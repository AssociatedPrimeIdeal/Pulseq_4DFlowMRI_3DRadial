import sys

module_path = '/nas-data/ryy_rawdata/gropt-master/gropt-master/python'
sys.path.append(module_path)


class MRISequence:
    def __init__(self, sys, The_list, Phi_list, Encoding_list, TERange, FOV, Nxyz, FA, dwell, tbw, max_grad, max_slew,
                 spoilratio=0.5):
        # Initialize sequence parameters
        self.FOVx, self.FOVy, self.FOVz = FOV[0], FOV[1], FOV[2]
        self.Nx, self.Ny, self.Nz = Nxyz[0], Nxyz[1], Nxyz[2]
        self.tbw = tbw  # Flip angle in degrees
        self.FA = FA
        self.max_grad = max_grad
        self.max_slew = max_slew
        # Calculate derived parameters
        self.dx = 1 / FOV[0]
        self.dy = 1 / FOV[1]
        self.dz = 1 / FOV[2]

        # Constants
        self.RF_SPOIL_INC = 117.0
        self.gradient_cache = {}

        # Initialize sequence object
        self.sys = sys
        self.seq = pp.Sequence(system=self.sys)
        # Initialize RF phase variables
        self.rf_phase = 0
        self.rf_inc = 0
        self.spoilratio = spoilratio
        self.dwell = dwell
        self.TERange = TERange
        self.The_list = The_list
        self.Phi_list = Phi_list
        self.Encoding_list = Encoding_list
        self.max_spoil_time = 0
        self.max_vel_time = 0
        self.max_ro_time = 0

    def make_gradient(self, axis, moment_params, min_TE=0.1, max_TE=5, verbose=0):
        T_lo = min_TE
        T_hi = max_TE
        T_range = T_hi - T_lo
        best_time = 999999.9
        params = {
            'mode': 'free',
            'dt': self.seq.grad_raster_time,
            'gmax': self.max_grad,
            'smax': self.max_slew,
            'moment_params': moment_params,
        }
        while ((T_range * 1e-3) > (self.seq.grad_raster_time / 4.0)):
            params['TE'] = T_lo + (T_range) / 2.0
            if verbose:
                print(' %.3f' % params['TE'], end='', flush=True)
            G, ddebug = gropt.gropt(params)
            lim_break = ddebug[14]
            if lim_break == 0:
                T_hi = params['TE']
                if T_hi < best_time:
                    G_out = G
                    T_out = T_hi
                    best_time = T_hi
            else:
                T_lo = params['TE']
            T_range = T_hi - T_lo
        G_out = np.squeeze(G_out)
        g_vel = pp.make_arbitrary_grad(channel=axis,
                                       waveform=G_out * self.sys.gamma,
                                       system=self.sys)
        return g_vel

    def make_gradient_fixedTE(self, axis, moment_params, TE):
        params = {
            'mode': 'free',
            'dt': self.seq.grad_raster_time,
            'gmax': self.max_grad,
            'smax': self.max_slew,
            'moment_params': moment_params,
            'TE': TE * 1e3,
        }
        G_out, ddebug = gropt.gropt(params)
        if ddebug[14]:
            return False, False
        G_out = np.squeeze(G_out)
        G_time = [10e-6 * i for i in range(len(G_out))]
        g_vel = pp.make_extended_trapezoid(channel=axis,
                                           amplitudes=G_out * self.sys.gamma,
                                           times=G_time,
                                           system=self.sys)
        return True, g_vel

    def CalTrapM1(self, A, w, r, t0, second_half=False):
        # A : gradient amplitude
        # w : flat time
        # r : rise time
        # t0: start time
        absA = np.abs(A)
        s = A / r
        M0 = A * (w + r / 2)

        if second_half:
            p1 = A * (absA ** 3 + (3 * w + 6 * t0) * s ** 2 * w * absA + (3 * w + 3 * t0) * s * A ** 2)
        else:
            p1 = A * (absA ** 3 + (3 * w + 6 * t0) * s ** 2 * w * absA + (6 * w + 3 * t0) * s * A ** 2)
        p2 = (6 * s ** 2 * absA)
        if p1 == 0 and p2 == 0:
            M1 = 0
        else:
            M1 = p1 / p2
        return M0, M1

    def CheckTE(self, useTE):
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.FA * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.FOVz * 2,
            apodization=0.42,
            time_bw_product=self.tbw,
            system=self.sys,
            return_gz=True
        )
        tstart = pp.calc_duration(gz) / 2

        gx = pp.make_trapezoid(channel='x',
                               flat_area=self.Nx * self.dx,
                               flat_time=self.dwell * self.Nx,
                               system=self.sys)
        amp = gx.amplitude
        self.max_rise_time = gx.rise_time
        for i in tqdm(range(len(The_list))):
            VE = self.Encoding_list[i]
            The = self.The_list[i]
            Phi = self.Phi_list[i]

            areax = amp * np.sin(The) * np.cos(Phi) * self.dwell * self.Nx
            areay = amp * np.sin(The) * np.sin(Phi) * self.dwell * self.Nx
            areaz = amp * np.cos(The) * self.dwell * self.Nx

            gx_spoil = pp.make_trapezoid(channel='x', area=areax * self.spoilratio,
                                         system=self.sys)
            gy_spoil = pp.make_trapezoid(channel='y', area=areay * self.spoilratio,
                                         system=self.sys)
            gz_spoil = pp.make_trapezoid(channel='z', area=areaz * self.spoilratio,
                                         system=self.sys)

            spoil_time = np.max([pp.calc_duration(gx_spoil), pp.calc_duration(gy_spoil), pp.calc_duration(gz_spoil)])
            if spoil_time > self.max_spoil_time:
                self.max_spoil_time = spoil_time

            gx_ro = pp.make_trapezoid(channel='x',
                                      flat_area=areax,
                                      flat_time=self.dwell * self.Nx,
                                      rise_time=self.max_rise_time,
                                      system=self.sys)
            gy_ro = pp.make_trapezoid(channel='y',
                                      flat_area=areay,
                                      flat_time=self.dwell * self.Nx,
                                      rise_time=self.max_rise_time,
                                      system=self.sys)
            gz_ro = pp.make_trapezoid(channel='z',
                                      flat_area=areaz,
                                      flat_time=self.dwell * self.Nx,
                                      rise_time=self.max_rise_time,
                                      system=self.sys)

            ro_time = np.max([pp.calc_duration(gx_ro), pp.calc_duration(gy_ro), pp.calc_duration(gz_ro)])
            if ro_time > self.max_spoil_time:
                self.max_ro_time = ro_time

            M0y_reph, M1y_reph = self.CalTrapM1(gy_ro.amplitude / self.sys.gamma * 1e3,
                                                gy_ro.flat_time * 1000 / 2,
                                                gy_ro.rise_time * 1000,
                                                (useTE - pp.calc_duration(gy_ro) / 2) * 1000)

            y_moment_params = [
                [0, 0, tstart, -1, -1, -M0y_reph, 1.0e-6],
                [0, 1, tstart, -1, -1, M1s[VE, 1] - M1y_reph, 1.0e-6]
            ]
            gy_vel = self.make_gradient_fixedTE('y', y_moment_params, (useTE - tstart - pp.calc_duration(gy_ro) / 2))
            if not gy_vel[0]:
                return False
            # Flow gx
            M0x_reph, M1x_reph = self.CalTrapM1(gx_ro.amplitude / self.sys.gamma * 1e3,
                                                gx_ro.flat_time * 1000 / 2,
                                                gx_ro.rise_time * 1000,
                                                (useTE - pp.calc_duration(gx_ro) / 2) * 1000)

            x_moment_params = [
                [0, 0, tstart, -1, -1, -M0x_reph, 1.0e-6],
                [0, 1, tstart, -1, -1, (M1s[VE, 0] - M1x_reph), 1.0e-6]
            ]
            gx_vel = self.make_gradient_fixedTE('x', x_moment_params, (useTE - tstart - pp.calc_duration(gx_ro) / 2))
            if not gx_vel[0]:
                return False
            # Flow gz
            M0ss_reph, M1ss_reph = self.CalTrapM1(gz.amplitude / self.sys.gamma * 1e3,
                                                  gz.flat_time * 1000 / 2,
                                                  gz.rise_time * 1000,
                                                  0.00,
                                                  second_half=True)
            M0z_reph, M1z_reph = self.CalTrapM1(gz_ro.amplitude / self.sys.gamma * 1e3,
                                                gz_ro.flat_time * 1000 / 2,
                                                gz_ro.rise_time * 1000,
                                                (useTE - pp.calc_duration(gz_ro) / 2) * 1000)

            z_moment_params = [
                [0, 0, tstart, -1, -1, - M0ss_reph - M0z_reph, 1.0e-6],
                [0, 1, tstart, -1, -1, M1s[VE, 2] - M1z_reph - M1ss_reph, 1.0e-6]
            ]
            gz_vel = self.make_gradient_fixedTE('z', z_moment_params, (useTE - tstart - pp.calc_duration(gz_ro) / 2))
            if not gz_vel[0]:
                return False

        self.TE = useTE
        self.TE = math.ceil(self.TE / self.seq.grad_raster_time) * self.seq.grad_raster_time
        self.TR = useTE + self.max_spoil_time + self.max_ro_time / 2 + tstart
        self.TR = math.ceil(self.TR / self.seq.grad_raster_time) * self.seq.grad_raster_time

        print("FINGDING BEST TE---------------------------TE(ms)", self.TE * 1e3, "TR(ms)", self.TR * 1e3)
        return True

    def prep(self):
        left = self.TERange[0] * 1e5
        right = self.TERange[1] * 1e5
        epsilon = 1
        result = self.TERange[1]
        while right - left > epsilon:
            mid = left + (right - left) // 2
            print(mid, left, right)
            if self.CheckTE(mid / 1e5):
                result = mid
                right = mid
            else:
                left = mid

        if self.CheckTE(left / 1e5):
            result = left
        elif self.CheckTE(right / 1e5):
            result = right

    def make_tr(self, The, Phi, M1s, plot=False, labels=None):
        # RF pulse
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.FA * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.FOVz * 2,
            apodization=0.42,
            time_bw_product=self.tbw,
            system=self.sys,
            return_gz=True
        )
        tstart = pp.calc_duration(gz) / 2

        gx = pp.make_trapezoid(channel='x',
                               flat_area=self.Nx * self.dx,
                               flat_time=self.dwell * self.Nx,
                               system=self.sys)

        amp = gx.amplitude
        areax = amp * np.sin(The) * np.cos(Phi) * self.dwell * self.Nx
        areay = amp * np.sin(The) * np.sin(Phi) * self.dwell * self.Nx
        areaz = amp * np.cos(The) * self.dwell * self.Nx

        gx_spoil = pp.make_trapezoid(channel='x', area=areax * self.spoilratio, duration=self.max_spoil_time,
                                     system=self.sys)
        gy_spoil = pp.make_trapezoid(channel='y', area=areay * self.spoilratio, duration=self.max_spoil_time,
                                     system=self.sys)
        gz_spoil = pp.make_trapezoid(channel='z', area=areaz * self.spoilratio, duration=self.max_spoil_time,
                                     system=self.sys)

        gx_ro = pp.make_trapezoid(channel='x',
                                  flat_area=areax,
                                  flat_time=self.dwell * self.Nx,
                                  rise_time=self.max_rise_time,
                                  system=self.sys)
        gy_ro = pp.make_trapezoid(channel='y',
                                  flat_area=areay,
                                  rise_time=self.max_rise_time,
                                  flat_time=self.dwell * self.Nx,
                                  system=self.sys)
        gz_ro = pp.make_trapezoid(channel='z',
                                  flat_area=areaz,
                                  rise_time=self.max_rise_time,
                                  flat_time=self.dwell * self.Nx,
                                  system=self.sys)

        adc = pp.make_adc(num_samples=self.Nx / 2,
                          delay=self.max_rise_time,
                          dwell=self.dwell * 2,
                          system=self.sys)

        M0y_reph, M1y_reph = self.CalTrapM1(gy_ro.amplitude / self.sys.gamma * 1e3,
                                            gy_ro.flat_time * 1000 / 2,
                                            gy_ro.rise_time * 1000,
                                            (self.TE - pp.calc_duration(gy_ro) / 2) * 1000)

        y_moment_params = [
            [0, 0, tstart, -1, -1, -M0y_reph, 1.0e-6],
            [0, 1, tstart, -1, -1, M1s[1] - M1y_reph, 1.0e-6]
        ]
        gy_vel = self.make_gradient_fixedTE('y', y_moment_params, (self.TE - tstart - pp.calc_duration(gy_ro) / 2))

        # Flow gx
        M0x_reph, M1x_reph = self.CalTrapM1(gx_ro.amplitude / self.sys.gamma * 1e3,
                                            gx_ro.flat_time * 1000 / 2,
                                            gx_ro.rise_time * 1000,
                                            (self.TE - pp.calc_duration(gx_ro) / 2) * 1000)

        x_moment_params = [
            [0, 0, tstart, -1, -1, -M0x_reph, 1.0e-6],
            [0, 1, tstart, -1, -1, (M1s[0] - M1x_reph), 1.0e-6]
        ]
        gx_vel = self.make_gradient_fixedTE('x', x_moment_params, (self.TE - tstart - pp.calc_duration(gx_ro) / 2))
        # Flow gz
        M0ss_reph, M1ss_reph = self.CalTrapM1(gz.amplitude / self.sys.gamma * 1e3,
                                              gz.flat_time * 1000 / 2,
                                              gz.rise_time * 1000,
                                              0.00,
                                              second_half=True)
        M0z_reph, M1z_reph = self.CalTrapM1(gz_ro.amplitude / self.sys.gamma * 1e3,
                                            gz_ro.flat_time * 1000 / 2,
                                            gz_ro.rise_time * 1000,
                                            (self.TE - pp.calc_duration(gz_ro) / 2) * 1000)

        z_moment_params = [
            [0, 0, tstart, -1, -1, - M0ss_reph - M0z_reph, 1.0e-6],
            [0, 1, tstart, -1, -1, M1s[2] - M1z_reph - M1ss_reph, 1.0e-6]
        ]
        gz_vel = self.make_gradient_fixedTE('z', z_moment_params, (self.TE - tstart - pp.calc_duration(gz_ro) / 2))

        self.TR = math.ceil(self.TR / self.seq.grad_raster_time) * self.seq.grad_raster_time

        adc.dwell = np.round(adc.dwell / self.seq.adc_raster_time) * self.seq.adc_raster_time
        # assemble sequence

        rf_1.phase_offset = self.rf_phase / 180 * np.pi
        adc.phase_offset = self.rf_phase / 180 * np.pi
        self.rf_inc = np.mod(self.rf_inc + self.RF_SPOIL_INC, 360.0)
        self.rf_phase = np.mod(self.rf_phase + self.rf_inc, 360.0)
        # Add blocks to sequence
        self.seq.add_block(rf_1, gz)
        self.seq.add_block(gx_vel[1], gy_vel[1], gz_vel[1])
        self.seq.add_block(gx_ro, gy_ro, gz_ro, adc)
        self.seq.add_block(gx_spoil, gy_spoil, gz_spoil)
        if plot:
            self.seq.plot(grad_disp='mT/m')


def gen_3DRadial_traj_new(m_lSegments, m_lSpokePerSeg, lDummys, lPrescan, lEncodingNum):
    lSGNum = m_lSpokePerSeg
    UMR_PI = np.pi
    lEquatorIndex = int(np.ceil(m_lSpokePerSeg * m_lSegments / 2.0))
    lScanLines = m_lSegments * m_lSpokePerSeg

    dPhi_list_scan = [0.0] * lScanLines
    dThe_list_scan = [0.0] * lScanLines
    dEncoding = [0] * (lScanLines + lDummys + lPrescan)

    for lSegIdx in range(1, m_lSegments + 1):
        for lSpkIdx in range(1, m_lSpokePerSeg + 1):
            if lSegIdx % 2 == 0:
                lCurIndex = m_lSpokePerSeg * (lSegIdx - 1) + (m_lSpokePerSeg - lSpkIdx)
            else:
                lCurIndex = m_lSpokePerSeg * (lSegIdx - 1) + (lSpkIdx - 1)
            lDirIndex = lSegIdx + (lSpkIdx - 1) * m_lSegments
            dPhi = lDirIndex * 137.51 * UMR_PI / 180.0
            if lDirIndex <= lEquatorIndex:
                dThe = UMR_PI / 2.0 * np.sqrt(lDirIndex / lEquatorIndex)
            else:
                dThe = UMR_PI - UMR_PI / 2.0 * np.sqrt(
                    (m_lSegments * m_lSpokePerSeg - lDirIndex) / lEquatorIndex)
            if (lCurIndex + 1) % m_lSpokePerSeg == 0:
                dThe = 0.0
            dThe_list_scan[lCurIndex] = dThe
            dPhi_list_scan[lCurIndex] = dPhi
    dThe_list_pre = [0.0] * lPrescan
    dPhi_list_pre = [0.0] * lPrescan
    n = lPrescan // 9 // lEncodingNum

    for nv in range(lEncodingNum):
        encoding_offset = nv * n * 9
        dThe_list_pre[encoding_offset:encoding_offset + 3 * n] = [UMR_PI / 2] * (3 * n)
        for i in range(n):
            dPhi_list_pre[encoding_offset + i] = dPhi_list_scan[i]
        for i in range(n):
            dPhi_list_pre[encoding_offset + n + i] = dPhi_list_scan[i] + UMR_PI
        for i in range(n):
            dPhi_list_pre[encoding_offset + 2 * n + i] = dPhi_list_scan[i] + UMR_PI / 2
        dPhi_list_pre[encoding_offset + 3 * n:encoding_offset + 6 * n] = [0.0] * (3 * n)
        for i in range(n):
            dThe_list_pre[encoding_offset + 3 * n + i] = dThe_list_scan[i]
        for i in range(n):
            dThe_list_pre[encoding_offset + 4 * n + i] = dThe_list_scan[i] + UMR_PI
        for i in range(n):
            dThe_list_pre[encoding_offset + 5 * n + i] = dThe_list_scan[i] + UMR_PI / 2
        dPhi_list_pre[encoding_offset + 6 * n:encoding_offset + 9 * n] = [UMR_PI / 2] * (3 * n)
        for i in range(n):
            dThe_list_pre[encoding_offset + 6 * n + i] = dThe_list_scan[i]
        for i in range(n):
            dThe_list_pre[encoding_offset + 7 * n + i] = dThe_list_scan[i] + UMR_PI
        for i in range(n):
            dThe_list_pre[encoding_offset + 8 * n + i] = dThe_list_scan[i] + UMR_PI / 2

    lAllLines = lScanLines + lDummys + lPrescan
    dPhi_list_all = [0.0] * lAllLines
    dThe_list_all = [0.0] * lAllLines
    print(lAllLines, lDummys, len(dPhi_list_all), len(dPhi_list_scan))
    for i in range(lDummys):
        dPhi_list_all[i] = dPhi_list_scan[i]
        dThe_list_all[i] = dThe_list_scan[i]
        dEncoding[i] = 0

    for i in range(lPrescan):
        dPhi_list_all[lDummys + i] = dPhi_list_pre[i]
        dThe_list_all[lDummys + i] = dThe_list_pre[i]
        dEncoding[lDummys + i] = (i // (n * 9))

    dThe_list_all[lDummys + lPrescan:lDummys + lPrescan + lScanLines] = dThe_list_scan
    dPhi_list_all[lDummys + lPrescan:lDummys + lPrescan + lScanLines] = dPhi_list_scan

    # Post processing
    for i in range(len(dPhi_list_all)):
        dPhi_list_all[i] = np.mod(dPhi_list_all[i], 2 * UMR_PI)
        dThe_list_all[i] = np.mod(dThe_list_all[i], 2 * UMR_PI)
        if i > lDummys + lPrescan - 1:
            dEncoding[i] = (i - lDummys - lPrescan) % lEncodingNum
            if ((i - lDummys - lPrescan) % lSGNum) == 0:
                dEncoding[i] = 0

    return dThe_list_all, dPhi_list_all, dEncoding


if __name__ == "__main__":
    import numpy as np
    import pypulseq as pp
    import sys
    import gropt

    import math
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    pnsratio = 0.5
    max_grad = 114.26
    max_slew = 190.47 * pnsratio
    sys = pp.Opts(max_grad=max_grad, grad_unit='mT/m', max_slew=max_slew, slew_unit='T/m/s',
                  rf_ringdown_time=10e-6, rf_dead_time=400e-6, adc_dead_time=70e-6, grad_raster_time=10e-6)

    Nxyz = [320, 320, 320]
    FOV = [256e-3, 256e-3, 256e-3]
    VENC = [150, 150, 150, ]

    VENC1 = 150
    VENC2 = 50
    M1s = np.zeros((7, 3))
    MaxM1 = 1e11 / (sys.gamma * VENC1) / 4
    M1s[0, :] = [-MaxM1, -MaxM1, -MaxM1]
    M1s[1, :] = [MaxM1, -MaxM1, -MaxM1]
    M1s[2, :] = [-MaxM1, MaxM1, -MaxM1]
    M1s[3, :] = [-MaxM1, -MaxM1, MaxM1]
    factor = (-1 + 2 * (VENC1 / VENC2))
    M1s[4, :] = [factor * MaxM1, -MaxM1, -MaxM1]
    M1s[5, :] = [-MaxM1, factor * MaxM1, -MaxM1]
    M1s[6, :] = [-MaxM1, -MaxM1, factor * MaxM1]

    num_pre = 3000
    num_pre = num_pre // (9 * M1s.shape[0]) * (9 * M1s.shape[0])
    num_dummy = 100
    spokeperseg = 10
    segments = 10000
    The_list, Phi_list, Encoding_list = gen_3DRadial_traj_new(segments, spokeperseg, num_dummy, num_pre, M1s.shape[0])

    seq = MRISequence(
        The_list=The_list,
        Phi_list=Phi_list,
        Encoding_list=Encoding_list,
        sys=sys,
        TERange=[2e-3, 5e-3],
        FOV=FOV,
        Nxyz=Nxyz,
        FA=7,
        dwell=5e-6,
        tbw=4,
        max_grad=max_grad,
        max_slew=max_slew,
    )
    end_points = []
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    RO = Nxyz[0]
    r = RO // 2
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([-r, r])
    ax.view_init(elev=0, azim=0)
    for i in range(segments * spokeperseg + num_pre + num_dummy):
        dThe = The_list[i]
        dPhi = Phi_list[i]
        x = np.linspace(-r, r, RO) * np.sin(dThe) * np.cos(dPhi)
        y = np.linspace(-r, r, RO) * np.sin(dThe) * np.sin(dPhi)
        z = np.linspace(-r, r, RO) * np.cos(dThe)
        if i >= num_dummy + num_pre and i <= num_dummy + num_pre + spokeperseg - 1:
            end_points.append([x[-1], y[-1], z[-1]])
            ax.plot([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], marker='o')
    end_points = np.array(end_points)
    ax.plot(end_points[:, 0], end_points[:, 1], end_points[:, 2], marker='o', color='b')
    plt.show()

    seq.prep()
    for i in tqdm(range(len(The_list))):
        if i == 0:
            ifplot = True
        else:
            ifplot = False
        seq.make_tr(The_list[i], Phi_list[i], M1s=M1s[Encoding_list[i]], plot=ifplot)
    (
        ok,
        error_report,
    ) = seq.seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]
    seq.seq.set_definition('FOV', [seq.FOVx, seq.FOVy, seq.FOVz])
    seq.seq.set_definition('Dimension', 3)
    seq.seq.set_definition('SliceNumber', Nxyz[-1])
    seq.seq.set_definition('SliceThickness', FOV[-1] / Nxyz[-1])
    seq.seq.write(f'/nas-data/ryy_rawdata/pulseq/4Dflow_3dradial.seq')
    # Show K-space sequence
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.seq.calculate_kspace()
    plt.figure(figsize=(30, 30))
    plt.plot(k_traj[0], k_traj[1])
    plt.plot(k_traj_adc[0], k_traj_adc[1], '.', ms=5)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
