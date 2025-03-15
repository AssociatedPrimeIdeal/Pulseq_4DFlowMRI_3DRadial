import copy
import numpy as np
import pypulseq as pp
import sys

module_path = '/nas-data/ryy_rawdata/gropt-master/gropt-master/python'
sys.path.append(module_path)
import gropt
import math
from scipy.signal import medfilt
from scipy.integrate import simpson
import multiprocessing
from functools import partial

import pandas as pd


class MRISequence:
    def __init__(self, TE, TR, FOV, Nxyz, FA, dwell, tbw, pnsratio=1.0):
        # Initialize sequence parameters
        self.FOVx, self.FOVy, self.FOVz = FOV[0], FOV[1], FOV[2]
        self.Nx, self.Ny, self.Nz = Nxyz[0], Nxyz[1], Nxyz[2]
        self.tbw = tbw  # Flip angle in degrees
        self.FA = FA
        # Calculate derived parameters
        self.dx = 1 / FOV[0]
        self.dy = 1 / FOV[1]
        self.dz = 1 / FOV[2]
        self.pnsratio = pnsratio
        self.max_grad = 114.26
        self.max_slew = 190.47 * self.pnsratio
        # Constants
        self.RF_SPOIL_INC = 117.0
        self.gradient_cache = {}

        # Initialize sequence object
        self.sys = pp.Opts(max_grad=self.max_grad, grad_unit='mT/m', max_slew=self.max_slew, slew_unit='T/m/s',
                           rf_ringdown_time=10e-6, rf_dead_time=400e-6, adc_dead_time=70e-6, grad_raster_time=10e-6)
        self.seq = pp.Sequence(system=self.sys)
        # Initialize RF phase variables
        self.rf_phase = 0
        self.rf_inc = 0

        self.dwell = dwell
        self.TE = TE
        self.TR = TR

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
        G_out = np.squeeze(G_out)
        G_time = [10e-6 * i for i in range(len(G_out))]
        g_vel = pp.make_extended_trapezoid(channel=axis,
                                           amplitudes=G_out * self.sys.gamma,
                                           times=G_time,
                                           system=self.sys)
        return g_vel

    def choose_alpha(self, alpha, M01, M02, M11, M12, tstart, axis):
        moment_params = [
            [0, 0, tstart, -1, -1, M01, 1.0e-6],
            [0, 1, tstart, -1, -1, M11, 1.0e-6]
        ]
        g_vel = self.make_gradient(axis, moment_params)
        d1 = pp.calc_duration(g_vel)
        moment_params = [
            [0, 0, tstart, -1, -1, M02, 1.0e-6],
            [0, 1, tstart, -1, -1, M12, 1.0e-6]
        ]
        g_vel = self.make_gradient(axis, moment_params)
        d2 = pp.calc_duration(g_vel)
        return (alpha, M01, M02, M11, M12, d1, d2)

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

    def prep(self, M1):
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.FA * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.FOVz,
            apodization=0.42,
            time_bw_product=self.tbw,
            system=self.sys,
            return_gz=True
        )
        M0z_reph, M1z_reph = self.CalTrapM1(gz.amplitude / self.sys.gamma * 1e3,
                                            gz.flat_time * 1000 / 2,
                                            gz.rise_time * 1000,
                                            0.00,
                                            second_half=True)

        gx = pp.make_trapezoid(channel='x',
                               flat_area=self.Nx * self.dx,
                               flat_time=self.dwell * self.Nx,
                               system=self.sys)

        tstart = pp.calc_duration(gz) / 2
        areay = (-(np.arange(seq.Ny) - seq.Ny / 2) * seq.dy).tolist()
        areaz = (-(np.arange(seq.Nz) - seq.Nz / 2) * seq.dz).tolist()

        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        alpha_list = []
        M01_list = []
        M02_list = []
        M11_list = []
        M12_list = []
        tstart_list = []
        axis_list = []
        for alpha in np.linspace(0, 0.5, 51):
            alpha_list.extend([alpha for a in areay])
            M01_list.extend(list(np.array(areay) / self.sys.gamma * 1e6))
            M02_list.extend(list(np.array(areay) / self.sys.gamma * 1e6))
            M11_list.extend([M1[1] * alpha for a in areay])
            M12_list.extend([M1[1] * (alpha - 1) for a in areay])
            tstart_list.extend([tstart for a in areay])
            axis_list.extend(['y' for a in areay])
        para_zip = list(
            zip(np.array(alpha_list), np.array(M01_list), np.array(M02_list), np.array(M11_list), np.array(M12_list),
                np.array(tstart_list), np.array(axis_list)))
        results = pool.starmap(self.choose_alpha, para_zip)
        pool.close()
        pool.join()

        pool = multiprocessing.Pool(processes=num_processes)
        results = pd.DataFrame(results, columns=['alpha', 'M01', 'M02', 'M11', 'M12', 'd1', 'd2'])
        results['maxtime'] = np.max([results['d1'], results['d2']])
        maxtime_per_alpha = results.groupby('alpha')['maxtime'].max().reset_index()
        miny = maxtime_per_alpha['maxtime'].min()
        alphay = maxtime_per_alpha.loc[maxtime_per_alpha['maxtime'].idxmin(), 'alpha']
        print("Y", alphay, miny)

        alpha_list = []
        M01_list = []
        M02_list = []
        M11_list = []
        M12_list = []
        tstart_list = []
        axis_list = []
        for alpha in np.linspace(0, 1, 101):
            alpha_list.extend([alpha for a in areay])
            M01_list.extend(list(np.array(areaz) / self.sys.gamma * 1e6 - M0z_reph))
            M02_list.extend(list(np.array(areaz) / self.sys.gamma * 1e6 - M0z_reph))
            M11_list.extend([M1[2] * alpha - M1z_reph for a in areay])
            M12_list.extend([M1[2] * (alpha - 1) - M1z_reph for a in areay])
            tstart_list.extend([tstart for a in areay])
            axis_list.extend(['z' for a in areay])
        para_zip = list(
            zip(np.array(alpha_list), np.array(M01_list), np.array(M02_list), np.array(M11_list), np.array(M12_list),
                np.array(tstart_list), np.array(axis_list)))
        results = pool.starmap(self.choose_alpha, para_zip)
        pool.close()
        pool.join()

        pool = multiprocessing.Pool(processes=num_processes)
        results = pd.DataFrame(results, columns=['alpha', 'M01', 'M02', 'M11', 'M12', 'd1', 'd2'])
        results['maxtime'] = np.max([results['d1'], results['d2']])
        maxtime_per_alpha = results.groupby('alpha')['maxtime'].max().reset_index()
        minz = maxtime_per_alpha['maxtime'].min()
        alphaz = maxtime_per_alpha.loc[maxtime_per_alpha['maxtime'].idxmin(), 'alpha']
        print("Z", alphaz, minz)

        M0x_reph, M1x_reph = self.CalTrapM1(gx.amplitude / self.sys.gamma * 1e3,
                                            gx.flat_time * 1000 / 2,
                                            gx.rise_time * 1000,
                                            (np.max([minz, miny]) + pp.calc_duration(gz) / 2) * 1000)
        alpha_list = []
        M01_list = []
        M02_list = []
        M11_list = []
        M12_list = []
        tstart_list = []
        axis_list = []
        for alpha in np.linspace(0, 1, 101):
            alpha_list.extend([alpha for a in areay])
            M01_list.extend([-M0x_reph for a in areay])
            M02_list.extend([-M0x_reph for a in areay])
            M11_list.extend([M1[0] * alpha - M1x_reph for a in areay])
            M12_list.extend([M1[0] * (alpha - 1) - M1x_reph for a in areay])
            tstart_list.extend([tstart for a in areay])
            axis_list.extend(['x' for a in areay])
        para_zip = list(
            zip(np.array(alpha_list), np.array(M01_list), np.array(M02_list), np.array(M11_list), np.array(M12_list),
                np.array(tstart_list), np.array(axis_list)))
        results = pool.starmap(self.choose_alpha, para_zip)
        pool.close()
        pool.join()

        results = pd.DataFrame(results, columns=['alpha', 'M01', 'M02', 'M11', 'M12', 'd1', 'd2'])
        results['maxtime'] = np.max([results['d1'], results['d2']])
        maxtime_per_alpha = results.groupby('alpha')['maxtime'].max().reset_index()
        minx = maxtime_per_alpha['maxtime'].min()
        alphax = maxtime_per_alpha.loc[maxtime_per_alpha['maxtime'].idxmin(), 'alpha']
        print("X", alphax, minx)

        t_vel = np.max([minx, miny, minz])
        self.t_vel = math.ceil(t_vel / self.seq.grad_raster_time) * self.seq.grad_raster_time
        self.TE = math.ceil(
            (gz.fall_time + gz.flat_time / 2 + t_vel + pp.calc_duration(gx) / 2) / self.seq.grad_raster_time) \
                  * self.seq.grad_raster_time
        self.alphas = [alphax, alphay, alphaz]
        print(self.t_vel, self.TE, self.alphas)

    def prep_nonc(self, The_list, Phi_list, M1):
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.FA * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.FOVz,
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
        maxrisetime = gx.rise_time
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)

        alpha_list = []
        M01_list = []
        M02_list = []
        M11_list = []
        M12_list = []
        tstart_list = []
        axis_list = []
        # y_moment_params = [
        #     [0, 0, tstart, -1, -1, M0y, 1.0e-6],
        #     [0, 1, tstart, -1, -1, M1s[1], 1.0e-6]
        # ]

        # for alpha in np.linspace(0, 1, 101):
        for alpha in [0]:
            areay = amp * np.sin(The_list) * np.sin(Phi_list) * self.dwell * self.Nx

            alpha_list.extend([alpha for a in areay])
            tstart_list.extend([tstart for a in areay])
            axis_list.extend(['y' for a in areay])

            for a in areay:
                gy_ro = pp.make_trapezoid(channel='y',
                                          flat_area=a,
                                          rise_time=maxrisetime,
                                          flat_time=self.dwell * self.Nx,
                                          system=self.sys)

                M0y_reph, M1y_reph = self.CalTrapM1(gy_ro.amplitude / self.sys.gamma * 1e3,
                                                    gy_ro.flat_time * 1000 / 2,
                                                    gy_ro.rise_time * 1000,
                                                    (self.TE - pp.calc_duration(gy_ro) / 2) * 1000)
                M01_list.append(-M0y_reph)
                M02_list.append(-M0y_reph)
                M11_list.append(M1[1] * alpha - M1y_reph)
                M12_list.append(M1[1] * (alpha - 1) - M1y_reph)

        para_zip = list(
            zip(np.array(alpha_list), np.array(M01_list), np.array(M02_list), np.array(M11_list), np.array(M12_list),
                np.array(tstart_list), np.array(axis_list)))
        results = pool.starmap(self.choose_alpha, para_zip)
        pool.close()
        pool.join()

        pool = multiprocessing.Pool(processes=num_processes)
        results = pd.DataFrame(results, columns=['alpha', 'M01', 'M02', 'M11', 'M12', 'd1', 'd2'])
        results['maxtime'] = np.max([results['d1'], results['d2']])
        maxtime_per_alpha = results.groupby('alpha')['maxtime'].max().reset_index()
        miny = maxtime_per_alpha['maxtime'].min()
        alphay = maxtime_per_alpha.loc[maxtime_per_alpha['maxtime'].idxmin(), 'alpha']
        print("Y", alphay, miny)

        alpha_list = []
        M01_list = []
        M02_list = []
        M11_list = []
        M12_list = []
        tstart_list = []
        axis_list = []
        # z_moment_params = [
        #     [0, 0, tstart, -1, -1, M0z_total, 1.0e-6],
        #     [0, 1, tstart, -1, -1, M1z_total, 1.0e-6]
        # ]
        # for alpha in np.linspace(0, 1, 101):
        for alpha in [0]:
            areaz = amp * np.cos(The_list) * self.dwell * self.Nx

            alpha_list.extend([alpha for a in areaz])
            tstart_list.extend([tstart for a in areaz])
            axis_list.extend(['z' for a in areaz])
            for a in areaz:

                gz_ro = pp.make_trapezoid(channel='z',
                                          flat_area=a,
                                          rise_time=maxrisetime,
                                          flat_time=self.dwell * self.Nx,
                                          system=self.sys)
                M0ss_reph, M1ss_reph = self.CalTrapM1(gz.amplitude / self.sys.gamma * 1e3,
                                                      gz.flat_time * 1000 / 2,
                                                      gz.rise_time * 1000,
                                                      0.00,
                                                      second_half=True)
                M0z_reph, M1z_reph = self.CalTrapM1(gz_ro.amplitude / self.sys.gamma * 1e3,
                                                    gz_ro.flat_time * 1000 / 2,
                                                    gz_ro.rise_time * 1000,
                                                    (self.TE - pp.calc_duration(gz_ro) / 2) * 1000)
                M01_list.append(- M0ss_reph - M0z_reph)
                M02_list.append(- M0ss_reph - M0z_reph)
                M11_list.append(M1[2] * alpha - M1z_reph - M0ss_reph)
                M12_list.append(M1[2] * (alpha - 1) - M1z_reph - M0ss_reph)

        para_zip = list(
            zip(np.array(alpha_list), np.array(M01_list), np.array(M02_list), np.array(M11_list), np.array(M12_list),
                np.array(tstart_list), np.array(axis_list)))
        results = pool.starmap(self.choose_alpha, para_zip)
        pool.close()
        pool.join()

        pool = multiprocessing.Pool(processes=num_processes)
        results = pd.DataFrame(results, columns=['alpha', 'M01', 'M02', 'M11', 'M12', 'd1', 'd2'])
        results['maxtime'] = np.max([results['d1'], results['d2']])
        maxtime_per_alpha = results.groupby('alpha')['maxtime'].max().reset_index()
        minz = maxtime_per_alpha['maxtime'].min()
        alphaz = maxtime_per_alpha.loc[maxtime_per_alpha['maxtime'].idxmin(), 'alpha']
        print("Z", alphaz, minz)

        M0x_reph, M1x_reph = self.CalTrapM1(gx.amplitude / self.sys.gamma * 1e3,
                                            gx.flat_time * 1000 / 2,
                                            gx.rise_time * 1000,
                                            (np.max([minz, miny]) + pp.calc_duration(gz) / 2) * 1000)
        alpha_list = []
        M01_list = []
        M02_list = []
        M11_list = []
        M12_list = []
        tstart_list = []
        axis_list = []
        # x_moment_params = [
        #     [0, 0, tstart, -1, -1, -M0x_reph, 1.0e-6],
        #     [0, 1, tstart, -1, -1, (M1s[0] - M1x_reph), 1.0e-6]
        # ]
        # for alpha in np.linspace(0, 1, 101):
        for alpha in [0]:
            areax = amp * np.sin(The_list) * np.cos(Phi_list) * self.dwell * self.Nx
            alpha_list.extend([alpha for a in areay])
            tstart_list.extend([tstart for a in areay])
            axis_list.extend(['x' for a in areay])
            for a in areax:

                gx_temp = pp.make_trapezoid(channel='x',
                                            flat_area=a,
                                            rise_time=maxrisetime,
                                            flat_time=self.dwell * self.Nx,
                                            system=self.sys)
                M0x_reph, M1x_reph = self.CalTrapM1(gx_temp.amplitude / self.sys.gamma * 1e3,
                                                    gx_temp.flat_time * 1000 / 2,
                                                    gx_temp.rise_time * 1000,
                                                    (self.TE - pp.calc_duration(gx_temp) / 2) * 1000)
                M01_list.append(-M0x_reph)
                M02_list.append(-M0x_reph)
                M11_list.append(M1[0] * alpha - M1x_reph)
                M12_list.append(M1[0] * (alpha - 1) - M1x_reph)

        para_zip = list(
            zip(np.array(alpha_list), np.array(M01_list), np.array(M02_list), np.array(M11_list), np.array(M12_list),
                np.array(tstart_list), np.array(axis_list)))
        results = pool.starmap(self.choose_alpha, para_zip)
        pool.close()
        pool.join()

        results = pd.DataFrame(results, columns=['alpha', 'M01', 'M02', 'M11', 'M12', 'd1', 'd2'])
        results['maxtime'] = np.max([results['d1'], results['d2']])
        maxtime_per_alpha = results.groupby('alpha')['maxtime'].max().reset_index()
        minx = maxtime_per_alpha['maxtime'].min()
        alphax = maxtime_per_alpha.loc[maxtime_per_alpha['maxtime'].idxmin(), 'alpha']
        print("X", alphax, minx)

        t_vel = np.max([minx, miny, minz])
        self.t_vel = math.ceil(t_vel / self.seq.grad_raster_time) * self.seq.grad_raster_time
        self.TE = math.ceil(
            (gz.fall_time + gz.flat_time / 2 + t_vel + pp.calc_duration(gx) / 2) / self.seq.grad_raster_time) \
                  * self.seq.grad_raster_time
        self.alphas = [alphax, alphay, alphaz]
        print(self.t_vel, self.TE, self.alphas)

    def make_tr(self, areay, areaz, M1s, plot=False, labels=None):
        # RF pulse
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.FA * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.FOVz,
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
        adc = pp.make_adc(num_samples=self.Nx,
                          delay=gx.rise_time,
                          duration=gx.flat_time,
                          system=self.sys)

        # Spoil
        # gy_reph = pp.make_trapezoid(channel='y', area=-areay, system=self.sys)
        gx_spoil = pp.make_trapezoid(channel='x', area=self.Nx * self.dx, system=self.sys)
        gz_spoil = pp.make_trapezoid(channel='z', area=4 / self.dz - areaz, system=self.sys)
        # Flow gy
        M0y = areay / self.sys.gamma * 1e6

        y_moment_params = [
            [0, 0, tstart, -1, -1, M0y, 1.0e-6],
            [0, 1, tstart, -1, -1, M1s[1], 1.0e-6]
        ]

        gy_vel = self.make_gradient_fixedTE('y', y_moment_params, self.t_vel)
        # gy_vel = self.make_gradient('y', y_moment_params)

        # Flow gx
        M0x_reph, M1x_reph = self.CalTrapM1(gx.amplitude / self.sys.gamma * 1e3,
                                            gx.flat_time * 1000 / 2,
                                            gx.rise_time * 1000,
                                            (self.TE - pp.calc_duration(gx) / 2) * 1000)

        x_moment_params = [
            [0, 0, tstart, -1, -1, -M0x_reph, 1.0e-6],
            [0, 1, tstart, -1, -1, (M1s[0] - M1x_reph), 1.0e-6]
        ]
        gx_vel = self.make_gradient_fixedTE('x', x_moment_params, self.t_vel)
        # gx_vel = self.make_gradient('x', x_moment_params)

        # Flow gz
        M0z_reph, M1z_reph = self.CalTrapM1(gz.amplitude / self.sys.gamma * 1e3,
                                            gz.flat_time * 1000 / 2,
                                            gz.rise_time * 1000,
                                            0.00,
                                            second_half=True)

        M1z_total = M1s[2] - M1z_reph
        M0z_ex = (areaz) / (self.sys.gamma) * 1e6
        M0z_total = M0z_ex - M0z_reph

        z_moment_params = [
            [0, 0, tstart, -1, -1, M0z_total, 1.0e-6],
            [0, 1, tstart, -1, -1, M1z_total, 1.0e-6]
        ]
        gz_vel = self.make_gradient_fixedTE('z', z_moment_params, self.t_vel)
        # gz_vel = self.make_gradient('z', z_moment_params)
        # spoiling

        t_vel = np.max([pp.calc_duration(gx_vel), pp.calc_duration(gy_vel), pp.calc_duration(gz_vel)])
        self.TR = math.ceil(
            (pp.calc_duration(gz) + t_vel + pp.calc_duration(gx) + np.max([
                pp.calc_duration(gx_spoil),
                pp.calc_duration(gz_spoil)]))
            / self.seq.grad_raster_time) \
                  * self.seq.grad_raster_time

        adc.dwell = np.round(adc.dwell / self.seq.adc_raster_time) * self.seq.adc_raster_time
        # assemble sequence

        rf_1.phase_offset = self.rf_phase / 180 * np.pi
        adc.phase_offset = self.rf_phase / 180 * np.pi
        self.rf_inc = np.mod(self.rf_inc + self.RF_SPOIL_INC, 360.0)
        self.rf_phase = np.mod(self.rf_phase + self.rf_inc, 360.0)

        # Add blocks to sequence
        self.seq.add_block(rf_1, gz)
        self.seq.add_block(gx_vel, gy_vel, gz_vel)
        if labels != None:
            self.seq.add_block(gx, adc, *labels)
        else:
            self.seq.add_block(gx, adc)
        spoil_block_contents = [gx_spoil, gz_spoil]
        self.seq.add_block(*spoil_block_contents)
        # print(pp.calc_duration(gy_vel), pp.calc_duration(gx_vel), pp.calc_duration(gz_vel))
        # print(pp.calc_duration(gx), pp.calc_duration(adc))
        # print(pp.calc_duration(gy_reph), pp.calc_duration(gx_spoil), pp.calc_duration(gz_spoil))
        if plot:
            self.seq.plot(grad_disp='mT/m')

    def make_tr_nonc(self, The, Phi, M1s, plot=False, labels=None):
        # RF pulse
        rf_1, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.FA * np.pi / 180,
            duration=0.6e-3,
            slice_thickness=self.FOVz,
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

        maxrisetime = gx.rise_time
        amp = gx.amplitude
        areax = amp * np.sin(The) * np.cos(Phi) * self.dwell * self.Nx
        areay = amp * np.sin(The) * np.sin(Phi) * self.dwell * self.Nx
        areaz = amp * np.cos(The) * self.dwell * self.Nx
        gx_spoil = pp.make_trapezoid(channel='x', area=amp * self.dwell * self.Nx * np.sign(areax) * 0.5, system=self.sys)
        gy_spoil = pp.make_trapezoid(channel='y', area=amp * self.dwell * self.Nx * np.sign(areay) * 0.5, system=self.sys)
        gz_spoil = pp.make_trapezoid(channel='z', area=amp * self.dwell * self.Nx * np.sign(areaz) * 0.5, system=self.sys)

        gx = pp.make_trapezoid(channel='x',
                               flat_area=areax,
                               flat_time=self.dwell * self.Nx,
                               rise_time=maxrisetime,
                               system=self.sys)
        gy_ro = pp.make_trapezoid(channel='y',
                                  flat_area=areay,
                                  rise_time=maxrisetime,
                                  flat_time=self.dwell * self.Nx,
                                  system=self.sys)
        gz_ro = pp.make_trapezoid(channel='z',
                                  flat_area=areaz,
                                  rise_time=maxrisetime,
                                  flat_time=self.dwell * self.Nx,
                                  system=self.sys)
        adc = pp.make_adc(num_samples=self.Nx,
                          delay=maxrisetime,
                          duration=self.dwell * self.Nx,
                          system=self.sys)

        M0y_reph, M1y_reph = self.CalTrapM1(gy_ro.amplitude / self.sys.gamma * 1e3,
                                            gy_ro.flat_time * 1000 / 2,
                                            gy_ro.rise_time * 1000,
                                            (self.TE - pp.calc_duration(gy_ro) / 2) * 1000)

        y_moment_params = [
            [0, 0, tstart, -1, -1, -M0y_reph, 1.0e-6],
            [0, 1, tstart, -1, -1, M1s[1] - M1y_reph, 1.0e-6]
        ]
        gy_vel = self.make_gradient_fixedTE('y', y_moment_params, self.t_vel)

        # Flow gx
        M0x_reph, M1x_reph = self.CalTrapM1(gx.amplitude / self.sys.gamma * 1e3,
                                            gx.flat_time * 1000 / 2,
                                            gx.rise_time * 1000,
                                            (self.TE - pp.calc_duration(gx) / 2) * 1000)

        x_moment_params = [
            [0, 0, tstart, -1, -1, -M0x_reph, 1.0e-6],
            [0, 1, tstart, -1, -1, (M1s[0] - M1x_reph), 1.0e-6]
        ]
        gx_vel = self.make_gradient_fixedTE('x', x_moment_params, self.t_vel)

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
        gz_vel = self.make_gradient_fixedTE('z', z_moment_params, self.t_vel)

        t_vel = np.max([pp.calc_duration(gx_vel), pp.calc_duration(gy_vel), pp.calc_duration(gz_vel)])
        self.TR = math.ceil(
            (pp.calc_duration(gz) + t_vel + pp.calc_duration(gx) + np.max([
                pp.calc_duration(gx_spoil),
                pp.calc_duration(gz_spoil)]))
            / self.seq.grad_raster_time) \
                  * self.seq.grad_raster_time

        adc.dwell = np.round(adc.dwell / self.seq.adc_raster_time) * self.seq.adc_raster_time
        # assemble sequence

        rf_1.phase_offset = self.rf_phase / 180 * np.pi
        adc.phase_offset = self.rf_phase / 180 * np.pi
        self.rf_inc = np.mod(self.rf_inc + self.RF_SPOIL_INC, 360.0)
        self.rf_phase = np.mod(self.rf_phase + self.rf_inc, 360.0)

        # Add blocks to sequence
        self.seq.add_block(rf_1, gz)
        self.seq.add_block(gx_vel, gy_vel, gz_vel)
        if labels != None:
            self.seq.add_block(gx, gy_ro, gz_ro, adc, *labels)
        else:
            self.seq.add_block(gx, gy_ro, gz_ro, adc)
        spoil_block_contents = [gx_spoil, gy_spoil, gz_spoil]
        self.seq.add_block(*spoil_block_contents)
        if plot:
            self.seq.plot(grad_disp='mT/m')


def gen_3DRadial_traj(m_lSegments, m_lSpokePerSeg, lPrescan, lDummys):
    UMR_PI = np.pi
    lEquatorIndex = int(np.ceil(m_lSpokePerSeg * m_lSegments / 2.0))
    lScanLines = m_lSegments * m_lSpokePerSeg
    dThe_list_scan = np.zeros(lScanLines)
    dPhi_list_scan = np.zeros(lScanLines)
    for lSegIdx in range(1, m_lSegments + 1):
        for lSpkIdx in range(1, m_lSpokePerSeg + 1):
            if lSegIdx % 2 == 0:
                lCurIndex = m_lSpokePerSeg * (lSegIdx - 1) + (m_lSpokePerSeg - lSpkIdx)
            else:
                lCurIndex = m_lSpokePerSeg * (lSegIdx - 1) + (lSpkIdx - 1)
            lDirIndex = lSegIdx + (lSpkIdx - 1) * m_lSegments
            dPhi = lDirIndex * 137.51 * UMR_PI / 180.0
            if lDirIndex <= lEquatorIndex:
                dThe = UMR_PI / 2 * np.sqrt(float(lDirIndex) / float(lEquatorIndex))
            else:
                dThe = UMR_PI - UMR_PI / 2 * np.sqrt(
                    float(m_lSegments * m_lSpokePerSeg - lDirIndex) / float(lEquatorIndex))
            if (lCurIndex + 1) % m_lSpokePerSeg == 0:
                dThe = 0
            dThe_list_scan[lCurIndex] = dThe
            dPhi_list_scan[lCurIndex] = dPhi

    dThe_list_pre = np.zeros(lPrescan)
    dPhi_list_pre = np.zeros(lPrescan)
    n = lPrescan // 9
    dThe_list_pre[:3 * n] = UMR_PI / 2
    dPhi_list_pre[:n] = dPhi_list_scan[:n]
    for i in range(n, 2 * n):
        dPhi_list_pre[i] = dPhi_list_scan[i - n] + UMR_PI
    for i in range(2 * n, 3 * n):
        dPhi_list_pre[i] = dPhi_list_scan[i - 2 * n] + UMR_PI / 2

    dPhi_list_pre[3 * n:6 * n] = 0.0
    dThe_list_pre[3 * n:4 * n] = dThe_list_scan[:n]
    for i in range(4 * n, 5 * n):
        dThe_list_pre[i] = dThe_list_scan[i - 4 * n] + UMR_PI
    for i in range(5 * n, 6 * n):
        dThe_list_pre[i] = dThe_list_scan[i - 5 * n] + UMR_PI / 2

    dPhi_list_pre[6 * n:9 * n] = UMR_PI / 2
    dThe_list_pre[6 * n:7 * n] = dThe_list_scan[:n]
    for i in range(7 * n, 8 * n):
        dThe_list_pre[i] = dThe_list_scan[i - 7 * n] + UMR_PI
    for i in range(8 * n, 9 * n):
        dThe_list_pre[i] = dThe_list_scan[i - 8 * n] + UMR_PI / 2

    lAllLines = lScanLines + lDummys + lPrescan
    dThe_list_all = np.zeros(lAllLines)
    dPhi_list_all = np.zeros(lAllLines)
    dThe_list_all[:lDummys] = dThe_list_scan[:lDummys]
    dPhi_list_all[:lDummys] = dPhi_list_scan[:lDummys]
    dThe_list_all[lDummys:lDummys + lPrescan] = dThe_list_pre
    dPhi_list_all[lDummys:lDummys + lPrescan] = dPhi_list_pre
    dThe_list_all[lDummys + lPrescan:] = dThe_list_scan
    dPhi_list_all[lDummys + lPrescan:] = dPhi_list_scan
    dThe_list_all = np.mod(dThe_list_all, 2 * UMR_PI)
    dPhi_list_all = np.mod(dPhi_list_all, 2 * UMR_PI)
    return dThe_list_all, dPhi_list_all


if __name__ == "__main__":
    import matplotlib
    import numpy as np
    import pypulseq as pp
    import sys
    import gropt

    import math
    import matplotlib.pyplot as plt
    from scipy.signal import medfilt
    from scipy.integrate import simpson
    from tqdm import tqdm

    DO_TRIGGERING = False
    PLOT_KSPACE = False
    HEART_RATE = 70
    Nxyz = [64, 64, 64]
    FOV = [320e-3, 320e-3, 320e-3]
    VENC = [150, 150, 150,]
    seq = MRISequence(
        TE=3e-3,
        TR=10e-3,
        FOV=FOV,
        Nxyz=Nxyz,
        FA=7,
        dwell=10e-6,
        tbw=4,
        pnsratio=0.3,
    )
    # Calculate areay and areaz
    # areay = (-(np.arange(seq.Ny) - seq.Ny / 2) * seq.dy).tolist()
    # areaz = (-(np.arange(seq.Nz) - seq.Nz / 2) * seq.dz).tolist()
    TargetM1 = [0.5e11 / (seq.sys.gamma * VENC[0]),
                0.5e11 / (seq.sys.gamma * VENC[1]),
                0.5e11 / (seq.sys.gamma * VENC[2])]
    # M1 values in mT*ms^2/m

    num_pre = 1998
    num_dummy = 50
    spokeperseg = 200
    segments = 100

    The_list, Phi_list = gen_3DRadial_traj(segments, spokeperseg, num_pre, num_dummy)
    import matplotlib.pyplot as plt

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
    # seq.prep(TargetM1)
    # seq.alphas = [0, 0, 0]
    # seq.TE = 0.00298
    # seq.t_vel = 0.00178

    seq.prep_nonc(The_list, Phi_list, TargetM1)
    # seq.alphas = [0, 0, 0]
    # seq.TE = 0.00274
    # seq.t_vel = 0.00171

    alphas = seq.alphas
    M1s = [
        [TargetM1[0] * alphas[0], TargetM1[1] * alphas[1], TargetM1[2] * alphas[2]],
        [TargetM1[0] * (alphas[0] - 1), TargetM1[1] * alphas[1], TargetM1[2] * alphas[2]],
        [TargetM1[0] * alphas[0], TargetM1[1] * (alphas[1] - 1), TargetM1[2] * alphas[2]],
        [TargetM1[0] * alphas[0], TargetM1[1] * alphas[1], TargetM1[2] * (alphas[2] - 1)],
    ]
    # for islice in tqdm(range(len(areaz))):
    #     for iphase in tqdm(range(len(areay))):
    #         labels = []
    #         labels.append(pp.make_label(type="SET", label="PAR", value=islice))
    #         labels.append(pp.make_label(type="SET", label="LIN", value=iphase))
    #         for tr_index in range(len(M1s)):
    #             # if islice == len(areaz) - 1 and iphase == len(areay) - 1 and tr_index == len(M1s) -1:
    #                 # ifplot = True
    #             labels.append(pp.make_label(type="SET", label="SET", value=tr_index))
    #             seq.make_tr(areay[iphase], areaz[islice], M1s=M1s[tr_index], labels=labels)
    #             labels.pop()
    for i in tqdm(range(len(The_list))):
        # for i in range(4):
        for tr_index in range(4):
            if i==0 and tr_index == 3:
                seq.make_tr_nonc(The_list[i], Phi_list[i], M1s=M1s[tr_index], plot=True)
            else:
                seq.make_tr_nonc(The_list[i], Phi_list[i], M1s=M1s[tr_index], plot=False)

    print(seq.TE, seq.TR)
    print("TE, TR", seq.TE, seq.TR)
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
    if PLOT_KSPACE:
        # Show K-space sequence
        k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.seq.calculate_kspace()
        plt.figure(figsize=(30, 30))
        plt.plot(k_traj[0], k_traj[1])
        plt.plot(k_traj_adc[0], k_traj_adc[1], '.', ms=5)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()
