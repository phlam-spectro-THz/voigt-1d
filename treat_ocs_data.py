#! encoding = utf-8

""" Treat OCS data systematically
Steps:
Pre:
    * Split the dual chirp .tdf into 2 single chirps with correct LO header
    * Convert data to voltage and apply HP filter. Save the voltage tdf

1. Load data and header
2. FFT, get main frequency
3. Fit time domain using exp(-a(t+t0)^2-b(t+t0)sin(2pi f(t+t0) + phi)
   Fix a to Doppler, t0 to the expected relaxation time, free b, f and phi
4. Construct window function grid and make the grid
   for each (a, b), do window -> FFT -> Fit -> output fit parameter.
5. Construct Kaiser window function for reference
"""

import numpy as np
from os.path import join as path_join
from scipy.signal import cheby2, sosfiltfilt
import lmfit
import sqlite3
from lib import calc_delta_g_coeff, best_snr_zero_a, \
    best_snr_fwhm_zero_a, fwhm_ab, sig2vol, \
    apply_kaiser, apply_voigt1d, fwhm_ab_fid, \
    find_v1d_ab_at_fwhm


def pre_treatement(filelist, dir_='sample_data/OCS/'):
    """ Pre-treatement
    * Split the dual chirp .tdf into 2 single chirps with correct LO header
    * Convert data to voltage and apply HP filter. Save the voltage tdf
    """

    sos = cheby2(4, 60, 10, output='sos', fs=1e3, btype='highpass')
    counter = 1
    conn = sqlite3.connect('OCS_freq.db')
    c = conn.cursor()

    for f, phi01, phi02 in filelist:
        with open(path_join(dir_, f), 'r') as fh:
            hd = fh.readline().strip()
        hd_array = hd.split('|')
        if len(hd_array) == 14:     # int64
            raw = np.loadtxt(path_join(dir_, f), skiprows=1, dtype='int64')
            total_avg = int(hd_array[6])
            vol = sig2vol(raw / total_avg) * 1e-3
        else:
            vol = np.loadtxt(path_join(dir_, f), skiprows=1)
        fmin = float(hd_array[0])
        bw = float(hd_array[1])
        fmax = fmin + bw
        len_ = int(hd_array[4])
        delay = int(hd_array[5])
        c.execute("SELECT species, mass FROM ocs WHERE freq >= {:g} AND freq <= {:g}".format(fmin, fmax))
        try:
            species, mass = c.fetchone()
        except TypeError:
            print('cannot find record for ', counter, f)
            species = ''
            mass = 60

        if isinstance(phi01, type(None)):     # bad FID, skip
            pass
        else:
            new_hd1 = 'fmin={:g}|fmax={:g}|bw={:g}|lo={:g}|len={:d}|delay={:d}|' \
                      'species={:s}|mass={:g}|phi0={:g}'.format(
                       fmin, fmax, bw, fmin, len_, delay, species, mass, phi01)
            fid1 = vol[:, 0] * 1e3
            fid1 -= np.average(fid1[-100:])
            y = sosfiltfilt(sos, fid1)
            np.savetxt(path_join(dir_, 'OCS_fid_{:d}.dat'.format(counter)),
                       y, header=new_hd1, fmt='%9.6f')
            counter += 1

        if isinstance(phi02, type(None)):    # bad FID, skip
            pass
        else:
            new_hd2 = 'fmin={:g}|fmax={:g}|bw={:g}|lo={:g}|len={:d}|delay={:d}|' \
                      'species={:s}|mass={:g}|phi0={:g}'.format(
                       fmin, fmax, bw, fmax, len_, delay, species, mass, phi02)
            fid2 = vol[:, 1] * 1e3
            fid2 -= np.average(fid2[-100:])
            y = sosfiltfilt(sos, fid2)
            np.savetxt(path_join(dir_, 'OCS_fid_{:d}.dat'.format(counter)),
                       y, header=new_hd2, fmt='%9.6f')
            counter += 1

    conn.close()


def f2min_tds_1f(lmpar, x, y):
    """ Time domain function to minimize. Single freq component """

    v = lmpar.valuesdict()
    t0 = v['t0']
    s = v['s']
    a = v['a']
    b = v['b']
    base = v['base']
    phi = v['phi']
    f = v['f']

    y_exp = s * np.exp(- a * (x+t0)**2 - b * (x+t0))
    y_sin = np.sin(2*np.pi * f * x + phi)

    return y_exp * y_sin + base - y


def fit_time_domain(log='fit.log', dir_='../data/OCS/', cut=30, no_t0=False, num=51):
    """ Fit TDS """

    log_fmt = '{:>2d} {:^8s} {:>6g} {:>6g} {:>3g} {:>6g} {:>6.4f} ' \
              '{:>8.4f}({:6.4f}) {:>6.4f} {:>6.4f}({:>6.4f}) {:>7.2f}({:>4.2f}) ' \
              '{:>6.4f}({:6.4f}) {:>6.3f}({:>5.3f}) {:>8.4f} {:>9.3e}\n'
    with open(path_join(dir_, log), 'w') as f_log:
        f_log.write('#id species   fmin   fmax  bw    flo t0(fix)    freq(err)'
                    '    a(fix)     b(err)         phi(err)      s(err)'
                    '      base*1e-3(err) chisqr redchisqr\n'.format())
        for i in range(num):
            with open(path_join(dir_, 'OCS_fid_{:d}.dat'.format(i+1)), 'r') as f:
                hd = f.readline().strip()
            hd_array = hd.split('|')
            len_ = int(hd_array[4].split('=')[1])
            delay = int(hd_array[5].split('=')[1])
            fmin = float(hd_array[0].split('=')[1])
            fmax = float(hd_array[1].split('=')[1])
            bw = float(hd_array[2].split('=')[1])
            flo = float(hd_array[3].split('=')[1])
            species = hd_array[6].split('=')[1]
            mass = float(hd_array[7].split('=')[1])
            phi0 = float(hd_array[8].split('=')[1]) / 180 * np.pi

            fid = np.loadtxt(path_join(dir_, 'OCS_fid_{:d}.dat'.format(i+1)), skiprows=1)
            # do fft and find the main freq component
            test_y = np.abs(np.fft.rfft(fid))
            x = np.fft.rfftfreq(len(fid)) * 1e3
            f0 = x[np.argmax(test_y)]
            t_array = np.arange(len(fid)) * 1e-3
            # find excitation time
            t0 = (delay - len_ * (bw-f0) / bw) * 1e-3
            # find a0 and b0. a0 = pi^2 gamma_D^2 / 4ln2
            if flo == fmax:
                gammaD = calc_delta_g_coeff(mass, 300) * (fmax - f0)
            else:
                gammaD = calc_delta_g_coeff(mass, 300) * (fmin + f0)
            a0 = (np.pi * gammaD)**2 / (4 * np.log(2))
            if fmax < 75000:
                b0 = 0.1
            elif 75000 < fmax < 120000:
                b0 = 0.3
            elif 140000 < fmax < 200000:
                b0 = 0.03
            else:
                b0 = 0.00

            lmpar = lmfit.Parameters()
            if no_t0:
                s0 = np.max(fid[cut:]) * np.exp(a0 * cut**2 * 1e-6 + b0 * cut * 1e-3)
                lmpar.add('t0', value=0, vary=False)
                lmpar.add('b', value=b0+2*a0*t0, min=max(b0+2*a0*t0-1, 0),
                          max=b0+2*a0*t0+0.8, vary=True)
            else:
                # find scalar
                s0 = np.max(fid[cut:]) * np.exp(a0*(t0 + cut*1e-3)**2 + b0*(t0 + cut*1e-3))
                if fmax > 200000:
                    lmpar.add('t0', value=t0, min=t0-0.05, max=t0+0.05, vary=False)
                    lmpar.add('b', value=0, vary=False)
                else:
                    lmpar.add('t0', value=t0, min=0.3, max=0.832, vary=False)
                    lmpar.add('b', value=b0, min=0, max=b0+0.1, vary=True)
            lmpar.add('s', value=s0, min=s0*0.5, max=s0*2, vary=True)
            lmpar.add('a', value=a0, vary=False)
            lmpar.add('base', value=0, vary=True)
            lmpar.add('phi', value=phi0, min=phi0-np.pi/4, max=phi0+np.pi/4, vary=True)
            lmpar.add('f', value=f0, min=f0-2, max=f0+2, vary=True)

            minner = lmfit.Minimizer(f2min_tds_1f, lmpar,
                                     fcn_args=(t_array[cut:], fid[cut:]))
            res = minner.minimize()
            # save fit log result
            try:
                f_log.write(log_fmt.format(
                        i+1, species, fmin, fmax, bw, flo,
                        res.params['t0'].value,
                        res.params['f'].value, res.params['f'].stderr, a0,
                        res.params['b'].value, res.params['b'].stderr,
                        res.params['phi'].value*180/np.pi, res.params['phi'].stderr*180/np.pi,
                        res.params['s'].value, res.params['s'].stderr,
                        res.params['base'].value*1e3, res.params['base'].stderr*1e3,
                        res.chisqr, res.redchi
                ))
                print(i + 1, '{:.4f}({:.4f})'.format(res.params['b'].value, res.params['b'].stderr))
                hd = 'FID_fit|redchi={:.2e}|'.format(res.redchi)
                hd += '|'.join(
                        list('{:s}={:.6f}({:.6f})'.format(name, param.value, param.stderr)
                             for name, param in res.params.items()))
            except TypeError:
                f_log.write(log_fmt.format(
                        i + 1, species, fmin, fmax, bw, flo,
                        res.params['t0'].value,
                        res.params['f'].value, 0., a0,
                        res.params['b'].value, 0.,
                        res.params['phi'].value * 180 / np.pi, 0.,
                        res.params['s'].value, 0.,
                        res.params['base'].value*1e3, 0.,
                        0., 0.
                ))
                print(i + 1, '{:.4f}(Nan)'.format(res.params['b'].value))
                hd = 'FID_fit|Bad fit|'
                hd += '|'.join(
                        list('{:s}={:.6f}'.format(name, param.value)
                             for name, param in res.params.items()))
            # save fit file
            if no_t0:
                np.savetxt(path_join(dir_, 'OCS_fid_fit_{:d}_no-t0.fit'.format(i+1)),
                           np.column_stack((t_array[cut:], fid[cut:], res.residual)),
                           fmt=['%6.3f', '%9.6f', '%9.6f'], header=hd)
            else:
                np.savetxt(path_join(dir_, 'OCS_fid_fit_{:d}.fit'.format(i+1)),
                           np.column_stack((t_array[cut:], fid[cut:], res.residual)),
                           fmt=['%6.3f', '%9.6f', '%9.6f'], header=hd)


def load_fid_log(logfile, skip=1):
    """ Load information from the FID fit log """
    with open(logfile, 'r') as f:
        for i in range(skip):
            f.readline()
        for a_line in f:
            a_list = a_line.split()
            id_ = int(a_list[0])
            fmin = float(a_list[2])
            fmax = float(a_list[3])
            bw = float(a_list[4])
            flo = float(a_list[5])
            f = float(a_list[7].split('(')[0])
            a = float(a_list[8])
            b = float(a_list[9].split('(')[0])
            yield id_, fmin, fmax, bw, flo, f, a, b


def fit_ocs_fid():

    pre_treatement([
        ('OCS_60860.tdf', None, 110),
        ('OCS_60900.tdf', -170, None),
        ('OCS_71140.tdf', 90, None),
        ('OCS_73020.tdf', 25, -150),
        ('OCS_73100.tdf', -130, None),
        ('OCS_82980.tdf', -120, 90),
        ('OCS_85300.tdf', None, 60),
        ('OCS_94900.tdf', None, -90),
        ('OCS_106740.tdf', 135, -60),
        ('OCS_154180.tdf', 60, -55),
        ('OCS_158180.tdf', -90, -15),
        ('OCS_165980.tdf', 45, None),  # 2nd bad
        ('OCS_170380.tdf', 135, None),  # 2nd bad
        ('OCS_177780.tdf', -65, None),  # 2nd bad
        ('OCS_182580.tdf', 30, -140),
        ('OCS_189780.tdf', -135, None),
        ('OCS_194780.tdf', 35, None),
        ('OCS_201580.tdf', 120, None),
        ('OCS_225280.tdf', -60, 170),
        ('OCS_236980.tdf', 170, 120),
        ('OCS_243280.tdf', 120, -145),
        ('OCS_243580.tdf', -60, 150),
        ('OCS_248980.tdf', -20, 120),
        ('OCS_255580.tdf', 60, -35),
        ('OCS_260680.tdf', -35, None),
        ('OCS_267580.tdf', 15, 145),
        ('OCS_267880.tdf', -60, -45),
        ('OCS_272680.tdf', 110, 150),
        ('OCS_279880.tdf', -160, -20),
        ('OCS_280180.tdf', 0, 45),
        ('OCS_316480.tdf', -90, -120),
        ('OCS_328480.tdf', 90, 90),
    ])

    fit_time_domain(log='fid_fit_no-t0.log', no_t0=True)
    fit_time_domain(log='fid_fit.log', no_t0=False)


def run(outfile, skip=1, num=51, zp=20000, dir_='sample_data/OCS/',
        fid_log='fid_fit_no-t0.log'):
    """ New routine that runs selective OCS lines.
    Procedure:
        1. Read the fit.log file to get initial (a0, b0)
        2. Use analytical functions to find best (a, b) for SnR,
           and best (a_fwhm, b_fwhm) for SnR/FWHM
        3. Apply two window functions, fit Voigt profile and
           calculate SnR & FWHM
        4. Calculate T for truncated Kaisser:
           -a0*T^2 - b0*T = -1, and Kaisser length = 2T
        5. Apply truncated Kaisser & full length Kaisser,
           fit Voigt profile and calculate SnR & FWHM
    :arguments
        outfile: str    Output result filename
        skip: int       skipped lines in the fid log
        num: int        treated lines in the fid log
        zp: int         FID zero-padding length
        dir_: str       file directory
        fid_log: str    FID log
    """

    iter_fid = load_fid_log(path_join(dir_, fid_log), skip=skip)
    out_fmt = '{:>3d} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f} ' \
              '{:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f} ' \
              '{:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f} ' \
              '{:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f}  ' \
              '{:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f} ' \
              '{:>6d} {:>6.4f} {:>7.2f} {:>6d} {:>6.4f} {:>7.2f}\n'
    out_fmt2 = '{:>3d} {:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f} ' \
               '{:>6.4f} {:>6.4f} {:>6.4f} {:>6.4f} {:>7.2f} ' \
               '{:>6s} {:>6s} {:>6s} {:>6s} {:>7s} ' \
               '{:>6s} {:>6s} {:>6s} {:>6s} {:>7s} ' \
               '{:>6s} {:>6s} {:>6s} {:>6s} {:>7s} ' \
               '{:>6d} {:>6.4f} {:>7.2f} {:>6d} {:>6.4f} {:>7.2f}\n'
    hd = '# Column meaning\n' \
         '# ID: OCS data id\n' \
         '# A0: initial a0 parameter, fitted in time domain \n' \
         '# B0: initial b0 parameter, fitted in time domain \n' \
         '# VC_AB: FWHM of the unwindowed complex voigt line in theory, calculated using A & B values \n' \
         '# VC_VOIGT: actual FWHM of unwindowed complex voigt line, determined by the complex Voigt fit \n' \
         '# SNR: actual SnR of the unwindowed line \n' \
         '# A_MAX: Voigt-1D window function parameter a to get max SnR \n' \
         '# B_MAX: Voigt-1D window function parameter b to get max SnR \n' \
         '# V_AB: FWHM of the windowed line in theory, calculated using A & B values \n' \
         '# V_VOIGT: actual FWHM of the windowed line, determined by the Voigt fit \n' \
         '# SNR: actual SnR of the line windowed by (A_MAX, B_MAX)  \n' \
         '# A_FWHM: Voigt-1D window function parameter a to get max SnR/FWHM \n' \
         '# B_FWHM: Voigt-1D window function parameter b to get max SnR/FWHM \n' \
         '# V_AB: FWHM of the windowed line in theory, calculated using A & B values \n' \
         '# V_VOIGT: actual FWHM of the windowed line, determined by the Voigt fit \n' \
         '# SNR: actual SnR of the line windowed by (A_FWHM, B_FWHM) \n' \
         '# A_KB_F: Voigt-1D window function parameter a to get the same Voigt FWHM as the full length Kaiser window \n' \
         '# B_KB_F: Voigt-1D window function parameter b to get the same Voigt FWHM as the full length Kaiser window \n' \
         '# V_AB: FWHM of the windowed line in theory, calculated using A & B values \n' \
         '# V_VOIGT: actual FWHM of the windowed line, determined by the Voigt fit \n' \
         '# SNR: actual SnR of the line windowed by (A_KB_F, B_KB_F) \n' \
         '# A_KB_TR: Voigt-1D window function parameter a to get the same Voigt FWHM as the truncated Kaiser window \n' \
         '# B_KB_TR: Voigt-1D window function parameter b to get the same Voigt FWHM as the truncated Kaiser window \n' \
         '# V_AB: FWHM of the windowed line in theory, calculated using A & B values \n' \
         '# V_VOIGT: actual FWHM of the windowed line, determined by the Voigt fit \n' \
         '# SNR: actual SnR of the line windowed by (A_KB_TR, B_KB_TR) \n' \
         '# LEN_KBF: length of the full length Kaiser (pi*a=8)\n' \
         '# V_KBF: actual FWHM of the line windowed by full length Kaiser (pi*a=8)\n' \
         '# SNR: actual SnR of the line windowed by full length Kaiser (pi*a=8)\n' \
         '# LEN_KBTR: length of the truncated Kaiser (pi*a=8)\n' \
         '# V_KBTR: actual FWHM of the line windowed by truncated Kaiser (pi*a=8) \n' \
         '# SNR: actual SnR of the line windowed by truncated Kaiser (pi*a=8) \n'

    log_lines = [hd,
                 '# 1    2      3      4      5      6   |'
                 '   7      8      9     10      11  |'
                 '  12     13     14     15      16  |'
                 '  17      18     19     20      21  |'
                 '  22      23     24     25     26  |'
                 '   27    28      29  |   30    31      32  \n',
                 '#ID   A0     B0  |VC_AB VC_VOIGT   SNR |'
                 ' A_MAX  B_MAX  V_AB  V_VOIGT  SNR  |'
                 'A_FWHM B_FWHM  V_AB V_VOIGT   SNR  |'
                 'A_KB_F  B_KB_F  V_AB  V_VOIGT  SNR  |'
                 'A_KB_TR B_KB_TR V_AB  V_VOIGT SNR  |'
                 'LEN_KBF V_KBF    SNR |LEN_KBTR V_KBTR  SNR\n']

    count = 0
    while count < num:
        # load fit log file
        try:
            id_, fmin, fmax, bw, flo, fpk, a0, b0 = next(iter_fid)
        except StopIteration:
            break
        count += 1
        # load FID data
        fid = np.loadtxt(path_join(dir_, 'OCS_fid_{:d}.dat'.format(id_)), skiprows=1)

        # get (a, b) for highest SnR
        a_snr, b_snr = best_snr_zero_a(a0, b0, x0=np.array([np.pi * (a0 + b0)]))
        vv_ab_snr = fwhm_ab(a_snr + a0, b_snr + b0)
        # get (a, b) for highest SnR/FWHM
        a_fwhm, b_fwhm = best_snr_fwhm_zero_a(a0, b0, x0=(1e-2, b0))
        vv_ab_fwhm = fwhm_ab(a_fwhm + a0, b_fwhm + b0)

        # apply winf to FID and calculate actual SnR & FWHM

        # Voigt-1D for max SnR
        snr_max_snr, vv_fit_max_snr = apply_voigt1d(
                fid, (fpk, ), (1,), a0, b0, a_snr, b_snr, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30, ftype='voigt',
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_voigt1d_max_snr.fit'.format(id_))
        )
        # Voigt-1D for max SnR/FWHM
        snr_max_snr_fwhm, vv_fit_max_snr_fwhm = apply_voigt1d(
                fid, (fpk,), (1,), a0, b0, a_fwhm, b_fwhm, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30,
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_voigt1d_max_snr_fwhm.fit'.format(id_))
        )
        # Full length Kaiser
        snr_kaiser, vv_fit_kaiser, len_kb = apply_kaiser(
                fid, (fpk, ), (1, ), a0, b0, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30, trunc=False,
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_kaiser_full.fit'.format(id_))
        )
        # Now find the (a, b) setting to get the same FWHM as full length kaiser
        a_kb_f, b_kb_f = find_v1d_ab_at_fwhm(fid, zp, bw, vv_fit_kaiser,
                                             vv_fit_kaiser / 2, 0,
                                             (fpk, ), (1, ), a0, b0, ftype='voigt')
        vv_ab_kbf = fwhm_ab(a_kb_f, b_kb_f)
        snr_kbf, vv_fit_kbf = apply_voigt1d(
                fid, (fpk, ), (1,), a0, b0, a_kb_f, b_kb_f, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30, ftype='voigt',
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_voigt1d_fwhm_eq_kaiser_full.fit'.format(id_))
        )
        # truncated Kaiser
        snr_kaiser_tr, vv_fit_kaiser_tr, len_kb_tr = apply_kaiser(
                fid, (fpk,), (1,), a0, b0, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30, trunc=True,
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_kaiser_trunc.fit'.format(id_))
        )
        # Now find the (a, b) setting to get the same FWHM as truncated Kaiser
        a_kb_tr, b_kb_tr = find_v1d_ab_at_fwhm(fid, zp, bw, vv_fit_kaiser_tr,
                                               vv_fit_kaiser_tr, 0,
                                               (fpk,), (1,), a0, b0, ftype='voigt')
        vv_ab_kb_tr = fwhm_ab(a_kb_tr, b_kb_tr)
        snr_kb_tr, vv_fit_kb_tr = apply_voigt1d(
                fid, (fpk,), (1,), a0, b0, a_kb_tr, b_kb_tr, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30, ftype='voigt',
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_voigt1d_fwhm_eq_kaiser_trunc.fit'.format(id_))
        )
        # finally, unwindowed line (complex voigt), using the length of truncated Kaiser
        snr_raw, vv_fit_raw = apply_voigt1d(
                fid[:len_kb_tr], (fpk,), (1,), a0, b0, 0, 0, zp, bw, flo,
                up=(flo == fmax), f_cutoff=30, ftype='complex-voigt',
                dx_snr=20, dx_snr_mode='inside',
                outfile=path_join(dir_, 'OCS_{:d}_raw_fid.fit'.format(id_))
        )
        vv_ab_raw = fwhm_ab_fid(a0, b0)

        this_line = out_fmt.format(
                id_, a0, b0, vv_ab_raw, vv_fit_raw, snr_raw,
                a_snr, b_snr, vv_ab_snr, vv_fit_max_snr, snr_max_snr,
                a_fwhm, b_fwhm, vv_ab_fwhm, vv_fit_max_snr_fwhm, snr_max_snr_fwhm,
                a_kb_f, b_kb_f, vv_ab_kbf, vv_fit_kbf, snr_kbf,
                a_kb_tr, b_kb_tr, vv_ab_kb_tr, vv_fit_kb_tr, snr_kb_tr,
                len_kb, vv_fit_kaiser, snr_kaiser,
                len_kb_tr, vv_fit_kaiser_tr, snr_kaiser_tr
        )
       
        log_lines.append(this_line)
        print(id_)

    with open(outfile, 'w') as fh:
        for a_line in log_lines:
            fh.write(a_line)
        last_line = '      0      0      0      1       0' \
                    '      0      0      0      1       0' \
                    '      0      0      0      1       0' \
                    '      0      0      0      1       0' \
                    '       0      0      0      1       0' \
                    '      0      1       0      0      1       0'
        # this last line is all 0. this is for gnuplot to make closed fillsteps
        fh.write('{:>3d}{:s}\n'.format(num + 1, last_line))


if __name__ == '__main__':

    fit_ocs_fid()
    run('sample_data/OCS_voigt1d_vs_kaiser.log', skip=1, fid_log='fid_fit_no-t0.log')
