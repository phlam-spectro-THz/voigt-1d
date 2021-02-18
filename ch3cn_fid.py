#! encoding = utf-8

"""
Fit time domain FID signal
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
import lmfit
from lib import calc_delta_g_coeff, sig2vol
from v1d_theory import run_v1d_theory as run_sim_snr


def load_fid(filename, t1=256, t2=256 + 4096):
    """ Load FID from file
    :arguments
        filename: str           data file
        t1: int                 rect window starting time
        t2: int                 rect window end time
    :returns
        vol: np1darray          FID voltage (mV)
    """

    raw = np.loadtxt(filename, skiprows=1, dtype='int64')
    with open(filename, 'r') as f:
        hd_array = f.readline().strip().split('|')
    total_avg = int(hd_array[6])
    # the raw data is accumulative, take the average of it
    vol = sig2vol(raw / total_avg)
    # remove baseline drift using the last 100 points
    shift = np.average(vol[-100:])
    vol -= shift

    return vol[t1:t2]


def f2min_tds_simple(lmpar, x, y):
    """ Fitting function directly for time domain data, simple form """

    v = lmpar.valuesdict()
    t01 = v['t01']
    t02 = v['t02']
    s1 = v['s1']
    s2 = v['s2']
    a0 = v['a0']
    b0 = v['b0']
    base = v['base']
    phir1 = v['phi1'] / 180 * np.pi  # phase in radian
    phir2 = v['phi2'] / 180 * np.pi  # phase in radian
    f1 = v['f1']
    f2 = v['f2']

    # f_exp = s * np.exp(- a0 * (x + t0)**2 - b0 * (x + t0))
    f_exp = np.exp(- a0 * x ** 2 - b0 * x)
    f_sin = s1 * np.sin(2 * np.pi * f1 * x + phir1) \
            + s2 * np.sin(2 * np.pi * f2 * x + phir2)

    return f_exp * f_sin + base - y


def fit_fid_simple(fid, t1, f1, f2, s1, s2, a0, b0, t01, t02, ftol=0.2, phi=0., phitol=180.,
                   fix_phi=True, is_print=True, is_save=False, savefile=''):
    """ Fit the time domain spectrum directly to the voigt decay profile
    :arguments
        fid: np1darray      time domain data array
        t1: int             start time
        t0: float           pre-DAQ dead time
        t0tol: float        t0 tolerance
        fdiff: float        freq difference
        fsum: float         freq sum
        a0: float           Gaussian term (in freq domain)
        a0: float           Exponential term (in freq domain)
        phi: float          phase (deg)
        phitol: float       phase tolerance (deg)

        fix_t0: bool        Fix t0
        fix_fdiff: bool     Fix fdiff
        fix_fsum: bool      Fix fsum
        fix_delta: bool     Fix delta
        fix_phi: bool       Fix phase

        is_print: bool      Print fit result
        is_save: bool       Save fit result on disk

    :returns
        res: lmfit.result object
    """

    x = np.arange(len(fid)) * 1e-3

    lmpar = lmfit.Parameters()
    lmpar.add('s1', s1, min=1e-3, vary=True)
    lmpar.add('s2', s2, min=1e-3, expr='{:g}*s1'.format(s2/s1))
    lmpar.add('t01', t01, vary=False)
    lmpar.add('t02', t02, vary=False) #, expr='t01+{:.6f}'.format(t02-t01))
    lmpar.add('a0', a0, vary=False)
    lmpar.add('b0', b0, min=0, max=2.0, vary=True)
    lmpar.add('base', 0, min=-1e-2, max=1e-2, vary=True)
    lmpar.add('f1', f1, min=f1-ftol, max=f1+ftol, vary=True)
    lmpar.add('f2', f2, min=f2-ftol, max=f2+ftol, vary=True)
    lmpar.add('phi1', phi, min=phi - phitol, max=phi + phitol, vary=(not fix_phi))
    lmpar.add('phi2', phi, min=phi - phitol, max=phi + phitol, vary=(not fix_phi))
    # lmpar.add('psi', t0/fdiff, vary=True)

    # the first few points seem to be affected by filtering, and need to be excluded in the fit
    minner = lmfit.Minimizer(f2min_tds_simple, lmpar, fcn_args=(x, fid))
    # minner = lmfit.Minimizer(f2min_tds_no_t0, lmpar, fcn_args=(x, fid))
    res = minner.minimize()

    if is_print:
        print('chisqr: ', str(res.chisqr))
    par_list = []
    for name, param in res.params.items():
        try:
            par_list.append('{:s}={:.6f}({:.6f})'.format(name, param.value, param.stderr))
            if is_print:
                print('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, param.stderr))
        except (TypeError, ValueError):
            par_list.append('{:s}={:.6f}(nan)'.format(name, param.value))
            if is_print:
                print('{:7s} {:11.5f} nan'.format(name, param.value))

    if is_save and savefile:
        hd = 'FID_t1={:d}|redchi={:.2e}|'.format(t1, res.redchi)
        hd += '|'.join(par_list)
        np.savetxt(savefile, np.column_stack((x, fid, res.residual)),
                   fmt=['%.3f', '%9.6f', '%9.6f'], header=hd)

    return res


def run_simple(pp, t1=256, savefile='', sim_snr_file='', is_print=True):
    # settings
    ch3cn_f1 = 183675.9549
    ch3cn_f2 = (183676.6903 + 183676.7873) / 2
    ch3cn_inten1 = np.power(10, -0.4435)
    ch3cn_inten2 = np.power(10, -0.3996) + np.power(10, -0.4875)

    delta = calc_delta_g_coeff(41, 300) * (ch3cn_f1 + ch3cn_f2) / 2
    a0 = np.pi**2 * delta**2 / (4 * np.log(2))
    chirp_f_max = 183720.0  # LO
    chirp_bw = 60.0
    if1 = chirp_f_max - ch3cn_f1
    if2 = chirp_f_max - ch3cn_f2
    s1 = ch3cn_inten1 * 2
    s2 = ch3cn_inten2 * 2
    t2 = t1 + 4096  # rect window ending time
    fsum = 2 * chirp_f_max - (ch3cn_f2 + ch3cn_f1)
    # the actual chirp exication time. used for "t+t0" correction
    t0 = (1 - fsum / chirp_bw * 0.5) * 0.512 + t1 * 1e-3
    t01 = (1 - if1 / chirp_bw) * 0.512 + t1 * 1e-3
    t02 = (1 - if2 / chirp_bw) * 0.512 + t1 * 1e-3

    # iterate over all short chirp range data

    vol = load_fid('sample_data/ch3cn/short_3mTorr.tdf', t1=t1, t2=t2)
    # high pass filter
    sos = butter(3, 15, output='sos', fs=1e3, btype='highpass')
    fid = sosfiltfilt(sos, vol)
    res = fit_fid_simple(fid, t1, if1, if2, s1, s2, a0, 2*a0*t0, t01, t02, phi=0, phitol=180,
                         fix_phi=False, is_print=is_print, is_save=bool(savefile),
                         savefile=savefile)
    print('pp  a0     b0     b0_std   redchi')
    b0 = res.params['b0']
    print('{:>3d} {:6.4f} {:6.4f} {:6.4f} {:6.3e}'.format(
           pp, a0, b0.value, b0.stderr, res.redchi))
    # for n, p in res.params.items():
    #     print(n, p.value, p.stderr)

    if sim_snr_file:
        run_sim_snr(sim_snr_file, a0, res.params['b0'].value)


if __name__ == '__main__':

    run_simple(3, t1=450, savefile='sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit',)
    run_simple(3, t1=256, savefile='sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit')

