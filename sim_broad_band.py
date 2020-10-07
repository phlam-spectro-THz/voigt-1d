#! encoding = utf-8

import numpy as np
from scipy.signal.windows import kaiser
from lib import best_snr_zero_a, best_snr_fwhm_zero_a

CONST_AV = 6.02214086e23        # constant, Avogadro (mol-1)
CONST_KB = 1.38064852e-23       # constant, Boltzmann (J/K)
CONST_C = 2.99792458e8          # constant, speed of light (m_v1d/s)


def calc_a0(f, mass, temp):
    """ Calculate a0 parameter

    Arguments
        f: np1darray        line frequency, GHz
        mass: float         molecular mass (g/mol)
        temp: float         temperature (K)

    Returns
        a0: np1darray       a0 param, as size of f, GHz^2
    """

    coeff = CONST_KB * CONST_AV / CONST_C**2 * 1e3
    return 2 * coeff * np.pi**2 * temp * np.power(f, 2) / mass


def print_v1d_par(fc, cplen=500, delay=250, mass=60, temp=300, ll0=0.1,):
    """ Print Voigt-1D window function parameter """

    # delay times for each peak
    a0_avg = calc_a0(fc, mass, temp) * 1e6  # convert to MHz^2
    t0_avg = 0.5 * cplen + delay
    b0_avg = ll0 * np.pi + 2 * a0_avg * t0_avg * 1e-3  # MHz
    a_snr, b_snr = best_snr_zero_a(a0_avg, b0_avg)
    a_fwhm, b_fwhm = best_snr_fwhm_zero_a(a0_avg, b0_avg, x0=np.array([b0_avg]))
    print('a0_avg={:6.4f}, b0_avg={:6.4f}, a_snr={:6.4f}, b_snr={:6.4f}, '
          'a_fwhm={:6.4f}, b_fwhm={:6.4f}'.format(
            a0_avg, b0_avg, a_snr, b_snr, a_fwhm, b_fwhm))


def run(catfile='', fidfile='', ftfile='', band=(10, 20), fs=50, cplen=500,
        delay=250, datalen=10000, mass=60, temp=300, ll0=0.1, num=1000,
        a_pow=0.5, rand_phi=False, noise=0.01):
    """ Simulate a broadband FID signal
    :arguments
        catfile: str            JPL/CDMS catalog file (to sim real molecular spectra)
        fidfile: str            FID output filename
        ftfile: str             FT output filename
        band: tuple             chirp band (start, stop), GHz
        fs: float               sampling frequency, GHz
        cplen: int              chirp lenght, ns
        delay: int              DAQ delay, ns
        datalen: int            data length, ns
        mass: float             molecular weight, g/mol
        temp: float             temperature, K
        ll0: float              Lorentzian FWHM, MHz
        num: int                number of peaks
        a_pow: float            power distribution param a, for intensity generation
                                the peak intensities are generated using a power distribution
        rand_phi: bool          If True, randomize the initial phase of each peak
                                If False, generate pure cosine wave for each peak
        noise: float            noise level. Note that max(FID)=1.0
    """

    # note all time units are in ns, and frequency units in GHz.
    flow, fup = band
    bw = fup - flow

    if catfile:     # if catalog file is specified, read real spectral lines
        data = np.loadtxt(catfile, usecols=[0, 2])
        fpks_mol = data[:, 0] * 1e-3    # convert to GHz
        logint = data[:, 1]
        intens = np.power(10, logint - np.max(logint))
        # make the strongest line as intensity 1
        fpks_if = fpks_mol - flow
        if rand_phi:
            phis = np.random.rand(len(fpks_mol)) * np.pi * 2
        else:
            phis = np.zeros_like(fpks_mol)
    else:
        fpks_if = np.random.rand(num) * bw      # IF of peaks, [0, bw]
        fpks_mol = fpks_if + flow               # actual freq of peaks, [flow, fup]
        intens = np.random.power(a_pow, num)
        if rand_phi:
            phis = np.random.rand(num) * np.pi * 2
        else:
            phis = np.zeros(num)

    # delay times for each peak
    t0s = (1 - fpks_if / bw) * cplen + delay
    # calculate initial a0 & b0 values.
    a0s = calc_a0(fpks_mol, mass, temp)
    a0_avg = calc_a0((flow + fup)/2, mass, temp) * 1e6  # convert to MHz^2
    t0_avg = 0.5 * cplen + delay
    b0 = ll0 * np.pi * 1e-3             # GHz
    b0_avg = ll0 * np.pi + 2 * a0_avg * t0_avg * 1e-3   # MHz

    t = np.arange(0, datalen, 1/fs)     # unit: ns
    fid = np.zeros_like(t)
    for i, fpk in enumerate(fpks_if):
        fid += intens[i] * np.cos(2*np.pi * fpk * (t+t0s[i]) + phis[i]) \
               * np.exp(-a0s[i] * (t+t0s[i])**2 - b0 * (t+t0s[i]))
    # add noise
    fid += np.random.randn(len(t)) * noise

    np.savetxt(fidfile, fid, fmt='%.8f')

    zp = 4
    x = np.fft.rfftfreq(len(t)*zp) * fs
    y = np.abs(np.fft.rfft(fid, len(t)*zp, norm='ortho'))
    idx = x < bw

    a_snr, b_snr = best_snr_zero_a(a0_avg, b0_avg)
    a_fwhm, b_fwhm = best_snr_fwhm_zero_a(a0_avg, b0_avg, x0=np.array([b0_avg]))
    print('a0_avg={:6.4f}, b0_avg={:6.4f}, a_snr={:6.4f}, b_snr={:6.4f}, '
          'a_fwhm={:6.4f}, b_fwhm={:6.4f}'.format(
            a0_avg, b0_avg, a_snr, b_snr, a_fwhm, b_fwhm))

    wf = np.exp(- a_snr * (t*1e-3)**2 - b_snr * 1e-3 * t) * t * 1e-3
    y_v1d = np.abs(np.fft.rfft(fid * wf, len(t)*zp, norm='ortho'))
    wf = np.exp(- a_fwhm * (t*1e-3)**2 - b_fwhm * 1e-3 * t) * t * 1e-3
    y_v1d_fwhm = np.abs(np.fft.rfft(fid * wf, len(t)*zp, norm='ortho'))
    y_kaiser = np.abs(np.fft.rfft(fid * kaiser(len(t), 8), len(t)*zp, norm='ortho'))

    data = np.column_stack((x, y/np.max(y), y_v1d/np.max(y_v1d),
                            y_v1d_fwhm/np.max(y_v1d_fwhm),
                            y_kaiser/np.max(y_kaiser)))

    np.savetxt(ftfile, data[idx, :], fmt=['%8.6f'] + ['%9.6f']*4,
               header='   x       yraw    v1d_snr   v1d_fwhm   kaiser')


def simulate():
    run(catfile='sample_data/Catalogs/CAT_furcis_10-20.cat',
        fidfile='sample_data/broadband_sim_chirp_cis-furfural_10-20_FID.dat',  # FID output filename
        ftfile='sample_data/broadband_sim_chirp_cis-furfural_10-20_FT.txt',  # FT output filename
        band=(10, 20),  # bandwidth, GHz
        fs=50,  # sampling frequency, GHz
        cplen=500,  # chirp length, ns
        delay=250,  # DAQ delay, ns
        datalen=10000,  # data length, ns
        mass=96,   # molecular mass, g/mol
        temp=300,  # temperature, K
        ll0=0.1,  # Lorentzian FWHM, MHz
        rand_phi=True,  # randomize phase if True. If False, generate pure cosine wave for all peaks
        noise=0.3,  # noise level,
        )

    run(catfile='sample_data/Catalogs/CAT_nmf.cat',
        fidfile='sample_data/broadband_sim_chirp_N-MF_640-650_FID.dat',  # FID output filename
        ftfile='sample_data/broadband_sim_chirp_N-MF_640-650_FT.txt',  # FT output filename
        band=(640, 650),  # bandwidth, GHz
        fs=50,  # sampling frequency, GHz
        cplen=500,  # chirp length, ns
        delay=250,  # DAQ delay, ns
        datalen=625,  # data length, ns
        mass=59,  # molecular mass, g/mol
        temp=300,  # temperature, K
        ll0=0.1,  # Lorentzian FWHM, MHz
        rand_phi=True,  # randomize phase if True. If False, generate pure cosine wave for all peaks
        noise=0.1,  # noise level
        )


if __name__ == '__main__':

    print('cis-furfural 10-20 GHz')
    print_v1d_par(15, mass=96)
    print('N-methylformamide 640-650 GHz')
    print_v1d_par(645, mass=59)

    simulate()      # simulate spectrum takes long so wrap it for easier disabling


