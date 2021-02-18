#! encoding = utf-8

""" Calculate theoretical SnR & other metrics of Kaiser-Bessel window.
Some Values are calculated numerically.
"""


import numpy as np
from scipy.signal.windows import kaiser
from lib import fwhm_num, fwxdb_num, interp_symm


def run_kaiser_theory(filename, a0, b0, pa_range=(0, 16), t_range=(0, 5),
                      pa_step=0.1, t_step=0.1, array_len=2**10, zp_len=2**16):
    """
    :arguments
        filename: str               export data to this file
        a0, b0: float               a0, b0 parameters for the FID envelope shape
        pa_range: tuple             pi*alpha range for the Kaiser window
        t_range: tuple              time range for the Kaiser window
        pa_step: float              pi*alpha incremental step
        t_step: float               time incremental step
        array_len: int              length of simulation array
        zp_len: int                 zero-padding length to get numerical fwhm
    """
    t_min, t_max = t_range
    pa_min, pa_max = pa_range

    with open(filename, 'w') as f:
        f.write('#{:^4s} {:^4s} {:^8s} {:^8s} {:^9s} {:^9s} {:^9s}\n'.format(
                't', 'pi*a', 'snr', 'fwhm', 'fw10db', 'fw20db', 'fw30db'))
        for t in np.arange(t_min + t_step, t_max + t_step, t_step):
            print('t = {:.2f}'.format(t))
            # generate time array
            x = np.linspace(0, t, num=array_len)
            # generate FID envelope array
            y = np.exp(-a0 * x**2 - b0*x)
            for pa in np.arange(pa_min, pa_max + pa_step, pa_step):
                # generate kaiser window
                kw = kaiser(array_len, pa)
                # calculate numerical FWHM
                yft = np.fft.fftshift(np.abs(np.fft.fft(kw * y, zp_len)))
                xft = np.fft.fftshift(np.fft.fftfreq(zp_len)) / t * array_len
                fwhm = fwhm_num(xft, yft)
                # calculate noise
                noise = np.sqrt(np.sum(kw**2))
                # calculate peak intensity
                # because yft is zero-padded, we need to rescale it back to the
                # same scale of the noise
                peak = yft.max() / np.sqrt(zp_len // array_len)
                snr = peak / noise
                # calculate baseline resolution FW30DB
                # interpolate the profile to remove ringing
                yintrp = interp_symm(xft, yft, zp_len // array_len)
                fw10db = fwxdb_num(xft, yintrp, 10)
                fw20db = fwxdb_num(xft, yintrp, 20)
                fw30db = fwxdb_num(xft, yintrp, 30)
                f.write('{:>5.2f} {:>4.1f} {:>8.6f} {:>8.4f} {:>9.4f} {:>9.4f} {:>9.4f}\n'.format(
                        t, pa, snr, fwhm, fw10db, fw20db, fw30db
                ))
            f.write('\n')


if __name__ == '__main__':

    for a0, b0 in [(1, 0), (0, 1), (0.25, 1)]:
        print('a0={:.2f}  b0={:d}'.format(a0, b0))
        run_kaiser_theory('sample_data/kaiser_theory_{:.2f}-{:d}.dat'.format(a0, b0),
                          a0, b0, pa_step=0.1, t_step=0.1)
        print()
