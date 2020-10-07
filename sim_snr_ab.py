#! encoding = utf-8

""" Simulate SnR/FWHM values for a grid of (a, b) """

import numpy as np
from lib import snr_theo, fwhm_ab


def run(a0, b0, a_range=(0, 5), b_range=(-5, 5), res=0.02, outfile='sim_snr_ab.dat'):

    a_min, a_max = a_range
    b_min, b_max = b_range
    with open(outfile, 'w') as fh:
        fh.write('# a0={:g} b0={:g} \n'.format(a0, b0))
        fh.write('# {:<2s} {:^5s} {:^9s} {:^8s} {:^8s} \n'.format('a', 'b', 'VV_AB', 'SNR', 'SNR/FWHM'))
        for a in np.arange(a_min, a_max+res, res):
            for b in np.arange(b_min, b_max+res, res):
                this_snr = snr_theo((a, b), a0, b0)
                this_fwhm = fwhm_ab(a0+a, b0+b)
                if this_fwhm == float('inf'):
                    fh.write('{:>4.2f} {:>5.2f} {:^9s} {:>8s} {:>8s}\n'.format(
                            a, b, 'nan', 'nan', 'nan'
                    ))
                else:
                    fh.write('{:>4.2f} {:>5.2f} {:>9.6f} {:>8.6f} {:>8.6f}\n'.format(
                        a, b, this_fwhm, this_snr, this_snr / this_fwhm
                    ))
            fh.write('\n')
            print('a = {:g}'.format(a))


if __name__ == '__main__':

    run(1.0, 0.0, a_range=(0, 5), b_range=(-5, 5), res=0.02,
        outfile='sample_data/sim_snr_ab_1.dat')
    run(0.0, 1.0, a_range=(0, 5), b_range=(-5, 5), res=0.02,
        outfile='sample_data/sim_snr_ab_2.dat')
    run(0.25, 1.0, a_range=(0, 5), b_range=(-5, 5), res=0.02,
        outfile='sample_data/sim_snr_ab_3.dat')
