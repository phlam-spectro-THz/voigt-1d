#! encoding = utf-8

""" Calculate theoretical SnR & FWHM of Voigt-1D window,
With fixed a, b parameters, and varying a0, b0
Compare "averaged" window parameter v.s. optimal window parameter
"""

from lib import *


def calc_a0b0_array(fgrid, ll0, gcf0, tcp, td):
    """ Calculate the a0 & b0 array
    a0 = pi^2 * (gcf0 * fi)^2 / 4ln2
    b0 = pi*ll0 + 2 * a0 * [td + (fup-fi)/bw * tcp]
    :arguments
        fgrid: array        frequency grid
    """
    flow = fgrid.min()
    fup = fgrid.max()
    fc = (fup + flow) / 2
    a0_array = (np.pi * gcf0 * fgrid) ** 2 / (4 * np.log(2))
    t = td + (fup - fgrid) / (fup - flow) * tcp
    b0_array = np.pi * ll0 + 2 * a0_array * t
    a0_avg = (np.pi * gcf0 * fc) ** 2 / (4 * np.log(2))
    b0_avg = np.pi * ll0 + 2 * a0_avg * (td + tcp / 2)
    return a0_array, b0_array, a0_avg, b0_avg


def run_v1d(filename, fcn, fgrid, a0_avg, b0_avg,
            a0_array, b0_array, coeff=1.0):
    """ b parameter is calculated on the fly with fcn
    We calc 2 sets, one opt par (0, b) for (a0_avg, b0_avg)
    one opt par (0, b) for (a0, b0) that loops the a0_array
    so we compare the metrics of these 2 sets of data
    :arguments
        filename: str               export data to this file
        fcn: function               optimization function
        fgrid: array                frequency array (used for file output)
        a0_array, b0_array          FID parameters
        coeff: float                guess coefficient for fcn
    """
    x = np.linspace(-100, 100, 10**5+1)
    a_avg, b_avg = fcn(a0_avg, b0_avg, x0=np.array([coeff * (a0_avg + b0_avg)]))
    with open(filename, 'w') as f:
        f.write('# a0_avg={:.4f}, b0_avg = {:.4f}, b_avg = {:.4f}\n'.format(
                a0_avg, b0_avg, b_avg))
        f.write('#{:^5s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}\n'.format(
                'f', 'a0', 'b0', 'b-each', 'snr-each', 'fwhm-each', 'snr-avg', 'fwhm-avg'))
        for fi, a0, b0 in zip(fgrid, a0_array, b0_array):
            a_each, b_each = fcn(a0, b0, x0=np.array([coeff * (a0 + b0)]))
            # calculate metric for b-each
            snr_each = pk_v1d(a0 + a_each, b_each + b0) / np.sqrt(q_v1d(a_each, b_each))
            y = voigt1d_ft_ana(x, a0 + a_each, b_each + b0, yshift=0)
            fwhm_each = fwhm_num(x, y)
            # calculate metric for b_avg
            snr_avg = pk_v1d(a0 + a_avg, b_avg + b0) / np.sqrt(q_v1d(a_avg, b_avg))
            y = voigt1d_ft_ana(x, a0 + a_avg, b_avg + b0, yshift=0)
            fwhm_avg = fwhm_num(x, y)
            f.write('{:>6.1f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.6f} {:>8.4f} {:>8.6f} {:>8.4f}\n'.format(
                    fi, a0, b0, b_each, snr_each, fwhm_each, snr_avg, fwhm_avg
                    ))


if __name__ == '__main__':

    # microwave, doppler dominated: a0 range with large span
    fgrid = np.arange(2, 8.1, 0.1)
    a0_arr, b0_arr, a0_avg, b0_avg = calc_a0b0_array(fgrid, 0, 0.05, 4, 0)
    print('a0: {:.4f} - {:.4f}'.format(a0_arr[0], a0_arr[-1]))
    print('b0: {:.4f} - {:.4f}'.format(b0_arr[0], b0_arr[-1]))
    run_v1d('sample_data/SI_v1d_fixed_ab_mw_doppler_snr.dat',
            best_snr_fix_1p, fgrid, a0_avg, b0_avg, a0_arr, b0_arr,
            coeff=np.pi)
    run_v1d('sample_data/SI_v1d_fixed_ab_mw_doppler_snr-fwhm.dat',
            best_snr_fwhm_fix_1p, fgrid, a0_avg, b0_avg, a0_arr, b0_arr,
            coeff=1)

    # submillimeter, doppler dominated: a0 range with small span but large value
    fgrid = np.arange(590, 600.05, 0.1)
    a0_arr, b0_arr, a0_avg, b0_avg = calc_a0b0_array(fgrid, 0, 0.003, 0.025, 0)
    print('a0: {:.4f} - {:.4f}'.format(a0_arr[0], a0_arr[-1]))
    print('b0: {:.4f} - {:.4f}'.format(b0_arr[0], b0_arr[-1]))
    run_v1d('sample_data/SI_v1d_fixed_ab_submm_doppler_snr.dat',
            best_snr_fix_1p, fgrid, a0_avg, b0_avg, a0_arr, b0_arr,
            coeff=np.pi)
    run_v1d('sample_data/SI_v1d_fixed_ab_submm_doppler_snr-fwhm.dat',
            best_snr_fwhm_fix_1p, fgrid, a0_avg, b0_avg, a0_arr, b0_arr,
            coeff=1)

    # Lorentzian dominated: b0 range much larger than a0
    fgrid = np.arange(2, 8.1, 0.1)
    a0_arr, b0_arr, a0_avg, b0_avg = calc_a0b0_array(fgrid, 2, 0.05, 4, 1)
    print('a0: {:.4f} - {:.4f}'.format(a0_arr[0], a0_arr[-1]))
    print('b0: {:.4f} - {:.4f}'.format(b0_arr[0], b0_arr[-1]))
    run_v1d('sample_data/SI_v1d_fixed_ab_lorentzian_snr.dat',
            best_snr_fix_1p, fgrid, a0_avg, b0_avg, a0_arr, b0_arr,
            coeff=np.pi)
    run_v1d('sample_data/SI_v1d_fixed_ab_lorentzian_snr-fwhm.dat',
            best_snr_fwhm_fix_1p, fgrid, a0_avg, b0_avg, a0_arr, b0_arr,
            coeff=1)
