#! encoding = utf-8

""" Calculate theoretical SnR & other metrics of Voigt-1D window. """


from lib import *


def run_v1d_theory(filename, a0, b0, arange=(0, 5), brange=(-5, 5),
                   astep=0.02, bstep=0.02):
    """
    :arguments
        filename: str               export data to this file
        a0, b0: float               a0, b0 parameters for the FID envelope shape
        pa_range: tuple             pi*alpha range for the Kaiser window
        t_range: tuple              time range for the Kaiser window
        astep: float              pi*alpha incremental step
        bstep: float               time incremental step
        array_len: int              length of simulation array
    """
    a_min, a_max = arange
    b_min, b_max = brange
    x = np.linspace(-100, 100, 10**5+1)
    with open(filename, 'w') as f:
        f.write('#{:^4s} {:^4s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}\n'.format(
                'b', 'a', 'snr', 'fwhm', 'fw10db', 'fw20db', 'fw30db'))
        for b in np.arange(b_min, b_max + bstep, bstep):
            if not(int(b / bstep) % 10):
                print('b = {:.2f}'.format(b))
            for a in np.arange(a_min, a_max + astep, astep):
                peak = pk_v1d(a + a0, b + b0)
                snr = peak / np.sqrt(q_v1d(a, b))
                if a == 0 and b <= 0:     # constraint
                    f.write('{:>5.2f} {:>4.2f} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}\n'.format(
                            b, a, *['nan'] * 5
                    ))
                else:
                    # calculate spectral concentration: flux within 2*FWHM / total flux
                    y = voigt1d_ft_ana(x, a + a0, b + b0, yshift=0)
                    this_fwhm = fwhm_num(x, y)
                    this_fw10db = fwxdb_num(x, y, 10)
                    this_fw20db = fwxdb_num(x, y, 20)
                    this_fw30db = fwxdb_num(x, y, 30)
                    f.write('{:>5.2f} {:>4.2f} {:>8.6f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}\n'.format(
                        b, a, snr, this_fwhm, this_fw10db, this_fw20db, this_fw30db)
                    )
            f.write('\n')


if __name__ == '__main__':

    for a0, b0 in [(1, 0), (0, 1), (0.25, 1)]:
        print('a0={:.2f}  b0={:d}'.format(a0, b0))
        run_v1d_theory('sample_data/v1d_theory_{:.2f}-{:d}.dat'.format(a0, b0),
                       a0, b0, astep=0.05, bstep=0.05)
        print()
