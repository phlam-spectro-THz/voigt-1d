#! encoding = utf-8

"""
Fit time domain FID signal
"""

import numpy as np
from lib import best_snr_zero_a, best_snr_fwhm_zero_a, apply_kaiser, apply_voigt1d, \
    best_snr_fix_1p, best_snr_fwhm_fix_1p


def run_si(rawfile, out_prefix=''):

    # load FID
    fid = np.loadtxt(rawfile, skiprows=1, usecols=[1])

    with open(rawfile, 'r') as fh:
        hd_str = fh.readline()
    hd_list = hd_str.split('|')
    t1 = int(hd_list[0].split('=')[1])
    a0 = float(hd_list[6][3:11])
    b0 = float(hd_list[7][3:11]) # + 2*a0*float(hd_list[3][4:11])
    b0_err = float(hd_list[7][13:20])
    # b0 = 2*a0*float(hd_list[3][4:11])
    print(t1, a0, b0, '+|-', b0_err)

    ch3cn_f1 = 183675.9549
    ch3cn_f2 = (183676.6903 + 183676.7873) / 2
    bw = 60
    flo = 183660 + bw
    pk1 = ch3cn_f1
    pk2 = ch3cn_f2
    s1 = 0.5
    s2 = 1.0
    zp = 20000

    # get (a, b) for highest SnR
    a_snr, b_snr = best_snr_zero_a(a0, b0, x0=np.array([np.pi * (a0 + b0)]))
    # get (a, b) for highest SnR/FWHM
    a_fwhm, b_fwhm = best_snr_fwhm_zero_a(a0, b0, np.array([b0]))

    # apply winf to FID and calculate actual SnR & FWHM
    apply_kaiser(fid, (pk1, pk2), (s1, s2), a0, b0, zp, bw, flo,
                 up=True, f_cutoff=20., trunc=True, dx_snr=3,
                 outfile='sample_data/ch3cn/{:s}ch3cn_kaiser_trunc.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, 0, 0, zp, bw, flo, up=True,
                  f_cutoff=20., ftype='complex-voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_raw.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_snr, b_snr,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_fwhm, b_fwhm,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr_fwhm.fit'.format(out_prefix))

    # semi-manual options:
    # fixing b=-b0 for max SnR and SnR/FWHM
    a_snr_fwhm_neg_b, _ = best_snr_fwhm_fix_1p(a0, b0, np.array([a0 + b0 / 2]), pfix='b', pvalue=-b0)
    a_snr_neg_b, _ = best_snr_fix_1p(a0, b0, np.array([3*(a0+b0)]), pfix='b', pvalue=-b0)
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_snr_neg_b, -b0,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr_neg_b.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_snr_fwhm_neg_b, -b0,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr_fwhm_neg_b.fit'.format(out_prefix))


def run_article(rawfile, out_prefix=''):

    # load FID
    fid = np.loadtxt(rawfile, skiprows=1, usecols=[1])

    with open(rawfile, 'r') as fh:
        hd_str = fh.readline()
    hd_list = hd_str.split('|')
    t1 = int(hd_list[0].split('=')[1])
    a0 = float(hd_list[6][3:11])
    b0 = float(hd_list[7][3:11]) # + 2*a0*float(hd_list[3][4:11])
    b0_err = float(hd_list[7][13:20])
    # b0 = 2*a0*float(hd_list[3][4:11])
    print(t1, a0, b0, '+|-', b0_err)

    ch3cn_f1 = 183675.9549
    ch3cn_f2 = (183676.6903 + 183676.7873) / 2
    bw = 60
    flo = 183660 + bw
    pk1 = ch3cn_f1
    pk2 = ch3cn_f2
    s1 = 0.5
    s2 = 1.0
    pkc = (pk1 * s1 + pk2 * s2) / (s1 + s2)
    zp = 20000

    # get (a, b) for highest SnR
    a_snr, b_snr = best_snr_zero_a(a0, b0, x0=np.array([np.pi * (a0 + b0)]))
    # get (a, b) for highest SnR/FWHM
    a_fwhm, b_fwhm = best_snr_fwhm_zero_a(a0, b0, np.array([b0]))
    # get (a, b) for resolution
    a_res = a0
    b_res = - 2 * np.sqrt(a0)

    # apply winf to FID and calculate actual SnR & FWHM
    # manually trunc fid to t=3300 ns
    apply_kaiser(fid[:3300], (pk1, pk2), (s1, s2), a0, b0, zp, bw,
                 flo=flo, up=True, f_cutoff=20., trunc=False,
                 outfile='sample_data/ch3cn/{:s}_kaiser.fit'.format(out_prefix))
    apply_voigt1d(fid, (pkc,), (1,), a0, b0, 0, 0, zp, bw, up=True,
                  flo=flo, f_cutoff=20., ftype='complex-voigt',
                  outfile='sample_data/ch3cn/{:s}_raw.fit'.format(out_prefix))
    apply_voigt1d(fid, (pkc,), (1,), a0, b0, a_snr, b_snr,
                  zp, bw, flo=flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}_v1d_max_snr.fit'.format(out_prefix))
    apply_voigt1d(fid, (pkc,), (1,), a0, b0, a_fwhm, b_fwhm,
                  zp, bw, flo=flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}_v1d_max_snr-fwhm.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_res, b_res,
                  zp, bw, flo=flo, up=True, f_cutoff=20., ftype='voigt',
                  outfile='sample_data/ch3cn/{:s}_v1d_resolution.fit'.format(out_prefix))


if __name__ == '__main__':

    run_article('sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit', out_prefix='t450')
    run_article('sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit', out_prefix='t256')
    run_si('sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit', out_prefix='SI_t450_')
    run_si('sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit', out_prefix='SI_t256_')
