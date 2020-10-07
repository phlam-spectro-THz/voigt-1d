#! encoding = utf-8

"""
Fit time domain FID signal
"""

import numpy as np
from lib import best_snr_zero_a, best_snr_zero_b, best_snr_fwhm_zero_a, apply_kaiser, apply_voigt1d, \
    best_snr_fwhm_zero_b, find_v1d_ab_at_fwhm, best_snr_fix_1p, best_snr_fwhn_fix_1p


def run(rawfile, out_prefix=''):

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
    flo = 183660
    bw = 60
    pk1 = flo + bw - ch3cn_f1
    pk2 = flo + bw - ch3cn_f2
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
    snr, vv0, klen = apply_kaiser(fid, (pk1, pk2), (s1, s2), a0, b0, zp, bw, flo,
                                  up=True, f_cutoff=20., trunc=False, dx_snr=3,
                                  outfile='sample_data/ch3cn/{:s}ch3cn_kaiser_full.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, 0, 0, zp, bw, flo, up=True,
                  f_cutoff=20., ftype='complex-voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_raw.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_snr, b_snr,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_fwhm, b_fwhm,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr_fwhm.fit'.format(out_prefix))
    # now treat the FID again with wf that gets the same vv0 as Kaiser
    a, b = find_v1d_ab_at_fwhm(fid, zp, bw, vv0, vv0 / 2, -b0,
                               (pk1, pk2), (s1, s2), a0, b0, ftype='voigt')
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a, b, zp, bw, flo,
                  up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_fwhm_eq_kaiser.fit'.format(out_prefix))

    # semi-manual options:
    # fixing b=-b0 for max SnR and SnR/FWHM
    a_snr_fwhm_neg_b, _ = best_snr_fwhn_fix_1p(a0, b0, np.array([a0+b0/2]), pfix='b', pvalue=-b0)
    a_snr_neg_b, _ = best_snr_fix_1p(a0, b0, np.array([3*(a0+b0)]), pfix='b', pvalue=-b0)
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_snr_neg_b, -b0,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr_neg_b.fit'.format(out_prefix))
    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a_snr_fwhm_neg_b, -b0,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='voigt', dx_snr=3,
                  outfile='sample_data/ch3cn/{:s}ch3cn_voigt1d_max_snr_fwhm_neg_b.fit'.format(out_prefix))


def test_run():
    # load FID
    fid = np.loadtxt('sample_data/ch3cn/FID_3mTorr.fit', skiprows=1, usecols=[1])

    with open('sample_data/ch3cn/FID_3mTorr.fit', 'r') as fh:
        hd_str = fh.readline()
    hd_list = hd_str.split('|')
    a0 = float(hd_list[6][3:11])
    b0 = float(hd_list[7][3:11])  # + 2*a0*float(hd_list[3][4:11])
    # b0 = 2*a0*float(hd_list[3][4:11])
    print(a0, b0)

    ch3cn_f1 = 183675.9549
    ch3cn_f2 = (183676.6903 + 183676.7873) / 2
    flo = 183660
    bw = 60
    pk1 = flo + bw - ch3cn_f1
    pk2 = flo + bw - ch3cn_f2
    s1 = 0.5
    s2 = 1.0
    zp = 20000
    vv0 = 0.6088
    a_init = 0.3
    b_init = -b0
    a, b = find_v1d_ab_at_fwhm(fid, zp, bw, vv0, a_init, b_init,
                               (pk1, pk2), (s1, s2), a0, b0, ftype='gaussian')
    print("FWHM at 0.6088")
    print('a_init, b_init = ', a_init, b_init)
    print('a_fit, b_fit = ', a, b)

    apply_voigt1d(fid, (pk1, pk2), (s1, s2), a0, b0, a, b,
                  zp, bw, flo, up=True, f_cutoff=20., ftype='gaussian',
                  outfile='sample_data/ch3cn/ch3cn_voigt1d_target_fwhm.fit')


if __name__ == '__main__':

    run('sample_data/ch3cn/ch3cn_FID_3mTorr_450ns.fit', out_prefix='t450_')
    run('sample_data/ch3cn/ch3cn_FID_3mTorr_256ns.fit', out_prefix='t256_')
