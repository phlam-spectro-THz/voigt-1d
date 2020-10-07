#! encoding = utf-8

import numpy as np
from math import floor, log10
from copy import deepcopy
from scipy.special import erfc, wofz, erfcx
from scipy.optimize import minimize, root
from scipy.signal.windows import kaiser
import lmfit

CONST_AV = 6.02214086e23        # constant, Avogadro (mol-1)
CONST_KB = 1.38064852e-23       # constant, Boltzmann (J/K)
CONST_C = 2.99792458e8          # constant, speed of light (m_v1d/s)


def _alpha(delta):
    """ Convert Gaussian FWHM delta to short notation _alpha """

    return np.pi**2 * delta**2 / (4 * np.log(2))


def _beta(lambda_, delta, t0=0):
    """ Convert Gaussian FWHM delta to short notation _alpha """

    return np.pi * lambda_ + 2 * _alpha(delta) * t0


def _norm2x(xp, xrange, xcenter):
    """ Convert normalized x in (-1, 1) back to data x unit

    Arguments
        xp: float | np1darray       in (-1, 1)
        xrange: float               MHz
        xcenter: float              MHz

    Returns
        x: float | np1darray
    """

    return xp / 2 * xrange + xcenter


def _x2norm(x, xrange, xcenter):
    """ Convert x in data unit to normalized x in [-100: 100] %

    Arguments
        x: float | np1darray
        xrange: float (MHz)
        xcenter: float (MHz)

    Returns
        xp: float | np1darray       in (-1, 1)
    """

    return (x - xcenter) / xrange * 2


def _norm2res(res_norm, xrange, xcenter, yscale):
    """ Convert normalized fit result back to original values

    Arguments
        res_norm: lmfit.result object
        xrange: float       for rescaling x
        xcenter: float      for rescaling x
        yscale: float       for rescaling y

    Returns
        res: lmfit.result object, value updated
    """

    if isinstance(res_norm, type(None)):
        return None
    else:
        res = deepcopy(res_norm)

    for name, param in res.params.items():
        # for each parameter rescale value
        # get original value from res_norm to avoid self iteration of linked
        # parameters
        if name.startswith('p') or name.startswith('s'):
            param.set(value=res_norm.params[name].value*yscale,
                      min=res_norm.params[name].min*yscale,
                      max=res_norm.params[name].max*yscale)
        elif name.startswith('x'):
            # peak position: convert from (-1, 1) to original
            param.set(value=_norm2x(res_norm.params[name].value, xrange, xcenter),
                      min=_norm2x(res_norm.params[name].min, xrange, xcenter),
                      max=_norm2x(res_norm.params[name].max, xrange, xcenter))
        else:
            param.set(value=_norm2x(res_norm.params[name].value, xrange, 0),
                      min=_norm2x(res_norm.params[name].min, xrange, 0),
                      max=_norm2x(res_norm.params[name].max, xrange, 0))
        # rescale stderr if there is stderr
        if isinstance(param.stderr, float):
            if name.startswith('p') or name.startswith('s'):
                param.stderr = res_norm.params[name].stderr * yscale
            else:
                param.stderr = _norm2x(res_norm.params[name].stderr, xrange, 0)
        else:
            pass

    res.residual *= yscale
    res.chisqr *= yscale**2
    res.redchi *= yscale**2

    return res


def sig2vol(raw_int):
    """ Convert ADC card signal integer to voltage

    Arguments
        raw_int: np.array int64     raw signal
        v_offset: int               raw2vol voltage offset
        v_res: int                  raw2vol voltage resolution
        v_range: float              raw2vol voltage range (V)
        v_dc: int                   raw2vol DC offset

    Returns
        vol: np.array float64       voltage array
    """

    v_offset = -1  # voltage offset
    v_res = -32768  # voltage resolution
    v_range = 2000  # voltage range (mV)
    v_dc = 0  # voltage DC offset
    vol = (v_offset - raw_int) * v_range / (2 * v_res) + v_dc

    return vol


def lorentzian(x, ll, yshift=0.):
    """ Peak normalized Lorentzian function
    :arguments
        x: float / np1darray
        ll: float       Lorentzian FWHM
        yshift: float   shift of y value (used for root finding)
    :returns
        y: float / np1darray
    """

    return ll**2 / (4 * x**2 + ll**2) - yshift


def gaussian(x, gg, yshift=0.):
    """ Peak normalized Gaussian function
    :arguments
        x: float / np1darray
        ll: float       Lorentzian FWHM
        yshift: float   shift of y value (used for root finding)
    :returns
        y: float / np1darray
    """

    return np.exp(-(x / gg)**2 * 4 * np.log(2)) - yshift


def voigt(x, gg, ll, yshift=0.):
    """ Peak normalized Voigt function
    :arguments
        x: float / np1darray
        gg: float       Gaussian FWHM
        ll: float       Lorentzian FWHM
        yshift: float   shift of y value (used for root finding)
    :returns
        y: float / np1darray
    """

    # calculate Gaussian stdev and Lorentzian gamma
    sigma = gg / (2 * np.sqrt(2 * np.log(2)))
    gamma = ll / 2
    # the complex z for the Faddeeva function
    # z = (x/gamma + 1j) * np.sqrt(2) * sigma / (4*gamma)
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    ymax = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    return np.real(wofz(z)) / ymax - yshift


def complex_voigt(x, gg, ll, yshift=0.):
    """ Peak normalized complex Voigt function
    :arguments
        x: float / np1darray
        gg: float       Gaussian FWHM
        ll: float       Lorentzian FWHM
        yshift: float   shift of y value (used for root finding)
    :returns
        y: float / np1darray
    """

    # calculate Gaussian stdev and Lorentzian gamma
    sigma = gg / (2 * np.sqrt(2 * np.log(2)))
    gamma = ll / 2
    # the complex z for the Faddeeva function
    # z = (x/gamma + 1j) * np.sqrt(2) * sigma / (4*gamma)
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    ymax = np.abs(wofz(1j * gamma / (sigma * np.sqrt(2))))
    return np.abs(wofz(z)) / ymax - yshift


def fid_waveform(t, lambda_, delta, t0, norm=False):
    """ The FID waveform
    :argument
        t: np1darray        time array
        lambda_: float      Lorentzian FWHM
        delta: float        Gaussian FWHM
        t0: float           pre-acquisition dead time
        norm: bool          normalization by area
    """

    y = np.exp(-_alpha(delta) * (t + t0)**2 - np.pi * lambda_ * (t + t0))

    if norm:
        y0 = np.exp(-_alpha(delta) * t0**2 - np.pi * lambda_ * t0)
        if _alpha(delta) > 0:
            area = y0 * np.sqrt(np.pi / (4 * _alpha(delta))) \
                   * erfc(_beta(lambda_, delta, t0) / (2 * np.sqrt(_alpha(delta)))) \
                   * np.exp(_beta(lambda_, delta, t0)**2 / (4 * _alpha(delta)))
        else:
            area = y0 / (np.pi * lambda_)
        return y / area
    else:
        return y


def f2root_raw_fid_max_snr(x, a0, b0):
    """ Function for root finding for getting the truncate length of FID
    root x0 for y == 0 gives the maximum x^(-1/2) * intg[fid(t)dt, t=0,x],
    i.e., the maximum SnR for a unwindowed FID
    :argument
        x: float        integral upper bound T for intg[fid(t)dt, t=0,x]
        a0: float       a0 parameter for FID exp(-a0*t^2-b0*t)
        b0: float       b0 parameter for FID exp(-a0*t^2-b0*t)
    :returns
        y: float        function value
    """
    if x >= 0:  # x needs to be positive
        if a0 == 0:      # pure exponential
            return (2 * b0 * x + 1) * np.exp(-b0 * x) - 1
        else:
            return pk_raw_tup(x, a0, b0) - 2 * x * np.exp(-a0 * x**2 - b0 * x)
    else:
        return float('inf')


def get_fid_tr_len(a0, b0, tunit=1e3):
    """ Return the truncate length of the raw FID for highest SnR
    It is finding the maximum t_max for function:
        1/sqrt(T) * integral[exp(-at^2-bt), t=[0, T]]
    :argument
        a0: float       a0 parameter for raw FID
        b0: float       b0 parameter for raw FID
        tunit: float    specify time unit conversion
    :return
        tr_len: int     int(t_max * tunit)
    """

    if a0 < 0:
        raise ValueError('a0 cannot be negative')
    elif a0 == 0:     # taking f(tau)=1/e for initial guess
        x0 = 1 / b0
    else:
        x0 = (np.sqrt(b0**2 + 4 * a0) - b0) / (2*a0)
    sol = root(f2root_raw_fid_max_snr, x0, args=(a0, b0))
    return int(sol.x[0] * tunit)


def fwhm_num(x, y):
    """ Numerically determine the FWHM: find the 1/2 maximum intercept
    and calculate the width
    :argument
        x: np1darray        x data
        y: np1darray        y data
    :returns
        fwhm_ab: float         fwhm_ab
    """

    y_max = np.max(y)
    idx_y_max = np.argmax(y)
    # find x where y(x) = y_max / 2
    # the data may not have points exactly at y_max / 2, so we need to find the
    # points closest to it
    # find 4 points: 2 to the left and 2 to the right of the peak
    # 2 just above y_max / 2 and 2 just below y_max / 2
    y_diff = y - y_max / 2
    # to the left
    x_left = x[:idx_y_max]
    y_left = y[:idx_y_max]
    y_diff_left = y_diff[:idx_y_max]
    idx = np.argmin(y_diff_left[y_diff_left >= 0])
    x1 = x_left[y_diff_left >= 0][idx]
    y1 = y_left[y_diff_left >= 0][idx]
    idx = np.argmax(y_diff_left[y_diff_left < 0])
    x2 = x_left[y_diff_left < 0][idx]
    y2 = y_left[y_diff_left < 0][idx]
    # to the right
    x_right = x[idx_y_max:]
    y_right = y[idx_y_max:]
    y_diff_right = y_diff[idx_y_max:]
    idx = np.argmin(y_diff_right[y_diff_right >= 0])
    x3 = x_right[y_diff_right >= 0][idx]
    y3 = y_right[y_diff_right >= 0][idx]
    idx = np.argmax(y_diff_right[y_diff_right < 0])
    x4 = x_right[y_diff_right < 0][idx]
    y4 = y_right[y_diff_right < 0][idx]
    # now fit a quadratic function to it
    p = np.polyfit([x1, x2, x3, x4], [y1, y2, y3, y4], deg=2)
    # find the root poly[p] = y_max / 2. p starts from highest order
    # the difference of the two roots is just 2 * sqrt(b^2-4ac)/2a
    a, b, c = p
    c -= y_max / 2
    fwhm = np.sqrt(b**2 - 4*a*c) / abs(a)

    return fwhm


def calc_delta_g_coeff(mass, temp):
    """ Calculate Guassian line width coefficient

    Arguments
        mass: float         molecular mass (g/mol)
        temp: float         temperature (K)

    Returns
        coeff: float        Gaussian FWHM coefficient
    """

    mass_si = mass * 1e-3 / CONST_AV
    coeff = np.sqrt(8 * np.log(2) * CONST_KB * temp / mass_si) / CONST_C
    return coeff


def convert_fds_freq(x, y, bw, flo=None, f_cutoff=0, is_dual=False):
    """ Convert the frequency axis of the frequency domain spectrum.

    Arguments
        x: np1darray            fft frequency array
        y: np1darray            fft amplitude array
        bw: float               Bandwidth (MHz)
        flo: float              LO frequency (MHz). If None, return fft freq
        f_cutoff: float         Frequency to cutoff at the edge
        is_dual: bool           Dual cut

    Returns
        xc: np1darray           converted frequency array
        yc: np1darray           corresponding amplitude array

    flo is the maximum frequency (detection frequency) of the pulse.
    The last frequency in the xc array == flo-f_cutoff
    If flo==None, then return the fft freq array.

    If is_dual, cut off both (0, f_cutoff) and (flo-f_cutoff, flo).
    Else, cut off only (0, f_cutoff) range.
    The edge of the xc array is adjusted so that when concatenating
    two freq domain spectra, the frequency spacing is kept constant at
    the joint, e.g. always 0.1 in this example
        0, 0.1, 0.2, ..., 0.9~|~0, 0.1, 0.2, ..., 0.9 | ...
    """

    # a small tolerance to hanlde possible float point errors in comparison
    tol = 1e-6

    if is_dual:
        idx = np.logical_and(x > f_cutoff - tol, x < bw - f_cutoff - tol)
    else:
        idx = np.logical_and(x > f_cutoff - tol, x < bw - tol)

    if isinstance(flo, type(None)):
        xc = x[idx]
        yc = y[idx]
    else:
        xc = flo - x[idx]
        yc = y[idx]

    # check the direction of x. if x is monotonic decreasing, flip it
    if xc[0] > xc[-1]:
        return np.flip(xc), np.flip(yc)
    else:
        return xc, yc


def sciformat(a, a_std, err=1, digit=2, exp='auto'):
    """ convert number a to string in format used in scientific publication

    Number formated to xxx.xx(xx),
    where the uncertainty is expressed in the paranthesis,
    to the last digit of the value

    Arguments
        a: float        the value
        a_std: float    the standard deviation of the value
        err: float/int  the multiplier of a_std to put in the paranthesis.
                        Default is 1, i.e., the 1-sigma value
        digit: int      the digit of uncertainty in the paranthesis.
                        Default is 2 digits.
        exp: str        exponential option
            'auto'      automatic.
                        For both the value and the uncertainty,
                        if there are >3 leading 0s after the decimal point,
                        or >3 ending 0s before the decimal point,
                        use E+00 style to specify the order of magnitude.
                        Else, use normal plain digits.
            'plain'     force to plain digits.
            'force'     force to 0.00E+00 format

    Returns
        a_str: str      formatted string
    """

    # check if a_std is > 0
    if isinstance(a, type(None)):
        return 'Nan(Nan)'
    else:
        if exp == 'plain':
            a_fmt = '{:.4f}'
        elif exp == 'force':
            a_fmt = '{:.3e}'
        else:
            if 1e-3 < abs(a) < 1e3 or a == 0:
                a_fmt = '{:.3f}'
            else:
                a_fmt = '{:.3e}'

        if isinstance(a_std, type(None)):
            return a_fmt.format(a) + '(Nan)'
        elif a_std == 0:
            return a_fmt.format(a) + '(fix)'
        elif a_std == float('inf') or a_std == float('-inf'):
            return a_fmt.format(a) + '(inf)'
        elif abs(a) > abs(a_std*err):
            pass
        else:
            return a_fmt.format(a) + '(Nan)'
    try:
        mag_a = floor(log10(abs(a)))
        mag_e = floor(log10(abs(a_std*err)))
    except ValueError:
        mag_a = 0
        mag_e = 0

    # check if needs exponential
    # if both value and std have decimal points
    # (such as 32.2(1.2)), still need exponential. otherwise no need
    # this is top prioprity
    if mag_e >= 0 and digit-mag_e >= 2:
        is_exp = True
    else:
        if exp == 'force':
            is_exp = True
        elif exp == 'plain':
            is_exp = False
        elif exp == 'auto':
            if (mag_a < -3 and mag_e < -3) or (mag_a > 3 and mag_e > 3):
                is_exp = True
            else:
                is_exp = False
        else:
            raise KeyError('invalid "exp" option')

    # total digits that a needs
    digits_a = mag_a - mag_e + digit

    # calculate padding 0s
    n_zeros_before_decimal = max(mag_e+1-digit, 0)
    n_zeros_after_decimal = max(-mag_a-1, 0)

    # prepare exponential case
    if is_exp:
        # the leading value string is always x.xxx
        _v_str = '{{:.{:d}f}}'.format(digits_a-1).format(a*10**(-mag_a))
        # the err in paranthesis is alwasy xxx
        _err_str = '{:.0f}'.format(a_std*err*10**(digit-1-mag_e))
        # concatenate. separate positive and negative magnitude
        if mag_a >= 0:
            a_str = '{:s}({:s})E{:0=2d}'.format(_v_str, _err_str, mag_a)
        else:
            a_str = '{:s}({:s})E{:0=3d}'.format(_v_str, _err_str, mag_a)
    # prepare non-exponential case
    else:
        # if both value and error are large and need 0-padding before decimal
        if n_zeros_before_decimal > 0:
            _v_str = '{:.0f}'.format(a*10**(digits_a-1-mag_a)) \
                     + '0'*(mag_a+1-digits_a)
            _err_str = '{:.0f}'.format(a_std*err*10**(digit-1-mag_e)) \
                        + '0'*n_zeros_before_decimal
        # if both value and error are small and need 0-padding after decimal
        elif n_zeros_after_decimal > 0:
            # total digits == necessary digits + padding 0s.
            _v_str = '{{:.{:d}f}}'.format(n_zeros_after_decimal+digits_a).format(a)
            # error string is then without decimal
            _err_str = '{:.0f}'.format(a_std*err*10**(digit-1-mag_e))
        else:
            _v_str = '{{:.{:d}f}}'.format(digits_a-mag_a-1).format(a)
            _err_str = '{:.0f}'.format(a_std*err*10**(digit-1-mag_e))
        a_str = '{:s}({:s})'.format(_v_str, _err_str)

    return a_str


def f2min_voigt(lmpar, x, y, is_flip=False, order=0):
    """ Voigt function to be minimized.

    Arguments
        lmpar: lmfit params object
        x: np1darray            x array
        y: np1darray            y array
        der: int                derivative of the voigt profile
        is_flip: bool           flip intensity of the function

    Returns
        res: np.array = f(x)-y
    """

    # x = np.linspace(-100, 100, num=len(y), endpoint=True)
    v = lmpar.valuesdict()
    model = 0

    for n in range((len(v)-order-1)//4):
        model += voigt(x - v['x' + str(n)], v['gg' + str(n)],
                       v['ll' + str(n)]) * v['s' + str(n)]

    if is_flip:          # flip sign
        model = -model
    else:
        pass

    for i in range(order+1):
        model += v['p'+str(i)]*x**i

    return model - y


def f2min_gaussian(lmpar, x, y, order=0):
    """ Gaussian function to be minimized """

    v = lmpar.valuesdict()
    model = 0
    for i in range((len(v) - order - 1) // 3):
        model += gaussian(x - v['x'+str(i)], v['gg'+str(i)], yshift=0) * v['s'+str(i)]
    for i in range(order+1):
        model += v['p'+str(i)] * x**i

    return model - y


def f2min_lorentzian(lmpar, x, y, order=0):
    """ Lorentzian function to be minimized """

    v = lmpar.valuesdict()
    model = 0
    for i in range((len(v) - order - 1) // 3):
        model += lorentzian(x - v['x'+str(i)], v['ll'+str(i)], yshift=0) * v['s'+str(i)]
    for i in range(order+1):
        model += v['p'+str(i)] * x**i

    return model - y


def f2min_complex_voigt(lmpar, x, y, is_flip=False, order=0):
    """ Magnitude of complex Voigt function to be minimized.

    Arguments
        lmpar: lmfit params object
        x: np1darray            x array
        y: np1darray            y array
        der: int                derivative of the voigt profile
        is_flip: bool           flip intensity of the function

    Returns
        res: np.array = f(x)-y
    """

    # x = np.linspace(-100, 100, num=len(y), endpoint=True)
    v = lmpar.valuesdict()
    model = 0

    for n in range((len(v)-order-1)//4):
        model += complex_voigt(x - v['x' + str(n)], v['gg' + str(n)],
                               v['ll' + str(n)]) * v['s' + str(n)]

    if is_flip:          # flip sign
        model = -model
    else:
        pass

    for i in range(order+1):
        model += v['p'+str(i)]*x**i

    return model - y


def fit_spectrum(x, y, pks, snorms, gg, ll, link_gg=True, link_ll=True,
                 dx=0.5, ftype='voigt'):
    """ Fit frequency domain spectrum.
    :arguments
        x: np1darray        x array
        y: np1darray        y array
        pks: list of float          peak positions
        snorms: list of normalized intensity
                                    intensity values of each peak
        gg: float           Gaussian FWHM
        ll: float           Lorentzian FWHM
        link_gg: bool       link all gg
        link_ll: bool       link all ll
        dx: float           x boundary
        ftype: str          function type
            'voigt'
            'gaussian'
            'lorentzian'
            'complex-voigt'
    """

    xrange = np.ptp(x)
    xcenter = np.median(x)
    x_norm = _x2norm(x, xrange, xcenter)
    yscale = np.max(y)
    y_norm = y / yscale

    dxn = _x2norm(dx, xrange, 0)
    ggn = _x2norm(gg, xrange, 0)
    lln = _x2norm(ll, xrange, 0)

    # prepare init lmfit parameter for the
    lmpar = lmfit.Parameters()
    lmpar.add('p0', value=0, vary=True)
    for i in range(len(pks)):
        x0n = _x2norm(pks[i], xrange, xcenter)
        lmpar.add('x'+str(i), value=x0n, min=x0n - dxn, max=x0n + dxn)
        lmpar.add('s'+str(i), value=snorms[i], min=0, max=2)
        if ftype == 'gaussian':
            if link_gg and (i > 0):
                lmpar.add('gg'+str(i), value=ggn, min=0, max=max(2.5*ggn, ggn+1e-3),
                          expr='gg0')
            else:
                lmpar.add('gg'+str(i), value=ggn, min=0, max=max(2.5*ggn, ggn+1e-3))
        elif ftype == 'lorentzian':
            if link_ll and (i > 0):
                lmpar.add('ll'+str(i), value=lln, min=0, max=max(2.5*lln, lln+1e-3),
                          expr='ll0')
            else:
                lmpar.add('ll'+str(i), value=lln, min=0, max=max(2.5*lln, lln+1e-3))
        else:
            if link_gg and (i > 0):
                lmpar.add('gg'+str(i), value=ggn, min=0, max=max(2.5*ggn, ggn+1e-3),
                          expr='gg0')
            else:
                lmpar.add('gg'+str(i), value=ggn, min=0, max=max(2.5*ggn, ggn+1e-3))
            if link_ll and (i > 0):
                lmpar.add('ll'+str(i), value=lln, min=0, max=max(2.5*lln, lln+1e-3),
                          expr='ll0')
            else:
                lmpar.add('ll'+str(i), value=lln, min=0, max=max(2.5*lln, lln+1e-3))

    # fit the normalized data
    if ftype == 'gaussian':
        minner = lmfit.Minimizer(f2min_gaussian, lmpar, fcn_args=(x_norm, y_norm),
                                 nan_policy='propagate')
    elif ftype == 'lorentzian':
        minner = lmfit.Minimizer(f2min_lorentzian, lmpar, fcn_args=(x_norm, y_norm),
                                 nan_policy='propagate')
    elif ftype == 'voigt':
        minner = lmfit.Minimizer(f2min_voigt, lmpar, fcn_args=(x_norm, y_norm),
                                 nan_policy='propagate')
    elif ftype == 'complex-voigt':
        minner = lmfit.Minimizer(f2min_complex_voigt, lmpar, fcn_args=(x_norm, y_norm),
                                 nan_policy='propagate')
    else:
        raise ValueError('Unknown function type')
    res_norm = minner.minimize()
    # convert normalized result back to original scale
    res = _norm2res(res_norm, xrange, xcenter, yscale)

    return res


def to_mol_freq(x, y, bw, flo=None, up=True, f_cutoff=0.):
    """ Convert the frequency axis of the frequency domain spectrum to molecular freq

    Arguments
        x: np1darray            fft frequency array
        y: np1darray            fft amplitude array
        bw: float               Bandwidth (MHz)
        flo: float              LO frequency (MHz). If None, return fft freq
        f_cutoff: float         Frequency to cutoff at the edge
        up: bool                Chirp going up, flo=fmax

    Returns
        xc: np1darray           converted frequency array
        yc: np1darray           corresponding amplitude array

    flo is the detection frequency of the pulse.
    The last frequency in the xc array == flo-f_cutoff
    If flo==None, then return the fft freq array.

    The edge of the xc array is adjusted so that when concatenating
    two freq domain spectra, the frequency spacing is kept constant at
    the joint, e.g. always 0.1 in this example
        0, 0.1, 0.2, ..., 0.9~|~0, 0.1, 0.2, ..., 0.9 | ...
    """

    # a small tolerance to hanlde possible float point errors in comparison
    tol = 1e-6

    idx = np.logical_and(x > f_cutoff - tol, x < bw - tol)

    if isinstance(flo, type(None)):
        xc = x[idx]
        yc = y[idx]
    else:
        if up:
            xc = flo - x[idx]
            yc = y[idx]
        else:
            xc = x[idx] + flo
            yc = y[idx]

    # check the direction of x. if x is monotonic decreasing, flip it
    if xc[0] > xc[-1]:
        return np.flip(xc), np.flip(yc)
    else:
        return xc, yc


def pk_raw_inf(a, b):
    """ Peak intensity of DFT of raw FID signal exp(-at^2-bt) from 0 to inf """
    if a > 0:
        u = 0.5 * b / np.sqrt(a)
        return 0.5 * np.sqrt(np.pi / a) * erfcx(u)
    elif a == 0:
        return 1 / b
    else:
        return 0


def pk_raw_tup(t_up, a, b):
    """ Peak intensity of DFT of raw FID signal exp(-at^2-bt) from 0 to tup """

    if t_up <= 0:
        return 0
    elif a == 0:
        return (1 - np.exp(-b * t_up)) / b
    else:
        u1 = 0.5 * b / np.sqrt(a)
        u2 = np.sqrt(a) * t_up + u1
        fid = np.exp(-a * t_up**2 - b * t_up)
        return 0.5 * np.sqrt(np.pi / a) * (erfcx(u1) - fid * erfcx(u2))


def pk_v1d(a, b):
    """ Peak intenisty of DFT of sig(a, b) = t * exp(-at^2-bt)
    Note that b includes Lorentzian and t0 term.
    a=0 case needs special treatment because it degenerates to a pure exponential
    """

    if a > 0:
        u = b / (2*np.sqrt(a))
        return (1 - np.sqrt(np.pi) * u * erfcx(u)) / (2*a)
    elif a == 0:
        return 1 / b**2
    else:   # a < 0
        return 0


def q_v1d(a, b):
    """ Integral of w^2(t) = t^2 * exp(-2at^2 - 2bt)
        a=0 case needs special treatment because it degenerates to a pure exponential
    """

    if a > 0:
        u = b / np.sqrt(2 * a)
        q = np.sqrt(2 * np.pi) * (a + b**2) * erfcx(u) / (16 * np.sqrt(a**5)) \
                - b / (8 * a**2)
        return q
    elif a == 0 and b > 0:
        return 1 / (4 * b**3)
    else:
        return float('+inf')


def m_v1d(a, b):
    """ normalization factor of window function """
    if a > 0:
        x = (np.sqrt(b**2 + 8*a) - b) / (4*a)
        wf_max = x * np.exp(-0.5 * (1 + b * x))
    else:
        if b > 0:
            wf_max = np.exp(-1) / b
        else:
            wf_max = float('+inf')
    return wf_max


def snr_theo(x, a0, b0, neg=False):
    """ Get theoretical SnR of (a, b, a0, b0)
    :arguments
        x: nparray      (a, b)
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo: float      SnR value
    """
    a = x[0]
    b = x[1]
    if (a == 0 and b == 0) or (a0+a == 0 and b0+b == 0):
        return 0
    elif a < 0:     # negative a is not allowed
        return 0
    else:
        if neg:
            return - pk_v1d(a0 + a, b0 + b) / np.sqrt(q_v1d(a, b))
        else:
            return pk_v1d(a0 + a, b0 + b) / np.sqrt(q_v1d(a, b))


def snr_theo_fix_1p(x, a0, b0, neg=False, pfix='a0', pvalue=0):
    """ Get theoretical SnR of (a, b, a0, b0) with 1 initial parameter fixed to
    preset value
    :arguments
        x: nparray      (a, b)
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
        pfix: str       name of the fixed parameter 'a0' / 'b0'
        pvalue: float   the fixed value
    :returns:
        snr_theo: float      SnR value
    """

    if pfix == 'a':    # fix a, p is b
        b = x[0]
        a = pvalue
        # if a < 0:
        #     raise ValueError('a cannot be negative')
        # else:
        #     if b == 0 or (a0 == 0 and b0 + b == 0):
        #         return 0
        #     else:
        #         if neg:
        #             return - pk_v1d(a0 + a, b + b0) / np.sqrt(q_v1d(a, b))
        #         else:
        #             return - pk_v1d(a0 + b, b + b0) / np.sqrt(q_v1d(a, b))
    elif pfix == 'b':  # fix b, p is a
        a = x[0]
        b = pvalue
        # if a < 0:
        #     return 0
        # elif (a == 0 and b == 0) or (a0 + a == 0 and b0 + b == 0):
        #     return 0
        # else:
        #     if neg:
        #         return - pk_v1d(a0 + a, b0 + b) / np.sqrt(q_v1d(a, b))
        #     else:
        #         return pk_v1d(a0 + a, b0 + b) / np.sqrt(q_v1d(a, b))
    else:
        raise ValueError('unknown pfix')

    return snr_theo([a, b], a0, b0, neg=neg)


def snr_zero_a(x, a0, b0, neg=False):
    """ SnR a==0 for optimization of max[SnR] because it is always found when a=0
    :arguments
        x: nparray      (b, )
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo: float      SnR value
    """
    b = x[0]
    if (b == 0) or (a0 == 0 and b0+b == 0):
        return 0
    else:
        if neg:
            return - pk_v1d(a0, b0 + b) / np.sqrt(q_v1d(0, b))
        else:
            return pk_v1d(a0, b0 + b) / np.sqrt(q_v1d(0, b))


def snr_zero_b(x, a0, b0, neg=False):
    """ SnR a==0 for optimization of max[SnR] because it is always found when b=0
    :arguments
        x: nparray      (a, )
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo: float      SnR value
    """
    a = x[0]
    if a <= 0:
        return 0
    else:
        if neg:
            return - pk_v1d(a0 + a, b0) / np.sqrt(q_v1d(a, 0))
        else:
            return pk_v1d(a0 + a, b0) / np.sqrt(q_v1d(a, 0))


def snr_per_fwhm(x, a0, b0, neg=False):
    """ SnR per FWHM
    :arguments
        x: nparray      (a, b)
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo/fwhm_ab: float      SnR / fwhm_ab
    """

    return snr_theo(x, a0, b0, neg=neg) / fwhm_ab(x[0] + a0, x[1] + b0)


def snr_per_fwhm_fix_1p(x, a0, b0, neg=False, pfix='a0', pvalue=0):
    """ Get theoretical SnR of (a, b, a0, b0) with 1 initial parameter fixed to
    preset value
    :arguments
        x: nparray      (a, b)
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
        pfix: str       name of the fixed parameter 'a0' / 'b0'
        pvalue: float   the fixed value
    :returns:
        snr_theo: float      SnR value
    """

    if pfix == 'a':    # fix a, p is b
        b = x[0]
        a = pvalue
    elif pfix == 'b':  # fix b, p is a
        a = x[0]
        b = pvalue
    else:
        raise ValueError('unknown pfix')
    return snr_per_fwhm([a, b], a0, b0, neg=neg)


def snr_per_fwhm_zero_a(x, a0, b0, neg=False):
    """ SnR per FWHM, fixing a == 0
    :arguments
        x: float        b
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo/fwhm_ab: float      SnR / fwhm_ab
    """

    b = x[0]
    if a0 > 0 or (a0 == 0 and b > -b0):
        return snr_theo([0, b], a0, b0, neg=neg) / fwhm_ab(a0, b0 + b)
    else:
        return 0


def snr_per_fwhm_zero_b(x, a0, b0, neg=False):
    """ SnR per FWHM, fixing b == 0
    :arguments
        x: float        a
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo/fwhm_ab: float      SnR / fwhm_ab
    """

    a = x[0]
    if a > 0:
        return snr_theo([a, 0], a0, b0, neg=neg) / fwhm_ab(a + a0, b0)
    else:
        return 0


def best_snr_ab(a0, b0, x0=np.array([1e-3, 1])):
    """ Return the (a, b) pair that gives highest SnR of initial (a0, b0),
    """

    res = minimize(snr_theo, x0=x0, args=(a0, b0, True))
    return 0, res.x[0]


def best_snr_zero_a(a0, b0, x0=np.array([1])):
    """ Return the (a, b) pair that gives highest SnR of initial (a0, b0),
    where a=0 is fixed
    """

    res = minimize(snr_zero_a, x0=x0, args=(a0, b0, True))
    return 0, res.x[0]


def best_snr_zero_b(a0, b0, x0=np.array([1])):
    """ Return the (a, b) pair that gives highest SnR of initial (a0, b0),
    where b=0 is fixed
    """

    res = minimize(snr_zero_b, x0=x0, args=(a0, b0, True))
    return res.x[0], 0


def best_snr_fix_1p(a0, b0, x0=np.array([1]), pfix='a0', pvalue=0):

    res = minimize(snr_theo_fix_1p, x0=x0, args=(a0, b0, True, pfix, pvalue))
    if pfix == 'a':
        return pvalue, res.x[0]
    elif pfix == 'b':
        return res.x[0], pvalue
    else:
        raise ValueError('unknown pfix')


def best_snr_per_fwhm(a0, b0, x0=np.array([1e-3, 1])):
    """ Return the (a, b) pair that gives highest SnR / FWHM of initial (a0, b0)
    """
    res = minimize(snr_per_fwhm, x0=x0, args=(a0, b0, True))
    return res.x[0], res.x[1]


def best_snr_fwhm_zero_a(a0, b0, x0):
    """ Return the a value that gives highest SnR / FWHM of initial (a0, b0),
    where a=0 is fixed.
    """

    res = minimize(snr_per_fwhm_zero_a, x0=x0, args=(a0, b0, True))
    return 0, res.x[0]


def best_snr_fwhm_zero_b(a0, b0, x0):
    """ Return the a value that gives highest SnR / FWHM of initial (a0, b0),
    where b=0 is fixed.
    """

    res = minimize(snr_per_fwhm_zero_b, x0=x0, args=(a0, b0, True))
    return res.x[0], 0


def best_snr_fwhn_fix_1p(a0, b0, x0=np.array([1]), pfix='a0', pvalue=0):

    res = minimize(snr_per_fwhm_fix_1p, x0=x0, args=(a0, b0, True, pfix, pvalue))
    if pfix == 'a':
        return pvalue, res.x[0]
    elif pfix == 'b':
        return res.x[0], pvalue
    else:
        raise ValueError('unknown pfix')


def fid_ft_ana(x, a, b, yshift=0.5):
    """ The analytical magitude FT of raw FID exp(-a*x^2-b*x).
    :arguments
        x: float / np1darray    Frequency domain independent variable
        a: float
        b: float
        yshift: float       the relative y shift applied to the function
                            (with respect to ymax)
    """
    omega = 2 * np.pi * x
    if a > 0:
        a_sq = 2 * np.sqrt(a)
        pi_sq = np.sqrt(np.pi)
        z = (1j * omega + b) / a_sq
        y = np.abs(erfcx(z)) * pi_sq / a_sq
        y_max = np.abs(erfcx(b / a_sq)) * pi_sq / a_sq
        return y - yshift * y_max
    elif a == 0:  # pure lorentzian
        if b > 0:
            y = 1 / np.abs(b + 1j * omega)
            return y - yshift / b
        else:
            return np.zeros_like(x)
    else:
        return np.zeros_like(x)


def voigt1d_ft_ana(x, a, b, yshift=0.5):
    """ The analytical magitude FT of Voigt-1D function x*exp(-a*x^2-b*x).
    :arguments
        x: float / np1darray    Frequency domain independent variable
        a: float
        b: float
        yshift: float        the relative y shift applied to the function
                            (with respect to ymax)
    """

    omega = 2 * np.pi * x
    if a > 0:
        a_sq = 2 * np.sqrt(a)
        pi_sq = np.sqrt(np.pi)
        z = (-omega + 1j * b) / a_sq
        y = np.abs(1 + 1j * pi_sq * z * wofz(z))
        y_max = np.abs(1 - b * pi_sq / a_sq * wofz(1j * b / a_sq))
        return y - yshift * y_max
    elif a == 0:    # pure lorentzian
        if b > 0:
            y = 1 / np.abs((b + 1j * omega)**2)
            return y - yshift / (b**2)
        else:
            return np.zeros_like(x)
    else:
        return np.zeros_like(x)


def fwhm_voigt_fit(ll, gg, eq='accurate'):
    """ FWHM of Voigt fit, using approximation
    :arguments
        ll: float       fitted Lorentzian FWHM component
        gg: float       fitted Gaussian FWHM component
        eq: str   'rough' or 'accurate'
            'rough'  Rough equation vv = ll / 2 + sqrt(ll^2/4 + gg^2)
            'accurate':  More accurate approximation by Olivero and Longbothum 1977
    :returns
        fwhm: float     FWHM
    """

    if eq == 'rough':
        return ll / 2 + np.sqrt(ll**2 / 4 + gg**2)
    elif eq == 'accurate':
        return 0.5346 * ll + np.sqrt(0.2166 * ll**2 + gg**2)
    else:
        raise ValueError('invalid eq')


def fwhm_complex_voigt_fit(ll, gg):
    """ FWHM of complex voigt fit.
    We don't know an approximation, so the FWHM is determined by
    finding the root v(x0) = 0.5*v(0)

    :arguments
        ll: float       fitted Lorentzian FWHM component
        gg: float       fitted Gaussian FWHM component
    :returns
        fwhm: float     FWHM
    """
    # start initial guess by using the Voigt FWHM
    x0 = fwhm_voigt_fit(ll, gg)
    sol = root(complex_voigt, x0, args=(gg, ll, 0.5))
    return abs(sol.x[0]) * 2


def fwhm_ab_fid(a, b):
    """ Return the FWHM of the magnitude FT of the raw FID, i.e.,
    the magnitude of a complex voigt profile.
    The FWHM is determined numerically by finding the root v(x0) = 0.5 * v(0)

    :arguments:
        a: float
        b: float
    :returns
        fwhm_ab: float
    """

    if a <= 0 or (a == 0 and b <= 0):   # prevent the illegal case
        return float('inf')
    else:
        x0 = b / (4*np.pi) + 0.5 * np.sqrt(b**2 / (4*np.pi**2) + 4 * a * np.log(2) / np.pi**2)
        sol = root(fid_ft_ana, x0, args=(a, b))

    return 2 * sol.x[0]


def fwhm_ab(a, b):
    """ Return the FWHM of the magnitude FT of Voigt-1D function
    x*exp(-a*x^2-b*x).

    The FWHM is determined numerically by finding the root v(x0) = 0.5 * v(0)

    :arguments
        a: float
        b: float
    :returns
        fwhm_ab: float
    """

    if a < 0 or (a == 0 and b <= 0):   # prevent the illegal case
        return float('inf')
    else:
        x0 = b / (4*np.pi) + 0.5 * np.sqrt(b**2 / (4*np.pi**2) + 4 * a * np.log(2) / np.pi**2)
        # sol = root_scalar(mag_ft_ana, args=(a, b, pk_v1d(a, b)*0.5), method='bisect',
        #                   x0=x_init, bracket=[0, 2*x_init], xtol=1e-6)
        sol = root(voigt1d_ft_ana, x0, args=(a, b))

    return 2 * sol.x[0]


def fwhm_approx(a, b, eq='rough'):
    """ Approximate Voigt FWHM
    :arguments
        a: float    pi^2 * gg^2 / (4*log2)
        b: float    pi * ll
        eq: str   'rough' or 'accurate'
            'rough'  Rough equation vv = ll / 2 + sqrt(ll^2/4 + gg^2)
            'accurate':  Accurate equation
    :returns
        vv: float   Voigt FWHM approximated
    """

    ll = b / np.pi
    gg_sqr = 4 * a * np.log(2) / (np.pi**2)

    if eq == 'rough':
        return ll / 2 + np.sqrt(ll**2 / 4 + gg_sqr)
    elif eq == 'accurate':
        return 0.5346 * ll + np.sqrt(0.2166 * ll**2 + gg_sqr)
    else:
        raise ValueError('invalid eq')


def snr_per_fwhm_approx(x, a0, b0, neg=False):
    """ SnR per FWHM using approximated Voigt FWHM
    :arguments
        x: nparray      (a, b)
        a0: float       initial a0
        b0: float       initial b0
        neg: bool       return -snr_theo (for optimize.minimize)
    :returns:
        snr_theo/fwhm_ab: float      SnR / fwhm_ab
    """

    return snr_theo(x, a0, b0, neg=neg) / fwhm_approx(x[0] + a0, x[1] + b0)


def best_snr_per_fwhm_approx(a0, b0, x0=np.array([1, 1])):
    """ Return the (a, b) pair that gives highest SnR / FWHM of initial (a0, b0)
    Using approximated Voigt FWHM
    """
    res = minimize(snr_per_fwhm_approx, x0=x0, args=(a0, b0, True))
    return res.x


def f2min_at_fwhm(lmpar, vv0, fid, zp, bw, pks, snorms, a0, b0, ftype='voigt'):
    """ Minimizer Function to find the window function a, b value for an targeted FWHM
    :arguments
        lmpar: lmfit parameter object
        vv0: float              target FWHM
        fid: np1darray
        zp: int                 zero-padding length
        pks: list of float          peak positions
        snorms: list of normalized intensity
                                    intensity values of each peak
        ftype: str              lineshape function type
            'voigt'
            'gaussian'
            'lorentzian'
            'complex-voigt'
    """

    v = lmpar.valuesdict()
    t = np.arange(len(fid)) * 1e-3
    wf = np.exp(- v['a'] * t**2 - v['b'] * t) * t
    y = np.abs(np.fft.rfft(fid * wf, zp))
    x = np.fft.rfftfreq(zp) * 1e3
    xc, yc = to_mol_freq(x, y / np.max(y), bw, f_cutoff=20.)

    gg = 2 * np.sqrt((v['a'] + a0) * np.log(2)) / np.pi
    ll = (v['b'] + b0) / np.pi

    res = fit_spectrum(xc, yc, pks, snorms, gg, ll, ftype=ftype)
    if ftype == 'voigt':
        gg = res.params['gg0'].value
        ll = res.params['ll0'].value
        vv = fwhm_voigt_fit(ll, gg)
    elif ftype == 'gaussian':
        vv = res.params['gg0'].value
    elif ftype == 'lorentzian':
        vv = res.params['ll0'].value
    else:
        raise ValueError('Illegal target type')

    return vv0 - vv


def find_v1d_ab_at_fwhm(fid, zp, bw, vv0, a_init, b_init, pks, snorms, a0, b0,
                        ftype='voigt'):
    """ Return the window function a, b value for an FID so that it yields
    an targeted FWHM vv0
    : arguments
        fid: np1darray      FID signal
        zp: int             zero-padding length
        vv0: float          target FWHM
        a_init: float       initial guess a
        b_init: float       initial guess b
        a0: float           a0 value of the FID
        b0: float           b0 value of the FID
        ftype: str          lineshape function type
            'voigt'
            'gaussian'
            'lorentzian'
            'complex-voigt'
    """

    lmpar = lmfit.Parameters()
    lmpar.add('a', value=a_init, min=max(0, a_init-0.25), max=a_init+0.25, vary=True)
    lmpar.add('b', value=b_init, vary=False)

    minner = lmfit.Minimizer(f2min_at_fwhm, lmpar,
                             fcn_args=(vv0, fid, zp, bw, pks, snorms, a0, b0),
                             fcn_kws={'ftype': ftype},
                             nan_policy='propagate')
    res = minner.minimize(method='least_squares')
    return res.params['a'].value, res.params['b'].value


def apply_voigt1d(fid, pks, snorms, a0, b0, a, b, zp, bw, flo,
                  up=True, link_gg=True, link_ll=True, dx_snr=None, dx_snr_mode='outside',
                  f_cutoff=0., ftype='voigt', outfile=''):

    """ Apply Voigt-1D window function, fit the spectrum, and return the treated SnR & FWHM
    :arguments
        fid: np1darray      FID signal
        pks: list of float  peak positions
        snorms: float       The normalized intensities of the peaks
        a0: float           FID initial a0
        b0: float           FID initial b0
        a: float            Voigt-1D window parameter a
        b: float            Voigt-1D window parameter b
        zp: int             zero-padding length
        fpk: float          peak frequency
        bw: float           chirp bandwidth
        flo: float          LO frequency
        up: bool            is chirp going up
        link_gg: bool       link all Gaussian FWHM
        link_ll: bool       link all Lorentzian FWHM
        dx_snr: float       +/- dx_snr from pk to calculate noise & SnR
        dx_snr_mode: str
            'inside'        take pk - dx_snr < x < pk + dx_snr as noise range
            'outside'       take x < pk - dx_snr || x > pk + dx_snr as noise range
        f_cutoff: float     low frequency cutoff
        outfile: str        filename to save fit
        ftype: str          lineshape function type
            'voigt'
            'gaussian'
            'lorentzian'
            'complex-voigt'
    :returns
        snr: float          SnR of the spectral line
        vv: float           FWHM of the spectral line determined by the voigt fit
    """

    t = np.arange(len(fid)) * 1e-3
    if a == 0 and b == 0:
        wf = np.ones_like(t)
    else:
        wf = np.exp(- a * t**2 - b * t) * t
    y = np.abs(np.fft.rfft(fid * wf, zp))
    x = np.fft.rfftfreq(zp) * 1e3
    # convert to IF normalized spectrum
    xc, yc = to_mol_freq(x, y / np.max(y), bw, up=up, f_cutoff=f_cutoff)
    # initial guess
    gg0 = 2 * np.sqrt(np.log(2) * (a + a0)) / np.pi
    ll0 = max((b + b0) / np.pi, 0)
    vv_ab = fwhm_ab(a0 + a, b0 + b)

    if dx_snr:
        idx = np.array(np.ones_like(xc), dtype='bool')
        if dx_snr_mode == 'outside':
            for _pk in pks:
                idx = np.logical_and(idx, np.logical_or(xc < _pk - dx_snr, xc > _pk + dx_snr))
            # fit this xc & yc
            res = fit_spectrum(xc, yc, pks, snorms, gg0, ll0, link_gg=link_gg,
                               link_ll=link_ll, ftype=ftype)
            noise = np.std(res.residual[idx])
        elif dx_snr_mode == 'inside':       # if SnR mode is inside, we only need to fit the data inside
            for _pk in pks:
                idx = np.logical_or(idx, np.logical_and(xc > _pk - dx_snr, xc < _pk + dx_snr))
            # cut xc & yc
            xc = xc[idx]
            yc = yc[idx]
            # fit this xc & yc
            res = fit_spectrum(xc, yc, pks, snorms, gg0, ll0, link_gg=link_gg,
                               link_ll=link_ll, ftype=ftype)
            noise = np.std(res.residual)
        else:
            raise ValueError('Unknown dx_snr_mode string')
    else:
        res = fit_spectrum(xc, yc, pks, snorms, gg0, ll0, link_gg=link_gg,
                           link_ll=link_ll, ftype=ftype)
        noise = np.std(res.residual)

    snr = (np.max(yc) - res.params['p0'].value) / noise

    par_list = []
    for name, p in res.params.items():
        if isinstance(p.stderr, float):
            par_list.append('{:s}={:>8.4f}({:>8.4f})'.format(name, p.value, p.stderr))
        else:
            par_list.append('{:s}={:>8.4f}({:^8s})'.format(name, p.value, 'nan'))
    if isinstance(res.redchi, float):
        par_list.append('redchi={:>7.4f}'.format(res.redchi))
    else:
        par_list.append('redchi=nan')

    if ftype == 'voigt':
        vv_fit = fwhm_voigt_fit(res.params['ll0'].value, res.params['gg0'].value)
    elif ftype == 'complex-voigt':
        vv_fit = fwhm_complex_voigt_fit(res.params['ll0'].value, res.params['gg0'].value)
    elif ftype == 'gaussian':
        vv_fit = res.params['gg0'].value
    elif ftype == 'lorentzian':
        vv_fit = res.params['ll0'].value
    else:
        raise ValueError('Unknown function type')
    print('a={:>6.4f}'.format(a), 'b={:>7.4f}'.format(b), 'vv_ab={:>6.4f}'.format(vv_ab),
          'vv_fit={:>6.4f}'.format(vv_fit), 'snr={:>6.2f}'.format(snr), ' '.join(par_list))

    if outfile:  # save fit
        yfits = []  # a list of yfit for each peak
        for i in range(len(pks)):
            if ftype == 'voigt':
                _x0 = res.params['x'+str(i)].value
                _gg = res.params['gg'+str(i)].value
                _ll = res.params['ll'+str(i)].value
                _s = res.params['s'+str(i)].value
                yfits.append(voigt(xc - _x0, _gg, _ll) * _s)
            elif ftype == 'complex-voigt':
                _x0 = res.params['x' + str(i)].value
                _gg = res.params['gg' + str(i)].value
                _ll = res.params['ll' + str(i)].value
                _s = res.params['s' + str(i)].value
                yfits.append(complex_voigt(xc - _x0, _gg, _ll) * _s)
            elif ftype == 'gaussian':
                _x0 = res.params['x' + str(i)].value
                _gg = res.params['gg' + str(i)].value
                _s = res.params['s' + str(i)].value
                _y = np.exp(-((xc - _x0) / _gg)**2 * 4 * np.log(2)) * _s
                yfits.append(_y)
            elif ftype == 'lorentzian':
                _x0 = res.params['x' + str(i)].value
                _ll = res.params['ll' + str(i)].value
                _s = res.params['s' + str(i)].value
                _y = _ll / (2 * np.pi * ((xc - _x0)**2 + _ll**2 / 4))
                yfits.append(_y)
            else:
                raise ValueError('Unknown function type')
        if len(pks) == 1:
            outdata = np.column_stack((xc, yc, res.residual))
            outfmt = ['%6.2f', '%9.6f', '%9.6f']
        else:
            outdata = np.column_stack((xc, yc, res.residual, *yfits))
            outfmt = ['%6.2f', '%9.6f', '%9.6f'] + ['%9.6f'] * len(pks)
        hd_list = ['a0={:>6.4f} b0={:>6.4f} FLO={:g}MHz {:s}'.format(a0, b0, flo, 'UP' if up else 'DOWN'),
                   'Voigt-1D: a={:>6.4f} b={:>6.4f}'.format(a, b),
                   'SNR={:>6.2f} FWHM_AB={:>6.4f} FWHM_FIT={:>6.4f}'.format(snr, vv_ab, vv_fit),
                   ' | '.join(par_list),
                   '{:>5s} {:^10s} {:^10s}'.format('freq', 'inten', 'residual')
                   ]
        np.savetxt(outfile, outdata, fmt=outfmt, header='\n'.join(hd_list))
    return snr, vv_fit


def apply_kaiser(fid, pks, snorms, a0, b0, zp, bw, flo, up=True, f_cutoff=0.,
                 dx_snr=None, dx_snr_mode='outside', link_gg=True, link_ll=True, trunc=False,
                 ftype='gaussian', outfile=''):

    """ Apply Kaiser window function, fit the spectrum, and return the treated SnR & FWHM
    :arguments
        fid: np1darray      FID signal
        pks: list of float  peak positions
        snorms: float       The normalized intensities of the peaks
        a0: float           FID initial a0
        b0: float           FID initial b0
        zp: int             zero-padding length
        bw: float           chirp bandwidth
        flo: float          LO frequency
        up: bool            is chirp going up
        f_cutoff: float     low frequency cutoff
        dx_snr: float       +/- dx_snr from pk to calculate noise & SnR
        link_gg: bool       link all Gaussian FWHM
        link_ll: bool       link all Lorentzian FWHM
        trunc: bool         Truncate Kaiser to length = 2 * T[1/e]
        outfile: str        filename to save fit
        ftype: str          lineshape function type
            'voigt'
            'gaussian'
            'lorentzian'
            'complex-voigt'
    :returns
        snr: float          SnR of the spectral line
        vv: float           FWHM of the spectral line determined by the voigt fit
        klen: int           length of the window
    """

    if trunc:
        if a0 == 0:
            klen = int(2 / b0 * 1e3)
        else:
            klen = int((np.sqrt(b0**2 + 4 * a0) - b0) / a0 * 1e3)
        wf = kaiser(klen, 8)
        y = np.abs(np.fft.rfft(fid[:klen] * wf, zp))
    else:
        klen = len(fid)
        wf = kaiser(klen, 8)
        y = np.abs(np.fft.rfft(fid * wf, zp))

    x = np.fft.rfftfreq(zp) * 1e3
    # xc, yc = to_mol_freq(x, y, bw, flo=flo, up=up, f_cutoff=f_cutoff)
    xc, yc = to_mol_freq(x, y / np.max(y), bw, up=up, f_cutoff=f_cutoff)

    # fit this xc & yc
    gg0 = fwhm_num(xc, yc) / 2
    res = fit_spectrum(xc, yc, pks, snorms, gg0, 0,
                       link_gg=link_gg, link_ll=link_ll, ftype=ftype)

    if dx_snr:
        idx = np.array(np.ones_like(xc), dtype='bool')
        if dx_snr_mode == 'outside':
            for i in range(len(pks)):
                _pk = res.params['x' + str(i)].value
                idx = np.logical_and(idx, np.logical_or(xc < _pk - dx_snr, xc > _pk + dx_snr))
        elif dx_snr_mode == 'inside':
            for i in range(len(pks)):
                _pk = res.params['x' + str(i)].value
                idx = np.logical_or(idx, np.logical_and(xc > _pk - dx_snr, xc < _pk + dx_snr))
        else:
            raise ValueError('Unknown dx_snr_mode string')
        noise = np.std(res.residual[idx])
    else:
        noise = np.std(res.residual)
    snr = (np.max(yc) - res.params['p0'].value) / noise

    par_list = []
    for name, p in res.params.items():
        if isinstance(p.stderr, float):
            par_list.append('{:s}={:>8.4f}({:>8.4f})'.format(name, p.value, p.stderr))
        else:
            par_list.append('{:s}={:>8.4f}({:^8s})'.format(name, p.value, 'nan'))
    if isinstance(res.redchi, float):
        par_list.append('redchi={:>7.4f}'.format(res.redchi))
    else:
        par_list.append('redchi=nan')

    if ftype == 'voigt':
        vv_fit = fwhm_voigt_fit(res.params['ll0'].value, res.params['gg0'].value)
    elif ftype == 'complex-voigt':
        vv_fit = fwhm_complex_voigt_fit(res.params['ll0'].value, res.params['gg0'].value)
    elif ftype == 'gaussian':
        vv_fit = res.params['gg0'].value
    elif ftype == 'lorentzian':
        vv_fit = res.params['ll0'].value
    else:
        raise ValueError('Unknown function type')
    print('a={:^6s}'.format(' '), 'b={:^7s}'.format(' '),
          'wflen={:>6d}'.format(klen), 'vv_fit={:>6.4f}'.format(vv_fit),
          'snr={:>6.2f}'.format(snr), ' '.join(par_list))

    if outfile:  # save fit
        yfits = []  # a list of yfit for each peak
        for i in range(len(pks)):
            if ftype == 'voigt':
                _x0 = res.params['x'+str(i)].value
                _gg = res.params['gg'+str(i)].value
                _ll = res.params['ll'+str(i)].value
                _s = res.params['s'+str(i)].value
                yfits.append(voigt(xc - _x0, _gg, _ll) * _s)
            elif ftype == 'complex-voigt':
                _x0 = res.params['x' + str(i)].value
                _gg = res.params['gg' + str(i)].value
                _ll = res.params['ll' + str(i)].value
                _s = res.params['s' + str(i)].value
                yfits.append(complex_voigt(xc - _x0, _gg, _ll) * _s)
            elif ftype == 'gaussian':
                _x0 = res.params['x' + str(i)].value
                _gg = res.params['gg' + str(i)].value
                _s = res.params['s' + str(i)].value
                _y = np.exp(-((xc - _x0) / _gg)**2 * 4 * np.log(2)) * _s
                yfits.append(_y)
            elif ftype == 'lorentzian':
                _x0 = res.params['x' + str(i)].value
                _ll = res.params['ll' + str(i)].value
                _s = res.params['s' + str(i)].value
                _y = _ll / (2 * np.pi * ((xc - _x0)**2 + _ll**2 / 4))
                yfits.append(_y)
            else:
                raise ValueError('Unknown function type')
        if len(pks) == 1:
            outdata = np.column_stack((xc, yc, res.residual))
            outfmt = ['%6.2f', '%9.6f', '%9.6f']
        else:
            outdata = np.column_stack((xc, yc, res.residual, *yfits))
            outfmt = ['%6.2f', '%9.6f', '%9.6f'] + ['%9.6f'] * len(pks)
        hd_list = ['a0={:>6.4f} b0={:>6.4f} FLO={:g}MHz {:s}'.format(a0, b0, flo, 'UP' if up else 'DOWN'),
                   'Kaiser Window: pi*a=8 len={:>4d}'.format(klen),
                   'SNR={:>6.2f} FWHM_FIT={:>6.4f}'.format(snr, vv_fit),
                   ' | '.join(par_list),
                   '{:>5s} {:^10s} {:^10s}'.format('freq', 'inten', 'residual')
                   ]
        np.savetxt(outfile, outdata, fmt=outfmt, header='\n'.join(hd_list))
    return snr, vv_fit, klen


def _test_fit(gg0, ll0, zp=2**16, tmax=10):
    """ Test voigt fit """

    a0 = np.pi**2 * gg0**2 / (4 * np.log(2))
    b0 = np.pi * ll0
    fs = tmax / zp  # sampling frequency

    t = np.linspace(0, tmax, num=zp, endpoint=False)
    fid = np.exp(- a0 * t**2 - b0 * t) * t
    x = np.fft.fftshift(np.fft.fftfreq(zp)) / fs
    y = np.fft.fftshift(np.abs(np.fft.fft(fid)))
    res = fit_spectrum(x, y, (0, ), (1, ), gg0, ll0)
    # for n, p in res.params.items():
    #    print(n, p.value, p.stderr)
    print('{:>4.2f} {:>4.2f} {:>6.4f} {:>6.4f}'.format(
            gg0, ll0, res.params['gg0'].value, res.params['ll0'].value))


def _test_vv(a, b):
    """ Test FWHM of voigt, theo vs fit """

    vv_ab = fwhm_ab(a, b)
    x = np.linspace(-10, 10, num=10001)
    y = voigt1d_ft_ana(x, a, b, yshift=0)
    if a == 0:
        res = fit_spectrum(x, y, (0,), (1,), 0.53 * a, 0.8 * b, ftype='lorentzian')
        gg = 0
        ll = res.params['ll0'].value
    elif b == 0:
        res = fit_spectrum(x, y, (0,), (1,), 0.53 * a, 0.8 * b, ftype='gaussian')
        gg = res.params['gg0'].value
        ll = 0
    else:
        res = fit_spectrum(x, y, (0,), (1,), 0.53 * a, 0.8 * b, ftype='voigt')
        gg = res.params['gg0'].value
        ll = res.params['ll0'].value
    vv_fit = fwhm_voigt_fit(ll, gg)
    vv_num = fwhm_num(x, y)

    print('a={:>4.2f} b={:>4.2f} gg={:>6.4f} ll={:>6.4f} '
          'vv_ab={:>6.4f} vv_fit={:>6.4f} vv_num={:>6.4f} redchi={:g}'.format(
            a, b, gg, ll, vv_ab, vv_fit, vv_num, res.redchi))


if __name__ == '__main__':

    for _g in np.arange(0.0, 0.5, 0.2):
        for _l in np.arange(0.1, 0.5, 0.2):
            #_test_fit(_g, _l)
            _test_vv(_g, _l)
