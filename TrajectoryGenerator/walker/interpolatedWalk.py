import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

def interpolated_walk(x, y, factor=10, kind="cubic"):
    r""" Smooths a given trace by interpolation to give it a more natural appearance.

    Parameters
    ----------
    x,y : *array_like*
        Lists containing the individual values of a trace
    factor : *int*
        Factor which determines the number of points of the smoother trace
        (/Factor * len(x)/). Default is 10.
    kind : *str*
        Specifies the kind of interpolation as a string, used by
        `scipy.interpolate.interp1d`. Possible values are:
        ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’,
        ‘cubic’ and ‘univariate’ where ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation
        of zeroth, first, second or third order). If ‘univariate’ is
        chose, the function
        `scipy.interpolate.InterpolatedUnivariateSpline` will be
        used. Default is ‘cubic’.

    Returns
    -------
    xi, yi : *array_like*
        Lists of smoothed Trace-Positions. Now no longer as /integers/.

    """
    original_len = len(x)
    # Padding necessary for closed-loop traces.
    if np.sqrt((x[-1]-x[0])**2+(y[-1]-y[0])**2) < 10:
        xpad = np.concatenate((x[-3:-1],x,x[1:3]))
        ypad = np.concatenate((y[-3:-1],y,y[1:3]))
        ti = np.linspace(2, original_len+1, int(factor*original_len))
    else:
        xpad = x
        ypad = y
        ti = np.linspace(0, original_len-1, factor*original_len)
    t = np.arange(len(xpad))
    if kind == "univariate":
        xi = InterpolatedUnivariateSpline(t, xpad)(ti)
        yi = InterpolatedUnivariateSpline(t, ypad)(ti)
    else:
        xi = interp1d(t, xpad, kind=kind)(ti)
        yi = interp1d(t, ypad, kind=kind)(ti)
    return xi, yi

