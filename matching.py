import numpy as _np
import scipy.optimize as _scyopt


def matching(acc, objectives=None, constraints=None,variables=None, covariables=None):
    """ Performs the matching of optical functions using least squares.

    variables : Must be a list of dictionaries with keys:
        'elements': family name or list of indices of ring elements to be varied.
        'separated': Optional, Boolean. True, if each element is to be treated as
            independent knobs or False, if they should be seen as a family.
            Default is False.
        'atribute': name of the attribute to be varied.
        'index'   : Optional. In case the attribute is a vector or a matrix,
            this key defines which index of the attribute must be varied. It
            must be an integer for 1D or tuple for multidimensional arrays.
        'min' : lower bound for the attribute value. Define None for unbounded (default).
        'max' : upper bound for the attribute value. Define None for unbounded (default).
    objectives: List of dictionaries defining the objectives (penalties) of
        the optimization. Each dictionary must have the keys:
        'quantities': string or tuple of strings defining which quantities will
            be used in 'fun'. The full list of possible candidates: 'betax','betay',
            'alphax','alphay','etax','etay','etaxp','etayp','mux','muy','tunex',
            'tuney','rx','px','ry','py'
        'where': family name or list of indices of the elements where to calculate.
            Can also be a string 'first' or 'last' to indicate the beginning or
            the end of the lattice, or 'all' for all indices of the lattice.
            Default is 'last'.
        'fun' : function which takes the quantities defined in 'quantities', in that
            order, and returns a float or numpy_ndarray whose values will be
            minimized. Can also be a string 'max','min','ave'. If not passed or
            None, the own value will be compared.
        'type': type of the comparison to make ('eq','==','equal') for equality,
            ('lt','<','<=') for upper bounded limits and ('gt','>','>=') for
            lower bounded limits.
        'value': Value to which the comparison will be made. Must have the same
            dimension an size as the return of 'fun'.
        'weight': defines the weight of the penalty. Default is 1.
        'scale': defines at which scale of the comparison must be satisfied.
            Default is 1.
            np.log(np.exp(-(x-val)/scale) + 1)


    """




res = _scyopt.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
            bounds=None, constraints=(), tol=None, callback=None, options=None)
