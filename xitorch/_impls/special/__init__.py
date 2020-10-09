from xitorch._impls.special.generated.pyfuncs import *

# docstring
j0.__doc__ = """
    Bessel function of the first kind of order 0.

    Arguments
    ---------
    x: torch.Tensor
        The input argument.

    Returns
    -------
    torch.Tensor
        The value of Bessel function of the first kind of order 0 at x

    Notes
    -----
    * This function is a wrapper of the Cephes' j0 function [1]_.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library, http://www.netlib.org/cephes/
"""

y0.__doc__ = """
    Bessel function of the second kind of order 0.

    Arguments
    ---------
    x: torch.Tensor
        The input argument.

    Returns
    -------
    torch.Tensor
        The value of Bessel function of the second kind of order 0 at x

    Notes
    -----
    * This function is a wrapper of the Cephes' y0 function [1]_.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library, http://www.netlib.org/cephes/
"""

igam.__doc__ = """
    Regularized lower incomplete gamma function:

    .. math::

        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a-1}e^-t\ \mathrm{d}t

    Arguments
    ---------
    a: torch.Tensor
        The first input argument (must be positive).

    x: torch.Tensor
        The second input argument.

    Returns
    -------
    torch.Tensor
        The value of regularized lower incomplete gamma function

    Notes
    -----
    * This function is a wrapper of the Cephes' igam function [1]_.

    References
    ----------
    .. [1] Cephes Mathematical Functions Library, http://www.netlib.org/cephes/
"""
