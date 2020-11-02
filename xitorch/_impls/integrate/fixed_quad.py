import numpy as np
import torch

# no gradient flowing in the following functions

def leggauss(fcn, xl, xu, params, n=100, **unused):
    """
    Performing 1D integration using Legendre-Gaussian quadrature

    Keyword arguments
    -----------------
    n: int
        The number of integration points.
    """
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    ndim = len(xu.shape)
    xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg *= 0.5 * (xu - xl)
    xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))  # (n, *nx)

    res = wlg[0] * fcn(xs[0], *params)
    for i in range(1, n):
        res += wlg[i] * fcn(xs[i], *params)
    return res
