import numpy as np
import torch

# no gradient flowing in the following functions

def leggaussquad(fcn, xl, xu, params, n, **unused):
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    ndim = len(xu.shape)
    xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,)+(None,)*ndim] # (n, *nx)
    wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,)+(None,)*ndim] # (n, *nx)
    wlg *= 0.5 * (xu - xl)
    xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl)) # (n, *nx)

    res = [wlg[0] * f for f in fcn(xs[0], *params)]
    nres = len(res)
    for i in range(1,n):
        w = wlg[i]
        f = fcn(xs[i], *params)
        for j in range(nres):
            res[j] += w * f[j]
    return res
