import torch
import matplotlib.pyplot as plt
import numpy as np

################### metropolis hastings ###################
def mh(logpfcn, x0, pparams, nsamples=10000, nburnout=5000, step_size=1.0, **unused):
    dtype = x0.dtype
    device = x0.device

    x = _mh_sample(logpfcn, x0, pparams, nburnout, step_size, False, dtype, device)
    samples = _mh_sample(logpfcn, x, pparams, nsamples, step_size, True, dtype, device)
    weights = torch.zeros((samples.shape[0],), dtype=dtype, device=device) + (1./samples.shape[0])
    return samples, weights

def _mh_sample(logpfcn, x0, pparams, nsamples, step_size, collect_samples, dtype, device):
    x = x0
    logpx = logpfcn(x0, *pparams)
    log_rand = torch.log(torch.rand((nsamples,), dtype=dtype, device=device))
    if collect_samples:
        samples = torch.empty((nsamples, *x0.shape), dtype=dtype, device=device)

    for i in range(nsamples):
        xnext = x + step_size * torch.randn_like(x)
        logpnext = logpfcn(xnext, *pparams)
        logpratio = logpnext - logpx

        # decide if we should accept the next point
        if logpratio > 0:
            accept = True
        else:
            accept = log_rand[i] < logpratio

        # if accept, move the x into the new points
        if accept:
            logpx = logpnext
            x = xnext
        if collect_samples:
            samples[i] = x

    # return the samples if collect_samples, otherwise just return the last x
    if collect_samples:
        return samples
    else:
        return x

################### dummy sampler just for 1D ###################
def dummy1d(logpfcn, x0, pparams, nsamples=100, lb=-np.inf, ub=np.inf, **unused):
    dtype = x0.dtype
    device = x0.device

    # convert the bound to finite range
    ub = torch.tensor(ub, dtype=dtype, device=device)
    lb = torch.tensor(lb, dtype=dtype, device=device)
    tu = torch.atan(ub)
    tl = torch.atan(lb)

    assert torch.numel(x0) == 1, "This dummy operation can only be done in 1D space"
    tlg, wlg = np.polynomial.legendre.leggauss(nsamples)
    tlg = torch.tensor(tlg, dtype=dtype, device=device)
    wlg = torch.tensor(wlg, dtype=dtype, device=device)
    wlg *= 0.5 * (tu - tl)
    tsamples = tlg * (0.5 * (tu - tl)) + (0.5 * (tu + tl)) # (n, *nx)
    xsamples = torch.tan(tsamples)
    wt = torch.cos(tsamples)**(-2.)
    wp = torch.empty_like(wt)
    for i in range(nsamples):
        wp[i] = torch.exp(logpfcn(xsamples[i], *pparams))

    wsamples = wt * wlg * wp
    wsamples = wsamples / wsamples.sum()
    return xsamples, wsamples
