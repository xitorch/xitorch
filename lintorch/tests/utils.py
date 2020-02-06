import torch
import pytest
import argparse
from lintorch.utils.fd import finite_differences

__all__ = ["device_dtype_float_test", "compare_grad_with_fd"]

def device_dtype_float_test(only64=False, onlycpu=False):
    dtypes = [torch.float, torch.float64]
    devices = [torch.device("cpu"), torch.device("cuda")]
    if only64:
        dtypes = [torch.float64]
    if onlycpu:
        devices = [torch.device("cpu")]

    def device_dtype_float_test_fcn(fcn):
        def fcn_all():
            for dtype in dtypes:
                for device in devices:
                    fcn(dtype, device)
        return fcn_all
    return device_dtype_float_test_fcn

def compare_grad_with_fd(fcn, args, idxs, eps=1e-6, max_rtol=1e-3,
        max_median_rtol=1e-3, fd_to64=True, verbose=False):

    args = [a.clone().detach().requires_grad_() if type(a) == torch.Tensor else a for a in args]
    device = args[idxs[0]].device
    if not hasattr(eps, "__iter__"):
        eps = [eps for i in range(len(idxs))]

    # calculate the differentiable loss
    loss0 = fcn(*args)

    # zeroing the grad
    for idx in idxs:
        if args[idx].grad is not None:
            args[idx].grad.zero_()

    loss0.backward()
    grads = [args[idx].grad.data for idx in idxs]

    # compare with finite differences
    if fd_to64:
        argsfd = [arg.to(torch.float64) \
            if (type(arg) == torch.Tensor and arg.dtype == torch.float) \
            else arg \
            for arg in args]
    else:
        argsfd = args
    fds = [finite_differences(fcn, argsfd, idx, eps=eps[i]) for i,idx in enumerate(idxs)]

    for i in range(len(idxs)):
        ratio = grads[i] / fds[i]
        if verbose:
            print("Params #%d" % (idxs[i]))
            print("* grad:")
            print(grads[i])
            print("* fd:")
            print(fds[i])
            print("* ratio:")
            print(ratio)

        dev = (ratio-1.0).abs()
        if max_rtol is not None:
            assert dev.max() < max_rtol, "Max dev ratio: %.3e (tolerated: %.3e)" % (dev.max(), max_rtol)
        if max_median_rtol is not None:
            assert dev.median() < max_median_rtol, "Median dev ratio: %.3e (tolerated: %.3e)" % (dev.median(), max_median_rtol)
