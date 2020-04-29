import torch
import lintorch as lt
import pytest
import argparse
from lintorch.utils.fd import finite_differences

__all__ = ["device_dtype_float_test", "compare_grad_with_fd", "get_diagonally_dominant_class"]

def device_dtype_float_test(only64=False, onlycpu=False):
    dtypes = [torch.float, torch.float64]
    devices = [torch.device("cpu"), torch.device("cuda")]
    if only64:
        dtypes = [torch.float64]
    if onlycpu or not torch.cuda.is_available():
        devices = [torch.device("cpu")]

    def device_dtype_float_test_fcn(fcn):
        def fcn_all():
            for dtype in dtypes:
                for device in devices:
                    fcn(dtype, device)
        return fcn_all
    return device_dtype_float_test_fcn

def compare_grad_with_fd(fcn, args, idxs, eps=1e-6, rtol=1e-5,
        atol=1e-8, fd_to64=True, verbose=False, step=1):

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
    fds = [finite_differences(fcn, argsfd, idx, eps=eps[i], step=step) for i,idx in enumerate(idxs)]

    for i in range(len(idxs)):
        torch.allclose(grads[i], fds[i], rtol=rtol, atol=atol)

def get_diagonally_dominant_class(na):
    class Acls(lt.Module):
        def __init__(self):
            super(Acls, self).__init__(shape=(na,na))

        def forward(self, x, A1, diag):
            Amatrix = (A1 + A1.transpose(-2,-1))
            A = Amatrix + diag.diag_embed(dim1=-2, dim2=-1)
            y = torch.bmm(A, x)
            return y

        def precond(self, y, A1, dg, biases=None, M=None, mparams=None):
            # return y
            # y: (nbatch, na, ncols)
            # dg: (nbatch, na)
            # biases: (nbatch, ncols) or None
            Adiag = A1.diagonal(dim1=-2, dim2=-1) * 2
            dd = (Adiag + dg).unsqueeze(-1)

            if biases is not None:
                if M is not None:
                    M1, Mdg = mparams
                    Mdiag = M1.diagonal(dim1=-2, dim2=-1) * 2
                    md = (Mdiag + Mdg).unsqueeze(-1)
                    dd = dd - biases.unsqueeze(1) * md
                else:
                    dd = dd - biases.unsqueeze(1) # (nbatch, na, ncols)
            dd[dd.abs() < 1e-6] = 1.0
            yprec = y / dd
            return yprec
    return Acls

def get_lower_mat_class(na):
    class Acls(lt.Module):
        def __init__(self):
            super(Acls, self).__init__(shape=(na,na), is_symmetric=False)

        def forward(self, x, A1):
            Amatrix = torch.tril(A1)
            y = torch.bmm(Amatrix, x)
            return y

        def transpose(self, x, A1):
            Amatrix = torch.tril(A1).transpose(-2,-1)
            y = torch.bmm(Amatrix, x)
            return y
    return Acls
