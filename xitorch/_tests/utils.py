import torch
import xitorch as xt
import pytest
import argparse
from xitorch._utils.fd import finite_differences

__all__ = ["device_dtype_float_test", "get_diagonally_dominant_class"]

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

def get_diagonally_dominant_class(na):
    class Acls(xt.Module):
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
    class Acls(xt.Module):
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
