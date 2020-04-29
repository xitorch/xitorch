import time
import torch
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.tests.utils import compare_grad_with_fd, device_dtype_float_test, \
    get_diagonally_dominant_class, get_lower_mat_class

@device_dtype_float_test(only64=True)
def test_grad_lsymeig(dtype, device):
    # generate the matrix
    na = 4
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)

    def getloss(A1, diag, method="davidson"):
        A = Acls().to(dtype).to(device)
        neig = 4
        options = {
            "method": method,
            # "verbose": False,
            # "nguess": neig,
            # "v_init": "randn",
        }
        bck_options = {
            "verbose": False,
            "min_eps": 1e-9,
        }
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=(A1, diag,),
            fwd_options=options,
            bck_options=bck_options)
        return evals, evecs

    gradcheck(getloss, (A1, diag, "davidson"))
    gradcheck(getloss, (A1, diag, "exacteig"))
    gradgradcheck(getloss, (A1, diag, "exacteig"))
    gradgradcheck(getloss, (A1, diag, "davidson"), rtol=1e-4, atol=1e-4, eps=1e-3)

@device_dtype_float_test(only64=True)
def test_grad_lsymeig_with_M(dtype, device):
    na = 4
    dtype = torch.float64
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).unsqueeze(0).requires_grad_(True)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).requires_grad_(True)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)

    def getloss(A1, diag, M1, mdiag, method="davidson"):
        A = Acls().to(dtype)
        M = Acls().to(dtype)
        neig = 4
        options = {
            "method": method,
        }
        bck_options = {
            "verbose": True,
            "min_eps": 1e-9,
        }
        with torch.enable_grad():
            A1.requires_grad_()
            diag.requires_grad_()
            M1.requires_grad_()
            mdiag.requires_grad_()
            evals, evecs = lt.lsymeig(A,
                neig=neig,
                params=(A1, diag,),
                M=M,
                mparams=(M1, mdiag,),
                fwd_options=options,
                bck_options=bck_options)
        return evals, evecs

    gradcheck(getloss, (A1, diag, M1, mdiag, "davidson"))
    gradcheck(getloss, (A1, diag, M1, mdiag, "exacteig"))
    gradgradcheck(getloss, (A1, diag, M1, mdiag, "davidson"))
    gradgradcheck(getloss, (A1, diag, M1, mdiag, "exacteig"))

@device_dtype_float_test(only64=True)
def test_grad_solve(dtype, device):
    # generate the matrix
    na = 4
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))+1).to(dtype).to(device).requires_grad_(True)
    Acls = get_lower_mat_class(na)
    M1 = (torch.rand((1,na,na))+1).to(dtype).to(device).requires_grad_(True)
    Mcls = get_lower_mat_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    M = Mcls().to(dtype).to(device)
    biases = torch.rand(1,ncols).to(dtype).to(device) * 0.1
    b = (A(xtrue, A1) - biases.unsqueeze(1) * M(xtrue, M1)).detach().requires_grad_()

    def getloss(A1, b, biases, M1):
        fwd_options = {
            "min_eps": 1e-12,
        }
        bck_options = {
            "verbose": False,
            "min_eps": 1e-12,
        }
        xinv = lt.solve(A, (A1,), b,
            biases = biases,
            M = M,
            mparams = (M1,),
            fwd_options = fwd_options)
        return xinv

    gradcheck(getloss, (A1, b, biases, M1))
    gradgradcheck(getloss, (A1, b, biases, M1), atol=7e-4)
