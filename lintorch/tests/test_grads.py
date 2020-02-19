import time
import torch
import lintorch as lt
from lintorch.tests.utils import compare_grad_with_fd, device_dtype_float_test, get_diagonally_dominant_class

@device_dtype_float_test(only64=True)
def test_grad_lsymeig(dtype, device):
    # generate the matrix
    na = 4
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)

    def getloss(A1, diag, contrib):
        A = Acls().to(dtype).to(device)
        neig = 4
        options = {
            # "method": "davidson",
            # "verbose": False,
            # "nguess": neig,
            # "v_init": "randn",
        }
        bck_options = {
            "verbose": False,
            "min_eps": 1e-7,
        }
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=(A1, diag,),
            fwd_options=options,
            bck_options=bck_options)
        loss = 0
        if contrib == "eigvals":
            loss = loss + (evals**2).sum()
        elif contrib == "eigvecs":
            loss = loss + (evecs**4).sum()
        return loss

    compare_grad_with_fd(getloss, (A1, diag, "eigvals"), [0, 1])
    compare_grad_with_fd(getloss, (A1, diag, "eigvecs"), [0, 1])

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

    def getloss(A1, diag, M1, mdiag, contrib):
        A = Acls().to(dtype)
        M = Acls().to(dtype)
        neig = 4
        options = {
            "method": "davidson",
            "verbose": False,
            "nguess": neig,
            "v_init": "randn",
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

            lss = 0
            if contrib == "eigvals":
                lss = lss + (evals**1).abs().sum()
            elif contrib == "eigvecs":
                lss = lss + (evecs**1).abs().sum()
        return lss

    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvals"), [0, 1, 2, 3])
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvecs"), [0, 1, 2, 3])

@device_dtype_float_test(only64=True)
def test_grad_solve(dtype, device):
    # generate the matrix
    na = 4
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Mcls = get_diagonally_dominant_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    M = Mcls().to(dtype).to(device)
    biases = torch.rand(1,ncols).to(dtype).to(device)
    b = (A(xtrue, A1, diag) - biases.unsqueeze(1) * M(xtrue, M1, mdiag)).detach().requires_grad_()

    def getloss(A1, diag, b, biases, M1, mdiag):
        fwd_options = {
            "min_eps": 1e-9
        }
        bck_options = {
            "verbose": False,
        }
        xinv = lt.solve(A, (A1, diag), b,
            biases = biases,
            M = M,
            mparams = (M1, mdiag),
            fwd_options = fwd_options)
        lss = (xinv**2).sum()
        return lss

    compare_grad_with_fd(getloss, (A1, diag, b, biases, M1, mdiag), [0,1,2,3,4,5])

@device_dtype_float_test(only64=True)
def test_2grad_lsymeig(dtype, device):
    # generate the matrix
    na = 5
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)

    def getloss(A1, diag, contrib, contrib2):
        A = Acls().to(dtype).to(device)
        neig = 4
        options = {
            "method": "davidson",
            "verbose": False,
            "min_eps": 1e-10,
        }
        bck_options = {
            "verbose": False,
            "min_eps": 1e-10,
        }
        with torch.enable_grad():
            A1.requires_grad_()
            diag.requires_grad_()
            evals, evecs = lt.lsymeig(A,
                neig=neig,
                params=(A1, diag,),
                fwd_options=options,
                bck_options=bck_options)

            lss = 0
            if contrib == "eigvals":
                lss = lss + evals.abs().sum()
            elif contrib == "eigvecs":
                lss = lss + evecs.abs().sum()
            grad_A1, grad_diag = torch.autograd.grad(lss, (A1, diag),
                create_graph=True)

        loss = 0
        if contrib2 == "A1":
            loss = loss + (grad_A1**2).abs().sum()
        elif contrib2 == "diag":
            loss = loss + (grad_diag**2).abs().sum()
        return loss

    compare_grad_with_fd(getloss, (A1, diag, "eigvals", "A1"), [0, 1])
    compare_grad_with_fd(getloss, (A1, diag, "eigvals", "diag"), [0, 1])
    # larger step size in eigvecs to avoid errors due to non-exactness in the iterative solutions
    compare_grad_with_fd(getloss, (A1, diag, "eigvecs", "A1"), [0, 1], eps=1e-2, step=2)
    compare_grad_with_fd(getloss, (A1, diag, "eigvecs", "diag"), [0, 1], eps=1e-2, step=2)

@device_dtype_float_test(only64=True)
def test_2grad_lsymeig_with_M(dtype, device):
    # generate the matrix
    na = 5
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)

    def getloss(A1, diag, M1, mdiag, contrib, contrib2):
        A = Acls().to(dtype).to(device)
        M = Acls().to(dtype).to(device)
        neig = 4
        options = {
            "method": "davidson",
            "verbose": False,
            "min_eps": 1e-10,
        }
        bck_options = {
            "verbose": False,
            "min_eps": 1e-10,
        }
        with torch.enable_grad():
            A1.requires_grad_()
            diag.requires_grad_()
            evals, evecs = lt.lsymeig(A,
                neig=neig,
                params=(A1, diag,),
                M=M,
                mparams=(M1, mdiag,),
                fwd_options=options,
                bck_options=bck_options)

            lss = 0
            if contrib == "eigvals":
                lss = lss + evals.abs().sum()
            elif contrib == "eigvecs":
                lss = lss + evecs.abs().sum()
            grad_A1, grad_diag, grad_M1, grad_mdiag = \
                torch.autograd.grad(lss, (A1, diag, M1, mdiag), create_graph=True)

        loss = 0
        if contrib2 == "A1":
            loss = loss + (grad_A1**2).abs().sum()
        elif contrib2 == "diag":
            loss = loss + (grad_diag**2).abs().sum()
        elif contrib2 == "mdiag":
            loss = loss + (grad_mdiag**2).abs().sum()
        elif contrib2 == "M1":
            loss = loss + (grad_M1**2).abs().sum()
        return loss

    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvals", "A1"), [0, 1])
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvals", "diag"), [0, 1])
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvals", "M1"), [0, 1])
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvals", "mdiag"), [0, 1])
    # larger step size in eigvecs to avoid errors due to non-exactness in the iterative solutions
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvecs", "A1"), [0, 1], eps=1e-2, step=2)
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvecs", "diag"), [0, 1], eps=1e-2, step=2)
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvecs", "M1"), [0, 1], eps=1e-2, step=2)
    compare_grad_with_fd(getloss, (A1, diag, M1, mdiag, "eigvecs", "mdiag"), [0, 1], eps=1e-2, step=2)

@device_dtype_float_test(only64=True)
def test_2grad_solve(dtype, device):
    # generate the matrix
    na = 4
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Acls = get_diagonally_dominant_class(na)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
    Mcls = get_diagonally_dominant_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    M = Mcls().to(dtype).to(device)
    biases = torch.rand(1,ncols).to(dtype).to(device)
    b = (A(xtrue, A1, diag) - biases.unsqueeze(1) * M(xtrue, M1, mdiag)).detach().requires_grad_()

    def getloss(A1, diag, b, biases, M1, mdiag, contrib):
        fwd_options = {
            "min_eps": 1e-9
        }
        bck_options = {
            "min_eps": 1e-9
        }
        bck_options = {
            "verbose": False,
        }
        with torch.enable_grad():
            A1.requires_grad_()
            b.requires_grad_()
            diag.requires_grad_()
            biases.requires_grad_()
            M1.requires_grad_()
            mdiag.requires_grad_()
            xinv = lt.solve(A, (A1, diag), b,
                biases = biases,
                M = M,
                mparams = (M1, mdiag),
                fwd_options = fwd_options,
                bck_options = bck_options)
            lss = (xinv**2).sum()
            grad_A1, grad_diag, grad_b, grad_biases, grad_M1, grad_mdiag = \
                    torch.autograd.grad(
                        lss,
                        (A1, diag, b, biases, M1, mdiag), create_graph=True)
        loss = 0
        if contrib == "params":
            loss = loss + (grad_A1**2).sum()
            loss = loss + (grad_diag**2).sum()
        elif contrib == "mparams":
            loss = loss + (grad_M1**2).sum()
            loss = loss + (grad_mdiag**2).sum()
        elif contrib == "b":
            loss = loss + (grad_b**2).sum()
        elif contrib == "biases":
            loss = loss + (grad_biases**2).sum()
        return loss

    compare_grad_with_fd(getloss, (A1, diag, b, biases, M1, mdiag, "params"), [0,1,2,3,4,5])
    compare_grad_with_fd(getloss, (A1, diag, b, biases, M1, mdiag, "mparams"), [0,1,2,3,4,5])
    compare_grad_with_fd(getloss, (A1, diag, b, biases, M1, mdiag, "b"), [0,1,2,3,4,5])
    compare_grad_with_fd(getloss, (A1, diag, b, biases, M1, mdiag, "biases"), [0,1,2,3,4,5])
