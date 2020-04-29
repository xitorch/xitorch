import torch
import lintorch as lt
from lintorch.tests.utils import compare_grad_with_fd, device_dtype_float_test, \
    get_diagonally_dominant_class, get_lower_mat_class

@device_dtype_float_test()
def test_lsymeig(dtype, device):
    # generate the matrix
    def runtest(options):
        na = 10
        torch.manual_seed(123)
        A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
        diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
        Acls = get_diagonally_dominant_class(na)
        params = (A1, diag)

        A = Acls().to(dtype).to(device)
        neig = 4
        # evals: (nbatch, neig)
        # evecs: (nbatch, na, neig)
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=params,
            fwd_options=options)

        # check with the eigendecomposition equation
        AU = A(evecs, *params)
        UE = evals.unsqueeze(1) * evecs
        assert torch.allclose(AU, UE, atol=1e-5, rtol=1e-5)

        # check orthogonality
        UTU = torch.bmm(evecs.transpose(-2,-1), evecs)
        eye = torch.eye(UTU.shape[-1]).unsqueeze(0).to(UTU.dtype).to(UTU.device)
        assert torch.allclose(UTU, eye, atol=1e-5, rtol=1e-5)

    all_options = [{
        "method": "davidson",
        "min_eps": 1e-9,
        },
        {
        "method": "exacteig",
        }]
    for options in all_options:
        runtest(options)

@device_dtype_float_test()
def test_lsymeig_with_M(dtype, device):
    # generate the matrix
    def runtest(options):
        na = 10
        torch.manual_seed(123)
        A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
        diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
        Acls = get_diagonally_dominant_class(na)
        params = (A1, diag)
        M1 = (torch.rand((1,na,na))*0.01).to(dtype).to(device)
        mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
        Mcls = get_diagonally_dominant_class(na)
        mparams = (M1, mdiag)

        A = Acls().to(dtype).to(device)
        M = Mcls().to(dtype).to(device)
        neig = 4
        # evals: (nbatch, neig)
        # evecs: (nbatch, na, neig)
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=params,
            M=M,
            mparams=mparams,
            fwd_options=options)

        AU = A(evecs, *params)
        MUE = M(evals.unsqueeze(1) * evecs, *mparams)
        assert torch.allclose(AU, MUE, atol=1e-5, rtol=1e-4)

        # check orthogonality
        UMU = torch.bmm(evecs.transpose(-2,-1), M(evecs, *mparams))
        eye = torch.eye(UMU.shape[-1]).to(UMU.dtype).to(UMU.device)
        assert torch.allclose(UMU, eye, atol=1e-4, rtol=1e-6)

    all_options = [
        {
            "method": "exacteig",
        },
        {
            "method": "davidson",
            "nguess": 10,
            "min_eps": 1e-9
        },
        {
            "method": "davidson",
            "nguess": 4,
            "min_eps": 1e-9
        }
    ]
    for options in all_options:
        runtest(options)

@device_dtype_float_test()
def test_solve(dtype, device):
    # generate the matrix
    na = 10
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))+1).to(dtype).to(device)
    Acls = get_lower_mat_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    biases = (torch.rand(1,ncols)*0.1).to(dtype).to(device)
    b = (A(xtrue, A1) - biases.unsqueeze(1) * xtrue)

    fwd_options = {
        "min_eps": 1e-9,
    }
    x = lt.solve(A, (A1,), b,
        biases = biases,
        fwd_options = fwd_options)

    assert torch.allclose(x, xtrue, atol=1e-5, rtol=1e-4)

@device_dtype_float_test()
def test_solve_with_M(dtype, device):
    # generate the matrix
    na = 10
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))+1).to(dtype).to(device)
    Acls = get_lower_mat_class(na)
    M1 = (torch.rand((1,na,na))+1).to(dtype).to(device)
    Mcls = get_lower_mat_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    M = Mcls().to(dtype).to(device)
    biases = torch.rand(1,ncols).to(dtype).to(device) * 0.1
    b = (A(xtrue, A1) - biases.unsqueeze(1) * M(xtrue, M1))

    fwd_options = {
        "min_eps": 1e-9,
    }
    x = lt.solve(A, (A1,), b,
        biases = biases,
        M = M,
        mparams = (M1,),
        fwd_options = fwd_options)

    assert torch.allclose(x, xtrue, atol=1e-5, rtol=1e-4)

@device_dtype_float_test(only64=True)
def test_converge_solve(dtype, device):
    # generate the matrix
    na = 1000
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))*0.01).to(dtype).to(device)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
    Acls = get_diagonally_dominant_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    biases = torch.rand(1,ncols).to(dtype).to(device)
    params = (A1, diag)
    b = (A(xtrue, *params) - biases.unsqueeze(1) * xtrue)

    fwd_options = {
        "min_eps": 1e-10,
        "verbose": True,
        "max_niter": 20,
    }
    x = lt.solve(A, (A1, diag), b,
        biases = biases,
        fwd_options = fwd_options)

    Axmb = A(x, *params) - biases.unsqueeze(1) * x
    assert torch.allclose(Axmb, b, atol=1e-5, rtol=1e-4)
    assert torch.allclose(xtrue, x, atol=1e-5, rtol=1e-4)

@device_dtype_float_test(only64=True)
def test_converge_solve_with_M(dtype, device):
    # generate the matrix
    na = 1000
    ncols = 2
    torch.manual_seed(124)
    A1 = (torch.rand((1,na,na))*0.01).to(dtype).to(device)
    diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
    Acls = get_diagonally_dominant_class(na)
    M1 = (torch.rand((1,na,na))*0.01).to(dtype).to(device)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
    Mcls = get_diagonally_dominant_class(na)
    xtrue = torch.rand(1,na,ncols).to(dtype).to(device)
    A = Acls().to(dtype).to(device)
    M = Mcls().to(dtype).to(device)
    params = (A1, diag)
    mparams = (M1, mdiag)
    biases = torch.rand(1,ncols).to(dtype).to(device)
    b = (A(xtrue, *params) - biases.unsqueeze(1) * M(xtrue, *mparams))

    fwd_options = {
        "min_eps": 1e-10,
        "verbose": True,
        "max_niter": 20,
    }
    x = lt.solve(A, (A1, diag), b,
        biases = biases,
        M = M,
        mparams = (M1, mdiag),
        fwd_options = fwd_options)

    Axmb = A(x, *params) - biases.unsqueeze(1) * M(x, *mparams)
    assert torch.allclose(Axmb, b, atol=1e-5, rtol=1e-4)
    assert torch.allclose(x, xtrue, atol=1e-5, rtol=1e-4)


@device_dtype_float_test(only64=True)
def test_converge_lsymeig(dtype, device):
    # generate the matrix
    def runtest(options):
        na = 1000
        torch.manual_seed(123)
        A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
        diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
        Acls = get_diagonally_dominant_class(na)
        params = (A1, diag)

        A = Acls().to(dtype).to(device)
        neig = 4
        # evals: (nbatch, neig)
        # evecs: (nbatch, na, neig)
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=params,
            fwd_options=options)

        # check with the eigendecomposition equation
        AU = A(evecs, *params)
        UE = evals.unsqueeze(1) * evecs
        assert torch.allclose(AU, UE, atol=1e-5, rtol=1e-5)

        # check orthogonality
        UTU = torch.bmm(evecs.transpose(-2,-1), evecs)
        eye = torch.eye(UTU.shape[-1]).unsqueeze(0).to(UTU.dtype).to(UTU.device)
        assert torch.allclose(UTU, eye, atol=1e-5, rtol=1e-5)

    all_options = [{
        "method": "davidson",
        "min_eps": 1e-9,
        "max_niter": 20,
        "verbose": True,
        }]
    for options in all_options:
        runtest(options)

@device_dtype_float_test()
def test_converge_lsymeig_with_M(dtype, device):
    # generate the matrix
    def runtest(options):
        na = 1000
        torch.manual_seed(123)
        A1 = (torch.rand((1,na,na))*0.1).to(dtype).to(device).requires_grad_(True)
        diag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0).requires_grad_(True)
        Acls = get_diagonally_dominant_class(na)
        params = (A1, diag)
        M1 = (torch.rand((1,na,na))*0.01).to(dtype).to(device)
        mdiag = (torch.arange(na, dtype=dtype)+1.0).to(device).unsqueeze(0)
        Mcls = get_diagonally_dominant_class(na)
        mparams = (M1, mdiag)

        A = Acls().to(dtype).to(device)
        M = Mcls().to(dtype).to(device)
        neig = 4
        # evals: (nbatch, neig)
        # evecs: (nbatch, na, neig)
        evals, evecs = lt.lsymeig(A,
            neig=neig,
            params=params,
            M=M,
            mparams=mparams,
            fwd_options=options)

        AU = A(evecs, *params)
        MUE = M(evals.unsqueeze(1) * evecs, *mparams)
        assert torch.allclose(AU, MUE, atol=1e-5, rtol=1e-4)

        # check orthogonality
        UMU = torch.bmm(evecs.transpose(-2,-1), M(evecs, *mparams))
        eye = torch.eye(UMU.shape[-1]).to(UMU.dtype).to(UMU.device)
        assert torch.allclose(UMU, eye, atol=1e-4, rtol=1e-6)

    all_options = [
        {
            "method": "davidson",
            "min_eps": 1e-9,
            "max_niter": 20,
            "verbose": True,
        }
    ]
    for options in all_options:
        runtest(options)
