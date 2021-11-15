import warnings
import torch
import pytest
from torch.autograd import gradcheck, gradgradcheck
from xitorch.debug.modes import enable_debug
from xitorch import LinearOperator
from xitorch.linalg.symeig import lsymeig, symeig, svd
from xitorch.linalg.solve import solve
from xitorch._utils.bcast import get_bcasted_dims
from xitorch._utils.exceptions import MathWarning
from xitorch._tests.utils import device_dtype_float_test

seed = 12345

############## lsymeig ##############
@device_dtype_float_test()
def test_lsymeig_nonhermit_err(dtype, device):
    torch.manual_seed(seed)
    mat = torch.rand((3, 3), dtype=dtype, device=device)
    linop = LinearOperator.m(mat, False)
    linop2 = LinearOperator.m(mat + mat.transpose(-2, -1), True)

    try:
        res = lsymeig(linop)
        assert False, "A RuntimeError must be raised if the A linear operator in lsymeig is not Hermitian"
    except RuntimeError:
        pass

    try:
        res = lsymeig(linop2, M=linop)
        assert False, "A RuntimeError must be raised if the M linear operator in lsymeig is not Hermitian"
    except RuntimeError:
        pass

@device_dtype_float_test()
def test_lsymeig_mismatch_err(dtype, device):
    torch.manual_seed(seed)
    mat1 = torch.rand((3, 3), dtype=dtype, device=device)
    mat2 = torch.rand((2, 2), dtype=dtype, device=device)
    mat1 = mat1 + mat1.transpose(-2, -1)
    mat2 = mat2 + mat2.transpose(-2, -1)
    linop1 = LinearOperator.m(mat1, True)
    linop2 = LinearOperator.m(mat2, True)

    try:
        res = lsymeig(linop1, M=linop2)
        assert False, "A RuntimeError must be raised if A & M shape are mismatch"
    except RuntimeError:
        pass


@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "shape": [(4, 4), (2, 4, 4), (2, 3, 4, 4)],
    "method": ["exacteig", "custom_exacteig"],  # only 2 of methods, because both gradient implementations are covered
})
def test_lsymeig_A(dtype, device, shape, method):
    torch.manual_seed(seed)
    mat1 = torch.rand(shape, dtype=dtype, device=device)
    mat1 = mat1 + mat1.transpose(-2, -1).conj()
    mat1 = mat1.requires_grad_()
    linop1 = LinearOperator.m(mat1, True)
    fwd_options = {"method": method}

    for neig in [2, shape[-1]]:
        eigvals, eigvecs = lsymeig(linop1, neig=neig, **fwd_options)  # eigvals: (..., neig), eigvecs: (..., na, neig)
        eigvals = eigvals.to(eigvecs.dtype)
        assert list(eigvecs.shape) == list([*linop1.shape[:-1], neig])
        assert list(eigvals.shape) == list([*linop1.shape[:-2], neig])

        ax = linop1.mm(eigvecs)
        xe = torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1))
        assert torch.allclose(ax, xe)

        # only perform gradcheck if neig is full, to reduce the computational cost
        if neig == shape[-1]:
            def lsymeig_fcn(amat):
                amat = (amat + amat.transpose(-2, -1).conj()) * 0.5  # symmetrize
                alinop = LinearOperator.m(amat, is_hermitian=True)
                eigvals_, eigvecs_ = lsymeig(alinop, neig=neig, **fwd_options)
                return eigvals_, eigvecs_.abs()

            gradcheck(lsymeig_fcn, (mat1,))
            gradgradcheck(lsymeig_fcn, (mat1,))

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "ashape": [(3, 3), (2, 3, 3), (2, 1, 3, 3)],
    "mshape": [(3, 3), (2, 3, 3), (2, 1, 3, 3)],
    "method": ["exacteig", "custom_exacteig"],  # only 2 of methods, because both gradient implementations are covered
})
def test_lsymeig_AM(dtype, device, ashape, mshape, method):
    torch.manual_seed(seed)
    mata = torch.rand(ashape, dtype=dtype, device=device)
    matm = torch.rand(mshape, dtype=dtype, device=device) + \
        torch.eye(mshape[-1], dtype=dtype, device=device)  # make sure it's not singular
    mata = mata + mata.transpose(-2, -1).conj()
    matm = matm + matm.transpose(-2, -1).conj()
    mata = mata.requires_grad_()
    matm = matm.requires_grad_()
    linopa = LinearOperator.m(mata, is_hermitian=True)
    linopm = LinearOperator.m(matm, is_hermitian=True)
    fwd_options = {"method": method}

    na = ashape[-1]
    bshape = get_bcasted_dims(ashape[:-2], mshape[:-2])
    for neig in [2, ashape[-1]]:
        eigvals, eigvecs = lsymeig(linopa, M=linopm, neig=neig, **fwd_options)  # eigvals: (..., neig)
        eigvals = eigvals.to(eigvecs.dtype)
        assert list(eigvals.shape) == list([*bshape, neig])
        assert list(eigvecs.shape) == list([*bshape, na, neig])

        ax = linopa.mm(eigvecs)
        mxe = linopm.mm(torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1)))
        assert torch.allclose(ax, mxe)

        # only perform gradcheck if neig is full, to reduce the computational cost
        if neig == ashape[-1]:
            def lsymeig_fcn(amat, mmat):
                # symmetrize
                amat = (amat + amat.transpose(-2, -1).conj()) * 0.5
                mmat = (mmat + mmat.transpose(-2, -1).conj()) * 0.5
                alinop = LinearOperator.m(amat, is_hermitian=True)
                mlinop = LinearOperator.m(mmat, is_hermitian=True)
                eigvals_, eigvecs_ = lsymeig(alinop, M=mlinop, neig=neig, **fwd_options)
                return eigvals_, eigvecs_.abs()

            gradcheck(lsymeig_fcn, (mata, matm))
            gradgradcheck(lsymeig_fcn, (mata, matm))

@device_dtype_float_test(only64=True, additional_kwargs={
    "shape": [(1000, 1000), (2, 1000, 1000), (2, 3, 1000, 1000)],
    "method": ["davidson"],  # list the methods here
    "mode": ["uppermost", "lowest"],
})
def test_symeig_A_large_methods(dtype, device, shape, method, mode):
    torch.manual_seed(seed)

    class ALarge(LinearOperator):
        def __init__(self, shape, dtype, device):
            super(ALarge, self).__init__(shape,
                                         is_hermitian=True,
                                         dtype=dtype,
                                         device=device)
            na = shape[-1]
            self.b = torch.arange(na, dtype=dtype, device=device).repeat(*shape[:-2], 1)

        def _mv(self, x):
            # x: (*BX, na)
            xb = x * self.b
            xsmall = x * 1e-3
            xp1 = torch.roll(xsmall, shifts=1, dims=-1)
            xm1 = torch.roll(xsmall, shifts=-1, dims=-1)
            return xb + xp1 + xm1

        def _getparamnames(self, prefix=""):
            return [prefix + "b"]

    neig = 2
    na = shape[-1]
    linop1 = ALarge(shape, dtype=dtype, device=device)
    fwd_options = {"method": method, "min_eps": 1e-8}

    # eigvals: (..., neig), eigvecs: (..., na, neig)
    eigvals, eigvecs = symeig(linop1, mode=mode, neig=neig, **fwd_options)

    # the matrix's eigenvalues will be around arange(na)
    if mode == "lowest":
        assert (eigvals < neig * 2).all()
    elif mode == "uppermost":
        assert (eigvals > na - neig * 2).all()

    assert list(eigvecs.shape) == list([*linop1.shape[:-1], neig])
    assert list(eigvals.shape) == list([*linop1.shape[:-2], neig])

    ax = linop1.mm(eigvecs)
    xe = torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1))
    assert torch.allclose(ax, xe)

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "eivaloffset": [0, -4],
    "method": ["exacteig", "custom_exacteig"]
})
def test_symeig_A_degenerate(dtype, device, eivaloffset, method):
    # test if the gradient can be stably propagated if the loss function
    # does not depend on which degenerate eigenvectors are used
    # (note: the variable is changed in a certain way so that the degeneracy
    # is kept)

    torch.manual_seed(seed)
    n = 5
    neig = 3
    kwargs = {
        "dtype": dtype,
        "device": device,
    }
    print(dtype)
    # random matrix to be orthogonalized for the eigenvectors
    mat = torch.randn((n, n), **kwargs).requires_grad_()

    # matrix for the loss function
    P2 = torch.randn((n, n), **kwargs).requires_grad_()

    # the degenerate eigenvalues
    a = (torch.tensor([1.0, 2.0, 3.0], **kwargs) + eivaloffset)
    if torch.is_complex(a):
        a = a.real
    a = a.requires_grad_()

    bck_options = {
        "method": "exactsolve",
    }

    def get_loss(a, mat, P2):
        # get the orthogonal vector for the eigenvectors
        P, _ = torch.linalg.qr(mat)

        # line up the eigenvalues
        b = torch.cat((a[:2], a[1:2], a[2:], a[2:])).to(dtype)

        # construct the matrix
        diag = torch.diag_embed(b)
        A = torch.matmul(torch.matmul(P.transpose(-2, -1).conj(), diag), P)
        Alinop = LinearOperator.m(A, is_hermitian=True)

        eivals, eivecs = symeig(
            Alinop, neig=neig,
            method=method,
            bck_options=bck_options)
        U = eivecs[:, 1:3]  # the degenerate eigenvectors

        loss = torch.einsum("rc,rc->", torch.matmul(P2, U), U.conj())
        return loss

    gradcheck(get_loss, (a, mat, P2))
    gradgradcheck(get_loss, (a, mat, P2))

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "method": ["exacteig", "custom_exacteig"]
})
def test_symeig_AM_degenerate(dtype, device, method):
    # same as test_symeig_A_degenerate, but now with the overlap matrix

    torch.manual_seed(seed)
    n = 5
    neig = 3
    kwargs = {
        "dtype": dtype,
        "device": device,
    }
    # random matrix to be orthogonalized for the eigenvectors
    matA = torch.randn((n, n), **kwargs)
    matM = torch.rand((n, n), **kwargs)

    # matrix for the loss function
    P2 = torch.randn((n, n), **kwargs).requires_grad_()

    # the degenerate eigenvalues
    a = torch.tensor([1.0, 2.0, 3.0], **kwargs)
    if torch.is_complex(a):
        a = a.real
    a = a.requires_grad_()

    bck_options = {
        "method": "exactsolve",
    }

    def get_loss(a, matA, matM, P2):
        # get the orthogonal vector for the eigenvectors
        P, _ = torch.linalg.qr(matA)
        PM, _ = torch.linalg.qr(matM)

        # line up the eigenvalues
        b = torch.cat((a[:2], a[1:2], a[2:], a[2:])).to(dtype)

        # construct the matrix
        diag = torch.diag_embed(b)
        A = torch.matmul(torch.matmul(P.transpose(-2, -1).conj(), diag), P)
        M = torch.matmul(PM.transpose(-2, -1).conj(), PM)
        Alinop = LinearOperator.m(A, is_hermitian=True)
        Mlinop = LinearOperator.m(M, is_hermitian=True)

        eivals, eivecs = symeig(
            Alinop, M=Mlinop, neig=neig,
            method=method,
            bck_options=bck_options)
        U = eivecs[:, 1:3]  # the degenerate eigenvectors

        loss = torch.einsum("rc,rc->", torch.matmul(P2, U), U.conj())
        return loss

    gradcheck(get_loss, (a, matA, matM, P2))
    gradgradcheck(get_loss, (a, matA, matM, P2))


@device_dtype_float_test(only64=True)
def test_symeig_A_degenerate_req_not_sat(dtype, device):
    # test if the degenerate gradient returns nan if the requirments are not satisfied

    torch.manual_seed(seed)
    n = 5
    neig = 3
    kwargs = {
        "dtype": dtype,
        "device": device,
    }
    # random matrix to be orthogonalized for the eigenvectors
    mat = torch.randn((n, n), **kwargs).requires_grad_()

    # the degenerate eigenvalues
    a = torch.tensor([1.0, 2.0, 3.0], **kwargs).requires_grad_()
    bck_options = {
        "method": "exactsolve",
    }

    def get_loss(a, mat):
        # get the orthogonal vector for the eigenvectors
        P, _ = torch.linalg.qr(mat)

        # line up the eigenvalues
        b = torch.cat((a[:2], a[1:2], a[2:], a[2:]))

        # construct the matrix
        diag = torch.diag_embed(b)
        A = torch.matmul(torch.matmul(P.T, diag), P)
        Alinop = LinearOperator.m(A)

        eivals, eivecs = symeig(
            Alinop, neig=neig,
            method="custom_exacteig",
            bck_options=bck_options)
        U = eivecs[:, :3]  # the degenerate eigenvectors are in 1,2
        loss = torch.sum(U ** 4)
        return loss

    with warnings.catch_warnings(record=True) as w, enable_debug():
        loss = get_loss(a, mat)
        loss.backward()
        assert len(w) == 1
        wmsg = str(w[0].message).lower()
        assert "degener" in wmsg
        assert "loss function" in wmsg
        assert "incorrect" in wmsg
        assert w[0].category == MathWarning

############## svd #############
@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "shape": [(4, 3), (2, 1, 3, 4)],
    "method": ["exacteig", "custom_exacteig"],
})
def test_svd_A(dtype, device, shape, method):
    torch.manual_seed(seed)
    mat1 = torch.rand(shape, dtype=dtype, device=device)
    mat1 = mat1.requires_grad_()
    linop1 = LinearOperator.m(mat1, is_hermitian=False)
    fwd_options = {"method": method}

    min_mn = min(shape[-1], shape[-2])
    for k in [min_mn]:
        u, s, vh = svd(linop1, k=k, **fwd_options)  # u: (..., m, k), s: (..., k), vh: (..., k, n)
        assert list(u.shape) == list([*linop1.shape[:-1], k])
        assert list(s.shape) == list([*linop1.shape[:-2], k])
        assert list(vh.shape) == list([*linop1.shape[:-2], k, linop1.shape[-1]])

        keye = torch.zeros((*shape[:-2], k, k), dtype=dtype, device=device) + \
            torch.eye(k, dtype=dtype, device=device)
        assert torch.allclose(u.transpose(-2, -1).conj() @ u, keye)
        assert torch.allclose(vh @ vh.transpose(-2, -1).conj(), keye)
        if k == min_mn:
            assert torch.allclose(mat1, u @ torch.diag_embed(s.to(u.dtype)) @ vh)

        def svd_fcn(amat, only_s=False):
            alinop = LinearOperator.m(amat, is_hermitian=False)
            u_, s_, vh_ = svd(alinop, k=k, **fwd_options)
            if only_s:
                return s_
            else:
                return u_.abs(), s_, vh_.abs()

        gradcheck(svd_fcn, (mat1,))
        gradgradcheck(svd_fcn, (mat1,))

############## solve ##############
@device_dtype_float_test()
def test_solve_nonsquare_err(dtype, device):
    torch.manual_seed(seed)
    mat = torch.rand((3, 2), dtype=dtype, device=device)
    mat2 = torch.rand((3, 3), dtype=dtype, device=device)
    linop = LinearOperator.m(mat)
    linop2 = LinearOperator.m(mat2)
    B = torch.rand((3, 1), dtype=dtype, device=device)

    try:
        res = solve(linop, B)
        assert False, "A RuntimeError must be raised if the A linear operator in solve not square"
    except RuntimeError:
        pass

    try:
        res = solve(linop2, B, M=linop)
        assert False, "A RuntimeError must be raised if the M linear operator in solve is not square"
    except RuntimeError:
        pass

@device_dtype_float_test()
def test_solve_mismatch_err(dtype, device):
    torch.manual_seed(seed)
    shapes = [
        #   A      B      M
        ([(3, 3), (2, 1), (3, 3)], "the B shape does not match with A"),
        ([(3, 3), (3, 2), (2, 2)], "the M shape does not match with A"),
    ]
    for (ashape, bshape, mshape), msg in shapes:
        amat = torch.rand(ashape, dtype=dtype, device=device)
        bmat = torch.rand(bshape, dtype=dtype, device=device)
        mmat = torch.rand(mshape, dtype=dtype, device=device) + \
            torch.eye(mshape[-1], dtype=dtype, device=device)
        amat = amat + amat.transpose(-2, -1)
        mmat = mmat + mmat.transpose(-2, -1)

        alinop = LinearOperator.m(amat)
        mlinop = LinearOperator.m(mmat)
        try:
            res = solve(alinop, B=bmat, M=mlinop)
            assert False, "A RuntimeError must be raised if %s" % msg
        except RuntimeError:
            pass

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "ashape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "bshape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "method": ["exactsolve", "custom_exactsolve"],
    "hermit": [False, True],
})
def test_solve_A(dtype, device, ashape, bshape, method, hermit):
    torch.manual_seed(seed)
    na = ashape[-1]
    checkgrad = method.endswith("exactsolve") and len(ashape) == len(bshape) == 2

    ncols = bshape[-1] - 1
    bshape = [*bshape[:-1], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2])) + [na, ncols]
    fwd_options = {"method": method, "min_eps": 1e-9}
    bck_options = {"method": method}

    amat = torch.rand(ashape, dtype=dtype, device=device) * 0.1 + \
        torch.eye(ashape[-1], dtype=dtype, device=device)
    bmat = torch.rand(bshape, dtype=dtype, device=device)

    amat = amat.requires_grad_()
    bmat = bmat.requires_grad_()

    def prepare(amat):
        if hermit:
            return (amat + amat.transpose(-2, -1).conj()) * 0.5
        return amat

    def solvefcn(amat, bmat):
        # is_hermitian=hermit is required to force the hermitian status in numerical gradient
        alinop = LinearOperator.m(prepare(amat), is_hermitian=hermit)
        x = solve(A=alinop, B=bmat,
                  **fwd_options,
                  bck_options=bck_options)
        return x

    x = solvefcn(amat, bmat)
    assert list(x.shape) == xshape

    ax = LinearOperator.m(prepare(amat)).mm(x)
    assert torch.allclose(ax, bmat)

    if checkgrad:
        gradcheck(solvefcn, (amat, bmat))
        gradgradcheck(solvefcn, (amat, bmat))

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "method": ["scipy_gmres", "broyden1", "cg", "bicgstab", "gmres"],
})
def test_solve_A_methods(dtype, device, method):

    if dtype in [torch.complex128, torch.complex64]:
        if method in ["scipy_gmres", "gmres"]:
            pytest.xfail("%s does not work for complex input" % method)

    torch.manual_seed(seed)
    na = 100
    ashape = (na, na)
    bshape = (2, na, na)
    options = {
        "scipy_gmres": {},
        "broyden1": {
            "alpha": -0.2,
        },
        "cg": {
            "rtol": 1e-8  # stringent rtol required to meet the torch.allclose tols
        },
        "bicgstab": {
            "rtol": 1e-8,
        },
        "gmres": {}
    }[method]
    fwd_options = {"method": method, **options}

    ncols = bshape[-1] - 1
    bshape = [*bshape[:-1], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2])) + [na, ncols]

    amat = torch.rand(ashape, dtype=dtype, device=device) * 0.1 + \
        torch.eye(ashape[-1], dtype=dtype, device=device)
    bmat = torch.rand(bshape, dtype=dtype, device=device) + 0.1
    amat = (amat + amat.transpose(-2, -1).conj()) * 0.5

    amat = amat.requires_grad_()
    bmat = bmat.requires_grad_()

    def solvefcn(amat, bmat):
        alinop = LinearOperator.m(amat)
        x = solve(A=alinop, B=bmat, **fwd_options)
        return x

    x = solvefcn(amat, bmat)
    assert list(x.shape) == xshape

    ax = LinearOperator.m(amat).mm(x)

    assert torch.allclose(ax, bmat)

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "ashape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "bshape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "eshape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "method": ["exactsolve", "custom_exactsolve"],
})
def test_solve_AE(dtype, device, ashape, bshape, eshape, method):
    torch.manual_seed(seed)
    na = ashape[-1]

    # save time by enabling gradchecker only on some cases
    checkgrad = method.endswith("exactsolve") and len(ashape) == len(bshape) == len(eshape) == 2

    ncols = bshape[-1] - 1
    bshape = [*bshape[:-1], ncols]
    eshape = [*eshape[:-2], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2], eshape[:-1])) + [na, ncols]
    fwd_options = {"method": method}
    bck_options = {"method": method}

    amat = torch.rand(ashape, dtype=dtype, device=device) * 0.1 + \
        torch.eye(ashape[-1], dtype=dtype, device=device)
    bmat = torch.rand(bshape, dtype=dtype, device=device)
    emat = torch.rand(eshape, dtype=dtype, device=device)

    amat = amat.requires_grad_()
    bmat = bmat.requires_grad_()
    emat = emat.requires_grad_()

    def solvefcn(amat, bmat, emat):
        alinop = LinearOperator.m(amat)
        x = solve(A=alinop, B=bmat, E=emat,
                  **fwd_options,
                  bck_options=bck_options)
        return x

    x = solvefcn(amat, bmat, emat)
    assert list(x.shape) == xshape

    ax = LinearOperator.m(amat).mm(x)
    xe = torch.matmul(x, torch.diag_embed(emat, dim2=-1, dim1=-2))

    assert torch.allclose(ax - xe, bmat)

    if checkgrad:
        gradcheck(solvefcn, (amat, bmat, emat))
        gradgradcheck(solvefcn, (amat, bmat, emat))

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "abeshape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "mshape": [(2, 2), (2, 2, 2), (2, 1, 2, 2)],
    "method": ["exactsolve", "custom_exactsolve"],
})
def test_solve_AEM(dtype, device, abeshape, mshape, method):
    torch.manual_seed(seed)
    na = abeshape[-1]
    ashape = abeshape
    bshape = abeshape
    eshape = abeshape

    # save time by enabling gradchecker only on some cases
    checkgrad = method.endswith("exactsolve") and len(abeshape) == len(mshape) == 2

    ncols = bshape[-1] - 1
    bshape = [*bshape[:-1], ncols]
    eshape = [*eshape[:-2], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2], eshape[:-1], mshape[:-2])) + [na, ncols]
    fwd_options = {"method": method, "min_eps": 1e-9}
    bck_options = {"method": method}  # exactsolve at backward just to test the forward solve

    amat = torch.rand(ashape, dtype=dtype, device=device) * 0.1 + \
        torch.eye(ashape[-1], dtype=dtype, device=device)
    mmat = torch.rand(mshape, dtype=dtype, device=device) * 0.1 + \
        torch.eye(mshape[-1], dtype=dtype, device=device) * 0.5
    bmat = torch.rand(bshape, dtype=dtype, device=device)
    emat = torch.rand(eshape, dtype=dtype, device=device)
    mmat = (mmat + mmat.transpose(-2, -1).conj()) * 0.5

    amat = amat.requires_grad_()
    mmat = mmat.requires_grad_()
    bmat = bmat.requires_grad_()
    emat = emat.requires_grad_()

    def solvefcn(amat, mmat, bmat, emat):
        mmat = (mmat + mmat.transpose(-2, -1).conj()) * 0.5
        alinop = LinearOperator.m(amat)
        mlinop = LinearOperator.m(mmat, is_hermitian=True)
        x = solve(A=alinop, B=bmat, E=emat, M=mlinop,
                  **fwd_options,
                  bck_options=bck_options)
        return x

    x = solvefcn(amat, mmat, bmat, emat)
    assert list(x.shape) == xshape

    ax = LinearOperator.m(amat).mm(x)
    mxe = LinearOperator.m(mmat).mm(torch.matmul(x, torch.diag_embed(emat.to(x.dtype), dim2=-1, dim1=-2)))
    y = ax - mxe
    assert torch.allclose(y, bmat)

    # gradient checker
    if checkgrad:
        gradcheck(solvefcn, (amat, mmat, bmat, emat))
        gradgradcheck(solvefcn, (amat, mmat, bmat, emat))

@device_dtype_float_test(only64=True, include_complex=True, additional_kwargs={
    "method": ["broyden1", "cg", "bicgstab"],
})
def test_solve_AEM_methods(dtype, device, method):
    torch.manual_seed(seed)
    na = 100
    nc = na // 2

    amshape = (na, na)
    eshape = (nc,)
    bshape = (2, na, nc)
    options = {
        "scipy_gmres": {},
        "broyden1": {
            "alpha": -0.2,
        },
        "cg": {
            "rtol": 1e-8  # stringent rtol required to meet the torch.allclose tols
        },
        "bicgstab": {},
    }[method]
    fwd_options = {"method": method, **options}

    amat = torch.rand(amshape, dtype=dtype, device=device) * 0.1 + \
        torch.eye(amshape[-1], dtype=dtype, device=device)
    mmat = torch.rand(amshape, dtype=dtype, device=device) * 0.05 + \
        torch.eye(amshape[-1], dtype=dtype, device=device) * 0.5
    bmat = torch.rand(bshape, dtype=dtype, device=device) + 0.1
    emat = torch.rand(eshape, dtype=dtype, device=device) * 0.1
    amat = (amat + amat.transpose(-2, -1).conj()) * 0.5
    mmat = (mmat + mmat.transpose(-2, -1).conj()) * 0.5

    amat = amat.requires_grad_()
    bmat = bmat.requires_grad_()
    emat = emat.requires_grad_()

    def solvefcn(amat, bmat, emat, mmat):
        alinop = LinearOperator.m(amat)
        mlinop = LinearOperator.m(mmat)
        x = solve(A=alinop, B=bmat, E=emat, M=mlinop, **fwd_options)
        return x

    x = solvefcn(amat, bmat, emat, mmat)
    ax = LinearOperator.m(amat).mm(x)
    mxe = LinearOperator.m(mmat).mm(x) @ torch.diag_embed(emat)
    assert torch.allclose(ax - mxe, bmat)
