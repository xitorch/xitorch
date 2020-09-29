import itertools
import torch
import pytest
from torch.autograd import gradcheck, gradgradcheck
from xitorch import LinearOperator
from xitorch.linalg.symeig import lsymeig, symeig, svd
from xitorch.linalg.solve import solve
from xitorch._utils.bcast import get_bcasted_dims
from xitorch._tests.utils import device_dtype_float_test

seed = 12345

############## lsymeig ##############
@device_dtype_float_test()
def test_lsymeig_nonhermit_err(dtype, device):
    torch.manual_seed(seed)
    mat = torch.rand((3,3), dtype=dtype, device=device)
    linop = LinearOperator.m(mat, False)
    linop2 = LinearOperator.m(mat+mat.transpose(-2,-1), True)

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
    mat1 = torch.rand((3,3), dtype=dtype, device=device)
    mat2 = torch.rand((2,2), dtype=dtype, device=device)
    mat1 = mat1 + mat1.transpose(-2,-1)
    mat2 = mat2 + mat2.transpose(-2,-1)
    linop1 = LinearOperator.m(mat1, True)
    linop2 = LinearOperator.m(mat2, True)

    try:
        res = lsymeig(linop1, M=linop2)
        assert False, "A RuntimeError must be raised if A & M shape are mismatch"
    except RuntimeError:
        pass


@device_dtype_float_test(only64=True, additional_kwargs={
    "shape": [(4,4), (2,4,4), (2,3,4,4)],
    "method": ["exacteig", "custom_exacteig"], # only 2 of methods, because both gradient implementations are covered
})
def test_lsymeig_A(dtype, device, shape, method):
    torch.manual_seed(seed)
    mat1 = torch.rand(shape, dtype=dtype, device=device)
    mat1 = mat1 + mat1.transpose(-2,-1)
    mat1 = mat1.requires_grad_()
    linop1 = LinearOperator.m(mat1, True)
    fwd_options = {"method": method}

    for neig in [2,shape[-1]]:
        eigvals, eigvecs = lsymeig(linop1, neig=neig, **fwd_options) # eigvals: (..., neig), eigvecs: (..., na, neig)
        assert list(eigvecs.shape) == list([*linop1.shape[:-1], neig])
        assert list(eigvals.shape) == list([*linop1.shape[:-2], neig])

        ax = linop1.mm(eigvecs)
        xe = torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1))
        assert torch.allclose(ax, xe)

        # only perform gradcheck if neig is full, to reduce the computational cost
        if neig == shape[-1]:
            def lsymeig_fcn(amat):
                amat = (amat + amat.transpose(-2,-1)) * 0.5 # symmetrize
                alinop = LinearOperator.m(amat, is_hermitian=True)
                eigvals_, eigvecs_ = lsymeig(alinop, neig=neig, **fwd_options)
                return eigvals_, eigvecs_

            gradcheck(lsymeig_fcn, (mat1,))
            gradgradcheck(lsymeig_fcn, (mat1,))

@device_dtype_float_test(only64=True, additional_kwargs={
    "ashape": [(3,3), (2,3,3), (2,1,3,3)],
    "mshape": [(3,3), (2,3,3), (2,1,3,3)],
    "method": ["exacteig", "custom_exacteig"], # only 2 of methods, because both gradient implementations are covered
})
def test_lsymeig_AM(dtype, device, ashape, mshape, method):
    torch.manual_seed(seed)
    mata = torch.rand(ashape, dtype=dtype, device=device)
    matm = torch.rand(mshape, dtype=dtype, device=device) + \
           torch.eye(mshape[-1], dtype=dtype, device=device) # make sure it's not singular
    mata = mata + mata.transpose(-2,-1)
    matm = matm + matm.transpose(-2,-1)
    mata = mata.requires_grad_()
    matm = matm.requires_grad_()
    linopa = LinearOperator.m(mata, True)
    linopm = LinearOperator.m(matm, True)
    fwd_options = {"method": method}

    na = ashape[-1]
    bshape = get_bcasted_dims(ashape[:-2], mshape[:-2])
    for neig in [2,ashape[-1]]:
        eigvals, eigvecs = lsymeig(linopa, M=linopm, neig=neig, **fwd_options) # eigvals: (..., neig)
        assert list(eigvals.shape) == list([*bshape, neig])
        assert list(eigvecs.shape) == list([*bshape, na, neig])

        ax = linopa.mm(eigvecs)
        mxe = linopm.mm(torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1)))
        assert torch.allclose(ax, mxe)

        # only perform gradcheck if neig is full, to reduce the computational cost
        if neig == ashape[-1]:
            def lsymeig_fcn(amat, mmat):
                # symmetrize
                amat = (amat + amat.transpose(-2,-1)) * 0.5
                mmat = (mmat + mmat.transpose(-2,-1)) * 0.5
                alinop = LinearOperator.m(amat, is_hermitian=True)
                mlinop = LinearOperator.m(mmat, is_hermitian=True)
                eigvals_, eigvecs_ = lsymeig(alinop, M=mlinop, neig=neig, **fwd_options)
                return eigvals_, eigvecs_

            gradcheck(lsymeig_fcn, (mata, matm))
            gradgradcheck(lsymeig_fcn, (mata, matm))

@device_dtype_float_test(only64=True, additional_kwargs={
    "shape": [(1000,1000), (2,1000,1000), (2,3,1000,1000)],
    "method": ["davidson"], # list the methods here
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
            return [prefix+"b"]

    neig = 2
    na = shape[-1]
    linop1 = ALarge(shape, dtype=dtype, device=device)
    fwd_options = {"method": method, "min_eps": 1e-8}

    eigvals, eigvecs = symeig(linop1, mode=mode, neig=neig, **fwd_options) # eigvals: (..., neig), eigvecs: (..., na, neig)

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

############## svd #############
@device_dtype_float_test(only64=True, additional_kwargs={
    "shape": [(4,3), (2,1,3,4)],
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
        u, s, vh = svd(linop1, k=k, **fwd_options) # u: (..., m, k), s: (..., k), vh: (..., k, n)
        assert list(u.shape) == list([*linop1.shape[:-1], k])
        assert list(s.shape) == list([*linop1.shape[:-2], k])
        assert list(vh.shape) == list([*linop1.shape[:-2], k, linop1.shape[-1]])

        keye = torch.zeros((*shape[:-2], k, k), dtype=dtype, device=device) + \
               torch.eye(k, dtype=dtype, device=device)
        assert torch.allclose(u.transpose(-2,-1) @ u, keye)
        assert torch.allclose(vh @ vh.transpose(-2, -1), keye)
        if k == min_mn:
            assert torch.allclose(mat1, u @ torch.diag_embed(s) @ vh)

        def svd_fcn(amat, only_s=False):
            alinop = LinearOperator.m(amat, is_hermitian=False)
            u_, s_, vh_ = svd(alinop, k=k, **fwd_options)
            if only_s:
                return s_
            else:
                return u_, s_, vh_

        gradcheck(svd_fcn, (mat1,))
        gradgradcheck(svd_fcn, (mat1, True))

############## solve ##############
@device_dtype_float_test()
def test_solve_nonsquare_err(dtype, device):
    torch.manual_seed(seed)
    mat = torch.rand((3,2), dtype=dtype, device=device)
    mat2 = torch.rand((3,3), dtype=dtype, device=device)
    linop = LinearOperator.m(mat)
    linop2 = LinearOperator.m(mat2)
    B = torch.rand((3,1), dtype=dtype, device=device)

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
        ([(3,3), (2,1), (3,3)], "the B shape does not match with A"),
        ([(3,3), (3,2), (2,2)], "the M shape does not match with A"),
    ]
    for (ashape, bshape, mshape), msg in shapes:
        amat = torch.rand(ashape, dtype=dtype, device=device)
        bmat = torch.rand(bshape, dtype=dtype, device=device)
        mmat = torch.rand(mshape, dtype=dtype, device=device) + \
               torch.eye(mshape[-1], dtype=dtype, device=device)
        amat = amat + amat.transpose(-2,-1)
        mmat = mmat + mmat.transpose(-2,-1)

        alinop = LinearOperator.m(amat)
        mlinop = LinearOperator.m(mmat)
        try:
            res = solve(alinop, B=bmat, M=mlinop)
            assert False, "A RuntimeError must be raised if %s" % msg
        except RuntimeError:
            pass

@device_dtype_float_test(only64=True, additional_kwargs={
    "ashape": [(2,2), (2,2,2), (2,1,2,2)],
    "bshape": [(2,2), (2,2,2), (2,1,2,2)],
    "method": ["exactsolve", "custom_exactsolve"],
    "hermit": [False, True],
})
def test_solve_A(dtype, device, ashape, bshape, method, hermit):
    torch.manual_seed(seed)
    na = ashape[-1]
    checkgrad = method.endswith("exactsolve")

    ncols = bshape[-1]-1
    bshape = [*bshape[:-1], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2])) + [na, ncols]
    fwd_options = {"method": method, "min_eps": 1e-9}
    bck_options = {"method": method}

    amat = torch.rand(ashape, dtype=dtype, device=device) * 0.1 + \
           torch.eye(ashape[-1], dtype=dtype, device=device)
    bmat = torch.rand(bshape, dtype=dtype, device=device)
    if hermit:
        amat = (amat + amat.transpose(-2,-1)) * 0.5

    amat = amat.requires_grad_()
    bmat = bmat.requires_grad_()

    def solvefcn(amat, bmat):
        # is_hermitian=hermit is required to force the hermitian status in numerical gradient
        alinop = LinearOperator.m(amat, is_hermitian=hermit)
        x = solve(A=alinop, B=bmat,
            **fwd_options,
            bck_options=bck_options)
        return x

    x = solvefcn(amat, bmat)
    assert list(x.shape) == xshape

    ax = LinearOperator.m(amat).mm(x)
    assert torch.allclose(ax, bmat)

    if checkgrad:
        gradcheck(solvefcn, (amat, bmat))
        gradgradcheck(solvefcn, (amat, bmat))

@device_dtype_float_test(only64=True, additional_kwargs={
    "method": ["gmres", "broyden1"],
})
def test_solve_A_methods(dtype, device, method):
    torch.manual_seed(seed)
    na = 3
    ashape = (na, na)
    bshape = (2, na, na)
    fwd_options = {"method": method}

    ncols = bshape[-1]-1
    bshape = [*bshape[:-1], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2])) + [na, ncols]

    amat = torch.rand(ashape, dtype=dtype, device=device) + \
           torch.eye(ashape[-1], dtype=dtype, device=device)
    bmat = torch.rand(bshape, dtype=dtype, device=device)
    amat = amat + amat.transpose(-2,-1)

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

@device_dtype_float_test(only64=True, additional_kwargs={
    "ashape": [(2,2), (2,2,2), (2,1,2,2)],
    "bshape": [(2,2), (2,2,2), (2,1,2,2)],
    "eshape": [(2,2), (2,2,2), (2,1,2,2)],
    "method": ["exactsolve", "custom_exactsolve"],
})
def test_solve_AE(dtype, device, ashape, bshape, eshape, method):
    torch.manual_seed(seed)
    na = ashape[-1]
    checkgrad = method.endswith("exactsolve")

    ncols = bshape[-1]-1
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

    # grad check only performed at AEM, to save time
    # if checkgrad:
    #     gradcheck(solvefcn, (amat, bmat, emat))
    #     gradgradcheck(solvefcn, (amat, bmat, emat))

@device_dtype_float_test(only64=True, additional_kwargs={
    "abeshape": [(2,2), (2,2,2), (2,1,2,2)],
    "mshape": [(2,2), (2,2,2), (2,1,2,2)],
    "method": ["exactsolve", "custom_exactsolve"],
})
def test_solve_AEM(dtype, device, abeshape, mshape, method):
    torch.manual_seed(seed)
    na = abeshape[-1]
    ashape = abeshape
    bshape = abeshape
    eshape = abeshape
    checkgrad = method.endswith("exactsolve")

    ncols = bshape[-1]-1
    bshape = [*bshape[:-1], ncols]
    eshape = [*eshape[:-2], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2], eshape[:-1], mshape[:-2])) + [na, ncols]
    fwd_options = {"method": method, "min_eps": 1e-9}
    bck_options = {"method": method} # exactsolve at backward just to test the forward solve

    amat = torch.rand(ashape, dtype=dtype, device=device) * 0.1 + \
           torch.eye(ashape[-1], dtype=dtype, device=device)
    mmat = torch.rand(mshape, dtype=dtype, device=device) * 0.1 + \
           torch.eye(mshape[-1], dtype=dtype, device=device) * 0.5
    bmat = torch.rand(bshape, dtype=dtype, device=device)
    emat = torch.rand(eshape, dtype=dtype, device=device)
    mmat = (mmat + mmat.transpose(-2,-1)) * 0.5

    amat = amat.requires_grad_()
    mmat = mmat.requires_grad_()
    bmat = bmat.requires_grad_()
    emat = emat.requires_grad_()

    def solvefcn(amat, mmat, bmat, emat):
        mmat = (mmat + mmat.transpose(-2,-1)) * 0.5
        alinop = LinearOperator.m(amat)
        mlinop = LinearOperator.m(mmat)
        x = solve(A=alinop, B=bmat, E=emat, M=mlinop,
            **fwd_options,
            bck_options=bck_options)
        return x

    x = solvefcn(amat, mmat, bmat, emat)
    assert list(x.shape) == xshape

    ax = LinearOperator.m(amat).mm(x)
    mxe = LinearOperator.m(mmat).mm(torch.matmul(x, torch.diag_embed(emat, dim2=-1, dim1=-2)))
    y = ax - mxe
    assert torch.allclose(y, bmat)

    # gradient checker
    if checkgrad:
        gradcheck(solvefcn, (amat, mmat, bmat, emat))
        gradgradcheck(solvefcn, (amat, mmat, bmat, emat))
