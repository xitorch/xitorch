import itertools
import torch
import pytest
from torch.autograd import gradcheck, gradgradcheck
from lintorch.core.linop import LinearOperator
from lintorch.linop.lsymeig import lsymeig
from lintorch.linop.solve import solve
from lintorch.utils.bcast import get_bcasted_dims

torch.manual_seed(12345)

class LinOp(LinearOperator):
    def __init__(self, mat, is_hermitian=False):
        super(LinOp, self).__init__(
            shape = mat.shape,
            is_hermitian = is_hermitian,
            dtype = mat.dtype,
            device = mat.device
        )
        self.mat = mat
        self.implemented_methods = ["_mv"]

    def _mv(self, x):
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _rmv(self, x):
        return torch.matmul(self.mat.T, x.unsqueeze(-1)).squeeze(-1)

    def _getparamnames(self):
        return ["mat"]

############## lsymeig ##############
def test_lsymeig_nonhermit_err():
    mat = torch.rand((3,3))
    linop = LinOp(mat, False)
    linop2 = LinOp(mat+mat.transpose(-2,-1), True)

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

def test_lsymeig_mismatch_err():
    mat1 = torch.rand((3,3))
    mat2 = torch.rand((2,2))
    mat1 = mat1 + mat1.transpose(-2,-1)
    mat2 = mat2 + mat2.transpose(-2,-1)
    linop1 = LinOp(mat1, True)
    linop2 = LinOp(mat2, True)

    try:
        res = lsymeig(linop1, M=linop2)
        assert False, "A RuntimeError must be raised if A & M shape are mismatch"
    except RuntimeError:
        pass

def test_lsymeig_A():
    shapes = [(4,4), (2,4,4), (2,3,4,4)]
    for shape in shapes:
        mat1 = torch.rand(shape, dtype=torch.float64)
        mat1 = mat1 + mat1.transpose(-2,-1)
        linop1 = LinOp(mat1, True)

        for neig in [2,shape[-1]]:
            eigvals, eigvecs = lsymeig(linop1, neig=neig) # eigvals: (..., neig), eigvecs: (..., na, neig)
            assert list(eigvecs.shape) == list([*linop1.shape[:-1], neig])
            assert list(eigvals.shape) == list([*linop1.shape[:-2], neig])

            ax = linop1.mm(eigvecs)
            xe = torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1))
            assert torch.allclose(ax, xe)

def test_lsymeig_AM():
    shapes = [(3,3), (2,3,3), (2,1,3,3)]
    dtype = torch.float64
    for ashape,mshape in itertools.product(shapes, shapes):
        mata = torch.rand(ashape, dtype=dtype)
        matm = torch.rand(mshape, dtype=dtype) + torch.eye(mshape[-1], dtype=dtype) # make sure it's not singular
        mata = mata + mata.transpose(-2,-1)
        matm = matm + matm.transpose(-2,-1)
        linopa = LinOp(mata, True)
        linopm = LinOp(matm, True)

        na = ashape[-1]
        bshape = get_bcasted_dims(ashape[:-2], mshape[:-2])
        for neig in [2,ashape[-1]]:
            eigvals, eigvecs = lsymeig(linopa, M=linopm, neig=neig) # eigvals: (..., neig)
            assert list(eigvals.shape) == list([*bshape, neig])
            assert list(eigvecs.shape) == list([*bshape, na, neig])

            ax = linopa.mm(eigvecs)
            mxe = linopm.mm(torch.matmul(eigvecs, torch.diag_embed(eigvals, dim1=-2, dim2=-1)))
            assert torch.allclose(ax, mxe)

############## solve ##############
def test_solve_nonsquare_err():
    mat = torch.rand((3,2))
    mat2 = torch.rand((3,3))
    linop = LinOp(mat)
    linop2 = LinOp(mat2)
    B = torch.rand(3,1)

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

def test_solve_mismatch_err():
    shapes = [
        #   A      B      M
        ([(3,3), (2,1), (3,3)], "the B shape does not match with A"),
        ([(3,3), (3,2), (2,2)], "the M shape does not match with A"),
    ]
    dtype = torch.float64
    for (ashape, bshape, mshape), msg in shapes:
        amat = torch.rand(ashape, dtype=dtype)
        bmat = torch.rand(bshape, dtype=dtype)
        mmat = torch.rand(mshape, dtype=dtype) + torch.eye(mshape[-1], dtype=dtype)
        amat = amat + amat.transpose(-2,-1)
        mmat = mmat + mmat.transpose(-2,-1)

        alinop = LinOp(amat)
        mlinop = LinOp(mmat)
        try:
            res = solve(alinop, B=bmat, M=mlinop)
            assert False, "A RuntimeError must be raised if %s" % msg
        except RuntimeError:
            pass

def test_solve_A():
    na = 3
    shapes = [(na,na), (2,na,na), (2,1,na,na)]
    dtype = torch.float64
    for ashape, bshape in itertools.product(shapes, shapes):
        ncols = bshape[-1]-1
        bshape = [*bshape[:-1], ncols]
        xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2])) + [na, ncols]

        amat = torch.rand(ashape, dtype=dtype) + torch.eye(ashape[-1], dtype=dtype)
        bmat = torch.rand(bshape, dtype=dtype)
        amat = amat + amat.transpose(-2,-1)

        amat = amat.requires_grad_()
        bmat = bmat.requires_grad_()

        def solvefcn(amat, bmat):
            alinop = LinOp(amat)
            x = solve(A=alinop, B=bmat)
            return x

        x = solvefcn(amat, bmat)
        assert list(x.shape) == xshape

        ax = LinOp(amat).mm(x)
        assert torch.allclose(ax, bmat)

        # gradcheck
        gradcheck(solvefcn, (amat, bmat))
        gradgradcheck(solvefcn, (amat, bmat))

# TODO: use fixtures' params to iterate the methods
def test_solve_A_gmres():
    na = 3
    dtype = torch.float64
    ashape = (na, na)
    bshape = (2, na, na)
    fwd_options = {"method": "gmres"}

    ncols = bshape[-1]-1
    bshape = [*bshape[:-1], ncols]
    xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2])) + [na, ncols]

    amat = torch.rand(ashape, dtype=dtype) + torch.eye(ashape[-1], dtype=dtype)
    bmat = torch.rand(bshape, dtype=dtype)
    amat = amat + amat.transpose(-2,-1)

    amat = amat.requires_grad_()
    bmat = bmat.requires_grad_()

    def solvefcn(amat, bmat):
        alinop = LinOp(amat)
        x = solve(A=alinop, B=bmat, fwd_options=fwd_options)
        return x

    x = solvefcn(amat, bmat)
    assert list(x.shape) == xshape

    ax = LinOp(amat).mm(x)
    assert torch.allclose(ax, bmat)

    gradcheck(solvefcn, (amat, bmat))
    gradgradcheck(solvefcn, (amat, bmat))

def test_solve_AE():
    na = 3
    shapes = [(na,na), (2,na,na), (2,1,na,na)]
    dtype = torch.float64
    for ashape, bshape, eshape in itertools.product(shapes, shapes, shapes):
        ncols = bshape[-1]-1
        bshape = [*bshape[:-1], ncols]
        eshape = [*eshape[:-2], ncols]
        xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2], eshape[:-1])) + [na, ncols]

        amat = torch.rand(ashape, dtype=dtype) + torch.eye(ashape[-1], dtype=dtype)
        bmat = torch.rand(bshape, dtype=dtype)
        emat = torch.rand(eshape, dtype=dtype)
        amat = amat + amat.transpose(-2,-1)

        alinop = LinOp(amat)

        x = solve(A=alinop, B=bmat, E=emat)
        assert list(x.shape) == xshape

        ax = alinop.mm(x)
        xe = torch.matmul(x, torch.diag_embed(emat, dim2=-1, dim1=-2))
        assert torch.allclose(ax - xe, bmat)

def test_solve_AEM():
    na = 3
    shapes = [(na,na), (2,na,na), (2,1,na,na)]
    dtype = torch.float64
    for ashape, bshape, eshape, mshape in itertools.product(shapes, shapes, shapes, shapes):
        ncols = bshape[-1]-1
        bshape = [*bshape[:-1], ncols]
        eshape = [*eshape[:-2], ncols]
        xshape = list(get_bcasted_dims(ashape[:-2], bshape[:-2], eshape[:-1], mshape[:-2])) + [na, ncols]

        amat = torch.rand(ashape, dtype=dtype) + torch.eye(ashape[-1], dtype=dtype)
        mmat = torch.rand(mshape, dtype=dtype) + torch.eye(mshape[-1], dtype=dtype)
        bmat = torch.rand(bshape, dtype=dtype)
        emat = torch.rand(eshape, dtype=dtype)
        amat = amat + amat.transpose(-2,-1)
        mmat = mmat + mmat.transpose(-2,-1)

        alinop = LinOp(amat)
        mlinop = LinOp(mmat)

        x = solve(A=alinop, B=bmat, E=emat, M=mlinop)
        assert list(x.shape) == xshape

        ax = alinop.mm(x)
        mxe = mlinop.mm(torch.matmul(x, torch.diag_embed(emat, dim2=-1, dim1=-2)))
        assert torch.allclose(ax - mxe, bmat)
