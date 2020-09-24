import torch
from xitorch import LinearOperator

class BaseLinOp(LinearOperator):
    def __init__(self, mat, is_hermitian=False):
        super(BaseLinOp, self).__init__(
            shape = mat.shape,
            is_hermitian = is_hermitian,
            dtype = mat.dtype,
            device = mat.device
        )
        self.mat = mat
        self.implemented_methods = []

    def _getparamnames(self, prefix=""):
        return [prefix+"mat"]

class LinOp0(BaseLinOp):
    # LinearOperator where only nothing is implemented (should produce an error)
    def __init__(self, mat, is_hermitian=False):
        super(LinOp0, self).__init__(mat, is_hermitian)
        self.implemented_methods = []

class LinOp1(BaseLinOp):
    # LinearOperator where only ._mv is implemented (bare minimum)
    def __init__(self, mat, is_hermitian=False):
        super(LinOp1, self).__init__(mat, is_hermitian)
        self.implemented_methods = ["_mv"]

    def _mv(self, x):
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

class LinOp2(BaseLinOp):
    # LinearOperator where ._mv and ._rmv are implemented
    def __init__(self, mat, is_hermitian=False):
        super(LinOp2, self).__init__(mat, is_hermitian)
        self.implemented_methods = ["_mv", "_rmv"]

    def _mv(self, x):
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _rmv(self, x):
        return torch.matmul(self.mat.T, x.unsqueeze(-1)).squeeze(-1)

def test_linop0_err():
    mat = torch.rand((3,1,2))
    try:
        linop = LinOp0(mat)
        assert False, "A RuntimeError must be raised when creating a LinearOperator without ._mv()"
    except RuntimeError:
        pass

def test_linop1_shape1_err():
    mat = torch.rand(3)
    mat2 = torch.rand((3,2))
    try:
        linop = LinOp1(mat)
        assert False, "A RuntimeError must be raised when creating a LinearOperator with 1 dimension"
    except RuntimeError:
        pass

    try:
        linop = LinOp1(mat2, is_hermitian=True)
        assert False, "A RuntimeError must be raised when creating a Hermitian LinearOperator without square shape"
    except RuntimeError:
        pass

def test_linop1_mm():
    # test if mm done correctly if not implemented
    mat = torch.rand((2,4,2,3))
    x = torch.rand((3,2))
    xv = torch.rand((3,))

    linop = LinOp1(mat)
    ymv = linop.mv(xv)
    ymm = linop.mm(x)

    assert torch.allclose(ymv, torch.matmul(mat, xv))
    assert torch.allclose(ymm, torch.matmul(mat, x))

def test_linop1_fullmatrix():
    mat = torch.rand((2,4,2,3), dtype=torch.float64)
    linop = LinOp1(mat)
    linop_mat = linop.fullmatrix()
    assert torch.allclose(linop_mat, mat)

def test_linop1_adj_err():
    # see if an error is raised if .H.mv() is called without .rmv() implementation
    mat = torch.rand((2,4,2,3))
    xv = torch.rand((2,))
    x = torch.rand((2,3))
    linop = LinOp1(mat)

    # try the rmv and rmm
    try:
        ymv = linop.rmv(xv)
        assert False, "A RuntimeError must be raised when calling .rmv() if ._rmv is not implemented"
    except RuntimeError:
        pass
    try:
        ymv = linop.rmm(x)
        assert False, "A RuntimeError must be raised when calling .rmm() if ._rmv or ._rmm is not implemented"
    except RuntimeError:
        pass

    # try the adjoint .mv and .mm
    linopH = linop.H
    try:
        ymv = linopH.mv(xv)
        assert False, "A RuntimeError must be raised when calling .H.mv() if ._rmv is not implemented"
    except RuntimeError:
        pass
    try:
        ymv = linopH.mm(x)
        assert False, "A RuntimeError must be raised when calling .H.mm() if ._rmv or ._rmm is not implemented"
    except RuntimeError:
        pass

def test_linop2_rmm():
    # test if rmm done correctly if not implemented
    mat = torch.rand((2,4,2,3))
    rx = torch.rand((2,3))
    rxv = torch.rand((2,))

    linop = LinOp2(mat)
    ymv = linop.rmv(rxv)
    assert torch.allclose(ymv, torch.matmul(mat.T, rxv))

    ymm = linop.rmm(rx)
    assert torch.allclose(ymm, torch.matmul(mat.T, rx))
