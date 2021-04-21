import warnings
import torch
from xitorch import LinearOperator, EditableModule
from xitorch.linalg import solve, symeig

class LinOpWithoutGetParamNames(LinearOperator):
    def __init__(self, mat, is_hermitian=False):
        super(LinOpWithoutGetParamNames, self).__init__(
            shape=mat.shape,
            is_hermitian=is_hermitian,
            dtype=mat.dtype,
            device=mat.device
        )
        self.mat = mat
        self.implemented_methods = []

    def _mv(self, x):
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

class LinOpWithoutInit(LinearOperator):
    def __init__(self, mat, is_hermitian=False):
        self.mat = mat
        self.implemented_methods = []

    def _mv(self, x):
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

class BaseLinOp(LinearOperator):
    def __init__(self, mat, is_hermitian=False):
        super(BaseLinOp, self).__init__(
            shape=mat.shape,
            is_hermitian=is_hermitian,
            dtype=mat.dtype,
            device=mat.device
        )
        self.mat = mat
        self.implemented_methods = []

    def _getparamnames(self, prefix=""):
        return [prefix + "mat"]

class LinOpWithoutMv(BaseLinOp):
    # LinearOperator where only nothing is implemented (should produce an error)
    def __init__(self, mat, is_hermitian=False):
        super(LinOpWithoutMv, self).__init__(mat, is_hermitian)
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
        return torch.matmul(self.mat.transpose(-2, -1), x.unsqueeze(-1)).squeeze(-1)

class NotLinOp1(BaseLinOp):
    # LinearOperator where only ._mv is implemented (bare minimum)
    def __init__(self, mat, is_hermitian=False):
        super(NotLinOp1, self).__init__(mat, is_hermitian)
        self.implemented_methods = ["_mv"]

    def _mv(self, x):
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1) + 1

def test_linop0_err():
    mat = torch.rand((3, 1, 2))
    try:
        linop = LinOpWithoutMv(mat)
        msg = ("A RuntimeError must be raised when creating a "
               "LinearOperator without ._mv()")
        assert False, msg
    except RuntimeError:
        pass

def test_linop_no_getparamnames_err():
    mat = torch.rand((3, 2, 2))
    linop = LinOpWithoutGetParamNames(mat)
    x = torch.rand(2, 4)
    b = linop.mm(x)
    with torch.no_grad():
        solve(linop, b)
    try:
        with torch.enable_grad():
            solve(linop, b)
        msg = ("A RuntimeError must be raised when using a LinearOperator "
               "without ._getparamnames() in xitorch functions")
        assert False, msg
    except RuntimeError:
        pass

def test_linop_no_init_err():
    mat = torch.rand((3, 1, 2))
    x = torch.rand(2)
    linop = LinOpWithoutInit(mat)
    try:
        y = linop.mv(x)
        msg = ("A RuntimeError must be raised when a LinearOperator "
               "is called without .__init__() called first")
        assert False, msg
    except RuntimeError as e:
        assert "__init__" in str(e), 'The error message must contain "__init__"'

def test_linop1_shape1_err():
    mat = torch.rand(3)
    mat2 = torch.rand((3, 2))
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
    mat = torch.rand((2, 4, 2, 3))
    x = torch.rand((3, 2))
    xv = torch.rand((3,))

    linop = LinOp1(mat)
    ymv = linop.mv(xv)
    ymm = linop.mm(x)

    assert torch.allclose(ymv, torch.matmul(mat, xv))
    assert torch.allclose(ymm, torch.matmul(mat, x))

def test_linop1_fullmatrix():
    mat = torch.rand((2, 4, 2, 3), dtype=torch.float64)
    linop = LinOp1(mat)
    linop_mat = linop.fullmatrix()
    assert torch.allclose(linop_mat, mat)

def test_linop1_rmm():
    # test if rmv and rmm done correctly if not implemented
    mat = torch.rand((2, 4, 2, 3))
    rx = torch.rand((2, 3))
    rxv = torch.rand((2,))

    linop = LinOp1(mat)
    ymv = linop.rmv(rxv)
    ymv_true = torch.matmul(mat.transpose(-2, -1), rxv)
    assert torch.allclose(ymv, ymv_true)

    ymm = linop.rmm(rx)
    ymm_true = torch.matmul(mat.transpose(-2, -1), rx)
    assert torch.allclose(ymm, ymm_true)

def test_linop2_rmm():
    # test if rmm done correctly if not implemented
    mat = torch.rand((2, 4, 2, 3))
    rx = torch.rand((2, 3))
    rxv = torch.rand((2,))

    linop = LinOp2(mat)
    ymv = linop.rmv(rxv)
    assert torch.allclose(ymv, torch.matmul(mat.transpose(-2, -1), rxv))

    ymm = linop.rmm(rx)
    assert torch.allclose(ymm, torch.matmul(mat.transpose(-2, -1), rx))

def test_linop_repr():
    dtype = torch.float32
    device = torch.device("cpu")
    mat1 = torch.randn((2, 3, 2), dtype=dtype, device=device)
    a = LinOp1(mat1)

    arepr = ["LinearOperator", "LinOp1", "(2, 3, 2)", "float32", "cpu"]
    _assert_str_contains(a.__repr__(), arepr)

    b = a.H
    brepr = arepr + ["AdjointLinearOperator", "(2, 2, 3)"]
    _assert_str_contains(b.__repr__(), brepr)

    c = b.matmul(a)
    crepr = brepr + ["MatmulLinearOperator", "(2, 2, 2)"]
    _assert_str_contains(c.__repr__(), crepr)

    d = LinearOperator.m(mat1)
    drepr = ["MatrixLinearOperator", "(2, 3, 2)"]
    _assert_str_contains(d.__repr__(), drepr)

    e = d + a
    erepr = ["AddLinearOperator", "(2, 3, 2)"]
    _assert_str_contains(d.__repr__(), drepr)

def test_linop_mat_hermit_err():
    torch.manual_seed(100)
    mat = torch.rand(3, 3)
    mat2 = torch.rand(3, 4)
    msg = "Expecting a RuntimeError for non-symmetric matrix "\
          "indicated as a Hermitian"

    try:
        LinearOperator.m(mat, is_hermitian=True)
        assert False, msg
    except RuntimeError:
        pass

    try:
        LinearOperator.m(mat2, is_hermitian=True)
        assert False, msg
    except RuntimeError:
        pass

def test_linop_op_err():
    mat1 = torch.rand(3, 4)
    vec1 = torch.rand(3)
    vec2 = torch.rand(4)
    errmsg = "Mismatch shape should raise a RuntimeError"
    linop1 = LinOp1(mat1)
    linop2 = LinOp2(mat1)

    try:
        linop1.mv(vec1)
        assert False, errmsg
    except RuntimeError:
        pass

    try:
        linop1.mm(mat1)
        assert False, errmsg
    except RuntimeError:
        pass

    try:
        linop1.rmv(vec2)
        assert False, errmsg
    except RuntimeError:
        pass

    try:
        linop1.rmm(mat1.transpose(-2, -1))
        assert False, errmsg
    except RuntimeError:
        pass

    try:
        linop1.matmul(linop2)
        assert False, errmsg
    except RuntimeError:
        pass

    try:
        linop1.H + linop2
        assert False, errmsg
    except RuntimeError:
        pass

def test_check_linop():
    mat1 = torch.rand(3, 4)
    linop1 = LinOp1(mat1)
    nlinop1 = NotLinOp1(mat1)

    with warnings.catch_warnings(record=True) as w:
        linop1.check()
        assert len(w) == 1
        assert "slow down" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        try:
            nlinop1.check()
            assert False, "AssertionError must be raised for non-LinearOperator"
        except AssertionError:
            pass
        assert len(w) == 1
        assert "slow down" in str(w[0].message)

def test_linop_add():
    mat = torch.randn((2, 3, 2))
    linop1 = LinOp1(mat)
    linop2 = LinOp2(mat + 1)

    # test using non-matrix linop
    c = linop1 + linop2
    assert torch.allclose(c.fullmatrix(), 2 * mat + 1)

    # test using matrix linear operator
    m1 = LinearOperator.m(mat)
    m2 = LinearOperator.m(mat + 1)
    m12 = m1 + m2
    assert torch.allclose(m12.fullmatrix(), 2 * mat + 1)

def test_linop_sub():
    # test the behaviour of subtraction of LinearOperators
    mat = torch.randn((2, 3, 2))
    linop1 = LinOp1(mat)
    linop2 = LinOp2(-mat + 1)

    # test using non-matrix linop
    c = linop1 - linop2
    assert torch.allclose(c.fullmatrix(), 2 * mat - 1)

    # test using matrix linear operator
    m1 = LinearOperator.m(mat)
    m2 = LinearOperator.m(-mat + 1)
    m12 = m1 - m2
    assert torch.allclose(m12.fullmatrix(), 2 * mat - 1)

def test_linop_mul():
    # test the behaviour of multiplication of LinearOperator with a number
    mat = torch.randn((2, 3, 2))
    linop1 = LinOp1(mat)
    linop2 = LinearOperator.m(mat)

    for f1 in [2, 4.0]:
        print(f1)
        # test using non-matrix linop multiplier
        c11l = linop1 * f1
        c11r = f1 * linop1
        # test using matrix linop multiplier
        c12l = linop2 * f1
        c12r = f1 * linop2
        assert torch.allclose(c11l.fullmatrix(), f1 * mat)
        assert torch.allclose(c11r.fullmatrix(), f1 * mat)
        assert torch.allclose(c12l.fullmatrix(), f1 * mat)
        assert torch.allclose(c12r.fullmatrix(), f1 * mat)

def _assert_str_contains(s, slist):
    for c in slist:
        assert c in s

############# special case tests #############

# #1 case: various cases uses linop in editable module
class ClassWithLinop(EditableModule):
    def __init__(self):
        mat = torch.randn(3, 3)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        # using LinOp1 instead of LinearOperator.m to detach .fullmatrix()
        # output with mat
        self.linop = LinOp1(mat, is_hermitian=True)

    def symeig1(self, method):
        return symeig(self.linop, method=method)[1]

    def getparamnames(self, methodname, prefix=""):
        if methodname == "symeig1":
            return self.linop._getparamnames(prefix=prefix + "linop.")
        else:
            raise KeyError()

def test_assertparams_if_fullmatrix_is_called_1():
    a = ClassWithLinop()
    # if fullmatrix stores cache, then assertparams will fail because the
    # method a.symeig1 changes the object's states
    a.assertparams(a.symeig1, "exacteig")

def test_assertparams_if_fullmatrix_is_called_2():
    # This test is involving the linear algebra method that requires the
    # fullmatrix.
    # If ClassWithLinop.symeig2 is called, the method depends on the full matrix
    # of the linop, so if fullmatrix() cache is stored, then it does
    # not directly depend on parameters of linop (but related to the
    # fullmatrix of linop), which makes assertparams raises warnings

    a = ClassWithLinop()

    # calling symeig1 first
    a.symeig1(method="exacteig")
    # doing assert params to make sure there is no cache is stored in
    # fullmatrix (in the old version, this failed)
    with warnings.catch_warnings(record=True) as w:
        a.assertparams(a.symeig1, "exacteig")
        assert len(w) == 0
