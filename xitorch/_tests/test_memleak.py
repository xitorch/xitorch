import random
import torch
from xitorch import LinearOperator, EditableModule
from xitorch.linalg import lsymeig, solve
from xitorch.optimize import rootfinder, equilibrium, minimize
from xitorch._utils.bcast import get_bcasted_dims
from xitorch._tests.utils import device_dtype_float_test, assert_no_memleak

# memory leak tests

seed = 12345

############## linear algebra functions ##############
@device_dtype_float_test(only64=True, onlycpu=True, additional_kwargs={
    "ashape": [(3, 3)],
    "mshape": [(3, 3)],
    "method": ["custom_exacteig"],
})
def test_lsymeig_AM_mem(dtype, device, ashape, mshape, method):

    def _test_lsymeig():
        torch.manual_seed(seed)
        mata = torch.rand(ashape, dtype=dtype, device=device)
        matm = torch.rand(mshape, dtype=dtype, device=device) + \
            torch.eye(mshape[-1], dtype=dtype, device=device)  # make sure it's not singular
        mata = mata + mata.transpose(-2, -1)
        matm = matm + matm.transpose(-2, -1)
        mata = mata.requires_grad_()
        matm = matm.requires_grad_()
        fwd_options = {"method": method}

        na = ashape[-1]
        bshape = get_bcasted_dims(ashape[:-2], mshape[:-2])
        neig = ashape[-1]

        def lsymeig_fcn(amat, mmat):
            # symmetrize
            amat = (amat + amat.transpose(-2, -1)) * 0.5
            mmat = (mmat + mmat.transpose(-2, -1)) * 0.5
            alinop = LinearOperator.m(amat, is_hermitian=True)
            mlinop = LinearOperator.m(mmat, is_hermitian=True)
            eigvals_, eigvecs_ = lsymeig(alinop, M=mlinop, neig=neig, **fwd_options)
            return eigvals_, eigvecs_

        eival, eivec = lsymeig_fcn(mata, matm)
        loss = (eival * eival).sum() + (eivec * eivec).sum()
        # using autograd.grad instead of .backward because backward has a known
        # memory leak problem
        grads = torch.autograd.grad(loss, (mata, matm), create_graph=True)

    assert_no_memleak(_test_lsymeig)

@device_dtype_float_test(only64=True, additional_kwargs={
    "abeshape": [(2, 2)],
    "mshape": [(2, 2)],
    "method": ["custom_exactsolve"],
})
def test_solve_AEM_mem(dtype, device, abeshape, mshape, method):
    def _test_solve():
        torch.manual_seed(seed)
        na = abeshape[-1]
        ashape = abeshape
        bshape = abeshape
        eshape = abeshape
        checkgrad = method.endswith("exactsolve")

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
        mmat = (mmat + mmat.transpose(-2, -1)) * 0.5

        amat = amat.requires_grad_()
        mmat = mmat.requires_grad_()
        bmat = bmat.requires_grad_()
        emat = emat.requires_grad_()

        def solvefcn(amat, mmat, bmat, emat):
            mmat = (mmat + mmat.transpose(-2, -1)) * 0.5
            alinop = LinearOperator.m(amat)
            mlinop = LinearOperator.m(mmat)
            x = solve(A=alinop, B=bmat, E=emat, M=mlinop,
                      **fwd_options,
                      bck_options=bck_options)
            return x

        loss = (solvefcn(amat, mmat, bmat, emat) ** 2).sum()
        # using autograd.grad instead of .backward because backward has a known
        # memory leak problem
        grads = torch.autograd.grad(loss, (amat, mmat, bmat, emat), create_graph=True)

    assert_no_memleak(_test_solve)

############## optimize functions ##############
class DummyModule(EditableModule):
    def __init__(self, a, sumoutput=False):
        super().__init__()
        self.a = a
        self.sumoutput = sumoutput

    def forward(self, x):
        y = self.a ** 2 * x ** 4 - 1
        if self.sumoutput:
            return y.sum()
        else:
            return y

    def getparamnames(self, methodname, prefix=""):
        return [prefix + "a"]

@device_dtype_float_test(only64=True, onlycpu=True)
def test_rootfinder_mem(dtype, device):
    def _test_rf():
        clss = DummyModule  # only need to test it with one clsas
        torch.manual_seed(100)
        random.seed(100)

        nbatch = 2000
        fwd_options = {
            "method": "broyden1",
            "f_tol": 1e-9,
            "alpha": -0.5,
        }

        a = (torch.ones((nbatch,), dtype=dtype) + 0.5).requires_grad_()
        y0 = torch.ones((nbatch,), dtype=dtype)

        def getloss(a):
            model = clss(a)
            y = rootfinder(model.forward, y0, **fwd_options)
            return y

        loss = (getloss(a) ** 2).sum()
        # using torch.autograd.grad instead of backward because backward
        # has a memory leak problem
        grads = torch.autograd.grad(loss, (a,), create_graph=True)

    assert_no_memleak(_test_rf)

@device_dtype_float_test(only64=True, onlycpu=True)
def test_equil_mem(dtype, device):
    def _test_equil():
        clss = DummyModule
        torch.manual_seed(100)
        random.seed(100)

        nbatch = 2000
        fwd_options = {
            "method": "broyden1",
            "f_tol": 1e-9,
            "alpha": -0.5,
        }

        a = (torch.ones((nbatch,), dtype=dtype) + 0.5).requires_grad_()
        y0 = torch.ones((nbatch,), dtype=dtype)

        def getloss(a):
            model = clss(a)
            y = equilibrium(model.forward, y0, **fwd_options)
            return y

        loss = (getloss(a) ** 2).sum()
        grads = torch.autograd.grad(loss, (a,), create_graph=True)

    assert_no_memleak(_test_equil)

@device_dtype_float_test(only64=True, onlycpu=True)
def test_minimize_mem(dtype, device):
    def _test_minimize():
        clss = DummyModule
        torch.manual_seed(400)
        random.seed(100)

        nbatch = 2000
        fwd_options = {
            "method": "broyden1",
            "f_tol": 1e-9,
            "alpha": -0.5,
        }

        a = (torch.ones((nbatch,), dtype=dtype) + 0.5).requires_grad_()
        y0 = torch.ones((nbatch,), dtype=dtype)

        def getloss(a):
            model = clss(a, sumoutput=True)
            y = minimize(model.forward, y0, **fwd_options)
            return y

        loss = (getloss(a) ** 2).sum()
        grads = torch.autograd.grad(loss, (a,), create_graph=True)

    assert_no_memleak(_test_minimize)
