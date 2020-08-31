import random
import torch
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.optimize import rootfinder, equilibrium
from lintorch._tests.utils import device_dtype_float_test

class DummyModule(lt.EditableModule):
    def __init__(self, A, addx=True):
        super(DummyModule, self).__init__()
        self.A = A # (nr, nr)
        self.addx = addx
        self.sigmoid = torch.nn.Sigmoid()

    def set_diag_bias(self, diag, bias):
        self.diag = diag
        self.bias = bias

    def forward(self, x):
        # x: (nbatch, nr)
        # diag: (nbatch, nr)
        # bias: (nbatch, nr)
        nbatch, nr = x.shape
        x = x.unsqueeze(-1)
        A = self.A.unsqueeze(0).expand(nbatch, -1, -1) # (nbatch, nr, nr)
        A = A + torch.diag_embed(self.diag) # (nbatch, nr, nr)
        y = torch.bmm(A, x).squeeze(-1) # (nbatch, nr)
        yr = self.sigmoid(2*y) + 2*self.bias
        if self.addx:
            yr = yr + x.squeeze(-1)
        return yr

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"A", prefix+"diag", prefix+"bias"]

class DummyNNModule(torch.nn.Module):
    def __init__(self, A, addx=True):
        super(DummyNNModule, self).__init__()
        self.A = A
        self.addx = addx
        self.sigmoid = torch.nn.Sigmoid()

    def set_diag_bias(self, diag, bias):
        self.diag = diag
        self.bias = bias

    def forward(self, x):
        # x: (nbatch, nr)
        # diag: (nbatch, nr)
        # bias: (nbatch, nr)
        nbatch, nr = x.shape
        x = x.unsqueeze(-1)
        A = self.A.unsqueeze(0).expand(nbatch, -1, -1) # (nbatch, nr, nr)
        A = A + torch.diag_embed(self.diag) # (nbatch, nr, nr)
        y = torch.bmm(A, x).squeeze(-1) # (nbatch, nr)
        yr = self.sigmoid(2*y) + 2*self.bias
        if self.addx:
            yr = yr + x.squeeze(-1)
        return yr

class DummyModuleExplicit(lt.EditableModule):
    def __init__(self, addx=True):
        super(DummyModuleExplicit, self).__init__()
        self.addx = addx
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, A, diag, bias):
        nbatch, nr = x.shape
        x = x.unsqueeze(-1)
        A = A.unsqueeze(0).expand(nbatch, -1, -1)
        A = A + torch.diag_embed(diag)
        y = torch.bmm(A, x).squeeze(-1)
        yr = self.sigmoid(2*y) + 2*bias
        if self.addx:
            yr = yr + x.squeeze(-1)
        return yr

    def getparamnames(self, methodname, prefix=""):
        return []

@device_dtype_float_test(only64=True)
def test_rootfinder(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1

    for clss in [DummyModule, DummyNNModule]:
        A    = torch.nn.Parameter((torch.randn((nr, nr))*0.5).to(dtype).requires_grad_())
        diag = torch.nn.Parameter(torch.randn((nbatch, nr)).to(dtype).requires_grad_())
        bias = torch.nn.Parameter(torch.zeros((nbatch, nr)).to(dtype).requires_grad_())
        y0 = torch.randn((nbatch, nr)).to(dtype)

        model = clss(A, addx=True)
        model.set_diag_bias(diag, bias)
        y = rootfinder(model.forward, y0)
        f = model.forward(y)
        assert torch.allclose(f*0, f)

        def getloss(A, y0, diag, bias):
            model = clss(A, addx=True)
            model.set_diag_bias(diag, bias)
            y = rootfinder(model.forward, y0)
            return y

        gradcheck(getloss, (A, y0, diag, bias))
        gradgradcheck(getloss, (A, y0, diag, bias))

@device_dtype_float_test(only64=True)
def test_equil(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1

    for clss in [DummyModule, DummyNNModule]:
        A    = torch.nn.Parameter((torch.randn((nr, nr))*0.5).to(dtype).requires_grad_())
        diag = torch.nn.Parameter(torch.randn((nbatch, nr)).to(dtype).requires_grad_())
        bias = torch.nn.Parameter(torch.zeros((nbatch, nr)).to(dtype).requires_grad_())
        y0 = torch.randn((nbatch, nr)).to(dtype)

        model = clss(A, addx=False)
        model.set_diag_bias(diag, bias)
        y = equilibrium(model.forward, y0)
        f = model.forward(y)
        assert torch.allclose(y, f)

        def getloss(A, y0, diag, bias):
            model = clss(A, addx=False)
            model.set_diag_bias(diag, bias)
            y = equilibrium(model.forward, y0)
            return y

        gradcheck(getloss, (A, y0, diag, bias))
        gradgradcheck(getloss, (A, y0, diag, bias))

@device_dtype_float_test(only64=True)
def test_rootfinder_with_params(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1

    clss = DummyModuleExplicit
    for bias_is_tensor in [True, False]:
        A    = (torch.randn((nr, nr))*0.5).to(dtype).requires_grad_()
        diag = torch.randn((nbatch, nr)).to(dtype).requires_grad_()
        if bias_is_tensor:
            bias = torch.zeros((nbatch, nr)).to(dtype).requires_grad_()
        else:
            bias = 0.0
        y0 = torch.randn((nbatch, nr)).to(dtype)

        model = clss(addx=True)
        y = rootfinder(model.forward, y0, (A, diag, bias))
        f = model.forward(y, A, diag, bias)
        assert torch.allclose(f*0, f)

        def getloss(y0, A, diag, bias):
            model = clss(addx=True)
            y = rootfinder(model.forward, y0, (A, diag, bias))
            return y

        gradcheck(getloss, (y0, A, diag, bias))
        gradgradcheck(getloss, (y0, A, diag, bias))

# @device_dtype_float_test(only64=True)
# def test_optimize(dtype, device):
#     torch.manual_seed(100)
#     random.seed(100)
#     dtype = torch.float64
#
#     nr = 3
#     nbatch = 1
#     x  = torch.tensor([-1, 1, 4]).unsqueeze(0).to(dtype).requires_grad_()
#     y0 = torch.rand((nbatch, 1)).to(dtype)
#     params = (y0, x)
#
#     model = PolynomialModule()
#     zopt, (yopt,) = lt.optimize(model, (y0,), (x,))
#     z = model(yopt, x)
#     zmin = x[:,0]-x[:,1]**2/(4*x[:,2])
#     assert torch.allclose(zmin, z)
#     assert torch.allclose(zmin, zopt)
#
#     def getloss(x, y0):
#         model = PolynomialModule()
#         zopt, (yopt,) = lt.optimize(model, (y0,), (x,))
#         return zopt#, yopt
#
#     gradcheck(getloss, (x, y0))
#     gradgradcheck(getloss, (x, y0))
