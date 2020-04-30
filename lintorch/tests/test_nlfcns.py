import random
import torch
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.tests.utils import device_dtype_float_test

class DummyModule(lt.EditableModule):
    def __init__(self, A, addx=True):
        super(DummyModule, self).__init__()
        self.A = A # (nr, nr)
        self.addx = addx
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, diag):
        # x: (nbatch, nr)
        # diag: (nbatch, nr)
        nbatch, nr = x.shape
        x = x.unsqueeze(-1)
        A = self.A.unsqueeze(0).expand(nbatch, -1, -1) # (nbatch, nr, nr)
        A = A + torch.diag_embed(diag) # (nbatch, nr, nr)
        y = torch.bmm(A, x) # (nbatch, nr, ncols)
        yr = self.sigmoid(2*y)
        if self.addx:
            yr = yr + x
        return yr.squeeze(-1)

    def getparams(self, methodname):
        return [self.A]

    def setparams(self, methodname, *params):
        self.A, = params[:1]
        return 1

@device_dtype_float_test(only64=True)
def test_rootfinder(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1
    A = (torch.randn((nr, nr))*0.5).to(dtype).requires_grad_()
    diag = torch.randn((nbatch, nr)).to(dtype).requires_grad_()
    y0 = torch.randn((nbatch, nr)).to(dtype)
    params = (diag,)

    model = DummyModule(A, addx=True)
    y = lt.rootfinder(model.forward, y0, params)
    f = model.forward(y, *params)
    assert torch.allclose(f*0, f)

    def getloss(A, y0, diag):
        model = DummyModule(A, addx=True)
        y = lt.rootfinder(model.forward, y0, (diag,))
        return y

    gradcheck(getloss, (A, y0, diag))
    # gradgradcheck(getloss, (A, y0, diag))

@device_dtype_float_test(only64=True)
def test_equil(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1
    A = (torch.randn((nr, nr))*0.5).to(dtype).requires_grad_()
    diag = torch.randn((nbatch, nr)).to(dtype).requires_grad_()
    y0 = torch.randn((nbatch, nr)).to(dtype)
    params = (diag,)

    model = DummyModule(A, addx=False)
    y = lt.equilibrium(model.forward, y0, params)
    f = model.forward(y, *params)
    assert torch.allclose(y, f)

    def getloss(A, y0, diag):
        model = DummyModule(A, addx=False)
        y = lt.equilibrium(model.forward, y0, (diag,))
        return y

    gradcheck(getloss, (A, y0, diag))
    gradgradcheck(getloss, (A, y0, diag))

@device_dtype_float_test(only64=True)
def test_optimize(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 3
    nbatch = 1
    x  = torch.tensor([-1, 1, 4]).unsqueeze(0).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, 1)).to(dtype)
    params = (y0, x)

    model = PolynomialModule()
    zopt, (yopt,) = lt.optimize(model, (y0,), (x,))
    z = model(yopt, x)
    zmin = x[:,0]-x[:,1]**2/(4*x[:,2])
    assert torch.allclose(zmin, z)
    assert torch.allclose(zmin, zopt)

    def getloss(x, y0):
        model = PolynomialModule()
        zopt, (yopt,) = lt.optimize(model, (y0,), (x,))
        return zopt#, yopt

    gradcheck(getloss, (x, y0))
    gradgradcheck(getloss, (x, y0))
