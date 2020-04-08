import random
import torch
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.tests.utils import device_dtype_float_test

class PolynomialModule(torch.nn.Module):
    def __init__(self):
        super(PolynomialModule, self).__init__()

    def forward(self, y, c):
        # y: (nbatch, 1)
        # c: (nbatch, nr)
        nr = c.shape[1]
        power = torch.arange(nr)
        b = (y ** power * c).sum(dim=-1, keepdim=True) # (nbatch, 1)
        return b

@device_dtype_float_test(only64=True)
def test_rootfinder(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1
    x  = torch.tensor([-1, 1, 4, 1]).unsqueeze(0).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, 1)).to(dtype)
    params = (x,)

    model = PolynomialModule()
    y = lt.rootfinder(model, y0, params)
    f = model(y, *params)
    assert torch.allclose(f*0, f)

    def getloss(x, y0):
        model = PolynomialModule()
        y = lt.equilibrium(model, y0, (x,))
        return y

    gradcheck(getloss, (x, y0))
    gradgradcheck(getloss, (x, y0))

@device_dtype_float_test(only64=True)
def test_equil(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1
    x  = torch.tensor([-1, 1, 4, 1]).unsqueeze(0).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, 1)).to(dtype)
    params = (x,)

    model = PolynomialModule()
    y = lt.equilibrium(model, y0, params)
    assert torch.allclose(y, model(y, *params))

    def getloss(x, y0):
        model = PolynomialModule()
        y = lt.equilibrium(model, y0, (x,))
        return y

    gradcheck(getloss, (x, y0))
    gradgradcheck(getloss, (x, y0))

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
