import random
import torch
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.tests.utils import device_dtype_float_test

@device_dtype_float_test()
def test_equil(dtype, device):
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()

        def forward(self, y, c):
            # y: (nbatch, 1)
            # c: (nbatch, nr)
            nr = c.shape[1]
            power = torch.arange(nr)
            b = (y ** power * c).sum(dim=-1, keepdim=True) # (nbatch, 1)
            return b

    torch.manual_seed(100)
    random.seed(100)

    dtype = torch.float64
    nr = 4
    nbatch = 1
    x  = torch.tensor([-1, 1, 4, 1]).unsqueeze(0).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, 1)).to(dtype)
    params = (x,)

    model = DummyModule()
    y = lt.equilibrium(model, y0, params)
    assert torch.allclose(y, model(y, *params))

    def getloss(x, y0):
        model = DummyModule()
        y = lt.equilibrium(model, y0, (x,))
        return y

    gradcheck(getloss, (x, y0))
    gradgradcheck(getloss, (x, y0))
