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

        def forward(self, y, x, A):
            # y: (nbatch, nr)
            # x: (nbatch, nr)
            nbatch = y.shape[0]
            tanh = torch.nn.Tanh()
            A = A.unsqueeze(0).expand(nbatch, -1, -1)
            Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
            Ayx = Ay + x
            return tanh(0.1 * Ayx)

    torch.manual_seed(100)
    random.seed(100)

    dtype = torch.float64
    nr = 3
    nbatch = 1
    A  = torch.randn((nr, nr)).to(dtype).requires_grad_()
    x  = torch.rand((nbatch, nr)).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, nr)).to(dtype)
    params = (x, A)

    model = DummyModule()
    y = lt.equilibrium(model, y0, params)
    assert torch.allclose(y, model(y, *params))

    def getloss(x, A, y0):
        model = DummyModule()
        y = lt.equilibrium(model, y0, (x, A))
        return y

    gradcheck(getloss, (x, A, y0))
    # gradgradcheck(getloss, (x, A, y0), eps=1e-5)
