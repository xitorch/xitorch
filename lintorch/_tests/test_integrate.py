import random
import torch
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.grad.jachess import hess
from lintorch.integrate import quad
from lintorch._tests.utils import device_dtype_float_test

class IntegrationNNModule(torch.nn.Module):
    # cos(a*x + b * c)
    def __init__(self, a, b):
        super(IntegrationNNModule, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x, c):
        return torch.cos(self.a * x + self.b * c)

class IntegrationModule(lt.EditableModule):
    # cos(a*x + b * c)
    def __init__(self, a, b):
        # super(IntegrationModule, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x, c):
        return torch.cos(self.a * x + self.b * c)

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"a", prefix+"b"]

@device_dtype_float_test(only64=True)
def test_quad(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    nr = 2
    fwd_options = {
        "method": "leggauss",
        "n": 100,
    }

    a = torch.nn.Parameter(torch.rand((nr,), dtype=dtype, device=device).requires_grad_())
    b = torch.nn.Parameter(torch.randn((nr,), dtype=dtype, device=device).requires_grad_())
    c = torch.randn((nr,), dtype=dtype, device=device).requires_grad_()
    xl = torch.zeros((1,), dtype=dtype, device=device).requires_grad_()
    xu = (torch.ones ((1,), dtype=dtype, device=device) * 0.5).requires_grad_()

    for clss in [IntegrationModule, IntegrationNNModule]:

        module = clss(a, b)
        y = quad(module.forward, xl, xu, params=(c,), fwd_options=fwd_options)
        ytrue = (torch.sin(a * xu + b * c) - torch.sin(a * xl + b * c)) / a
        assert torch.allclose(y, ytrue)

        def getloss(a, b, c, xl, xu):
            module = clss(a, b)
            y = quad(module.forward, xl, xu, params=(c,), fwd_options=fwd_options)
            return y

        gradcheck(getloss, (a, b, c, xl, xu))
        gradgradcheck(getloss, (a, b, c, xl, xu))

if __name__ == "__main__":
    test_quad()
