import itertools
import torch
from comptorch.core.module import Module
from comptorch.core.editmodule import get_wrap_fcn

class NNModule(torch.nn.Module):
    def __init__(self, a):
        super(NNModule, self).__init__()
        if isinstance(a, torch.nn.Parameter):
            self.a = a
        else:
            self.a = torch.nn.Parameter(a)

    def forward(self, x):
        return self.a + x

class PlainModule(Module):
    def __init__(self, a, b, amodule=True):
        super(PlainModule, self).__init__()
        self.a = self.register(a)
        self.b = b

    def forward(self, x):
        if isinstance(self.a, torch.nn.Module):
            x = self.a(x)
        else:
            x = x + self.a
        return x + self.b

class ListModule(Module):
    def __init__(self, alist, b, amodule=True):
        super(ListModule, self).__init__()
        self.a = self.register(alist)
        self.b = b

    def forward(self, x):
        for i in range(len(self.a)):
            x = x + self.a[i]
        return x + self.b

a = torch.tensor([0.])
a1 = a+1.
a2 = a+2.
a3 = a+3.
b = torch.tensor([10.])
b1 = b+1.
b2 = b+2.
b3 = b+3.
aparam  = torch.nn.Parameter(a )
aparam1 = torch.nn.Parameter(a1)
aparam2 = torch.nn.Parameter(a2)
aparam3 = torch.nn.Parameter(a3)
bparam  = torch.nn.Parameter(b )
bparam1 = torch.nn.Parameter(b1)
bparam2 = torch.nn.Parameter(b2)
bparam3 = torch.nn.Parameter(b3)
x = torch.tensor([100.])
y = torch.tensor([200.])
zero = torch.tensor([0.])

def test_simple():
    nnmod1 = NNModule(aparam)
    mod1 = PlainModule(nnmod1, b) # b is not registered as parameters
    fcn, params = get_wrap_fcn(mod1, [x])
    assert len(params) == 2
    assert params[0] is x
    assert params[1] is aparam

    assert torch.allclose(fcn(y, b1), zero+221) # y + b1 + b (b here is not a parameter)
    assert torch.allclose(fcn(x, b2), zero+122) # x + b2 + b (b here is not a parameter)

def test_module_with_list():
    mod1 = ListModule([aparam, aparam1], bparam)
    sum_fcn, params = get_wrap_fcn(mod1, [x])
    assert len(params) == 4
    assert params[0] is x

    assert torch.allclose(sum_fcn(y,a,a1,a2), zero+203)
    assert torch.allclose(sum_fcn(x,aparam,aparam1,aparam2), zero+103)

def test_duplicate():
    nnmod1 = NNModule(aparam)
    mod1 = PlainModule(nnmod1, aparam) # aparam is a duplicate here
    fcn, params = get_wrap_fcn(mod1, [x])
    assert len(params) == 2
    assert params[0] is x
    assert params[1] is aparam

    assert torch.allclose(fcn(y, b1), zero+222) # y + b1 + b1 (b1 param is counted twice)
    assert torch.allclose(fcn(x, b2), zero+124) # x + b2 + b2
