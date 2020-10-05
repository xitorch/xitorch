import xitorch
import torch
import pytest
from xitorch._core.pure_function import get_pure_function, make_sibling, PureFunction

def func1(x, a, b):
    return x * a + b

@torch.jit.script
def jitfunc1(x, a, b):
    return x * a + b

class TorchModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x * self.a + self.b

class MyModule(xitorch.EditableModule):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, x):
        return x * self.a + self.b

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"a", prefix+"b"]

@pytest.mark.parametrize("fcn", [func1, jitfunc1])
def test_pure_function(fcn):
    pfunc1 = get_pure_function(fcn)
    assert isinstance(pfunc1, PureFunction)
    assert len(pfunc1.objparams()) == 0
    a = torch.tensor(2.0)
    b = torch.tensor(1.0)
    x = torch.tensor(1.5)
    res = x * a + b
    assert torch.allclose(pfunc1(x, a, b), res)

def test_nnmodule_pfunc():
    a = torch.nn.Parameter(torch.tensor(2.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    x = torch.tensor(1.5)
    module = TorchModule(a, b)
    pfunc = get_pure_function(module)
    assert isinstance(pfunc, PureFunction)
    assert len(pfunc.objparams()) == 2
    res = x * a + b
    assert torch.allclose(pfunc(x), res)

def test_edmodule_pfunc():
    a = torch.tensor(2.0)
    b = torch.tensor(1.0)
    x = torch.tensor(1.5)
    module = MyModule(a, b)
    pfunc = get_pure_function(module.forward)
    assert isinstance(pfunc, PureFunction)
    assert len(pfunc.objparams()) == 2
    res = x * a + b
    assert torch.allclose(pfunc(x), res)

@pytest.mark.parametrize("fcn", [func1, jitfunc1])
def test_make_sibling_fcn(fcn):
    a = torch.tensor(2.0)
    b = torch.tensor(1.0)
    x = torch.tensor(1.5)

    @xitorch.make_sibling(fcn)
    def fminusx(x, a, b):
        return fcn(x, a, b) - x

    assert isinstance(fminusx, PureFunction)
    assert len(fminusx.objparams()) == 0
    res = x * a + b
    assert torch.allclose(fminusx(x, a, b), res - x)

@pytest.mark.parametrize("clss", [TorchModule, MyModule])
def test_make_sibling_method(clss):
    a = torch.nn.Parameter(torch.tensor(2.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    x = torch.tensor(1.5)
    module = clss(a, b)

    @xitorch.make_sibling(module.forward)
    def fminusx(x):
        return module.forward(x) - x

    assert isinstance(fminusx, PureFunction)
    assert len(fminusx.objparams()) == 2
    res = x * a + b
    assert torch.allclose(fminusx(x), res - x)

    # test if the state is changed
    a2 = torch.tensor(3.)
    b2 = torch.tensor(2.)
    params = (a2, b2)
    with fminusx.useobjparams(params):
        res2 = x * a2 + b2
        assert torch.allclose(fminusx(x), res2 - x)
