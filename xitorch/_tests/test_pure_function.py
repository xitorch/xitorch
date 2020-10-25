import xitorch
import torch
import pytest
from xitorch._core.pure_function import get_pure_function, PureFunction

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

    def forward2(self, x):
        return x + self.a

    def getparamnames(self, methodname, prefix=""):
        if methodname == "forward":
            return [prefix + "a", prefix + "b"]
        elif methodname == "forward2":
            return [prefix + "a"]
        else:
            raise KeyError()

@pytest.mark.parametrize("fcn", [func1, jitfunc1])
def test_pure_function(fcn):
    pfunc1 = get_pure_function(fcn)
    assert isinstance(pfunc1, PureFunction)
    assert len(pfunc1.objparams()) == 0
    a = torch.tensor(2.0)
    b = torch.tensor(1.0)
    x = torch.tensor(1.5)
    res = x * a + b

    expr = lambda x, a, b: x * a + b
    runtest_pfunc(pfunc1, (x, a, b), expr)

@pytest.mark.parametrize("clss", [TorchModule, MyModule])
def test_module_pfunc(clss):
    a = torch.nn.Parameter(torch.tensor(2.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    x = torch.tensor(1.5)
    module = clss(a, b)
    pfunc = get_pure_function(module.forward)

    expr = lambda x, a, b: x * a + b
    runtest_pfunc(pfunc, (x,), expr)

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

    expr = lambda x, a, b: x * a + b - x
    runtest_pfunc(fminusx, (x, a, b), expr)

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

    expr = lambda x, a, b: x * a + b - x
    runtest_pfunc(fminusx, (x,), expr)

def test_make_sibling_multiple():
    a = torch.nn.Parameter(torch.tensor(2.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    x = torch.tensor(1.5)
    module1 = TorchModule(a, b)
    module2 = MyModule(a, b * 2)

    @xitorch.make_sibling(module1.forward, module2.forward)
    def newfcn(x):
        return module1.forward(x) + module2.forward(x) * 2

    assert isinstance(newfcn, PureFunction)
    assert len(newfcn.objparams()) == 3  # not 4, because a is identical

    expr = lambda x, a, b, b2: (x * a + b) + (x * a + b2) * 2
    runtest_pfunc(newfcn, (x,), expr)

def test_make_sibling_multiple_nonunique_objs():
    a = torch.nn.Parameter(torch.tensor(2.0))
    b = torch.nn.Parameter(torch.tensor(1.0))
    x = torch.tensor(1.5)
    module = MyModule(a, b)

    @xitorch.make_sibling(module.forward, module.forward2)
    def newfcn(x):
        return module.forward(x) + module.forward2(x) * 2

    assert isinstance(newfcn, PureFunction)
    assert len(newfcn.objparams()) == 2

    expr = lambda x, a, b: (x * a + b) + (x + a) * 2
    runtest_pfunc(newfcn, (x,), expr)

def runtest_pfunc(pfunc, params, expr):
    objparams = pfunc.objparams()

    # test original values
    res0 = pfunc(*params)
    res0_true = expr(*params, *objparams)
    assert torch.allclose(res0, res0_true)

    # test changing obj params
    objparams1 = [p + 1.0 for p in objparams]
    res1_true = expr(*params, *objparams1)
    with pfunc.useobjparams(objparams1):
        res1 = pfunc(*params)
    assert torch.allclose(res1, res1_true)

    # test recovery
    res2 = pfunc(*params)
    res2_true = res0_true
    assert torch.allclose(res2, res2_true)

    # test nested
    objparams3 = [p * 2.0 for p in objparams]
    res3_true = expr(*params, *objparams3)
    with pfunc.useobjparams(objparams1):
        assert torch.allclose(pfunc(*params), res1_true)
        with pfunc.useobjparams(objparams3):
            assert torch.allclose(pfunc(*params), res3_true)
        assert torch.allclose(pfunc(*params), res1_true)
    assert torch.allclose(pfunc(*params), res0_true)
