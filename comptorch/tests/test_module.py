import itertools
import torch
from comptorch.core.module import CModule

class NNModule(torch.nn.Module):
    def __init__(self, a):
        super(NNModule, self).__init__()
        if isinstance(a, torch.nn.Parameter):
            self.a = a
        else:
            self.a = torch.nn.Parameter(a)

class PlainModule(CModule):
    def __init__(self, a, b):
        super(PlainModule, self).__init__()
        self.a = self.register(a)
        self.b = b

class NestedModule(CModule):
    def __init__(self, mod, a, b):
        super(NestedModule, self).__init__()
        self.mod = mod
        self.a = self.register(a)
        self.b = b

def test_plain_module():
    # plain module apply self.register to a and plain assignment for b

    a = torch.tensor([1.])
    b = torch.tensor([2.])
    aparam = torch.nn.Parameter(a)
    bparam = torch.nn.Parameter(b)
    mod1 = PlainModule(a, b)
    named_params_dict1 = {
        "a": a
    }
    assert_mod_dict(mod1, named_params_dict1)

    mod2 = PlainModule(aparam, b)
    named_params_dict2 = {
        "a": aparam
    }
    assert_mod_dict(mod2, named_params_dict2)

    mod3 = PlainModule(a, bparam)
    named_params_dict3 = {
        "a": a,
        "b": bparam # b should be registered as well because it is a parameter
    }
    assert_mod_dict(mod3, named_params_dict3)

    mod4 = PlainModule(aparam, bparam)
    named_params_dict4 = {
        "a": aparam,
        "b": bparam
    }
    assert_mod_dict(mod4, named_params_dict4)

def test_nested_module_simple():
    # nested module apply self.register to a and plain assignment for b
    a = torch.tensor([1.])
    b = torch.tensor([2.])
    a1 = a + 1
    a2 = 2 * a
    a3 = 3 * a
    b2 = 2 * b
    aparam = torch.nn.Parameter(a)
    bparam = torch.nn.Parameter(b)
    aparam2 = torch.nn.Parameter(a2)
    bparam2 = torch.nn.Parameter(b2)
    nnmod = NNModule(aparam)
    plmod1 = PlainModule(a2, b)
    plmod2 = PlainModule(aparam2, bparam)

    mod1 = NestedModule(nnmod, a1, b)
    named_params_dict1 = {
        "mod.a": aparam,
        "a": a1,
    }
    assert_mod_dict(mod1, named_params_dict1)

    mod2 = NestedModule(plmod1, a, b)
    named_params_dict2 = {
        "mod.a": a2,
        "a": a
    }
    assert_mod_dict(mod2, named_params_dict2)

    mod3 = NestedModule(plmod2, a3, b)
    named_params_dict3 = {
        "mod.a": aparam2,
        "mod.b": bparam,
        "a": a3
    }
    assert_mod_dict(mod3, named_params_dict3)

    mod4 = NestedModule(mod3, a2, b)
    named_params_dict4 = {
        "mod.mod.a": aparam2,
        "mod.mod.b": bparam,
        "mod.a": a3,
        "a": a2
    }
    assert_mod_dict(mod4, named_params_dict4)

    mod5 = NestedModule(mod3, mod2, b)
    named_params_dict5 = {
        "mod.mod.a": aparam2,
        "mod.mod.b": bparam,
        "mod.a": a3,
        "a.mod.a": a2,
        "a.a": a
    }
    assert_mod_dict(mod5, named_params_dict5)

    mod6 = NestedModule(mod3, mod2, mod1)
    named_params_dict6 = {
        "mod.mod.a": aparam2,
        "mod.mod.b": bparam,
        "mod.a": a3,
        # from mod2
        "a.mod.a": a2,
        "a.a": a,
        # from mod1
        "b.mod.a": aparam,
        "b.a": a1,
    }
    assert_mod_dict(mod6, named_params_dict6)

def test_module_with_list():
    a = torch.tensor([1.])
    b = torch.tensor([2.])
    a2 = 2*a
    a3 = 3*a
    b2 = 2*b
    aparam = torch.nn.Parameter(a)
    bparam = torch.nn.Parameter(b)
    bparam2 = torch.nn.Parameter(b2)
    aparam2 = torch.nn.Parameter(a2)
    aparam3 = torch.nn.Parameter(a3)
    nnmod = NNModule(aparam)

    listmod1 = PlainModule([a, b], bparam)
    named_params_dict1 = {
        "a.0": a,
        "a.1": b,
        "b": bparam,
    }
    assert_mod_dict(listmod1, named_params_dict1)

    listmod2 = PlainModule([aparam, b], a)
    named_params_dict2 = {
        "a.0": aparam,
        "a.1": b,
    }
    assert_mod_dict(listmod2, named_params_dict2)

    listmod3 = PlainModule([aparam, bparam], aparam2)
    named_params_dict3 = {
        "a.0": aparam,
        "a.1": bparam,
        "b": aparam2,
    }
    assert_mod_dict(listmod3, named_params_dict3)

    listmod4 = PlainModule([aparam3, bparam2], listmod3)
    named_params_dict4 = {
        "a.0": aparam3,
        "a.1": bparam2,
        # from listmod3
        "b.a.0": aparam,
        "b.a.1": bparam,
        "b.b": aparam2
    }
    assert_mod_dict(listmod4, named_params_dict4)

def test_list_module_err():
    a = torch.tensor([1.])
    b = torch.tensor([2.])
    a2 = 2*a
    aparam = torch.nn.Parameter(a)
    aparam2 = torch.nn.Parameter(a2)
    bparam = torch.nn.Parameter(b)
    nnmod1 = NNModule(aparam)
    nnmod2 = NNModule(bparam)

    try:
        mod1 = PlainModule(a=[nnmod1, nnmod2], b=aparam2)
        assert False, "a TypeError must be raised if list of modules are registered"
    except TypeError:
        pass

    mod2 = PlainModule(a=torch.nn.ModuleList([nnmod1, nnmod2]), b=aparam2)
    named_params_dict2 = {
        "a.0.a": aparam,
        "a.1.a": bparam,
        "b": aparam2,
    }
    assert_mod_dict(mod2, named_params_dict2)

def assert_mod_dict(mod, dct2_orig):
    recurse_modes = [True, False]
    nnparam_only_modes = [True, False]
    for recurse, nnparam_only in itertools.product(recurse_modes, nnparam_only_modes):
        dct2 = select_dct(dct2_orig, recurse, nnparam_only)
        dct1 = dict(mod.named_parameters(recurse=recurse, nnparam_only=nnparam_only))
        assert len(dct1) == len(dct2)
        for k,v1 in dct1.items():
            v2 = dct2[k]
            assert v1 is v2

def select_dct(dct, recurse=True, nnparam_only=False):
    res = {}
    for key, val in dct.items():
        cond1 = recurse or "." not in key
        cond2 = not nnparam_only or isinstance(val, torch.nn.Parameter)
        if cond1 and cond2:
            res[key] = val
    return res
