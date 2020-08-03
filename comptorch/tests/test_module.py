import torch
from comptorch.core.module import Module

class NNModule(torch.nn.Module):
    def __init__(self, a):
        super(NNModule, self).__init__()
        if isinstance(a, torch.nn.Parameter):
            self.a = a
        else:
            self.a = torch.nn.Parameter(a)

class PlainModule(Module):
    def __init__(self, a, b):
        super(PlainModule, self).__init__()
        self.a = self.register(a)
        self.b = b

class ModuleWithList(Module):
    def __init__(self, a, b):
        super(ModuleWithList, self).__init__()
        self.ab = self.register([a, b])

class NestedModule(Module):
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
    aparam = torch.nn.Parameter(a)
    bparam = torch.nn.Parameter(b)
    nnmod = NNModule(aparam)
    plmod1 = PlainModule(a, b)
    plmod2 = PlainModule(a, bparam)

    mod1 = NestedModule(nnmod, a, b)
    named_params_dict1 = {
        "mod.a": aparam,
        "a": a,
    }
    assert_mod_dict(mod1, named_params_dict1)

    mod2 = NestedModule(plmod1, a, b)
    named_params_dict2 = {
        "mod.a": a,
        "a": a
    }
    assert_mod_dict(mod2, named_params_dict2)

    mod3 = NestedModule(plmod2, a, b)
    named_params_dict3 = {
        "mod.a": a,
        "mod.b": bparam,
        "a": a
    }
    assert_mod_dict(mod3, named_params_dict3)

    mod4 = NestedModule(mod3, a, b)
    named_params_dict4 = {
        "mod.mod.a": a,
        "mod.mod.b": bparam,
        "mod.a": a,
        "a": a
    }
    assert_mod_dict(mod4, named_params_dict4)

def assert_mod_dict(mod, dct2_orig):
    recurse_modes = [True, False]
    nnparam_only_modes = [True, False]
    for recurse in recurse_modes:
        for nnparam_only in nnparam_only_modes:
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
