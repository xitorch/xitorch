import torch
from comptorch.core.module import Module

class NNModule(torch.nn.Module):
    def __init__(self, a):
        super(NNModule, self).__init__()
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
    def __init__(self, amod, at, a):
        super(NestedModule, self).__init__()
        self.mod = amod
        self.modt = at
        self.a = self.register(a)

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
    assert_tensor_dict(dict(mod1.named_fullparams()), named_params_dict1, assert_obj=True)

    mod2 = PlainModule(aparam, b)
    named_params_dict2 = {
        "a": aparam
    }
    assert_tensor_dict(dict(mod2.named_fullparams()), named_params_dict2, assert_obj=True)

    mod3 = PlainModule(a, bparam)
    named_params_dict3 = {
        "a": a,
        "b": bparam # b should be registered as well because it is a parameter
    }
    assert_tensor_dict(dict(mod3.named_fullparams()), named_params_dict3, assert_obj=True)

    mod4 = PlainModule(aparam, bparam)
    named_params_dict4 = {
        "a": aparam,
        "b": bparam
    }
    assert_tensor_dict(dict(mod4.named_fullparams()), named_params_dict4, assert_obj=True)

def assert_tensor_dict(dct1, dct2, assert_obj=True):
    assert len(dct1) == len(dct2)
    for k,v1 in dct1.items():
        v2 = dct2[k]
        if assert_obj:
            assert v1 is v2
        else:
            assert torch.allclose(v1, v2)
