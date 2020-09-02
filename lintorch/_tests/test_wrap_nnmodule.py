import torch
from lintorch._core.pure_function import wrap_fcn

class NNModule(torch.nn.Module):
    def __init__(self, a):
        super(NNModule, self).__init__()
        self.a = a # must be a param

    def forward(self, x):
        return self.a * x

class NNModule0(torch.nn.Module):
    def __init__(self):
        super(NNModule0, self).__init__()

    def forward(self, x):
        return 2 * x

dtype = torch.float64

def test_wrap_nnmodule():
    a = torch.nn.Parameter(torch.tensor([1.2], dtype=dtype))
    x = torch.tensor([1.4], dtype=dtype, requires_grad=True)

    modules = [NNModule(a), NNModule0()]
    nparams = [2, 1]

    for i in range(len(modules)):
        module = modules[i]
        fcn, params = wrap_fcn(module, (x,))
        assert len(params) == nparams[i]
        assert params[0] is x
        if len(params) == 2:
            assert params[1] is a
