import torch
from xitorch._core.pure_function import get_pure_function

class NNModule(torch.nn.Module):
    def __init__(self, a):
        super(NNModule, self).__init__()
        self.a = a  # must be a param

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
        pfcn = get_pure_function(module)
        objparams = pfcn.objparams()
        assert len(objparams) == nparams[i] - 1  # 1 for the parameter x
        if len(objparams) == 1:
            assert objparams[0] is a
