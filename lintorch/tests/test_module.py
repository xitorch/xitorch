import torch
import lintorch as lt
from lintorch.tests.utils import device_dtype_float_test

@device_dtype_float_test()
def test_decor(dtype, device):
    na = 25

    @lt.module(shape=(na,na))
    def A(x, diag):
        return x * diag

    # check the properties
    assert isinstance(A, torch.nn.Module)
    assert A.is_symmetric
    assert A.is_real
    assert A.shape == (na, na)
    assert A.is_forward_set()
    assert not A.is_precond_set()

    @A.set_precond
    def precond(x, diag, biases=None, M=None, mparams=None):
        return x / diag

    assert A.is_precond_set()
    assert A.is_transpose_set()

    A = A.to(dtype).to(device)
    x = torch.ones(1,na,1).to(dtype).to(device)
    diag = (torch.arange(na)+1.0).unsqueeze(0).unsqueeze(-1).to(dtype).to(device)
    y = A(x, diag)
    x0 = A.precond(y, diag)
    y0 = A(x, diag)

    assert torch.allclose(y, x * diag)
    assert torch.allclose(x, x0)
    assert torch.allclose(y, y0)

@device_dtype_float_test()
def test_class(dtype, device):
    na = 25
    class Acls(lt.Module):
        def __init__(self):
            super(Acls, self).__init__(shape=(na, na))

        def forward(self, x, diag):
            return x * diag

        def precond(self, y, diag, biases=None, M=None, mparams=None):
            return y / diag

    # check the properties
    A = Acls()
    A = A.to(dtype).to(device)
    assert isinstance(A, torch.nn.Module)
    assert A.is_symmetric
    assert A.is_real
    assert A.shape == (na, na)
    assert A.is_forward_set()
    assert A.is_transpose_set()
    assert A.is_precond_set()

    x = torch.ones(1,na,1).to(dtype).to(device)
    diag = (torch.arange(na)+1.0).unsqueeze(0).unsqueeze(-1).to(dtype).to(device)
    y = A(x, diag)
    x0 = A.precond(y, diag)
    y0 = A.transpose(x, diag)
    assert torch.allclose(y, x * diag)
    assert torch.allclose(x, x0)
    assert torch.allclose(y, y0)
