import torch
from torch import tensor
from torch.autograd import gradcheck, gradgradcheck
import xitorch.special as xsp
from xitorch._tests.utils import device_dtype_float_test

special_funcs_inouts = {
    "j0": [ # list of inputs and outputs
        [(tensor([1.5, -1.5]),), tensor([0.5118277072906494, 0.5118277072906494])],
    ],
    "y0": [
        [(tensor([1.5, 2.5]),), tensor([0.3824489712715149, 0.4980703592300415])],
    ],
}
special_funcs_fcnnames = list(special_funcs_inouts.keys())

def getfcn(fcnname):
    return xsp.__dict__[fcnname]

def _to(inps, dtype, device):
    if isinstance(inps, torch.Tensor):
        return inps.to(dtype).to(device)
    else:
        return tuple(inp.to(dtype).to(device) for inp in inps)

def _call(fcnname, inp):
    fcn = getfcn(fcnname)
    return fcn(*inp)

def special_func_skipif(dtype, device, fcnname, *unused):
    inp, out = special_funcs_inouts[fcnname][0]
    inp = _to(inp, dtype, device)
    try:
        _call(fcnname, inp)
    except NotImplementedError:
        return True, "No implementation of %s for %s and %s" % (fcnname, dtype, device) # skip
    return False, ""

def special_func_skipderiv(dtype, device, fcnname, *unused):
    skip_func, msg = special_func_skipif(dtype, device, fcnname, *unused)
    if skip_func:
        return True, msg
    inp, out = special_funcs_inouts[fcnname][0]
    inp = tuple(ip.requires_grad_() for ip in _to(inp, dtype, device))
    out = _call(fcnname, inp)
    if not isinstance(out, torch.Tensor):
        out = out[0]
    try:
        torch.autograd.grad(out.sum(), inp)
    # temporary RuntimeError (should be NotImplementedError) due to PyTorch's bug #46088
    # https://github.com/pytorch/pytorch/issues/46088
    except RuntimeError:
        return True, "Backward operation of %s is not implemented" % fcnname
    return False, ""

# test all special functions on all available dtype and device, skip if no
# implementation is available
def special_func_decor(skip_fcn=None, **kwargs):
    if skip_fcn is None:
        skip_fcn = special_func_skipif

    return device_dtype_float_test(
        additional_kwargs={
            "fcnname": special_funcs_fcnnames
        },
        skip_fcn=skip_fcn, **kwargs)

@special_func_decor()
def test_special_fcn(dtype, device, fcnname):
    inouts = special_funcs_inouts[fcnname]
    fcn = getfcn(fcnname)

    for inp, out in inouts:
        inp = _to(inp, dtype, device)
        out = _to(out, dtype, device)
        fout = fcn(*inp)

        if isinstance(fout, torch.Tensor):
            assert torch.allclose(fout, out)
        else:
            for out0, fout0 in zip(out, fout):
                assert torch.allclose(fout0, out0)

@special_func_decor()
def test_transpose(dtype, device, fcnname):
    inptemp, outtemp = special_funcs_inouts[fcnname][0]
    fcn = getfcn(fcnname)

    ninp = len(inptemp)
    nout = 1 if isinstance(outtemp, torch.Tensor) else len(outtemp)
    shape = (3,2)

    # calculate the original output
    inp0 = tuple(torch.rand(shape, dtype=dtype, device=device) for i in range(ninp))
    out0 = fcn(*inp0)
    assert tuple(out0.shape) == shape

    # set the transposed inp0
    inp1 = tuple(inp.transpose(-2,-1) for inp in inp0) # non-contiguous
    out1 = fcn(*inp1)
    if nout == 1:
        assert torch.allclose(out0.transpose(-2,-1), out1)
    else:
        for out,o1 in zip(out0, out1):
            assert torch.allclose(out.transpose(-2,-1), o1)

@special_func_decor(only64=True, skip_fcn=special_func_skipderiv)
def test_deriv(dtype, device, fcnname):
    inptemp, outtemp = special_funcs_inouts[fcnname][0]
    fcn = getfcn(fcnname)
    inps = _to(inptemp, dtype=dtype, device=device)
    inps = tuple(inp.clone().requires_grad_() for inp in inps)

    gradcheck(fcn, inps)
    gradgradcheck(fcn, inps)
