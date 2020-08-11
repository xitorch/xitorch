import torch
from torch.autograd import gradcheck, gradgradcheck
from lintorch.funcs.jac import jac
from lintorch.core.editable_module import wrap_fcn

dtype = torch.float64

def func1(A, x0, b):
    x = torch.matmul(A, x0) + b
    x = torch.nn.Softplus()(x)
    return x

def getnnparams(na):
    A = torch.rand((na,na), dtype=dtype, requires_grad=True)
    x = torch.rand((na,1 ), dtype=dtype, requires_grad=True)
    b = torch.rand((na,1 ), dtype=dtype, requires_grad=True)
    return (A, x, b)

def test_jac_func():
    na = 3
    params = getnnparams(na)
    jacs = jac(func1, params)
    assert len(jacs) == len(params)

    y = func1(*params)
    nout = torch.numel(y)
    nins = [torch.numel(p) for p in params]
    v = torch.rand_like(y).requires_grad_()
    for i in range(len(jacs)):
        assert list(jacs[i].shape) == [nout, nins[i]]

    # get rmv
    jacs_rmv = torch.autograd.grad(y, params, grad_outputs=v, create_graph=True)
    # the jac LinearOperator has shape of (nout, nin), so we need to flatten v
    jacs_rmv0 = [jc.rmv(v.view(-1)) for jc in jacs]

    # calculate the mv
    w = [torch.rand_like(p) for p in params]
    jacs_lmv = [torch.autograd.grad(jacs_rmv[i], (v,), grad_outputs=w[i], retain_graph=True)[0] for i in range(len(jacs))]
    jacs_lmv0 = [jacs[i].mv(w[i].view(-1)) for i in range(len(jacs))]

    for i in range(len(jacs)):
        assert torch.allclose(jacs_rmv[i].view(-1), jacs_rmv0[i].view(-1))
        assert torch.allclose(jacs_lmv[i].view(-1), jacs_lmv0[i].view(-1))

def test_jac_grad():
    na = 3
    params = getnnparams(na)
    params2 = [torch.rand(1, dtype=dtype).requires_grad_() for p in params]
    jacs = jac(func1, params)
    nout = jacs[0].shape[-2]

    def fcnr(i, v, *params):
        jacs = jac(func1, params)
        return jacs[i].rmv(v.view(-1))

    def fcnl(i, w, *params):
        jacs = jac(func1, params)
        return jacs[i].mv(w.view(-1))

    def fcnr2(i, v, *params2):
        fmv, vparams = wrap_fcn(jacs[i].rmv, (v.view(-1),))
        params1 = vparams[1:]
        params12 = [p1*p2 for (p1,p2) in zip(params1, params2)]
        return fmv(vparams[0], *params12)

    def fcnl2(i, w, *params2):
        fmv, vparams = wrap_fcn(jacs[i].mv, (w.view(-1),))
        params1 = vparams[1:]
        params12 = [p1*p2 for (p1,p2) in zip(params1, params2)]
        return fmv(vparams[0], *params12)

    v = torch.rand((na,), dtype=dtype, requires_grad=True)
    w = [torch.rand_like(p).requires_grad_() for p in params]
    for i in range(len(jacs)):
        gradcheck    (fcnr, (i, v, *params))
        gradgradcheck(fcnr, (i, v, *params))
        gradcheck    (fcnl, (i, w[i], *params))
        gradgradcheck(fcnl, (i, w[i], *params))
        gradcheck    (fcnr2, (i, v, *params2))
        gradgradcheck(fcnr2, (i, v, *params2))
        gradcheck    (fcnl2, (i, w[i], *params2))
        gradgradcheck(fcnl2, (i, w[i], *params2))
