import torch
from torch.autograd import gradcheck, gradgradcheck
from xitorch.grad.jachess import jac, hess
from xitorch._core.editable_module import EditableModule
from xitorch._core.pure_function import get_pure_function

dtype = torch.float64

def func1(A, b, x0):
    x = torch.matmul(A, x0) + b
    x = torch.nn.Softplus()(x)
    return x

def hfunc1(A, b, x0):
    x = torch.matmul(A, x0) + b
    x = torch.nn.Softplus()(x)
    return x.sum()

def scalar_func(A, b, x0):
    x = A * x0 + b
    x = torch.nn.Softplus()(x)
    return x

class func2(EditableModule):
    def __init__(self, b):
        self.b = b

    def getparamnames(self, methodname, prefix=""):
        if methodname == "__call__":
            return [prefix + "b"]
        else:
            raise KeyError("Params for method %s cannot be found" % methodname)

    def __call__(self, A, b, x0):
        x = torch.matmul(A * self.b, x0) + b
        x = torch.nn.Softplus()(x)
        return x

class hfunc2(EditableModule):
    def __init__(self, b):
        self.b = b

    def getparamnames(self, methodname, prefix=""):
        if methodname == "__call__":
            return [prefix + "b"]
        else:
            raise KeyError("Params for method %s cannot be found" % methodname)

    def __call__(self, A, b, x0):
        x = torch.matmul(A * self.b, x0) + b
        x = torch.nn.Softplus()(x)
        return x.sum()

def getfnscalarparams():
    A = torch.rand((), dtype=dtype, requires_grad=True)
    b = torch.rand((), dtype=dtype, requires_grad=True)
    x = torch.rand((), dtype=dtype, requires_grad=True)
    return (A, b, x)

def getfnparams(na):
    A = torch.rand((na, na), dtype=dtype, requires_grad=True)
    b = torch.rand((na, 1), dtype=dtype, requires_grad=True)
    x = torch.rand((na, 1), dtype=dtype, requires_grad=True)
    return (A, b, x)

def getnnparams(na):
    b = torch.rand((na, 1), dtype=dtype, requires_grad=True)
    return (b,)

def test_jac_func():
    na = 3
    params = getfnparams(na)
    nnparams = getnnparams(na)
    nparams = len(params)
    all_idxs = [None, (0,), (1,), (0, 1), (0, 1, 2)]
    funcs = [func1, func2(*nnparams)]

    for func in funcs:
        for idxs in all_idxs:
            if idxs is None:
                gradparams = params
            else:
                gradparams = [params[i] for i in idxs]

            jacs = jac(func, params, idxs=idxs)
            assert len(jacs) == len(gradparams)

            y = func(*params)
            nout = torch.numel(y)
            nins = [torch.numel(p) for p in gradparams]
            v = torch.rand_like(y).requires_grad_()
            for i in range(len(jacs)):
                assert list(jacs[i].shape) == [nout, nins[i]]

            # get rmv
            jacs_rmv = torch.autograd.grad(y, gradparams, grad_outputs=v, create_graph=True)
            # the jac LinearOperator has shape of (nout, nin), so we need to flatten v
            jacs_rmv0 = [jc.rmv(v.view(-1)) for jc in jacs]

            # calculate the mv
            w = [torch.rand_like(p) for p in gradparams]
            jacs_lmv = [torch.autograd.grad(jacs_rmv[i], (v,), grad_outputs=w[i], retain_graph=True)[
                0] for i in range(len(jacs))]
            jacs_lmv0 = [jacs[i].mv(w[i].view(-1)) for i in range(len(jacs))]

            for i in range(len(jacs)):
                assert torch.allclose(jacs_rmv[i].view(-1), jacs_rmv0[i].view(-1))
                assert torch.allclose(jacs_lmv[i].view(-1), jacs_lmv0[i].view(-1))

def test_jac_grad():
    na = 3
    params = getfnparams(na)
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
        fmv = get_pure_function(jacs[i].rmv)
        params0 = v.view(-1)
        params1 = fmv.objparams()
        params12 = [p1 * p2 for p1, p2 in zip(params1, params2)]
        with fmv.useobjparams(params12):
            return fmv(params0)

    def fcnl2(i, w, *params2):
        fmv = get_pure_function(jacs[i].mv)
        params0 = w.view(-1)
        params1 = fmv.objparams()
        params12 = [p1 * p2 for (p1, p2) in zip(params1, params2)]
        with fmv.useobjparams(params12):
            return fmv(params0)

    v = torch.rand((na,), dtype=dtype, requires_grad=True)
    w = [torch.rand_like(p).requires_grad_() for p in params]
    for i in range(len(jacs)):
        gradcheck(fcnr, (i, v, *params))
        gradgradcheck(fcnr, (i, v, *params))
        gradcheck(fcnl, (i, w[i], *params))
        gradgradcheck(fcnl, (i, w[i], *params))
        gradcheck(fcnr2, (i, v, *params2))
        gradgradcheck(fcnr2, (i, v, *params2))
        gradcheck(fcnl2, (i, w[i], *params2))
        gradgradcheck(fcnl2, (i, w[i], *params2))

def test_jac_method_grad():
    na = 3
    params = getfnparams(na)
    nnparams = getnnparams(na)
    num_nnparams = len(nnparams)
    jacs = jac(func2(*nnparams), params)
    nout = jacs[0].shape[-2]

    def fcnr(i, v, *allparams):
        nnparams = allparams[:num_nnparams]
        params = allparams[num_nnparams:]
        jacs = jac(func2(*nnparams), params)
        return jacs[i].rmv(v.view(-1))

    def fcnl(i, v, *allparams):
        nnparams = allparams[:num_nnparams]
        params = allparams[num_nnparams:]
        jacs = jac(func2(*nnparams), params)
        return jacs[i].mv(v.view(-1))

    v = torch.rand((na,), dtype=dtype, requires_grad=True)
    w = [torch.rand_like(p).requires_grad_() for p in params]
    for i in range(len(jacs)):
        gradcheck(fcnr, (i, v, *nnparams, *params))
        gradgradcheck(fcnr, (i, v, *nnparams, *params))
        gradcheck(fcnl, (i, w[i], *nnparams, *params))
        gradgradcheck(fcnl, (i, w[i], *nnparams, *params))

def test_jac_scalar_func():
    params = getfnscalarparams()
    nparams = len(params)
    all_idxs = [None, (0,), (1,), (0, 1), (0, 1, 2)]
    func = scalar_func

    for idxs in all_idxs:
        if idxs is None:
            gradparams = params
        else:
            gradparams = [params[i] for i in idxs]

        jacs = jac(func, params, idxs=idxs)
        assert len(jacs) == len(gradparams)

        y = func(*params)
        nout = torch.numel(y)
        nins = [torch.numel(p) for p in gradparams]
        v = torch.rand_like(y).requires_grad_()
        for i in range(len(jacs)):
            assert list(jacs[i].shape) == [nout, nins[i]]

        # get rmv
        jacs_rmv = torch.autograd.grad(y, gradparams, grad_outputs=v, create_graph=True)
        # the jac LinearOperator has shape of (nout, nin), so we need to flatten v
        jacs_rmv0 = [jc.rmv(v.view(-1)) for jc in jacs]

        # calculate the mv
        w = [torch.rand_like(p) for p in gradparams]
        jacs_lmv = [torch.autograd.grad(jacs_rmv[i], (v,), grad_outputs=w[i], retain_graph=True)[0]
                    for i in range(len(jacs))]
        jacs_lmv0 = [jacs[i].mv(w[i].view(-1)) for i in range(len(jacs))]

        for i in range(len(jacs)):
            assert torch.allclose(jacs_rmv[i].view(-1), jacs_rmv0[i].view(-1))
            assert torch.allclose(jacs_lmv[i].view(-1), jacs_lmv0[i].view(-1))

def test_hess_func():
    na = 3
    params = getfnparams(na)
    nnparams = getnnparams(na)
    nparams = len(params)
    all_idxs = [None, (0,), (1,), (0, 1), (0, 1, 2)]
    funcs = [hfunc1, hfunc2(*nnparams)]

    for func in funcs:
        for idxs in all_idxs:
            if idxs is None:
                gradparams = params
            else:
                gradparams = [params[i] for i in idxs]

            hs = hess(func, params, idxs=idxs)
            assert len(hs) == len(gradparams)

            y = func(*params)
            nins = [torch.numel(p) for p in gradparams]
            w = [torch.rand_like(p) for p in gradparams]
            for i in range(len(hs)):
                assert list(hs[i].shape) == [nins[i], nins[i]]

            # assert the values
            dfdy = torch.autograd.grad(y, gradparams, create_graph=True)
            hs_mv_man = [torch.autograd.grad(dfdy[i], (gradparams[i],), grad_outputs=w[i],
                                             retain_graph=True)[0] for i in range(len(dfdy))]
            hs_mv = [hs[i].mv(w[i].reshape(-1, nins[i])) for i in range(len(dfdy))]
            for i in range(len(dfdy)):
                assert torch.allclose(hs_mv[i].view(-1), hs_mv_man[i].view(-1))

def test_hess_grad():
    na = 3
    params = getfnparams(na)
    params2 = [torch.rand(1, dtype=dtype).requires_grad_() for p in params]
    hs = hess(hfunc1, params)

    def fcnl(i, v, *params):
        hs = hess(hfunc1, params)
        return hs[i].mv(v.view(-1))

    def fcnl2(i, v, *params2):
        fmv = get_pure_function(hs[i].mv)
        params0 = v.view(-1)
        params1 = fmv.objparams()
        params12 = [p1 * p2 for (p1, p2) in zip(params1, params2)]
        with fmv.useobjparams(params12):
            return fmv(params0)

    w = [torch.rand_like(p).requires_grad_() for p in params]
    for i in range(len(hs)):
        gradcheck(fcnl, (i, w[i], *params))
        gradgradcheck(fcnl, (i, w[i], *params))
        gradcheck(fcnl2, (i, w[i], *params2))
        gradgradcheck(fcnl2, (i, w[i], *params2))
