import random
import torch
import numpy as np
from torch.autograd import gradcheck, gradgradcheck
import lintorch as lt
from lintorch.grad.jachess import hess
from lintorch.integrate import quad, solve_ivp, mcquad
from lintorch._tests.utils import device_dtype_float_test

################################## quadrature ##################################
class IntegrationNNModule(torch.nn.Module):
    # cos(a*x + b * c)
    def __init__(self, a, b):
        super(IntegrationNNModule, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x, c):
        return torch.cos(self.a * x + self.b * c)

class IntegrationModule(lt.EditableModule):
    # cos(a*x + b * c)
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, x, c):
        return torch.cos(self.a * x + self.b * c)

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"a", prefix+"b"]

class IntegrationNNMultiModule(torch.nn.Module):
    # cos(a*x + b * c), sin(a*x + b*c)
    def __init__(self, a, b):
        super(IntegrationNNMultiModule, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x, c):
        return torch.cos(self.a * x + self.b * c), torch.sin(self.a * x + self.b * c)

class IntegrationMultiModule(lt.EditableModule):
    # cos(a*x + b * c), sin(a*x + b*c)
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, x, c):
        return torch.cos(self.a * x + self.b * c), torch.sin(self.a * x + self.b * c)

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"a", prefix+"b"]

class IntegrationInfModule(torch.nn.Module):
    def __init__(self, w):
        super(IntegrationInfModule, self).__init__()
        self.w = w

    def forward(self, x):
        return torch.exp(-x*x/(2*self.w*self.w))

@device_dtype_float_test(only64=True)
def test_quad(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    nr = 2
    fwd_options = {
        "method": "leggauss",
        "n": 100,
    }

    a = torch.nn.Parameter(torch.rand((nr,), dtype=dtype, device=device).requires_grad_())
    b = torch.nn.Parameter(torch.randn((nr,), dtype=dtype, device=device).requires_grad_())
    c = torch.randn((nr,), dtype=dtype, device=device).requires_grad_()
    xl = torch.zeros((1,), dtype=dtype, device=device).requires_grad_()
    xu = (torch.ones ((1,), dtype=dtype, device=device) * 0.5).requires_grad_()

    for clss in [IntegrationModule, IntegrationNNModule]:

        module = clss(a, b)
        y = quad(module.forward, xl, xu, params=(c,), fwd_options=fwd_options)
        ytrue = (torch.sin(a * xu + b * c) - torch.sin(a * xl + b * c)) / a
        assert torch.allclose(y, ytrue)

        def getloss(a, b, c, xl, xu):
            module = clss(a, b)
            y = quad(module.forward, xl, xu, params=(c,), fwd_options=fwd_options)
            return y

        gradcheck    (getloss, (a, b, c, xl, xu))
        gradgradcheck(getloss, (a, b, c, xl, xu))
        # check if not all parameters require grad
        gradcheck    (getloss, (a, b.detach(), c, xl, xu))

@device_dtype_float_test(only64=True)
def test_quad_multi(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    nr = 4
    fwd_options = {
        "method": "leggauss",
        "n": 100,
    }

    a = torch.nn.Parameter(torch.rand((nr,), dtype=dtype, device=device).requires_grad_())
    b = torch.nn.Parameter(torch.randn((nr,), dtype=dtype, device=device).requires_grad_())
    c = torch.randn((nr,), dtype=dtype, device=device).requires_grad_()
    xl = torch.zeros((1,), dtype=dtype, device=device).requires_grad_()
    xu = (torch.ones ((1,), dtype=dtype, device=device) * 0.5).requires_grad_()

    for clss in [IntegrationMultiModule, IntegrationNNMultiModule]:
        module = clss(a, b)
        y = quad(module.forward, xl, xu, params=(c,), fwd_options=fwd_options)
        ytrue0 = (torch.sin(a * xu + b * c) - torch.sin(a * xl + b * c)) / a
        ytrue1 = (-torch.cos(a * xu + b * c) + torch.cos(a * xl + b * c)) / a
        assert len(y) == 2
        assert torch.allclose(y[0], ytrue0)
        assert torch.allclose(y[1], ytrue1)

@device_dtype_float_test(only64=True)
def test_quad_inf(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    nr = 4
    fwd_options = {
        "method": "leggauss",
        "n": 100,
    }
    w = torch.nn.Parameter(torch.abs(torch.randn((nr,), dtype=dtype, device=device)).requires_grad_())
    i = 0
    for totensor in [True, False]:
        if totensor:
            xl = torch.tensor(-float("inf"), dtype=dtype, device=device)
            xu = torch.tensor( float("inf"), dtype=dtype, device=device)
        else:
            xl = -float("inf")
            xu =  float("inf")

        def get_loss(w):
            module = IntegrationInfModule(w)
            y = quad(module.forward, xl, xu, params=[], fwd_options=fwd_options)
            return y

        y = get_loss(w)
        ytrue = w * np.sqrt(2*np.pi)
        assert torch.allclose(y, ytrue)
        if i == 0:
            gradcheck(get_loss, (w,))
            gradgradcheck(get_loss, (w,))
        i += 1

################################## ivp ##################################
class IVPNNModule(torch.nn.Module):
    # dydt: -a * y * t - b * y - c * y
    def __init__(self, a, b):
        super(IVPNNModule, self).__init__()
        self.a = a
        self.b = b

    def forward(self, t, y, c):
        return -self.a * y * t - self.b * y - c * y

class IVPModule(lt.EditableModule):
    # dydt: -a * y * t - b * y - c * y
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self, t, y, c):
        return -self.a * y * t - self.b * y - c * y

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"a", prefix+"b"]

@device_dtype_float_test(only64=True)
def test_ivp(dtype, device):
    torch.manual_seed(100)
    random.seed(100)
    nr = 2
    nt = 5
    t0 = 0.0
    t1 = 0.2
    fwd_options = {
        "method": "rk4",
    }

    a = torch.nn.Parameter(torch.rand((nr,), dtype=dtype, device=device).requires_grad_())
    b = torch.nn.Parameter(torch.randn((nr,), dtype=dtype, device=device).requires_grad_())
    c = torch.randn((nr,), dtype=dtype, device=device).requires_grad_()
    ts = torch.linspace(t0, t1, nt, dtype=dtype, device=device).requires_grad_()
    y0 = torch.rand((nr,), dtype=dtype, device=device).requires_grad_()
    ts1 = ts.unsqueeze(-1)

    for clss in [IVPModule, IVPNNModule]:
        def getoutput(a, b, c, ts, y0):
            module = clss(a, b)
            yt = solve_ivp(module.forward, ts, y0, params=(c,), fwd_options=fwd_options)
            return yt

        yt = getoutput(a, b, c, ts, y0)
        yt_true = y0 * torch.exp(-(0.5 * a * (ts1 + t0) + b + c) * (ts1 - t0))
        assert torch.allclose(yt, yt_true)

        gradcheck(getoutput, (a, b, c, ts, y0))
        gradgradcheck(getoutput, (a, b, c, ts, y0))

################################## mcquad ##################################
class MCQuadLogProbNNModule(torch.nn.Module):
    def __init__(self, w):
        super(MCQuadLogProbNNModule, self).__init__()
        self.w = w

    def forward(self, x):
        # x, w are single-element tensors
        return -x*x/(2*self.w*self.w)

class MCQuadFcnModule(lt.EditableModule):
    def __init__(self, a):
        self.a = a

    def forward(self, x):
        # return self.a*self.a * x * x
        return torch.exp(-x*x/(2*self.a*self.a))

    def getparamnames(self, methodname, prefix=""):
        return [prefix+"a"]

def get_true_output(w, a):
    # return a*a*w*w
    return 1.0 / torch.sqrt(1 + w * w / (a * a))

@device_dtype_float_test(only64=True)
def test_mcquad(dtype, device):
    torch.manual_seed(100)
    random.seed(100)

    w = torch.nn.Parameter(torch.tensor(1.2, dtype=dtype, device=device))
    a = torch.tensor(0.3, dtype=dtype, device=device).requires_grad_()
    x0 = torch.tensor(0.0, dtype=dtype, device=device)
    fwd_options = {
        "method": "mh",
        "step_size": 0.6,
        "nsamples": 10000,
        "nburnout": 2,
    }

    def getoutput(w, a, x0):
        logp = MCQuadLogProbNNModule(w)
        fcn = MCQuadFcnModule(a)
        res = mcquad(fcn.forward, logp.forward, x0, fparams=[], pparams=[], fwd_options=fwd_options)
        return res

    rtol = 15e-2 # relatively large error is acceptable because of the stochasticity
    epf = getoutput(w, a, x0)
    epf_true = get_true_output(w, a)
    assert torch.allclose(epf, epf_true, rtol=rtol)

    # manually check the gradient
    g = torch.tensor(0.7, dtype=dtype, device=device).reshape(epf.shape).requires_grad_()
    ga     , gw      = torch.autograd.grad(epf     , (a, w), grad_outputs=g, create_graph=True)
    # different implementation
    ga2    , gw2     = torch.autograd.grad(epf     , (a, w), grad_outputs=g, retain_graph=True, create_graph=False)
    ga_true, gw_true = torch.autograd.grad(epf_true, (a, w), grad_outputs=g, create_graph=True)
    assert torch.allclose(gw, gw_true, rtol=rtol)
    assert torch.allclose(ga, ga_true, rtol=rtol)
    assert torch.allclose(gw2, gw_true, rtol=rtol)
    assert torch.allclose(ga2, ga_true, rtol=rtol)

    ggaw     , ggaa     , ggag      = torch.autograd.grad(ga     , (w, a, g), retain_graph=True, allow_unused=True)
    ggaw_true, ggaa_true, ggag_true = torch.autograd.grad(ga_true, (w, a, g), retain_graph=True, allow_unused=True)
    print("ggaw", ggaw, ggaw_true, (ggaw - ggaw_true) / ggaw_true)
    print("ggaa", ggaa, ggaa_true, (ggaa - ggaa_true) / ggaa_true)
    print("ggag", ggag, ggag_true, (ggag - ggag_true) / ggag_true)
    assert torch.allclose(ggaa, ggaa_true, rtol=rtol)
    assert torch.allclose(ggag, ggag_true, rtol=rtol)
    ggww     , ggwa     , ggwg      = torch.autograd.grad(gw     , (w, a, g), allow_unused=True)
    ggww_true, ggwa_true, ggwg_true = torch.autograd.grad(gw_true, (w, a, g), allow_unused=True)
    print("ggwa", ggwa, ggwa_true, (ggwa - ggwa_true) / ggwa_true)
    print("ggwg", ggwg, ggwg_true, (ggwg - ggwg_true) / ggwg_true)
    print("ggww", ggww, ggww_true, (ggww - ggww_true) / ggww_true)
    assert torch.allclose(ggwa, ggwa_true, rtol=rtol)
    assert torch.allclose(ggwg, ggwg_true, rtol=rtol)
    assert torch.allclose(ggww, ggww_true, rtol=rtol)

if __name__ == "__main__":
    # with torch.autograd.detect_anomaly():
    test_mcquad()
