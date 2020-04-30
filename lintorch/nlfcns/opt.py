import torch
import numpy as np
from lintorch.utils.misc import set_default_option
from lintorch.fcns.solve import solve
from lintorch.core.base import Module as LintorchModule
from lintorch.nlfcns.util import wrap_fcn

__all__ = ["optimize"]

def optimize(fcn, xparams, yparams=[], fwd_options={}, bck_options={}):
    """
    Find the set of parameters that minimize the output of the function `fcn`,

        min_x fcn(*x, *y)

    where `fcn` is the function with output of a tensor of (nbatch,)
    and input of a series of tensors, *x, and *y. *x is the optimized parameters
    while *y is the external parameters.
    The output of this block is the minimum value of `fcn` and the best
    parameters, `x`.
    The gradient is calculated with respect to `y`.
    """
    raise RuntimeError("This function is not ready. Please do not use it.")
    wrapped_fcn, all_params = wrap_fcn(fcn, (*xparams, *yparams))
    all_params = all_params[len(xparams):] # to exclude xparams
    res = _Optimize.apply(wrapped_fcn, xparams, fwd_options, bck_options, *all_params)
    fcn_min = res[0]
    xmin = res[1:]
    return fcn_min, xmin

class _Optimize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, x0, fwd_options, bck_options, *yparams):
        # set default options
        config = set_default_option({
            "max_niter": 100,
            "min_eps": 1e-12,
            "verbose": False,
            "lr": 1e-2,
            "method": "lbfgs",
        }, fwd_options)
        ctx.bck_options = bck_options

        verbose = config["verbose"]
        min_eps = config["min_eps"]

        # get the algorithm class
        method = config["method"].lower()
        if method == "lbfgs":
            opt_cls = torch.optim.LBFGS
            opt_kwds = ["lr"]
        elif method == "sgd":
            opt_cls = torch.optim.SGD
            opt_kwds = ["lr", "momentum", "dampening", "weight_decay", "nesterov"]

        opt_kwargs = {x:config[x] for x in opt_kwds if x in config}

        class Recorder:
            def __init__(self):
                self.previ = -1
                self.zprev = None
                self.dz = None

        with torch.enable_grad():
            x = [p.detach().clone().requires_grad_() for p in x0]
            y = [p.detach().clone() for p in yparams]
            params = x
            if len(params) == 0:
                raise RuntimeError("There is no parameters to be optimized in the OptimizationModule")

            opt = opt_cls(params, **opt_kwargs)
            recorder = Recorder()
            for i in range(config["max_niter"]):
                def closure():
                    opt.zero_grad()
                    z = fcn(*x, *y)
                    # NOTE: using backward here will make .grad attribute
                    # non-zero for differentiable parameters that are not in
                    # params, so we will have to reset it later
                    # TODO: implement algorithm without backward
                    z.sum().backward()

                    if i != recorder.previ:
                        if recorder.zprev is not None:
                            recorder.dz = (recorder.zprev - z).abs().max()
                        recorder.zprev = z
                        recorder.previ = i

                    if verbose and i%10 == 0 and recorder.dz is not None:
                        print("Iter %3d: %.3e dloss: %.3e" % (i, z, recorder.dz))
                    return z

                opt.step(closure)
                if recorder.dz is not None and recorder.dz < min_eps:
                    break

        # reset all the gradients
        opt.zero_grad()

        xopt = x
        zopt = fcn(*xopt, *yparams)
        res = (zopt, *xopt)

        # save for the backward calculation
        ctx.fcn = fcn
        ctx.xopt = xopt
        ctx.zopt = zopt
        ctx.yparams = yparams
        return res

    @staticmethod
    def backward(ctx, grad_zopt, *grad_xopt):
        import time
        t0 = time.time()

        yparams = ctx.yparams
        fcn = ctx.fcn
        xopt = ctx.xopt
        zopt = ctx.zopt

        # get the contribution from zopt
        with torch.enable_grad():
            yparams_copy = [y.clone() for y in yparams]
            f = ctx.fcn(*ctx.xopt, *yparams_copy)

        grad_yparams = torch.autograd.grad(f, yparams_copy, grad_outputs=grad_zopt,
            create_graph=torch.is_grad_enabled())

        # check if the grad_xopt is all zeros
        allzeros = True
        for gx in grad_xopt:
            if not torch.allclose(gx, gx*0):
                allzeros = False
                break
        # if grad_xopt is all zeros, don't bother to calculate the contribution
        # from grad_xopt
        if allzeros:
            return (None, None, None, None, *grad_yparams)

        # get the contribution from xopt
        unpacker = _Unpacker(ctx.xopt)
        # calculate (d2f/dx2)^(-1) * (grad_xopt)
        _apply_D2fDx2 = _D2fDx2(unpacker, fcn, len(ctx.xopt))
        gx = unpacker.pack(grad_xopt) # (nbatch, nr)
        gxfcn = solve(_apply_D2fDx2, [*xopt, *yparams], gx.unsqueeze(-1),
            fwd_options=ctx.bck_options, bck_options=ctx.bck_options).squeeze(-1) # (nbatch, nr)
        gx_unpack = unpacker.unpack(gxfcn) # list of tensors: (nbatch, ...)

        # calculate (d2f/dxdy)^(-1) * (gx)
        with torch.enable_grad():
            yparams_copy = [y.clone() for y in yparams]
            f = fcn(*xopt, *yparams_copy)
            dfdx = torch.autograd.grad(f, xopt, create_graph=True)
        grad_yparams2 = torch.autograd.grad(dfdx, yparams_copy, grad_outputs=gx_unpack)

        # sum the contribution from z and x
        gyparams = []
        for i in range(len(grad_yparams)):
            gyparams.append(grad_yparams[i] - grad_yparams2[i])
        return (None, None, None, None, *gyparams)

class _D2fDx2(LintorchModule):
    def __init__(self, unpacker, fcn, noptparams):
        nr = unpacker.get_size()
        super(_D2fDx2, self).__init__(shape=(nr,nr), is_symmetric=False)
        self.fcn = fcn
        self.unpacker = unpacker
        self.noptparams = noptparams

    def forward(self, gx, *params):#xopt, yparams):
        # gx: (nbatch, nr, 1)
        xopt = params[:self.noptparams]
        yparams = params[self.noptparams:]
        grad_xopt = self.unpacker.unpack(gx.squeeze(-1)) # list of tensors: (nbatch, ...)
        with torch.enable_grad():
            f = self.fcn(*xopt, *yparams)
            dfdx = torch.autograd.grad(f, xopt, create_graph=True)
        d2fdx2 = torch.autograd.grad(dfdx, xopt, grad_outputs=grad_xopt,
            retain_graph=True, create_graph=torch.is_grad_enabled()) # (list of tensors, (nbatch,...))
        d2fdx2_line = self.unpacker.pack(d2fdx2).unsqueeze(-1) # (nbatch, nr, 1)
        return d2fdx2_line

class _Unpacker(object):
    def __init__(self, xopt):
        self.nx = len(xopt)
        nr = 1
        shapes = []
        nrs = []
        for i in range(self.nx):
            shapes.append(xopt[i].shape)
            nr_elmt = np.prod(xopt[i].shape[1:]) if (xopt[i].ndim > 1) else 1
            nrs.append(nr_elmt)
            nr *= nr_elmt
        self.nr = nr
        self.shapes = shapes
        self.nrs = nrs

    def get_size(self):
        return self.nr

    def unpack(self, xline):
        nbatch, nr = xline.shape
        xvars = []
        inr = 0
        for i in range(self.nx):
            nr_elmt = self.nrs[i]
            xvars.append(xline[:,inr:inr+nr_elmt].clone().view(self.shapes[i]))
            inr += nr_elmt
        return xvars

    def pack(self, xvars):
        # xvars: list of (nbatch, ...)
        nbatch = xvars[0].shape[0]
        return torch.cat([x.view(nbatch, -1) for x in xvars], dim=-1) # (nbatch, nr)
