import torch
import lintorch as lt
from lintorch.utils.misc import set_default_option
from lintorch.maths.rootfinder import lbfgs, selfconsistent, broyden, diis, gradrca
from lintorch.fcns.solve import solve
from lintorch.core.base import Module as LintorchModule

__all__ = ["equilibrium"]

def equilibrium(fcn, y0, params, fwd_options={}, bck_options={}):
    """
    Solving nonlinear equation to solve the equation

        y = fcn(y, *params)

    where `fcn` is a function that can be non-linear and produce output of shape
    of `y`.
    """
    return _Equilibrium.apply(fcn, y0, fwd_options, bck_options, *params)

class _Equilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, y0, options, bck_options, *params):
        # set default options
        config = set_default_option({
            "max_niter": 50,
            "min_eps": 1e-9,
            "verbose": False,
            "linesearch": True,
            "jinv0": 0.5,
            "method": "lbfgs",
        }, options)
        ctx.bck_options = set_default_option({
            "max_niter": 50,
            "min_eps": 1e-9,
            "verbose": False,
        }, bck_options)

        def loss(y):
            yfcn = fcn(y, *params)
            return y - yfcn

        jinv0 = config["jinv0"]
        method = config["method"].lower()
        if method == "lbfgs":
            y = lbfgs(loss, y0, **config)
        elif method == "selfconsistent":
            y = selfconsistent(loss, y0, **config)
        elif method == "broyden":
            y = broyden(loss, y0, **config)
        elif method == "diis":
            y = diis(loss, y0, **config)
        elif method == "gradrca":
            y = gradrca(loss, y0, **config)
        else:
            raise RuntimeError("Unknown method: %s" % config["method"])

        ctx.fcn = fcn
        ctx.params = params
        ctx.yout = y
        return y

    @staticmethod
    def backward(ctx, grad_yout):
        yout = ctx.yout
        params = ctx.params
        nr = yout.shape[-1]
        # dL/df
        _apply_ImDfDy = _ImDfDy(nr, ctx.fcn)
        gyfcn = solve(_apply_ImDfDy, [yout, *params], grad_yout.unsqueeze(-1),
            fwd_options=ctx.bck_options, bck_options=ctx.bck_options).squeeze(-1)

        # get the grad for the params
        with torch.enable_grad():
            params_copy = [p.clone() for p in params]
            yfcn = ctx.fcn(yout, *params_copy)
        grad_params = torch.autograd.grad(yfcn, params_copy, grad_outputs=gyfcn,
            create_graph=torch.is_grad_enabled())

        return (None, None, None, None, *grad_params)

class _ImDfDy(LintorchModule):
    def __init__(self, nr, fcn):
        super(_ImDfDy, self).__init__(shape=(nr,nr), is_symmetric=False)
        self.fcn = fcn
        self.yinp = None
        self.yfcn = None

    def forward(self, gy, yfcn, *params):
        gy = gy.squeeze(-1)
        if torch.is_grad_enabled():
            yfcn.requires_grad_()
            with torch.enable_grad():
                yout = self.fcn(yfcn, *params)
            dfdy, = torch.autograd.grad(yout, (yfcn,), grad_outputs=gy,
                create_graph=torch.is_grad_enabled())
        else:
            # only evaluate once to save running time
            if self.yfcn is None:
                with torch.enable_grad():
                    self.yinp = yfcn.clone().detach().requires_grad_()
                    self.yfcn = self.fcn(self.yinp, *params)
            dfdy, = torch.autograd.grad(self.yfcn, (self.yinp,), gy,
                retain_graph=True, create_graph=torch.is_grad_enabled())

        res = gy - dfdy
        res = res.unsqueeze(-1)
        return res
