import inspect
import torch
import scipy.optimize
import lintorch as lt
from lintorch.utils.misc import set_default_option
from lintorch.maths.rootfinder import lbfgs, selfconsistent, broyden, diis, gradrca
from lintorch.fcns.solve import solve
from lintorch.core.base import Module as LintorchModule
from lintorch.core.editable_module import wrap_fcn
from lintorch.funcs.utils.fcncheck import assertfcn
from lintorch.utils.decorators import deprecated

__all__ = ["equilibrium", "rootfinder"]

@deprecated
def rootfinder(fcn, y0, params=[], fwd_options={}, bck_options={}):
    """
    Solving the rootfinder equation of a given function,

        0 = fcn(y, *params)

    where `fcn` is a function that can be non-linear and produce output of shape
    `y`. The output of this block is `y` that produces the 0 as the output.
    """
    assertfcn(fcn)
    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0
    return _RootFinder.apply(wrapped_fcn, y0, fwd_options, bck_options, *all_params)#, *model_params)

@deprecated
def equilibrium(fcn, y0, params=[], fwd_options={}, bck_options={}):
    """
    Solving nonlinear equation to solve the equation

        y = fcn(y, *params)

    where `fcn` is a function that can be non-linear and produce output of shape
    of `y`.
    """
    assertfcn(fcn)
    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0
    def new_fcn(y, *params):
        return y - wrapped_fcn(y, *params)

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, *all_params)#

class _RootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, y0, options, bck_options, *params):
        # set default options
        config = set_default_option({
            "min_eps": 1e-9,
            "verbose": False,
            "linesearch": True,
            "jinv0": 0.5,
            "method": "lbfgs",
        }, options)
        ctx.bck_options = set_default_option({
            "method": "gmres"
        }, bck_options)

        def loss(y):
            yfcn = fcn(y, *params)
            return yfcn

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
        elif method.startswith("np_"):
            nbatch = y0.shape[0]

            def loss_np(y):
                yt = torch.tensor(y, dtype=y0.dtype, device=y0.device).view(nbatch,-1)
                yfcn = fcn(yt, *params)
                return yfcn.reshape(-1).cpu().detach().numpy()

            y0_np = y0.squeeze(0).cpu().detach().numpy()
            if method == "np_broyden1":
                opt = set_default_option({
                    "verbose": False,
                    "max_niter": None,
                    "min_eps": None,
                    "linesearch": "armijo",
                }, config)
                y_np = scipy.optimize.broyden1(loss_np, y0_np,
                    verbose=opt["verbose"],
                    maxiter=opt["max_niter"],
                    alpha=-config["jinv0"],
                    f_tol=opt["min_eps"],
                    line_search=opt["linesearch"])
            elif method == "np_fsolve":
                opt = set_default_option({
                    "max_niter": 0,
                }, config)
                y_np, info, ier, msg = scipy.optimize.fsolve(loss_np, y0_np,
                    maxfev=opt["max_niter"])
                if ier != 1:
                    warnings.warn(msg)
            else:
                raise RuntimeError("Unknown method: %s" % config["method"])

            y = torch.tensor(y_np, dtype=y0.dtype, device=y0.device).unsqueeze(0)
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
        _apply_DfDy = _DfDy(nr, ctx.fcn, dtype=yout.dtype, device=yout.device)
        gyfcn = solve(_apply_DfDy, [yout, *params], grad_yout.unsqueeze(-1),
            fwd_options=ctx.bck_options, bck_options=ctx.bck_options).squeeze(-1)

        # get the grad for the params
        with torch.enable_grad():
            params_copy = [p.clone().requires_grad_() for p in params]
            yfcn = ctx.fcn(yout, *params_copy)
        grad_params = torch.autograd.grad(yfcn, params_copy, grad_outputs=gyfcn,
            create_graph=torch.is_grad_enabled())

        return (None, None, None, None, *grad_params)

class _DfDy(LintorchModule):
    def __init__(self, nr, fcn, dtype=None, device=None):
        super(_DfDy, self).__init__(shape=(nr,nr), is_symmetric=False,
            dtype=dtype, device=device)
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

        # connect the params to dfdy, in case there is one or more missing
        if torch.is_grad_enabled():
            dfdy = connect_graph(dfdy, params)

        res = -dfdy.unsqueeze(-1)
        return res

    def transpose(self, gy, yfcn, *params):
        gy = gy.squeeze(-1)
        yfcn.requires_grad_()
        v = torch.ones_like(gy).to(gy.device).requires_grad_()
        with torch.enable_grad():
            yout = self.fcn(yfcn, *params)
            dfdy, = torch.autograd.grad(yout, (yfcn,), grad_outputs=v,
                create_graph=True)
        dfdyt, = torch.autograd.grad(dfdy, v, grad_outputs=gy,
            create_graph=torch.is_grad_enabled())

        # connect the params to dfdy, in case there is one or more missing
        if torch.is_grad_enabled():
            dfdyt = connect_graph(dfdyt, params)

        res = -dfdyt.unsqueeze(-1)
        return res

    def getparams(self, methodname):
        return []

    def setparams(self, methodname, *params):
        return 0

def connect_graph(out, params):
    # just to have a dummy graph, in case there is a parameter that
    # is disconnected in calculating df/dy
    return out + sum([p.view(-1)[0]*0 for p in params])
