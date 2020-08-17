import inspect
from typing import Callable, Iterable, Mapping, Any, Sequence
import torch
import numpy as np
import scipy.optimize
import lintorch as lt
from lintorch.utils.misc import set_default_option
from lintorch.maths.rootfinder import lbfgs, selfconsistent, broyden, diis, gradrca
from lintorch.funcs.solve import solve
from lintorch.funcs.jac import jac
from lintorch.core.linop import LinearOperator, checklinop
from lintorch.core.editable_module import EditableModule, wrap_fcn
from lintorch.utils.debugmodes import is_debug_enabled

__all__ = ["equilibrium", "rootfinder"]

def rootfinder(
        fcn:Callable[[torch.Tensor],torch.Tensor],
        y0:torch.Tensor,
        params:Sequence[Any]=[],
        fwd_options:Mapping[str,Any]={},
        bck_options:Mapping[str,Any]={}):
    """
    Solving the rootfinder equation of a given function,

        0 = fcn(y, *params)

    where `fcn` is a function that can be non-linear and produce output of shape
    `y`. The output of this block is `y` that produces the 0 as the output.

    Arguments
    ---------
    * fcn: callable with output tensor (*ny)
        The function
    * y0: torch.tensor with shape (*ny)
        Initial guess of the solution
    * params: list
        List of any other parameters to be put in fcn
    * fwd_options: dict
        Options for the rootfinder method
    * bck_options: dict
        Options for the backward solve method

    Returns
    -------
    * yout: torch.tensor with shape (*ny)
        The solution which satisfies 0 = fcn(yout)

    Note
    ----
    * To obtain the correct gradient and higher order gradients, the fcn must be:
        - a torch.nn.Module with fcn.parameters() list the tensors that determine
            the output of the fcn.
        - a method in lt.EditableModule object with no out-of-scope parameters.
        - a function with no out-of-scope parameters.
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        if inspect.ismethod(fcn) and isinstance(fcn.__self__, EditableModule):
            fcn.__self__.assertparams(fcn, y0, *params)

    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0
    return _RootFinder.apply(wrapped_fcn, y0, fwd_options, bck_options, *all_params)#, *model_params)

def equilibrium(
        fcn:Callable[[torch.Tensor],torch.Tensor],
        y0:torch.Tensor,
        params:Sequence[Any]=[],
        fwd_options:Mapping[str,Any]={},
        bck_options:Mapping[str,Any]={}):
    """
    Solving the equilibrium equation of a given function,

        y = fcn(y, *params)

    where `fcn` is a function that can be non-linear and produce output of shape
    `y`. The output of this block is `y` that produces the 0 as the output.

    Arguments
    ---------
    * fcn: callable with output tensor (*ny)
        The function
    * y0: torch.tensor with shape (*ny)
        Initial guess of the solution
    * params: list
        List of any other parameters to be put in fcn
    * fwd_options: dict
        Options for the rootfinder method
    * bck_options: dict
        Options for the backward solve method

    Returns
    -------
    * yout: torch.tensor with shape (*ny)
        The solution which satisfies yout = fcn(yout)

    Note
    ----
    * To obtain the correct gradient and higher order gradients, the fcn must be:
        - a torch.nn.Module with fcn.parameters() list the tensors that determine
            the output of the fcn.
        - a method in lt.EditableModule object with no out-of-scope parameters.
        - a function with no out-of-scope parameters.
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        if inspect.ismethod(fcn) and isinstance(fcn.__self__, EditableModule):
            fcn.__self__.assertparams(fcn, y0, *params)

    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0
    def new_fcn(y, *params):
        return y - wrapped_fcn(y, *params)

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, *all_params)#

class _RootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn:Callable[[torch.Tensor],torch.Tensor],
            y0:torch.Tensor,
            options:Mapping[str,Any],
            bck_options:Mapping[str,Any], *params) -> torch.Tensor:
        # set default options
        config = set_default_option({
            "min_eps": 1e-9,
            "verbose": False,
            "linesearch": True,
            "jinv0": 0.5,
            "method": "lbfgs",
        }, options)
        ctx.bck_options = set_default_option({
            "method": "exactsolve"
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
    def backward(ctx, grad_yout:torch.Tensor):
        yout = ctx.yout # (*ny)
        params = ctx.params
        # dL/df
        jac_dfdy = jac(ctx.fcn, params=(yout, *params), idxs=[0])[0]
        gyfcn = solve(A=jac_dfdy.H, B=-grad_yout.unsqueeze(-1),
            fwd_options=ctx.bck_options, bck_options=ctx.bck_options).squeeze(-1)

        # get the grad for the params
        with torch.enable_grad():
            params_copy = [p.clone().requires_grad_() for p in params]
            yfcn = ctx.fcn(yout, *params_copy)
        grad_params = torch.autograd.grad(yfcn, params_copy, grad_outputs=gyfcn,
            create_graph=torch.is_grad_enabled())

        return (None, None, None, None, *grad_params)
