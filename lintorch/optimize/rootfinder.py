import inspect
from typing import Callable, Iterable, Mapping, Any, Sequence
import torch
import numpy as np
import scipy.optimize
import lintorch as lt
from lintorch._utils.misc import set_default_option, TensorNonTensorSeparator
from lintorch._utils.assertfuncs import assert_fcn_params
from lintorch._impls.optimize.rootfinder import lbfgs, selfconsistent, broyden, diis, gradrca
from lintorch.linalg.solve import solve
from lintorch.grad.jachess import jac
from lintorch.linalg.linop import LinearOperator, checklinop
from lintorch._core.editable_module import EditableModule
from lintorch._core.pure_function import wrap_fcn
from lintorch.debug.modes import is_debug_enabled

__all__ = ["equilibrium", "rootfinder", "minimize"]

def rootfinder(
        fcn:Callable[...,torch.Tensor],
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
        assert_fcn_params(fcn, (y0, *params))

    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0
    return _RootFinder.apply(wrapped_fcn, y0, fwd_options, bck_options, *all_params)#, *model_params)

def equilibrium(
        fcn:Callable[...,torch.Tensor],
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
        assert_fcn_params(fcn, (y0, *params))

    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0
    def new_fcn(y, *params):
        return y - wrapped_fcn(y, *params)

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, *all_params)#

def minimize(
        fcn:Callable[...,torch.Tensor],
        y0:torch.Tensor,
        params:Sequence[Any]=[],
        fwd_options:Mapping[str,Any]={},
        bck_options:Mapping[str,Any]={}) -> torch.Tensor:
    """
    Solve the minimization problem:

        z = (argmin_y) fcn(y, *params)

    to find the best `y` that minimizes the output of the function `fcn`.
    The output of `fcn` must be a single element tensor.

    Arguments
    ---------
    * fcn: callable with output tensor (numel=1)
        The function
    * y0: torch.tensor with shape (*ny)
        Initial guess of the solution
    * params: list
        List of any other parameters to be put in fcn
    * fwd_options: dict
        Options for the minimizer method
    * bck_options: dict
        Options for the backward solve method

    Returns
    -------
    * y: torch.tensor with shape (*ny)
        The solution of the minimization
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (y0, *params))

    wrapped_fcn, all_params = wrap_fcn(fcn, (y0, *params))
    all_params = all_params[1:] # to exclude y0

    # the rootfinder algorithms are designed to move to the opposite direction
    # of the output of the function, so the output of this function is just
    # the grad of z w.r.t. y
    def new_fcn(y, *params):
        with torch.enable_grad():
            y1 = y.clone().requires_grad_()
            z = wrapped_fcn(y1, *params)
        grady, = torch.autograd.grad(z, (y1,), retain_graph=True,
            create_graph=torch.is_grad_enabled())
        return grady

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, *all_params)

class _RootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn:Callable[...,torch.Tensor],
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

        # split tensors and non-tensors params
        ctx.param_sep = TensorNonTensorSeparator(params)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(y, *tensor_params)

        return y

    @staticmethod
    def backward(ctx, grad_yout:torch.Tensor):
        param_sep = ctx.param_sep
        yout = ctx.saved_tensors[0]

        # merge the tensor and nontensor parameters
        tensor_params = ctx.saved_tensors[1:]
        params = param_sep.reconstruct_params(tensor_params)

        # dL/df
        jac_dfdy = jac(ctx.fcn, params=(yout, *params), idxs=[0])[0]
        gyfcn = solve(A=jac_dfdy.H, B=-grad_yout.unsqueeze(-1),
            fwd_options=ctx.bck_options, bck_options=ctx.bck_options).squeeze(-1)

        # get the grad for the params
        with torch.enable_grad():
            tensor_params_copy = [p.clone().requires_grad_() for p in tensor_params]
            params_copy = param_sep.reconstruct_params(tensor_params_copy)
            yfcn = ctx.fcn(yout, *params_copy)
        grad_tensor_params = torch.autograd.grad(yfcn, tensor_params_copy, grad_outputs=gyfcn,
            create_graph=torch.is_grad_enabled())
        grad_nontensor_params = [None for _ in range(param_sep.nnontensors())]
        grad_params = param_sep.reconstruct_params(grad_tensor_params, grad_nontensor_params)

        return (None, None, None, None, *grad_params)
