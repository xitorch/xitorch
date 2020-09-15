import inspect
from typing import Callable, Iterable, Mapping, Any, Sequence
import torch
import numpy as np
import scipy.optimize
import lintorch as lt
from lintorch._utils.misc import set_default_option, TensorNonTensorSeparator
from lintorch._utils.assertfuncs import assert_fcn_params
from lintorch._impls.optimize.root.rootsolver import broyden1
from lintorch.linalg.solve import solve
from lintorch.grad.jachess import jac
from lintorch.linalg.linop import LinearOperator, checklinop
from lintorch._core.editable_module import EditableModule
from lintorch._core.pure_function import get_pure_function, make_sibling
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

    pfunc = get_pure_function(fcn)
    return _RootFinder.apply(pfunc, y0, fwd_options, bck_options, len(params), *params, *pfunc.objparams())

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

    pfunc = get_pure_function(fcn)

    @make_sibling(pfunc)
    def new_fcn(y, *params):
        return y - pfunc(y, *params)

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, len(params), *params, *pfunc.getobjparams())

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

    pfunc = get_pure_function(fcn)

    # the rootfinder algorithms are designed to move to the opposite direction
    # of the output of the function, so the output of this function is just
    # the grad of z w.r.t. y
    @make_sibling(pfunc)
    def new_fcn(y, *params):
        with torch.enable_grad():
            y1 = y.clone().requires_grad_()
            z = pfunc(y1, *params)
        grady, = torch.autograd.grad(z, (y1,), retain_graph=True,
            create_graph=torch.is_grad_enabled())
        return grady

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, len(params), *params, *pfunc.objparams())

class _RootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn:Callable[...,torch.Tensor],
            y0:torch.Tensor,
            options:Mapping[str,Any],
            bck_options:Mapping[str,Any],
            nparams:int,
            *allparams) -> torch.Tensor:

        # set default options
        config = set_default_option({
            "method": "broyden1",
        }, options)
        ctx.bck_options = set_default_option({
            "method": "exactsolve"
        }, bck_options)

        params = allparams[:nparams]
        objparams = allparams[nparams:]

        with fcn.useobjparams(objparams):

            orig_method = config.pop("method")
            method = orig_method.lower()
            if method == "broyden1":
                y = broyden1(fcn, y0, params, **config)
            else:
                raise RuntimeError("Unknown rootfinder method: %s" % orig_method)

        ctx.fcn = fcn

        # split tensors and non-tensors params
        ctx.nparams = nparams
        ctx.param_sep = TensorNonTensorSeparator(allparams)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(y, *tensor_params)

        return y

    @staticmethod
    def backward(ctx, grad_yout:torch.Tensor):
        param_sep = ctx.param_sep
        yout = ctx.saved_tensors[0]
        nparams = ctx.nparams
        fcn = ctx.fcn

        # merge the tensor and nontensor parameters
        tensor_params = ctx.saved_tensors[1:]
        allparams = param_sep.reconstruct_params(tensor_params)
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        # dL/df
        with fcn.useobjparams(objparams):

            jac_dfdy = jac(ctx.fcn, params=(yout, *params), idxs=[0])[0]
            gyfcn = solve(A=jac_dfdy.H, B=-grad_yout.reshape(-1,1),
                fwd_options=ctx.bck_options, bck_options=ctx.bck_options).reshape(grad_yout.shape)

            # get the grad for the params
            with torch.enable_grad():
                tensor_params_copy = [p.clone().requires_grad_() for p in tensor_params]
                allparams_copy = param_sep.reconstruct_params(tensor_params_copy)
                params_copy = allparams_copy[:nparams]
                objparams_copy = allparams_copy[nparams:]
                with fcn.useobjparams(objparams_copy):
                    yfcn = fcn(yout, *params_copy)

            grad_tensor_params = torch.autograd.grad(yfcn, tensor_params_copy, grad_outputs=gyfcn,
                create_graph=torch.is_grad_enabled())
            grad_nontensor_params = [None for _ in range(param_sep.nnontensors())]
            grad_params = param_sep.reconstruct_params(grad_tensor_params, grad_nontensor_params)

        return (None, None, None, None, None, *grad_params)
