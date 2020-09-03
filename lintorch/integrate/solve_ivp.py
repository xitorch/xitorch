import torch
from typing import Callable, Union, Mapping, Any, Sequence
from lintorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from lintorch._core.pure_function import get_pure_function
from lintorch._impls.integrate.ivp import rk4_ivp
from lintorch._utils.misc import set_default_option, TensorNonTensorSeparator
from lintorch.debug.modes import is_debug_enabled

def solve_ivp(fcn:Callable[...,torch.Tensor],
              t:torch.Tensor,
              y0:torch.Tensor,
              params:Sequence[Any]=[],
              fwd_options:Mapping[str,Any]={},
              bck_options:Mapping[str,Any]={}):
    """
    Solve the initial value problem (IVP) which given the initial value `y0`,
    the function is then solve

        y(t) = y0 + int_t0^t f(t', y, *params) dt'

    Arguments
    ---------
    * fcn: callable with output a tensor with shape (*ny) or a list of tensors
        The function that represents dy/dt. The function takes an input of a
        single time `t` and `y` with shape (*ny) and produce dydt with shape (*ny).
    * t: torch.tensor with shape (nt,)
        The time points where the value of `y` is returned.
        It must be monotonically increasing or decreasing.
    * y0: torch.tensor with shape (*ny) or a list of tensors
        The initial value of y, i.e. y(t[0]) == y0
    * params: list
        List of other parameters required in the function.
    * fwd_options: dict
        Options for the forward solve_ivp method.
    * bck_options: dict
        Options for the backward solve_ivp method.

    Returns
    -------
    * yt: torch.tensor with shape (nt,*ny) or a list of tensors
        The values of `y` for each time step in `t`.
    """
    if is_debug_enabled():
        assert_fcn_params(fcn, (t, y0, *params))
    assert_runtime(len(t.shape) == 1, "Argument t must be a 1D tensor")

    # run once to see if the outputs is a tuple or a single tensor
    is_y0_list = isinstance(y0, list) or isinstance(y0, tuple)
    dydt = fcn(t[0], y0, *params)
    is_dydt_list = isinstance(dydt, list) or isinstance(y0, tuple)
    is_dydt_list = not isinstance(dydt, torch.Tensor)
    if is_y0_list != is_dydt_list:
        raise RuntimeError("The y0 and output of fcn must both be tuple or a tensor")

    pfcn = get_pure_function(fcn)
    if not is_y0_list:
        # make it a tuple
        @make_pure_function_sibling(pfcn)
        def pfcn2(t, y, *params):
            return (pfcn(t, y[0], *params),)
        y0 = (y0,)
        return _SolveIVP.apply(pfcn2, t, y0, fwd_options, bck_options, len(params), *params, *pfcn.objparams())[0]
    else:
        return _SolveIVP.apply(pfcn , t, y0, fwd_options, bck_options, len(params), *params, *pfcn.objparams())

class _SolveIVP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pfcn, t, y0, fwd_options, bck_options, nparams, *allparams):
        config = set_default_option({
            "method": "rk4",
        }, fwd_options)
        ctx.bck_config = set_default_option(config, bck_options)

        params = allparams[:nparams]
        objparams = allparams[nparams:]

        method = config["method"].lower()
        if method == "rk4":
            yt = rk4_ivp(pfcn, t, y0, params, **config)
        else:
            raise RuntimeError("Unknown solve_ivp method: %s" % config["method"])

        # save the parameters for backward
        ctx.param_sep = TensorNonTensorSeparator(allparams)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(t, y0, *tensor_params)
        ctx.pfcn = pfcn
        ctx.nparams = nparams
        ctx.ny = len(yt)
        ctx.yt = yt

        return yt # list of (nt,*ny)

    @staticmethod
    def backward(ctx, *grad_yt):
        # grad_yt: list of (nt, *ny)
        nparams = ctx.nparams
        pfcn = ctx.pfcn
        param_sep = ctx.param_sep
        ny = ctx.ny
        yt = ctx.yt

        # restore the parameters
        saved_tensors = ctx.saved_tensors
        t = saved_tensors[0]
        y0 = saved_tensors[1]
        tensor_params = saved_tensors[2:]
        allparams = param_sep.reconstruct_params(tensor_params)
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        def new_pfunc(t, inps, *tensor_params):
            # t: single-element
            y = inps[:ny] # list of (*ny)
            dLdy = inps[ny:2*ny] # list of (*ny)
            # TODO: get params from tensor_params here
            with torch.enable_grad():
                ycopy = y.clone().requires_grad_()
                f = pfcn(t, ycopy, *params) # list of (*ny)
            dfdy = torch.autograd.grad(f, ycopy,
                grad_outputs=dLdy,
                retain_graph=True,
                create_graph=torch.is_grad_enabled()) # list of (*ny)
            outs = (*f, # dydt
                    *[-d for d in dfdy], # d/dt (dL/dy)
                    )
            return outs

        t_flip = t.flip(0)
        t_flip_idx = -1
        inps = (*[yyt[t_flip_idx] for yyt in yt], # y
                *[gyt[t_flip_idx] for gyt in grad_yt], # dL/dy
                )
        for i in range(len(t_flip)-1):
            t_flip_idx -= 1
            outs = solve_ivp(new_pfunc, t_flip[i:i+2], inps, params,
                fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
            inps = (*[yyt[t_flip_idx] for yyt in yt], # y
                    *[gyt[t_flip_idx] + gy0 for (gyt,gy0) in zip(grad_yt, outs[ny:2*ny])], # dL/dy
                    )
        grad_y0 = inps[ny:2*ny] # dL/dy0, list of (*ny)
        grad_t = None # ???
        grad_params = [None for _ in range(len(nparams))]
        return (None, grad_t, grad_y0, None, None, None, *grad_params)
