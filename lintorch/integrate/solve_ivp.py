import torch
from typing import Callable, Union, Mapping, Any, Sequence
from lintorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from lintorch._core.editable_module import wrap_fcn
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
    * fcn: callable with output a tensor with shape (*ny)
        The function that represents dy/dt
    * t: torch.tensor with shape (nt,)
        The time points where the value of `y` is returned.
        It must be always increasing.
    * y0: torch.tensor with shape (*ny)
        The initial value of y, i.e. y(t[0]) == y0
    * params: list
        List of other parameters required in the function.
    * fwd_options: dict
        Options for the forward solve_ivp method.
    * bck_options: dict
        Options for the backward solve_ivp method.

    Returns
    -------
    * yt: torch.tensor with shape (nt, *ny)
        The values of `y` for each time step in `t`.
    """
    if is_debug_enabled():
        assert_fcn_params(fcn, (t, y0, *params))
    assert_runtime(len(t.shape) == 1, "Argument t must be a 1D tensor")

    wrapped_fcn, all_params = wrap_fcn(fcn, (t, y0, *params))
    all_params = all_params[2:] # take out t and y0
    return _SolveIVP.apply(wrapped_fcn, t, y0, fwd_options, bck_options, *all_params)

class _SolveIVP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, t, y0, fwd_options, bck_options, *all_params):
        config = set_default_option({
            "method": "rk4",
        }, fwd_options)
        ctx.bck_config = set_default_option(config, bck_options)

        method = config["method"].lower()
        if method == "rk4":
            yt = rk4_ivp(fcn, t, y0, all_params, **config)
        else:
            raise RuntimeError("Unknown solve_ivp method: %s" % config["method"])

        # save the parameters for backward
        ctx.param_sep = TensorNonTensorSeparator(all_params)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(t, y0, *tensor_params)
        ctx.fcn = fcn

        return yt

    @staticmethod
    def backward(ctx, grad_yt):
        # restore the parameters
        saved_tensors = ctx.saved_tensors
        t = saved_tensors[0]
        y0 = saved_tensors[1]
        tensor_params = saved_tensors[2:]
        params = ctx.param_sep.reconstruct_params(tensor_params)
        fcn = ctx.fcn
