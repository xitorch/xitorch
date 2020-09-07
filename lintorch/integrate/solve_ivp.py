import torch
from typing import Callable, Union, Mapping, Any, Sequence
from lintorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from lintorch._core.pure_function import get_pure_function, make_sibling
from lintorch._impls.integrate.ivp.explicit_rk import rk4_ivp, rk38_ivp
from lintorch._utils.misc import set_default_option, TensorNonTensorSeparator
from lintorch._utils.tensor import convert_none_grads_to_zeros
from lintorch.debug.modes import is_debug_enabled

__all__ = ["solve_ivp"]

def solve_ivp(fcn:Callable[...,torch.Tensor],
              ts:torch.Tensor,
              y0:torch.Tensor,
              params:Sequence[Any]=[],
              fwd_options:Mapping[str,Any]={},
              bck_options:Mapping[str,Any]={}) -> torch.Tensor:
    """
    Solve the initial value problem (IVP) which given the initial value `y0`,
    the function is then solve

        y(t) = y0 + int_t0^t f(t', y, *params) dt'

    Arguments
    ---------
    * fcn: callable with output a tensor with shape (*ny) or a list of tensors
        The function that represents dy/dt. The function takes an input of a
        single time `t` and `y` with shape (*ny) and produce dydt with shape (*ny).
    * ts: torch.tensor with shape (nt,)
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
        The values of `y` for each time step in `ts`.
    """
    if is_debug_enabled():
        assert_fcn_params(fcn, (ts[0], y0, *params))
    assert_runtime(len(ts.shape) == 1, "Argument ts must be a 1D tensor")

    # run once to see if the outputs is a tuple or a single tensor
    is_y0_list = isinstance(y0, list) or isinstance(y0, tuple)
    dydt = fcn(ts[0], y0, *params)
    is_dydt_list = isinstance(dydt, list) or isinstance(dydt, tuple)
    if is_y0_list != is_dydt_list:
        raise RuntimeError("The y0 and output of fcn must both be tuple or a tensor")

    pfcn = get_pure_function(fcn)
    if not is_y0_list:
        # make it a tuple
        @make_sibling(pfcn)
        def pfcn2(t, y, *params):
            return (pfcn(t, y[0], *params),)
        y0 = (y0,)
        return _SolveIVP.apply(pfcn2, ts, fwd_options, bck_options, len(y0), len(params), *y0, *params, *pfcn.objparams())[0]
    else:
        return _SolveIVP.apply(pfcn , ts, fwd_options, bck_options, len(y0), len(params), *y0, *params, *pfcn.objparams())

class _SolveIVP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pfcn, ts, fwd_options, bck_options, ny, nparams, *ally0params):
        config = set_default_option({
            "method": "rk4",
        }, fwd_options)
        ctx.bck_config = set_default_option(config, bck_options)

        y0 = ally0params[:ny]
        allparams = ally0params[ny:]
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        method = config["method"].lower()
        if method == "rk4":
            yt = rk4_ivp(pfcn, ts, y0, params, **config)
        elif method == "rk38":
            yt = rk38_ivp(pfcn, ts, y0, params, **config)
        else:
            raise RuntimeError("Unknown solve_ivp method: %s" % config["method"])

        # save the parameters for backward
        ctx.param_sep = TensorNonTensorSeparator(allparams, varonly=True)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(ts, *y0, *tensor_params)
        ctx.pfcn = pfcn
        ctx.nparams = nparams
        ctx.ny = ny
        ctx.yt = yt
        ctx.ts_requires_grad = ts.requires_grad

        return tuple(yt) # list of (nt,*ny)

    @staticmethod
    def backward(ctx, *grad_yt):
        # grad_yt: list of (nt, *ny)
        nparams = ctx.nparams
        pfcn = ctx.pfcn
        param_sep = ctx.param_sep
        ny = ctx.ny
        yt = ctx.yt
        ts_requires_grad = ctx.ts_requires_grad

        # restore the parameters
        saved_tensors = ctx.saved_tensors
        ts = saved_tensors[0]
        y0 = saved_tensors[1:1+ny]
        tensor_params = list(saved_tensors[1+ny:])
        allparams = param_sep.reconstruct_params(tensor_params)
        ntensor_params = len(tensor_params)
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        grad_enabled = torch.is_grad_enabled()
        # custom function to evaluate the input `pfcn` based on whether we want
        # to connect the graph or not
        def pfunc2(t, y, tensor_params):
            if not grad_enabled:
                # if graph is not constructed, then use the default tensor_params
                ycopy = [yi.detach().requires_grad_() for yi in y]
                tcopy = t.detach().requires_grad_()
                f = pfcn(tcopy, ycopy, *params)
                return f, tcopy, ycopy, tensor_params
            else:
                # if graph is constructed, then use the clone of the tensor params
                # so that infinite loop of backward can be avoided
                tensor_params_copy = [p.clone().requires_grad_() for p in tensor_params]
                ycopy = [yi.clone().requires_grad_() for yi in y]
                tcopy = t.clone().requires_grad_()
                allparams_copy = param_sep.reconstruct_params(tensor_params_copy)
                params_copy = allparams_copy[:nparams]
                objparams_copy = allparams_copy[nparams:]
                with pfcn.useobjparams(objparams_copy):
                    f = pfcn(tcopy, ycopy, *params_copy)
                return f, tcopy, ycopy, tensor_params_copy

        # slices and indices definitions on the augmented states
        y_slice = slice(None, ny, None) # [:ny]
        dLdy_slice = slice(ny, 2*ny, None) # [ny:2*ny]
        dLdt_index = 2*ny
        dLdt_slice = slice(dLdt_index, dLdt_index+1, None) # [2*ny:2*ny+1]
        dLdp_slice = slice(-ntensor_params, None, None) # [-ntensor_params:]
        state_size = 2*ny+1 + ntensor_params
        states = [None for _ in range(state_size)]

        def new_pfunc(t, states, *tensor_params):
            # t: single-element
            y = states[y_slice] # list of (*ny)
            dLdy = [-fi for fi in states[dLdy_slice]] # list of (*ny)
            with torch.enable_grad():
                f, t2, y2, tensor_params2 = pfunc2(t, y, tensor_params)
            allgradinputs = (list(y2) + [t2] + list(tensor_params2))
            allgrads = torch.autograd.grad(f,
                inputs=allgradinputs,
                grad_outputs=dLdy,
                retain_graph=True,
                allow_unused=True,
                create_graph=torch.is_grad_enabled()) # list of (*ny)
            allgrads = convert_none_grads_to_zeros(allgrads, allgradinputs)
            outs = (
                *f, # dydt
                *allgrads,
            )
            return outs

        ts_flip = ts.flip(0)
        t_flip_idx = -1
        states[y_slice   ] = [yyt[t_flip_idx] for yyt in yt]
        states[dLdy_slice] = [gyt[t_flip_idx] for gyt in grad_yt]
        states[dLdt_slice] = [torch.zeros_like(ts[0])]
        states[dLdp_slice] = [torch.zeros_like(tp) for tp in tensor_params]
        grad_ts = [None for _ in range(len(ts))] if ts_requires_grad else None

        for i in range(len(ts_flip)-1):
            if ts_requires_grad:
                fevals = pfunc2(ts_flip[i], states[y_slice], tensor_params)[0]
                dLdt1 = sum([torch.dot(feval.reshape(-1), gyt[t_flip_idx].reshape(-1))  for feval,gyt in zip(fevals, grad_yt)])
                states[dLdt_index] -= dLdt1
                grad_ts[t_flip_idx] = dLdt1.view(-1)

            t_flip_idx -= 1
            outs = solve_ivp(new_pfunc, ts_flip[i:i+2], states, tensor_params,
                fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
            # only take the output for the earliest time
            states = [out[-1] for out in outs]
            states[   y_slice] = [yyt[t_flip_idx] for yyt in yt]
            # gyt is the contribution from the input grad_y
            # gy0 is the propagated gradients from the later time step
            states[dLdy_slice] = [gyt[t_flip_idx] + gy0 for (gyt,gy0) in zip(grad_yt, states[dLdy_slice])]

        if ts_requires_grad:
            grad_ts[0] = states[dLdt_index].view(-1)

        grad_y0 = states[dLdy_slice] # dL/dy0, list of (*ny)
        if ts_requires_grad:
            grad_ts = torch.cat(grad_ts).view(*ts.shape)
        grad_tensor_params = states[dLdp_slice]
        grad_ntensor_params = [None for _ in range(len(allparams)-ntensor_params)]
        grad_params = param_sep.reconstruct_params(grad_tensor_params, grad_ntensor_params)
        return (None, grad_ts, None, None, None, None, *grad_y0, *grad_params)
