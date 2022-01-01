import torch
import copy
from typing import Callable, Union, Mapping, Any, Sequence, Dict
from xitorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from xitorch._core.pure_function import get_pure_function, make_sibling
from xitorch._impls.integrate.ivp.explicit_rk import rk4_ivp, rk38_ivp, fwd_euler_ivp
from xitorch._impls.integrate.ivp.adaptive_rk import rk23_adaptive, rk45_adaptive
from xitorch._utils.misc import set_default_option, TensorNonTensorSeparator, \
    TensorPacker, get_method
from xitorch._utils.tensor import convert_none_grads_to_zeros
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch.debug.modes import is_debug_enabled

__all__ = ["solve_ivp"]

def solve_ivp(fcn: Union[Callable[..., torch.Tensor], Callable[..., Sequence[torch.Tensor]]],
              ts: torch.Tensor,
              y0: torch.Tensor,
              params: Sequence[Any] = [],
              bck_options: Mapping[str, Any] = {},
              method: Union[str, Callable, None] = None,
              **fwd_options) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    r"""
    Solve the initial value problem (IVP) or also commonly known as ordinary
    differential equations (ODE), where given the initial value :math:`\mathbf{y_0}`,
    it then solves

    .. math::

        \mathbf{y}(t) = \mathbf{y_0} + \int_{t_0}^{t} \mathbf{f}(t', \mathbf{y}, \theta)\ \mathrm{d}t'

    Although the original ``solve_ivp`` does not accept batched ``ts``, it can
    be batched using functorch's ``vmap`` (only for explicit solver, though,
    e.g. ``rk38``, ``rk4``, and ``euler``). Adaptive steps cannot be vmapped
    at the moment.

    Arguments
    ---------
    fcn: callable
        The function that represents dy/dt. The function takes an input of a
        single time ``t`` and tensor ``y`` with shape ``(*ny)`` and
        produce :math:`\mathrm{d}\mathbf{y}/\mathrm{d}t` with shape ``(*ny)``.
        The output of the function must be a tensor with shape ``(*ny)`` or
        a list of tensors.
    ts: torch.tensor
        The time points where the value of `y` will be returned.
        It must be monotonically increasing or decreasing.
        It is a tensor with shape ``(nt,)``.
    y0: torch.tensor
        The initial value of ``y``, i.e. ``y(t[0]) == y0``.
        It is a tensor with shape ``(*ny)`` or a list of tensors.
    params: list
        Sequence of other parameters required in the function.
    bck_options: dict
        Options for the backward solve_ivp method. If not specified, it will
        take the same options as fwd_options.
    method: str or callable or None
        Initial value problem solver. If None, it will choose ``"rk45"``.
    **fwd_options
        Method-specific option (see method section below).

    Returns
    -------
    torch.tensor or a list of tensors
        The values of ``y`` for each time step in ``ts``.
        It is a tensor with shape ``(nt,*ny)`` or a list of tensors
    """
    if is_debug_enabled():
        assert_fcn_params(fcn, (ts[0], y0, *params))
    assert_runtime(len(ts.shape) == 1, "Argument ts must be a 1D tensor")

    if method is None:  # set the default method
        method = "rk45"
    fwd_options["method"] = method

    is_y0_list = isinstance(y0, (list, tuple))
    pfcn = get_pure_function(fcn)
    if is_y0_list:
        nt = len(ts)
        roller = TensorPacker(y0)

        @make_sibling(pfcn)
        def pfcn2(t, ytensor, *params):
            ylist = roller.pack(ytensor)
            res_list = pfcn(t, ylist, *params)
            if not isinstance(res_list, (list, tuple)):
                raise RuntimeError("The y0 and output of fcn must both be tuple or a tensor")
            res = roller.flatten(res_list)
            return res

        y0 = roller.flatten(y0)
        res = _SolveIVP.apply(pfcn2, ts, fwd_options, bck_options, len(params), y0, *params, *pfcn.objparams())
        return roller.pack(res)
    else:
        return _SolveIVP.apply(pfcn, ts, fwd_options, bck_options, len(params), y0, *params, *pfcn.objparams())

class _SolveIVP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pfcn, ts, fwd_options, bck_options, nparams, y0, *allparams):
        config = fwd_options
        ctx.bck_config = set_default_option(config, bck_options)

        params = allparams[:nparams]
        objparams = allparams[nparams:]

        method = config.pop("method")
        methods = {
            "rk4": rk4_ivp,
            "rk38": rk38_ivp,
            "rk23": rk23_adaptive,
            "rk45": rk45_adaptive,
            "euler": fwd_euler_ivp,
        }
        solver = get_method("solve_ivp", methods, method)
        yt = solver(pfcn, ts, y0, params, **config)

        # save the parameters for backward
        ctx.param_sep = TensorNonTensorSeparator(allparams, varonly=True)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(ts, y0, *tensor_params)
        ctx.pfcn = pfcn
        ctx.nparams = nparams
        ctx.yt = yt
        ctx.ts_requires_grad = ts.requires_grad

        return yt

    @staticmethod
    def backward(ctx, grad_yt):
        # grad_yt: (nt, *ny)
        nparams = ctx.nparams
        pfcn = ctx.pfcn
        param_sep = ctx.param_sep
        yt = ctx.yt
        ts_requires_grad = ctx.ts_requires_grad

        # restore the parameters
        saved_tensors = ctx.saved_tensors
        ts = saved_tensors[0]
        y0 = saved_tensors[1]
        tensor_params = list(saved_tensors[2:])
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
                ycopy = y.detach().requires_grad_()  # [yi.detach().requires_grad_() for yi in y]
                tcopy = t.detach().requires_grad_()
                f = pfcn(tcopy, ycopy, *params)
                return f, tcopy, ycopy, tensor_params
            else:
                # if graph is constructed, then use the clone of the tensor params
                # so that infinite loop of backward can be avoided
                tensor_params_copy = [p.clone().requires_grad_() for p in tensor_params]
                ycopy = y.clone().requires_grad_()
                tcopy = t.clone().requires_grad_()
                allparams_copy = param_sep.reconstruct_params(tensor_params_copy)
                params_copy = allparams_copy[:nparams]
                objparams_copy = allparams_copy[nparams:]
                with pfcn.useobjparams(objparams_copy):
                    f = pfcn(tcopy, ycopy, *params_copy)
                return f, tcopy, ycopy, tensor_params_copy

        # slices and indices definitions on the augmented states
        y_index = 0
        dLdy_index = 1
        dLdt_index = 2
        dLdt_slice = slice(dLdt_index, dLdt_index + 1, None)  # [2:3]
        dLdp_slice = slice(-ntensor_params, None, None) if ntensor_params > 0 else slice(0,
                                                                                         0, None)  # [-ntensor_params:]
        state_size = 3 + ntensor_params
        states = [None for _ in range(state_size)]

        def new_pfunc(t, states, *tensor_params):
            # t: single-element
            y = states[y_index]
            dLdy = -states[dLdy_index]
            with torch.enable_grad():
                f, t2, y2, tensor_params2 = pfunc2(t, y, tensor_params)
            allgradinputs = ([y2] + [t2] + list(tensor_params2))
            allgrads = torch.autograd.grad(f,
                                           inputs=allgradinputs,
                                           grad_outputs=dLdy,
                                           retain_graph=True,
                                           allow_unused=True,
                                           create_graph=torch.is_grad_enabled())  # list of (*ny)
            allgrads = convert_none_grads_to_zeros(allgrads, allgradinputs)
            outs = (
                f,  # dydt
                *allgrads,
            )
            return outs

        ts_flip = ts.flip(0)
        t_flip_idx = -1
        states[y_index] = yt[t_flip_idx]
        states[dLdy_index] = grad_yt[t_flip_idx]
        states[dLdt_index] = torch.zeros_like(ts[0])
        states[dLdp_slice] = [torch.zeros_like(tp) for tp in tensor_params]
        grad_ts = [None for _ in range(len(ts))] if ts_requires_grad else None

        # define a new function for the augmented dynamics
        bkw_roller = TensorPacker(states)

        @make_sibling(new_pfunc)
        def pfcn_back(t, ytensor, *params):
            ylist = bkw_roller.pack(ytensor)
            res_list = new_pfunc(t, ylist, *params)
            res = bkw_roller.flatten(res_list)
            return res

        for i in range(len(ts_flip) - 1):
            if ts_requires_grad:
                feval = pfunc2(ts_flip[i], states[y_index], tensor_params)[0]
                dLdt1 = torch.dot(feval.reshape(-1), grad_yt[t_flip_idx].reshape(-1))
                states[dLdt_index] -= dLdt1
                grad_ts[t_flip_idx] = dLdt1.reshape(-1)

            t_flip_idx -= 1
            states_flatten = bkw_roller.flatten(states)
            fwd_config = copy.copy(ctx.bck_config)
            bck_config = copy.copy(ctx.bck_config)
            outs_flatten = _SolveIVP.apply(
                pfcn_back, ts_flip[i:i + 2], fwd_config, bck_config, len(tensor_params),
                states_flatten, *tensor_params)
            outs = bkw_roller.pack(outs_flatten)

            # only take the output for the earliest time
            states = [out[-1] for out in outs]
            states[y_index] = yt[t_flip_idx]
            # gyt is the contribution from the input grad_y
            # gy0 is the propagated gradients from the later time step
            states[dLdy_index] = grad_yt[t_flip_idx] + states[dLdy_index]

        if ts_requires_grad:
            grad_ts[0] = states[dLdt_index].reshape(-1)

        grad_y0 = states[dLdy_index]  # dL/dy0, (*ny)
        if ts_requires_grad:
            grad_ts = torch.cat(grad_ts).reshape(*ts.shape)
        grad_tensor_params = states[dLdp_slice]
        grad_ntensor_params = [None for _ in range(len(allparams) - ntensor_params)]
        grad_params = param_sep.reconstruct_params(grad_tensor_params, grad_ntensor_params)
        return (None, grad_ts, None, None, None, grad_y0, *grad_params)


# docstring completion
ivp_methods: Dict[str, Callable] = {
    "rk45": rk45_adaptive,
    "rk23": rk23_adaptive,
    "rk4": rk4_ivp,
    "rk38": rk38_ivp,
    "euler": fwd_euler_ivp,
}
solve_ivp.__doc__ = get_methods_docstr(solve_ivp, ivp_methods)
