import torch
from typing import Callable, Union, Mapping, Any, Sequence, List
from lintorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from lintorch._core.editable_module import wrap_fcn
from lintorch._utils.misc import set_default_option, TensorNonTensorSeparator
from lintorch._impls.integrate.fixed_quad import leggaussquad
from lintorch.debug.modes import is_debug_enabled

__all__ = ["quad"]

def quad(
        fcn:Union[Callable[...,torch.Tensor], Callable[...,List[torch.Tensor]]],
        xl:Union[float,int,torch.Tensor],
        xu:Union[float,int,torch.Tensor],
        params:Sequence[Any]=[],
        fwd_options:Mapping[str,Any]={},
        bck_options:Mapping[str,Any]={}):
    """
    Calculate the quadrature of the function `fcn` from `x0` to `xf`:

        y = int_xl^xu fcn(x, *params)

    Arguments
    ---------
    * fcn: callable with output tensor with shape (*nout) or list of tensors
        The function to be integrated.
    * xl, xu: float, int, or 1-element torch.Tensor
        The lower and upper bound of the integration.
    * params: list
        List of any other parameters for the function `fcn`.
    * fwd_options: dict
        Options for the forward quadrature method.
    * bck_options: dict
        Options for the backward quadrature method.

    Returns
    -------
    * y: torch.tensor with shape (*nout) or list of tensors
        The quadrature results.
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (xl, *params))
    assert_runtime(torch.numel(xl) == 1, "xl must be a 1-element tensors")
    assert_runtime(torch.numel(xu) == 1, "xu must be a 1-element tensors")

    out = fcn(xl, *params)
    is_tuple_out = not isinstance(out, torch.Tensor)

    wrapped_fcn, all_params = wrap_fcn(fcn, (xl, *params))
    all_params = all_params[1:] # to exclude x
    if not is_tuple_out:
        wrapped_fcn2 = lambda x,*params: (wrapped_fcn(x,*params),)
        return _Quadrature.apply(wrapped_fcn2, xl, xu, fwd_options, bck_options, *all_params)[0]
    else:
        return _Quadrature.apply(wrapped_fcn , xl, xu, fwd_options, bck_options, *all_params)

class _Quadrature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, xl, xu, fwd_options, bck_options, *all_params):
        config = set_default_option({
            "method": "leggauss",
            "n": 100,
        }, fwd_options)
        ctx.bck_config = set_default_option(config, bck_options)

        method = config["method"].lower()
        if method == "leggauss":
            y = leggaussquad(fcn, xl, xu, all_params, **config)
        else:
            raise RuntimeError("Unknown quad method: %s" % config["method"])

        # save the parameters for backward
        ctx.param_sep = TensorNonTensorSeparator(all_params)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.xltensor = isinstance(xl, torch.Tensor)
        ctx.xutensor = isinstance(xu, torch.Tensor)
        xlxu_tensor = ([xl] if ctx.xltensor else []) + \
                      ([xu] if ctx.xutensor else [])
        ctx.xlxu_nontensor = ([xl] if not ctx.xltensor else []) + \
                             ([xu] if not ctx.xutensor else [])
        ctx.save_for_backward(*xlxu_tensor, *tensor_params)
        ctx.fcn = fcn

        return tuple(y)

    @staticmethod
    def backward(ctx, *grad_ys):
        # retrieve the params
        nparams = ctx.param_sep.ntensors()
        tensor_params = ctx.saved_tensors[-nparams:]
        params = ctx.param_sep.reconstruct_params(tensor_params)
        fcn = ctx.fcn
        ngrady = len(grad_ys)

        # restore xl, and xu
        xlxu_tensor = ctx.saved_tensors[:-nparams]
        if ctx.xltensor and ctx.xutensor:
            xl, xu = xlxu_tensor
        elif ctx.xltensor:
            xl = xlxu_tensor[0]
            xu = ctx.xlxu_nontensor[0]
        elif ctx.xutensor:
            xu = xlxu_tensor[0]
            xl = ctx.xlxu_nontensor[0]
        else:
            xl, xu = ctx.xlxu_nontensor

        # calculate the gradient for the boundaries
        grad_xl = -sum([torch.sum(gy * f).reshape(xl.shape) for (gy,f) in zip(grad_ys, fcn(xl, *params))]) if ctx.xltensor else None
        grad_xu =  sum([torch.sum(gy * f).reshape(xu.shape) for (gy,f) in zip(grad_ys, fcn(xu, *params))]) if ctx.xutensor else None

        # calculate the gradients for the integrands
        def new_fcn(x, *grad_y_params):
            grad_ys = grad_y_params[:ngrady]
            tensor_params = grad_y_params[ngrady:]
            params = ctx.param_sep.reconstruct_params(tensor_params)
            with torch.enable_grad():
                f = fcn(x, *params)
            dfdts = torch.autograd.grad(f, tensor_params,
                grad_outputs=grad_ys,
                retain_graph=True,
                create_graph=torch.is_grad_enabled())
            return dfdts

        # reconstruct grad_params
        dydts = quad(new_fcn, xl, xu, params=(*grad_ys, *tensor_params),
                     fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
        dydns = [None for _ in range(ctx.param_sep.nnontensors())]
        grad_params = ctx.param_sep.reconstruct_params(dydts, dydns)

        return (None, grad_xl, grad_xu, None, None, *grad_params)
