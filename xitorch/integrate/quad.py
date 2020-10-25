from abc import abstractmethod
import torch
from typing import Callable, Union, Mapping, Any, Sequence
from xitorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from xitorch._core.pure_function import get_pure_function, make_sibling
from xitorch._utils.misc import set_default_option, TensorNonTensorSeparator, \
    TensorPacker, get_method
from xitorch._impls.integrate.fixed_quad import leggauss
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch.debug.modes import is_debug_enabled

__all__ = ["quad"]

def quad(
        fcn: Union[Callable[..., torch.Tensor], Callable[..., Sequence[torch.Tensor]]],
        xl: Union[float, int, torch.Tensor],
        xu: Union[float, int, torch.Tensor],
        params: Sequence[Any] = [],
        bck_options: Mapping[str, Any] = {},
        method: Union[str, Callable, None] = None,
        **fwd_options) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    r"""
    Calculate the quadrature:

    .. math::

        y = \int_{x_l}^{x_u} f(x, \theta)\ \mathrm{d}x

    Arguments
    ---------
    fcn: callable
        The function to be integrated. Its output must be a tensor with
        shape ``(*nout)`` or list of tensors.
    xl: float, int or 1-element torch.Tensor
        The lower bound of the integration.
    xu: float, int or 1-element torch.Tensor
        The upper bound of the integration.
    params: list
        Sequence of any other parameters for the function ``fcn``.
    bck_options: dict
        Options for the backward quadrature method.
    method: str or callable or None
        Quadrature method. If None, it will choose ``"leggauss"``.
    **fwd_options
        Method-specific options (see method section).

    Returns
    -------
    torch.tensor or a list of tensors
        The quadrature results with shape ``(*nout)`` or list of tensors.
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (xl, *params))
    if isinstance(xl, torch.Tensor):
        assert_runtime(torch.numel(xl) == 1, "xl must be a 1-element tensors")
    if isinstance(xu, torch.Tensor):
        assert_runtime(torch.numel(xu) == 1, "xu must be a 1-element tensors")
    if method is None:
        method = "leggauss"
    fwd_options["method"] = method

    out = fcn(xl, *params)
    if isinstance(out, torch.Tensor):
        dtype = out.dtype
        device = out.device
        is_tuple_out = False
    elif len(out) > 0:
        dtype = out[0].dtype
        device = out[0].device
        is_tuple_out = True
    else:
        raise RuntimeError("The output of the fcn must be non-empty")

    pfunc = get_pure_function(fcn)
    nparams = len(params)
    if is_tuple_out:
        packer = TensorPacker(out)

        @make_sibling(pfunc)
        def pfunc2(x, *params):
            y = fcn(x, *params)
            return packer.flatten(y)

        res = _Quadrature.apply(pfunc2, xl, xu, fwd_options, bck_options, nparams,
                                dtype, device, *params, *pfunc.objparams())
        return packer.pack(res)
    else:
        return _Quadrature.apply(pfunc, xl, xu, fwd_options, bck_options, nparams,
                                 dtype, device, *params, *pfunc.objparams())

class _Quadrature(torch.autograd.Function):
    # NOTE: _Quadrature method do not involve changing the state (objparams) of
    # fcn, so there is no need in using `with fcn.useobjparams(objparams)`
    # statements.
    # The function `disable_state_change()` is used to disable state change of
    # the pure function during the execution of the forward and backward
    # calculations

    @staticmethod
    def forward(ctx, fcn, xl, xu, fwd_options, bck_options, nparams,
                dtype, device, *all_params):

        with fcn.disable_state_change():

            config = fwd_options
            ctx.bck_config = set_default_option(config, bck_options)

            params = all_params[:nparams]
            objparams = all_params[nparams:]

            # convert to tensor
            xl = torch.as_tensor(xl, dtype=dtype, device=device)
            xu = torch.as_tensor(xu, dtype=dtype, device=device)

            # apply transformation if the boundaries contain inf
            if _isinf(xl) or _isinf(xu):
                tfm = _TanInfTransform()

                @make_sibling(fcn)
                def fcn2(t, *params):
                    ys = fcn(tfm.forward(t), *params)
                    dxdt = tfm.dxdt(t)
                    return ys * dxdt

                tl = tfm.x2t(xl)
                tu = tfm.x2t(xu)
            else:
                fcn2 = fcn
                tl = xl
                tu = xu

            method = config.pop("method")
            methods = {
                "leggauss": leggauss
            }
            method_fcn = get_method("quad", methods, method)
            y = method_fcn(fcn2, tl, tu, params, **config)

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
            ctx.nparams = nparams
            return y

    @staticmethod
    def backward(ctx, grad_ys):
        # retrieve the params
        ntensor_params = ctx.param_sep.ntensors()
        tensor_params = ctx.saved_tensors[-ntensor_params:]
        allparams = ctx.param_sep.reconstruct_params(tensor_params)
        nparams = ctx.nparams
        params = allparams[:nparams]
        fcn = ctx.fcn

        with fcn.disable_state_change():

            # restore xl, and xu
            xlxu_tensor = ctx.saved_tensors[:-ntensor_params]
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
            grad_xl = -torch.dot(grad_ys.reshape(-1), fcn(xl, *params).reshape(-1)
                                 ).reshape(xl.shape) if ctx.xltensor else None
            grad_xu = torch.dot(grad_ys.reshape(-1), fcn(xu, *params).reshape(-1)
                                ).reshape(xu.shape) if ctx.xutensor else None

            def new_fcn(x, *grad_y_params):
                grad_ys = grad_y_params[0]
                # not setting objparams and params because the params and objparams
                # are still the same objects as the objects outside
                with torch.enable_grad():
                    f = fcn(x, *params)
                dfdts = torch.autograd.grad(f, tensor_params,
                                            grad_outputs=grad_ys,
                                            retain_graph=True,
                                            create_graph=torch.is_grad_enabled())
                return dfdts

            # reconstruct grad_params
            # listing tensor_params in the params of quad to make sure it gets
            # the gradient calculated
            dydts = quad(new_fcn, xl, xu, params=(grad_ys, *tensor_params),
                         fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
            dydns = [None for _ in range(ctx.param_sep.nnontensors())]
            grad_params = ctx.param_sep.reconstruct_params(dydts, dydns)

            return (None, grad_xl, grad_xu, None, None, None, None, None, *grad_params)

def _isinf(x):
    return torch.any(torch.isinf(x))

class _BaseInfTransform(object):
    @abstractmethod
    def forward(self, t):
        pass

    @abstractmethod
    def dxdt(self, t):
        pass

    @abstractmethod
    def x2t(self, x):
        pass

class _TanInfTransform(_BaseInfTransform):
    def forward(self, t):
        return torch.tan(t)

    def dxdt(self, t):
        sec = 1. / torch.cos(t)
        return sec * sec

    def x2t(self, x):
        return torch.atan(x)


# docstring completion
quad.__doc__ = get_methods_docstr(quad, [leggauss])
