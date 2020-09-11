import math
from abc import abstractmethod
import torch
from typing import Callable, Union, Mapping, Any, Sequence, List
from lintorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from lintorch._core.editable_module import EditableModule
from lintorch._core.pure_function import get_pure_function, make_sibling
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
    if isinstance(xl, torch.Tensor):
        assert_runtime(torch.numel(xl) == 1, "xl must be a 1-element tensors")
    if isinstance(xu, torch.Tensor):
        assert_runtime(torch.numel(xu) == 1, "xu must be a 1-element tensors")

    out = fcn(xl, *params)
    is_tuple_out = not isinstance(out, torch.Tensor)
    if not is_tuple_out:
        dtype = out.dtype
        device = out.device
    elif len(out) > 0:
        dtype = out[0].dtype
        device = out[0].device
    else:
        raise RuntimeError("The output of the fcn must be non-empty")

    pfunc = get_pure_function(fcn)
    nparams = len(params)
    if not is_tuple_out:
        @make_sibling(pfunc)
        def pfunc2(x, *params):
            return (pfunc(x,*params),)
        return _Quadrature.apply(pfunc2, xl, xu, fwd_options, bck_options, nparams,
            dtype, device, *params, *pfunc.objparams())[0]
    else:
        return _Quadrature.apply(pfunc , xl, xu, fwd_options, bck_options, nparams,
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

            config = set_default_option({
                "method": "leggauss",
                "n": 100,
            }, fwd_options)
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
                    return [y * dxdt for y in ys]
                tl = tfm.x2t(xl)
                tu = tfm.x2t(xu)
            else:
                fcn2 = fcn
                tl = xl
                tu = xu

            method = config["method"].lower()
            if method == "leggauss":
                y = leggaussquad(fcn2, tl, tu, params, **config)
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
            ctx.nparams = nparams

            return tuple(y)

    @staticmethod
    def backward(ctx, *grad_ys):
        # retrieve the params
        ntensor_params = ctx.param_sep.ntensors()
        tensor_params = ctx.saved_tensors[-ntensor_params:]
        allparams = ctx.param_sep.reconstruct_params(tensor_params)
        nparams = ctx.nparams
        params = allparams[:nparams]
        fcn = ctx.fcn
        ngrady = len(grad_ys)

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
            grad_xl = -sum([torch.sum(gy * f).reshape(xl.shape) for (gy,f) in zip(grad_ys, fcn(xl, *params))]) if ctx.xltensor else None
            grad_xu =  sum([torch.sum(gy * f).reshape(xu.shape) for (gy,f) in zip(grad_ys, fcn(xu, *params))]) if ctx.xutensor else None

            def new_fcn(x, *grad_y_params):
                grad_ys = grad_y_params[:ngrady]
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
            dydts = quad(new_fcn, xl, xu, params=(*grad_ys, *tensor_params),
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
        sec = 1./torch.cos(t)
        return sec*sec

    def x2t(self, x):
        return torch.atan(x)
