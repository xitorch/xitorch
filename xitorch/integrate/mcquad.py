import torch
from typing import Union, Sequence, Any, Callable, Mapping
from xitorch.debug.modes import is_debug_enabled
from xitorch._core.pure_function import get_pure_function, make_sibling
from xitorch._utils.misc import set_default_option, TensorNonTensorSeparator, \
    TensorPacker, get_method
from xitorch._utils.assertfuncs import assert_fcn_params
from xitorch._impls.integrate.mcsamples.mcmc import mh, mhcustom, dummy1d
from xitorch._docstr.api_docstr import get_methods_docstr

__all__ = ["mcquad"]

def mcquad(
        ffcn: Union[Callable[..., torch.Tensor], Callable[..., Sequence[torch.Tensor]]],
        log_pfcn: Callable[..., torch.Tensor],
        x0: torch.Tensor,
        fparams: Sequence[Any] = [],
        pparams: Sequence[Any] = [],
        bck_options: Mapping[str, Any] = {},
        method: Union[str, Callable, None] = None,
        **fwd_options) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    r"""
    Performing monte carlo quadrature to calculate the expectation value:

    .. math::
       \mathbb{E}_p[f] = \frac{\int f(\mathbf{x},\theta_f) p(\mathbf{x},\theta_p)
       \ \mathrm{d}\mathbf{x} }{ \int p(\mathbf{x},\theta_p)\ \mathrm{d}\mathbf{x} }

    Arguments
    ---------
    ffcn: Callable
        The function with to be integrated. Its outputs is a tensor or a
        list of tensors. To call the function: ``ffcn(x, *fparams)``
    log_pfcn: Callable
        The natural logarithm of the probability function. The output should be
        a one-element tensor. To call the function: ``log_pfcn(x, *pparams)``
    x0: torch.Tensor
        Tensor with any size as the initial position.
        The call ``ffcn(x0,*fparams)`` must work.
    fparams: list
        Sequence of any other parameters for ``ffcn``.
    pparams: list
        Sequence of any other parameters for ``gfcn``.
    bck_options: dict
        Options for the backward mcquad operation. Unspecified fields will be
        taken from ``fwd_options``.
    method: str or callable or None
        Monte Carlo quadrature method. If None, it will choose ``"mh"``.
    **fwd_options: dict
        Method-specific options (see method section below).

    Returns
    -------
    torch.Tensor or a list of torch.Tensor
        The expectation values of the function ``ffcn`` over the space of ``x``.
        If the output of ``ffcn`` is a list, then this is also a list.
    """
    if method is None:
        method = "mh"
    return _mcquad(ffcn, log_pfcn, x0, None, None, fparams, pparams,
                   method, bck_options, **fwd_options)

def _mcquad(ffcn, log_pfcn, x0, xsamples, wsamples, fparams, pparams, method,
            bck_options, **fwd_options):
    # this is mcquad with an additional xsamples argument, to prevent xsamples being set by users

    if is_debug_enabled():
        assert_fcn_params(ffcn, (x0, *fparams))
        assert_fcn_params(log_pfcn, (x0, *pparams))

    # check if ffcn produces a list / tuple
    out = ffcn(x0, *fparams)
    is_tuple_out = isinstance(out, list) or isinstance(out, tuple)

    # get the pure functions
    pure_ffcn = get_pure_function(ffcn)
    pure_logpfcn = get_pure_function(log_pfcn)
    nfparams = len(fparams)
    npparams = len(pparams)
    fobjparams = pure_ffcn.objparams()
    pobjparams = pure_logpfcn.objparams()
    nf_objparams = len(fobjparams)

    if is_tuple_out:
        packer = TensorPacker(out)

        @make_sibling(pure_ffcn)
        def pure_ffcn2(x, *fparams):
            y = pure_ffcn(x, *fparams)
            return packer.flatten(y)
        res = _MCQuad.apply(pure_ffcn2, pure_logpfcn, x0, None, None,
                            method, fwd_options, bck_options,
                            nfparams, nf_objparams, npparams, *fparams, *fobjparams, *pparams, *pobjparams)
        return packer.pack(res)
    else:
        return _MCQuad.apply(pure_ffcn, pure_logpfcn, x0, None, None,
                             method, fwd_options, bck_options,
                             nfparams, nf_objparams, npparams, *fparams, *fobjparams, *pparams, *pobjparams)

class _MCQuad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ffcn, log_pfcn, x0, xsamples, wsamples,
                method, fwd_options, bck_options,
                nfparams, nf_objparams, npparams, *all_fpparams):
        # set up the default options
        config = fwd_options
        ctx.bck_config = set_default_option(config, bck_options)

        # split the parameters
        fparams    = all_fpparams[:nfparams]
        fobjparams = all_fpparams[nfparams:nfparams + nf_objparams]
        pparams    = all_fpparams[nfparams + nf_objparams:nfparams + nf_objparams + npparams]
        pobjparams = all_fpparams[nfparams + nf_objparams + npparams:]

        # select the method for the sampling
        if xsamples is None:
            methods = {
                "mh": mh,
                "_dummy1d": dummy1d,
                "mhcustom": mhcustom,
            }
            method_fcn = get_method("mcquad", methods, method)
            xsamples, wsamples = method_fcn(log_pfcn, x0, pparams, **config)
        epf = _integrate(ffcn, xsamples, wsamples, fparams)

        # save parameters for backward calculations
        ctx.xsamples = xsamples
        ctx.wsamples = wsamples
        ctx.ffcn = ffcn
        ctx.log_pfcn = log_pfcn
        ctx.fparam_sep = TensorNonTensorSeparator((*fparams, *fobjparams))
        ctx.pparam_sep = TensorNonTensorSeparator((*pparams, *pobjparams))
        ctx.nfparams = len(fparams)
        ctx.npparams = len(pparams)
        ctx.method = method

        # save for backward
        ftensor_params = ctx.fparam_sep.get_tensor_params()
        ptensor_params = ctx.pparam_sep.get_tensor_params()
        ctx.nftensorparams = len(ftensor_params)
        ctx.nptensorparams = len(ptensor_params)
        ctx.save_for_backward(epf, *ftensor_params, *ptensor_params)

        return epf

    @staticmethod
    def backward(ctx, grad_epf):
        # restore the parameters
        alltensors = ctx.saved_tensors
        nftensorparams = ctx.nftensorparams
        nptensorparams = ctx.nptensorparams
        epf = alltensors[0]
        ftensor_params = alltensors[1:1 + nftensorparams]
        ptensor_params = alltensors[1 + nftensorparams:]
        fptensor_params = alltensors[1:]

        # get the parameters and the object parameters
        nfparams = ctx.nfparams
        npparams = ctx.npparams
        fall_params = ctx.fparam_sep.reconstruct_params(ftensor_params)
        pall_params = ctx.pparam_sep.reconstruct_params(ptensor_params)
        fparams = fall_params[:nfparams]
        fobjparams = fall_params[nfparams:]
        pparams = pall_params[:npparams]
        pobjparams = pall_params[npparams:]

        # get other things from the forward
        ffcn = ctx.ffcn
        log_pfcn = ctx.log_pfcn
        xsamples = ctx.xsamples
        wsamples = ctx.wsamples
        grad_enabled = torch.is_grad_enabled()

        def function_wrap(fcn, param_sep, nparams, x, tensor_params):
            all_params = param_sep.reconstruct_params(tensor_params)
            params = all_params[:nparams]
            objparams = all_params[nparams:]
            with fcn.useobjparams(objparams):
                f = fcn(x, *params)
            return f

        def aug_function(x, *grad_and_fptensor_params):
            local_grad_enabled = torch.is_grad_enabled()
            grad_epf = grad_and_fptensor_params[0]
            epf = grad_and_fptensor_params[1]
            fptensor_params = grad_and_fptensor_params[2:]
            ftensor_params = fptensor_params[:nftensorparams]
            ptensor_params = fptensor_params[nftensorparams:]
            with torch.enable_grad():
                # if graph is constructed, then fptensor_params is a clone of
                # fptensor_params from outside, therefore, it needs to be put
                # in the pure function's objects (that's what function_wrap does)
                if grad_enabled:
                    fout = function_wrap(ffcn, ctx.fparam_sep, nfparams, x, ftensor_params)
                    pout = function_wrap(log_pfcn, ctx.pparam_sep, npparams, x, ptensor_params)
                # if graph is not constructed, then fptensor_params in this
                # function *is* fptensor_params in the outside, so we can
                # just use fparams and pparams from the outside
                else:
                    fout = ffcn(x, *fparams)
                    pout = log_pfcn(x, *pparams)
            # derivative of fparams
            dLdthetaf = []
            if len(ftensor_params) > 0:
                dLdthetaf = torch.autograd.grad(fout, ftensor_params,
                                                grad_outputs=grad_epf,
                                                retain_graph=True,
                                                create_graph=local_grad_enabled)
            # derivative of pparams
            dLdthetap = []
            if len(ptensor_params) > 0:
                dLdef = torch.dot((fout - epf).reshape(-1), grad_epf.reshape(-1))
                dLdthetap = torch.autograd.grad(pout, ptensor_params,
                                                grad_outputs=dLdef.reshape(pout.shape),
                                                retain_graph=True,
                                                create_graph=local_grad_enabled)
            # combine the states needed for backward
            outs = (
                *dLdthetaf,
                *dLdthetap,
            )
            return outs

        if grad_enabled:
            fptensor_params_copy = [y.clone().requires_grad_() for y in fptensor_params]
        else:
            fptensor_params_copy = fptensor_params

        aug_epfs = _mcquad(aug_function, log_pfcn,
                           x0=xsamples[0],  # unused because xsamples is set
                           xsamples=xsamples,
                           wsamples=wsamples,
                           fparams=(grad_epf, epf, *fptensor_params_copy),
                           pparams=pparams,
                           method=ctx.method,
                           bck_options=ctx.bck_config,
                           **ctx.bck_config)
        dLdthetaf = aug_epfs[:nftensorparams]
        dLdthetap = aug_epfs[nftensorparams:]

        # combine the gradient for all fparams
        dLdfnontensor = [None for _ in range(ctx.fparam_sep.nnontensors())]
        dLdpnontensor = [None for _ in range(ctx.pparam_sep.nnontensors())]
        dLdtf = ctx.fparam_sep.reconstruct_params(dLdthetaf, dLdfnontensor)
        dLdtp = ctx.pparam_sep.reconstruct_params(dLdthetap, dLdpnontensor)
        return (None, None, None, None, None, None, None, None, None, None, None,
                *dLdtf, *dLdtp)

def _integrate(ffcn, xsamples, wsamples, fparams):
    nsamples = len(xsamples)
    res = 0.0
    for x, w in zip(xsamples, wsamples):
        res = res + ffcn(x, *fparams) * w
    return res


# docstring completion
mcquad.__doc__ = get_methods_docstr(mcquad, [mh, mhcustom])
