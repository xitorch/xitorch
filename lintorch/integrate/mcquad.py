import torch
from lintorch.debug.modes import is_debug_enabled
from lintorch._core.pure_function import get_pure_function, make_sibling
from lintorch._utils.misc import set_default_option, TensorNonTensorSeparator
from lintorch._utils.tupleops import tuple_axpy1
from lintorch._impls.integrate.mcsamples.mh import mh

__all__ = ["mcquad"]

def mcquad(ffcn, log_pfcn, x0, fparams, pparams, fwd_options={}, bck_options={}):
    """
    Performing monte carlo quadrature to calculate the expectation value:

    E[f] = int[f(x,*fparams) g(x,*pparams) dx] / int[g(x,*pparams) dx]

    Arguments
    ---------
    * ffcn: Callable
        The function with to be integrated. Its outputs should be a tensor or a
        list of tensors. To call the function: ffcn(x, *fparams)
    * log_pfcn: Callable
        The natural logarithm of the probability function. The output should be
        one-element tensor. To call the function: gfcn(x, *pparams)
    * x0: torch.Tensor
        Tensor with any size as the initial position.
        The call `ffcn(x0,*fparams)` must work.
    * fparams: list
        List of any other parameters for `ffcn`.
    * pparams: list
        List of any other parameters for `gfcn`.
    * fwd_options: dict
        Options for the forward operation.
    * bck_options: dict
        Options for the backward mcquad operation.

    Returns
    -------
    * epf: torch.Tensor or a list of torch.Tensor
        The expectation values of the function `ffcn` over the space of `x`.
        If the output of `ffcn` is a list, then this is also a list
    """
    return _mcquad(ffcn, log_pfcn, x0, None, fparams, pparams, fwd_options, bck_options)

def _mcquad(ffcn, log_pfcn, x0, xsamples, fparams, pparams, fwd_options, bck_options):
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

    if not is_tuple_out:
        @make_sibling(pure_ffcn)
        def pure_ffcn2(x, *fparams):
            return (pure_ffcn(x, *fparams),)
        return _MCQuad.apply(pure_ffcn2, pure_logpfcn, 1, x0, None, fwd_options, bck_options,
            nfparams, nf_objparams, npparams, *fparams, *fobjparams, *pparams, *pobjparams)[0]
    else:
        nf = len(out)
        return _MCQuad.apply(pure_ffcn, pure_logpfcn, nf, x0, None, fwd_options, bck_options,
            nfparams, nf_objparams, npparams, *fparams, *fobjparams, *pparams, *pobjparams)

class _MCQuad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ffcn, log_pfcn, nf, x0, xsamples, fwd_options, bck_options,
            nfparams, nf_objparams, npparams, *all_fpparams):
        # set up the default options
        config = set_default_option({
            "method": "mh",
        }, fwd_options)
        ctx.bck_config = set_default_option(config, bck_options)

        # split the parameters
        fparams    = all_fpparams[:nfparams]
        fobjparams = all_fpparams[nfparams:nfparams+nf_objparams]
        pparams    = all_fpparams[nfparams+nf_objparams:nfparams+nf_objparams+npparams]
        pobjparams = all_fpparams[nfparams+nf_objparams+npparams:]

        # select the method for the sampling
        if xsamples is None:
            method = config["method"].lower()
            method_fcn = {
                "mh": mh,
                # "mhcustom": mhcustom_mcquad,
            }
            if method not in method_fcn:
                raise RuntimeError("Unknown mcquad method: %s" % config["method"])
            xsamples = method_fcn[method](log_pfcn, x0, pparams, **config)
        epfs = _integrate(ffcn, xsamples, fparams, nf)

        # save parameters for backward calculations
        ctx.xsamples = xsamples
        ctx.ffcn = ffcn
        ctx.log_pfcn = log_pfcn
        ctx.fparam_sep = TensorNonTensorSeparator((*fparams, *fobjparams))
        ctx.pparam_sep = TensorNonTensorSeparator((*pparams, *pobjparams))
        ctx.nfparams = len(fparams)
        ctx.npparams = len(pparams)

        # save for backward
        ftensor_params = ctx.fparam_sep.get_tensor_params()
        ptensor_params = ctx.pparam_sep.get_tensor_params()
        ctx.nftensorparams = len(ftensor_params)
        ctx.nptensorparams = len(ptensor_params)
        ctx.nout = len(epfs)
        ctx.save_for_backward(*epfs, *ftensor_params, *ptensor_params)

        return tuple(epfs)

    @staticmethod
    def backward(ctx, *grad_epfs):
        # restore the parameters
        alltensors = ctx.saved_tensors
        nout = ctx.nout
        nftensorparams = ctx.nftensorparams
        nptensorparams = ctx.nptensorparams
        epfs = alltensors[:nout]
        ftensor_params = alltensors[nout:nout+nftensorparams]
        ptensor_params = alltensors[nout+nftensorparams:]
        fptensor_params = alltensors[nout:]

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
            grad_epfs = grad_and_fptensor_params[:nout]
            epfs = grad_and_fptensor_params[nout:2*nout]
            fptensor_params = grad_and_fptensor_params[2*nout:]
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
                    grad_outputs=grad_epfs,
                    retain_graph=True,
                    create_graph=local_grad_enabled)
            # derivative of pparams
            dLdthetap = []
            if len(ptensor_params) > 0:
                dLdef = sum([torch.dot((f-y).reshape(-1), grad_epf.reshape(-1)) for (f, y, grad_epf) in zip(fout, epfs, grad_epfs)])
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
            x0=xsamples[0], # unused because xsamples is set
            xsamples=xsamples,
            fparams=(*grad_epfs, *epfs, *fptensor_params_copy),
            pparams=pparams,
            fwd_options=ctx.bck_config,
            bck_options=ctx.bck_config)
        dLdthetaf = aug_epfs[:nftensorparams]
        dLdthetap = aug_epfs[nftensorparams:]

        # combine the gradient for all fparams
        dLdfnontensor = [None for _ in range(ctx.fparam_sep.nnontensors())]
        dLdpnontensor = [None for _ in range(ctx.pparam_sep.nnontensors())]
        dLdtf = ctx.fparam_sep.reconstruct_params(dLdthetaf, dLdfnontensor)
        dLdtp = ctx.pparam_sep.reconstruct_params(dLdthetap, dLdpnontensor)
        return (None, None, None, None, None, None, None, None, None, None,
                *dLdtf, *dLdtp)

def _integrate(ffcn, xsamples, fparams, nf):
    nsamples = len(xsamples)
    sumfs = [0.0 for _ in range(nf)]
    for x in xsamples:
        sumfs = [s + f for s,f in zip(sumfs, ffcn(x, *fparams))]
    meanfs = [sumf / nsamples for sumf in sumfs]
    return meanfs
