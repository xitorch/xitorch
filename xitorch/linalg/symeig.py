import torch
from typing import Union, Mapping, Any
import functools
from xitorch import LinearOperator
from xitorch.linalg.solve import solve
from xitorch.debug.modes import is_debug_enabled
from xitorch._utils.assertfuncs import assert_runtime
from xitorch._utils.misc import set_default_option, dummy_context_manager
from xitorch._utils.tensor import ortho
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch._impls.linalg.symeig import exacteig, davidson

__all__ = ["lsymeig", "usymeig", "symeig"]

def lsymeig(A:LinearOperator, neig:Union[int,None]=None,
        M:Union[LinearOperator,None]=None,
        bck_options:Mapping[str,Any]={},
        **fwd_options):
    return symeig(A, neig, "lowest", M, **fwd_options, bck_options=bck_options)

def usymeig(A:LinearOperator, neig:Union[int,None]=None,
        M:Union[LinearOperator,None]=None,
        bck_options:Mapping[str,Any]={},
        method:Union[str,None]=None,
        **fwd_options):
    return symeig(A, neig, "uppest", M, **fwd_options, bck_options=bck_options)

def symeig(A:LinearOperator, neig:Union[int,None]=None,
        mode:str="lowest", M:Union[LinearOperator,None]=None,
        bck_options:Mapping[str,Any]={},
        method:Union[str,None]=None,
        **fwd_options):
    """
    Obtain ``neig`` lowest eigenvalues and eigenvectors of a linear operator,

    .. math::

        \mathbf{AX = MXE}

    where :math:`\mathbf{A}, \mathbf{M}` are linear operators,
    :math:`\mathbf{E}` is a diagonal matrix containing the eigenvalues, and
    :math:`\mathbf{X}` is a matrix containing the eigenvectors.

    Arguments
    ---------
    A: xitorch.LinearOperator
        The linear operator object on which the eigenpairs are constructed.
        It must be a Hermitian linear operator with shape ``(*BA, q, q)``
    neig: int or None
        The number of eigenpairs to be retrieved. If ``None``, all eigenpairs are
        retrieved
    mode: str
        ``"lowest"`` or ``"uppermost"``/``"uppest"``. If ``"lowest"``,
        it will take the lowest ``neig`` eigenpairs.
        If ``"uppest"``, it will take the uppermost ``neig``.
    M: xitorch.LinearOperator
        The transformation on the right hand side. If ``None``, then ``M=I``.
        If specified, it must be a Hermitian with shape ``(*BM, q, q)``.
    bck_options: dict
        Method-specific options for :func:`solve` which used in backpropagation
        calculation.
    method: str or None
        Method for the eigendecomposition. If None, it will choose exacteig.
    **fwd_options
        Method-specific options (see method section below).

    Returns
    -------
    tuple of tensors
        It will return eigenvalues and eigenvectors with shapes respectively
        ``(*BAM, neig)`` and ``(*BAM, na, neig)``, where ``*BAM`` is the
        broadcasted shape of ``*BA`` and ``*BM``.
    """
    assert_runtime(A.is_hermitian, "The linear operator A must be Hermitian")
    if M is not None:
        assert_runtime(M.is_hermitian, "The linear operator M must be Hermitian")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))
    mode = mode.lower()
    if mode == "uppermost":
        mode = "uppest"
    if method is None: # TODO: do a proper method selection based on size
        method = "exacteig"

    # perform expensive check if debug mode is enabled
    if is_debug_enabled():
        A.check()
        if M is not None:
            M.check()

    if method == "exacteig":
        return exacteig(A, neig, mode, M)
    else:
        fwd_options["method"] = method
        # get the unique parameters of A & M
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return symeig_torchfcn.apply(A, neig, mode, M,
            fwd_options, bck_options,
            na, *params, *mparams)

class symeig_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, neig, mode, M, fwd_options, bck_options, na, *amparams):
        # A: LinearOperator (*BA, q, q)
        # M: LinearOperator (*BM, q, q) or None

        # separate the sets of parameters
        params = amparams[:na]
        mparams = amparams[na:]

        config = set_default_option({
        }, fwd_options)
        ctx.bck_config = set_default_option({
            # "method": ???
        }, bck_options)

        method = config["method"].lower()
        if method == "davidson":
            evals, evecs = davidson(A, params, neig, mode, M, mparams, **config)
        elif method == "custom_exacteig":
            evals, evecs = custom_exacteig(A, params, neig, mode, M, mparams, **config)
        else:
            raise RuntimeError("Unknown eigen decomposition method: %s" % config["method"])

        # save for the backward
        ctx.evals = evals # (*BAM, neig)
        ctx.evecs = evecs # (*BAM, na, neig)
        ctx.params = params
        ctx.A = A
        ctx.M = M
        ctx.mparams = mparams
        return evals, evecs

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        # grad_evals: (*BAM, neig)
        # grad_evecs: (*BAM, na, neig)

        # get the variables from ctx
        evals = ctx.evals
        evecs = ctx.evecs
        M = ctx.M
        A = ctx.A

        # the loss function where the gradient will be retrieved
        # warnings: if not all params have the connection to the output of A,
        # it could cause an infinite loop because pytorch will keep looking
        # for the *params node and propagate further backward via the `evecs`
        # path. So make sure all the *params are all connected in the graph.
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in ctx.params]
            with A.uselinopparams(*params):
                loss = A.mm(evecs) # (*BAM, na, neig)

        # calculate the contributions from the eigenvalues
        gevalsA = grad_evals.unsqueeze(-2) * evecs # (*BAM, na, neig)

        # calculate the contributions from the eigenvectors
        with M.uselinopparams(*ctx.mparams) if M is not None else dummy_context_manager():
            # orthogonalize the grad_evecs with evecs
            B = ortho(grad_evecs, evecs, dim=-2, M=M, mright=False)
            with A.uselinopparams(*ctx.params):
                gevecs = solve(A, -B, evals, M, fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
            # orthogonalize gevecs w.r.t. evecs
            gevecsA = ortho(gevecs, evecs, dim=-2, M=M, mright=True)

        # accummulate the gradient contributions
        gaccumA = gevalsA + gevecsA
        grad_params = torch.autograd.grad(
            outputs=(loss,),
            inputs=params,
            grad_outputs=(gaccumA,),
            create_graph=torch.is_grad_enabled(),
        )

        grad_mparams = []
        if ctx.M is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in ctx.mparams]
                with M.uselinopparams(*mparams):
                    mloss = M.mm(evecs) # (*BAM, na, neig)
            gevalsM = -gevalsA * evals.unsqueeze(-2)
            gevecsM = -gevecsA * evals.unsqueeze(-2)

            # the contribution from the parallel elements
            gevecsM_par = (-0.5 * torch.einsum("...ae,...ae->...e", grad_evecs, evecs)).unsqueeze(-2) * evecs # (*BAM, na, neig)

            gaccumM = gevalsM + gevecsM + gevecsM_par
            grad_mparams = torch.autograd.grad(
                outputs=(mloss,),
                inputs=mparams,
                grad_outputs=(gaccumM,),
                create_graph=torch.is_grad_enabled(),
            )

        return (None, None, None, None, None, None, None, *grad_params, *grad_mparams)

def custom_exacteig(A, params, neig, mode, M=None, mparams=[], **options):
    with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
        return exacteig(A, neig, mode, M)

# docstring completion
_symeig_methods = {
    "exacteig": exacteig,
    "davidson": davidson,
}
ignore_kwargs = ["M", "mparams"]
symeig.__doc__ = get_methods_docstr(symeig, _symeig_methods, ignore_kwargs)
