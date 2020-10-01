import torch
import warnings
from typing import Union, Any, Mapping, Optional
import numpy as np
from xitorch import LinearOperator
from xitorch._utils.assertfuncs import assert_runtime
from xitorch._utils.misc import set_default_option, dummy_context_manager
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch.debug.modes import is_debug_enabled
from xitorch._impls.linalg.solve import exactsolve, wrap_gmres, \
    cg, broyden1_solve, _get_batchdims

def solve(A:LinearOperator, B:torch.Tensor, E:Union[torch.Tensor,None]=None,
          M:Optional[LinearOperator]=None,
          bck_options:Mapping[str,Any]={},
          method:Optional[str]=None,
          **fwd_options) -> torch.Tensor:
    r"""
    Performing iterative method to solve the equation

    .. math::

        \mathbf{AX=B}

    or

    .. math::

        \mathbf{AX-MXE=B}

    where :math:`\mathbf{E}` is a diagonal matrix.
    This function can also solve batched multiple inverse equation at the
    same time by applying :math:`\mathbf{A}` to a tensor :math:`\mathbf{X}`
    with shape ``(...,na,ncols)``.
    The applied :math:`\mathbf{E}` are not necessarily identical for each column.

    Arguments
    ---------
    A: xitorch.LinearOperator
        A linear operator that takes an input ``X`` and produce the vectors in the same
        space as ``B``.
        It should have the shape of ``(*BA, na, na)``
    B: torch.Tensor
        The tensor on the right hand side with shape ``(*BB, na, ncols)``
    E: torch.Tensor or None
        If a tensor, it will solve :math:`\mathbf{AX-MXE = B}`.
        It will be regarded as the diagonal of the matrix.
        Otherwise, it just solves :math:`\mathbf{AX = B}` and ``M`` is ignored.
        If it is a tensor, it should have shape of ``(*BE, ncols)``.
    M: xitorch.LinearOperator or None
        The transformation on the ``E`` side. If ``E`` is ``None``,
        then this argument is ignored.
        If E is not ``None`` and ``M`` is ``None``, then ``M=I``.
        If LinearOperator, it must be Hermitian with shape ``(*BM, na, na)``.
    bck_options: dict
        Options of the iterative solver in the backward calculation.
    method: str or None
        The method of linear equation solver. If ``None``, it will choose
        ``"exactsolve"`` or ``"broyden1"`` depending on the size
    **fwd_options
        Method-specific options (see method below)

    Returns
    -------
    torch.Tensor
        The tensor :math:`\mathbf{X}` that satisfies :math:`\mathbf{AX-MXE=B}`.
    """
    assert_runtime(A.shape[-1] == A.shape[-2], "The linear operator A must have a square shape")
    assert_runtime(A.shape[-1] == B.shape[-2], "Mismatch shape of A & B (A: %s, B: %s)" % (A.shape, B.shape))
    if M is not None:
        assert_runtime(M.shape[-1] == M.shape[-2], "The linear operator M must have a square shape")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))
        assert_runtime(M.is_hermitian, "The linear operator M must be a Hermitian matrix")
    if E is not None:
        assert_runtime(E.shape[-1] == B.shape[-1], "The last dimension of E & B must match (E: %s, B: %s)" % (E.shape, B.shape))
    if E is None and M is not None:
        warnings.warn("M is supplied but will be ignored because E is not supplied")

    # perform expensive check if debug mode is enabled
    if is_debug_enabled():
        A.check()
        if M is not None:
            M.check()

    if method is None:
        method = "exactsolve" if A.shape[-1] < 10 else "broyden1"

    if method == "exactsolve":
        return exactsolve(A, B, E, M)
    else:
        # get the unique parameters of A
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return solve_torchfcn.apply(
            A, B, E, M, method,
            fwd_options, bck_options,
            na, *params, *mparams)

class solve_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, E, M, method,
                fwd_options, bck_options,
                na, *all_params):
        # A: (*BA, nr, nr)
        # B: (*BB, nr, ncols)
        # E: (*BE, ncols) or None
        # M: (*BM, nr, nr) or None
        # all_params: list of tensor of any shape
        # returns: (*BABEM, nr, ncols)

        # separate the parameters for A and for M
        params = all_params[:na]
        mparams = all_params[na:]

        config = set_default_option({
        }, fwd_options)
        ctx.bck_config = set_default_option({
        }, bck_options)

        if torch.all(B == 0): # special case
            dims = (*_get_batchdims(A, B, E, M), *B.shape[-2:])
            x = torch.zeros(dims, dtype=B.dtype, device=B.device)
        elif method == "custom_exactsolve":
            x = custom_exactsolve(A, params, B, E=E, M=M, mparams=mparams, **config)
        elif method == "scipy_gmres":
            x = wrap_gmres(A, params, B, E=E, M=M, mparams=mparams, **config)
        elif method == "broyden1":
            x = broyden1_solve(A, params, B, E=E, M=M, mparams=mparams, **config)
        elif method == "cg":
            x = cg(A, params, B, E=E, M=M, mparams=mparams, **config)
        else:
            raise RuntimeError("Unknown solve method: %s" % method)

        ctx.A = A
        ctx.M = M
        ctx.E = E
        ctx.x = x
        ctx.params = params
        ctx.mparams = mparams
        ctx.na = na
        return x

    @staticmethod
    def backward(ctx, grad_x):
        # grad_x: (*BABEM, nr, ncols)
        # ctx.x: (*BABEM, nr, ncols)

        # solve (A-biases*M)^T v = grad_x
        # this is the grad of B
        AT = ctx.A.H # (*BA, nr, nr)
        MT = ctx.M.H if ctx.M is not None else None # (*BM, nr, nr)
        with AT.uselinopparams(*ctx.params), MT.uselinopparams(*ctx.mparams) if MT is not None else dummy_context_manager():
            v = solve(AT, grad_x, ctx.E, MT,
                fwd_options=ctx.bck_config, bck_options=ctx.bck_config) # (*BABEM, nr, ncols)
        grad_B = v

        # calculate the grad of matrices parameters
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in ctx.params]
            with ctx.A.uselinopparams(*params):
                loss = -ctx.A.mm(ctx.x) # (*BABEM, nr, ncols)

        grad_params = torch.autograd.grad((loss,), params, grad_outputs=(v,),
            create_graph=torch.is_grad_enabled())

        # calculate the biases gradient
        grad_E = None
        if ctx.E is not None:
            if ctx.M is None:
                Mx = ctx.x
            else:
                with ctx.M.uselinopparams(*ctx.mparams):
                    Mx = ctx.M.mm(ctx.x) # (*BABEM, nr, ncols)
            grad_E = torch.einsum('...rc,...rc->...c', v, Mx) # (*BABEM, ncols)

        # calculate the gradient to the biases matrices
        grad_mparams = []
        if ctx.M is not None and ctx.E is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in ctx.mparams]
                lmbdax = ctx.x * ctx.E.unsqueeze(-2)
                with ctx.M.uselinopparams(*mparams):
                    mloss = ctx.M.mm(lmbdax)

            grad_mparams = torch.autograd.grad((mloss,), mparams,
                grad_outputs=(v,),
                create_graph=torch.is_grad_enabled())

        return (None, grad_B, grad_E, None, None, None, None, None,
                *grad_params, *grad_mparams)

def custom_exactsolve(A, params, B, E=None,
                M=None, mparams=[], **options):
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
        return exactsolve(A, B, E, M)

# docstring completion
_solve_methods = {
    "broyden1": broyden1_solve,
    "exactsolve": exactsolve,
    "scipy_gmres": wrap_gmres,
    "cg": cg,
}
ignore_kwargs = ["E", "M", "mparams"]
solve.__doc__ = get_methods_docstr(solve, _solve_methods, ignore_kwargs)
