import torch
import numpy as np
from lintorch.utils.misc import set_default_option
from lintorch.utils.tensor import ortho
from lintorch.fcns.solve import solve_torchfcn

__all__ = ["solve2"]

def solve2(A, params, B, eigvals, eigvecs, c, M=None, mparams=[],
           fwd_options={}, bck_options={}):
    """
    Solve2 is solving the equation `(A-eigvals*M)y = B` where `eigvals` are the
    eigenvalues of matrix A with right-hand-side operator M
    (i.e. `A*x = lmbda*M*x`).
    Due to infinite solution is possible, then one need one more condition
    to be satisfied, which is `x^T*M*y = c` with `x` being the eigenvector of
    the corresponding eigenvalues.

    Arguments
    ---------
    * A: lintorch.Module
        The lintorch module
    * params: list of torch.tensor (nbatch, ...)
        List of parameters for the module A
    * B: torch.tensor (nbatch, na, neig)
        The tensor on the right hand side of the solve equation.
    * eigvals: torch.tensor (nbatch, neig)
        The eigenvalues of the matrix A & M
    * eigvecs: torch.tensor (nbatch, na, neig)
        The eigenvectors of the matrix A & M
    * c: torch.tensor (nbatch, neig)
        The parallel component, i.e. x^T*M*y = c
    * M: lintorch.Module
        The operator on the right hand side of the generalized
        eigendecomposition equation.
        If None, it is an identity.
    * mparams: list of torch.tensor (nbatch, ...)
        The list of parameters for M
    * fwd_options: dict
        Options of the iterative solver in the forward calculation
    * bck_options: dict
        Options of the iterative solver in the backward calculation

    Returns
    -------
    * y: torch.tensor (nbatch, na, neig)
        The tensor solution of the equation `(A-eigvals*M)y = B` and `x^T*M*y=c`
    """
    na = len(params)
    return solve2_torchfcn(A, B, eigvals, eigvecs, c, M, fwd_options, bck_options, na, *params, *mparams)

class solve2_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, eigvals, eigvecs, c, M, fwd_options, bck_options, na, *amparams):
        # B: (nbatch, na, neig)
        # eigvals: (nbatch, neig)
        # eigvecs: (nbatch, na, neig)
        # c: (nbatch, neig)
        # amparams: (nbatch, ...)

        # split the parameters
        params = amparams[:na]
        mparams = amparams[na:]

        # using solve to get the orthogonal component of the solution
        # yortho: (nbatch, na, neig)
        yortho = solve_torchfcn.apply(A, B, eigvals, M, False, fwd_options, bck_options, na, *amparams)
        # orthogonalize
        yortho = ortho(yortho, eigvecs, dim=-2, M=M, mparams=mparams, mright=True)

        # get the parallel part and sum them
        ypar = eigvecs * c.unsqueeze(1)
        ysol = ypar + yortho

        # save the parameters for backward operation
        ctx.A = A
        ctx.M = M
        ctx.eigvals = eigvals
        ctx.eigvecs = eigvecs
        ctx.c = c
        ctx.bck_options = bck_options
        ctx.params = params
        ctx.mparams = mparams
        ctx.y = ysol

        return ysol

    @staticmethod
    def backward(ctx, grad_ysol):
        grad_ysol_ortho = ortho(grad_ysol, ctx.eigvecs, dim=-2, M=ctx.M, mparams=ctx.mparams, mright=False)
        grad_ysol_par = grad_ysol - grad_ysol_ortho
        #dLdbc: (nbatch, na, neig)
        dLdbc = solve2(A=ctx.A, params=ctx.params,
                       B=grad_ysol_ortho, c=grad_ysol_par,
                       eigvals=ctx.eigvals, eigvecs=ctx.eigvecs,
                       M=ctx.M, mparams=ctx.mparams,
                       fwd_options=ctx.bck_options, bck_options=ctx.bck_options)
        grad_B = ortho(dLdbc, ctx.eigvecs, dim=-2, M=ctx.M, mparams=ctx.mparams, mright=True)
        pass
