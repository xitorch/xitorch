import torch
import numpy as np
from lintorch.utils.misc import set_default_option
from lintorch.utils.tensor import ortho
from lintorch.fcns.solve import solve_torchfcn

__all__ = ["solve2"]

def solve2(A, params, B, eigvals, eigvecs, cpar, M=None, mparams=[],
           fwd_options={}, bck_options={}):
    """
    Solve2 is solving the equation `(A-eigvals*M)y = B` where `eigvals` are the
    eigenvalues of matrix A with right-hand-side operator M
    (i.e. `A*x = lmbda*M*x`).
    Due to infinite solution is possible, then one need one more condition
    to be satisfied, which is `x*x^T*M*y = cpar` with `x` being the
    eigenvector of the corresponding eigenvalues.

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
    * cpar: torch.tensor (nbatch, na, neig)
        The parallel component, i.e. x*x^T*M*y = cpar
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
    return solve2_torchfcn.apply(A, B, eigvals, eigvecs, cpar, M, fwd_options, bck_options, na, *params, *mparams)

class solve2_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, eigvals, eigvecs, cpar, M, fwd_options, bck_options, na, *amparams):
        # B: (nbatch, na, neig)
        # eigvals: (nbatch, neig)
        # eigvecs: (nbatch, na, neig)
        # cpar: (nbatch, na, neig)
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
        ypar = cpar
        ysol = ypar + yortho

        # save the parameters for backward operation
        ctx.A = A
        ctx.M = M
        ctx.eigvals = eigvals
        ctx.eigvecs = eigvecs
        ctx.cpar = cpar
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
                       B=grad_ysol_ortho, cpar=grad_ysol_par,
                       eigvals=ctx.eigvals, eigvecs=ctx.eigvecs,
                       M=ctx.M, mparams=ctx.mparams,
                       fwd_options=ctx.bck_options, bck_options=ctx.bck_options)
        grad_B = ortho(dLdbc, ctx.eigvecs, dim=-2, M=ctx.M, mparams=ctx.mparams, mright=True)
        grad_cpar = dLdbc - grad_B

        grad_eigvals = None
        grad_eigvecs = None
        grad_params = [None] * len(ctx.params)
        grad_mparams = [None] * len(ctx.mparams)

        return (None, grad_B, grad_eigvals, grad_eigvecs, grad_cpar,
                None, None, None, None,
                *grad_params, *grad_mparams)

if __name__ == "__main__":
    import time
    from lintorch.fcns.lsymeig import lsymeig
    from lintorch.utils.fd import finite_differences
    from lintorch.core.base import Module

    # generate the matrix
    na = 20
    dtype = torch.float64
    torch.manual_seed(123)
    A1 = (torch.rand((1,na,na))*0.1).to(dtype).requires_grad_(True)
    diag = (torch.arange(na, dtype=dtype)+1.0).unsqueeze(0).requires_grad_(True)
    M1 = (torch.rand((1,na,na))*0.1).to(dtype).requires_grad_(True)
    mdiag = (torch.arange(na, dtype=dtype)+1.0).unsqueeze(0).requires_grad_(True)

    class Acls(Module):
        def __init__(self):
            super(Acls, self).__init__(shape=(na,na))

        def forward(self, x, A1, diag):
            Amatrix = (A1 + A1.transpose(-2,-1))
            A = Amatrix + diag.diag_embed(dim1=-2, dim2=-1)
            y = torch.bmm(A, x)
            return y

        def precond(self, y, A1, dg, biases=None, M=None, mparams=None):
            # return y
            # y: (nbatch, na, ncols)
            # dg: (nbatch, na)
            # biases: (nbatch, ncols) or None
            Adiag = A1.diagonal(dim1=-2, dim2=-1) * 2
            dd = (Adiag + dg).unsqueeze(-1)

            if biases is not None:
                if M is not None:
                    M1, Mdg = mparams
                    Mdiag = M1.diagonal(dim1=-2, dim2=-1) * 2
                    md = (Mdiag + Mdg).unsqueeze(-1)
                    dd = dd - biases.unsqueeze(1) * md
                else:
                    dd = dd - biases.unsqueeze(1) # (nbatch, na, ncols)
            dd[dd.abs() < 1e-6] = 1.0
            yprec = y / dd
            return yprec

    A = Acls().to(dtype)
    M = Acls().to(dtype)
    neig = 4
    options = {
        "method": "exacteig",
        "verbose": False,
        "nguess": neig,
        "v_init": "randn",
    }
    bck_options = {
        "verbose": True,
        "min_eps": 1e-9,
    }
    evals, evecs = lsymeig(A, (A1,diag), neig, M, (M1,mdiag))
    cpar = evecs * 0

    xtrue = torch.randn((1,na,neig)).to(dtype)
    B = A(xtrue, A1, diag) - M(xtrue, M1, mdiag) * evals.unsqueeze(1)
    cpar = (M(xtrue, M1, mdiag) * evecs).sum(dim=1, keepdim=True) * evecs
    x = solve2(A, (A1, diag), B, evals, evecs, cpar, M=M, mparams=[M1, mdiag])
    print(x/xtrue)
