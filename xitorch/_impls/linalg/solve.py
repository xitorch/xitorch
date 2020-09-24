import warnings
from typing import Union
import torch
import numpy as np
from xitorch import LinearOperator
from scipy.sparse.linalg import gmres
from xitorch._impls.optimize.rootfinder import lbfgs, broyden
from xitorch._utils.misc import dummy_context_manager
from xitorch._utils.bcast import normalize_bcast_dims, get_bcasted_dims

def wrap_gmres(A, params, B, E=None, M=None, mparams=[],
        min_eps=1e-9,
        max_niter=None,
        **unused):
    """
    Using SciPy's gmres method to solve the linear equation.

    Keyword arguments
    -----------------
    min_eps: float
        Relative tolerance for stopping conditions
    max_niter: int or None
        Maximum number of iterations. If ``None``, default to twice of the
        number of columns of ``A``.
    """
    # A: (*BA, nr, nr)
    # B: (*BB, nr, ncols)
    # E: (*BE, ncols) or None
    # M: (*BM, nr, nr) or None

    # NOTE: currently only works for batched B (1 batch dim), but unbatched A
    assert len(A.shape) == 2 and len(B.shape) == 3, "Currently only works for batched B (1 batch dim), but unbatched A"

    # check the parameters
    msg = "GMRES can only do AX=B"
    assert A.shape[-2] == A.shape[-1], "GMRES can only work for square operator for now"
    assert E is None, msg
    assert M is None, msg

    nbatch, na, ncols = B.shape
    if max_niter is None:
        max_niter = 2*na

    B = B.transpose(-1,-2) # (nbatch, ncols, na)

    # convert the numpy/scipy
    with A.uselinopparams(*params):
        op = A.scipy_linalg_op()
        B_np = B.detach().numpy()
        res_np = np.empty(B.shape, dtype=np.float64)
        for i in range(nbatch):
            for j in range(ncols):
                x, info = gmres(op, B_np[i,j,:], tol=min_eps, atol=1e-12, maxiter=max_niter)
                if info > 0:
                    msg = "The GMRES iteration does not converge to the desired value "\
                          "(%.3e) after %d iterations" % \
                          (config["min_eps"], info)
                    warnings.warn(msg)
                res_np[i,j,:] = x

        res = torch.tensor(res_np, dtype=B.dtype, device=B.device)
        res = res.transpose(-1,-2) # (nbatch, na, ncols)
        return res

def rootfinder_solve(alg, A, params, B, E=None, M=None, mparams=[], **options):
    # using rootfinder algorithm
    with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
        nr = A.shape[-1]
        ncols = B.shape[-1]

        # set up the function for the rootfinding
        def fcn_rootfinder(xi):
            # xi: (*BX, nr*ncols)
            x = xi.reshape(*xi.shape[:-1], nr, ncols) # (*BX, nr, ncols)
            y = A.mm(x) - B # (*BX, nr, ncols)
            if E is not None:
                MX = M.mm(x) if M is not None else x
                MXE = MX * E.unsqueeze(-2)
                y = y - MXE # (*BX, nr, ncols)
            y = y.reshape(*xi.shape[:-1], -1) # (*BX, nr*ncols)
            return y

        # setup the initial guess (the batch dimension must be the largest)
        batchdims = _get_batchdims(A, B, E, M)
        x0 = torch.zeros((*batchdims, nr*ncols), dtype=A.dtype, device=A.device)

        if alg == "lbfgs":
            x = lbfgs(fcn_rootfinder, x0, **options)
        elif alg == "broyden":
            x = broyden(fcn_rootfinder, x0, **options)
        else:
            raise RuntimeError("Unknown method %s" % alg)
        x = x.reshape(*x.shape[:-1], nr, ncols)
        return x

def exactsolve(A:LinearOperator, B:torch.Tensor,
               E:Union[torch.Tensor,None],
               M:Union[LinearOperator,None]):
    """
    Solve the linear equation by contructing the full matrix of LinearOperators.

    Warnings
    --------
    * As this method construct the linear operators explicitly, it might requires
      a large memory.
    """
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    if E is None:
        Amatrix = A.fullmatrix() # (*BA, na, na)
        x, _ = torch.solve(B, Amatrix) # (*BAB, na, ncols)
    elif M is None:
        Amatrix = A.fullmatrix()
        x = _solve_ABE(Amatrix, B, E)
    else:
        Mmatrix = M.fullmatrix() # (*BM, na, na)
        L = torch.cholesky(Mmatrix, upper=False) # (*BM, na, na)
        Linv = torch.inverse(L) # (*BM, na, na)
        LinvT = Linv.transpose(-2,-1) # (*BM, na, na)
        A2 = torch.matmul(Linv, A.mm(LinvT)) # (*BAM, na, na)
        B2 = torch.matmul(Linv, B) # (*BBM, na, ncols)

        X2 = _solve_ABE(A2, B2, E) # (*BABEM, na, ncols)
        x = torch.matmul(LinvT, X2) # (*BABEM, na, ncols)
    return x

def _solve_ABE(A:torch.Tensor, B:torch.Tensor, E:torch.Tensor):
    # A: (*BA, na, na) matrix
    # B: (*BB, na, ncols) matrix
    # E: (*BE, ncols) matrix
    na = A.shape[-1]
    BA, BB, BE = normalize_bcast_dims(A.shape[:-2], B.shape[:-2], E.shape[:-1])
    E = E.view(1, *BE, E.shape[-1]).transpose(0,-1) # (ncols, *BE, 1)
    B = B.view(1, *BB, *B.shape[-2:]).transpose(0,-1) # (ncols, *BB, na, 1)

    # NOTE: The line below is very inefficient for large na and ncols
    AE = A - torch.diag_embed(E.repeat_interleave(repeats=na, dim=-1), dim1=-2, dim2=-1) # (ncols, *BAE, na, na)
    r, _ = torch.solve(B, AE) # (ncols, *BAEM, na, 1)
    r = r.transpose(0,-1).squeeze(0) # (*BAEM, na, ncols)
    return r

def _get_batchdims(A:LinearOperator, B:torch.Tensor, E:Union[torch.Tensor,None], M:Union[LinearOperator,None]):
    batchdims = [A.shape[:-2], B.shape[:-2]]
    if E is not None:
        batchdims.append(E.shape[:-1])
        if M is not None:
            batchdims.append(M.shape[:-2])
    return get_bcasted_dims(*batchdims)
