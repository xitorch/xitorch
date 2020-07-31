import torch
import warnings
from typing import Union, Any, Mapping
from lintorch.linop.base import LinearOperator
from lintorch.utils.bcast import normalize_bcast_dims
from lintorch.utils.debug import assert_runtime

def solve(A:LinearOperator, B:torch.Tensor, E:Union[torch.Tensor,None]=None,
          M:Union[LinearOperator,None]=None,
          fwd_options:Mapping[str,Any]={},
          bck_options:Mapping[str,Any]={}):
    """
    Performing iterative method to solve the equation AX=B or
    AX-MXE=B, where E is a diagonal matrix.
    This function can also solve batched multiple inverse equation at the
    same time by applying A to a tensor X with shape (...,na,ncols).
    The applied E are not necessarily identical for each column.

    Arguments
    ---------
    * A: lintorch.LinearOperator instance with shape (*BA, na, na)
        A function that takes an input X and produce the vectors in the same
        space as B.
    * B: torch.tensor (*BB, na, ncols)
        The tensor on the right hand side.
    * E: torch.tensor (*BE, ncols) or None
        If not None, it will solve AX-MXE = B. Otherwise, it just solves
        AX = B and M is ignored. E would be applied to every column.
    * M: lintorch.LinearOperator instance (*BM, na, na) or None
        The transformation on the E side. If E is None,
        then this argument is ignored. I E is not None and M is None, then M=I.
    * fwd_options: dict
        Options of the iterative solver in the forward calculation
    * bck_options: dict
        Options of the iterative solver in the backward calculation
    """
    assert_runtime(A.shape[-1] == A.shape[-2], "The linear operator A must have a square shape")
    assert_runtime(A.shape[-1] == B.shape[-2], "Mismatch shape of A & B (A: %s, B: %s)" % (A.shape, B.shape))
    if M is not None:
        assert_runtime(M.shape[-1] == M.shape[-2], "The linear operator M must have a square shape")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))
    if E is not None:
        assert_runtime(E.shape[-1] == B.shape[-1], "The last dimension of E & B must match (E: %s, B: %s)" % (E.shape, B.shape))
    if E is None and M is not None:
        warnings.warn("M is supplied but will be ignored because E is not supplied")

    if "method" not in fwd_options or fwd_options["method"].lower() == "exactsolve":
        return exactsolve(A, B, E, M)
    else:
        raise RuntimeError("Method other than exactsolve has not been implemented")

def exactsolve(A:LinearOperator, B:torch.Tensor,
               E:Union[torch.Tensor,None],
               M:Union[LinearOperator,None]):
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
