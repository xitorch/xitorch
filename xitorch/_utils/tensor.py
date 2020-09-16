import torch

"""
This file contains functions for some linear algebra and basic operations of
torch.tensor.
"""

def tallqr(V, MV=None):
    # faster QR for tall and skinny matrix
    # V: (*BV, na, nguess)
    # MV: (*BM, na, nguess) where M is the basis to make Q M-orthogonal
    # if MV is None, then MV=V
    if MV is None:
        MV = V
    VTV = torch.matmul(V.transpose(-2,-1), MV) # (*BMV, nguess, nguess)
    R = torch.cholesky(VTV, upper=True) # (*BMV, nguess, nguess)
    Rinv = torch.inverse(R) # (*BMV, nguess, nguess)
    Q = torch.matmul(V, Rinv)
    return Q, R

def to_fortran_order(V):
    # V: (...,nrow,ncol)
    # outV: (...,nrow,ncol)

    # check if it is in C-contiguous
    if V.is_contiguous():
        # return V.set_(V.storage(), V.storage_offset(), V.size(), tuple(reversed(V.stride())))
        return V.transpose(-2,-1).contiguous().transpose(-2,-1)
    elif V.transpose(-2,-1).is_contiguous():
        return V
    else:
        raise RuntimeError("Only the last two dimensions can be made Fortran order.")

def ortho(A, B, dim=-2, M=None, mright=True):
    """
    Orthogonalize each column in matrix A w.r.t. matrix B in M basis.
    M is a LinearOperator.
    """
    if M is None:
        return A - (A * B).sum(dim=dim, keepdim=True) * B
    elif mright:
        return A - (M.mm(A) * B).sum(dim=dim, keepdim=True) * B
    else:
        return A - M.mm((A * B).sum(dim=dim, keepdim=True) * B)

def convert_none_grads_to_zeros(grads, inputs):
    is_tuple = isinstance(grads, tuple)
    if is_tuple:
        grads = list(grads)
    for i in range(len(grads)):
        g = grads[i]
        if g is None:
            grads[i] = torch.zeros_like(inputs[i])
    if is_tuple:
        grads = tuple(grads)
    return grads
