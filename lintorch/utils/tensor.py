import torch

"""
This file contains functions for some linear algebra and basic operations of
torch.tensor.
"""

def tallqr(V):
    # faster QR for tall and skinny matrix
    # V: (nbatch, na, nguess)
    VTV = torch.bmm(V.transpose(-2,-1), V) # (nbatch, nguess, nguess)
    R = torch.cholesky(VTV, upper=True) # (nbatch, nguess, nguess)
    Rinv = torch.inverse(R) # (nbatch, nguess, nguess)
    Q = torch.bmm(V, Rinv)
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
