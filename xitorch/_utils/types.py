import torch
import numpy as np

__all__ = ["get_np_dtype"]

def get_np_dtype(dtype: torch.dtype):
    # return the corresponding numpy dtype from the input pytorch's tensor dtype
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.float64:
        return np.float64
    elif dtype == torch.complex64:
        return np.complex64
    elif dtype == torch.complex128:
        return np.complex128
    else:
        raise TypeError("Unknown type: %s" % dtype)
