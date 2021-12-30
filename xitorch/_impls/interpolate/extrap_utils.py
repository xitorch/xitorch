import torch

def get_extrap_pos(xqextrap, extrap, xmin=0.0, xmax=1.0):
    # xqextrap: (nrq,)
    xqnorm = (xqextrap - xmin) / (xmax - xmin)
    if extrap == "periodic":
        xqinside = xqnorm % 1.0
    elif extrap == "mirror":
        xqnorm = xqnorm.abs()
        xqnorm_ceil = xqnorm.long() + 1
        xqhalf = torch.div(xqnorm_ceil, 2, rounding_mode="trunc")
        xqinside = (2 * xqhalf - xqnorm) * (1 - (xqnorm_ceil % 2.0) * 2)
    elif extrap == "bound":
        xqinside = torch.clamp(xqnorm, 0.0, 1.0)
    else:
        raise RuntimeError("get_extrap_pos only work for periodic and mirror extrapolation")
    return xqinside * (xmax - xmin) + xmin

def get_extrap_val(xqextrap, y, extrap):
    # xqextrap: (nrq,)
    # y: (*BY, nr)
    shape = (*y.shape[:-1], xqextrap.shape[-1])
    dtype = xqextrap.dtype
    device = xqextrap.device

    if extrap is None or extrap == "nan":
        return torch.empty(shape, dtype=dtype, device=device) * float("nan")
    elif isinstance(extrap, int) or isinstance(extrap, float) or \
            (isinstance(extrap, torch.Tensor) and torch.numel(extrap) == 1):
        return torch.zeros(shape, dtype=dtype, device=device) + extrap
    elif hasattr(extrap, "__call__"):
        return extrap(xqextrap).expand(*y.shape[:-1], -1)  # (*BY, nrq)
    else:
        raise RuntimeError("Invalid extrap type (type: %s): %s" % (type(extrap), extrap))
