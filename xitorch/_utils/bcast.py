from typing import Tuple
import torch

def normalize_bcast_dims(*shapes):
    """
    Normalize the lengths of the input shapes to have the same length.
    The shapes are padded at the front by 1 to make the lengths equal.
    """
    maxlens = max([len(shape) for shape in shapes])
    res = [[1] * (maxlens - len(shape)) + list(shape) for shape in shapes]
    return res

def get_bcasted_dims(*shapes):
    """
    Return the broadcasted shape of the given shapes.
    """
    shapes = normalize_bcast_dims(*shapes)
    return [max(*a) for a in zip(*shapes)]

def match_dim(*xs: torch.Tensor, contiguous: bool = False) -> Tuple[torch.Tensor, ...]:
    # match the N-1 dimensions of x and xq for searchsorted and gather with dim=-1
    orig_shapes = tuple(x.shape[:-1] for x in xs)
    shape = tuple(get_bcasted_dims(*orig_shapes))
    xs_new = tuple(x.expand(shape + (x.shape[-1],)) for x in xs)
    if contiguous:
        xs_new = tuple(x.contiguous() for x in xs_new)
    return xs_new
