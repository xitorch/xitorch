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
