def tuple_axpy1(a, xs, ys):  # a*x + y (only x and y are tuple)
    return [(a * x + y) for (x, y) in zip(xs, ys)]
