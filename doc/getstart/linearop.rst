Tutorial: linear operator
=========================

xitorch provides some linear algebra operations that does not need the explicit
matrix, such as :func:`xitorch.linalg.solve` and :func:`xitorch.linalg.symeig`.
To represent the matrix implicitly, base class :class:`xitorch.LinearOperator`
should be used to construct user-defined linear operators.
To write a LinearOperator class, ``_mv`` and ``_getparamnames``
must be implemented at the very minimum, and ``_rmv`` only if needed.

As an example, to write the matrix

.. math::

    \mathbf{A} = \begin{pmatrix}
    0 & 0 & ... & 0 & a \\
    0 & 0 & ... & a & 0 \\
    \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & a & ... & 0 & 0 \\
    a & 0 & ... & 0 & 0
    \end{pmatrix}

as a LinearOperator, we can write

.. testsetup:: tut_linop1

    import sys
    sys.stderr = sys.stdout

.. doctest:: tut_linop1

    >>> import torch
    >>> import xitorch
    >>> class Flip(xitorch.LinearOperator):
    ...     def __init__(self, a, size):
    ...         super().__init__(shape=(size,size))
    ...         self.a = a
    ...
    ...     def _mv(self, x):
    ...         return torch.flip(x * a, dims=(-1,))
    ...
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix+"a"]
    ...
    >>> a = torch.tensor(2.0).requires_grad_()
    >>> flip = Flip(a, 5)
    >>> print(flip)
    LinearOperator (Flip) with shape (5, 5), dtype = torch.float32, device = cpu

With ``_mv`` implemented, we can call ``.mv()``, ``.mm()``, and ``.fullmatrix()``,
but not ``.rmv()`` and ``.rmm()``
(unless ``is_hermitian=True`` is specified in ``super()`` initializer)

.. doctest:: tut_linop1

    >>> vec = torch.arange(5, dtype=torch.float)
    >>> mat = torch.cat((vec.unsqueeze(-1), 2*vec.unsqueeze(-1)), dim=-1)
    >>> print(flip.mv(vec))
    tensor([8., 6., 4., 2., 0.], grad_fn=<FlipBackward>)
    >>> print(flip.mm(mat))
    tensor([[ 8., 16.],
            [ 6., 12.],
            [ 4.,  8.],
            [ 2.,  4.],
            [ 0.,  0.]], grad_fn=<SqueezeBackward1>)
    >>> print(flip.fullmatrix())
    tensor([[0., 0., 0., 0., 2.],
            [0., 0., 0., 2., 0.],
            [0., 0., 2., 0., 0.],
            [0., 2., 0., 0., 0.],
            [2., 0., 0., 0., 0.]], grad_fn=<SqueezeBackward1>)

The LinearOperator instance can also be used for linear algebra's operations
in xitorch, such as :func:`xitorch.linalg.solve`

.. doctest:: tut_linop1

    >>> from xitorch.linalg import solve
    >>> mmres = flip.mm(mat)
    >>> mat2 = solve(flip, mmres)
    >>> print(mat2)
    tensor([[0., 0.],
            [1., 2.],
            [2., 4.],
            [3., 6.],
            [4., 8.]], grad_fn=<SolveBackward>)
