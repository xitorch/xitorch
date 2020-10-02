Tutorial: linear operator
=========================

xitorch provides some linear algebra operations that does not need the explicit
matrix, such as :func:`xitorch.linalg.solve` and :func:`xitorch.linalg.symeig`.
To represent the matrix implicitly, base class :class:`xitorch.LinearOperator`
should be used to construct user-defined linear operators.
To write a LinearOperator class, these 2 functions must be implemented at the
very minimum:

  * ``_mv`` (matrix-vector multiplication)
  * ``_getparamnames`` (returning list of parameters affecting the output, as in
    :obj:`EditableModule`)

As an example, to write the matrix

.. math::

    \mathbf{A} = \begin{pmatrix}
    0 & 0 & ... & 0 & a_0 \\
    0 & 0 & ... & a_1 & 0 \\
    \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & a_{n-2} & ... & 0 & 0 \\
    a_{n-1} & 0 & ... & 0 & 0
    \end{pmatrix}

as a LinearOperator, we can write

.. testsetup:: tut_linop1

    import sys
    sys.stderr = sys.stdout

.. doctest:: tut_linop1

    >>> import torch
    >>> import xitorch
    >>> class MyFlip(xitorch.LinearOperator):
    ...     def __init__(self, a, size):
    ...         super().__init__(shape=(size,size))
    ...         self.a = a
    ...
    ...     def _mv(self, x):
    ...         return torch.flip(x, dims=(-1,)) * a
    ...
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix+"a"]
    ...
    >>> a = torch.arange(1, 6, dtype=torch.float).requires_grad_()
    >>> flip = MyFlip(a, 5)
    >>> print(flip)
    LinearOperator (MyFlip) with shape (5, 5), dtype = torch.float32, device = cpu

With only ``_mv`` implemented, we can call all matrix operations, including

  * ``.mv()`` (matrix-vector multiplication),
  * ``.mm()`` (matrix-matrix multiplication),
  * ``.fullmatrix()`` (returns the dense representation of the linear operator),
  * ``.rmv()`` (matrix-vector right-multiplication), and
  * ``.rmm()`` (matrix-matrix right-multiplication).

The matrix-matrix multiplication is calculated by batched matrix-vector calculation,
while the right-multiplication is performed using the adjoint trick with the
help of PyTorch's autograd engine.

.. doctest:: tut_linop1

    >>> vec = torch.arange(5, dtype=torch.float)
    >>> mat = torch.cat((vec.unsqueeze(-1), 2*vec.unsqueeze(-1)), dim=-1)
    >>> print(flip.mv(vec))
    tensor([4., 6., 6., 4., 0.], grad_fn=<MulBackward0>)
    >>> print(flip.rmv(vec))
    tensor([20., 12.,  6.,  2.,  0.], grad_fn=<FlipBackward>)
    >>> print(flip.mm(mat))
    tensor([[ 4.,  8.],
            [ 6., 12.],
            [ 6., 12.],
            [ 4.,  8.],
            [ 0.,  0.]], grad_fn=<SqueezeBackward1>)
    >>> print(flip.fullmatrix())
    tensor([[0., 0., 0., 0., 1.],
            [0., 0., 0., 2., 0.],
            [0., 0., 3., 0., 0.],
            [0., 4., 0., 0., 0.],
            [5., 0., 0., 0., 0.]], grad_fn=<SqueezeBackward1>)

The LinearOperator instance can also be used for linear algebra's operations
in xitorch, such as :func:`xitorch.linalg.solve`

.. doctest:: tut_linop1

    >>> from xitorch.linalg import solve
    >>> mmres = flip.mm(mat)
    >>> mat2 = solve(flip, mmres)
    >>> print(mat2)
    tensor([[0.0000, 0.0000],
            [1.0000, 2.0000],
            [2.0000, 4.0000],
            [3.0000, 6.0000],
            [4.0000, 8.0000]], grad_fn=<solve_torchfcnBackward>)
