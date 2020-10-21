Building a custom linear operator
=================================

xitorch provides some linear algebra operations that does not need the explicit
matrix, such as :func:`xitorch.linalg.solve` and :func:`xitorch.linalg.symeig`.
To represent the matrix implicitly, base class :class:`xitorch.LinearOperator`
should be used to construct user-defined linear operators.
To write a LinearOperator class, the method ``_mv`` (matrix-vector
multiplication) must be implemented.

If the LinearOperator is used in xitorch's functional with grad enabled, e.g.
:func:`xitorch.linalg.symeig` or :func:`xitorch.linalg.solve`, it must have
the method ``_getparamnames`` implemented.
``_getparamnames`` returns a list of parameters affecting the output,
as in :obj:`xitorch.EditableModule`

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

.. jupyter-execute::

    import torch
    import xitorch
    class MyFlip(xitorch.LinearOperator):
       def __init__(self, a, size):
           super().__init__(shape=(size,size))
           self.a = a

       def _mv(self, x):
           return torch.flip(x, dims=(-1,)) * a

       def _getparamnames(self, prefix=""):
           return [prefix+"a"]

    a = torch.arange(1, 6, dtype=torch.float).requires_grad_()
    flip = MyFlip(a, 5)
    print(flip)

With only ``_mv`` implemented, we can call all matrix operations, including

  * ``.mv()`` (matrix-vector multiplication),
  * ``.mm()`` (matrix-matrix multiplication),
  * ``.fullmatrix()`` (returns the dense representation of the linear operator),
  * ``.rmv()`` (matrix-vector right-multiplication), and
  * ``.rmm()`` (matrix-matrix right-multiplication).

The matrix-matrix multiplication is calculated by batched matrix-vector calculation,
while the right-multiplication is performed using the adjoint trick with the
help of PyTorch's autograd engine.

.. jupyter-execute::

    vec = torch.arange(5, dtype=torch.float)
    mat = torch.cat((vec.unsqueeze(-1), 2*vec.unsqueeze(-1)), dim=-1)
    print(flip.mv(vec))

.. jupyter-execute::

    # matrix-vector right-multiplication
    print(flip.rmv(vec))

.. jupyter-execute::

    # matrix-matrix multiplication
    print(flip.mm(mat))

.. jupyter-execute::

    # getting the dense representation
    print(flip.fullmatrix())

The LinearOperator instance can also be used for linear algebra's operations
in xitorch, such as :func:`xitorch.linalg.solve`

.. jupyter-execute::

    from xitorch.linalg import solve
    mmres = flip.mm(mat)
    mat2 = solve(flip, mmres)
    print(mat2)
