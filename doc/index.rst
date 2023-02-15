.. xitorch documentation master file, created by
   sphinx-quickstart on Tue Sep 22 10:46:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xitorch: differentiable scientific computing library
===========================================================

xitorch (pronounced "`sigh-torch`") is a library based on PyTorch that provides
differentiable operations and functionals for scientific computing and deep
learning.
xitorch provides analytic first and higher order derivatives automatically
using PyTorch's autograd engine.
It is inspired by SciPy, a popular Python library for scientific computing.

Example operations available in xitorch:

  * :obj:`xitorch.linalg.symeig`: symetric eigendecomposition for large sparse
    matrix or implicit linear operator,
  * :obj:`xitorch.optimize.rootfinder`: multivariate root finder, and
  * :obj:`xitorch.integrate.solve_ivp`: initial value problem solver or commonly
    known as ordinary differential equations (ODE) solver.

Why use xitorch:

  * contains differentiable functionals;
  * provides 1st, 2nd, and higher order gradients of functionals;
  * enables the use of functionals in the object-oriented way.

Source code: https://github.com/xitorch/xitorch/

Example
-------

.. code-block:: python

    import torch
    from xitorch.optimize import rootfinder

    def func1(y, A):  # example function
        return torch.tanh(A @ y + 0.1) + y / 2.0

    # set up the parameters and the initial guess
    A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    y0 = torch.zeros((2, 1))  # zeros as the initial guess

    # finding a root
    yroot = rootfinder(func1, y0, params=(A,))

    # calculate the derivatives
    dydA, = torch.autograd.grad(yroot.sum(), (A,), create_graph=True)
    grad2A, = torch.autograd.grad(dydA.sum(), (A,), create_graph=True)


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getstart/installation
   getstart/functional
   getstart/linearop
   getstart/debugging
   getstart/custom_method
   getstart/contribute
   notes/index

.. toctree::
   :maxdepth: 1
   :caption: Modules

   api/xitorch/index
   api/xitorch_optimize/index
   api/xitorch_integrate/index
   api/xitorch_linalg/index
   api/xitorch_interpolate/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
