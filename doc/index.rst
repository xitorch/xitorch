.. xitorch documentation master file, created by
   sphinx-quickstart on Tue Sep 22 10:46:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xitorch: differentiable scientific computing library
===========================================================

xitorch (pronounced "sigh-torch") is a library based on PyTorch that provides
differentiable operations and functionals for scientific computing and deep
learning.
xitorch provides analytic first and higher order derivatives automatically
using PyTorch's autograd engine.
It is inspired by SciPy, a popular Python library for scientific computing.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getstart/installation
   getstart/functional

.. toctree::
   :maxdepth: 1
   :caption: Modules

   api/xitorch/index
   api/xitorch_optimize/index
   api/xitorch_integrate/index
   api/xitorch_linalg/index
   api/xitorch_interpolate/index

Example of use
--------------

.. doctest::

    >>> import torch
    >>> from xitorch.optimize import rootfinder
    >>> def func1(x, a):
    ...     return a[0]*x*x + a[1]*x + a[2]
    ...
    >>> a = torch.tensor([1.0, 3.0, -1.75], requires_grad=True)
    >>> x0 = torch.tensor(0.5)
    >>> xroot = rootfinder(func1, x0, params=(a,))
    >>> print(xroot)
    tensor(0.5000, grad_fn=<_RootFinderBackward>)
    >>> dxda, = torch.autograd.grad(xroot, (a,), create_graph=True) # first derivative
    >>> print(dxda)
    tensor([-0.0625, -0.1250, -0.2500], grad_fn=<AddBackward0>)
    >>> d2xda2, = torch.autograd.grad(dxda[0], (a,)) # second derivative
    >>> print(d2xda2)
    tensor([0.0293, 0.0430, 0.0547])

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
