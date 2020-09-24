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
   getstart/linearop

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
