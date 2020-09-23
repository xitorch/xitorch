.. xitorch documentation master file, created by
   sphinx-quickstart on Tue Sep 22 10:46:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xitorch: differentiable scientific computing library
===========================================================

xitorch (pronounced "sigh-torch") is a library based on PyTorch that provides
operations and functionals for scientific computing and deep learning.

Requirements
------------

* python >= 3.6
* pytorch >= 1.6 (install `here <https://pytorch.org/>`_)

Installation
------------

In your terminal, type:

.. code-block::

    git clone https://github.com/mfkasim91/xitorch
    cd xitorch
    pip install -e .

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


Table of content
================

.. toctree::
   :maxdepth: 2

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
