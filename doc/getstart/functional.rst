Tutorial: using functionals
===========================

xitorch contains functionals that are commonly used in scientific computing and
deep learning, such as rootfinder and initial value problem solver.
One advantage of xitorch is that it can provide the first and higher order
derivatives of the functional outputs.
However, it comes with a cost: the function input to the functionals are
restricted to

1. Pure functions (i.e functions with their outputs fully determined by their tensor inputs)
2. Methods of classes derived from ``torch.nn.Module``
3. Methods of classes derived from :class:`xitorch.EditableModule`

In this example, we will show how to use the functionals in xitorch with above
function inputs.

Pure function as input
----------------------

Let's say we want to find :math:`\mathbf{x}` that is a root of the equation

.. math::

    \mathbf{0}=\mathrm{tanh}(\mathbf{A}\mathbf{x+b}) + \mathbf{x}/2

where :math:`\mathbf{x}` and :math:`\mathbf{b}` are vectors of size :math:`n\times 1`,
and :math:`\mathbf{A}` is a matrix of size :math:`n\times n`.
The first step is to write the function with :math:`\mathbf{x}` as the first argument
as well as specifying the known parameters, i.e. :math:`\mathbf{A}` and
:math:`\mathbf{b}`:

.. jupyter-execute::

    import torch
    def func1(x, A, b):
        return torch.tanh(A @ x + b) + x / 2.0
    A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    b = torch.tensor([[0.3], [-0.2]]).requires_grad_()

Once the function and parameters have been defined, now we can call the
functional with an initial guess of the root.

.. jupyter-execute::

    from xitorch.optimize import rootfinder
    x0 = torch.zeros((2,1))  # zeros as the initial guess
    xroot = rootfinder(func1, x0, params=(A, b))
    print(xroot)

The function :func:`xitorch.optimize.rootfinder` and most other functionals
in xitorch takes the similar argument patterns.
It typically starts with the function as the first argument, the parameter of
interest as the second argument, then followed by other parameters required by
the function.

The output of the functional can be used to calculate the first order and
higher order derivatives.

.. jupyter-execute::

    dxdA, dxdb = torch.autograd.grad(xroot.sum(), (A, b), create_graph=True)  # first derivative
    grad2A, grad2b = torch.autograd.grad(dxdA.sum(), (A, b), create_graph=True)  # second derivative
    print(grad2A)

Methods of ``torch.nn.Module`` as input
---------------------------------------
Functionals in xitorch can also take methods from ``torch.nn.Module`` as their
inputs, given that all the affecting parameters are listed in
``.named_parameters()``.

Let's take the previous problem as an example: finding the root :math:`\mathbf{x}`
to satisfy

.. math::

    \mathbf{0}=\mathrm{tanh}(\mathbf{A}\mathbf{x+b}) + \mathbf{x}/2

where now :math:`\mathbf{A}` and :math:`\mathbf{b}` are parameters in a ``torch.nn.Module``.

.. jupyter-execute::

    import torch
    class NNModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.nn.Parameter(torch.tensor([[1.1, 0.4], [0.3, 0.8]]))
            self.b = torch.nn.Parameter(torch.tensor([[0.3], [-0.2]]))

        def forward(self, x):  # also called in __call__
            return torch.tanh(self.A @ x + self.b) + x / 2.0

The functional can then be applied similarly with the previous case, but now
without additional parameters

.. jupyter-execute::

    from xitorch.optimize import rootfinder
    module = NNModule()
    x0 = torch.zeros((2,1))  # zeros as the initial guess
    xroot = rootfinder(module.forward, x0, params=())  # module.forward only takes x
    print(xroot)

The output of the rootfinder can also be used to calculate the first and higher
order derivatives of the module's parameters

.. jupyter-execute::

    nnparams = list(module.parameters())  # (A, b)
    dxdA, dxdb = torch.autograd.grad(xroot.sum(), nnparams, create_graph=True)  # first derivative
    grad2A, grad2b = torch.autograd.grad(dxdA.sum(), nnparams, create_graph=True)  # second derivative
    print(grad2A)

Methods of :class:`xitorch.EditableModule` as input
---------------------------------------------------
The problem with ``torch.nn.Module`` classes is that they can only take leaves as
the parameters.
However, in large scientific simulations, sometimes we want processed variables
(non-leaf) as the parameters for efficiency.

To illustrate the use case of :class:`xitorch.EditableModule`, let's slightly
modify the test case above.
We want to find the root :math:`\mathbf{x}` to satisfy the equation

.. math::

    \mathbf{0}=\mathrm{tanh}[(\mathbf{E}^3)\mathbf{x+b}] + \mathbf{x}/2

where :math:`\mathbf{E}^3` is the matrix power operator.
Because the matrix power operand does not depend on :math:`\mathbf{x}`,
we should be able to precompute :math:`\mathbf{A}=\mathbf{E}^3` so
we don't have to compute it every time in the function.

To do this with :class:`xitorch.EditableModule`, we can write something like

.. jupyter-execute::

    import torch
    import xitorch
    class MyModule(xitorch.EditableModule):
        def __init__(self, E, b):
            self.E = E
            self.A = E @ E @ E
            self.b = b

        def forward(self, x):
            return torch.tanh(self.A @ x + self.b) + x / 2.0

        def getparamnames(self, methodname, prefix=""):
            if methodname == "forward":
                return [prefix+"A", prefix+"b"]
            else:
                raise KeyError()

The biggest difference here is that in :class:`xitorch.EditableModule`, a method
``getparamnames`` must be implemented.
It returns a list of parameters affecting the outputs of a method in that class.
To check if the list of parameters written manually in ``getparamnames`` is correct,
:func:`xitorch.EditableModule.assertparams` can be used.

To use the functional, it is similar to the previous test cases

.. jupyter-execute::

    from xitorch.optimize import rootfinder
    E = torch.tensor([[1.1, 0.4], [0.3, 0.9]]).requires_grad_()
    b = torch.tensor([[0.3], [-0.2]]).requires_grad_()
    mymodule = MyModule(E, b)
    x0 = torch.zeros((2,1))  # zeros as the initial guess
    xroot = rootfinder(mymodule.forward, x0, params=())  # .forward() only takes x
    print(xroot)

The output can then be used to get the derivatives with respect to direct parameters
(:math:`\mathbf{A}` and :math:`\mathbf{b}`) as well as indirect parameters
(:math:`\mathbf{E}`).

.. jupyter-execute::

    params = (mymodule.A, mymodule.b, mymodule.E)
    dxdA, dxdb, dxdE = torch.autograd.grad(xroot.sum(), params, create_graph=True)  # 1st deriv
    grad2A, grad2b, gradE = torch.autograd.grad(dxdE.sum(), params, create_graph=True)  # 2nd deriv
    print(grad2A)
