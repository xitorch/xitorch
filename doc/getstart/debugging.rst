Debugging EditableModule and LinearOperator
===========================================

If you are implementing :obj:`xitorch.EditableModule` or
:obj:`xitorch.LinearOperator`, how are you sure that your implementation is
correct?
For example, are parameters listed in ``getparamnames()`` method of
:obj:`xitorch.EditableModule` complete or excessive?
Does the implementation of :obj:`xitorch.LinearOperator` actually behave like
a proper linear operator?
We will answer those questions here.

Checking parameters in :obj:`xitorch.EditableModule`
----------------------------------------------------

Let's say we have a class derived from :obj:`xitorch.EditableModule`:

.. jupyter-execute::

    import torch
    import xitorch

    class AClass(xitorch.EditableModule):
        def __init__(self, a):
            self.a = a
            self.b = a*a

        def mult(self, x):
            return self.b * x

        def getparamnames(self, methodname, prefix=""):
            if methodname == "mult":
                return [prefix+"a"]  # intentionally wrong
            else:
                raise KeyError()

The method ``getparamnames`` returns the wrong parameters for method ``mult``
above: it returns ``a`` while it should be ``b``.
To detect the fault, you can use the method ``assertparams`` of
the classes derived from :obj:`xitorch.EditableModule`.

The method ``assertparams`` takes a method and its arguments and keyword
arguments as the inputs.
It raises warnings if it detects missing affecting variables and excessive
variables.
An example is shown below.

.. jupyter-execute::
    :stderr:

    a = torch.tensor(2.0).requires_grad_()
    x = torch.tensor(0.4).requires_grad_()
    A = AClass(a)
    A.assertparams(A.mult, x)

Is my :obj:`LinearOperator` actually a linear operator?
-------------------------------------------------------

Programmatically, to implement a :obj:`LinearOperator`, you just need to
implement the matrix-vector multiplication function, ``._mv()``.
But does the implemented operation behave like a linear operator?

To check if your implementation is correct, you can use the method ``.check()``
in classes derived from :obj:`LinearOperator`.
It does not take any input and it will perform several checks which will raise
an error if it fails.

Let's take an example of a wrong implementation of a linear operator.

.. jupyter-execute::
    :stderr:
    :raises: AssertionError

    import torch
    import xitorch

    class WrongLinearOp(xitorch.LinearOperator):
        def __init__(self, a):
            shape = (torch.numel(a), torch.numel(a))
            super().__init__(shape=shape, dtype=a.dtype, device=a.device)
            self.a = a

        def _mv(self, x):
            return self.a * x + 1.0  # not a linear operator

    a = torch.tensor(1.2, requires_grad=True)
    linop = WrongLinearOp(a)
    linop.check()

As expected, it raises an error where the check fails (i.e., it is in linearity
check).
This check should only be done in debugging mode as it takes considerable amount
of time.
