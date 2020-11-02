from typing import Callable, Mapping, Any, Sequence, Union
import torch
from xitorch._utils.misc import TensorNonTensorSeparator, get_method
from xitorch._utils.assertfuncs import assert_fcn_params
from xitorch._impls.optimize.root.rootsolver import broyden1, broyden2, \
    linearmixing
from xitorch.linalg.solve import solve
from xitorch.grad.jachess import jac
from xitorch._core.pure_function import get_pure_function, make_sibling
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch.debug.modes import is_debug_enabled

__all__ = ["equilibrium", "rootfinder", "minimize"]

def rootfinder(
        fcn: Callable[..., torch.Tensor],
        y0: torch.Tensor,
        params: Sequence[Any] = [],
        bck_options: Mapping[str, Any] = {},
        method: Union[str, Callable, None] = None,
        **fwd_options) -> torch.Tensor:
    r"""
    Solving the rootfinder equation of a given function,

    .. math::

        \mathbf{0} = \mathbf{f}(\mathbf{y}, \theta)

    where :math:`\mathbf{f}` is a function that can be non-linear and
    produce output of the same shape of :math:`\mathbf{y}`, and :math:`\theta`
    is other parameters required in the function.
    The output of this block is :math:`\mathbf{y}`
    that produces the :math:`\mathbf{0}` as the output.

    Arguments
    ---------
    fcn : callable
        The function :math:`\mathbf{f}` with output tensor ``(*ny)``
    y0 : torch.tensor
        Initial guess of the solution with shape ``(*ny)``
    params : list
        Sequence of any other parameters to be put in ``fcn``
    bck_options : dict
        Method-specific options for the backward solve (see :func:`xitorch.linalg.solve`)
    method : str or callable or None
        Rootfinder method. If None, it will choose ``"broyden1"``.
    **fwd_options
        Method-specific options (see method section)

    Returns
    -------
    torch.tensor
        The solution which satisfies
        :math:`\mathbf{0} = \mathbf{f}(\mathbf{y},\theta)`
        with shape ``(*ny)``

    Example
    -------
    .. testsetup:: root1

        import torch
        from xitorch.optimize import rootfinder

    .. doctest:: root1

        >>> def func1(y, A):  # example function
        ...     return torch.tanh(A @ y + 0.1) + y / 2.0
        >>> A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
        >>> y0 = torch.zeros((2,1))  # zeros as the initial guess
        >>> yroot = rootfinder(func1, y0, params=(A,))
        >>> print(yroot)
        tensor([[-0.0459],
                [-0.0663]], grad_fn=<_RootFinderBackward>)
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (y0, *params))

    pfunc = get_pure_function(fcn)
    fwd_options["method"] = _get_rootfinder_default_method(method)
    return _RootFinder.apply(pfunc, y0, fwd_options, bck_options, len(params), *params, *pfunc.objparams())

def equilibrium(
        fcn: Callable[..., torch.Tensor],
        y0: torch.Tensor,
        params: Sequence[Any] = [],
        bck_options: Mapping[str, Any] = {},
        method: Union[str, Callable, None] = None,
        **fwd_options) -> torch.Tensor:
    r"""
    Solving the equilibrium equation of a given function,

    .. math::

        \mathbf{y} = \mathbf{f}(\mathbf{y}, \theta)

    where :math:`\mathbf{f}` is a function that can be non-linear and
    produce output of the same shape of :math:`\mathbf{y}`, and :math:`\theta`
    is other parameters required in the function.
    The output of this block is :math:`\mathbf{y}`
    that produces the same :math:`\mathbf{y}` as the output.

    Arguments
    ---------
    fcn : callable
        The function :math:`\mathbf{f}` with output tensor ``(*ny)``
    y0 : torch.tensor
        Initial guess of the solution with shape ``(*ny)``
    params : list
        Sequence of any other parameters to be put in ``fcn``
    bck_options : dict
        Method-specific options for the backward solve (see :func:`xitorch.linalg.solve`)
    method : str or None
        Rootfinder method. If None, it will choose ``"broyden1"``.
    **fwd_options
        Method-specific options (see method section)

    Returns
    -------
    torch.tensor
        The solution which satisfies
        :math:`\mathbf{y} = \mathbf{f}(\mathbf{y},\theta)`
        with shape ``(*ny)``

    Example
    -------
    .. testsetup:: equil1

        import torch
        from xitorch.optimize import equilibrium

    .. doctest:: equil1

        >>> def func1(y, A):  # example function
        ...     return torch.tanh(A @ y + 0.1) + y / 2.0
        >>> A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
        >>> y0 = torch.zeros((2,1))  # zeros as the initial guess
        >>> yequil = equilibrium(func1, y0, params=(A,))
        >>> print(yequil)
        tensor([[ 0.2313],
                [-0.5957]], grad_fn=<_RootFinderBackward>)

    Note
    ----
    * This is a direct implementation of finding the root of
      :math:`\mathbf{g}(\mathbf{y}, \theta) = \mathbf{y} - \mathbf{f}(\mathbf{y}, \theta)`
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (y0, *params))

    pfunc = get_pure_function(fcn)

    @make_sibling(pfunc)
    def new_fcn(y, *params):
        return y - pfunc(y, *params)

    fwd_options["method"] = _get_rootfinder_default_method(method)
    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, len(params), *params, *pfunc.objparams())

def minimize(
        fcn: Callable[..., torch.Tensor],
        y0: torch.Tensor,
        params: Sequence[Any] = [],
        bck_options: Mapping[str, Any] = {},
        method: Union[str, Callable] = None,
        **fwd_options) -> torch.Tensor:
    r"""
    Solve the unbounded minimization problem:

    .. math::

        \mathbf{y^*} = \arg\min_\mathbf{y} f(\mathbf{y}, \theta)

    to find the best :math:`\mathbf{y}` that minimizes the output of the
    function :math:`f`.

    Arguments
    ---------
    fcn: callable
        The function to be optimized with output tensor with 1 element.
    y0: torch.tensor
        Initial guess of the solution with shape ``(*ny)``
    params: list
        Sequence of any other parameters to be put in ``fcn``
    bck_options: dict
        Method-specific options for the backward solve (see :func:`xitorch.linalg.solve`)
    method: str or callable or None
        Minimization method. If None, it will choose ``"broyden1"``.
    **fwd_options
        Method-specific options (see method section)

    Returns
    -------
    torch.tensor
        The solution of the minimization with shape ``(*ny)``

    Example
    -------
    .. testsetup:: root1

        import torch
        from xitorch.optimize import minimize

    .. doctest:: root1

        >>> def func1(y, A):  # example function
        ...     return torch.sum((A @ y)**2 + y / 2.0)
        >>> A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
        >>> y0 = torch.zeros((2,1))  # zeros as the initial guess
        >>> ymin = minimize(func1, y0, params=(A,))
        >>> print(ymin)
        tensor([[-0.0519],
                [-0.2684]], grad_fn=<_RootFinderBackward>)
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (y0, *params))

    pfunc = get_pure_function(fcn)

    fwd_options["method"] = _get_minimizer_default_method(method)
    # the rootfinder algorithms are designed to move to the opposite direction
    # of the output of the function, so the output of this function is just
    # the grad of z w.r.t. y

    @make_sibling(pfunc)
    def new_fcn(y, *params):
        with torch.enable_grad():
            y1 = y.clone().requires_grad_()
            z = pfunc(y1, *params)
        grady, = torch.autograd.grad(z, (y1,), retain_graph=True,
                                     create_graph=torch.is_grad_enabled())
        return grady

    return _RootFinder.apply(new_fcn, y0, fwd_options, bck_options, len(params), *params, *pfunc.objparams())

class _RootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, y0, options, bck_options, nparams, *allparams):

        # set default options
        config = options
        ctx.bck_options = bck_options

        params = allparams[:nparams]
        objparams = allparams[nparams:]

        with fcn.useobjparams(objparams):

            method = config.pop("method")
            methods = {
                "broyden1": broyden1,
                "broyden2": broyden2,
                "linearmixing": linearmixing,
            }
            method_fcn = get_method("rootfinder", methods, method)
            y = method_fcn(fcn, y0, params, **config)

        ctx.fcn = fcn

        # split tensors and non-tensors params
        ctx.nparams = nparams
        ctx.param_sep = TensorNonTensorSeparator(allparams)
        tensor_params = ctx.param_sep.get_tensor_params()
        ctx.save_for_backward(y, *tensor_params)

        return y

    @staticmethod
    def backward(ctx, grad_yout):
        param_sep = ctx.param_sep
        yout = ctx.saved_tensors[0]
        nparams = ctx.nparams
        fcn = ctx.fcn

        # merge the tensor and nontensor parameters
        tensor_params = ctx.saved_tensors[1:]
        allparams = param_sep.reconstruct_params(tensor_params)
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        # dL/df
        with fcn.useobjparams(objparams):

            jac_dfdy = jac(ctx.fcn, params=(yout, *params), idxs=[0])[0]
            gyfcn = solve(A=jac_dfdy.H, B=-grad_yout.reshape(-1, 1),
                          bck_options=ctx.bck_options, **ctx.bck_options)
            gyfcn = gyfcn.reshape(grad_yout.shape)

            # get the grad for the params
            with torch.enable_grad():
                tensor_params_copy = [p.clone().requires_grad_() for p in tensor_params]
                allparams_copy = param_sep.reconstruct_params(tensor_params_copy)
                params_copy = allparams_copy[:nparams]
                objparams_copy = allparams_copy[nparams:]
                with fcn.useobjparams(objparams_copy):
                    yfcn = fcn(yout, *params_copy)

            grad_tensor_params = torch.autograd.grad(yfcn, tensor_params_copy, grad_outputs=gyfcn,
                                                     create_graph=torch.is_grad_enabled())
            grad_nontensor_params = [None for _ in range(param_sep.nnontensors())]
            grad_params = param_sep.reconstruct_params(grad_tensor_params, grad_nontensor_params)

        return (None, None, None, None, None, *grad_params)

def _get_rootfinder_default_method(method):
    if method is None:
        return "broyden1"
    else:
        return method

def _get_minimizer_default_method(method):
    if method is None:
        return "broyden1"
    else:
        return method


# docstring completion
rf_methods: Sequence[Callable] = [broyden1, broyden2, linearmixing]
rootfinder.__doc__ = get_methods_docstr(rootfinder, rf_methods)
equilibrium.__doc__ = get_methods_docstr(equilibrium, rf_methods)
minimize.__doc__ = get_methods_docstr(minimize, rf_methods)
