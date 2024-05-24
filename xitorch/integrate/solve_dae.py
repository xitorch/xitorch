from typing import Callable, Any, Sequence, Optional, Dict
import torch
from xitorch._utils.misc import get_method
from xitorch._docstr.api_docstr import get_methods_docstr
from xitorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from xitorch._core.pure_function import get_pure_function
from xitorch.debug.modes import is_debug_enabled
from xitorch._impls.integrate.dae.dae_solver import bwd_euler_dae


def solve_dae(fcn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
              ts: torch.Tensor,
              y0: torch.Tensor,
              params: Sequence[Any] = [],
              method: Optional[str] = None,
              **fwd_options) -> torch.Tensor:
    r"""
    Solve the DAE system of equations using the given function and the time steps.

    .. math::

        f(t, y, y', \theta) = 0

    where the ``y`` is the state variable of size ``(ny,)``, ``y'`` is the derivative of the state variable,
    and ``\theta`` is the parameters. The function ``fcn`` should return a tensor of size ``(ny,)``.
    The initial condition is obtained by solving the equation above by assuming ``y' = 0`` (i.e., steady-state
    conditions).

    Arguments
    ---------
    fcn : callable
        The function that represents the DAE system of equations. It should have the signature
        ``fcn(t: torch.Tensor, y: torch.Tensor, yp: torch.Tensor, *params) -> torch.Tensor``,
        where ``t`` is a single element tensor, ``y`` and ``yp`` are tensors of shape ``(ny,)``,
        and ``params`` is a sequence of parameters.
    ts : torch.Tensor
        The time steps where the solution should be computed. The shape should be ``(nt,)``.
    y0 : torch.Tensor
        The initial condition. If there is an algebraic variable, the correctness will not be checked.
    params : sequence
        The parameters that are passed to the function ``fcn``.
    method : str
        The method to integrate the DAE system. If ``None``, it will use ``"bwdeuler"``.
    **fwd_options
        Method-specific option (see method section below).

    Returns
    -------
    torch.Tensor
        The solution of the DAE system at the time steps ``ts``. The shape is ``(nt, ny)``.
    """
    dtype = ts.dtype
    device = ts.device
    if is_debug_enabled():
        yp0 = torch.zeros_like(y0, dtype=dtype, device=device)
        assert_fcn_params(fcn, (ts[0], y0, yp0, *params))
    assert_runtime(len(ts.shape) == 1, "Argument ts must be a 1D tensor")

    if method is None:  # set the default method
        method = "bwdeuler"
    fwd_options["method"] = method

    # get the pure function format of fcn
    pfcn = get_pure_function(fcn)
    return _SolveDAE.apply(pfcn, ts, fwd_options, len(params), y0, *params, *pfcn.objparams())

class _SolveDAE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pfcn, ts, fwd_options, nparams, y0, *allparams):
        config = fwd_options
        params = allparams[:nparams]
        objparams = allparams[nparams:]

        # get the integration method and solver method
        integ_method = config.pop("method")

        # get the appropriate method
        methods = {
            "bwdeuler": bwd_euler_dae,
        }
        solver = get_method("solve_dae", methods, integ_method)
        yt = solver(pfcn, ts, y0, params, **config)

        # TODO: save the necessary variables for backward
        return yt  # (nt, ny)

    @staticmethod
    def backward(ctx, grad_yt):
        # grad_yt: (nt, ny)
        # TODO: implement the backward
        raise NotImplementedError("Backward of solve_dae is not implemented yet")


dae_methods: Dict[str, Callable] = {
    "bwdeuler": bwd_euler_dae,
}
solve_dae.__doc__ = get_methods_docstr(solve_dae, dae_methods)
