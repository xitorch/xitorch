from typing import Callable, Sequence, Any, Optional, Dict
import torch
from xitorch.optimize.rootfinder import rootfinder


__all__ = ["bwd_euler_dae"]

def bwd_euler_dae(fcn: Callable[..., torch.Tensor], ts: torch.Tensor, y0: torch.Tensor,
                  params: Sequence[Any], *,
                  solver_method: Optional[str] = None,
                  solver_kwargs: Optional[Dict[str, Any]] = None,
                  **unused) -> torch.Tensor:
    """
    Solve the DAE system of equations using backward Euler method.
    Specifically, it assumes ``y' = (y_{n+1} - y_n) / dt`` and solve the equation using rootfinder.

    Keyword arguments
    -----------------
    solver_method : str
        The method to solve the rootfinder. Available methods can be seen in :class:`xitorch.optimize.rootfinder`.
        The default is using ``rootfinder``'s default method.
    solver_kwargs : dict
        The keyword arguments that are passed to the rootfinder method. See :class:`xitorch.optimize.rootfinder`.
    """
    # y0: (ny,)
    # ts: (nt,)
    solver_kwargs = {} if solver_kwargs is None else solver_kwargs

    # initial condition, no check is performed here
    # TODO: perform the check of the initial condition, especially when algebraic variables are present
    ysol = [y0]

    # solve the DAE
    for i in range(1, len(ts)):
        ti = ts[i]
        dt = ti - ts[i - 1]

        def fcn_transient(y, *params):
            return fcn(ti, y, (y - y0) / dt, *params)

        y0 = rootfinder(fcn_transient, y0, params, method=solver_method, **solver_kwargs)
        ysol.append(y0)

    return torch.stack(ysol, dim=0)  # (nt, ny)
