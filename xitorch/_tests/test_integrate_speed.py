from typing import List
import time
import torch
from xitorch.integrate import solve_ivp

class RemoveTimeArg(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._module = module

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._module(y)

def test_ivp_speed():
    # test the speed of ivp compared to manually implement it
    dydt_module = torch.nn.Sequential(
        torch.nn.Linear(3, 10),
        torch.nn.LogSigmoid(),
        torch.nn.Linear(10, 10),
        torch.nn.LogSigmoid(),
        torch.nn.Linear(10, 3),
    )

    def manual_euler(dydt_module: torch.nn.Module, ts: torch.Tensor, y0: torch.Tensor):
        # ts: (nt,)
        # y0: (nb, ndim)
        y_lst: List[torch.Tensor] = []
        y_lst.append(y0)
        y = y0
        for i in range(1, ts.size(0)):
            dydt = dydt_module(y)  # (nb, ndim)
            y = y + dydt * (ts[i] - ts[i - 1])
            y_lst.append(y)
        yres = torch.stack(y_lst, dim=0)
        return yres

    dydt_module2 = RemoveTimeArg(dydt_module)
    ts = torch.linspace(0, 1.0, 100).requires_grad_()
    y0 = torch.randn((5, 3)).requires_grad_()

    t0 = time.time()
    for i in range(10):
        yt0 = manual_euler(dydt_module, ts, y0)
    t1 = time.time()
    for i in range(10):
        yt1 = solve_ivp(dydt_module2, ts, y0, method="euler")
    t2 = time.time()
    for i in range(10):
        _ = torch.autograd.grad(yt0, (ts, y0), torch.ones_like(yt0), retain_graph=True)
    t3 = time.time()
    for i in range(10):
        _ = torch.autograd.grad(yt1, (ts, y0), torch.ones_like(yt1), retain_graph=True)
    t4 = time.time()

    # forward timing
    dt1 = t1 - t0  # manual
    dt2 = t2 - t1  # xitorch
    assert dt2 < 1.5 * dt1  # cannot be much slower than the manual implementation

    # backward timing
    dtg1 = t3 - t2  # manual
    dtg2 = t4 - t3  # xitorch
    assert dtg2 < 15.0 * dtg1  # it is expected to be slower than manual implementation

if __name__ == "__main__":
    test_ivp_speed()
