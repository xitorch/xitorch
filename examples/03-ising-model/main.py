import torch
import numpy as np
from xitorch.integrate import mcquad
import matplotlib.pyplot as plt

@torch.jit.script
def sweep(x, JkT, BkT):
    # flip randomly one-by-one the spins according to the J, B, and kT
    # using periodic boundary condition
    nrows, ncols = x.shape
    log_rand = torch.log(torch.rand((nrows, ncols), dtype=JkT.dtype, device=JkT.device))
    for i in range(nrows):
        for j in range(ncols):
            # try to flip one
            scur = x[i,j]
            sflip = -scur
            sneighs = torch.cat([
                x[i-1,j].unsqueeze(0),
                x[(i+1)%nrows,j].unsqueeze(0),
                x[i,j-1].unsqueeze(0),
                x[i,(j+1)%ncols].unsqueeze(0),
            ], dim=0)
            sscur  = (sneighs * scur ).sum()
            ssflip = -sscur
            dene = -JkT * (ssflip - sscur) + BkT * (sflip - scur)
            logpratio = -dene

            # decide to flip
            if logpratio > 0:
                accept = True
            else:
                accept = bool(log_rand[i,j] < logpratio)
            if accept:
                x[i,j] *= -1.0
    return x

def logp(x, JkT, BkT):
    s_up = torch.roll(x,  1, 0)
    s_dn = torch.roll(x, -1, 0)
    s_lf = torch.roll(x,  1, 1)
    s_rt = torch.roll(x, -1, 1)
    ss = (s_up + s_dn + s_lf + s_rt) * x
    # 0.5 because there are double countings on the edges
    ene = -JkT * ss.sum() * 0.5 + BkT * x.sum()
    return -ene

def calculate_M(x):
    return torch.mean(x)
    # return torch.mean(x).abs()

def plot_grad(xs, ys, dydxs, dx, color):
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        dydx = dydxs[i]
        plt.plot([x-dx, x+dx], [y-dydx*dx, y+dydx*dx], color=color)

def main():
    n = 30
    dtype = torch.float64

    BkTvals = [0.2]#, 0.1]
    JkTvals = np.linspace(0, 1, 11)
    JkTval_grad = JkTvals[:]

    fwd_options = {
        "method": "mhcustom",
        "nsamples": 500*3,
        "nburnout": 100*3,
        "custom_step": sweep,
    }

    Ms  = torch.empty((len(BkTvals), len(JkTvals)), dtype=dtype)
    dMs = torch.empty((len(BkTvals), len(JkTval_grad)), dtype=dtype)
    for i,Bval in enumerate(BkTvals):
        jgrad = 0
        for j,Jval in enumerate(JkTvals):
            print(i,j)
            x0 = (torch.randn((n, n), dtype=dtype) > 0) * 2.0 - 1.0
            x0 = x0.to(dtype)
            JkT = torch.tensor(Jval, dtype=dtype).requires_grad_()
            BkT = torch.tensor(Bval, dtype=dtype).requires_grad_()
            M = mcquad(
                ffcn = calculate_M,
                log_pfcn = logp,
                x0 = x0,
                fparams = [],
                pparams = [JkT, BkT],
                fwd_options = fwd_options)#.abs()

            # calculate the gradient
            if Jval in JkTval_grad:
                dMdJkt, = torch.autograd.grad(M, JkT)
                dMs[i,jgrad] = dMdJkt
                jgrad += 1

            Ms[i,j] = M

        Mnp = Ms[i,:].detach().numpy()
        plt.plot(JkTvals, Mnp, "C%d-"%i, label="B/kT = %.1f" % Bval)
        plot_grad(JkTval_grad, Mnp, dMs[i,:].detach().numpy(), dx=0.03, color="C%d"%i)
    print(Ms, dMs)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
