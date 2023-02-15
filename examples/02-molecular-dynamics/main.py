import os
import torch
import numpy as np
from xitorch.integrate import solve_ivp
import matplotlib.pyplot as plt

######################## physics functions ########################
def dydt(t, y):
    # t: 1-element tensor
    # y: (2, nbatch, nparticles, ndim)
    nparticles = y.shape[-1] // 2
    pos = y[0]  # (nbatch, nparticles, ndim)
    vel = y[1]
    dposdt = vel.clone()  # (nbatch, nparticles, ndim)

    # calculate the distance among the particles
    dpos = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # (nbatch, nparticles, nparticles, ndim)
    dist = dpos.norm(dim=-1, keepdim=True)  # (nbatch, nparticles, nparticles, 1)
    dir = dpos / (dist + 1e-12)

    # get the force
    force = -(1. / torch.sqrt(dist * dist + 1e-1) * dir).sum(dim=-2)  # (nbatch, nparticles, ndim)
    dveldt = force
    dydt = torch.cat((dposdt.unsqueeze(0), dveldt.unsqueeze(0)), dim=0)
    return dydt  # (2, nbatch, nparticles, ndim)

def get_loss(pos0, vel0, ts, pos_target):
    y0 = torch.cat((pos0.unsqueeze(0), vel0.unsqueeze(0)), dim=0)
    yt = solve_ivp(dydt, ts, y0, method="rk4")
    posf = yt[-1, 0]  # (nbatch, nparticles, ndim)
    dev = posf - pos_target
    loss = torch.dot(dev.reshape(-1), dev.reshape(-1))
    return loss, yt

######################## bookkeepers ########################
def save_image(yt, fname_format, scale):
    nt = yt.shape[0]
    gap = scale / 4.0
    for i in range(0, nt, 1):
        pos = yt[i][0]
        plt.plot(pos[..., 0].detach(), pos[..., 1].detach(), 'o')
        plt.gca().set_xlim((-gap, scale + gap))
        plt.gca().set_ylim((-gap, scale + gap))
        plt.savefig(fname_format % i)
        plt.close()

def get_initial_pos(nparticles, scale, dtype):
    nrows = int(nparticles ** 0.5)
    ncols = int(np.ceil(nparticles / nrows))
    x0 = torch.linspace(0, scale, ncols, dtype=dtype)
    y0 = torch.linspace(0, scale, nrows, dtype=dtype)
    y, x = torch.meshgrid(y0, x0)  # (nrows, ncols)
    y = y.reshape(-1)[:nparticles]
    x = x.reshape(-1)[:nparticles]
    pos = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1).unsqueeze(0)  # (1, nparticles, 2)
    return pos

def get_target_pos(nparticles, scale, dtype):
    # half of the particles to letter O
    no = nparticles // 2
    nx = nparticles - no
    gap = 0.1 * scale

    # letter O
    radius = (scale - gap) * 0.25
    xcentre = radius
    ycentre = scale * 0.5
    theta = torch.linspace(0, 2 * np.pi, no, dtype=dtype)
    xo = xcentre + radius * torch.cos(theta)
    yo = ycentre + radius * torch.sin(theta)

    # letter X
    nxl = nx // 2
    nxr = nx - nxl
    xleft = (scale + gap) * 0.5
    xright = scale
    width = xright - xleft
    yup = (scale + width) * 0.5
    ydown = (scale - width) * 0.5
    dl = torch.linspace(0, 1, nxl, dtype=dtype)
    dr = torch.linspace(0, 1, nxr, dtype=dtype)
    xxl = xleft + (xright - xleft) * dl
    xxr = xleft + (xright - xleft) * dr
    yxl = yup + (ydown - yup) * dl
    yxr = ydown + (yup - ydown) * dr

    # combine all
    xall = torch.cat((xo, xxl, xxr), dim=-1)  # (nparticles,)
    yall = torch.cat((yo, yxl, yxr), dim=-1)  # (nparticles,)
    pos = torch.cat((xall.unsqueeze(-1), yall.unsqueeze(-1)), dim=-1).unsqueeze(0)  # (1, nparticles, 2)
    return pos

######################## main function ########################
def mainopt():
    torch.manual_seed(100)
    dtype = torch.float64
    nparticles, ndim, nt = 32, 2, 100
    # set up the initial positions (grid-like)
    scale = 4.0
    pos = get_initial_pos(nparticles, scale=scale, dtype=dtype)
    pos_target = get_target_pos(nparticles, scale=scale, dtype=dtype)
    vel = torch.randn((1, nparticles, ndim), dtype=dtype) * 2
    vel = vel.requires_grad_()
    ts = torch.linspace(0, 1, nt)

    params = (vel,)
    opt = torch.optim.Adam(params, lr=1e-3)
    for i in range(100000):
        opt.zero_grad()
        loss, yt = get_loss(pos, vel, ts, pos_target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(params, max_norm=5., norm_type="inf")
        opt.step()
        if i % 10 == 0 or i < 10:
            print("%5d: %.3e" % (i, float(loss)))
        if i % 500 == 0:
            fdir = "images/%06d/" % i
            try:
                os.mkdir(fdir)
            except FileExistsError:
                pass
            save_image(yt, fname_format=fdir + "time-%03d.png", scale=scale)

if __name__ == "__main__":
    mainopt()
