import torch
import numpy as np
from torch.autograd.functional import jvp
import xitorch as xt
from xitorch.optimize import rootfinder
import matplotlib.pyplot as plt

################### neural network for the surface ###################
class SimpleNN(torch.nn.Module):
    def __init__(self, ndim):
        super(SimpleNN, self).__init__()
        self.ndim = ndim
        ch = 100
        self.module = torch.nn.Sequential(
            torch.nn.Linear(self.ndim - 1, ch),
            torch.nn.Softplus(),
            torch.nn.Linear(ch, ch),
            torch.nn.Softplus(),
            torch.nn.Linear(ch, ch),
            torch.nn.Softplus(),
            torch.nn.Linear(ch, 1, bias=False),
        )

    def forward(self, rsurf):
        # rsurf: (nbatch, ndim-1)
        znn = self.module(rsurf)
        # add "wings" at the outer radius to guarantee the existence of rootfinder solution
        radsurf = rsurf.norm(dim=-1, keepdim=True)
        z = znn * (1 - torch.tanh((radsurf - 3.0) * 5.0))
        return torch.cat((rsurf, z), dim=-1)  # (nbatch, ndim)

################### physics functions ###################
def get_intersection(r0, v, fcn):
    # r0: (nbatch, ndim) initial point of the rays
    # v: (nbatch, ndim) the direction of travel of the rays
    # fcn: a function that takes (nbatch, ndim-1) and outputs (nbatch, ndim)
    @xt.make_sibling(fcn)
    def rootfinder_fcn(y, r0, v):
        surface_pos = fcn(y[..., :-1])  # (nbatch, ndim)
        raypos = r0 + v * y[..., -1:]  # (nbatch, ndim)
        return (raypos - surface_pos)

    y0 = torch.zeros_like(v)
    y = rootfinder(rootfinder_fcn, y0, params=(r0, v))
    return y[..., :-1], y[..., -1:]  # (nbatch, ndim-1) and (nbatch, 1)

def get_normal(rsurf, fcn):
    nbatch, ndimm1 = rsurf.shape
    # (nbatch, ndim-1, ndim-1)
    allv = torch.eye(ndimm1, dtype=rsurf.dtype, device=rsurf.device).unsqueeze(0).repeat(nbatch, 1, 1)

    # ndim-1, each (nbatch, ndim)
    dfdys = [jvp(fcn, rsurf, v=allv[..., i], create_graph=torch.is_grad_enabled())[1] for i in range(ndimm1)]
    normal = torch.cross(dfdys[0], dfdys[1], dim=-1)  # (nbatch, ndim)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    return normal

def get_reflection(r0, v, fcn):
    rsurf, t = get_intersection(r0, v, fcn)  # (nbatch, ndim-1) and (nbatch, 1)
    r1 = r0 + v * t  # (nbatch, ndim)
    # get the normal of the surface
    normal = get_normal(rsurf, fcn)  # (nbatch, ndim)
    v1 = v - 2 * torch.sum(normal * v, dim=-1, keepdim=True) * normal
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    return r1, v1

################### plotting functions ###################
def plot_rays(r0, v, t0, t1, xyidx=(0, 1)):
    rini = r0 + v * t0  # (nbatch, ndim)
    rfin = r0 + v * t1  # (nbatch, ndim)
    rini = rini[..., xyidx]  # (nbatch, 2)
    rfin = rfin[..., xyidx]  # (nbatch, 2)
    for i in range(rini.shape[0]):
        plt.plot((rini[i, 0], rfin[i, 0]), (rini[i, 1], rfin[i, 1]))

def plot_surface(fcn, dtype):
    xsurf = torch.linspace(-2, 2, 100, dtype=dtype)
    ysurf = torch.zeros_like(xsurf)
    rsurf = torch.cat([xsurf.unsqueeze(-1), ysurf.unsqueeze(-1)], dim=-1)  # (nbatch, ndim-1)
    zsurf = fcn(rsurf)[:, -1]  # (nbatch, 1)
    plt.plot(xsurf.view(-1).detach().numpy(), zsurf.view(-1).detach().numpy())

################### setup functions ###################
def generate_rays(nrays):
    phisource = torch.rand((nrays, 1), dtype=dtype) * (2 * np.pi)
    thetasource = torch.rand((nrays, 1), dtype=dtype) * (np.pi / 6.)
    vsource = torch.cat([torch.cos(thetasource),
                         torch.sin(thetasource) * torch.cos(phisource) * 0,
                         torch.sin(thetasource) * torch.sin(phisource)], dim=-1)  # (nrays, ndim)
    # rotate the source
    cos_45 = np.cos(np.pi / 4.)
    sin_45 = np.cos(np.pi / 4.)
    rotate_y = torch.tensor([[cos_45, 0., -sin_45],
                             [0., 1., 0.],
                             [sin_45, 0., cos_45]], dtype=dtype)  # (ndim, ndim)
    vsource = torch.matmul(vsource, rotate_y.transpose(-2, -1))
    vsource = vsource / vsource.norm(dim=-1, keepdim=True)
    return vsource

if __name__ == "__main__":
    torch.manual_seed(200)
    ndim = 3
    nrays = 10
    dtype = torch.float64

    # setting up the source
    d = 1.5
    r0source = torch.zeros((nrays, ndim), dtype=dtype)
    r0source[..., 0] = -d
    r0source[..., 2] = -d
    vsource0 = generate_rays(nrays)

    torch.manual_seed(100)
    nn = SimpleNN(ndim).to(dtype)

    # the screen located at z=-d
    def screen_fcn(rsurf):
        nbatch, _ = rsurf.shape
        zsurf = torch.zeros_like(rsurf[:, :1]) - d
        return torch.cat((rsurf, zsurf), dim=-1)  # (nbatch, ndim)

    def get_loss(r0source, vsource, plot=False, saveto=None):
        # reflected by the neural-network mirror
        r1, v1 = get_reflection(r0source, vsource, nn.forward)
        # captured by the screen
        rscreen, t1 = get_intersection(r1, v1, screen_fcn)
        # compute how far the rays are from the target
        xtarget = 1.0
        ytarget = 0.0
        devx = rscreen[:, 0].reshape(-1) - xtarget
        devy = rscreen[:, 1].reshape(-1) - ytarget
        loss = torch.dot(devx, devx) + torch.dot(devy, devy)
        if plot:
            rsurf, tint = get_intersection(r0source, vsource, nn.forward)
            plot_surface(nn.forward, dtype=dtype)
            plot_rays(r0source, vsource, 0., tint, xyidx=(0, 2))
            plot_rays(r1, v1, 0., t1, xyidx=(0, 2))
            plt.plot([xtarget], [-d], "C3x")
            if saveto is not None:
                plt.savefig(saveto)
                plt.close()
            else:
                plt.show()
        return loss

    opt = torch.optim.Adam(nn.parameters(), lr=3.0e-4)
    for iiter in range(1000):
        opt.zero_grad()
        if iiter % 10 == 0:  # validation
            loss = get_loss(r0source, vsource0, plot=True, saveto="images/%05d.png" % iiter)
            print("%5d: %.3e" % (iiter, loss))
        else:  # training
            loss = get_loss(r0source, vsource0)
            loss.backward()
            opt.step()
