# This file contains calculation of finite differences for debugging purposes
import torch

def finite_differences(fcn, args, iarg, eps=1e-6, step=1):
    if step == 1:
        return fd_basic(fcn, args, iarg, eps=eps)
    elif step == 2:
        fd1 = fd_basic(fcn, args, iarg, eps=eps)
        fd2 = fd_basic(fcn, args, iarg, eps=eps*1.5)
        return 1.8*fd1 - 0.8*fd2
    else:
        raise ValueError("Only step = 1 or 2 are supported.")

def fd_basic(fcn, args, iarg, eps=1e-6):
    with torch.no_grad():
        nelmt = args[iarg].numel()
        shape = args[iarg].shape
        device = args[iarg].device
        dxs = torch.eye(nelmt).to(device) * eps

        loss0 = fcn(*args)
        dlossdx = torch.empty(nelmt).to(args[iarg].dtype).to(device)
        for i in range(nelmt):
            newarg = args[iarg].detach() + dxs[i,:].view(shape)
            newargs = [(args[j] if j != iarg else newarg) for j in range(len(args))]
            loss = fcn(*newargs)

            newarg1 = args[iarg].detach() - dxs[i,:].view(shape)
            newargs1 = [(args[j] if j != iarg else newarg1) for j in range(len(args))]
            loss1 = fcn(*newargs1)
            dlossdx[i] = (loss - loss1).detach() / (2*eps)

        return dlossdx.view(shape)
