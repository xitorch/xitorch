import random
import torch
from torch.autograd import gradcheck, gradgradcheck
from comptorch.core.module import Module
from comptorch.nlfcns.rootfinder import rootfinder, equilibrium

class DummyModule(Module):
    def __init__(self, A, addx=True):
        super(DummyModule, self).__init__()
        self.A = self.register(A) # (nr, nr)
        self.addx = addx
        self.A2 = self.register(A * 2)
        self.A3 = self.register(self.A2 + 1.)
        self.sigmoid = torch.nn.Sigmoid()

    def set_diag_bias(self, diag, bias):
        self.diag = self.register(diag)
        self.bias = self.register(bias)

    def forward(self, x):
        # x: (nbatch, nr)
        # diag: (nbatch, nr)
        # bias: (nbatch, nr)
        nbatch, nr = x.shape
        x = x.unsqueeze(-1)
        A = self.A.unsqueeze(0).expand(nbatch, -1, -1) # (nbatch, nr, nr)
        A = A + torch.diag_embed(self.diag) # (nbatch, nr, nr)
        y = torch.bmm(A, x).squeeze(-1) # (nbatch, nr)
        yr = self.sigmoid(2*y) + 2*self.bias
        if self.addx:
            yr = yr + x.squeeze(-1)
        return yr

class DummyNNModule(torch.nn.Module):
    def __init__(self, A, addx=True):
        super(DummyNNModule, self).__init__()
        self.A = A
        self.addx = addx
        self.sigmoid = torch.nn.Sigmoid()

    def set_diag_bias(self, diag, bias):
        self.diag = diag
        self.bias = bias

    def forward(self, x):
        # x: (nbatch, nr)
        # diag: (nbatch, nr)
        # bias: (nbatch, nr)
        nbatch, nr = x.shape
        x = x.unsqueeze(-1)
        A = self.A.unsqueeze(0).expand(nbatch, -1, -1) # (nbatch, nr, nr)
        A = A + torch.diag_embed(self.diag) # (nbatch, nr, nr)
        y = torch.bmm(A, x).squeeze(-1) # (nbatch, nr)
        yr = self.sigmoid(2*y) + 2*self.bias
        if self.addx:
            yr = yr + x.squeeze(-1)
        return yr

def test_rootfinder():
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1

    for clss in [DummyModule, DummyNNModule][:1]:
        A    = (torch.randn((nr, nr))*0.5).to(dtype).requires_grad_()
        diag = torch.randn((nbatch, nr)).to(dtype).requires_grad_()
        bias = torch.zeros((nbatch, nr)).to(dtype).requires_grad_()
        if clss is DummyNNModule:
            A = torch.nn.Parameter(A)
            diag = torch.nn.Parameter(diag)
            bias = torch.nn.Parameter(bias)

        y0 = torch.randn((nbatch, nr)).to(dtype)

        model = clss(A, addx=True)
        model.set_diag_bias(diag, bias)

        y = rootfinder(model.forward, y0)
        f = model.forward(y)
        assert torch.allclose(f*0, f)

        def getloss(A, y0, diag, bias):
            model = clss(A, addx=True)
            model.set_diag_bias(diag, bias)
            y = rootfinder(model.forward, y0)
            return y

        gradcheck(getloss, (A, y0, diag, bias))
        gradgradcheck(getloss, (A, y0, diag, bias))

def test_equil():
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1

    for clss in [DummyModule, DummyNNModule][:1]:
        A    = (torch.randn((nr, nr))*0.5).to(dtype).requires_grad_()
        diag = torch.randn((nbatch, nr)).to(dtype).requires_grad_()
        bias = torch.zeros((nbatch, nr)).to(dtype).requires_grad_()
        if clss is DummyNNModule:
            A = torch.nn.Parameter(A)
            diag = torch.nn.Parameter(diag)
            bias = torch.nn.Parameter(bias)
        y0 = torch.randn((nbatch, nr)).to(dtype)

        model = DummyModule(A, addx=False)
        model.set_diag_bias(diag, bias)
        y = equilibrium(model.forward, y0)
        f = model.forward(y)
        assert torch.allclose(y, f)

        def getloss(A, y0, diag, bias):
            model = DummyModule(A, addx=False)
            model.set_diag_bias(diag, bias)
            y = equilibrium(model.forward, y0)
            return y

        gradcheck(getloss, (A, y0, diag, bias))
        gradgradcheck(getloss, (A, y0, diag, bias))
