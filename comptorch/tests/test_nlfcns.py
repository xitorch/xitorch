import random
import torch
from torch.autograd import gradcheck, gradgradcheck
from comptorch.core.module import CModule
from comptorch.nlfcns.rootfinder import rootfinder, equilibrium

class DummyModule(CModule):
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
    def assert_rf_fcn(model, y):
        f = model.forward(y)
        assert torch.allclose(f*0, f)

    assert_nlfcns(rootfinder, assert_rf_fcn, addx=True)

def test_equil():
    def assert_equil_fcn(model, y):
        f = model.forward(y)
        assert torch.allclose(y, f)

    assert_nlfcns(equilibrium, assert_equil_fcn, addx=False)

def assert_nlfcns(nlfcns, assert_fcn, addx):
    torch.manual_seed(100)
    random.seed(100)
    dtype = torch.float64

    nr = 4
    nbatch = 1

    for clss in [DummyModule, DummyNNModule]:
        # if the class is DummyModule, it should take Parameter and non-Parameter
        # because it registers all the tensors
        to_paramss = [True, False] if clss is DummyModule else [True]

        for to_params in to_paramss:
            # set up the nonlinear function parameters
            A    = (torch.randn((nr, nr))*0.5).to(dtype).requires_grad_()
            diag = torch.randn((nbatch, nr)).to(dtype).requires_grad_()
            bias = torch.zeros((nbatch, nr)).to(dtype).requires_grad_()
            if to_params:
                A = torch.nn.Parameter(A)
                diag = torch.nn.Parameter(diag)
                bias = torch.nn.Parameter(bias)

            # set up the initial guess
            y0 = torch.randn((nbatch, nr)).to(dtype)

            model = clss(A, addx=addx)
            model.set_diag_bias(diag, bias)
            y = nlfcns(model.forward, y0)
            assert_fcn(model, y)
            # f = model.forward(y)
            # assert torch.allclose(y, f)

            def getloss(A, y0, diag, bias):
                model = clss(A, addx=addx)
                model.set_diag_bias(diag, bias)
                y = nlfcns(model.forward, y0)
                return y

            gradcheck(getloss, (A, y0, diag, bias))
            gradgradcheck(getloss, (A, y0, diag, bias))
