import torch
import xitorch as xt

@xt.module(shape=(25,25))
def A(x, diag):
    return x * diag

@A.set_precond
def precond(y, diag, biases=None):
    return y / diag

@xt.module_like(A)
def AA(x, diag2):
    return x*diag2*diag2

class Aclass(xt.Module):
    def forward(self, x, diag):
        return x*diag

    def precond(self, y, diag):
        return y/diag

eigvals, eigvecs = xt.lsymeig(A, (diag,), 3)
B = torch.random((nbatch, A.shape[1], 3))
c = xt.solve(A, (diag,), B)
