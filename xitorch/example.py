import torch
import lintorch as lt

@lt.module(shape=(25,25))
def A(x, diag):
    return x * diag

@A.set_precond
def precond(y, diag, biases=None):
    return y / diag

@lt.module_like(A)
def AA(x, diag2):
    return x*diag2*diag2

class Aclass(lt.Module):
    def forward(self, x, diag):
        return x*diag

    def precond(self, y, diag):
        return y/diag

eigvals, eigvecs = lt.lsymeig(A, (diag,), 3)
B = torch.random((nbatch, A.shape[1], 3))
c = lt.solve(A, (diag,), B)
