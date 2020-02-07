from lintorch.core.base import Module

__all__ = ["eye"]

class eye(Module):
    def __init__(self, nr):
        super(eye, self).__init__(shape=(nr,nr))

    def forward(self, x, *params):
        return x

    def precond(self, y, *params):
        return y
