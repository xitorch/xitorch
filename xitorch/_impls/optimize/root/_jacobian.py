import torch
from abc import abstractmethod

# taking most of the part from SciPy

class Jacobian(object):
    """
    Base class for the Jacobians used in rootfinder algorithms.
    """
    @abstractmethod
    def setup(self, x0, y0, func):
        pass

    @abstractmethod
    def solve(self, v, tol=0):
        pass

    @abstractmethod
    def update(self, x, y):
        pass

class BroydenFirst(Jacobian):
    def __init__(self, alpha=None, max_rank=None):
        self.alpha = alpha
        self.max_rank = max_rank

    def setup(self, x0, y0):
        self.x_prev = x0
        self.y_prev = y0

        if self.max_rank is None:
            self.max_rank = float('inf')

        if self.alpha is None:
            normy0 = torch.norm(y0)
            ones = torch.ones_like(normy0)
            if normy0:
                self.alpha = 0.5 * torch.max(torch.norm(x0), ones) / normy0
            else:
                self.alpha = ones

        # setup the approximate inverse Jacobian
        self.Gm = LowRankMatrix(-self.alpha, "restart")
        self._reduce = lambda: self.Gm.reduce(self.max_rank)

    def solve(self, v, tol=0):
        res = self.Gm.mv(v)
        return res

    def update(self, x, y):
        dy = y - self.y_prev
        dx = x - self.x_prev
        # update Gm
        self._update(x, y, dx, dy, dx.norm(), dy.norm())

        self.y_prev = y
        self.x_prev = x

    def _update(self, x, y, dx, dy, dxnorm, dynorm):
        # keep the rank small
        self._reduce()

        v = self.Gm.rmv(dx)
        c = dx - self.Gm.mv(dy)
        d = v / torch.dot(dy, v)
        self.Gm = self.Gm.append(c, d)

class LowRankMatrix(object):
    # represents a matrix of `\alpha * I + \sum_n c_n d_n^T`
    def __init__(self, alpha, reduce_method):
        self.alpha = alpha
        self.cns = []
        self.dns = []
        self.reduce_method = {
            "restart": 0,
            "simple": 1
        }[reduce_method]

    def mv(self, v):
        res = self.alpha * v
        for i in range(len(self.dns)):
            res += self.cns[i] * torch.dot(self.dns[i], v)
        return res

    def rmv(self, v):
        res = self.alpha * v
        for i in range(len(self.dns)):
            res += self.dns[i] * torch.dot(self.cns[i], v)
        return res

    def append(self, c, d):
        self.cns.append(c)
        self.dns.append(d)
        if len(self.cns) >= torch.numel(c):
            return FullRankMatrix(self.alpha, self.cns, self.dns)
        return self

    def reduce(self, max_rank, **otherparams):
        if len(self.cns) > max_rank:
            if self.reduce_method == 0: # restart
                del self.cns[:]
                del self.dns[:]
            elif self.reduce_method == 1: # simple
                n = len(self.cns)
                del self.cns[:n-max_rank]
                del self.dns[:n-max_rank]

class FullRankMatrix(object):
    def __init__(self, alpha, cns, dns):
        size = torch.numel(cns[0])
        dtype, device = cns[0].dtype, cns[0].device
        self.mat = torch.eye(size, dtype=dtype, device=device)
        self.mat *= alpha
        for i in range(len(cns)):
            self.mat += torch.ger(cns[i], dns[i])

    def mv(self, v):
        res = torch.matmul(self.mat, v)
        return res

    def rmv(self, v):
        return torch.matmul(self.mat.T, v)

    def append(self, c, d):
        self.mat += torch.ger(c, d)
        return self

    def reduce(self, max_rank, **kwargs):
        pass # ???
