from abc import abstractmethod

class BaseInterp(object):
    @abstractmethod
    def __init__(self, x, y=None, **kwargs):
        pass

    @abstractmethod
    def __call__(self, xq, y=None):
        pass

    @abstractmethod
    def getparamnames(self):
        pass
