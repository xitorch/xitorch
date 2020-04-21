from abc import abstractmethod
from contextlib import contextmanager

__all__ = ["EditableModule"]

class EditableModule(object):
    @abstractmethod
    def getparams(self, methodname):
        """
        Returns a list of tensor parameters used in the object's operations
        """
        pass

    @abstractmethod
    def setparams(self, methodname, *params):
        """
        Set the input parameters to the object's parameters to make a copy of
        the operations.
        """
        pass

    @contextmanager
    def useparams(self, methodname, *params):
        try:
            _orig_params_ = self.getparams(methodname)
            self.setparams(methodname, *params)
            yield self
        finally:
            self.setparams(methodname, *_orig_params_)
