from abc import abstractmethod
from contextlib import contextmanager

__all__ = ["EditableModule"]

class EditableModule(object):
    @abstractmethod
    def getparams(self):
        """
        Returns a list of tensor parameters used in the object's operations
        """
        pass

    @abstractmethod
    def setparams(self, *params):
        """
        Set the input parameters to the object's parameters to make a copy of
        the operations.
        """
        pass

    @contextmanager
    def useparams(self, *params):
        try:
            _orig_params_ = self.getparams()
            self.setparams(*params)
            yield self
        finally:
            self.setparams(*_orig_params_)
