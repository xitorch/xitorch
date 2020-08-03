import torch
from abc import abstractmethod

class _BaseCModule(torch.nn.Module):
    # dummy base module for Module, to check the instance within the module definition
    pass

class Module(_BaseCModule):
    def __init__(self):
        self.__dict__["_cparameters"] = {}
        super(Module, self).__init__()

    def register(self, x):
        if isinstance(x, torch.Tensor):
            return CParameter(x)
        elif hasattr(x, "__iter__"):
            # check is performed in the CParameterList
            return CParameterList(x)
        elif isinstance(x, torch.nn.Parameter) or \
             isinstance(x, torch.nn.Module) or \
             isinstance(x, _BaseCModule):
            return x
        else:
            raise RuntimeError("Type %s cannot be registered" % type(x))

    ################## torch.nn.Module overriden functions ##################
    def parameters(self):
        for name, val in self._param_generator():
            yield val

    def named_parameters(self):
        for name, val in self._param_generator():
            yield name, val

    def _param_generator(self):
        for name,val in self._cparameters.items():
            yield name,val
        for name,module in self.named_children():
            for mod_name,mod_val in module.named_parameters():
                fullname = "%s.%s"%(name,mod_name)
                yield fullname, mod_val

    ################## __*attr__ functions ##################
    def __setattr__(self, name, value):
        if ("_cparameters" not in self.__dict__):
            raise RuntimeError("__init__() must be called before doing assignments")

        # substituting the old parameters (must retain the type)
        if name in self._cparameters:
            # TODO: decide on what type can substitute the old parameters
            if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
                self._cparameters[name] = value
            else:
                raise TypeError("Cannot assign type %s to self.%s (torch.Tensor and not torch.nn.Parameter is required)" %\
                    (type(value), name))

        # adding new parameters
        else:
            # TODO: add type
            if isinstance(value, CParameter):
                self._cparameters[name] = value.tensor

            # regular type
            else:
                super(Module, self).__setattr__(name, value)

    def __getattr__(self, name):
        # called when `name` is not in the usual place
        if name in self._cparameters:
            return self._cparameters[name]

        return super(Module, self).__getattr__(name)

    def __delattr__(self, name):
        if name in self._cparameters:
            del self._cparameters[name]
            return

        super(Module, self).__delattr__(name)

class CParameter(object):
    # CParameter is needed to differentiate the values that are going to be
    # registered from the ordinary values
    def __init__(self, x):
        self._val = x

    @property
    def tensor(self):
        return self._val

class CParameterList(Module):
    def __init__(self, xlist):
        super(CParameterList, self).__init__()
        self._cparamlen = len(xlist)
        for i,x in enumerate(xlist):
            if not isinstance(x, torch.Tensor):
                raise TypeError("The %d-th element is not a tensor" % i)
            setattr(self, "%d"%i, self.register(x))

    def __getitem__(self, key):
        if key < 0:
            key = key + self._cparamlen
        if key >= self._cparamlen:
            raise IndexError("Cannot access index %d from list with %d elements" % (key, self._cparamlen))
        return getattr(self, "%d"%key)

    def __len__(self):
        return self._cparamlen

if __name__ == "__main__":
    class NNModule(torch.nn.Module):
        def __init__(self, a):
            super(NNModule, self).__init__()
            self.a = torch.nn.Parameter(a)

    class NewModule(Module):
        def __init__(self, a, b):
            super(NewModule, self).__init__()
            self.ab = self.register([a, b])

    class Module2(Module):
        def __init__(self, amod, at, a):
            super(Module2, self).__init__()
            self.mod = amod
            self.modt = at
            self.a = self.register(a)

    atorch = torch.tensor([1.])
    btorch = torch.tensor([2.])
    a = NewModule(atorch, btorch)
    at = NNModule(atorch)
    a2 = Module2(a, at, atorch)
    print(list(a2.named_parameters()))
