import inspect
import functools

"""
Filler decorator is applied to methods of classes.
It fills some parameters with default parameters from the classes parameters.
Its role is to enable listing of parameters that have roles in the function so
    that the autograd backward can propagate the gradient and higher order
    gradient to the parameters.
"""

__all__ = ["filler", "clsfiller", "is_with_filler"]

def filler(**def_kwargs):
    ndef_kwargs = len(def_kwargs)

    def filler_decorator(fcn):
        # check the signature (it cannot have *args and **kwargs at the moment)
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(fcn)
        if varargs is not None and varkw is not None:
            raise RuntimeError("The filler decorator cannot accept functions with *args and **kwargs")

        # get the number of parameters
        sig = inspect.signature(fcn)
        nsig = len(sig.parameters)

        @functools.wraps(fcn)
        def wrapper(instance, *args):
            nparams = len(args) + 1 # +1 for self
            if nparams == nsig:
                return fcn(instance, *args)
            elif nparams + ndef_kwargs >= nsig:
                noffset = nparams + ndef_kwargs - nsig
                def_params = [def_kwargs[k](instance) for k in def_kwargs][noffset:]
                return fcn(instance, *args, *def_params)

        def set_default_params(wrapper, instance):
            wrapper.def_params = [wrapper.def_kwargs[k](instance) for k in def_kwargs]

        # set the attribute of the methods
        wrapper.is_with_filler = True
        wrapper.def_kwargs = def_kwargs
        wrapper.def_params = None
        wrapper.set_def_params = lambda instance: set_default_params(wrapper, instance)

        return wrapper
    return filler_decorator

def clsfiller(cls):
    # wraps a class so when an instance is created, all methods with filler
    # will set up its default parameters, so it can be accessed

    @functools.wraps(cls)
    def wrapper_cls(*args, **kwargs):
        # create the instance
        instance = cls(*args, **kwargs)
        methods = inspect.getmembers(instance, predicate=inspect.ismethod)
        for name, method in methods:
            if is_with_filler(method):
                method.set_def_params(instance)
        return instance
    return wrapper_cls

def is_with_filler(method):
    return hasattr(method, "is_with_filler")

if __name__ == "__main__":
    @clsfiller
    class A:
        def __init__(self, a=1):
            self.a = a
            self.a2 = 2*a

        @filler(self_a=lambda self:self.a,
                self_a2=lambda self:self.a2)
        def b(self, a, self_a, self_a2):
            return self_a + a + self_a2

    print("Class created")
    a = A(1)
    aa = A(2)
    print(a.b.def_params)
    print(is_with_filler(a.b))
    print("Instance created")
    print(a.b(1))
    print(a.b(1, *aa.b.def_params))
    print(a.b(1, 2))
