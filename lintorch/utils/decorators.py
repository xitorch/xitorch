import inspect
import warnings

def deprecated(obj):
    if inspect.ismethod(obj):
        name = "Method %s.%s" % (obj.__class__.__name__, obj.__name__)
    elif inspect.isfunction(obj):
        name = "Function %s" % (obj.__name__)
    elif inspect.isclass(obj):
        name = "Class %s" % (obj.__name__)

    if inspect.ismethod(obj) or inspect.isfunction(obj):
        def fcn(*args, **kwargs):
            warnings.warn("%s is deprecated" % name, stacklevel=2)
            return obj(*args, **kwargs)
        return fcn

    elif inspect.isclass(obj):
        # replace the __init__ function
        old_init = obj.__init__
        def newinit(*args, **kwargs):
            warnings.warn("%s is deprecated" % name, stacklevel=2)
            return old_init(*args, **kwargs)
        obj.__init__ = newinit
        return obj
