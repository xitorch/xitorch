import inspect
import warnings
import functools

def deprecated(date_str):
    return lambda obj: _deprecated(obj, date_str)

def _deprecated(obj, date_str):
    if inspect.isfunction(obj):
        name = "Function %s" % (obj.__str__())
    elif inspect.isclass(obj):
        name = "Class %s" % (obj.__name__)

    if inspect.ismethod(obj) or inspect.isfunction(obj):
        @functools.wraps(obj)
        def fcn(*args, **kwargs):
            warnings.warn(
                "%s is deprecated since %s" % (name, date_str),
                stacklevel=2)
            return obj(*args, **kwargs)
        return fcn

    elif inspect.isclass(obj):
        # replace the __init__ function
        old_init = obj.__init__

        @functools.wraps(old_init)
        def newinit(*args, **kwargs):
            warnings.warn(
                "%s is deprecated since %s" % (name, date_str),
                stacklevel=2)
            return old_init(*args, **kwargs)
        obj.__init__ = newinit
        return obj
