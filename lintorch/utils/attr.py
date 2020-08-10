__all__ = ["get_attr", "set_attr", "del_attr", "has_attr"]

def get_attr(obj, name):
    return _get_attr(obj, name.split("."))

def set_attr(obj, name, val):
    return _set_attr(obj, name.split("."), val)

def del_attr(obj, name):
    return _del_attr(obj, name.split("."))

def has_attr(obj, name):
    return _has_attr(obj, name.split("."))


def _get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return getattr(_get_attr(obj, [names[0]]), names[1])

def _set_attr(obj, names, val):
    if len(names) == 1:
        return setattr(obj, names[0], val)
    else:
        return setattr(_get_attr(obj, [names[0]]), names[1], val)

def _del_attr(obj, names):
    if len(names) == 1:
        return delattr(obj, names[0])
    else:
        return delattr(_get_attr(obj, [names[0]]), names[1])

def _has_attr(obj, names):
    if len(names) == 1:
        return hasattr(obj, names[0])
    elif not hasattr(obj, names[0]):
        return False
    else:
        return hasattr(_get_attr(obj, [names[0]]), names[1])
