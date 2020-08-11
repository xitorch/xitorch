import re
import ast

__all__ = ["get_attr", "set_attr", "del_attr", "has_attr"]

# pattern to split the names, e.g. "model.params[1]" into ["model", "params", "[1]"]
sp = re.compile(r"\[{0,1}[\"']{0,1}\w+[\"']{0,1}\]{0,1}")

def get_attr(obj, name):
    return _get_attr(obj, _preproc_name(name))

def set_attr(obj, name, val):
    return _set_attr(obj, _preproc_name(name), val)

def del_attr(obj, name):
    return _del_attr(obj, _preproc_name(name))


def _get_attr(obj, names):
    attrfcn = lambda obj, name: getattr(obj, name)
    itemfcn = lambda obj, key: obj.__getitem__(key)
    return _traverse_attr(obj, names, attrfcn, itemfcn)

def _set_attr(obj, names, val):
    attrfcn = lambda obj, name: setattr(obj, name, val)
    itemfcn = lambda obj, key: obj.__setitem__(key, val)
    return _traverse_attr(obj, names, attrfcn, itemfcn)

def _del_attr(obj, names):
    attrfcn = lambda obj, name: delattr(obj, name)
    itemfcn = lambda obj, key: obj.__delitem__(key)
    return _traverse_attr(obj, names, attrfcn, itemfcn)


def _preproc_name(name):
    return sp.findall(name)

def _traverse_attr(obj, names, attrfcn, itemfcn):
    if len(names) == 1:
        if names[0].startswith("["):
            key = ast.literal_eval(names[0][1:-1])
            if not hasattr(obj, "keys"):
                raise TypeError("The parameter with [] must be a dictionary")
            return itemfcn(obj, key)
        else:
            return attrfcn(obj, names[0])
    else:
        return attrfcn(_get_attr(obj, names[:-1]), names[-1])
