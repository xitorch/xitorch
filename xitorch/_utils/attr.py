import re
import ast

__all__ = ["get_attr", "set_attr", "del_attr"]

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
    dictfcn = lambda obj, key: obj.__getitem__(key)
    listfcn = lambda obj, key: obj.__getitem__(key)
    return _traverse_attr(obj, names, attrfcn, dictfcn, listfcn)

def _set_attr(obj, names, val):
    attrfcn = lambda obj, name: setattr(obj, name, val)
    dictfcn = lambda obj, key: obj.__setitem__(key, val)
    listfcn = lambda obj, key: obj.__setitem__(key, val)
    return _traverse_attr(obj, names, attrfcn, dictfcn, listfcn)

def _del_attr(obj, names):
    attrfcn = lambda obj, name: delattr(obj, name)
    dictfcn = lambda obj, key: obj.__delitem__(key)

    def listfcn(obj, key):
        obj.__delitem__(key)
        obj.insert(key, None)  # to preserve the length
    return _traverse_attr(obj, names, attrfcn, dictfcn, listfcn)


def _preproc_name(name):
    return sp.findall(name)

def _traverse_attr(obj, names, attrfcn, dictfcn, listfcn):
    if len(names) == 1:
        return _applyfcn(obj, names[0], attrfcn, dictfcn, listfcn)
    else:
        return _applyfcn(_get_attr(obj, names[:-1]), names[-1], attrfcn, dictfcn, listfcn)

def _applyfcn(obj, name, attrfcn, dictfcn, listfcn):
    if name[0] == "[":
        key = ast.literal_eval(name[1:-1])
        if isinstance(obj, dict):
            return dictfcn(obj, key)
        elif isinstance(obj, list):
            return listfcn(obj, key)
        else:
            msg = "The parameter with [] must be either a dictionary or a list. "
            msg += "Got type: %s" % type(obj)
            raise TypeError(msg)
    else:
        return attrfcn(obj, name)
