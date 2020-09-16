import ast
import typing as tp
from importlib import import_module

class TypeParser(ast.NodeVisitor):
    """
    TypeParser evaluate the possible type of the members of a class.
    """

    def __init__(self):
        self.typed_classes = []
        # to be able to get the actual type, we need to import what the file imports
        # TODO: how about the type is defined in the file (???)
        self.imported_modules = {}
        self.imported_obj = {}
        self.deftypes = ["int", "float", "str", "bool"] # default types in Python

    def visit_Import(self, node):
        # save the imported modules in the dictionary with the alias name as the key
        for imp in node.names:
            name = imp.name if imp.asname is None else imp.asname
            self.imported_modules[name] = import_module(imp.name)

    def visit_ImportFrom(self, node):
        # save the imported objects in the dictionary with the alias name as the key
        module = import_module(node.module)
        for attr in node.names:
            name = attr.name if attr.asname is None else attr.asname
            self.imported_obj[name] = getattr(module, attr.name)

    def visit_ClassDef(self, node):
        cls = TypedClass(node.name)
        for nd in node.body:
            if not isinstance(nd, ast.FunctionDef):
                continue
            methodname = nd.name
            methodtype = self.get_type(nd.returns)
            cls.add_method_rettype(methodname, methodtype)

            # TODO: evaluate the variables

        self.typed_classes.append(cls)

    def get_type(self, node):
        if node is None:
            return tp.Any
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
            s = self.get_str(node)
            return self.eval_type(s)
        elif isinstance(node, ast.Subscript):
            return self.get_type(node.value)[self.get_type(node.slice.value)]
        elif isinstance(node, ast.Tuple):
            return tuple([self.get_type(elt) for elt in node.elts])

    def eval_type(self, s):
        # search s in the imported object first
        if s in self.imported_obj.keys():
            return self.imported_obj[s]

        # then check if it is Python keyword, then eval_type
        elif s in self.deftypes:
            return eval(s)

        # then search in the module
        # this is a bit tricky because the object is described by [module].[objname]
        else:
            modulenames = list(self.imported_modules.keys())
            modulenames = sorted(modulenames, key=lambda x: -len(x))
            # print(modulenames, s, type(s), s[len("torch")+1])
            for modulename in modulenames:
                if s.startswith(modulename) and s[len(modulename)] == ".":

                    # it can have multiple dots, so we need to traverse the attributes
                    attrs = s[len(modulename)+1:].split(".")
                    obj = self.imported_modules[modulename]
                    for attr in attrs:
                        obj = getattr(obj, attr)
                    return obj

            # if it can reach this point, then no match found
            raise RuntimeError("There is no match for %s" % s)

    def get_str(self, node):
        if isinstance(node, str):
            return node
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return "%s.%s" % (self.get_str(node.value), self.get_str(node.attr))
        elif isinstance(node, ast.Subscript):
            return "%s[%s]" % (self.get_str(node.value), self.get_str(node.slice))

class TypedClass(object):
    def __init__(self, clsname):
        self.clsname = clsname
        self.var_types = {}
        self.method_rettypes = {}

    def add_var_type(self, varname, vartype):
        self._add_type(self.var_types, varname, vartype)

    def add_method_rettype(self, methodname, methodtype):
        self._add_type(self.method_rettypes, methodname, methodtype)

    def _add_type(self, dct, name, tpe):
        if tpe is None:
            # override all the types
            dct[name] = tpe

        if name not in dct:
            dct[name] = tpe
        else:
            dct[name] = tp.Union[dct[name], tpe]

    def __str__(self):
        s = "Typed Class: %s\n" % self.clsname
        s += "Methods (returned type):\n"
        for methodname, methodtype in self.method_rettypes.items():
            s += "* %s: %s\n" % (methodname, methodtype)
        return s

if __name__ == "__main__":
    with open("../linop/base.py", "r") as f:
        tree = ast.parse(f.read())

    t = TypeParser()
    t.visit(tree)
    for cls in t.typed_classes:
        print(str(cls))
