def get_methods_docstr(cls_or_func, methods):
    """
    Get the full docstring of a class or a function. Full docstring is the
    main docstring of the class or function plus docstrings of the methods
    (usually to describe the options available specifically for a specific
    method).

    Arguments
    ---------
    * cls_or_func: class or callable
        Main class or function
    * methods: list or dict
        If a list, it lists the methods with the same method name options.
        If a dict, it contains the method name as the keys and the method
        function or class as the value.

    Returns
    -------
    * full_docstr: str
        The full docstring of cls_or_func
    """
    method_template = """
    Methods
    -------
    method="{name}"
    """
    res = cls_or_func.__doc__

    if isinstance(methods, dict):
        generator = methods.items()
    elif isinstance(methods, list):
        generator = ((method.__name__, method) for method in methods)
    else:
        raise TypeError("methods must be a list or a dict")

    for name, method in generator:
        methoddoc = method.__doc__
        res = res + method_template.format(name=name)
        if methoddoc is not None:
            method_doc = _method_indent(method.__doc__)
            res = res + method_doc
    return res

def _method_indent(method_doc):
    # add indentation to all parts of method doc except the args part
    # this is due to styling in Sphinx to make the "method" description
    # compatible with Sphinx styling

    lines = method_doc.split("\n")
    prev_line_strip = lines[0].strip()
    indent = " "*4
    arg_list = ["args", "arguments", "parameters", "params"]
    i0_arg = len(lines)
    i1_arg = len(lines)
    in_arg = False
    for i,line in enumerate(lines[1:], start=1):
        line_strip = line.strip()
        underline = line_strip[0] * len(line_strip) if len(line_strip) > 0 else "invalid"
        if prev_line_strip.lower() in arg_list and line_strip == underline:
            i0_arg = i-1
            if in_arg:
                raise RuntimeError("Unexpected arguments header")
            in_arg = True
        elif in_arg and line_strip == underline:
            i1_arg = i-2
            in_arg = False
        prev_line_strip = line_strip

    # add the indentation of all part, except args
    for i,line in enumerate(lines):
        # not in arg mode
        if i < i0_arg or i >= i1_arg or True:
            line = indent + line
            lines[i] = line
    return "\n".join(lines)

def _add_indent(s, indent):
    return "\n".join([indent+line for line in s.split("\n")])
