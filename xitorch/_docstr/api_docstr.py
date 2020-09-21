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
    * methods: list
        Dictionary with the method name as the keys and the method function or
        class as the value.

    Returns
    -------
    * full_docstr: str
        The full docstring of cls_or_func
    """
    method_template = """
    Method: {name}
    ========{namelength}
    """
    res = cls_or_func.__doc__
    for method in methods:
        name = method.__name__
        methoddoc = method.__doc__
        res = res + method_template.format(name=name, namelength=("="*len(name)))
        if methoddoc is not None:
            res = res + method.__doc__
    return res
