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
    Method: {name}
    ========{namelength}
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
        res = res + method_template.format(name=name, namelength=("="*len(name)))
        if methoddoc is not None:
            res = res + method.__doc__
    return res
