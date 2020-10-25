from contextlib import contextmanager

__all__ = ["is_debug_enabled", "set_debug_mode", "enable_debug", "disable_debug"]

class DebugSingleton(object):
    class __DebugSingleton:
        def __init__(self):
            self._isdebug = False  # default mode is not in the debug mode

        def set_debug_mode(self, mode):
            self._isdebug = mode

        def get_debug_mode(self):
            return self._isdebug

    instance = None

    def __init__(self):
        if DebugSingleton.instance is None:
            DebugSingleton.instance = DebugSingleton.__DebugSingleton()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, val):
        return setattr(self.instance, name, val)

def set_debug_mode(mode):
    dbg_obj = DebugSingleton()
    dbg_obj.set_debug_mode(mode)

def is_debug_enabled():
    dbg_obj = DebugSingleton()
    return dbg_obj.get_debug_mode()

@contextmanager
def enable_debug():
    try:
        dbg_mode = is_debug_enabled()
        set_debug_mode(True)
        yield
    except Exception as e:
        raise e
    finally:
        set_debug_mode(dbg_mode)

@contextmanager
def disable_debug():
    try:
        dbg_mode = is_debug_enabled()
        set_debug_mode(False)
        yield
    except Exception as e:
        raise e
    finally:
        set_debug_mode(dbg_mode)
