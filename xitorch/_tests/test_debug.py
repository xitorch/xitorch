import xitorch

def test_debug_default():
    assert not xitorch.is_debug_enabled()

def test_set_debug_mode():
    xitorch.set_debug_mode(True)
    assert xitorch.is_debug_enabled()
    xitorch.set_debug_mode(False)
    assert not xitorch.is_debug_enabled()

def test_debug_context_manager():
    assert not xitorch.is_debug_enabled()
    with xitorch.enable_debug():
        assert xitorch.is_debug_enabled()
        with xitorch.disable_debug():
            assert not xitorch.is_debug_enabled()
        assert xitorch.is_debug_enabled()
    assert not xitorch.is_debug_enabled()
