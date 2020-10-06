from xitorch._utils.unique import Uniquifier

def test_uniquifier():
    obj1 = [1]
    obj2 = [2]
    obj3 = [3]
    obj4 = [4]
    obj5 = [5]
    objs = [obj1, obj2, obj1, obj2, obj3, obj3]
    objsA = [obj2, obj1, obj2, obj1, obj4, obj4]

    uniq = Uniquifier(objs)
    unique_objs = uniq.get_unique_objs()
    assert len(unique_objs) == 3
    assert unique_objs[0] is obj1
    assert unique_objs[1] is obj2
    assert unique_objs[2] is obj3

    unique_objsA = uniq.get_unique_objs(objsA)
    assert len(unique_objsA) == 3
    assert unique_objsA[0] is obj2
    assert unique_objsA[1] is obj1
    assert unique_objsA[2] is obj4

    unique_objs2 = [obj3, obj4, obj5]
    objs2 = uniq.map_unique_objs(unique_objs2)
    assert len(objs2) == len(objs)
    assert objs2[0] is obj3
    assert objs2[1] is obj4
    assert objs2[2] is obj3
    assert objs2[3] is obj4
    assert objs2[4] is obj5
    assert objs2[5] is obj5

def test_uniquifier_error():
    obj1 = [1]
    obj2 = [2]
    obj3 = [3]
    obj4 = [4]
    objs = [obj1, obj2, obj1, obj1, obj2]
    objs2 = [obj1, obj2, obj1, obj1, obj2, obj2]

    uniq = Uniquifier(objs)
    try:
        unique_objs = uniq.get_unique_objs(objs2)
        assert False, "Expected a RuntimeError"
    except RuntimeError:
        pass

    try:
        objs3 = uniq.map_unique_objs(objs)
        assert False, "Expected a RuntimeError"
    except RuntimeError:
        pass
