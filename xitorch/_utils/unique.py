from typing import List, Optional, Dict
from xitorch._utils.assertfuncs import assert_runtime

class Uniquifier(object):
    def __init__(self, allobjs: List):
        self.nobjs = len(allobjs)

        id2idx: Dict[int, int] = {}
        unique_objs: List[int] = []
        unique_idxs: List[int] = []
        nonunique_map_idxs: List[int] = [-self.nobjs * 2] * self.nobjs
        num_unique = 0
        for i, obj in enumerate(allobjs):
            id_obj = id(obj)
            if id_obj in id2idx:
                nonunique_map_idxs[i] = id2idx[id_obj]
                continue
            id2idx[id_obj] = num_unique
            unique_objs.append(obj)
            nonunique_map_idxs[i] = num_unique
            unique_idxs.append(i)
            num_unique += 1

        self.unique_objs = unique_objs
        self.unique_idxs = unique_idxs
        self.nonunique_map_idxs = nonunique_map_idxs
        self.num_unique = num_unique
        self.all_unique = self.nobjs == self.num_unique

    def get_unique_objs(self, allobjs: Optional[List] = None) -> List:
        if allobjs is None:
            return self.unique_objs
        assert_runtime(len(allobjs) == self.nobjs, "The allobjs must have %d elements" % self.nobjs)
        if self.all_unique:
            return allobjs
        return [allobjs[i] for i in self.unique_idxs]

    def map_unique_objs(self, uniqueobjs: List) -> List:
        assert_runtime(len(uniqueobjs) == self.num_unique, "The uniqueobjs must have %d elements" % self.num_unique)
        if self.all_unique:
            return uniqueobjs
        return [uniqueobjs[idx] for idx in self.nonunique_map_idxs]
