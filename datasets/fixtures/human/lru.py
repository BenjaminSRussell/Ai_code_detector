from collections import OrderedDict


class LRU:
    def __init__(self, cap):
        self.cap = cap
        self._d = OrderedDict()

    def get(self, k, default=None):
        if k not in self._d:
            return default
        self._d.move_to_end(k)
        return self._d[k]

    def put(self, k, v):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.cap:
            self._d.popitem(last=False)
