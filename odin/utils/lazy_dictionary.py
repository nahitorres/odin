from collections import MutableMapping


class LazyDict(MutableMapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        data = self._raw_dict.__getitem__(key)
        if isinstance(data, tuple):
            if len(data) > 1:
                func, *args = data
                if callable(func):
                    value = func(*args)
                    self._raw_dict[key] = value
                    return value
            elif callable(data):
                value = data()
                self._raw_dict[key] = value
                return value
        return data

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)

    def __delitem__(self, v):
        del self._raw_dict[v]

    def __setitem__(self, k, v):
        self._raw_dict[k] = v
