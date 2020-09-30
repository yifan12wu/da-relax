

def get_flags(flags, key, default_val=None):
    if hasattr(flags, key):
        return getattr(flags, key)
    else:
        return default_val


class EmptyClass(object):
    pass


class BaseConfig(object):

    def __init__(self, flags):
        self._flags = self._default_flags()
        self._update_flags(flags)
        self._setup_flags()
        self._build()

    def _default_flags(self):
        flags = EmptyClass()
        return flags

    def _update_flags(self, flags):
        flags_dict = vars(flags)
        for key, val in flags_dict.items():
            if key[0] != '_' and hasattr(self._flags, key):
                setattr(self._flags, key, val)

    def _setup_flags(self):
        pass

    def _build(self):
        pass
