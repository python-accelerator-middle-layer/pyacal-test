"""."""
from mathphys.functions import save_pickle as _save_pickle, \
    load_pickle as _load_pickle


class BaseClass:
    """."""

    def __init__(self, params=None):
        """."""
        self.params = params
        self.data = dict()
        self.devices = dict()
        self.pvs = dict()

    @property
    def connected(self):
        """."""
        conn = all([dev.connected for dev in self.devices.values()])
        conn &= all([pv.connected for pv in self.pvs.values()])
        return conn

    def wait_for_connection(self, timeout=None):
        """."""
        obs = list(self.devices.values()) + list(self.pvs.values())
        for dev in obs:
            if not dev.wait_for_connection(timeout=timeout):
                return False
        return True

    def save_data(self, fname, overwrite=False):
        """."""
        data = dict(params=self.params, data=self.data)
        _save_pickle(data, fname, overwrite=overwrite)

    def load_and_apply(self, fname):
        """."""
        data = self.load_data(fname)
        self.data = data['data']
        self.params = data['params']

    @staticmethod
    def load_data(fname):
        """."""
        return _load_pickle(fname)
