#!/usr/bin/env python-sirius
"""."""

import pickle as _pickle


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

    def save_data(self, fname):
        """."""
        data = dict(params=self.params, data=self.data)
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as fil:
            _pickle.dump(data, fil)

    @staticmethod
    def load_data(fname):
        """."""
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'rb') as fil:
            data = _pickle.load(fil)
        return data
