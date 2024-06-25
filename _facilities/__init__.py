"""."""

import os

FacilityOptions = [fac for fac in os.listdir() if os.path.isdir(fac)]


class FacilityBase:
    """."""

    def __init__(self, name, control, simulator):
        """."""
        self.name = name
        self.control = control
        self.simulator = simulator
        self.alias_map = {}
        self.accelerators = {}
        self.default_accelerator = ''
