"""."""

FacilityOptions = ('esrf', 'sirius', 'soleil')


class FacilityBase:
    """."""

    def __init__(self, name, control_system, simulator):
        """."""
        self.name = name
        self.control_system = control_system
        self.simulator = simulator
        self.alias_map = dict()
        self.accelerators = dict()
        self.default_accelerator = ''
