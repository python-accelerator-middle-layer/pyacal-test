"""."""

import os

ControlSystemOptions = [
    fac for fac in os.listdir(__file__) if os.path.isdir(fac)
]
