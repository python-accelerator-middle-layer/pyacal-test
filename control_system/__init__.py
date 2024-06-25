"""."""

import os

ControlSystemOptions = [fac for fac in os.listdir() if os.path.isdir(fac)]
