"""Epics-related subpackage."""

import multiprocessing as _mp
from threading import Thread

from .._handle_pvs import create_pv, get_pv
from .pv import PV

del pv

Name = 'simulation'
ProcessSpawn = _mp.get_context('spawn').Process
