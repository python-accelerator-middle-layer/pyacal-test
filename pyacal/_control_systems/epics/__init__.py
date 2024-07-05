"""Epics-related subpackage."""

from .._handle_pvs import create_pv, get_pv
from .multiprocessing import ProcessSpawn
from .pv import PV
from .threading import Thread

del pv
del multiprocessing
del threading

Name = 'epics'
