"""Epics-related subpackage."""

from threading import Thread
import multiprocessing as _mp

from .pv import PV, ALL_CONNECTIONS

del pv, multiprocessing

Name = 'simulation'
ProcessSpawn = _mp.get_context('spawn').Process
