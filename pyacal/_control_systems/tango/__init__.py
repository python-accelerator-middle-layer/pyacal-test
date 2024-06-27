"""Epics-related subpackage."""

from threading import Thread
import multiprocessing as _mp

from .pv import PV

del pv

Name = 'tango'
ProcessSpawn = _mp.get_context('spawn').Process
