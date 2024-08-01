"""Epics Devices and Device Application."""

import math as _math
import operator as _opr
import time as _time
from functools import partial as _partial

import numpy as _np

from .._conversions import PV as _PV

_DEF_TIMEOUT = 10  # s
_TINY_INTERVAL = 0.050  # s


class Device:
    """Epics Device.

    Parameters
    ----------
        devname: str
            Device name, to be used as PVs prefix.
        props2init: ('all', None or (tuple, list)), optional
            Define which PVs will be connected in the class instantiation. If
            equal to 'all', all properties listed in PROPERTIES_DEFAULT will
            be initialized. If None or empty iterable, no property will be
            initialized. If iterable of strings, only the properties listed
            will be initialized. In this last option, it is possible to add
            properties that are not in PROPERTIES_DEFAULT. The other properties
            will be created whenever they are needed. Defaults to 'all'.

    """

    CONNECTION_TIMEOUT = 0.050  # [s]
    GET_TIMEOUT = 5.0  # [s]
    PROPERTIES_DEFAULT = ()

    def __init__(self, devname, props2init="all"):
        """."""
        self._devname = devname

        if isinstance(props2init, str) and props2init.lower() == "all":
            propties = self.PROPERTIES_DEFAULT
        elif not props2init:
            propties = []
        elif isinstance(props2init, (list, tuple)):
            propties = props2init
        else:
            raise ValueError("Wrong value for props2init.")
        self._pvs = {prpt: self._create_pv(prpt) for prpt in propties}

    @property
    def devname(self):
        """Return device name."""
        return self._devname

    @property
    def properties_in_use(self):
        """Return properties that were already added to the PV list."""
        return tuple(sorted(self._pvs.keys()))

    @property
    def properties_added(self):
        """Return properties added to PV list not in PROPERTIES_DEFAULT."""
        return tuple(
            sorted(set(self.properties_in_use) - set(self.PROPERTIES_DEFAULT))
        )

    @property
    def properties_all(self):
        """Return all properties of the device, connected or not."""
        return tuple(
            sorted(set(self.PROPERTIES_DEFAULT + self.properties_in_use))
        )

    @property
    def pvnames(self):
        """Return device PV names."""
        pvnames = {pv.pvname for pv in self._pvs.values()}
        return pvnames

    @property
    def connected(self):
        """Return PVs connection status."""
        for pvobj in self._pvs.values():
            if not pvobj.connected:
                return False
        return True

    @property
    def disconnected_pvnames(self):
        """Return list of disconnected device PVs."""
        set_ = set()
        for pvobj in self._pvs.values():
            if not pvobj.connected:
                set_.add(pvobj.pvname)
        return set_

    def update(self):
        """Update device properties."""
        for pvobj in self._pvs.values():
            pvobj.get()

    def pv_object(self, propty):
        """Return PV object for a given device property."""
        if propty not in self._pvs:
            pvobj = self._create_pv(propty)
            pvobj.wait_for_connection(Device.CONNECTION_TIMEOUT)
            self._pvs[propty] = pvobj
        return self._pvs[propty]

    def pv_attribute_values(self, attribute):
        """Return pvname-value dict of a given attribute for all PVs."""
        attributes = dict()
        for name, pvobj in self._pvs.items():
            attributes[(self.devname, name)] = getattr(pvobj, attribute)
        return attributes

    @property
    def hosts(self):
        """Return dict of IOC hosts providing device properties."""
        return self.pv_attribute_values("host")

    @property
    def values(self):
        """Return dict of property values."""
        return self.pv_attribute_values("value")

    def wait_for_connection(self, timeout=None):
        """Wait for connection."""
        for pvobj in self._pvs.values():
            res = pvobj.wait_for_connection(timeout)
            if not res:
                return False
        return True

    def __getitem__(self, propty):
        """Return value of property."""
        pvobj = self.pv_object(propty)
        try:
            value = pvobj.get(timeout=Device.GET_TIMEOUT)
        except Exception:
            # exceptions raised in a Virtual Circuit Disconnect (192)
            # event. If the PV IOC goes down, for example.
            print("Could not get value of {}".format(pvobj.pvname))
            value = None
        return value

    def __setitem__(self, propty, value):
        """Set value of property."""
        pvobj = self.pv_object(propty)
        try:
            pvobj.value = value
        except Exception:
            print("Could not set value of {}".format(pvobj.pvname))

    # --- private methods ---
    def _create_pv(self, propty):
        return _PV(
            self.devname, propty, connection_timeout=Device.CONNECTION_TIMEOUT
        )

    def wait(
        self,
        propty,
        value,
        timeout=None,
        comp="eq",
        rel_tol=0.0,
        abs_tol=0.1,
    ):
        """."""

        def comp_(val):
            boo = comp(self[propty], val)
            if isinstance(boo, _np.ndarray):
                boo = _np.all(boo)
            return boo

        if isinstance(comp, str):
            if comp.lower().endswith('close'):
                if isinstance(value, _np.ndarray):
                    comp = _partial(_np.isclose, atol=abs_tol, rtol=rel_tol)
                else:
                    comp = _partial(
                        _math.isclose, abs_tol=abs_tol, rel_tol=rel_tol)
            else:
                comp = getattr(_opr, comp)

        if not isinstance(timeout, str) and timeout != "never":
            timeout = _DEF_TIMEOUT if timeout is None else timeout
            timeout = 0 if timeout <= 0 else timeout
        t0_ = _time.time()
        while not comp_(value):
            if isinstance(timeout, str) and timeout == "never":
                pass
            else:
                if _time.time() - t0_ > timeout:
                    return False
            _time.sleep(_TINY_INTERVAL)
        return True

    def wait_set(
        self,
        props_values,
        timeout=None,
        comp="eq",
        abs_tol=0.0,
        rel_tol=0.1,
    ):
        """."""
        timeout = _DEF_TIMEOUT if timeout is None else timeout
        t0_ = _time.time()
        for propty, value in props_values.items():
            timeout_left = max(0, timeout - (_time.time() - t0_))
            if timeout_left == 0:
                return False
            if not self.wait(
                propty,
                value,
                timeout=timeout_left,
                comp=comp,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            ):
                return False
        return True

    def _enum_setter(self, propty, value, enums):
        value = self._enum_selector(value, enums)
        if value is not None:
            self[propty] = value

    @staticmethod
    def _enum_selector(value, enums):
        if value is None:
            return
        if hasattr(enums, "_fields"):
            enums = enums._fields
        if isinstance(value, str) and value in enums:
            return enums.index(value)
        elif 0 <= int(value) < len(enums):
            return value


class DeviceSet:
    """."""

    def __init__(self, devices, devname=""):
        """."""
        self._devices = devices
        self._devname = devname

    _enum_selector = staticmethod(Device._enum_selector)

    @property
    def devname(self):
        """Name of the Device set. May be empty in some cases."""
        return self._devname

    @property
    def simulators(self):
        """Return list of simulators."""
        sims = set()
        for dev in self._devices:
            sims.update(dev.simulators)
        return sims

    @property
    def pvnames(self):
        """Return device PV names."""
        set_ = set()
        for dev in self._devices:
            set_.update(dev.pvnames)
        return set_

    @property
    def connected(self):
        """Return PVs connection status."""
        for dev in self._devices:
            if not dev.connected:
                return False
        return True

    @property
    def disconnected_pvnames(self):
        """Return list of disconnected device PVs."""
        set_ = set()
        for dev in self._devices:
            set_.update(dev.disconnected_pvnames)
        return set_

    def update(self):
        """Update device properties."""
        for dev in self._devices:
            dev.update()

    def pv_attribute_values(self, attribute):
        """Return property-value dict of a given attribute for all PVs."""
        attributes = dict()
        for dev in self._devices:
            attrs = dev.pv_attribute_values(attribute)
            attributes.update(attrs)
        return attributes

    @property
    def hosts(self):
        """Return dict of IOC hosts providing device properties."""
        return self.pv_attribute_values("host")

    @property
    def values(self):
        """Return dict of property values."""
        return self.pv_attribute_values("value")

    def wait_for_connection(self, timeout=None):
        """Wait for connection."""
        for dev in self._devices:
            if not dev.wait_for_connection(timeout=timeout):
                return False
        return True

    @property
    def devices(self):
        """Return devices."""
        return self._devices

    def wait_devices_propty(
        self,
        devices,
        propty,
        values,
        comp="eq",
        timeout=None,
        abs_tol=0.0,
        rel_tol=0.1,
        return_prob=False,
    ):
        """Wait for devices property to reach value(s)."""
        dev2val = self._get_dev_2_val(devices, values)
        timeout = _DEF_TIMEOUT if timeout is None else timeout
        tini = _time.time()
        timeout_left = timeout
        for dev, val in dev2val.items():
            boo = dev.wait(
                propty,
                val,
                comp=comp,
                timeout=timeout_left,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            timeout_left = max(0, timeout - (_time.time() - tini))
            if not timeout_left or not boo:
                return (False, dev) if return_prob else False
        return (True, None) if return_prob else True

    # --- private methods ---

    def _set_devices_propty(self, devices, propty, values, wait=0):
        """Set devices property to value(s)."""
        dev2val = self._get_dev_2_val(devices, values)
        for dev, val in dev2val.items():
            if dev.pv_object(propty).wait_for_connection():
                dev[propty] = val
                _time.sleep(wait)

    def _get_dev_2_val(self, devices, values):
        """Get devices to values dict."""
        # always use an iterable object
        if not isinstance(devices, (tuple, list)):
            devices = [
                devices,
            ]
        # if 'values' is not iterable, consider the same value for all devices
        if not isinstance(values, (tuple, list, _np.ndarray)):
            values = len(devices) * [values]
        return {k: v for k, v in zip(devices, values)}

    def __getitem__(self, devidx):
        """Return device."""
        return self._devices[devidx]
