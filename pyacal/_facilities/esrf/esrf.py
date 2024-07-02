import at
from .. import Facility

# arbitrary, must be defined by the facility developers:
facility = Facility('esrf', 'tango', 'pyat')
facility.default_accelerator = 'EBS'

