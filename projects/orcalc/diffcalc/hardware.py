###
# Copyright 2008-2011 Diamond Light Source Ltd.
# This file is part of Diffcalc.
#
# Diffcalc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Diffcalc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Diffcalc.  If not, see <http://www.gnu.org/licenses/>.
###

from __future__ import absolute_import

from diffcalc.util import DiffcalcException
from diffcalc import settings

SMALL = 1e-8

from diffcalc.util import command

__all__ = ['hardware', 'setcut', 'setmin', 'setmax']


def getNameFromScannableOrString(o):
        try:  # it may be a scannable
            return o.getName()
        except AttributeError:
            return str(o)



@command
def hardware():
    """hardware  -- show diffcalc limits and cuts"""
    print(settings.hardware.repr_sector_limits_and_cuts()) # @UndefinedVariable

@command
def setcut(scannable_or_string=None, val=None):
    """setcut {name {val}} -- sets cut angle
    """
    if scannable_or_string is None and val is None:
        print(settings.hardware.repr_sector_limits_and_cuts()) # @UndefinedVariable
    else:
        name = getNameFromScannableOrString(scannable_or_string)
        if val is None:
            print('%s: %f' % (name, settings.hardware.get_cuts()[name])) # @UndefinedVariable
        else:
            oldcut = settings.hardware.get_cuts()[name]  # @UndefinedVariable
            settings.hardware.set_cut(name, float(val))  # @UndefinedVariable
            newcut = settings.hardware.get_cuts()[name]  # @UndefinedVariable

@command
def setmin(name=None, val=None):
    """setmin {axis {val}} -- set lower limits used by auto sector code (None to clear)""" #@IgnorePep8
    _setMinOrMax(name, val, settings.hardware.set_lower_limit)  # @UndefinedVariable

@command
def setmax(name=None, val=None):
    """setmax {name {val}} -- sets upper limits used by auto sector code (None to clear)""" #@IgnorePep8
    _setMinOrMax(name, val, settings.hardware.set_upper_limit)  # @UndefinedVariable

@command
def setrange(name=None, lower=None, upper=None):
    """setrange {axis {min} {max}} -- set lower and upper limits used by auto sector code (None to clear)""" #@IgnorePep8
    _setMinOrMax(name, lower, settings.hardware.set_lower_limit)  # @UndefinedVariable
    _setMinOrMax(name, upper, settings.hardware.set_upper_limit)  # @UndefinedVariable

def _setMinOrMax(name, val, setMethod):
    if name is None:
        print(settings.hardware.repr_sector_limits_and_cuts()) # @UndefinedVariable
    else:
        name = getNameFromScannableOrString(name)
        if val is None:
            print(settings.hardware.repr_sector_limits_and_cuts(name)) # @UndefinedVariable
        else:
            setMethod(name, float(val))


commands_for_help = ['Hardware',
                     hardware,
                     setcut,
                     setmin,
                     setmax]


class HardwareAdapter(object):

    def __init__(self, diffractometerAngleNames, defaultCuts={},
                 energyScannableMultiplierToGetKeV=1):

        self._diffractometerAngleNames = diffractometerAngleNames
        self._cut_angles = {}
        self._configure_cuts(defaultCuts)
        self.energyScannableMultiplierToGetKeV = \
            energyScannableMultiplierToGetKeV
        self._name = 'base'

    @property
    def name(self):
        return self._name

    def get_axes_names(self):
        return tuple(self._diffractometerAngleNames)

    def get_position(self):
        """pos = get_position() -- returns the current physical diffractometer
        position as a diffcalc.util object in degrees
        """
        raise NotImplementedError()

    def get_wavelength(self):
        """wavelength = get_wavelength() -- returns wavelength in Angstroms
        """
        return 12.39842 / self.get_energy()

    def get_energy(self):
        """energy = get_energy() -- returns energy in kEv  """
        raise NotImplementedError()

    def __str__(self):
        s = self.name + ":\n"
        s += "      energy : %9.4f keV\n" % self.get_energy()
        s += "  wavelength : %9.4f A\n" % self.get_wavelength()
        names = self._diffractometerAngleNames
        width = max(len(k) for k in names)
        fmt = '  %' + str(width) + 's : % 9.4f\n'
        for name, pos in zip(names, self.get_position()):
            s += fmt % (name, pos)
        return s

    def __repr__(self):
        return self.__str__()

    def get_position_by_name(self, angleName):
        names = list(self._diffractometerAngleNames)
        return self.get_position()[names.index(angleName)]

### Limits ###

    def get_lower_limit(self, name):
        '''returns lower limits by axis name. Limit may be None if not set
        '''
        raise NotImplementedError()

    def get_upper_limit(self, name):
        '''returns upper limit by axis name. Limit may be None if not set
        '''
        raise NotImplementedError()

    def set_lower_limit(self, name, value):
        """value may be None to remove limit"""
        raise NotImplementedError()

    def set_upper_limit(self, name, value):
        """value may be None to remove limit"""
        raise NotImplementedError()

    def is_axis_value_within_limits(self, axis_name, value):
        raise NotImplementedError()

    def is_position_within_limits(self, positionArray):
        """
        where position array is in degrees and cut to be between -180 and 180
        """
        names = self._diffractometerAngleNames
        for axis_name, value in zip(names, positionArray):
            if not self.is_axis_value_within_limits(axis_name, value):
                return False
        return True

    def repr_sector_limits_and_cuts(self, name=None):
        if name is None:
            s = ''
            for name in self.get_axes_names():
                s += self.repr_sector_limits_and_cuts(name) + '\n'
            s += "Note: When auto sector/transforms are used,\n "
            s += "      cuts are applied before checking limits."
            return s
        # limits:
        low = self.get_lower_limit(name)
        high = self.get_upper_limit(name)
        s = '  '
        if low is not None:
            s += "% 6.1f <= " % low
        else:
            s += ' ' * 10
        s += '%5s' % name
        if high is not None:
            s += " <= % 6.1f" % high
        else:
            s += ' ' * 10
        # cuts:
        try:
            if self.get_cuts()[name] is not None:
                s += " (cut: % 6.1f)" % self.get_cuts()[name]
        except KeyError:
            pass

        return s

### Cutting Stuff ###

    def _configure_cuts(self, defaultCutsDict):
        # 1. Set default cut angles
        self._cut_angles = dict.fromkeys(self._diffractometerAngleNames, -180.)
        if 'phi' in self._cut_angles:
            self._cut_angles['phi'] = 0.
        # 2. Overide with user-specified cuts
        for name, val in defaultCutsDict.items():
            self.set_cut(name, val)

    def set_cut(self, name, value):
        if name in self._cut_angles:
            self._cut_angles[name] = value
        else:
            raise KeyError("Diffractometer has no angle %s. Try: %s." %
                            (name, self._diffractometerAngleNames))

    def get_cuts(self):
        return self._cut_angles

    def cut_angles(self, positionArray):
        '''Assumes each angle in positionArray is between -360 and 360
        '''
        cutArray = []
        names = self._diffractometerAngleNames
        for axis_name, value in zip(names, positionArray):
            cutArray.append(self.cut_angle(axis_name, value))
        return tuple(cutArray)

    def cut_angle(self, axis_name, value):
        cut_angle = self._cut_angles[axis_name]
        if cut_angle is None:
            return value
        return cut_angle_at(cut_angle, value)


def cut_angle_at(cut_angle, value):
    if (cut_angle == 0 and (abs(value - 360) < SMALL) or
        (abs(value + 360) < SMALL) or
        (abs(value) < SMALL)):
        value = 0.
    if value < (cut_angle - SMALL):
        return value + 360.
    elif value >= cut_angle + 360. + SMALL:
        return value - 360.
    else:
        return value


class DummyHardwareAdapter(HardwareAdapter):

    def __init__(self, diffractometerAngleNames):
        super(self.__class__, self).__init__(diffractometerAngleNames)
#         HardwareAdapter.__init__(self, diffractometerAngleNames)

        self._position = [0.] * len(diffractometerAngleNames)
        self._upperLimitDict = {}
        self._lowerLimitDict = {}
        self._wavelength = 1.
        self.energyScannableMultiplierToGetKeV = 1
        self._name = "Dummy"

# Required methods

    def get_position(self):
        """
        pos = getDiffractometerPosition() -- returns the current physical
        diffractometer position as a list in degrees
        """
        return self._position

    def _set_position(self, pos):
        assert len(pos) == len(self.get_axes_names()), \
            "Wrong length of input list"
        self._position = pos

    position = property(get_position, _set_position)

    def get_energy(self):
        """energy = get_energy() -- returns energy in kEv  """
        if self._wavelength is None:
            raise DiffcalcException(
                "Energy or wavelength have not been set")
        return (12.39842 /
                (self._wavelength * self.energyScannableMultiplierToGetKeV))

    def _set_energy(self, energy):
        self._wavelength = 12.39842 / energy

    energy = property(get_energy, _set_energy)

    def get_wavelength(self):
        """wavelength = get_wavelength() -- returns wavelength in Angstroms"""
        if self._wavelength is None:
            raise DiffcalcException(
                "Energy or wavelength have not been set")
        return self._wavelength

    def _set_wavelength(self, wavelength):
        self._wavelength = wavelength

    wavelength = property(get_wavelength, _set_wavelength)

    def get_lower_limit(self, name):
        '''returns lower limits by axis name. Limit may be None if not set
        '''
        if name not in self._diffractometerAngleNames:
            raise ValueError("No angle called %s. Try one of: %s" %
                             (name, self._diffractometerAngleNames))
        return self._lowerLimitDict.get(name)

    def get_upper_limit(self, name):
        '''returns upper limit by axis name. Limit may be None if not set
        '''
        if name not in self._diffractometerAngleNames:
            raise ValueError("No angle called %s. Try one of: %s" %
                             name, self._diffractometerAngleNames)
        return self._upperLimitDict.get(name)

    def set_lower_limit(self, name, value):
        """value may be None to remove limit"""
        if name not in self._diffractometerAngleNames:
            raise ValueError(
                "Cannot set lower Diffcalc limit: No angle called %s. Try one "
                "of: %s" % (name, self._diffractometerAngleNames))
        if value is None:
            try:
                del self._lowerLimitDict[name]
            except KeyError:
                print ("WARNING: There was no lower Diffcalc limit %s set to "
                       "clear" % name)
        else:
            self._lowerLimitDict[name] = value

    def set_upper_limit(self, name, value):
        """value may be None to remove limit"""
        if name not in self._diffractometerAngleNames:
            raise ValueError(
                "Cannot set upper Diffcalc limit: No angle called %s. Try one "
                "of: %s" % (name, self._diffractometerAngleNames))
        if value is None:
            try:
                del self._upperLimitDict[name]
            except KeyError:
                print ("WARNING: There was no upper Diffcalc limit %s set to "
                       "clear" % name)
        else:
            self._upperLimitDict[name] = value

    def is_axis_value_within_limits(self, axis_name, value):
        if axis_name in self._upperLimitDict:
            if value > self._upperLimitDict[axis_name]:
                return False
        if axis_name in self._lowerLimitDict:
            if value < self._lowerLimitDict[axis_name]:
                return False
        return True


class ScannableHardwareAdapter(HardwareAdapter):

    def __init__(self, diffractometerScannable, energyScannable,
                 energyScannableMultiplierToGetKeV=1):
        input_names = diffractometerScannable.getInputNames()
        super(self.__class__, self).__init__(input_names)
#         HardwareAdapter.__init__(self, input_names)
        self.diffhw = diffractometerScannable
        self.energyhw = energyScannable
        self.energyScannableMultiplierToGetKeV = \
            energyScannableMultiplierToGetKeV
        self._name = "ScannableHarwdareMonitor"

# Required methods

    def get_position(self):
        """
        pos = getDiffractometerPosition() -- returns the current physical
        diffractometer position as a list in degrees
        """
        return self.diffhw.getPosition()

    def get_energy(self):
        """energy = get_energy() -- returns energy in keV (NOT eV!) """
        multiplier = self.energyScannableMultiplierToGetKeV
        energy = self.energyhw.getPosition() * multiplier
        if energy is None:
            raise DiffcalcException("Energy has not been set")
        return energy

    def get_wavelength(self):
        """wavelength = get_wavelength() -- returns wavelength in Angstroms"""
        energy = self.get_energy()
        return 12.39842 / energy

    @property
    def name(self):
        return self.diffhw.getName()

### Limits ###

    def get_lower_limit(self, name):
        '''returns lower limits by axis name. Limit may be None if not set
        '''
        try:
            scn = self.diffhw.getGroupMember(name)
        except AttributeError:
            scn = self.diffhw.getFieldScannable(name)
        try:
            limit = scn.getLowerInnerLimit()
            return limit
        except AttributeError:
            pass
        try:
            limits = scn.getLowerGdaLimits()
        except AttributeError:
            raise DiffcalcException("Cannot read lower limit for scannable {}".format(name))
        try:
            if len(limits) != 1:
                raise DiffcalcException("Lower limit for scannable {} has {} limit values".format(name, len(limits)))
            return limits[0]
        except TypeError:
            return limits

    def get_upper_limit(self, name):
        '''returns upper limit by axis name. Limit may be None if not set
        '''
        try:
            scn = self.diffhw.getGroupMember(name)
        except AttributeError:
            scn = self.diffhw.getFieldScannable(name)
        try:
            limit = scn.getUpperInnerLimit()
            return limit
        except AttributeError:
            pass
        try:
            limits = scn.getUpperGdaLimits()
        except AttributeError:
            raise DiffcalcException("Cannot read upper limit for scannable {}".format(name))
        try:
            if len(limits) != 1:
                raise DiffcalcException("Upper limit for scannable {} has {} limit values".format(name, len(limits)))
            return limits[0]
        except TypeError:
            return limits

    def set_lower_limit(self, name, value):
        scn = self.diffhw.getGroupMember(name)
        try:
            scn.setLowerDummyLimit(value)
        except AttributeError:
            raise DiffcalcException('This command is only implemented in dummy mode.\n'
                                    'Please use GDA/EPICS interface to set hardware limits.')

    def set_upper_limit(self, name, value):
        scn = self.diffhw.getGroupMember(name)
        try:
            scn.setUpperDummyLimit(value)
        except AttributeError:
            raise DiffcalcException('This command is only implemented in dummy mode.\n'
                                    'Please use GDA/EPICS interface to set hardware limits.')

    def is_position_within_limits(self, positionArray):
        """
        where position array is in degrees and cut to be between -180 and 180
        """
        try:
            res = False if self.diffhw.checkPositionValid(positionArray) else True
        except NotImplementedError:
            res = HardwareAdapter.is_position_within_limits(self, positionArray)
        return res

    def is_axis_value_within_limits(self, axis_name, value):
        scn = self.diffhw.getGroupMember(axis_name)
        res = False if scn.checkPositionValid([value,]) else True
        return res
