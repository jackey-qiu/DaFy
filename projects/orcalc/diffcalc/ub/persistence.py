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

from __future__ import with_statement

import os, glob
import datetime

try:
    import json
except ImportError:
    import simplejson as json


def is_writable(directory):
    """Return true if the file is writable from the current user
    """
    probe = os.path.join(directory, "probe")
    try:
        open(probe, 'w')
    except IOError:
        return False
    else:
        os.remove(probe)
        return True

def check_directory_appropriate(directory):

    if not os.path.exists(directory):
        raise IOError("'%s' does not exist")
    
    if not os.path.isdir(directory):
        raise IOError("'%s' is not a directory")
    
    if not is_writable(directory):
        raise IOError("'%s' is not writable")


class UBCalculationJSONPersister(object):

    def __init__(self, directory, encoder):
        check_directory_appropriate(directory)
        self.directory = directory
        self.description = directory
        self.encoder = encoder
        
    def filepath(self, name):
        return os.path.join(self.directory, name + '.json')
        
    def save(self, state, name):
        # FORMAT = '%Y-%m-%d %H:%M:%S'
        # time_string = datetime.datetime.strftime(datetime.datetime.now(), FORMAT)
        with open(self.filepath(name), 'w') as f:
            json.dump(state, f, indent=4, cls=self.encoder)

    def load(self, name):
        with open(self.filepath(name), 'r') as f:
            return json.load(f)

    def list(self):  # @ReservedAssignment
        files = self._get_save_files()
        return [os.path.basename(f + '.json').split('.json')[0] for f in files]

    def list_metadata(self):       
        metadata = []
        for f in self._get_save_files():            
            dt = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            metadata.append(dt.strftime('%d %b %Y (%H:%M)'))
        return metadata
    
    def _get_save_files(self):
        files = list(filter(os.path.isfile, glob.glob(os.path.join(self.directory, '*.json'))))
        files.sort(key=lambda x: os.path.getmtime(x))
        files.reverse()
        return files

    def remove(self, name):
        os.remove(self.filepath(name))



class UBCalculationPersister(object):
    """Attempts to the use the gda's database to store ub calculation state
    """
    def __init__(self):
        try:
            from uk.ac.diamond.daq.persistence.jythonshelf import LocalJythonShelfManager
            from uk.ac.diamond.daq.persistence.jythonshelf.LocalDatabase import \
                LocalDatabaseException
            self.shelf = LocalJythonShelfManager.getLocalObjectShelf(
                'diffcalc.ub')
        except ImportError as e:
            print ("!!! UBCalculationPersister could not import the gda database "
                   "code: " + repr(e))
            self.shelf = None
        except LocalDatabaseException as e:
            print ("UBCalculationPersister could not connect to the gda "
                   "database: " + repr(e))
            self.shelf = None
        self.description = 'GDA sql database'

    def save(self, state, key):
        if self.shelf is not None:
            self.shelf[key] = state
        else:
            print("<<<no database available to save UB calculation>>>")

    def load(self, name):
        if self.shelf is not None:
            return self.shelf[name]
        else:
            raise IOError("Could not load UB calculation: no database available")

    def list(self):  # @ReservedAssignment
        if self.shelf is not None:
            names = list(self.shelf.keys())
            names.sort()
            return names
        else:
            return []

    def remove(self, name):
        if self.shelf is not None:
            del self.shelf[name]
        else:
            raise IOError("Could not remove UB calculation: no database available")


class UbCalculationNonPersister(UBCalculationPersister):
    """
    A version of UBCalculationPersister that simply stores to a local dict
    rather than a database. Useful for testing.
    """
    def __init__(self):
        self.shelf = dict()
        self.description = 'memory only'
