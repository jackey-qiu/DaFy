"""Trying to make a real molecular viewer
"""

import numpy as np
import os

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtCore

from ..graphics.qt.qtviewer import QtViewer
from ..graphics import colors
from ..graphics.renderers import AtomRenderer, SphereImpostorRenderer
from ..graphics.pickers import SpherePicker
from .representations.ballandstick import BallAndStickRepresentation
from ..graphics.qt.qttrajectory import TrajectoryControls

from .QIPythonWidget import QIPythonWidget

from .. import resources

resources_dir = os.path.dirname(resources.__file__)


class IconButton(QtWidgets.QPushButton):
    def __init__(self, iconname, text):
        icon = QtWidgets.QIcon(iconname)
        super(IconButton, self).__init__(icon, '')
        self.setFixedSize(QSize(32, 32))
        self.setToolTip(text)

class QtMolecularViewer(QtViewer):
    def __init__(self):
        super(QtMolecularViewer, self).__init__()
        self.actions = {}
        
        self.representation = None
        self.widget.background_color = colors.black
        
        #####################################
        #  This is all UI stuff
        #####################################
        self.widget.clicked.connect(self.on_click)
        # Let's add some dock
        self.controls = QtWidgets.QDockWidget()
        # Eliminate the dock titlebar
        title_widget = QtWidgets.QWidget(self)
        self.controls.setTitleBarWidget(title_widget)
        hb = QtWidgets.QHBoxLayout() # For controls
        
        self.keys = {}
        
        self.actions['action1'] = QtWidgets.QPushButton('action1')
        self.actions['action2'] = QtWidgets.QPushButton('action2')
        
        # Sidebar definition        
        dock2 = QtWidgets.QDockWidget()

        self.status_bar = QtWidgets.QLabel()
        
        self.ipython = QIPythonWidget()
        self.ipython.initialize()        
        
        wrapper2 = QtWidgets.QWidget(self)
        vb = QtWidgets.QVBoxLayout(self)
        vb.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        
        self.traj_controls = TrajectoryControls()
        vb.addWidget(self.traj_controls, 1)
        vb.addWidget(self.ipython, 2)
        vb.addWidget(self.status_bar, 0)
        wrapper2.setLayout(vb)
        
        dock2.setWidget(wrapper2)
        self.addDockWidget(Qt.DockWidgetArea(Qt.BottomDockWidgetArea),
                           dock2)
        
        self.traj_controls.hide()
        ############################################
        # Initialization code
        ############################################

        self.namespace = self.ipython.get_user_namespace()
        self.namespace['__builtins__'].viewer = self
        
        self.ipython.ex('from chemlab.mviewer.api.init import *')
        
    def add_representation(self, Repr, system):
        self.system = system
        self.representation = Repr(self, system)
        
    def on_click(self, evt):
        if self.representation is not None:
            self.representation.on_click(evt.x(), evt.y())
    
    def select_all_atoms(self):
        mol = self.system
        which = self.representation.selection[0]
        at = mol.type_array[which]
        sel = mol.type_array == at
        self.representation.make_selection(sel.nonzero()[0])
    
    def select_all_molecules(self):
        mol = self.system
        which = self.representation.last_modified
        if which is None:
            return
        
        at = mol.type_array[which]
        sel = mol.type_array == at
        allmol = mol.atom_to_molecule_indices(sel)
        allmol = mol.mol_to_atom_indices(allmol)
        self.representation.make_selection(allmol, additive=True)

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_A:
            if 'a' in self.keys:
                self.keys['a']()

        if evt.key() == Qt.Key_S:
            if 's' in self.keys:
                self.keys['s']()
        return super(QtMolecularViewer, self).keyPressEvent(evt)
        
