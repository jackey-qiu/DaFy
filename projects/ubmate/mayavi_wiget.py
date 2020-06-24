from PyQt5 import  QtWidgets
import os
import numpy as np
from numpy import cos
from mayavi.mlab import contour3d
import mayavi

# os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi.api import Engine

# start visualisation
from mayavi import mlab
from tvtk.tools import visual
import reciprocal_space_plot_v4 as rsplt

try:
    import ConfigParser
except:
    import configparser as ConfigParser
import sys,os
sys.path.append('../../..')
sys.path.append('..')
import reciprocal_space_v5 as rsp
import numpy as np
try:
    from . import locate_path
except:
    import locate_path
script_path = locate_path.module_path_locator()

config = ConfigParser.RawConfigParser()
config.optionxform = str # make entries in config file case sensitive

config.read(os.path.join(script_path,'settings/Ag111_Cu001.cfg'))
#config.read('settings/Co_Oxides_Au001.cfg')
#config.read('settings/Ni_Oxides_Au111.cfg')
#config.read('settings/Fe_Oxides_Au111.cfg')

common_offset_angle = float(config.get('Plot', 'common_offset_angle'))
# common_offset_angle = 0


base_structures = dict()
structures = []

class Base_Structure():
    def __init__(self, id, a=1, b=1, c=1, alpha=90, beta=90, gamma=90, basis=[], filename=None, create_from_cif=False):
        self.id = id
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.basis = basis
        self.filename = filename
        self.create_from_cif = create_from_cif

    @staticmethod
    def from_cif(id, filename):
        return Base_Structure(id, filename=filename, create_from_cif=True)

class Structure():
    def __init__(self, base_structure, HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name):
        self.HKL_normal = HKL_normal
        self.HKL_para_x = HKL_para_x
        self.offset_angle = offset_angle
        self.is_reference_coordinate_system = is_reference_coordinate_system
        self.plot_peaks = plot_peaks
        self.plot_rods = plot_rods
        self.plot_grid = plot_grid
        self.plot_unitcell = plot_unitcell
        self.base_structure = base_structure
        self.color = color
        self.name = name
        if(base_structure.create_from_cif):
            self.lattice = rsp.lattice.from_cif(base_structure.filename, self.HKL_normal, self.HKL_para_x, offset_angle)
        else:
            a = base_structure.a
            b = base_structure.b
            c = base_structure.c
            alpha = base_structure.alpha
            beta = base_structure.beta
            gamma = base_structure.gamma
            basis = base_structure.basis
            self.lattice = rsp.lattice(a, b, c, alpha, beta, gamma, basis, HKL_normal, HKL_para_x, offset_angle)


# read from settings file
base_structures_ = config.items('BaseStructures')
for base_structure in base_structures_:
    toks = base_structure[1].split(',')
    if(len(toks) == 2):
        id = toks[0]
        base_structures[id] = Base_Structure.from_cif(id, os.path.join(script_path,toks[1]))
    else:
        id = toks[0]
        a = float(toks[1])
        b = float(toks[2])
        c = float(toks[3])
        alpha = float(toks[4])
        beta = float(toks[5])
        gamma = float(toks[6])
        basis = []
        for i in range(7, len(toks)):
            toks2 = toks[i].split(';')
            basis.append([toks2[0], float(toks2[1]), float(toks2[2]), float(toks2[3])])
        base_structures[id] = Base_Structure(id,a,b,c,alpha,beta,gamma,basis)

structures_ = config.items('Structures')
for structure_ in structures_:
    name = structure_[0]
    toks = structure_[1].split(',')
    id = toks[0]
    HKL_normal = toks[1].split(';')
    HKL_normal = [float(HKL_normal[0]), float(HKL_normal[1]), float(HKL_normal[2])]
    HKL_para_x = toks[2].split(';')
    HKL_para_x = [float(HKL_para_x[0]), float(HKL_para_x[1]), float(HKL_para_x[2])]
    offset_angle = float(toks[3]) + common_offset_angle
    is_reference_coordinate_system = int(toks[4])
    plot_peaks = int(toks[5])
    plot_rods = int(toks[6])
    plot_grid = int(toks[7])
    plot_unitcell = int(toks[8])
    color = toks[9].split(';')
    color = (float(color[0]), float(color[1]), float(color[2]))
    structures.append(Structure(base_structures[id], HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name))


# put reference structure at first position in list
for i in range(len(structures)):
    if(structures[i].is_reference_coordinate_system):
        structures[0], structures[i] = structures[i], structures[0]
        break


q_inplane_lim = config.get('Plot', 'q_inplane_lim')
qx_lim_low = config.get('Plot', 'qx_lim_low')
qx_lim_high = config.get('Plot', 'qx_lim_high')
qy_lim_low = config.get('Plot', 'qy_lim_low')
qy_lim_high = config.get('Plot', 'qy_lim_high')
qz_lim_low = config.get('Plot', 'qz_lim_low')
qz_lim_high = config.get('Plot', 'qz_lim_high')
q_mag_lim_low = config.get('Plot', 'q_mag_lim_low')
q_mag_lim_high = config.get('Plot', 'q_mag_lim_high')
plot_axes = int(config.get('Plot', 'plot_axes'))
energy_keV = float(config.get('Plot', 'energy_keV'))
k0 = rsp.get_k0(energy_keV)

q_inplane_lim = None if q_inplane_lim == 'None' else float(q_inplane_lim)
qx_lim_low = None if qx_lim_low == 'None' else float(qx_lim_low)
qx_lim_high = None if qx_lim_high == 'None' else float(qx_lim_high)
qy_lim_low = None if qy_lim_low == 'None' else float(qy_lim_low)
qy_lim_high = None if qy_lim_high == 'None' else float(qy_lim_high)
qz_lim_low = None if qz_lim_low == 'None' else float(qz_lim_low)
qz_lim_high = None if qz_lim_high == 'None' else float(qz_lim_high)
q_mag_lim_low = None if q_mag_lim_low == 'None' else float(q_mag_lim_low)
q_mag_lim_high = None if q_mag_lim_high == 'None' else float(q_mag_lim_high)


qx_lims = [qx_lim_low, qx_lim_high]
if(qx_lims[0] == None or qx_lims[1] == None):
    qx_lims = None
qy_lims = [qy_lim_low, qy_lim_high]
if(qy_lims[0] == None or qy_lims[1] == None):
    qy_lims = None
qz_lims = [qz_lim_low, qz_lim_high]
if(qz_lims[0] == None or qz_lims[1] == None):
    qz_lims = None
mag_q_lims = [q_mag_lim_low, q_mag_lim_high]
if(mag_q_lims[0] == None or mag_q_lims[1] == None):
    mag_q_lims = None

## create Mayavi Widget and show

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    scale_factor = 2

    def update_scale(self):
        self.scale_factor = self.scale_factor*2
        self.update_plot()

    @on_trait_change('scene.activated')
    def update_plot(self):
    ## PLot to Show        

        outline = mayavi.mlab.outline(line_width=3)
        outline.outline_mode = 'cornered'
        def Rx(theta_x):
            return np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x), np.cos(theta_x)]])
        def Ry(theta_y):
            return np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
        def Rz(theta_z):
            return np.array([[np.cos(theta_z), -np.sin(theta_z), 0],[np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

        HKL_aligned = np.array([0, 1, 2])
        mu = 0.34
        structures[0].lattice.set_E_keV(energy_keV)
        angles = structures[0].lattice.angles(HKL_aligned, mu)

        theta = angles[0][0]
        gamma = angles[0][1]
        delta = angles[0][2]

        print('theta =', theta)
        print('gamma =', gamma)
        print('delta =', delta)

        theta = angles[1][0]
        gamma = angles[1][1]
        delta = angles[1][2]

        print('theta =', theta)
        print('gamma =', gamma)
        print('delta =', delta)

        #ki = Ry(np.deg2rad(mu)).dot(Rz(np.deg2rad(theta)).dot([k0, 0, 0]))
        ki = np.dot(Rz(np.deg2rad(theta)), np.dot(Ry(np.deg2rad(mu)), [k0, 0, 0]))
        #kf = np.dot(Rz(np.deg2rad(theta)), np.dot(Ry(np.deg2rad(mu)), np.dot(Rz(np.deg2rad(delta)), np.dot(Ry(np.deg2rad(gamma)),[k0, 0, 0]))))
        kf = np.dot(Rz(np.deg2rad(delta)), np.dot(Ry(np.deg2rad(gamma)), np.dot(Rz(np.deg2rad(theta)), np.dot(Ry(np.deg2rad(mu)),[k0, 0, 0]))))

        #kf = Rz(np.deg2rad(theta-delta)).dot(Ry(np.deg2rad(mu-gamma)).dot([k0, 0, 0]))

        #ki = rsp.RotationMatrix(0, np.deg2rad(mu), np.deg2rad(theta)).dot([k0,0,0])
        #kf = rsp.RotationMatrix(0, np.deg2rad(-gamma), np.deg2rad(delta)).dot([k0,0,0])
        mlab.plot3d([-ki[0], 0], [-ki[1], 0], [-ki[2], 0], color=(0,0,1))
        mlab.plot3d([-ki[0], -ki[0]+kf[0]], [-ki[1], -ki[1]+kf[1]], [-ki[2], -ki[2]+kf[2]], color=(0,0,1))

        sphere = mlab.points3d(-ki[0], -ki[1], -ki[2], scale_mode='none', scale_factor=2*k0, color=(0.67, 0.77, 0.93), resolution=50, opacity=0.7)
        sphere.actor.property.specular = 0.45
        sphere.actor.property.specular_power = 5
        sphere.actor.property.backface_culling = True

        peaks = []
        space_plots = []
        for i in range(len(structures)):
            struc = structures[i]
            space_plots.append(rsplt.space_plot(struc.lattice))
            if(struc.plot_peaks):
                peaks.append(space_plots[i].plot_peaks(qx_lims=qx_lims, qy_lims=qy_lims, qz_lims=qz_lims, q_inplane_lim=q_inplane_lim, mag_q_lims=mag_q_lims, color=struc.color))
            if(struc.plot_rods):
                space_plots[i].plot_rods(qx_lims=qx_lims, qy_lims=qy_lims, qz_lims=qz_lims, q_inplane_lim=q_inplane_lim, color=struc.color)
            if(struc.plot_grid):
                space_plots[i].plot_grid(qx_lims=qx_lims, qy_lims=qy_lims, qz_lims=qz_lims, q_inplane_lim=q_inplane_lim, color=struc.color)
            if(struc.plot_unitcell):
                space_plots[i].plot_unit_cell()

        if(plot_axes):
            q1 = structures[0].lattice.q([1,0,0])
            q2 = structures[0].lattice.q([0,1,0])
            q3 = structures[0].lattice.q([0,0,1])

            rsplt.Arrow_From_A_to_B(0, 0, 0, q1[0], q1[1], q1[2], color=(0,0,0))
            rsplt.Arrow_From_A_to_B(0, 0, 0, q2[0], q2[1], q2[2], color=(0,0,0))
            rsplt.Arrow_From_A_to_B(0, 0, 0, q3[0], q3[1], q3[2], color=(0,0,0))

        number_of_peaks = []
        for peak in peaks:
            number_of_peaks.append(peak[0].glyph.glyph_source.glyph_source.output.points.to_array().shape[0])

        mayavi.mlab.figure(figure=mayavi.mlab.gcf(engine=None), bgcolor=(0,0,0))
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                        height=250, width=300, show_label=False),
                resizable=True )

class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        self.ui = self.visualization.edit_traits(parent=self,
                                                    kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

#### PyQt5 GUI ####
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

    ## MAIN WINDOW
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200,200,1100,700)

    ## CENTRAL WIDGET
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

    ## GRID LAYOUT
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")


    ## BUTTONS
    ## Mayavi Widget 1    
        container = QtGui.QWidget()
        mayavi_widget = MayaviQWidget(container)
        self.gridLayout.addWidget(mayavi_widget, 1, 0,1,1)


    ## SET TEXT 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simulator"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())