import sys,os,qdarkstyle
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import uic
import pyqtgraph as pg
import random,copy
import numpy as np
import matplotlib.pyplot as plt
try:
    from . import locate_path
except:
    import locate_path
script_path = locate_path.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
import pandas as pd
import time
import matplotlib
matplotlib.use("Qt5Agg")
import reciprocal_space_plot_v4 as rsplt
try:
    import ConfigParser
except:
    import configparser as ConfigParser
import sys,os
import reciprocal_space_v5 as rsp
# import scipy.signal.savgol_filter as savgol_filter

#from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
def error_pop_up(msg_text = 'error', window_title = ['Error','Information','Warning'][0]):
    msg = QMessageBox()
    if window_title == 'Error':
        msg.setIcon(QMessageBox.Critical)
    elif window_title == 'Warning':
        msg.setIcon(QMessageBox.Warning)
    else:
        msg.setIcon(QMessageBox.Information)

    msg.setText(msg_text)
    # msg.setInformativeText('More information')
    msg.setWindowTitle(window_title)
    msg.exec_()

class MyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        uic.loadUi(os.path.join(DaFy_path,'projects','ubmate','xrd_simulator.ui'),self)
        self.widget_terminal.update_name_space('main_gui',self)
        self.config = ConfigParser.RawConfigParser()
        self.config.optionxform = str # make entries in config file case sensitive
        self.base_structures = {}
        self.structures = []
        self.HKLs_dict = {}
        self.peaks_in_zoomin_viewer = {}
        self.pushButton_draw.clicked.connect(lambda:self.show_structure(widget_name = 'widget_glview'))
        self.pushButton_load.clicked.connect(self.load_config_file)
        self.pushButton_extract_in_viewer.clicked.connect(self.extract_peaks_in_zoom_viewer)
        self.pushButton_update.clicked.connect(self.update_config_file)
        self.pushButton_launch.clicked.connect(self.launch_config_file)
        self.comboBox_names.currentIndexChanged.connect(self.update_HKs_list)
        self.pushButton_panup.clicked.connect(lambda:self.widget_glview_zoomin.pan(0,0,-0.5))
        self.pushButton_pandown.clicked.connect(lambda:self.widget_glview_zoomin.pan(0,0,0.5))
        self.pushButton_plot_XRD_profiles.clicked.connect(self.draw_ctrs)
        # self.pushButton_draw.clicked.connect(self.prepare_peaks_for_render)


    def draw_ctrs(self):
        self.widget.canvas.figure.clear()
        num_plot = len(self.peaks_in_zoomin_viewer)
        resolution = 300
        l = np.linspace(0,self.qz_lim_high,resolution)
        intensity_dict = {'total':[self.widget.canvas.figure.add_subplot(num_plot+1,1,num_plot+1),l,np.zeros(resolution)]}
        for i in range(num_plot):
            name = list(self.peaks_in_zoomin_viewer.keys())[i]
            ax = self.widget.canvas.figure.add_subplot(num_plot+1,1,i+1)
            # ax.set_yscale('log')
            intensity_dict[name] = [ax]
            '''
            if name == self.structures[0].name:#main substrate then calculate ctr
                l = np.linspace(0,self.qz_lim_high,100)
                hk = self.structures[0].lattice.HKL(self.peaks_in_zoomin_viewer[name][0])
                I = np.array([self.structures[0].lattice.I([hk[0],hk[1],each]) for each in l])
                intensity_dict[name].append(l)
                intensity_dict[name].append(I)
            else:
            '''
            structure = [each_structure for each_structure in self.structures if each_structure.name == name][0]
            I = np.zeros(resolution)
            #l = np.linspace(0,self.qz_lim_high,300)
            for each_peak in self.peaks_in_zoomin_viewer[name]:
                hkl = structure.lattice.HKL(each_peak)
                if structure.name != self.structures[0].name:
                    I_this_point = structure.lattice.I(hkl)/10#scaled by 100 to consider for thin film
                else:
                    I_this_point = structure.lattice.I(hkl)
                l_wrt_main_substrate = self.structures[0].lattice.HKL(each_peak)[-1]
                # print(name, hkl, I_this_point)
                #Gaussian expansion, assume sigma = 0.2
                sigma = 0.06
                I_ = I_this_point/(sigma*(2*np.pi)**0.5)*np.exp(-0.5*((l-l_wrt_main_substrate)/sigma)**2)
                I = I + I_
            intensity_dict[name].append(l)
            intensity_dict[name].append(I)
            intensity_dict['total'][-1] = intensity_dict['total'][-1]+I
        for each in intensity_dict:
            ax, l, I = intensity_dict[each]
            ax.plot(l,I,label = each)
            ax.set_title(each)
        self.widget.canvas.draw()
        
    def update_camera_position(self,widget_name = 'widget_glview', angle_type="azimuth", angle=0):
        getattr(self,widget_name).setCameraPosition(pos=None, distance=None, \
            elevation=[None,angle][int(angle_type=="elevation")], \
                azimuth=[None,angle][int(angle_type=="azimuth")])

    def show_structure(self, widget_name = 'widget_glview'):
        getattr(self,widget_name).show_structure()
        self.update_camera_position(widget_name=widget_name, angle_type="elevation", angle=90)
        self.update_camera_position(widget_name=widget_name, angle_type="azimuth", angle=270)

    def extract_peaks_in_zoom_viewer(self):
        structure = None
        for each in self.structures:
            if each.name == self.comboBox_names.currentText():
                structure = each
                break
        hkl = list(eval(self.comboBox_HKs.currentText()))
        #hk.append(0)
        qx, qy,_ = each.lattice.RecTM.dot(hkl)
        # print('HK and QX and Qy',hk,qx,qy)
        peaks_temp = []
        text_temp = []
        self.peaks_in_zoomin_viewer = {}
        for key in self.peaks_dict:
            for each in self.peaks_dict[key]:
                if abs(each[0][0]-qx)<0.05 and abs(each[0][1]-qy)<0.05:
                    each_ = copy.deepcopy(each)
                    each_[0][0] = 0
                    each_[0][1] = 0
                    # each_[0][1] = each_[0][1]-0.5

                    # each_[-1] = 0.1
                    #print(each_)
                    peaks_temp.append(each_)
                    structure_temp = [each_structure for each_structure in self.structures if each_structure.name == key]#should be one item list
                    assert len(structure_temp)==1,'duplicate structures'
                    HKL_temp = [int(round(each_item,0)) for each_item in structure_temp[0].lattice.HKL(each[0])]
                    text_temp.append(each_[0]+['{}({})'.format(key,HKL_temp)])
                    if key in self.peaks_in_zoomin_viewer:
                        self.peaks_in_zoomin_viewer[key].append(each[0])
                    else:
                        self.peaks_in_zoomin_viewer[key] = [each[0]]
                    #print(each_)
                else:
                    pass
                    # print(each[0]) 
        # print('peaks',peaks_temp)
        self.widget_glview_zoomin.clear()
        self.widget_glview_zoomin.spheres = peaks_temp
        self.widget_glview_zoomin.texts = text_temp
        self.widget_glview.texts = [[qx,qy,self.qz_lims[1],'x']]
        self.widget_glview.show_structure()
        self.widget_glview_zoomin.show_structure()

    def update_HKs_list(self):
        name = self.comboBox_names.currentText()
        self.comboBox_HKs.clear()
        self.comboBox_HKs.addItems(list(map(str,self.HKLs_dict[name])))

    def load_config_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Config Files (*.cfg);;All Files (*.*)", options=options)
        if fileName:
            self.lineEdit_config_path.setText(fileName)
            with open(fileName,'r') as f:
                self.plainTextEdit_config.setPlainText(''.join(f.readlines()))
    
    def update_config_file(self):
        with open(self.lineEdit_config_path.text(),'w') as f:
            f.write(self.plainTextEdit_config.toPlainText())
            error_pop_up('The config file is overwritten!','Information')

    def launch_config_file(self):
        self.config = ConfigParser.RawConfigParser()
        self.config.optionxform = str # make entries in config file case sensitive
        self.config.read(self.lineEdit_config_path.text())
        self.common_offset_angle = float(self.config.get('Plot', 'common_offset_angle'))
        self.plot_axes = int(self.config.get('Plot', 'plot_axes'))
        self.energy_keV = float(self.config.get('Plot', 'energy_keV'))
        self.k0 = rsp.get_k0(self.energy_keV)
        self.load_base_structures_in_config()
        self.load_structures_in_config()
        self.update_draw_limits()
        self.prepare_objects_for_render()

    def load_base_structures_in_config(self):
        # read from settings file
        self.base_structures = {}
        base_structures_ = self.config.items('BaseStructures')
        for base_structure in base_structures_:
            toks = base_structure[1].split(',')
            if(len(toks) == 2):
                id = toks[0]
                self.base_structures[id] = Base_Structure.from_cif(id, os.path.join(script_path,toks[1]))
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
                self.base_structures[id] = Base_Structure(id,a,b,c,alpha,beta,gamma,basis)

    def load_structures_in_config(self):
        self.structures = []
        names = []
        structures_ = self.config.items('Structures')
        for structure_ in structures_:
            name = structure_[0]
            names.append(name)
            toks = structure_[1].split(',')
            id = toks[0]
            HKL_normal = toks[1].split(';')
            HKL_normal = [float(HKL_normal[0]), float(HKL_normal[1]), float(HKL_normal[2])]
            HKL_para_x = toks[2].split(';')
            HKL_para_x = [float(HKL_para_x[0]), float(HKL_para_x[1]), float(HKL_para_x[2])]
            offset_angle = float(toks[3]) + self.common_offset_angle
            is_reference_coordinate_system = int(toks[4])
            plot_peaks = int(toks[5])
            plot_rods = int(toks[6])
            plot_grid = int(toks[7])
            plot_unitcell = int(toks[8])
            color = toks[9].split(';')
            color = (float(color[0]), float(color[1]), float(color[2]))
            self.structures.append(Structure(self.base_structures[id], HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name))

        self.comboBox_names.clear()
        self.comboBox_names.addItems(names)
        # put reference structure at first position in list
        for i in range(len(self.structures)):
            if(self.structures[i].is_reference_coordinate_system):
                self.structures[0], self.structures[i] = self.structures[i], self.structures[0]
                break
    
    def update_draw_limits(self):
        q_inplane_lim = self.config.get('Plot', 'q_inplane_lim')
        qx_lim_low = self.config.get('Plot', 'qx_lim_low')
        qx_lim_high = self.config.get('Plot', 'qx_lim_high')
        qy_lim_low = self.config.get('Plot', 'qy_lim_low')
        qy_lim_high = self.config.get('Plot', 'qy_lim_high')
        qz_lim_low = self.config.get('Plot', 'qz_lim_low')
        qz_lim_high = self.config.get('Plot', 'qz_lim_high')
        q_mag_lim_low = self.config.get('Plot', 'q_mag_lim_low')
        q_mag_lim_high = self.config.get('Plot', 'q_mag_lim_high')

        self.q_inplane_lim = None if q_inplane_lim == 'None' else float(q_inplane_lim)
        qx_lim_low = None if qx_lim_low == 'None' else float(qx_lim_low)
        qx_lim_high = None if qx_lim_high == 'None' else float(qx_lim_high)
        qy_lim_low = None if qy_lim_low == 'None' else float(qy_lim_low)
        qy_lim_high = None if qy_lim_high == 'None' else float(qy_lim_high)
        qz_lim_low = None if qz_lim_low == 'None' else float(qz_lim_low)
        qz_lim_high = None if qz_lim_high == 'None' else float(qz_lim_high)
        q_mag_lim_low = None if q_mag_lim_low == 'None' else float(q_mag_lim_low)
        q_mag_lim_high = None if q_mag_lim_high == 'None' else float(q_mag_lim_high)

        if qz_lim_high == None:
            self.qz_lim_high = 5
        else:
            self.qz_lim_high = qz_lim_high

        self.qx_lims = [qx_lim_low, qx_lim_high]
        if(self.qx_lims[0] == None or self.qx_lims[1] == None):
            self.qx_lims = None
        self.qy_lims = [qy_lim_low, qy_lim_high]
        if(self.qy_lims[0] == None or self.qy_lims[1] == None):
            self.qy_lims = None
        self.qz_lims = [qz_lim_low, qz_lim_high]
        if(self.qz_lims[0] == None or self.qz_lims[1] == None):
            self.qz_lims = None
        self.mag_q_lims = [q_mag_lim_low, q_mag_lim_high]
        if(self.mag_q_lims[0] == None or self.mag_q_lims[1] == None):
            self.mag_q_lims = None

    def prepare_objects_for_render(self):
        self.peaks = []
        self.peaks_dict = {}
        self.rods = []
        self.grids = []
        self.axes = []
        space_plots = []
        for i in range(len(self.structures)):
            struc = self.structures[i]
            space_plots.append(rsplt.space_plot(struc.lattice))

            if(struc.plot_peaks):
                peaks_, _ = space_plots[i].get_peaks(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, mag_q_lims=self.mag_q_lims, color=struc.color)
                if len(peaks_)>0:
                    for each in peaks_:
                        self.peaks.append(each)
                self.peaks_dict[struc.name] = peaks_
                
            if(struc.plot_rods):
                rods_, HKLs = space_plots[i].get_rods(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, color=struc.color)
                if len(rods_)>0:
                    for each in rods_:
                        self.rods.append(each)
                    self.HKLs_dict[struc.name] = HKLs
            if(struc.plot_grid):
                grids_ = space_plots[i].get_grids(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, color=struc.color)
                if len(grids_)>0:
                    for each in grids_:
                        self.grids.append(each)
        if(self.plot_axes):
            q1 = self.structures[0].lattice.q([1,0,0])
            q2 = self.structures[0].lattice.q([0,1,0])
            q3 = self.structures[0].lattice.q([0,0,1])
            self.axes.append([[0,0,0],q1,0.1,0.2,(250,250,250,0.8)])
            self.axes.append([[0,0,0],q2,0.1,0.2,(1,1,1,0.8)])
            self.axes.append([[0,0,0],q3,0.1,0.2,(1,1,1,0.8)])
            #compose
            qx_min, qy_min = 10000, 10000
            for each in self.rods:
                qx_, qy_ = each[0][0:2]
                if qx_<qx_min:
                    qx_min = qx_
                if qy_<qy_min:
                    qy_min = qy_
            self.axes.append([[qx_min,qy_min,0],[qx_min+1,qy_min,0],0.1,0.2,(0,0,1,0.8)])
            self.axes.append([[qx_min,qy_min,0],[qx_min,qy_min+1,0],0.1,0.2,(0,1,0,0.8)])
            self.axes.append([[qx_min,qy_min,0],[qx_min,qy_min,1],0.1,0.2,(1,0,0,0.8)])
        self.widget_glview.spheres = self.peaks
        self.widget_glview.lines = self.rods
        self.widget_glview.grids = self.grids
        self.widget_glview.arrows = self.axes
        self.widget_glview.clear()


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


if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())