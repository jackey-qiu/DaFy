import sys,os,qdarkstyle
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import uic, QtCore
import pyqtgraph as pg
import random,copy
import numpy as np
from numpy.linalg import inv
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
matplotlib.use("TkAgg")
import reciprocal_space_plot_v4 as rsplt
try:
    import ConfigParser
except:
    import configparser as ConfigParser
import sys,os
import reciprocal_space_v5 as rsp
# import scipy.signal.savgol_filter as savgol_filter

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

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
        uic.loadUi(os.path.join(DaFy_path,'projects','ubmate','xrd_simulator_debug.ui'),self)
        self.widget_terminal.update_name_space('main_gui',self)
        self.config = ConfigParser.RawConfigParser()
        self.config.optionxform = str # make entries in config file case sensitive
        self.base_structures = {}
        self.structures = []
        self.HKLs_dict = {}
        self.peaks_in_zoomin_viewer = {}
        self.trajactory_pos = []
        self.ax_simulation =None
        self.pushButton_draw.clicked.connect(lambda:self.show_structure(widget_name = 'widget_glview'))
        self.pushButton_load.clicked.connect(self.load_config_file)
        self.pushButton_extract_in_viewer.clicked.connect(self.extract_peaks_in_zoom_viewer)
        self.pushButton_update.clicked.connect(self.update_config_file)
        self.pushButton_launch.clicked.connect(self.launch_config_file)
        self.comboBox_names.currentIndexChanged.connect(self.update_HKs_list)
        self.comboBox_names.currentIndexChanged.connect(self.update_Bragg_peak_list)
        self.comboBox_bragg_peak.currentIndexChanged.connect(self.select_rod_based_on_Bragg_peak)
        self.pushButton_panup.clicked.connect(lambda:self.widget_glview_zoomin.pan(0,0,-0.5))
        self.pushButton_pandown.clicked.connect(lambda:self.widget_glview_zoomin.pan(0,0,0.5))
        self.pushButton_plot_XRD_profiles.clicked.connect(self.draw_ctrs)
        # self.comboBox_working_substrate.currentIndexChanged.connect(self.fill_matrix)
        self.pushButton_convert_abc.clicked.connect(self.cal_xyz)
        self.pushButton_convert_xyz.clicked.connect(self.cal_abc)
        self.pushButton_convert_hkl.clicked.connect(self.cal_qxqyqz)
        self.pushButton_convert_qs.clicked.connect(self.cal_hkl)
        self.pushButton_calculate_hkl_reference.clicked.connect(self.cal_hkl_in_reference)
        self.pushButton_compute.clicked.connect(self.compute_angles)
        self.timer_spin = QtCore.QTimer(self)
        self.timer_spin.timeout.connect(self.spin)
        self.timer_spin_sample = QtCore.QTimer(self)
        self.timer_spin_sample.timeout.connect(self.rotate_sample)
        self.timer_spin_sample.timeout.connect(self.simulate_image)
        self.azimuth_angle = 0
        self.pushButton_azimuth0.clicked.connect(self.azimuth_0)
        self.pushButton_azimuth90.clicked.connect(self.azimuth_90)
        self.pushButton_azimuth180.clicked.connect(self.azimuth_180)
        self.pushButton_panup_2.clicked.connect(lambda:self.pan_view([0,0,-1]))
        self.pushButton_pandown_2.clicked.connect(lambda:self.pan_view([0,0,1]))
        self.pushButton_panleft.clicked.connect(lambda:self.pan_view([0,-1,0]))
        self.pushButton_panright.clicked.connect(lambda:self.pan_view([0,1,0]))
        self.pushButton_start_spin.clicked.connect(self.start_spin)
        self.pushButton_stop_spin.clicked.connect(self.stop_spin)
        #real space viewer control
        self.pushButton_azimuth0_2.clicked.connect(self.azimuth_0_2)
        self.pushButton_azimuth90_2.clicked.connect(self.azimuth_90_2)
        #self.pushButton_azimuth180_2.clicked.connect(self.azimuth_180_2)
        self.pushButton_elevation90.clicked.connect(self.elevation_90)
        self.pushButton_panup_4.clicked.connect(lambda:self.pan_view([0,0,-1],'widget_real_space'))
        self.pushButton_pandown_4.clicked.connect(lambda:self.pan_view([0,0,1],'widget_real_space'))

        self.pushButton_rotate.clicked.connect(self.rotate_sample)
        self.pushButton_rotate.clicked.connect(self.simulate_image)

        self.pushButton_spin.clicked.connect(self.spin_)
        self.pushButton_draw_real_space.clicked.connect(self.draw_real_space)
        self.pushButton_simulate.clicked.connect(self.simulate_image)
        # self.pushButton_draw.clicked.connect(self.prepare_peaks_for_render)
        ##set style for matplotlib figures
        plt.style.use('ggplot')
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        plt.rcParams.update({'axes.labelsize': 10})
        plt.rc('font',size = 10)
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = True
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['axes.facecolor']='0.7'
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['mathtext.default']='regular'
        #style.use('ggplot','regular')

    def simulate_image(self):
        pilatus_size = [float(self.lineEdit_detector_hor.text()), float(self.lineEdit_detector_ver.text())]
        pixel_size = [float(self.lineEdit_pixel.text())]*2
        self.widget_mpl.canvas.figure.clear()
        ax = self.widget_mpl.canvas.figure.add_subplot(1,1,1)
        self.widget_glview.calculate_index_on_pilatus_image_from_cross_points_info(
             pilatus_size = pilatus_size,
             pixel_size = pixel_size,
             distance_sample_detector = float(self.lineEdit_sample_detector_distance.text())
        )
        ax.imshow(self.widget_glview.cal_simuated_2d_pixel_image(pilatus_size = pilatus_size, pixel_size = pixel_size))
        for each in self.widget_glview.pixel_index_of_cross_points:
            for i in range(len(self.widget_glview.pixel_index_of_cross_points[each])):
                pos = self.widget_glview.pixel_index_of_cross_points[each][i]
                hkl = self.widget_glview.cross_points_info_HKL[each][i]
                if len(pos)!=0:
                    if pos not in self.trajactory_pos:
                        self.trajactory_pos.append(pos)
                    ax.text(*pos[::-1],'x',
                            horizontalalignment='center',
                            verticalalignment='center',color = 'r')
                    ax.text(*(list(pos[::-1]+np.array([20,20]))), '{}{}'.format(each,hkl),color = 'y',rotation = 'vertical',fontsize=8)
        ax.text(*self.widget_glview.primary_beam_position,'x',color = 'w')
        ax.text(*(self.widget_glview.primary_beam_position+np.array([50,-50])),'Primary Beam Pos',color = 'r')
        if self.timer_spin_sample.isActive():
            for each in self.trajactory_pos:
                ax.text(*each[::-1],'.',
                        horizontalalignment='center',
                        verticalalignment='center',color = 'w',fontsize = 15)
        ax.grid(False)
        self.widget_mpl.fig.tight_layout()
        self.widget_mpl.canvas.draw()
        self.ax_simulation = ax
        

    def draw_real_space(self):
        super_cell_size = [self.spinBox_repeat_x.value(), self.spinBox_repeat_y.value(), self.spinBox_repeat_z.value()]
        name = self.comboBox_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        self.widget_real_space.RealRM = structure.lattice.RealTM
        self.widget_real_space.show_structure(structure.lattice.basis, super_cell_size)

    def spin_(self):
        if self.timer_spin_sample.isActive():
            self.timer_spin_sample.stop()
        else:
            self.trajactory_pos = []
            self.timer_spin_sample.start(1000)

    def rotate_sample(self):
        theta_x = float(self.lineEdit_rot_x.text())
        theta_y = float(self.lineEdit_rot_y.text())
        theta_z = float(self.lineEdit_rot_z.text())
        self.widget_glview.theta_x = theta_x
        self.widget_glview.theta_y = theta_y
        self.widget_glview.theta_z = theta_z
        self.lineEdit_rotation.setText(str(round(float(self.lineEdit_rotation.text())+theta_z,1)))
        self.widget_glview.update_structure()
        self.extract_cross_point_info()

    #extract cross points between the rods and the Ewarld sphere
    def extract_cross_point_info(self):
        text = ['The cross points between rods and the Ewarld sphere is listed below']
        self.widget_glview.cross_points_info_HKL = {}
        for each in self.widget_glview.cross_points_info:
            self.widget_glview.cross_points_info_HKL[each] = []
            text.append('')
            text.append(each)
            structure = [self.structures[i] for i in range(len(self.structures)) if self.structures[i].name == each][0]
            #apply the rotation matrix due to sample rotation
            RecTMInv = inv(self.widget_glview.RM.dot(structure.lattice.RecTM))
            for each_q in self.widget_glview.cross_points_info[each]:
                #H, K, L = structure.lattice.HKL(each_q)
                H, K, L = RecTMInv.dot(each_q)
                self.widget_glview.cross_points_info_HKL[each].append([round(H,2),round(K,2),round(L,2)])
                text.append('HKL:{}'.format([round(H,3),round(K,3),round(L,3)]))
        self.plainTextEdit_cross_points_info.setPlainText('\n'.join(text))

    def compute_angles(self):
        if self.lineEdit_H.text()=='' or self.lineEdit_K.text()=='' or self.lineEdit_L.text()=='':
            error_pop_up('You must fill all qx qy qz blocks for this calculation!')
            return
        hkl = [float(self.lineEdit_H.text()),float(self.lineEdit_K.text()),float(self.lineEdit_L.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        energy_kev = float(self.lineEdit_energy.text())
        if self.comboBox_unit.currentText() != 'KeV':
            energy_kev = 12.398/energy_kev
        structure.lattice.set_E_keV(energy_kev)
        #negative because of the rotation sense
        mu = -float(self.lineEdit_mu.text())
        phi, gamma, delta = structure.lattice.calculate_diffr_angles(hkl,mu)
        self.lineEdit_phi.setText(str(round(phi,3)))
        self.lineEdit_gamma.setText(str(round(gamma,3)))
        self.lineEdit_delta.setText(str(round(delta,3)))

    def fill_matrix(self):
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        RecTM = structure.lattice.RecTM.flatten()
        RealTM = structure.lattice.RealTM.flatten()
        for i in range(1,10):
            exec(f'self.lineEdit_reaTM_{i}.setText(str(round(RealTM[i-1],3)))')
            exec(f'self.lineEdit_recTM_{i}.setText(str(round(RecTM[i-1],3)))')

    def cal_qxqyqz(self):
        if self.lineEdit_H.text()=='' or self.lineEdit_K.text()=='' or self.lineEdit_L.text()=='':
            error_pop_up('You must fill all qx qy qz blocks for this calculation!')
            return
        hkl = [float(self.lineEdit_H.text()),float(self.lineEdit_K.text()),float(self.lineEdit_L.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        qx,qy,qz = structure.lattice.q(hkl)
        self.lineEdit_qx.setText(str(round(qx,4)))
        self.lineEdit_qy.setText(str(round(qy,4)))
        self.lineEdit_qz.setText(str(round(qz,4)))
        self.cal_q_and_2theta()

    def cal_hkl(self):
        if self.lineEdit_qx.text()=='' or self.lineEdit_qy.text()=='' or self.lineEdit_qz.text()=='':
            error_pop_up('You must fill all qx qy qz blocks for this calculation!')
            return
        qx_qy_qz = [float(self.lineEdit_qx.text()),float(self.lineEdit_qy.text()),float(self.lineEdit_qz.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        H,K,L = structure.lattice.HKL(qx_qy_qz)
        self.lineEdit_H.setText(str(round(H,3)))
        self.lineEdit_K.setText(str(round(K,3)))
        self.lineEdit_L.setText(str(round(L,3)))
        self.cal_q_and_2theta()

    def cal_xyz(self):
        if self.lineEdit_a.text()=='' or self.lineEdit_b.text()=='' or self.lineEdit_c.text()=='':
            error_pop_up('You must fill all a b c blocks for this calculation!')
            return
        a_b_c = [float(self.lineEdit_a.text()),float(self.lineEdit_b.text()),float(self.lineEdit_c.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        x, y, z = structure.lattice.RealTM.dot(a_b_c)
        self.lineEdit_x.setText(str(round(x,3)))
        self.lineEdit_y.setText(str(round(y,3)))
        self.lineEdit_z.setText(str(round(z,3)))

    def cal_abc(self):
        if self.lineEdit_x.text()=='' or self.lineEdit_y.text()=='' or self.lineEdit_z.text()=='':
            error_pop_up('You must fill all x y z blocks for this calculation!')
            return
        x_y_z = [float(self.lineEdit_x.text()),float(self.lineEdit_y.text()),float(self.lineEdit_z.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        a, b, c = structure.lattice.RealTMInv.dot(x_y_z)
        self.lineEdit_a.setText(str(round(a,3)))
        self.lineEdit_b.setText(str(round(b,3)))
        self.lineEdit_c.setText(str(round(c,3)))

    def cal_hkl_in_reference(self):
        #name_work = self.comboBox_working_substrate.currentText()
        #structure_work = [each for each in self.structures if each.name == name_work][0]
        name_reference = self.comboBox_reference_substrate.currentText()
        structure_reference = [each for each in self.structures if each.name == name_reference][0]
        self.cal_qxqyqz()
        qx_qy_qz = np.array([float(self.lineEdit_qx.text()),float(self.lineEdit_qy.text()),float(self.lineEdit_qz.text())])
        hkl = [round(each,3) for each in structure_reference.lattice.HKL(qx_qy_qz)]
        self.lineEdit_hkl_reference.setText('[{},{},{}]'.format(*hkl))

    def cal_q_and_2theta(self):
        qx_qy_qz = [float(self.lineEdit_qx.text()),float(self.lineEdit_qy.text()),float(self.lineEdit_qz.text())]
        q = self._cal_q(qx_qy_qz)
        energy_anstrom = float(self.lineEdit_energy.text())
        if self.comboBox_unit.currentText() == 'KeV':
            energy_anstrom = 12.398/energy_anstrom
        _2theta = self._cal_2theta(q,energy_anstrom)
        self.lineEdit_q.setText(str(round(q,4)))
        self.lineEdit_2theta.setText(str(round(_2theta,2)))
        self.lineEdit_d.setText(str(round(energy_anstrom/2/np.sin(np.deg2rad(_2theta/2)),2)))

    def _cal_q(self,q):
        q = np.array(q)
        return np.sqrt(q.dot(q))

    def _cal_2theta(self,q,wl):
        return np.rad2deg(np.arcsin(q*wl/4/np.pi))*2

    def draw_ctrs(self):
        self.widget.canvas.figure.clear()
        num_plot = len(self.peaks_in_zoomin_viewer)
        resolution = 300
        l = np.linspace(0,self.qz_lim_high,resolution)
        intensity_dict = {'total':[self.widget.canvas.figure.add_subplot(num_plot+1,1,num_plot+1),l,np.zeros(resolution),[],[],[]]}
        self.peak_info_selected_rod = {}
        for i in range(num_plot):
            name = list(self.peaks_in_zoomin_viewer.keys())[i]
            self.peak_info_selected_rod[name] = []
            ax = self.widget.canvas.figure.add_subplot(num_plot+1,1,i+1)
            # ax.set_yscale('log')
            intensity_dict[name] = [ax]
            structure = [each_structure for each_structure in self.structures if each_structure.name == name][0]
            I = np.zeros(resolution)
            #l = np.linspace(0,self.qz_lim_high,300)
            l_text = []
            I_text = []
            text = []
            for each_peak in self.peaks_in_zoomin_viewer[name]:
                hkl = structure.lattice.HKL(each_peak)
                text.append([int(round(each,0)) for each in hkl])
                if structure.name != self.structures[0].name:
                    I_this_point = structure.lattice.I(hkl)/10#scaled by 100 to consider for thin film
                else:
                    I_this_point = structure.lattice.I(hkl)
                l_wrt_main_substrate = self.structures[0].lattice.HKL(each_peak)[-1]
                l_text.append(l_wrt_main_substrate)
                self.peak_info_selected_rod[name].append([l_wrt_main_substrate,str(tuple([int(round(each_,0)) for each_ in hkl]))])
                # print(name, hkl, I_this_point)
                #Gaussian expansion, assume sigma = 0.2
                sigma = 0.06
                I_ = I_this_point/(sigma*(2*np.pi)**0.5)*np.exp(-0.5*((l-l_wrt_main_substrate)/sigma)**2)
                I = I + I_
                I_text.append(I_this_point/(sigma*(2*np.pi)**0.5))
            intensity_dict[name].append(l)
            intensity_dict[name].append(I)
            intensity_dict[name].append(l_text)
            intensity_dict[name].append(I_text)
            intensity_dict[name].append(text)
            intensity_dict['total'][2] = intensity_dict['total'][2]+I
            intensity_dict['total'][3]+=l_text
            intensity_dict['total'][4]+=I_text
            intensity_dict['total'][5]+=text
        for each in intensity_dict:
            ax, l, I,l_text, I_text, text = intensity_dict[each]
            ax.plot(l,I,label = each)
            for i in range(len(text)):
                ax.text(l_text[i],I_text[i],str(text[i]),rotation ='vertical')
            ax.set_title(each)
        self.widget.canvas.figure.tight_layout()
        self.widget.canvas.draw()

    def pan_view(self,signs = [1,1,1], widget = 'widget_glview'):
        value = 0.5
        pan_values = list(np.array(signs)*value)
        getattr(self,widget).pan(*pan_values)
        #self.widget_glview.pan(*pan_values)
 
    def update_camera_position(self,widget_name = 'widget_glview', angle_type="azimuth", angle=0):
        getattr(self,widget_name).setCameraPosition(pos=None, distance=None, \
            elevation=[None,angle][int(angle_type=="elevation")], \
                azimuth=[None,angle][int(angle_type=="azimuth")])

    def azimuth_0(self):
        self.update_camera_position(angle_type="elevation", angle=0)
        self.update_camera_position(angle_type="azimuth", angle=0)

    def azimuth_90(self):
        self.update_camera_position(angle_type="elevation", angle=0)
        self.update_camera_position(angle_type="azimuth", angle=90)

    def azimuth_180(self):
        self.update_camera_position(angle_type="elevation", angle=0)
        self.update_camera_position(angle_type="azimuth", angle=180)

    def azimuth_0_2(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=0)
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="azimuth", angle=0)

    def azimuth_90_2(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=0)
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="azimuth", angle=90)

    def azimuth_180_2(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=0)
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="azimuth", angle=180)

    def elevation_90(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=90)

    def start_spin(self):
        self.timer_spin.start(100)

    def stop_spin(self):
        self.timer_spin.stop()

    def spin(self):
        #if self.azimuth > 360:
        self.update_camera_position(angle_type="azimuth", angle=self.azimuth_angle)
        self.azimuth_angle = self.azimuth_angle + 1

    def show_structure(self, widget_name = 'widget_glview'):
        getattr(self,widget_name).show_structure()
        if widget_name == 'widget_glview':
            self.extract_cross_point_info()
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
        self.widget_glview.text_selected_rod = list(self.widget_glview.RM.dot([qx,qy,self.qz_lims[1]]))+['x']
        self.widget_glview.update_text_item_selected_rod()
        self.widget_glview_zoomin.show_structure()

    def update_HKs_list(self):
        name = self.comboBox_names.currentText()
        self.comboBox_HKs.clear()
        self.comboBox_HKs.addItems(list(map(str,self.HKLs_dict[name])))

    def update_Bragg_peak_list(self):
        name = self.comboBox_names.currentText()
        structure = None
        for each in self.structures:
            if each.name == name:
                structure = each
                break
        peaks = self.peaks_dict[name]
        peaks_unique = []
        for i, peak in enumerate(peaks):
            qxqyqz, _, intensity = peak
            HKL = [int(round(each_, 0)) for each_ in structure.lattice.HKL(qxqyqz)]
            peaks_unique.append(HKL+[intensity])
        peaks_unique = np.array(peaks_unique)
        peaks_unique = peaks_unique[peaks_unique[:,-1].argsort()[::-1]]
        HKLs_unique = [str(list(map(int,each[:3]))) for each in peaks_unique]
        self.comboBox_bragg_peak.clear()
        self.comboBox_bragg_peak.addItems(HKLs_unique)

    def select_rod_based_on_Bragg_peak(self):
        name = self.comboBox_names.currentText()
        structure = None
        for each in self.structures:
            if each.name == name:
                structure = each
                break
        hkl_selected = list(eval(self.comboBox_bragg_peak.currentText()))
        qxqy_reference = np.array(structure.lattice.q(hkl_selected)[0:2])
        for each in self.HKLs_dict[name]:
            qxqy = np.array(structure.lattice.q(each)[0:2])
            if np.sum(np.abs(qxqy - qxqy_reference))<0.05:
                self.comboBox_HKs.setCurrentText(str(each))
                break

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
        self.lineEdit_energy.setText(str(self.energy_keV))
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
            self.structures.append(Structure(self.base_structures[id], HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name, self.energy_keV))

        self.comboBox_names.clear()
        self.comboBox_working_substrate.clear()
        self.comboBox_reference_substrate.clear()
        self.comboBox_substrate.clear()
        # self.comboBox_names.addItems(names)
        self.comboBox_working_substrate.addItems(names)
        self.comboBox_reference_substrate.addItems(names)
        self.comboBox_substrate.addItems(names)
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
        self.HKLs_dict = {}
        self.rods_dict = {}
        self.rods = []
        self.grids = []
        self.axes = []
        self.unit_cell_edges = {}
        space_plots = []
        names = []
        for i in range(len(self.structures)):
            struc = self.structures[i]
            names.append(struc.name)
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
                    self.rods_dict[struc.name] = rods_
            if(struc.plot_grid):
                grids_ = space_plots[i].get_grids(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, color=struc.color)
                if len(grids_)>0:
                    for each in grids_:
                        self.grids.append(each)
            if struc.plot_unitcell:
                self.unit_cell_edges[struc.name] = space_plots[i].get_unit_cell_edges(color = struc.color)

        if(self.plot_axes):
            q1 = self.structures[0].lattice.q([1,0,0])
            q2 = self.structures[0].lattice.q([0,1,0])
            q3 = self.structures[0].lattice.q([0,0,1])
            self.axes.append([[0,0,0],q1,0.1,0.2,(0,0,0,0.8)])
            self.axes.append([[0,0,0],q2,0.1,0.2,(0,0,0,0.8)])
            self.axes.append([[0,0,0],q3,0.1,0.2,(0,0,0,0.8)])
            #composer
            qx_min, qy_min = 10000, 10000
            for each in self.rods:
                qx_, qy_ = each[0][0:2]
                if qx_<qx_min:
                    qx_min = qx_
                if qy_<qy_min:
                    qy_min = qy_
            qx_min, qy_min = 0, self.structures[0].lattice.k0
            self.axes.append([[0,-qy_min,0],[1,-qy_min,0],0.1,0.2,(0,0,1,0.8)])
            self.axes.append([[0,-qy_min,0],[0,1-qy_min,0],0.1,0.2,(0,1,0,0.8)])
            self.axes.append([[0,-qy_min,0],[0,-qy_min,1],0.1,0.2,(1,0,0,0.8)])
        if self.checkBox_ewarld.isChecked():
            self.widget_glview.ewarld_sphere = [[0,-self.structures[0].lattice.k0,0],(0,0,1,0.3),self.structures[0].lattice.k0]
        else:
            self.widget_glview.ewarld_sphere = []
        self.comboBox_names.addItems(names)
        self.widget_glview.spheres = self.peaks
        self.widget_glview.lines = self.rods
        self.widget_glview.lines_dict = self.rods_dict
        self.widget_glview.grids = self.grids
        self.widget_glview.arrows = self.axes
        self.widget_glview.unit_cell_edges = self.unit_cell_edges
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
    def __init__(self, base_structure, HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name, energy_keV):
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
        self.energy_keV = energy_keV
        if(base_structure.create_from_cif):
            self.lattice = rsp.lattice.from_cif(base_structure.filename, self.HKL_normal, self.HKL_para_x, offset_angle, self.energy_keV)
        else:
            a = base_structure.a
            b = base_structure.b
            c = base_structure.c
            alpha = base_structure.alpha
            beta = base_structure.beta
            gamma = base_structure.gamma
            basis = base_structure.basis
            self.lattice = rsp.lattice(a, b, c, alpha, beta, gamma, basis, HKL_normal, HKL_para_x, offset_angle, self.energy_keV)


if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())