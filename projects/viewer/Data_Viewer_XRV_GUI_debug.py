import sys,os,qdarkstyle
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import uic
import random
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
from cv_tool import cvAnalysis
from charge_calculation import calculate_charge
from PlotSetup import data_viewer_plot_cv, RHE, plot_tafel_from_formatted_cv_info
import pandas as pd
import time
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator, FixedFormatter
matplotlib.use("Qt5Agg")
from scipy import signal
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
        uic.loadUi(os.path.join(DaFy_path,'projects','viewer','data_viewer__xrv_new.ui'),self)
        # self.setupUi(self)
        # plt.style.use('ggplot')
        self.cv_tool = cvAnalysis()
        self.widget_terminal.update_name_space('main_gui',self)
        self.addToolBar(self.mplwidget.navi_toolbar)
        self.setWindowTitle('XRV data Viewer')
        self.data_to_save = {}
        self.image_range_info = {}
        self.lineEdit_data_file_path.setText(os.path.join(DaFy_path,'dump_files','temp_xrv.csv'))
        #pot_offset is the difference between the spock value and those recorded by potentiostat
        #you need some calibration step to figure out this value not necessarily always 0.055 V
        #the correction will be real_pot = spock_value + pot_offset
        self.potential_offset = 0.055
        # plt.style.use('ggplot')
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        plt.rcParams.update({'axes.labelsize': 10})
        plt.rc('font',size = 10)
        plt.rcParams['axes.linewidth'] = 1.5
        # plt.rcParams['axes.grid'] = True
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams["errorbar.capsize"] = 5
        # plt.rcParams['axes.facecolor']='0.7'
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['mathtext.default']='regular'
        #style.use('ggplot')
        self.actionLoadData.triggered.connect(self.load_file)
        self.actionPlotData.triggered.connect(self.plot_figure_xrv)
        self.actionPlotData.triggered.connect(self.print_data_summary)
        self.actionPlotRate.triggered.connect(self.plot_data_summary_xrv)
        self.actionSaveData.triggered.connect(self.save_data_method)
        self.actionShowHide.triggered.connect(self.show_or_hide)
        self.pushButton_cal_charge.clicked.connect(self.plot_figure_xrv)
        self.pushButton_cal_charge.clicked.connect(self.print_data_summary)
        self.PushButton_append_scans.clicked.connect(self.append_scans_xrv)
        self.checkBox_time_scan.clicked.connect(self.set_plot_channels)
        self.checkBox_mask.clicked.connect(self.append_scans_xrv)
        self.actionLoadConfig.triggered.connect(self.load_config)
        self.actionSaveConfig.triggered.connect(self.save_config)
        self.pushButton_update.clicked.connect(self.update_plot_range)
        self.pushButton_update.clicked.connect(self.append_scans_xrv)
        self.pushButton_update_info.clicked.connect(self.make_plot_lib)
        self.pushButton_apply.clicked.connect(self.update_pot_offset)
        self.pushButton_load_cv_config.clicked.connect(self.load_cv_config_file)
        self.pushButton_update_cv_config.clicked.connect(self.update_cv_config_file)
        self.pushButton_plot_cv.clicked.connect(self.plot_cv_data)
        self.pushButton_cal_charge_2.clicked.connect(self.calculate_charge_2)
        # self.pushButton_plot_reaction_order.clicked.connect(self.cv_tool.plot_reaction_order_with_pH)
        self.pushButton_plot_reaction_order.clicked.connect(self.plot_reaction_order_and_tafel)
        
        #self.pushButton_save_data.clicked.connect(self.save_data_method)
        #self.pushButton_save_xrv_data.clicked.connect(self.save_xrv_data)
        #self.pushButton_plot_datasummary.clicked.connect(self.plot_data_summary_xrv)
        self.data = None
        self.data_summary = {} 
        self.data_range = None
        self.pot_range = None
        self.potential = []
        self.show_frame = True
        self.plot_lib = {}
        # self.grain_size_info = {'vertical':[],'horizontal':[]}
        self.charge_info = {}
        self.grain_size_info_all_scans = {}
        self.strain_info_all_scans = {}#key is scan_no, each_item is {(pot1,pot2):{"vertical":(abs_value,value_change),"horizontal":(abs_value,value_change)},"pH":pH value}}
        self.pot_ranges = {}
        self.cv_info = {}
        self.tick_label_settings = {}
        self.plot_tafel = plot_tafel_from_formatted_cv_info

    def extract_tick_label_settings(self):
        if hasattr(self,'plainTextEdit_tick_label_settings'):
            strings = self.plainTextEdit_tick_label_settings.toPlainText()
            lines = strings.rsplit('\n')
            for each_line in lines:
                if each_line.startswith('#'):
                    pass
                else:
                    items = each_line.rstrip().rsplit(';')
                    key,item,locator,padding,tick_num,fmt,func = items
                    locator = eval(locator)
                    if key in self.tick_label_settings:
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':padding,'tick_num':tick_num,'fmt':fmt,'func':func}
                    else:
                        self.tick_label_settings[key] = {}
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':padding,'tick_num':tick_num,'fmt':fmt,'func':func}
        else:
            pass

    def calculate_charge_2(self):
        output = self.cv_tool.calc_charge_all()
        self.plainTextEdit_cv_summary.setPlainText(output)

    def set_grain_info_all_scan(self,grain_object, scan, pot_ranges, direction, cases):
        pot_ranges = [tuple(each) for each in pot_ranges]
        if scan not in grain_object:
            grain_object[scan] = {}
        if pot_ranges[0] not in grain_object[scan]:
            for each_pot, case in zip(pot_ranges,cases):
                grain_object[scan][each_pot] = {direction:case,'pH':self.phs[self.scans.index(scan)]}
        else:
            for each_pot, case in zip(pot_ranges,cases):
                grain_object[scan][each_pot][direction] = case
                grain_object[scan][each_pot]['pH'] = self.phs[self.scans.index(scan)]
        return grain_object

    def reset_meta_data(self):
        self.charge_info = {}
        self.grain_size_info_all_scans = {}
        self.strain_info_all_scans = {}#key is scan_no, each_item is {(pot1,pot2):{"vertical":(abs_value,value_change),"horizontal":(abs_value,value_change)},"pH":pH value}}
        self.tick_label_settings = {}

    def update_pot_offset(self):
        self.potential_offset = eval(self.lineEdit_pot_offset.text())/1000

    def make_plot_lib(self):
        self.plot_lib = {}
        info = self.textEdit_plot_lib.toPlainText().rsplit('\n')
        folder = self.lineEdit_cv_folder.text()
        if info==[''] or folder=='':
            return
        for each in info:
            if not each.startswith('#'):
                scan, cv, cycle, cutoff,scale,color, ph, func = each.replace(" ","").rstrip().rsplit(',')
                cv_name = os.path.join(folder,cv)
                self.plot_lib[int(scan)] = [cv_name,int(cycle),eval(cutoff),eval(scale),color,eval(ph),func]

    #data format based on the output of IVIUM potentiostat
    def extract_ids_file(self,file_path,which_cycle=3):
        data = []
        current_cycle = 0
        with open(file_path,encoding="ISO-8859-1") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith('primary_data'):
                    print(current_cycle)
                    current_cycle=current_cycle+1
                    if current_cycle == which_cycle:
                        for j in range(i+3,i+3+int(lines[i+2].rstrip())):
                            data.append([float(each) for each in lines[j].rstrip().rsplit()])
                        break
                    else:
                        pass
                else:
                    pass
        #return (pot: V, current: mA)
        return np.array(data)[:,0], np.array(data)[:,1]*1000

    #data format based on Fouad's potentiostat
    def extract_cv_file(self,file_path='/home/qiu/apps/048_S221_CV', which_cycle=1):
        #return:time(s), pot(V), current (mA)
        skiprows = 0
        with open(file_path,'r') as f:
            for each in f.readlines():
                if each.startswith('Time(s)'):
                    skiprows+=1
                    break
                else:
                    skiprows+=1
        data = np.loadtxt(file_path,skiprows = skiprows)
        #nodes index saving all the valley pot positions
        nodes =[0]
        for i in range(len(data[:,1])):
            if i!=0 and i!=len(data[:,1])-1:
                if data[i,1]<data[i+1,1] and data[i,1]<data[i-1,1]:
                    nodes.append(i)
        nodes.append(len(data[:,1]))
        if which_cycle>len(nodes):
            print('Cycle number lager than the total cycles! Use the first cycle instead!')
            return data[nodes[1]:nodes[2],0], data[nodes[1]:nodes[2],1],data[nodes[1]:nodes[2],2]
        else:
            return data[nodes[which_cycle]:nodes[which_cycle+1],0],data[nodes[which_cycle]:nodes[which_cycle+1],1],data[nodes[which_cycle]:nodes[which_cycle+1],2]

    def plot_cv_from_external(self,ax,scan_no,marker_pos):
        file_name,which_cycle,cv_spike_cut,cv_scale_factor, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.{}'.format(func_name))
        t, pot,current = func(file_name, which_cycle)
        t_filtered, pot_filtered, current_filtered = t, pot, current
        for ii in range(4):
            filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
            filter_index = filter_index+1#index offset by 1
            t_filtered = t_filtered[(filter_index,)]
            pot_filtered = pot_filtered[(filter_index,)]
            current_filtered = current_filtered[(filter_index,)]
        pot_filtered = RHE(pot_filtered,pH=ph)
        ax.plot(pot_filtered,current_filtered*8*cv_scale_factor,label='',color = color)
        ax.plot(RHE(pot,pH=ph),current*8,label='',color = color)
        #get the position to show the scaling text on the plot
        current_temp = current_filtered[np.argmin(np.abs(pot_filtered[0:int(len(pot_filtered)/2)]-1.1))]*8*cv_scale_factor
        ax.text(1.1,current_temp+1.5,'x{}'.format(cv_scale_factor),color=color)
        #store the cv data
        self.cv_info[scan_no] = {'current_density':current*8,'potential':RHE(pot,pH = ph),'pH':ph, 'color':color}

        for each in marker_pos:
            ax.plot([each,each],[-100,100],':k')
        #ax.set_ylim([min(current_filtered*8*cv_scale_factor),max(current*8)])
        print('scan{} based on cv'.format(scan_no))
        # print([self.pot_range1, self.pot_range2, self.pot_range3])
        # print(self.get_integrated_charge(pot_filtered, current_filtered, t_filtered, plot = False))
        #scan rate in V/s
        scan_rate = float(self.lineEdit_scan_rate.text())
        #potential range in V_RHE
        """
        pot_range = eval('[{}]'.format(self.lineEdit_pot_range.text().rstrip()))
        charge_cv = calculate_charge(t, pot, current, which_cycle=0, ph=ph, cv_spike_cut=cv_spike_cut, cv_scale_factor=cv_scale_factor, scan_rate = scan_rate, pot_range = pot_range)
        if scan_no not in self.charge_info:
            self.charge_info[scan_no] = {'skin_charge':0,'film_charge':0,'total_charge':charge_cv}
        else:
            self.charge_info[scan_no]['total_charge'] = charge_cv
        # pot_ranges = [self.pot_range1, self.pot_range2, self.pot_range3]
        """
        pot_ranges = self.pot_ranges[scan_no]
        if scan_no not in self.charge_info:
            self.charge_info[scan_no] = {}
        for pot_range in pot_ranges:
            charge_cv = calculate_charge(t, pot, current, which_cycle=0, ph=ph, cv_spike_cut=cv_spike_cut, cv_scale_factor=cv_scale_factor, scan_rate = scan_rate, pot_range = pot_range)
            if pot_range not in self.charge_info[scan_no]:
                self.charge_info[scan_no][pot_range] = {'skin_charge':0,'film_charge':0,'total_charge':charge_cv}
            else:
                self.charge_info[scan_no][pot_range]['total_charge'] = charge_cv
        
        # print(self.charge_info)

        return min(current_filtered*8*cv_scale_factor),max(current*8)

    def get_integrated_charge(self, pot, current, t, pot_range_full = [1., 1.57], steps = 10,plot= False):
        trans_pot = [1.4,1.42,1.43,1.45,1.5]
        pot_ranges = []
        for each in trans_pot:
            pot_ranges.append([pot_range_full[0],each])
            pot_ranges.append([each,pot_range_full[1]])
        for pot_range in pot_ranges:
            Q_integrated = 0
            pot_step = (pot_range[1] - pot_range[0])/steps
            def _get_index(all_values, current_value, first_half = True):
                all_values = np.array(all_values)
                half_index = int(len(all_values)/2)
                if first_half:
                    return np.argmin(abs(all_values[0:half_index]-current_value))
                else:
                    return np.argmin(abs(all_values[half_index:]-current_value))
            for i in range(steps):
                pot_left, pot_right = pot_range[0] + pot_step*i, pot_range[0] + pot_step*(i+1)
                delta_t = abs(t[_get_index(pot, pot_left)] - t[_get_index(pot, pot_right)])
                i_top_left, i_top_right = current[_get_index(pot, pot_left)], current[_get_index(pot, pot_right)]
                i_bottom_left, i_bottom_right = current[_get_index(pot, pot_left,False)], current[_get_index(pot, pot_right,False)]
                Q_two_triangles = abs(i_top_left - i_top_right)*delta_t/2 + abs(i_bottom_left - i_bottom_right)*delta_t/2
                if i_top_left > i_bottom_left:
                    Q_retangle = abs(abs(min([i_top_left, i_top_right]))-abs(max([i_bottom_left, i_bottom_right])))*delta_t
                else:
                    Q_retangle = abs(abs(min([i_bottom_left, i_bottom_right]))-abs(max([i_top_left, i_top_right])))*delta_t
                Q_integrated = Q_integrated + Q_two_triangles + Q_retangle
            if plot:
                fig = plt.figure()
                plt.plot(t, current)
                plt.show()
            print('Integrated charge between E {} and E {} is {} mC'.format(pot_range[0], pot_range[1], Q_integrated/2))
            #factor of 2 is due to contributions from the anodic and cathodic cycle
        return Q_integrated/2

    def estimate_charge_from_skin_layer_thickness(self, slope, transition_E, pot_range,charge_per_unit_cell = 2, roughness_factor = 1):
        surface_area = 0.125 #cm2
        unitcell_area = 28.3 #A2
        single_layer_thickness = 0.5 #nm
        num_unit_cell_at_surface = surface_area/unitcell_area*10**16*roughness_factor
        single_layer_charge_transfer = num_unit_cell_at_surface * charge_per_unit_cell
        thickness_skin_layer = abs(slope * (pot_range[0] - pot_range[1]))
        percentage_oxidation = thickness_skin_layer/single_layer_thickness
        # print('percentage_oxidation={}'.format(percentage_oxidation))
        charge_transfer = percentage_oxidation * single_layer_charge_transfer * 1.6 * 10**-19 #in C
        print('Charge transfer between E {} and E {} is  {} mC based on skin layer thickness estimation'.format(pot_range[0], pot_range[1], charge_transfer*10**3))
        return charge_transfer*10**3

    #return size/strain change from slope info in vertical or horizontal direction, and the associated absolute size
    def calculate_size_strain_change(self, x0,x1,x2,y1,slope1,slope2,pot_range,roughness_factor = 1):
        y0 = slope1*(x0-x1)+y1
        y2 = slope2*(x2-x1)+y1
        pot_left, pot_right = pot_range

        size_at_pot_left, size_at_pot_right = 0, 0
        if pot_left<=x1:
            size_at_pot_left = slope1*(pot_left-x1)+y1
        else:
            size_at_pot_left = slope2*(pot_left-x1)+y1

        if pot_right<=x1:
            size_at_pot_right = slope1*(pot_right-x1)+y1
        else:
            size_at_pot_right = slope2*(pot_right-x1)+y1
        return max([size_at_pot_left, size_at_pot_right]),abs(size_at_pot_left - size_at_pot_right)

    def calculate_size_strain_change_from_plot_data(self, scan, label, data_range, pot_range, roughness_factor = 1):
        pot_lf, pot_rt = pot_range
        pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]]
        values = np.array(self.data_to_plot[scan][label][data_range[0]:data_range[1]])+self.data_to_plot[scan][label+'_max']
        values_smooth = signal.savgol_filter(values,41,2)
        index_lf = np.argmin(np.abs(pots - pot_lf))
        index_rt = np.argmin(np.abs(pots - pot_rt))
        return max([values_smooth[index_lf],values_smooth[index_rt]]), abs(values_smooth[index_lf]-values_smooth[index_rt])

    def estimate_charge_from_skin_layer_thickness_philippe_algorithm(self, size_info, q0 = 15.15):
        #q0 : number of electron transfered per nm^3 during redox chemistry(15.15 only for Co3O4 material)
        #check the document for details of this algorithm
        vertical, horizontal = size_info['vertical'], size_info['horizontal']
        vertical_size, vertical_size_change = vertical
        horizontal_size, horizontal_size_change = horizontal
        charge_per_electron = 1.6*(10**-19) # C
        v_skin = (vertical_size_change + 2*(horizontal_size_change*vertical_size/horizontal_size))*(10**14)
        q_skin = v_skin * q0 * charge_per_electron * 1000 # mC per m^2
        q_film = vertical_size*(10**14)*q0 * charge_per_electron * 1000
        return q_skin, q_film

    def print_data_summary(self):
        try:
            self.print_data_summary_()
        except:
            pass

    def print_data_summary_(self):
        header = "\t".join(["scan", "pH", "pot_lf", "pot_rt", "q_skin", "q_film", "q_cv", "hor_size","d_hor_size","ver_size","d_ver_size","hor_strain","d_hor_strain","ver_strain","d_ver_strain"])
        output_text = [header]
        for scan in self.scans:
            for pot_range in self.pot_ranges[scan]:
                #scan = each_scan
                ph = self.grain_size_info_all_scans[scan][pot_range]['pH']
                charges = [round(self.charge_info[scan][pot_range][each],2) for each in ['skin_charge','film_charge','total_charge']]
                size_hor = [round(each,2) for each in list(self.grain_size_info_all_scans[scan][pot_range]["horizontal"])]
                size_ver = [round(each, 2) for each in list(self.grain_size_info_all_scans[scan][pot_range]["vertical"])]
                strain_hor = [round(each,4) for each in list(self.strain_info_all_scans[scan][pot_range]["horizontal"])]
                strain_ver = [round(each,4) for each in list(self.strain_info_all_scans[scan][pot_range]["vertical"])]
                data_temp = [scan, ph] +[round(each,3) for each in list(pot_range)]+ charges + size_hor + size_ver + strain_hor + strain_ver
                output_text.append('\t'.join([str(each) for each in data_temp]))
        self.widget_terminal.update_name_space('charge_info',self.charge_info)
        self.widget_terminal.update_name_space('size_info',self.grain_size_info_all_scans)
        self.widget_terminal.update_name_space('strain_info',self.strain_info_all_scans)
        self.widget_terminal.update_name_space('main_win',self)
        self.widget_terminal.update_name_space('cv_info', self.cv_info)

        output_text.append("*********Notes*********")
        output_text.append("*scan: scan number")
        output_text.append("*pot_lf (V_RHE): left boundary of potential range considered ")
        output_text.append("*pot_rt (V_RHE): right boundary of potential range considered ")
        output_text.append("*q_skin(mc/m2): charge calculated based on skin layer thickness")
        output_text.append("*q_film(mc/m2): charge calculated assuming all Co2+ in the film material has been oxidized to Co3+")
        output_text.append("*q_cv(mc/m2): charge calculated from electrochemistry data (CV data)")
        output_text.append("*(d)_hor/ver_size(nm): horizontal/vertical size or the associated change with a d_ prefix")
        output_text.append("*(d)_hor/ver_strain(%): horizontal/vertical strain or the associated change with a d_ prefix")

        #print("\n".join(output_text))
        self.plainTextEdit_summary.setPlainText("\n".join(output_text))

    def show_or_hide(self):
        self.frame.setVisible(not self.show_frame)
        self.show_frame = not self.show_frame

    #this update the data range for the selected scan
    #it will take effect after you replot the figures
    def update_plot_range(self):
        scan = int(self.comboBox_scans.currentText())
        l,r = int(self.lineEdit_img_range_left.text()),int(self.lineEdit_img_range_right.text())
        self.image_range_info[scan] = [l,r]
        all_info=[]
        for each in self.image_range_info:
            all_info.append('{}:{}'.format(each,self.image_range_info[each]))
        self.plainTextEdit_img_range.setPlainText('\n'.join(all_info))

    #open data file
    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName)) 

    #save the data according to the specified data ranges
    def save_data_method(self):
        # print(self.data_to_save.keys())
        if len(self.data_to_save)==0:
            error_pop_up('No data prepared to be saved!','Error')
        else:
            for each in self.data_to_save:
                # print(self.data_to_save[each])
                self.data_to_save[each].to_csv(os.path.join(self.lineEdit_data_file_path.text(), self.lineEdit_data_file_name.text()+'_{}.csv'.format(each)),header = False, sep =' ',index=False)


    #save a segment of data to be formated for loading in superrod
    def save_xrv_data(self):
        key_map_lib = {
                       #'peak_intensity':1,
                       'strain_oop':2,
                       'strain_ip':3,
                       'grain_size_ip':4,
                       'grain_size_oop':5
                       }
        scan = self.scans
        ph = self.phs
        data_range = self.data_range
        data = self.data_to_plot
        for i in range(len(scan)):
            scan_ = scan[i]
            ph_ = ph[i]
            data_range_ = data_range[i]
            data_ = data[scan_]
            temp_data = {'potential':[],
                         'scan_no':[],
                         'items':[],
                         'Y':[],
                         #'I':[],
                         #'eI':[],
                         'e1':[],
                         'e2':[]}
            for key in key_map_lib:
                temp_data['potential'] = temp_data['potential'] + list(data_['potential'][data_range_[0]:])
                #temp_data['eI'] = temp_data['eI'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['Y'] = temp_data['Y'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['e1'] = temp_data['e1'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['e2'] = temp_data['e2'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['items'] = temp_data['items'] + [key_map_lib[key]]*len(data_['potential'][data_range_[0]:])
                temp_data['scan_no'] = temp_data['scan_no'] + [scan_]*len(data_['potential'][data_range_[0]:])
                #temp_data['I'] = temp_data['I'] + list(data_[key][data_range_[0]:])
            df = pd.DataFrame(temp_data)
            df.to_csv(self.lineEdit_data_file_path.text().replace('.csv','_{}.csv'.format(scan_)),\
                      header = False, sep =' ',columns = list(temp_data.keys()), index=False)

    #open the cv config file for cv analysis only
    def load_cv_config_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CV config Files (*.ini);;All Files (*.txt)", options=options)
        if fileName:
            self.lineEdit_cv_config_path.setText(fileName)
            with open(fileName,'r') as f:
                lines = f.readlines()
                self.plainTextEdit_cv_config.setPlainText(''.join(lines))

    #update the config file after edition in the plainText block
    def update_cv_config_file(self):
        with open(self.lineEdit_cv_config_path.text(),'w') as f:
            f.write(self.plainTextEdit_cv_config.toPlainText())
        missed_items = self.cv_tool._extract_parameter_from_config(self.lineEdit_cv_config_path.text(), sections = ['Global'])
        if len(missed_items)==0:
            self.cv_tool.extract_cv_info()
            error_pop_up('The config file is overwritten!','Information')
        else:
            error_pop_up(f'The config file is overwritten, but the config file has the following items missed:{missed_items}!','Error')

    #load config file
    def load_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.ini);;All Files (*.txt)", options=options)
        with open(fileName,'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.rstrip().rsplit(':')
                if len(items)>2:
                    channel,value = items[0], ':'.join(items[1:])
                else:
                    channel,value = items
                if value=='True':
                    getattr(self,channel).setChecked(True)
                elif value=='False':
                    getattr(self,channel).setChecked(False)
                else:
                    try:
                        if channel == "textEdit_plot_lib":
                            getattr(self,channel).setText(value.replace(";","\n"))
                        else:
                            getattr(self,channel).setText(value)
                    except:
                        getattr(self,channel).setPlainText(value.replace(";","\n"))
                        if value=="":
                            pass
                        else:
                            self.image_range_info = {}
                            items = value.rsplit(';')
                            for each_item in items:
                                a,b = each_item.rstrip().rsplit(":")
                                self.image_range_info[int(a)] = eval(b)
        self._load_file()
        self.append_scans_xrv()
        self.update_pot_offset()
        self.make_plot_lib()

    #save config file
    def save_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","config file (*.ini);Text Files (*.txt);all files(*.*)", options=options)
        with open(fileName,'w') as f:
            channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
                        'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
                        'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file']
            for channel in channels:
                try:
                    f.write(channel+':'+str(getattr(self,channel).isChecked())+'\n')
                except:
                    f.write(channel+':'+getattr(self,channel).text()+'\n')
            f.write("plainTextEdit_img_range:"+self.plainTextEdit_img_range.toPlainText().replace("\n",";")+'\n')
            f.write("textEdit_plot_lib:"+self.textEdit_plot_lib.toPlainText().replace("\n",";")+'\n')
            
    def set_plot_channels(self):
        time_scan = self.checkBox_time_scan.isChecked()
        if time_scan:
            self.lineEdit_x.setText('image_no')
            self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop')
        else:
            self.lineEdit_x.setText('potential')
            self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop') 

    #fill the info in the summary text block
    #and init the self.data attribute by reading data from excel file
    def _load_file(self):
        fileName = self.lineEdit_data_file.text()
        self.lineEdit_data_file.setText(fileName)
        self.data = pd.read_excel(fileName)
        col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
        scans = list(set(list(self.data['scan_no'])))
        self.scans_all = scans
        scans.sort()
        scan_numbers = 'scan_nos\n'+str(scans)+'\n'
        # print(list(self.data[self.data['scan_no']==scans[0]]['phs'])[0])
        self.phs_all = [list(self.data[self.data['scan_no']==scan]['phs'])[0] for scan in scans]
        phs = 'pHs\n'+str(self.phs_all)+'\n'
        self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers,phs]))

    #load excel data file
    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;All Files (*.csv)", options=options)
        if fileName:
            self.lineEdit_data_file.setText(fileName)
            self.data = pd.read_excel(fileName)
        col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
        scans = list(set(list(self.data['scan_no'])))
        self.scans_all = scans
        self.comboBox_scans.clear()
        self.comboBox_scans.addItems([str(each) for each in sorted(scans)])
        self.image_range_info = {}
        self.plainTextEdit_img_range.setPlainText("")
        scans.sort()
        scan_numbers = 'scan_nos\n'+str(scans)+'\n'
        # print(list(self.data[self.data['scan_no']==scans[0]]['phs'])[0])
        self.phs_all = [list(self.data[self.data['scan_no']==scan]['phs'])[0] for scan in scans]
        phs = 'pHs\n'+str(self.phs_all)+'\n'
        self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers,phs]))

    #plot bar chart using linear fit results
    def plot_data_summary_xrv_from_external_file(self):
        if self.data_summary!={}:
            self.mplwidget2.fig.clear()
            #label mapping
            y_label_map = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\Delta\varepsilon_\parallel$  (%/V)',
                        'strain_oop':r'$\Delta\varepsilon_\perp$  (%/V)',
                        'grain_size_oop':r'$\Delta d_\perp$  (nm/V)',
                        'grain_size_ip':r'$\Delta d_\parallel$  (nm/V)',
                        'peak_intensity':r'Intensity / a.u.'}
            #get color tags
            colors_bar = self.lineEdit_colors_bar.text().rsplit(',')
            if len(colors_bar) == 1:
                colors_bar = colors_bar*len(self.scans)
            else:
                if len(colors_bar) > len(self.scans):
                    colors_bar = colors_bar[0:len(self.scans)]
                elif len(colors_bar) < len(self.scans):
                    colors_bar = colors_bar + [colors_bar[-1]]*(len(self.scans)-len(colors_bar))
            plot_y_labels = [each for each in list(self.data_summary[self.scans[0]].keys()) if each in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']]
            #TODO this has to be changed to set the y_max automatically in different cases.
            lim_y_temp = {'strain_ip':-0.18,'strain_oop':-0.5,'grain_size_ip':-1.2,'grain_size_oop':-1.6}
            for each in plot_y_labels:
                for i in range(int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)):#each value come with an error value
                    # plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)],self.data_summary[each_scan][each][-1]] for each_scan in self.scans])
                    plot_data_y = np.array([[self.data_summary[each_scan][each][i*2],self.data_summary[each_scan][each][i*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    labels = ['pH {}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    count_pH13 = 1
                    for j,each_label in enumerate(labels):
                        if each_label == 'pH 13':
                            labels[j] = '{}_{}'.format(each_label,count_pH13)
                            count_pH13 += 1
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), int(len(self.data_summary[self.scans[0]]['strain_ip'])/2), i+1+int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)*plot_y_labels.index(each))
                    ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                    self._format_axis(ax_temp)
                    if each == 'strain_ip':
                        y_locator = [0,-0.05,-0.1,-0.15]
                    elif each =='strain_oop':
                        y_locator = [0,-0.1,-0.2,-0.3,-0.4]
                    elif each == 'grain_size_ip':
                        y_locator = [0,-0.25,-0.5,-0.75,-1]
                    elif each == 'grain_size_oop':
                        y_locator = [0,-0.3,-0.6,-0.9,-1.2,-1.5]
                    self._format_ax_tick_labels(ax = ax_temp,
                            fun_set_bounds = 'set_ylim', 
                            bounds = [lim_y_temp[each],0], 
                            bound_padding = 0., 
                            major_tick_location =y_locator, 
                            show_major_tick_label = i==0, #show major tick label for the first scan
                            num_of_minor_tick_marks=4, 
                            fmt_str = '{: 4.2f}')
                    if i == 0:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=13)
                        ax_temp.set_ylim([lim_y_temp[each],0])
                    else:
                        ax_temp.set_ylim([lim_y_temp[each],0])
                    # if each == plot_y_labels[0]:
                        # ax_temp.set_title('E range:{:4.2f}-->{:4.2f} V'.format(*each_pot), fontsize=13)
                    if each != plot_y_labels[-1]:
                        ax_temp.set_xticklabels([])
                    else:
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=10)
                    if i!=0:
                        ax_temp.set_yticklabels([])
                        
                    # ax_temp.set_xticklabels(plot_data_x,labels)
            self.mplwidget2.fig.subplots_adjust(wspace = 0.04,hspace=0.04)
            self.mplwidget2.canvas.draw()
        else:
            pass

    #for each item
    #eg self.data_summary[221]['strain_ip] = [slop_1st_seg, error_1st_seg, slope_2nd_seg, error_2nd_seg]
    # this fun is very much highly customized for one time purpose, not generic at all
    #be careful if you want to use it to extract info from a new fit file
    def make_data_summary_from_external_file(self):
        file = self.lineEdit_slope_file.text()
        if file=="":
            return
        data = pd.read_csv(file,sep='\t',comment = '#')
        summary = {}
        for each in self.scans:
            summary[each] = {}
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                summary[each][item] = []
                item_values = list(data['scan{}'.format(each)][item])[::-1]
                for each_item in item_values:
                    error = [0.05,0.15][int('size' in item)]
                    summary[each][item] = summary[each][item] + [each_item,error*abs(each_item)]
        self.data_summary = summary

    #bar chart based on slope values calculated directely from the xrv master figure
    def plot_data_summary_xrv(self):
        if self.checkBox_use_external_slope.isChecked():
            self.make_data_summary_from_external_file()
            self.plot_data_summary_xrv_from_external_file()
            print("new_data summary is built!")
            return 
        #here you should update the self.data_summary info
        self.plot_figure_xrv()
        #plain text to be displayed in the data summary tab
        plain_text = []

        if self.data_summary!={}:
            self.mplwidget2.fig.clear()
            y_label_map = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\Delta\varepsilon_\parallel$  (%/V)',
                        'strain_oop':r'$\Delta\varepsilon_\perp$  (%/V)',
                        'grain_size_oop':r'$\Delta d_\perp$  (nm/V)',
                        'grain_size_ip':r'$\Delta d_\parallel$  (nm/V)',
                        'peak_intensity':r'Intensity / a.u.'}
            y_label_map_abs = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\varepsilon_\parallel$  (%)',
                        'strain_oop':r'$\varepsilon_\perp$  (%)',
                        'grain_size_oop':r'$d_\perp$  (nm)',
                        'grain_size_ip':r'$d_\parallel$  (nm)',
                        'peak_intensity':r'Intensity / a.u.'}

            colors_bar = self.lineEdit_colors_bar.text().rsplit(',')
            if len(colors_bar) == len(self.scans):
                pass
            else:
                if len(colors_bar) > len(self.scans):
                    colors_bar = colors_bar[0:len(self.scans)]
                elif len(colors_bar) < len(self.scans):
                    colors_bar = colors_bar + [colors_bar[-1]]*(len(self.scans)-len(colors_bar))
            plot_y_labels = [each for each in list(self.data_summary[self.scans[0]].keys()) if each in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']]
            
            lim_y_temp = {'strain_ip':[],'strain_oop':[],'grain_size_ip':[],'grain_size_oop':[]}
            for each_pot in self.pot_range:
                #if pot_range = [1,1] for eg, the bar value is actually the associated absolute value at pot = 1
                #if pot_range = [1,1.5] for eg, the bar value is the value difference between 1 and 1.5 V
                use_absolute_value = each_pot[0] == each_pot[1]
                for each in lim_y_temp.keys():
                    for each_scan in self.scans:
                        if use_absolute_value:
                            lim_y_temp[each].append(self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2])
                        else:
                            lim_y_temp[each].append(-self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2])
            for each in lim_y_temp:
                lim_y_temp[each] = [min(lim_y_temp[each]),max(lim_y_temp[each])]
            for each in lim_y_temp:
                offset = (lim_y_temp[each][1]-lim_y_temp[each][0])*0.1
                lim_y_temp[each] = [lim_y_temp[each][0]-offset,lim_y_temp[each][1]+offset]
            if use_absolute_value:
               y_label_map = y_label_map_abs 
            #print(self.data_summary)
            for each_pot in self.pot_range:
                output_data = []
                use_absolute_value = each_pot[0] == each_pot[1]
                for each in plot_y_labels:
                    plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2],self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    labels = ['pH {}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), len(self.pot_range), self.pot_range.index(each_pot)+1+len(self.pot_range)*plot_y_labels.index(each))
                    if use_absolute_value:
                        ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                        ax_temp.plot(plot_data_x,plot_data_y[:,0], '*:',color='0.1')
                        output_data.append(plot_data_y[:,0])
                    else:
                        ax_temp.bar(plot_data_x,-plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                        ax_temp.plot(plot_data_x,-plot_data_y[:,0], '*:',color='0.1')
                        output_data.append(-plot_data_y[:,0])
                    if each_pot == self.pot_range[0]:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=12)
                        # ax_temp.set_ylim([lim_y_temp[each],0])
                        ax_temp.set_ylim(lim_y_temp[each])
                    else:
                        ax_temp.set_ylim(lim_y_temp[each])
                    if each == plot_y_labels[0]:
                        if use_absolute_value:
                            ax_temp.set_title('E = {:4.2f} V'.format(each_pot[0]), fontsize=12)
                        else:
                            ax_temp.set_title('E range:{:4.2f}-->{:4.2f} V'.format(*each_pot), fontsize=12)
                    if each != plot_y_labels[-1]:
                        ax_temp.set_xticklabels([])
                    else:
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=12)
                    if each_pot!=self.pot_range[0]:
                        ax_temp.set_yticklabels([])
                #print output data
                output_data = np.array(output_data).T
                output_data = np.append(output_data,np.array([int(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans])[:,np.newaxis],axis=1)
                output_data = np.append(np.array([int(each_) for each_ in self.scans])[:,np.newaxis],output_data,axis = 1)
                # print('\n')
                # print(each_pot)
                plain_text.append(f'\npot = {each_pot} V')
                plain_text.append('scan_no\tstrain_ip\tstrain_oop\tgrain_size_ip\tgrain_size_oop\tpH')
                for each_row in output_data:
                    # print("{:3.0f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:2.0f}".format(*each_row))
                    plain_text.append("{:3.0f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t\t{:2.0f}".format(*each_row))
            self.mplwidget2.fig.subplots_adjust(hspace=0.04)
            self.mplwidget2.canvas.draw()
            self.plainTextEdit_summary.setPlainText('\n'.join(plain_text))
        else:
            pass

    #extract the potential seperation in the fit file
    def return_seperator_values(self,scan):
        file = self.lineEdit_slope_file.text()
        data = pd.read_csv(file,sep='\t',comment = '#')
        summary = {}
        summary[scan] = {}
        for_current =[]
        try:
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                #_p1 is the associated potential value at the cross point (seperator)
                summary[scan][item] = [data['scan{}'.format(scan)]['{}_p1'.format(item)]+self.potential_offset]
                for_current.append(summary[scan][item])
            summary[scan]['current'] = for_current
        except:
            summary[scan] = None
        return summary

    #get slope/intercept values from fit files
    #dic of scan_number
    #each item is a dic of structural values, each structure value (eg strain_ip) corresponds to 6 items [p0, p1,p2,y1,a1,a2]
    #p's are potentials at the left bound(p0), right bound (p2), and cross point (p1), y1 is the associated value at p1, a1 and a2 are two slopes
    def return_slope_values(self):
        file = self.lineEdit_slope_file.text()
        data = pd.read_csv(file,sep='\t',comment = '#')
        summary = {}
        for each in self.scans:
            summary[each] = {}
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                try:
                    summary[each][item] = [data['scan{}'.format(each)]['{}_p{}'.format(item,i)]+self.potential_offset for i in range(3)]
                    summary[each][item] = summary[each][item] + [data['scan{}'.format(each)]['{}_y1'.format(item)]]
                    summary[each][item] = summary[each][item] +  list(data['scan{}'.format(each)]['{}'.format(item)])[::-1]
                except:
                    error_pop_up('Could not extract the slope value from fit file', 'Error')
        #each item = [p0,p1,p2,y1,a1,a2], as are slope values, (p1,y1) transition value, y0 and y2 are end points for potentials
        return summary

    #when using external_slope, the potential range is determined by the fitted cross point and the bounds you provied on the GUI (in tab of More setup, potential_range)
    #If not, just use the ranges specified in tab of Basic setup, potential ranges selector
    def cal_potential_ranges(self,scan):
        f = lambda x:(round(x[0],3),round(x[1],3))
        if self.checkBox_use_external_slope.isChecked():
            slope_info_temp = self.return_slope_values()
        else:
            slope_info_temp = None
        if slope_info_temp == None:
            self.pot_ranges[scan] = [f(each) for each in self.pot_range]
        else:
            _,p1,*_ = slope_info_temp[scan]["strain_ip"]#p1 y1 is the cross point, thus should be the same for all structural pars
            pot_range_specified = eval("({})".format(self.lineEdit_pot_range.text().rstrip()))
            if p1>pot_range_specified[1]:
                p1 = sum(pot_range_specified)/2
            pot_range1 = (pot_range_specified[0], p1)
            pot_range2 = (p1, pot_range_specified[1])
            pot_range3 = pot_range_specified
            self.pot_ranges[scan] = [f(pot_range1),f(pot_range2),f(pot_range3)]
        # print('now update potential range')
        # print(self.pot_ranges)

    def plot_reaction_order_and_tafel(self):
        self.widget_cv_view.canvas.figure.clear()
        ax_tafel = self.widget_cv_view.canvas.figure.add_subplot(1,2,1)
        ax_order = self.widget_cv_view.canvas.figure.add_subplot(1,2,2)
        if self.cv_tool.info['reaction_order_mode'] == 'constant_potential':
            constant_value = self.cv_tool.info['potential_reaction_order']
        elif self.cv_tool.info['reaction_order_mode'] == 'constant_current':
            constant_value = self.cv_tool.info['current_reaction_order']
        mode = self.cv_tool.info['reaction_order_mode']
        forward_cycle = True
        self.cv_tool.plot_tafel_with_reaction_order(ax_tafel, ax_order,constant_value = constant_value,mode = mode, forward_cycle = forward_cycle)
        self._format_axis(ax_tafel)
        self._format_axis(ax_order)
        self._format_ax_tick_labels(ax = ax_tafel,
                fun_set_bounds = 'set_ylim', 
                bounds = [0.05,10], 
                bound_padding = 0., 
                major_tick_location =[0.1,1,10], 
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks=10, 
                fmt_str = '{:.0e}')
        self._format_ax_tick_labels(ax = ax_tafel,
                fun_set_bounds = 'set_xlim', 
                bounds = [1.55,1.85], 
                bound_padding = 0., 
                major_tick_location =[1.55,1.6,1.65,1.7,1.75,1.8,1.85], 
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks=5, 
                fmt_str = '{: 4.2f}')
        self._format_ax_tick_labels(ax = ax_order,
                fun_set_bounds = 'set_ylim', 
                bounds = [1.62,1.82], 
                bound_padding = 0.01, 
                major_tick_location =[1.62,1.67,1.72,1.77,1.82], 
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks=5, 
                fmt_str = '{: 4.2f}')
        self.widget_cv_view.canvas.draw()


    def plot_cv_data(self):
        
        self.widget_cv_view.canvas.figure.clear()
        col_num = 4
        axs = [self.widget_cv_view.canvas.figure.add_subplot(len(self.cv_tool.cv_info), col_num, 1 + col_num*(i-1) ) for i in range(1,len(self.cv_tool.cv_info)+1)]
        self.cv_tool.plot_cv_files(axs = axs)
        axs_2 = [self.widget_cv_view.canvas.figure.add_subplot(len(self.cv_tool.cv_info), col_num, 1 + col_num*(i-1)+1) for i in range(1,len(self.cv_tool.cv_info)+1)]
        scans = list(self.cv_tool.cv_info.keys())
        scans = sorted(scans)
        min_x, max_x = 10000000, -10000000
        min_y, max_y = 10000000, -10000000

        for i in range(len(scans)):
            min_x_, max_x_, min_y_, max_y_ = self.cv_tool.plot_tafel_from_formatted_cv_info_one_scan(scans[i], axs_2[i])
            # min_x_, max_x_, min_y_, max_y_ = self.cv_tool.plot_tafel_from_formatted_cv_info_one_scan(scans[i], axs_2[0])
            # print(min_x_, max_x_, min_y_, max_y_)
            if min_x_<min_x:
                min_x = min_x_
            if min_y_<min_y:
                min_y = min_y_
            if max_x_>max_x:
                max_x = max_x_
            if max_y_>max_y:
                max_y = max_y_ 
        for each in axs_2:
            each.set_xlim(min_x, max_x)
            each.set_ylim(min_y, max_y)
            # each.yaxis.tick_right()
            each.yaxis.set_label_position("right")

        #self.widget_cv_view.fig.tight_layout()
        # print(self.data_summary)
        self.widget_cv_view.fig.subplots_adjust(wspace=0.24,hspace=0.04)
        self.widget_cv_view.canvas.draw()

    #plot the master figure
    def plot_figure_xrv(self):
        self.reset_meta_data()
        self.extract_tick_label_settings()

        if self.checkBox_use_external_slope.isChecked():
            slope_info_temp = self.return_slope_values()
        else:
            slope_info_temp = None

        self.mplwidget.fig.clear()
        plot_dim = [len(self.plot_labels_y), len(self.scans)]
        for scan in self.scans:
            setattr(self,'plot_axis_scan{}'.format(scan),[])
            j = self.scans.index(scan) + 1
            for i in range(plot_dim[0]):
                getattr(self,'plot_axis_scan{}'.format(scan)).append(self.mplwidget.canvas.figure.add_subplot(plot_dim[0], plot_dim[1],j+plot_dim[1]*i))
                self._format_axis(getattr(self,'plot_axis_scan{}'.format(scan))[-1])
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(which = 'major', axis="x", direction="in")
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(which = 'minor', axis="x", direction="in")
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(which = 'major', axis="y", direction="in")
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(which = 'minor', axis="y", direction="in")
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(which = 'major', bottom=True, top=True, left=True, right=True)
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(which = 'minor', bottom=True, top=True, left=True, right=True)
                # getattr(self,'plot_axis_scan{}'.format(scan))[-1].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                #getattr(self,'plot_axis_scan{}'.format(scan))[-1].xaxis.set_minor_locator(AutoMinorLocator())
        y_max_values,y_min_values = [-100000000]*len(self.plot_labels_y),[100000000]*len(self.plot_labels_y)

        #prepare ranges for viewing datasummary
        data_range = self.lineEdit_data_range.text().rsplit(',')
        if len(data_range) == 1:
            data_range = [list(map(int,data_range[0].rsplit('-')))]*len(self.scans)
        else:
            assert len(data_range) == len(self.scans)
            data_range = [list(map(int,each.rsplit('-'))) for each in data_range]
        self.data_range = data_range

        pot_range = self.lineEdit_potential_range.text().rsplit(',')
        if pot_range == ['']:
            self.pot_range = []
        else:
            self.pot_range = [list(map(float,each.rsplit('-'))) for each in pot_range]
            pot_range_ = []
            for each in self.pot_range:
                if len(each)==1:
                    pot_range_.append(each*2)
                elif len(each)==2:
                    pot_range_.append(each)
            self.pot_range = pot_range_
        #count the times of dataset for pH 13
        count_pH13_temp = 1
        for scan in self.scans:
            self.cal_potential_ranges(scan)
            self.data_summary[scan] = {}
            if 'potential' in self.plot_labels_y and self.plot_label_x == 'potential':
                plot_labels_y = [each for each in self.plot_labels_y if each!='potential']
            else:
                plot_labels_y = self.plot_labels_y
            for each in plot_labels_y:
                self.data_summary[scan][each] = []
                i = plot_labels_y.index(each)
                try:
                    fmt = self.lineEdit_fmt.text().rsplit(',')[self.scans.index(scan)].rsplit(";")
                except:
                    fmt = 'b-'
                y = self.data_to_plot[scan][plot_labels_y[i]]
                y_smooth_temp = signal.savgol_filter(self.data_to_plot[scan][plot_labels_y[i]],41,2)
                std_val = np.sum(np.abs(y_smooth_temp - y))/len(self.data_to_plot[scan][self.plot_label_x])
                marker_index_container = []
                for ii in range(len(self.pot_range)):
                    #if len(self.pot_range[ii])==1:
                    pot_range_temp = self.pot_range[ii]
                    data_range_temp = self.data_range[self.scans.index(scan)]
                    #print(list(self.data_to_plot.keys()))
                    assert 'potential' in list(self.data_to_plot[scan].keys())
                    index_left = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range_temp[0]:data_range_temp[1]] - pot_range_temp[0])) + data_range_temp[0]
                    index_right = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range_temp[0]:data_range_temp[1]] - pot_range_temp[1])) + data_range_temp[0]
                    marker_index_container.append(index_left)
                    marker_index_container.append(index_right)

                    pot_offset = abs(self.data_to_plot[scan]['potential'][index_left]-self.data_to_plot[scan]['potential'][index_right])
                    #data_temp = [(y_smooth_temp[index_left] - y_smooth_temp[index_right])/pot_offset,std_val/pot_offset]
                    if pot_offset==0:
                        self.data_summary[scan][each].append((y_smooth_temp[index_left]))
                        self.data_summary[scan][each].append(std_val)
                    else:
                        self.data_summary[scan][each].append((y_smooth_temp[index_left] - y_smooth_temp[index_right])/pot_offset)
                        self.data_summary[scan][each].append(std_val/pot_offset)
                
                #plot the results
                if len(fmt)==2:
                    fmt = fmt[int('size' in each)]
                else:
                    fmt = fmt[0]
                if self.plot_label_x == 'image_no':
                    if each != 'current':
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y,fmt,markersize = 3)
                        # getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y_smooth_temp,'-')
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([np.arange(len(y))[iii] for iii in marker_index_container],[y_smooth_temp[iii] for iii in marker_index_container],'k*')
                    else:
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y*8,fmt,markersize = 3)
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([np.arange(len(y))[iii] for iii in marker_index_container],[y[iii]*8 for iii in marker_index_container],'k*')
                else:
                    if self.checkBox_use_external_slope.isChecked():
                        seperators = self.return_seperator_values(scan)
                    else:
                        seperators = list(set(marker_index_container))
                    if each!='current':
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y,fmt,markersize = 2)
                        if self.checkBox_plot_slope.isChecked() and self.checkBox_use_external_slope.isChecked():
                            if slope_info_temp[scan][each]!=None:
                                p0,p1,p2,y1,a1,a2 = slope_info_temp[scan][each]
                                y0 = a1*(p0-p1)+y1
                                y2 = a2*(p2-p1)+y1
                                cases = [self.calculate_size_strain_change(p0,p1,p2,y1,a1,a2,pot_range = each_pot) for each_pot in self.pot_ranges[scan]]
                                if each=='grain_size_ip':
                                    self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_ranges[scan],'horizontal',cases)
                                elif each == 'grain_size_oop':
                                    self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_ranges[scan],'vertical',cases)
                                elif each == 'strain_ip':
                                    self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_ranges[scan],'horizontal',cases)
                                elif each == 'strain_oop':
                                    self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_ranges[scan],'vertical',cases)
                                getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([p0,p1,p2],np.array([y0,y1,y2])-self.data_to_plot[scan][each+"_max"],'k--')
                        else:
                            cases = [self.calculate_size_strain_change_from_plot_data(scan, each, self.data_range[self.scans.index(scan)], each_pot) for each_pot in self.pot_range]
                            #cases = [self.calculate_size_strain_change(p0,p1,p2,y1,a1,a2,pot_range = each) for each in self.pot_ranges[scan]]
                            if each=='grain_size_ip':
                                self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_range,'horizontal',cases)
                            elif each == 'grain_size_oop':
                                self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_range,'vertical',cases)
                            elif each == 'strain_ip':
                                self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_range,'horizontal',cases)
                            elif each == 'strain_oop':
                                self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_range,'vertical',cases)
                            #getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([p0,p1,p2],np.array([y0,y1,y2])-self.data_to_plot[scan][each+"_max"],'k--')
                        if self.checkBox_use_external_slope.isChecked():
                            try:
                                for pot in seperators[scan][each]:
                                    getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([pot,pot],[-100,100],'k:')
                            except:
                                pass
                        else:
                            if self.checkBox_show_marker.isChecked():
                                getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y_smooth_temp[iii] for iii in marker_index_container],'k*')
                                for each_index in seperators:
                                    pot = self.data_to_plot[scan][self.plot_label_x][each_index]
                                    getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([pot,pot],[-100,100],'k:')
                    else:
                        if self.checkBox_use_external_cv.isChecked():
                            try:
                                if self.checkBox_use_external_slope.isChecked():
                                    lim_y = self.plot_cv_from_external(getattr(self,'plot_axis_scan{}'.format(scan))[i],scan,seperators[scan][each])
                                else:
                                    pots_ = []
                                    for each_index in seperators:
                                        pots_.append([self.data_to_plot[scan][self.plot_label_x][each_index]])
                                    lim_y = self.plot_cv_from_external(getattr(self,'plot_axis_scan{}'.format(scan))[i],scan,pots_)
                                if self.checkBox_use_internal_cv.isChecked(): 
                                    getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y*8,fmt,markersize = 3)
                                    # getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y[iii]*8 for iii in marker_index_container],'k*')
                            except:
                                getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y*8,fmt,markersize = 3)
                                # getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y[iii]*8 for iii in marker_index_container],'k*')
                        else:
                            getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y*8,fmt,markersize = 3)
                            #getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y[iii]*8 for iii in marker_index_container],'k*')
                            if not self.checkBox_use_external_slope.isChecked():
                                if self.checkBox_show_marker.isChecked():
                                    for each_index in seperators:
                                        pot = self.data_to_plot[scan][self.plot_label_x][each_index]
                                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([pot,pot],[-100,100],'k:')
                            else:
                                for each_item in seperators[scan][each]:
                                    try:
                                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([each_item,each_item],[-100,100],':k')
                                    except:
                                        pass
                if i==0:
                    # getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}_scan{}'.format(self.phs[self.scans.index(scan)],scan),fontsize=11)
                    getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}'.format(self.phs[self.scans.index(scan)]),fontsize=11)
                    if self.phs[self.scans.index(scan)]==13:
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}_{}'.format(self.phs[self.scans.index(scan)],count_pH13_temp),fontsize=11)
                        count_pH13_temp+=1
                if each=='current':
                    try:
                        temp_min,temp_max = lim_y
                    except:
                        temp_max, temp_min = max(list(self.data_to_plot[scan][plot_labels_y[i]]*8)),min(list(self.data_to_plot[scan][plot_labels_y[i]]*8))
                else:
                    temp_max, temp_min = max(list(self.data_to_plot[scan][plot_labels_y[i]])),min(list(self.data_to_plot[scan][plot_labels_y[i]]))
                    
                if y_max_values[i]<temp_max:
                    y_max_values[i] = temp_max
                if y_min_values[i]>temp_min:
                    y_min_values[i] = temp_min
                if i!=(len(plot_labels_y)-1):
                    ax = getattr(self,'plot_axis_scan{}'.format(scan))[i]
                    ax.set_xticklabels([])
                else:
                    ax = getattr(self,'plot_axis_scan{}'.format(scan))[i]
                    x_label = [r'Image_no','E / V$_{RHE}$'][self.plot_label_x=='potential']
                    ax.set_xlabel(x_label, fontsize = 13)
                if scan!=self.scans[0]:
                    getattr(self,'plot_axis_scan{}'.format(scan))[i].set_yticklabels([])
                else:
                    y_label_map = {'potential':'E / V$_{RHE}$',
                                   'current':r'j / mAcm$^{-2}$',
                                   'strain_ip':r'$\Delta\varepsilon_\parallel$  (%)',
                                   'strain_oop':r'$\Delta\varepsilon_\perp$  (%)',
                                   'grain_size_oop':r'$\Delta d_\perp$ / nm',
                                   'grain_size_ip':r'$\Delta d_\parallel$ / nm',
                                   'peak_intensity':r'Intensity / a.u.'}
                    if each in y_label_map:
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].set_ylabel(y_label_map[each], fontsize = 13)
                    else:
                        pass

            for each_pot_range in self.pot_ranges[scan]:
                # if self.checkBox_use_external_slope.isChecked():
                try:
                    horizontal = self.grain_size_info_all_scans[scan][each_pot_range]['horizontal']
                    vertical = self.grain_size_info_all_scans[scan][each_pot_range]['vertical']
                    q_skin,q_film = self.estimate_charge_from_skin_layer_thickness_philippe_algorithm({"horizontal":horizontal,"vertical":vertical})
                    print("potential range:",each_pot_range)
                    print({"horizontal":horizontal,"vertical":vertical})
                    print('Skin charge calculated for scan{} using philippe algorithm is:{} mC/m2'.format(scan, q_skin))
                    if scan not in self.charge_info:
                        self.charge_info[scan] = {}
                        self.charge_info[scan][each_pot_range] = {'skin_charge':q_skin,'film_charge':q_film,'total_charge':0}
                    else:
                        self.charge_info[scan][each_pot_range]['skin_charge'] = q_skin
                        self.charge_info[scan][each_pot_range]['film_charge'] = q_film
                except:
                    print('Fail to cal charge info. Check!')
                # else:
                    # pass
        # print(self.strain_info_all_scans[221])

        for scan in self.scans:
            for each in self.plot_labels_y:
                i = self.plot_labels_y.index(each)
                if 'current' not in self.plot_labels_y:
                    pass
                else:
                    j = self.plot_labels_y.index('current')
                    if each != 'current':#synchronize the x_lim of all non-current to that for CV
                        x_lim = getattr(self,'plot_axis_scan{}'.format(scan))[j].get_xlim()
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].set_xlim(*x_lim)
                    else:
                        pass
                #padding_offset = 0.01
                #getattr(self,'plot_axis_scan{}'.format(scan))[i].set_ylim(y_min_values[i]-padding_offset, y_max_values[i]+padding_offset)
                ####The following lines are customized for a specific dataset, it is subject to change depending on the dataset you are using
                #getattr(self,'plot_axis_scan{}'.format(scan))[i].set_xlim(0.95,1.95)
                if self.plot_label_x == 'potential':
                    # x_locator = [1,1.3,1.6,1.9]
                    if 'potential' in self.tick_label_settings['master']:
                        {'locator':locator,'padding':padding,'tick_num':tick_num,'fmt':fmt,'func':func}
                        x_locator = [0.5,1,1.5,2]
                        self._format_ax_tick_labels(ax = getattr(self,'plot_axis_scan{}'.format(scan))[i],
                                                    fun_set_bounds = self.tick_label_settings['master']['func']#'set_xlim', 
                                                    bounds = x_lim#[0.4,2.1],#[0.95,1.95], 
                                                    bound_padding = float(self.tick_label_settings['master']['padding']), 
                                                    major_tick_location = self.tick_label_settings['master']['locator'], #x_locator
                                                    show_major_tick_label = (len(self.plot_labels_y)-1)==i, #show major tick label for the first scan
                                                    num_of_minor_tick_marks=self.tick_label_settings['master']['tick_num'], #4
                                                    fmt_str = self.tick_label_settings['master']['fmt'])#'{:3.1f}'

                y_locator = None
                if each == 'strain_ip':
                    # y_locator = [0, -0.02, -0.04, -0.06]
                    y_locator = [-0.3,-0.2,-0.1,0]
                elif each == 'strain_oop':
                    # y_locator = [0,-0.05,-0.10,-0.15,-0.20]
                    y_locator = [-0.6,-0.4,-0.2,0]
                elif each == 'grain_size_ip':
                    # y_locator = [0,-0.1,-0.2,-0.3,-0.40]
                    y_locator = [-4,-3,-2,-1,0]
                elif each == 'grain_size_oop':
                    # y_locator = [0,-0.2,-0.4,-0.6]
                    y_locator = [-2.5,-2,-1.5,-1,-0.5,0]
                elif each == 'current':
                    # y_locator = [-1,0,1,2,3,4]
                    y_locator = [-2,0,2,4,6]
                if y_locator!=None:
                    self._format_ax_tick_labels(ax = getattr(self,'plot_axis_scan{}'.format(scan))[i],
                                                fun_set_bounds = 'set_ylim', 
                                                bounds = [y_min_values[i],y_max_values[i]], 
                                                bound_padding = 0.01, 
                                                major_tick_location = y_locator, 
                                                show_major_tick_label = self.scans.index(scan)==0, #show major tick label for the first scan
                                                num_of_minor_tick_marks=4, 
                                                fmt_str = '{: 4.2f}')
        # self.print_data_summary()
        #self.actionPlotData.triggered.connect(self.print_data_summary)
        self.mplwidget.fig.tight_layout()
        # print(self.data_summary)
        self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

    #format the axis tick so that the tick facing inside, showing both major and minor tick marks
    #The tick marks on both sides (y tick marks on left and right side and x tick marks on top and bottom side)
    def _format_axis(self,ax):
        major_length = 4
        minor_length = 2
        if hasattr(ax,'__len__'):
            for each in ax:
                each.tick_params(which = 'major', axis="x", length = major_length, direction="in")
                each.tick_params(which = 'minor', axis="x", length = minor_length,direction="in")
                each.tick_params(which = 'major', axis="y", length = major_length, direction="in")
                each.tick_params(which = 'minor', axis="y", length = minor_length,direction="in")
                each.tick_params(which = 'major', bottom=True, top=True, left=True, right=True)
                each.tick_params(which = 'minor', bottom=True, top=True, left=True, right=True)
                each.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        else:
            ax.tick_params(which = 'major', axis="x", length = major_length,direction="in")
            ax.tick_params(which = 'minor', axis="x", length = minor_length,direction="in")
            ax.tick_params(which = 'major', axis="y", length = major_length,direction="in")
            ax.tick_params(which = 'minor', axis="y", length = minor_length,direction="in")
            ax.tick_params(which = 'major', bottom=True, top=True, left=True, right=True)
            ax.tick_params(which = 'minor', bottom=True, top=True, left=True, right=True)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    def _format_ax_tick_labels(self,ax,fun_set_bounds = 'set_ylim', bounds = [0,1], bound_padding = 0, major_tick_location = [], show_major_tick_label = True, num_of_minor_tick_marks=5, fmt_str = '{: 4.2f}'):
        mapping = {'set_ylim':'yaxis','set_xlim':'xaxis'}
        which_axis = mapping[fun_set_bounds]
        bounds_after_add_padding = bounds[0]-bound_padding, bounds[1]+bound_padding
        major_tick_labels = []
        for each in major_tick_location:
            if show_major_tick_label:
                major_tick_labels.append(fmt_str.format(each))
            else:
                major_tick_labels.append('')
        minor_tick_location = []
        minor_tick_labels = []
        for i in range(len(major_tick_location)-1):
            start = major_tick_location[i]
            end = major_tick_location[i+1]
            tick_spacing = (end-start)/(num_of_minor_tick_marks+1)
            for j in range(num_of_minor_tick_marks):
                minor_tick_location.append(start + tick_spacing*(j+1))
                minor_tick_labels.append('')#not showing minor tick labels
            #before starting point
            if i==0:
                count = 1
                while True:
                    if (start-count*abs(tick_spacing))<bounds_after_add_padding[0]:
                        break
                    else:
                        minor_tick_location.append(start - abs(tick_spacing)*count)
                        minor_tick_labels.append('')#not showing minor tick labels
                        count = count+1
            #after the last point
            elif i == (len(major_tick_location)-2):
                count = 1
                while True:
                    if (end+count*abs(tick_spacing))>bounds_after_add_padding[1]:
                        break
                    else:
                        minor_tick_location.append(end + abs(tick_spacing)*count)
                        minor_tick_labels.append('')#not showing minor tick labels
                        count = count+1

        #set limits
        getattr(ax,fun_set_bounds)(*bounds_after_add_padding)
        #set major tick and tick labels
        getattr(ax, which_axis).set_major_formatter(FixedFormatter(major_tick_labels))
        getattr(ax, which_axis).set_major_locator(FixedLocator(major_tick_location))
        #set minor tick and tick lables (not showing the lable)
        getattr(ax, which_axis).set_minor_formatter(FixedFormatter(minor_tick_labels))
        getattr(ax, which_axis).set_minor_locator(FixedLocator(minor_tick_location))
        

    def prepare_data_to_plot_xrv(self,plot_label_list, scan_number):
        if scan_number in self.image_range_info:
            l , r = self.image_range_info[scan_number]
        else:
            l, r = 0, 100000000
        if hasattr(self,'data_to_plot'):
            self.data_to_plot[scan_number] = {}
        else:
            self.data_to_plot = {}
            self.data_to_plot[scan_number] = {}
        if self.checkBox_mask.isChecked():
            condition = (self.data['mask_cv_xrd'] == True)&(self.data['scan_no'] == scan_number)
        else:
            condition = self.data['scan_no'] == scan_number
        #RHE potential, potential always in the y dataset
        self.data_to_plot[scan_number]['potential'] = self.potential_offset + 0.205+np.array(self.data[condition]['potential'])[l:r]+0.059*np.array(self.data[self.data['scan_no'] == scan_number]['phs'])[0]
        for each in plot_label_list:
            if each == 'potential':
                pass
            elif each=='current':#RHE potential
                self.data_to_plot[scan_number][each] = -np.array(self.data[condition][each])[l:r]
            else:
                if each in ['peak_intensity','peak_intensity_error','strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                    temp_data = np.array(self.data[condition][each])[l:r]
                    # self.data_to_plot[scan_number][each] = list(temp_data-max(temp_data))
                    if self.checkBox_max.isChecked():
                        self.data_to_plot[scan_number][each] = list(temp_data-max(temp_data))
                        self.data_to_plot[scan_number][each+'_max'] = max(temp_data) 
                    else:
                        self.data_to_plot[scan_number][each] = list(temp_data)
                        self.data_to_plot[scan_number][each+'_max'] = 0 
                else:
                    self.data_to_plot[scan_number][each] = list(self.data[condition][each])[l:r]

    def append_scans_xrv(self):
        text = self.scan_numbers_append.text()
        if text!='':
            scans = list(set([int(each) for each in text.rstrip().rsplit(',')]))
        else:
            return
        scans.sort()
        assert (self.lineEdit_x.text()!='' and self.lineEdit_y.text()!=''), 'No channels for plotting have been selected!'
        plot_labels = self.lineEdit_x.text() + ',' + self.lineEdit_y.text()
        plot_labels = plot_labels.rstrip().rsplit(',')
        for scan in scans:
            self.prepare_data_to_plot_xrv(plot_labels,scan)
            print('Prepare data for scan {} now!'.format(scan))
        self.scans = scans
        self.phs = [self.phs_all[self.scans_all.index(each)] for each in scans]
        self.plot_label_x = self.lineEdit_x.text()
        self.plot_labels_y = self.lineEdit_y.text().rstrip().rsplit(',')
        self.comboBox_scans.clear()
        self.comboBox_scans.addItems([str(each) for each in sorted(scans)])

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())