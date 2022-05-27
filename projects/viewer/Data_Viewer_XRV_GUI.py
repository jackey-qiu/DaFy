import sys,os,qdarkstyle
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import uic
import PyQt5
import random
import numpy as np
import matplotlib.pyplot as plt
import zipfile, pickle
try:
    from . import locate_path_viewer
except:
    import locate_path_viewer
script_path = locate_path_viewer.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
from UtilityFunctions import PandasModel, colorline
from cv_tool import cvAnalysis
from charge_calculation import calculate_charge
from PlotSetup import data_viewer_plot_cv, RHE, plot_tafel_from_formatted_cv_info
import pandas as pd
import time
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator, FixedFormatter
os.environ["QT_MAC_WANTS_LAYER"] = "1"
# matplotlib.use("TkAgg")
from scipy import signal, stats
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
        uic.loadUi(os.path.join(DaFy_path,'projects','viewer','data_viewer_xrv_gui.ui'),self)
        self.lineEdit_data_range.hide()
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
        plt.rcParams['savefig.dpi'] = 300
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
        self.pushButton_append_rows.clicked.connect(lambda:self.update_pandas_model_cv_setting())
        self.pushButton_apply.clicked.connect(self.update_pot_offset)
        self.pushButton_tweak.clicked.connect(self.tweak_one_channel)
        self.pushButton_load_cv_config.clicked.connect(self.load_cv_config_file)
        self.pushButton_update_cv_config.clicked.connect(self.update_cv_config_file)
        self.pushButton_plot_cv.clicked.connect(self.plot_cv_data)
        self.pushButton_cal_charge_2.clicked.connect(self.calculate_charge_2)
        # self.pushButton_plot_reaction_order.clicked.connect(self.cv_tool.plot_reaction_order_with_pH)
        self.pushButton_plot_reaction_order.clicked.connect(self.plot_reaction_order_and_tafel)
        self.pushButton_get_pars.clicked.connect(self.project_cv_settings)

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
        self.init_pandas_model_ax_format()
        self.init_pandas_model_cv_setting()
        self.pushButton_bkg_fit.clicked.connect(self.perform_bkg_fitting)
        self.pushButton_extract_cv.clicked.connect(self.extract_cv_data)
        self.pushButton_project.clicked.connect(self.project_to_master)
        self.pushButton_add_link.clicked.connect(self.add_one_link)
        self.pushButton_remove_selected.clicked.connect(self.remove_one_item)
        self.pushButton_remove_all.clicked.connect(lambda:self.comboBox_link_container.clear())
        self.GUI_metaparameter_channels  = ['lineEdit_data_file',
                                            'checkBox_time_scan',
                                            'checkBox_use',
                                            'checkBox_mask',
                                            'checkBox_max',
                                            'lineEdit_x',
                                            'lineEdit_y',
                                            'scan_numbers_append',
                                            'lineEdit_fmt',
                                            'lineEdit_potential_range', 
                                            'lineEdit_pot_range', 
                                            'lineEdit_scan_rate',
                                            'lineEdit_data_range',
                                            'lineEdit_colors_bar',
                                            'checkBox_use_external_cv',
                                            'checkBox_use_internal_cv',
                                            'checkBox_plot_slope',
                                            'checkBox_use_external_slope',
                                            'lineEdit_pot_offset',
                                            'lineEdit_cv_folder',
                                            'lineEdit_slope_file',
                                            'lineEdit_reference_potential',
                                            'checkBox_show_marker',
                                            'checkBox_merge']

    def add_one_link(self):
        link_1 = self.comboBox_link_1.currentText()
        link_2 = self.comboBox_link_2.currentText()
        link_group = f"{link_1}+{link_2}"
        current_list = [self.comboBox_link_container.itemText(i) for i in range(self.comboBox_link_container.count())]
        if link_group in current_list:
            return
        if len(current_list)==4:
            error_pop_up('You can add 4 group links at most. Remove some item and continue to add this new item.')
        self.comboBox_link_container.clear()
        self.comboBox_link_container.addItems(current_list + [link_group])

    def remove_one_item(self):
        current_list = [self.comboBox_link_container.itemText(i) for i in range(self.comboBox_link_container.count())]
        current_item = self.comboBox_link_container.currentText()
        new_list = [each for each in current_list if each!=current_item]
        self.comboBox_link_container.clear()
        self.comboBox_link_container.addItems(new_list)

    def perform_bkg_fitting(self):
        order = 0
        if self.checkBox_order1.isChecked():
            order+=1
        if self.checkBox_order2.isChecked():
            order+=2
        if self.checkBox_order3.isChecked():
            order+=3
        if self.checkBox_order4.isChecked():
            order+=4
        fct = 'atq'
        if self.radioButton_stq.isChecked():
            fct = 'stq'
        if self.radioButton_sh.isChecked():
            fct = 'sh'
        if self.radioButton_ah.isChecked():
            fct = 'ah'
        s = self.doubleSpinBox_ss_factor.value()
        scan_rate = float(self.lineEdit_scan_rate.text())
        charge = self.widget.perform_bkg_fitting(order, s, fct, scan_rate)
        self.lineEdit_charge.setText(f'{charge} mC/cm2')

    def project_to_master(self):
        scan_no = int(self.comboBox_scans_3.currentText())
        index_lf, index_rg = list(map(int,list(self.widget.region.getRegion())))
        pot = self.widget.p3_handle.getData()[1][index_lf:index_rg]
        current_bkg = self.widget.p1_bkg_handle.getData()[1][index_lf:index_rg]
        cv_scale_factor= self.plot_lib[scan_no][2]
        getattr(self,'plot_axis_scan{}'.format(scan_no))[0].plot(pot, current_bkg * cv_scale_factor, '--k')
        self.mplwidget.fig.tight_layout()
        self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

    def init_pandas_model_ax_format(self):
        data_ = {}
        data_['use'] = [True]*6 + [False] * 14
        data_['type'] = ['master']*6 + ['bar']*14
        data_['channel'] = ['potential','current','strain_ip','strain_oop','grain_size_ip','grain_size_oop']
        data_['channel'] = data_['channel'] + ['strain_ip','strain_oop','grain_size_ip','grain_size_oop'] + \
                           ['dV_bulk', 'dV_skin', '<dskin>','OER_E', 'OER_j', 'OER_j/<dskin>','pH', 'q_cv','q_film','input']
        data_['tick_locs'] = ['[0,1,2,3]']*(6+14)
        data_['padding'] = ['0.1']*(6+14)
        data_['#minor_tick'] = ['4']*(6+14)
        data_['fmt_str'] = ["{: 4.2f}"]*(6+14)
        data_['func'] = ['set_xlim'] + ['set_ylim']*(5+14)
        self.pandas_model_in_ax_format = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_ax_format, main_gui = self, check_columns = [0])
        self.tableView_ax_format.setModel(self.pandas_model_in_ax_format)
        self.tableView_ax_format.resizeColumnsToContents()
        self.tableView_ax_format.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def _init_meta_data_for_scans(self, scans):
        for each in scans:
            setattr(self, f'strain_ip_offset_{each}',0)
            setattr(self, f'strain_oop_offset_{each}',0)
            setattr(self, f'grain_size_ip_offset_{each}',0)
            setattr(self, f'grain_size_oop_offset_{each}',0)
            setattr(self, f'potential_offset_{each}',0)
            setattr(self, f'current_offset_{each}',0)
            setattr(self, f'image_no_offset_{each}',0)

    def tweak_one_channel(self):
        scan = int(self.comboBox_scans_2.currentText())
        channel = self.comboBox_tweak_channel.currentText()
        tweak_value = self.doubleSpinBox_offset.value()
        if channel == 'image_no':
            tweak_value = int(tweak_value)
        setattr(self, f'{channel}_offset_{scan}',tweak_value)
        self.plot_figure_xrv()

    def extract_tick_label_settings(self):
        if hasattr(self,'plainTextEdit_tick_label_settings'):
            strings = self.plainTextEdit_tick_label_settings.toPlainText()
            lines = strings.rsplit('\n')
            for each_line in lines:
                if each_line.startswith('#'):
                    pass
                else:
                    items = each_line.rstrip().rsplit('+')
                    key,item,locator,padding,tick_num,fmt,func = items
                    locator = eval(locator)
                    if key in self.tick_label_settings:
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}
                    else:
                        self.tick_label_settings[key] = {}
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}
        elif hasattr(self, 'tableView_ax_format'):
            cols = self.pandas_model_in_ax_format._data.shape[0]
            for i in range(cols):
                if self.pandas_model_in_ax_format._data.iloc[i,0]:
                    key,item,locator,padding,tick_num,fmt,func = self.pandas_model_in_ax_format._data.iloc[i,1:].tolist()
                    locator = eval(locator)
                    if key in self.tick_label_settings:
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}
                    else:
                        self.tick_label_settings[key] = {}
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}

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
        self.append_scans_xrv()
        self.plot_figure_xrv()

    def init_pandas_model_cv_setting(self):
        data_ = {}
        rows = self.spinBox_cv_rows.value()
        data_['use'] = [False] * rows
        data_['scan'] = [''] * rows
        data_['cv_name'] = [''] * rows
        data_['cycle'] = ['0'] * rows
        data_['scaling'] = ['30'] * rows
        data_['smooth_len'] = ['15'] * rows
        data_['smooth_order'] = ['1'] * rows
        data_['color'] = ['r'] * rows
        data_['pH'] = ['13'] * rows
        data_['extract_func'] = ['extract_cv_file_fouad'] * rows
        self.pandas_model_cv_setting = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_cv_setting, main_gui = self, check_columns = [0])
        self.tableView_cv_setting.setModel(self.pandas_model_cv_setting)
        self.tableView_cv_setting.resizeColumnsToContents()
        self.tableView_cv_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def update_pandas_model_cv_setting(self, reset = False, data = {}):
        if not reset:
            rows = self.spinBox_cv_rows.value() - self.pandas_model_cv_setting._data.shape[0]
            data_ = self.pandas_model_cv_setting._data.to_dict()
            data_new = {}
            if rows<=0:
                return
            else:
                for each in data_:
                    data_new[each] = [data_[each][self.pandas_model_cv_setting._data.shape[0]-1]]*rows
                self.pandas_model_cv_setting._data = pd.concat([self.pandas_model_cv_setting._data,pd.DataFrame(data_new, index = np.arange(self.pandas_model_cv_setting._data.shape[0],self.pandas_model_cv_setting._data.shape[0]+rows))])
                self.pandas_model_cv_setting = PandasModel(data = self.pandas_model_cv_setting._data, tableviewer = self.tableView_cv_setting, main_gui = self, check_columns = [0])
                self.tableView_cv_setting.setModel(self.pandas_model_cv_setting)
                self.tableView_cv_setting.resizeColumnsToContents()
                self.tableView_cv_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
        else:
            self.pandas_model_cv_setting._data = pd.DataFrame(data)
            self.pandas_model_cv_setting = PandasModel(data = self.pandas_model_cv_setting._data, tableviewer = self.tableView_cv_setting, main_gui = self, check_columns = [0])
            self.tableView_cv_setting.setModel(self.pandas_model_cv_setting)
            self.tableView_cv_setting.resizeColumnsToContents()
            self.tableView_cv_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def project_cv_settings(self):
        self.lineEdit_cv_config_path.setText(os.path.join(DaFy_path,'dump_files','temp.ini'))
        num_items = self.pandas_model_cv_setting._data.shape[0]
        pot_padding = eval(self.pandas_model_in_ax_format._data.iloc[0,4])
        pot_min = eval(self.pandas_model_in_ax_format._data.iloc[0,3])[0]-pot_padding
        pot_max = eval(self.pandas_model_in_ax_format._data.iloc[0,3])[-1]+pot_padding
        pot_bounds = f'[{round(pot_min,3)},{round(pot_max,3)}]'

        current_padding = eval(self.pandas_model_in_ax_format._data.iloc[1,4])
        current_min = eval(self.pandas_model_in_ax_format._data.iloc[1,3])[0]-current_padding
        current_max = eval(self.pandas_model_in_ax_format._data.iloc[1,3])[-1]+current_padding
        current_bounds = f'[{current_min},{current_max}]'

        key_value_map = {'Data_Info':
                                    {'sequence_id':str(list(map(eval,self.pandas_model_cv_setting._data['scan'].to_list()))),
                                     'selected_scan':str(list(map(eval,self.pandas_model_cv_setting._data['scan'].to_list()))),
                                     'cv_folder':self.lineEdit_cv_folder.text(),
                                     'path':str(self.pandas_model_cv_setting._data['cv_name'].to_list()),
                                     'ph':str(list(map(eval, self.pandas_model_cv_setting._data['pH'].to_list())))},
                          'General_Format_Settings':
                                     {'fmt':"['-']*{}".format(num_items),
                                      'color':str(self.pandas_model_cv_setting._data['color'].to_list()),
                                      'index_header_pos_offset_cv':str([0]*num_items),
                                      'index_header_pos_offset_tafel':str([0]*num_items),
                                      'index_header_pos_offset_order':str([0]*num_items),
                                      'tafel_show_tick_label_x_y':str([True]*num_items),
                                      'order_show_tick_label_x_y':str([True]*num_items),
                                      },
                          'Axis_Format_Settings':
                                      {'cv_bounds_pot':pot_bounds+'+'+'+'.join(self.pandas_model_in_ax_format._data.iloc[0,3:].to_list()),
                                       'cv_bounds_current':current_bounds+'+'+'+'.join(self.pandas_model_in_ax_format._data.iloc[1,3:].to_list()),
                                       'cv_show_tick_label_x':str([False]*(num_items-1) + [True]),
                                       'cv_show_tick_label_y':str([True]*num_items)
                                      },
                          'Data_Analysis_settings':
                                      {'cv_scale_factor':str([int(each) for each in self.pandas_model_cv_setting._data.iloc[:,4].to_list()]),
                                       'scale_factor_text_pos':str([(1.35,2.0)]*num_items),
                                       'cv_spike_cut':str([0.002]*num_items),
                                       'scan_rate':str([float(self.lineEdit_scan_rate.text())]*num_items),
                                       'resistance':str([50]*num_items),
                                       'which_cycle':str([eval(each)[0] for each in self.pandas_model_cv_setting._data.iloc[:,3].to_list()]),
                                       'method':str(self.pandas_model_cv_setting._data.iloc[:,-1].to_list()),
                                       'pot_range':str([[1.2,1.6]]*num_items),
                                       'pot_starts_tafel':str([1.68]*num_items),
                                       'pot_ends_tafel':str([1.72]*num_items),
                                      }
                        }

        for each_section in key_value_map:
            for each_item in key_value_map[each_section]:
                print(each_section, each_item)
                self.widget_par_tree.set_field(each_section, each_item, key_value_map[each_section][each_item])

    def make_plot_lib(self):
        self.plot_lib = {}
        if hasattr(self, 'textEdit_plot_lib'):
            info = self.textEdit_plot_lib.toPlainText().rsplit('\n')
            folder = self.lineEdit_cv_folder.text()
            if info==[''] or folder=='':
                return
            for each in info:
                if not each.startswith('#'):
                    # scan, cv, cycle, cutoff,scale,color, ph, func = each.replace(" ","").rstrip().rsplit(',')
                    scan, cv, cycle, scale, length, order, color, ph, func = each.replace(" ","").rstrip().rsplit(',')
                    cv_name = os.path.join(folder,cv)
                    self.plot_lib[int(scan)] = [cv_name,eval(cycle),eval(scale),eval(length), eval(order),color,eval(ph),func]
        if hasattr(self,'tableView_cv_setting'):
            folder = self.lineEdit_cv_folder.text()
            if folder=='':
                return
            for each in range(self.pandas_model_cv_setting._data.shape[0]):
                if self.pandas_model_cv_setting._data.iloc[each,0]:
                    scan, cv, cycle, scale, length, order, color, ph, func = self.pandas_model_cv_setting._data.iloc[each,1:].to_list()
                    cv_name = os.path.join(folder,cv)
                    self.plot_lib[int(scan)] = [cv_name,eval(cycle),eval(scale),eval(length), eval(order),color,eval(ph),func]


    #data format based on the output of IVIUM potentiostat
    def extract_ids_file(self,file_path,which_cycle=3):
        data = []
        current_cycle = 0
        with open(file_path,encoding="ISO-8859-1") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith('primary_data'):
                    # print(current_cycle)
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


    def plot_pot_step_current_from_external(self,ax,scan_no, plot_channel = 'potential'):
        file_name,which_cycle,cv_scale_factor, smooth_length, smooth_order, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.cv_tool.{}'.format(func_name))
        results = func(file_name, which_cycle, use_all = True)
        pot,current = results
        #pot_filtered, current_filtered = pot, current
        #pot_filtered = RHE(pot_filtered,pH=ph)
        # print(file_name,func_name,pot,current)
        #smooth the current due to beam-induced spikes
        #pot_filtered, current_filtered = self.cv_tool.filter_current(pot_filtered, current_filtered*cv_scale_factor, smooth_length, smooth_order)

        #ax.plot(pot_filtered,current_filtered*8,label='',color = color)
        if plot_channel=='current':
            ax.plot(range(len(current)),current*8,label='',color = color)
        elif plot_channel == 'potential':
            ax.plot(range(len(current)),RHE(pot,pH=ph),label='',color = color)

    def extract_cv_data(self):
        scan_no = int(self.comboBox_scans_3.currentText())
        file_name,which_cycle,cv_scale_factor, smooth_length, smooth_order, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.cv_tool.{}'.format(func_name))
        results = func(file_name, which_cycle)
        pot,current = results
        pot_filtered, current_filtered = pot, current
        pot_filtered = RHE(pot_filtered,pH=ph)
        # print(file_name,func_name,pot,current)
        #smooth the current due to beam-induced spikes
        pot_filtered, current_filtered = self.cv_tool.filter_current(pot_filtered, current_filtered*8, smooth_length, smooth_order)
        self.widget.set_data(pot_filtered, current_filtered)
        #return pot_filtered, current_filtered

    def plot_cv_from_external(self,ax,scan_no,marker_pos):
        file_name,which_cycle,cv_scale_factor, smooth_length, smooth_order, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.cv_tool.{}'.format(func_name))
        pot, current = [], []
        if type(which_cycle)==list:
            for each in which_cycle:
                _pot, _current = func(file_name, each)
                pot = pot + list(_pot)
                current = current + list(_current)
            pot, current = np.array(pot), np.array(current)
        elif type(which_cycle)==int:
            results = func(file_name, which_cycle)
            pot,current = results
        else:
            print('unsupported cycle index:', which_cycle)
        pot_filtered, current_filtered = pot, current
        pot_filtered = RHE(pot_filtered,pH=ph)
        # print(file_name,func_name,pot,current)
        #smooth the current due to beam-induced spikes
        pot_filtered, current_filtered = self.cv_tool.filter_current(pot_filtered, current_filtered*cv_scale_factor, smooth_length, smooth_order)
        '''
        if scan_no == 23999:
            self.ax_test = ax
            self.pot_test = pot_filtered
            self.current_test = current_filtered*8
        '''
        ax.plot(pot_filtered,current_filtered*8,label='',color = color)
        ax.plot(RHE(pot,pH=ph),current*8,ls=':',label='',color = color)
        #get the position to show the scaling text on the plot
        # current_temp = current_filtered[np.argmin(np.abs(pot_filtered[0:int(len(pot_filtered)/2)]-1.1))]*8*cv_scale_factor
        current_temp = 0
        ax.text(1.1,current_temp+1.5,'x{}'.format(cv_scale_factor),color=color)
        #store the cv data
        self.cv_info[scan_no] = {'current_density':current*8,'potential':RHE(pot,pH = ph),'pH':ph, 'color':color}
        if self.checkBox_show_marker.isChecked():
            for each in marker_pos:
                ax.plot([each,each],[-100,100],':k')
        #ax.set_ylim([min(current_filtered*8*cv_scale_factor),max(current*8)])
        print('scan{} based on cv'.format(scan_no))
        #scan rate in V/s
        scan_rate = float(self.lineEdit_scan_rate.text())
        #potential range in V_RHE
        pot_ranges = self.pot_ranges[scan_no]
        if scan_no not in self.charge_info:
            self.charge_info[scan_no] = {}
        for pot_range in pot_ranges:
            try:
                charge_cv, output = self.cv_tool.calculate_pseudocap_charge_stand_alone(pot_filtered, current_filtered/cv_scale_factor*8, scan_rate = scan_rate, pot_range = pot_range)
            except:
                charge_cv = 0
            if pot_range not in self.charge_info[scan_no]:
                self.charge_info[scan_no][pot_range] = {'skin_charge':0,'film_charge':0,'total_charge':charge_cv}
            else:
                self.charge_info[scan_no][pot_range]['total_charge'] = charge_cv

        return min(current_filtered*8),max(current*8)

    def plot_cv_from_external_original(self,ax,scan_no,marker_pos):
        file_name,which_cycle,cv_spike_cut,cv_scale_factor, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.{}'.format(func_name))
        results = func(file_name, which_cycle)
        if len(results) == 3:
            t, pot,current = results
            t_filtered, pot_filtered, current_filtered = t, pot, current
        elif len(results) == 2:
            pot,current = results
            pot_filtered, current_filtered = pot, current
        for ii in range(4):
            filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
            filter_index = filter_index+1#index offset by 1
            # t_filtered = t_filtered[(filter_index,)]
            pot_filtered = pot_filtered[(filter_index,)]
            current_filtered = current_filtered[(filter_index,)]
        pot_filtered = RHE(pot_filtered,pH=ph)
        ax.plot(pot_filtered,current_filtered*8*cv_scale_factor,label='',color = color)
        ax.plot(RHE(pot,pH=ph),current*8,label='',color = color)
        #get the position to show the scaling text on the plot
        # current_temp = current_filtered[np.argmin(np.abs(pot_filtered[0:int(len(pot_filtered)/2)]-1.1))]*8*cv_scale_factor
        current_temp = 0
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
        # pot_lf, pot_rt = pot_range
        # pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]]
        # values = np.array(self.data_to_plot[scan][label][data_range[0]:data_range[1]])+self.data_to_plot[scan][label+'_max']
        # values_smooth = signal.savgol_filter(values,41,2)
        # index_lf = np.argmin(np.abs(pots - pot_lf))
        # index_rt = np.argmin(np.abs(pots - pot_rt))
        pot_lf, pot_rt = pot_range
        if data_range[0]==data_range[1]:
            pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]+1]
        else:
            pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]]
        values = np.array(self.data_to_plot[scan][label])+self.data_to_plot[scan][label+'_max']
        values_smooth = signal.savgol_filter(values,41,2)
        index_lf = np.argmin(np.abs(pots - pot_lf)) + data_range[0]
        index_rt = np.argmin(np.abs(pots - pot_rt)) +data_range[0]
        return max([values_smooth[index_lf],values_smooth[index_rt]]), values_smooth[index_rt]-values_smooth[index_lf]

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
        header = ["scan", "pH", "pot_lf", "pot_rt", "q_skin", "q_film", "q_cv", "hor_size","d_hor_size","hor_size_err","ver_size","d_ver_size","ver_size_err","hor_strain","d_hor_strain","hor_strain_err","ver_strain","d_ver_strain","ver_strain_err","d_bulk_vol","d_bulk_vol_err","skin_vol_fraction","skin_vol_fraction_err","d_skin_avg", "d_skin_avg_err",'OER_E', 'OER_j']
        output_data = []
        for scan in self.scans:
            for pot_range in self.pot_ranges[scan]:
                which = self.pot_ranges[scan].index(pot_range)
                pot_range_value = 1
                if pot_range[0] != pot_range[1]:
                    pot_range_value = abs(pot_range[0]-pot_range[1])
                #scan = each_scan
                ph = self.grain_size_info_all_scans[scan][pot_range]['pH']
                charges = [round(self.charge_info[scan][pot_range][each],2) for each in ['skin_charge','film_charge','total_charge']]
                size_hor = [round(each,2) for each in list(self.grain_size_info_all_scans[scan][pot_range]["horizontal"])]
                size_hor_error = [round(self.data_summary[scan]['grain_size_ip'][which*2+1]*pot_range_value,4)]
                size_ver = [round(each, 2) for each in list(self.grain_size_info_all_scans[scan][pot_range]["vertical"])]
                size_ver_error = [round(self.data_summary[scan]['grain_size_oop'][which*2+1]*pot_range_value,4)]
                strain_hor = [round(each,4) for each in list(self.strain_info_all_scans[scan][pot_range]["horizontal"])]
                strain_hor_error = [round(self.data_summary[scan]['strain_ip'][which*2+1]*pot_range_value,4)]
                strain_ver = [round(each,4) for each in list(self.strain_info_all_scans[scan][pot_range]["vertical"])]
                strain_ver_error = [round(self.data_summary[scan]['strain_oop'][which*2+1]*pot_range_value,4)]
                d_bulk_vol = [2*abs(strain_hor[1])+abs(strain_ver[1])]
                d_bulk_vol_error = [round((4*strain_hor_error[0]**2 + strain_ver_error[0]**2)**0.5,4)]
                skin_vol_fraction = [round(abs(size_ver[1]/size_ver[0] + 2*size_hor[1]/size_hor[0])*100,3)]
                skin_vol_fraction_error = [round(((size_ver_error[0]/size_ver[0])**2 + 4 * (size_hor_error[0]/size_hor[0])**2)**0.5*100,4)]
                d_skin_avg = [round(abs(size_ver[1] + 2* size_ver[0]/size_hor[0] * size_hor[1]), 4)] #refer to ACS catalysis paper https://doi.org/10.1021/acscatal.1c05169 SI (Section 2)
                d_skin_avg_error = [round((size_ver_error[0]**2 + 4 * (size_ver[0]/size_hor[0])**2*size_hor_error[0]**2)**0.5,4)]
                try:
                    idx_OER_E = sorted(list(np.argpartition(abs(self.cv_info[scan]['potential']-float(self.lineEdit_OER_E.text())), 18)[0:18]))[0]
                    idx_OER_j = sorted(list(np.argpartition(abs(self.cv_info[scan]['current_density']-float(self.lineEdit_OER_j.text())), 18)[0:18]))[0]
                except:
                    idx_OER_E = sorted(list(np.argpartition(abs(self.data_to_plot[scan]['potential']-float(self.lineEdit_OER_E.text())), 18)[0:18]))[0]
                    idx_OER_j = sorted(list(np.argpartition(abs(self.data_to_plot[scan]['current']*8-float(self.lineEdit_OER_j.text())), 18)[0:18]))[0]
                OER_E = [round(self.cv_info[scan]['potential'][idx_OER_j],4)]
                OER_j = [round(self.cv_info[scan]['current_density'][idx_OER_E],4)]
                data_temp = [scan, ph] +[round(each,3) for each in list(pot_range)]+ charges + size_hor + size_hor_error + size_ver + size_ver_error + strain_hor + strain_hor_error + strain_ver + strain_ver_error + d_bulk_vol + d_bulk_vol_error + skin_vol_fraction + skin_vol_fraction_error + d_skin_avg + d_skin_avg_error + OER_E + OER_j
                output_data.append(data_temp)
        self.summary_data_df = pd.DataFrame(np.array(output_data),columns = header)
        self.widget_terminal.update_name_space('charge_info',self.charge_info)
        self.widget_terminal.update_name_space('size_info',self.grain_size_info_all_scans)
        self.widget_terminal.update_name_space('strain_info',self.strain_info_all_scans)
        self.widget_terminal.update_name_space('main_win',self)
        self.widget_terminal.update_name_space('cv_info', self.cv_info)
        self.widget_terminal.update_name_space('summary_data', self.summary_data_df)

        def _tag_p(text):
            return '<p>{}</p>'.format(text)
        output_text = []
        output_text.append("*********Notes*********")
        output_text.append("*scan: scan number")
        output_text.append("*pot_lf (V_RHE): left boundary of potential range considered ")
        output_text.append("*pot_rt (V_RHE): right boundary of potential range considered ")
        output_text.append("*q_skin(mc/m2): charge calculated based on skin layer thickness")
        output_text.append("*q_film(mc/m2): charge calculated assuming all Co2+ in the film material has been oxidized to Co3+")
        output_text.append("*q_cv(mc/m2): charge calculated from electrochemistry data (CV data)")
        output_text.append("*(d)_hor/ver_size(nm): horizontal/vertical size or the associated change with a d_ prefix")
        output_text.append("*hor/ver_size_err(nm): error for horizontal/vertical size")
        output_text.append("*(d)_hor/ver_strain(%): horizontal/vertical strain or the associated change with a d_ prefix")
        output_text.append("*hor/ver_strain_err(%): error for horizontal/vertical strain")
        output_text.append("*d_bulk_vol (%): The change of bulk vol wrt the total film volume: 2*d_hor_strain + d_ver_strain")
        output_text.append("*d_bulk_vol_err: error of d_bulk_vol")
        output_text.append("*skin_vol_fraction (%): The skin volume fraction wrt the total film volume")
        output_text.append("*skin_vol_fraction_err: error of skin_vol_fraction")
        output_text.append("*d_skin_avg (nm): The average thickness of the skin layer normalized to surface area of the crystal")
        output_text.append("*d_skin_avg_err: the error of d_skin_avg")
        output_text.append(f"*OER_E: The OER potential at j(mA/cm2) = {float(self.lineEdit_OER_j.text())}")
        output_text.append(f"*OER_j: The OER current at E (RHE/V) = {float(self.lineEdit_OER_E.text())}")
        self.output_text = output_text
        for i in range(len(output_text)):
            output_text[i] = _tag_p(output_text[i])

        self.plainTextEdit_summary.setHtml(self.summary_data_df.to_html(index = False)+''.join(output_text))
        #print("\n".join(output_text))
        #self.plainTextEdit_summary.setPlainText("\n".join(output_text))

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
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CV config Files (*.ini);;zip Files (*.zip)", options=options)
        if fileName:
            self.lineEdit_cv_config_path.setText(fileName)
            self.widget_par_tree.update_parameter(fileName)
            '''
            with open(fileName,'r') as f:
                lines = f.readlines()
                self.plainTextEdit_cv_config.setPlainText(''.join(lines))
            '''

    #update the config file after edition in the plainText block
    def update_cv_config_file(self):
        # with open(self.lineEdit_cv_config_path.text(),'w') as f:
            # f.write(self.plainTextEdit_cv_config.toPlainText())
        self.widget_par_tree.save_parameter(self.lineEdit_cv_config_path.text())
        if self.lineEdit_cv_config_path.text().endswith('.ini'):
            missed_items = self.cv_tool._extract_parameter_from_config(self.lineEdit_cv_config_path.text())
        elif self.lineEdit_cv_config_path.text().endswith('.zip'): 
            missed_items = self.cv_tool._extract_parameter_from_config(self.lineEdit_cv_config_path.text().replace('.zip','.ini'))
        if len(missed_items)==0:
            self.cv_tool.extract_cv_info()
            error_pop_up('The config file is overwritten!','Information')
        else:
            error_pop_up(f'The config file is overwritten, but the config file has the following items missed:{missed_items}!','Error')

    #load config file
    def load_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","zip Files (*.zip);;config Files (*.ini)", options=options)
        if fileName.endswith('.ini'):
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
                            if channel == 'plainTextEdit_img_range':
                                getattr(self,channel).setPlainText(value.replace(";","\n"))
                                if value=="":
                                    pass
                                else:
                                    self.image_range_info = {}
                                    items = value.rsplit(';')
                                    for each_item in items:
                                        a,b = each_item.rstrip().rsplit(":")
                                        self.image_range_info[int(a)] = eval(b)
                            elif channel == 'plainTextEdit_tick_label_settings':
                                getattr(self,channel).setPlainText(value.replace(";","\n"))
                            elif channel == 'tableView_ax_format':
                                value=eval(value)
                                cols = self.pandas_model_in_ax_format._data.columns.tolist()
                                data_shape = self.pandas_model_in_ax_format._data.shape
                                for i in range(len(cols)):
                                    #print(i,cols[i], value['use'])
                                    for j in value[cols[i]]:
                                        if j<data_shape[0] and i<data_shape[1]:
                                            self.pandas_model_in_ax_format._data.iloc[j,i] = value[cols[i]][j]
                                #self.pandas_model_in_ax_format._data = pd.DataFrame(eval(value))
                            elif channel == 'tableView_cv_setting':
                                self.update_pandas_model_cv_setting(reset = True, data = eval(value))
        elif fileName.endswith('.zip'):
            self.load_config_raw(zipfile.ZipFile(fileName,'r'))

        self._load_file()
        self.append_scans_xrv()
        self.update_pot_offset()
        self.make_plot_lib()

    #zipfile is of format zipfile.ZipFile('','r')
    #this step will be done first before lauching the plot func
    def _save_temp_cv_excel_file(self, zipfile):
        root_folder = os.path.join(DaFy_path,'dump_files')
        #pandas DataFrame
        xrv_data = pickle.loads(zipfile.read('xrv_data'))
        cv_data_list = pickle.loads(zipfile.read('cv_data_raw'))
        cv_data_names = pickle.loads(zipfile.read('cv_data_names'))
        xrv_data.to_excel(os.path.join(root_folder,zipfile.read('xrv_data_file_name').decode()))
        for i in range(len(cv_data_list)):
            #str format already
            cv_data = cv_data_list[i]
            _, cv_name = os.path.split(cv_data_names[i])
            with open(os.path.join(root_folder,cv_name), 'w') as f:
                f.write(cv_data)     

    def load_config_raw(self, zipfile):
        print('save temp cv and xrv excel file...')
        self._save_temp_cv_excel_file(zipfile)
        print('files being saved and set meta parameters now ...')
        #lineedit or checkbox channels
        #channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_use','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
        #            'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
        #            'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file','lineEdit_reference_potential',\
        #            'checkBox_show_marker','checkBox_merge']
        more_channels = ['plainTextEdit_img_range', 'tableView_ax_format', 'tableView_cv_setting']
        
        for channel in self.GUI_metaparameter_channels:
            if channel.startswith('lineEdit'):
                if channel == 'lineEdit_cv_folder':
                    getattr(self, channel).setText(os.path.join(DaFy_path,'dump_files'))
                elif channel == 'lineEdit_data_file':
                    filename = zipfile.read('xrv_data_file_name').decode()
                    getattr(self, channel).setText(os.path.join(DaFy_path,'dump_files',filename))
                else:
                    try:
                        getattr(self, channel).setText(zipfile.read(channel).decode())
                    except:
                        pass
            elif channel.startswith('checkBox'):
                try:
                    getattr(self, channel).setChecked(eval(zipfile.read(channel).decode()))
                except:
                    pass
            else:#scan_numbers_append
                getattr(self, channel).setText(zipfile.read(channel).decode())
        for channel in more_channels:
            value = zipfile.read(channel).decode()
            if channel == 'plainTextEdit_img_range':
                getattr(self,channel).setPlainText(value.replace(";","\n"))
                if value=="":
                    pass
                else:
                    self.image_range_info = {}
                    items = value.rsplit(';')
                    for each_item in items:
                        a,b = each_item.rstrip().rsplit(":")
                        self.image_range_info[int(a)] = eval(b)
            elif channel == 'tableView_ax_format':
                value=eval(value)
                cols = self.pandas_model_in_ax_format._data.columns.tolist()
                data_shape = self.pandas_model_in_ax_format._data.shape
                for i in range(len(cols)):
                    #print(i,cols[i], value['use'])
                    for j in value[cols[i]]:
                        if j<data_shape[0] and i<data_shape[1]:
                            self.pandas_model_in_ax_format._data.iloc[j,i] = value[cols[i]][j]
                #self.pandas_model_in_ax_format._data = pd.DataFrame(eval(value))
            elif channel == 'tableView_cv_setting':
                self.update_pandas_model_cv_setting(reset = True, data = eval(value))
        zipfile.close()
        print('everything for setup is finished now!')

    #save config file
    def save_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","zip File (*.zip);;config file (*.ini)", options=options)
        if fileName.endswith('.zip'):
            self.save_config_raw(fileName)
            return
        with open(fileName,'w') as f:
            #channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_use','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
            #            'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
            #            'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file','lineEdit_reference_potential',\
            #            'checkBox_show_marker','checkBox_merge']
            for channel in self.GUI_metaparameter_channels:
                try:
                    f.write(channel+':'+str(getattr(self,channel).isChecked())+'\n')
                except:
                    f.write(channel+':'+getattr(self,channel).text()+'\n')
            f.write("plainTextEdit_img_range:"+self.plainTextEdit_img_range.toPlainText().replace("\n",";")+'\n')

            #f.write("textEdit_plot_lib:"+self.textEdit_plot_lib.toPlainText().replace("\n",";")+'\n')
            if hasattr(self,'plainTextEdit_tick_label_settings'):
                f.write("plainTextEdit_tick_label_settings:"+self.plainTextEdit_tick_label_settings.toPlainText().replace("\n",";")+'\n')
            if hasattr(self, 'tableView_ax_format'):
                f.write("tableView_ax_format:"+str(self.pandas_model_in_ax_format._data.to_dict())+'\n')
            if hasattr(self,'tableView_cv_setting'):
                f.write("tableView_cv_setting:"+str(self.pandas_model_cv_setting._data.to_dict())+'\n')

    #save all meta-parameters and data files (xrv data and cv data) into a zip file
    def save_config_raw(self, filename):
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        #filename, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","config file (*.zip)", options=options)
        if filename:
            try:
                savefile = zipfile.ZipFile(filename, 'w')
            except Exception as e:
                raise IOError(str(e), filename)
            #channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_use','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
            #            'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
            #            'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file','lineEdit_reference_potential',\
            #            'checkBox_show_marker','checkBox_merge']
            for each in self.GUI_metaparameter_channels:
                if each.startswith('checkBox'):
                    savefile.writestr(each, str(getattr(self,each).isChecked()))
                else:
                    savefile.writestr(each, getattr(self,each).text())
            savefile.writestr("plainTextEdit_img_range", self.plainTextEdit_img_range.toPlainText().replace("\n",";"))
            # savefile.writestr("plainTextEdit_tick_label_settings", self.plainTextEdit_tick_label_settings.toPlainText().replace("\n",";"))
            savefile.writestr("tableView_ax_format", str(self.pandas_model_in_ax_format._data.to_dict()))
            savefile.writestr("tableView_cv_setting",str(self.pandas_model_cv_setting._data.to_dict()))
            savefile.writestr('xrv_data', pickle.dumps(self.data))
            savefile.writestr('xrv_data_file_name', os.path.split(self.lineEdit_data_file.text())[1])
            savefile.writestr('cv_data_raw', pickle.dumps([open(self.plot_lib[each][0],'r').read() for each in self.plot_lib]))
            savefile.writestr('cv_data_names', pickle.dumps([self.plot_lib[each][0] for each in self.plot_lib]))
            savefile.close()

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
        self.comboBox_scans_2.clear()
        self.comboBox_scans_2.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_3.clear()
        self.comboBox_scans_3.addItems([str(each) for each in sorted(scans)])
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
                        'strain_ip':r'$\partial\mid\Delta\varepsilon_\parallel\mid\slash\partial E$ (%/V)',
                        'strain_oop':r'$\partial\mid\Delta\varepsilon_\perp\mid\slash\partial E$ (%/V)',
                        'grain_size_oop':r'$\partial\mid\Delta d_\perp\mid\slash\partial E$ (nm/V)',
                        'grain_size_ip':r'$\partial\mid\Delta d_\parallel\mid\slash\partial E$ (nm/V)',
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
                    '''
                    plot_data_y_charge = []
                    for each_scan in self.scans:
                        key_ = list(self.charge_info[each_scan].keys())[-1]
                        plot_data_y_charge.append(self.charge_info[each_scan][key_]['total_charge']/abs(key_[0]-key_[1]))
                    '''
                    labels = ['pH {}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    count_pH13 = 1
                    for j,each_label in enumerate(labels):
                        if each_label == 'pH 13':
                            labels[j] = '{} ({})'.format(each_label,count_pH13)
                            count_pH13 += 1
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), int(len(self.data_summary[self.scans[0]]['strain_ip'])/2), i+1+int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)*plot_y_labels.index(each))
                    # ax_temp_2 = ax_temp.twinx()
                    if i==0 and each == plot_y_labels[0]:
                        #ax_temp.legend(labels)
                        for ii in range(len(plot_data_x)):
                            if labels[ii] in ['pH 13 (1)', 'pH 8', 'pH 7', 'pH 10']:
                                label = labels[ii]
                                if label == 'pH 13 (1)':
                                    label = 'pH 13'
                                ax_temp.bar(plot_data_x[ii],-plot_data_y[ii,0],0.5, yerr = plot_data_y[ii,-1], color = colors_bar[ii], label = label)
                            else:
                                ax_temp.bar(plot_data_x[ii],-plot_data_y[ii,0],0.5, yerr = plot_data_y[ii,-1], color = colors_bar[ii])
                        ax_temp.legend(loc = 2,ncol = 1)
                    else:
                        ax_temp.bar(plot_data_x,-plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                    # ax_temp_2.plot(plot_data_x,-np.array(plot_data_y_charge),'k:*')
                    self._format_axis(ax_temp)
                    # self._format_axis(ax_temp_2)
                    if 'bar' in self.tick_label_settings:
                        if (each in self.tick_label_settings['bar']) and self.checkBox_use.isChecked():
                            self._format_ax_tick_labels(ax = ax_temp,
                                    fun_set_bounds = self.tick_label_settings['bar'][each]['func'],
                                    bounds = [0,abs(lim_y_temp[each])], #[lim_y_temp[each],0],
                                    bound_padding = self.tick_label_settings['bar'][each]['padding'],
                                    major_tick_location =self.tick_label_settings['bar'][each]['locator'],
                                    show_major_tick_label = i==0, #show major tick label for the first scan
                                    num_of_minor_tick_marks=self.tick_label_settings['bar'][each]['tick_num'],
                                    fmt_str = self.tick_label_settings['bar'][each]['fmt'])
                    if i == 0:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=10)
                        ax_temp.set_ylim([0,abs(lim_y_temp[each])])

                    else:
                        ax_temp.set_ylim([0,abs(lim_y_temp[each])])
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
                        'peak_intensity':r'Intensity / a.u.',
                        '<dskin>': r'$d_{skin}$ / nm',
                        'dV_bulk':r'($\Delta V / V$) / %',
                        'dV_skin':r'($\Delta V_{skin} / V$) / %',
                        'OER_E': r'$\eta (1 mAcm^{-2}) / V$',
                        'OER_j':r'$j (1.65 V) / mAcm^{-2})$',
                        'q_cv':r'$Q_0\hspace{1} /\hspace{1} mC{\bullet}cm^{-2}$',
                        'q_film': r'$Q_0\hspace{1} /\hspace{1} mC{\bullet}cm^{-2}$',
                        'OER_j/<dskin>':r'$log(j/V_{skin})\hspace{1}/\hspace{1}mA{\bullet}nm^{-3}$'
                        }

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
                #force using the absolute value
                # use_absolute_value = True
                for each in lim_y_temp.keys():
                    for each_scan in self.scans:
                        if use_absolute_value:
                            lim_y_temp[each].append(self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2])
                        else:
                            lim_y_temp[each].append(self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2])
            for each in lim_y_temp:
                lim_y_temp[each] = [min(lim_y_temp[each]),max(lim_y_temp[each])]
            for each in lim_y_temp:
                offset = (lim_y_temp[each][1]-lim_y_temp[each][0])*0.1
                lim_y_temp[each] = [lim_y_temp[each][0]-offset,lim_y_temp[each][1]+offset]
            if use_absolute_value:
               y_label_map = y_label_map_abs
            gs_left = plt.GridSpec(len(plot_y_labels), len(self.pot_range)+1,hspace=0.02,wspace=0.2)
            hwspace = eval(self.lineEdit_hwspace.text())
            gs_right = plt.GridSpec(max([2,self.comboBox_link_container.count()]), len(self.pot_range)+1,hspace=hwspace[0],wspace=hwspace[1])
            #print(self.data_summary)
            def _extract_setting(channel):
                data = self.pandas_model_in_ax_format._data
                return data[(data['type'] == 'bar') & (data['channel'] == channel)].to_dict(orient = 'records')[0]

            for each_pot in self.pot_range:
                output_data = []
                #use_absolute_value = each_pot[0] == each_pot[1]
                # use_absolute_value = True
                for each in plot_y_labels:
                    plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2],self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    labels = ['{}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    # ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), len(self.pot_range)+1, self.pot_range.index(each_pot)+1+(len(self.pot_range)+1)*plot_y_labels.index(each))
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(gs_left[plot_y_labels.index(each), self.pot_range.index(each_pot)])
                    self._format_axis(ax_temp)
                    settings = _extract_setting(each)
                    if settings['use'] and self.checkBox_use.isChecked():
                        self._format_ax_tick_labels(ax = ax_temp,
                                                    fun_set_bounds = settings['func'],#'set_xlim',
                                                    bounds = [0,1],#will be replaced
                                                    bound_padding = float(settings['padding']),
                                                    major_tick_location = eval(settings['tick_locs']), #x_locator
                                                    show_major_tick_label = True, #show major tick label for the first scan
                                                    num_of_minor_tick_marks=int(settings['#minor_tick']), #4
                                                    fmt_str = settings['fmt_str'])#'{:3.1f}'
                    if use_absolute_value:
                        ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                        ax_temp.plot(plot_data_x,plot_data_y[:,0], '*:',color='0.1')
                        output_data.append(plot_data_y[:,0])
                    else:
                        ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                        ax_temp.plot(plot_data_x,plot_data_y[:,0], '*:',color='0.1')
                        output_data.append(plot_data_y[:,0])
                    if each_pot == self.pot_range[0]:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=10)
                        # ax_temp.set_ylim([lim_y_temp[each],0])
                        # ax_temp.set_ylim(lim_y_temp[each])
                    else:
                        pass
                        # ax_temp.set_ylim(lim_y_temp[each])
                    if each == plot_y_labels[0]:
                        if use_absolute_value:
                            ax_temp.set_title('E = {:4.2f} V'.format(each_pot[0]), fontsize=10)
                        else:
                            ax_temp.set_title('E range:{:4.2f}-->{:4.2f} V'.format(*each_pot), fontsize=10)
                    if each != plot_y_labels[-1]:
                        #ax_temp.set_xticklabels([])
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=10)
                    else:
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=10)
                        ax_temp.set_xlabel('pH')
                    if each_pot!=self.pot_range[0]:
                        ax_temp.set_yticklabels([])
                def _extract_data(channel, which_pot_range):
                    data_len = self.summary_data_df.shape[0]+1
                    name_map = {'dV_bulk':lambda:self.summary_data_df['d_bulk_vol'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'dV_skin':lambda:self.summary_data_df['skin_vol_fraction'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                '<dskin>':lambda:self.summary_data_df['d_skin_avg'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_E':lambda:self.summary_data_df['OER_E'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_j':lambda:self.summary_data_df['OER_j'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_j/<dskin>':lambda:(self.summary_data_df['OER_j']/self.summary_data_df['d_skin_avg']).to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'pH':lambda:self.summary_data_df['pH'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'q_film':lambda:self.summary_data_df['q_film'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'q_cv':lambda:self.summary_data_df['q_cv'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'input':lambda:eval(self.lineEdit_input_values.text())}
                    name_map_error = {'dV_bulk':lambda:self.summary_data_df['d_bulk_vol_err'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'dV_skin':lambda:self.summary_data_df['skin_vol_fraction_err'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                '<dskin>':lambda:self.summary_data_df['d_skin_avg_err'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_E':lambda:[None]*self.summary_data_df.shape[0],
                                'OER_j':lambda:[None]*self.summary_data_df.shape[0],
                                'OER_j/<dskin>':lambda:(self.summary_data_df['OER_j']/self.summary_data_df['d_skin_avg']**2*self.summary_data_df['d_skin_avg_err']).to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'pH':lambda:[None]*self.summary_data_df.shape[0],
                                'q_film':lambda:[None]*self.summary_data_df.shape[0],
                                'q_cv':lambda:[None]*self.summary_data_df.shape[0],
                                'input':lambda:[None]*self.summary_data_df.shape[0]}
                    if which_pot_range>len(self.pot_range)-1:
                        which_pot_range = 0
                    if self.checkBox_error.isChecked():
                        return name_map[channel](), name_map_error[channel]()
                    else:
                        return name_map[channel](), [None]*self.summary_data_df.shape[0]
                    #return [self.data_summary[each_][channel][which_pot_range*2] for each_ in self.scans]

                def _get_xy_for_linear_fit(panel_index, x, y):
                    tag = getattr(self,f'lineEdit_partial_set_p{panel_index+1}').text()
                    x_, y_ = [], []
                    if tag == '[*]': #use all
                        return x, y
                    else:
                        if tag.startswith('-'):
                            tag = [each for each in range(len(x)) if each not in eval(tag[1:])]
                        else:
                            tag = eval(tag)
                        if type(tag)!=list:
                            return x, y
                        else:
                            for each in tag:
                                x_.append(x[each])
                                y_.append(y[each])
                            return x_, y_

                for i in range(self.comboBox_link_container.count()):
                    channels = self.comboBox_link_container.itemText(i).rsplit('+')
                    x, x_error = _extract_data(channels[0],self.spinBox_pot_range_idx.value())
                    y, y_error = _extract_data(channels[1],self.spinBox_pot_range_idx.value())
                    x_, y_ = _get_xy_for_linear_fit(i, x, y)
                    # ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), len(self.pot_range)+1, 2+(len(self.pot_range)+1)*i)
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(gs_right[i,len(self.pot_range)])
                    self._format_axis(ax_temp)
                    if channels[0] in y_label_map:
                        ax_temp.set_xlabel(y_label_map[channels[0]])
                    else:
                        ax_temp.set_xlabel(channels[0])
                    if channels[1] in y_label_map:
                        ax_temp.set_ylabel(y_label_map[channels[1]])
                    else:
                        ax_temp.set_ylabel(channels[1])
                    if channels[0]=='input':
                        ax_temp.set_xlabel(self.lineEdit_input_name.text())
                    if channels[1]=='input':
                        ax_temp.set_ylabel(self.lineEdit_input_name.text())
                    if 'OER_j/<dskin>' == channels[1]:
                        #-14 term is due to the unit of cm-2 transformed to nm-2, the scalling factor will be 10-14 becoming -14 after applying the log
                        slope_, intercept_, r_value_, *_ = stats.linregress(x_, np.log10(y_)-14)
                        # ax_temp.set_ylabel('log({})'.format(channels[1]))
                        ax_temp.set_ylabel(y_label_map[channels[1]])
                        # [ax_temp.scatter(x[jj], np.log10(y[jj]), c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        for jj in range(len(x)):
                            if y_error[jj] == None:
                                ax_temp.errorbar(x[jj], np.log10(y[jj])-14, xerr=x_error[jj], yerr =None, c=colors_bar[jj], marker = 's', ms = 4)
                            else:
                                ax_temp.errorbar(x[jj], np.log10(y[jj])-14, xerr=x_error[jj], yerr = y_error[jj]/y[jj]/np.log(10), c=colors_bar[jj], marker = 's', ms = 4)
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(x[jj], np.log10(y[jj])-14, str(jj), c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(x, np.array(x)*slope_ + intercept_, ':k')
                    elif 'OER_j/<dskin>' == channels[0]:
                        slope_, intercept_, r_value_, *_ = stats.linregress(np.log10(x_)-14, y_)
                        # ax_temp.set_xlabel('log({})'.format(channels[0]))
                        ax_temp.set_xlabel(y_label_map[channels[0]])
                        # [ax_temp.scatter(np.log10(x[jj]), y[jj], c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        for jj in range(len(x)):
                            if x_error[jj] == None:
                                ax_temp.errorbar(np.log10(x[jj])-14, y[jj], xerr=None, yerr=y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                            else:
                                ax_temp.errorbar(np.log10(x[jj])-14, y[jj], xerr=x_error[jj]/x[jj]/np.log(10), yerr=y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                            [ax_temp.text(np.log10(x[jj]), y[jj], str(jj), c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(np.log10(x), np.log10(x)*slope_ + intercept_, ':k')
                    else:
                        slope_, intercept_, r_value_, *_ = stats.linregress(x_, y_)
                        # [ax_temp.scatter(x[jj], y[jj], c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        [ax_temp.errorbar(x[jj], y[jj], xerr = x_error[jj], yerr = y_error[jj], c=colors_bar[jj], marker = 's', ms = 4) for jj in range(len(x))]
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(x[jj], y[jj], str(jj), c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(x, np.array(x)*slope_ + intercept_, ':k')
                    for channel in channels:
                        settings = _extract_setting(channel)
                        if settings['use'] and self.checkBox_use.isChecked():
                            self._format_ax_tick_labels(ax = ax_temp,
                                                        fun_set_bounds = settings['func'],#'set_xlim',
                                                        bounds = [0,1],#will be replaced
                                                        bound_padding = float(settings['padding']),
                                                        major_tick_location = eval(settings['tick_locs']), #x_locator
                                                        show_major_tick_label = True, #show major tick label for the first scan
                                                        num_of_minor_tick_marks=int(settings['#minor_tick']), #4
                                                        fmt_str = settings['fmt_str'])#'{:3.1f}'

                #print output data
                output_data = np.array(output_data).T
                output_data = np.append(output_data,np.array([int(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans])[:,np.newaxis],axis=1)
                output_data = np.append(np.array([int(each_) for each_ in self.scans])[:,np.newaxis],output_data,axis = 1)
                # print('\n')
                # print(each_pot)
                plain_text.append(f'<p>\npot = {each_pot} V</p>')
                plain_text.append('<p>scan_no\tstrain_ip\tstrain_oop\tgrain_size_ip\tgrain_size_oop\tpH</p>')
                for each_row in output_data:
                    # print("{:3.0f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:2.0f}".format(*each_row))
                    plain_text.append("<p>{:3.0f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t\t{:2.0f}</p>".format(*each_row))
            #self.mplwidget2.fig.subplots_adjust(hspace=0.5, wspace=0.2)
            self.mplwidget2.canvas.draw()
            # self.plainTextEdit_summary.setPlainText('\n'.join(plain_text))
            self.plainTextEdit_summary.setHtml('<h3>Table of complete information of pseudocapacitive charge and film structure (results extracted from master figure)</h3>'\
                                               +self.summary_data_df.to_html(index = False)+''.join(self.output_text)
                                               +'<br><h3>structural change normalized to potential (delta/V) (data used for plotting bar chart)</h3>'+''.join(plain_text))
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
            #NOTE: this could be buggy, the text inside the lineEdit has to be like this "1.2,1.5"
            pot_range_specified = eval("({})".format(self.lineEdit_pot_range.text().rstrip()))
            if p1>pot_range_specified[1]:
                p1 = sum(pot_range_specified)/2
            pot_range1 = (pot_range_specified[0], p1)
            pot_range2 = (p1, pot_range_specified[1])
            pot_range3 = pot_range_specified
            self.pot_ranges[scan] = [f(pot_range1),f(pot_range2),f(pot_range3)]

    def plot_reaction_order_and_tafel(self,axs = []):
        if len(axs)== 0:
            self.widget_cv_view.canvas.figure.clear()
            ax_tafel = self.widget_cv_view.canvas.figure.add_subplot(1,2,1)
            ax_order = self.widget_cv_view.canvas.figure.add_subplot(1,2,2)
        else:
            assert len(axs) == 2, 'You need only two axis handle here!'
            ax_tafel, ax_order = axs
        #self.ax_tafel = ax_tafel
        #self.ax_order = ax_order
        if self.cv_tool.info['reaction_order_mode'] == 'constant_potential':
            constant_value = self.cv_tool.info['potential_reaction_order']
        elif self.cv_tool.info['reaction_order_mode'] == 'constant_current':
            constant_value = self.cv_tool.info['current_reaction_order']
        mode = self.cv_tool.info['reaction_order_mode']
        forward_cycle = True
        text_log_tafel = self.cv_tool.plot_tafel_with_reaction_order(ax_tafel, ax_order,constant_value = constant_value,mode = mode, forward_cycle = forward_cycle, use_marker = self.checkBox_use_marker.isChecked(), use_all = self.checkBox_use_all.isChecked())
        plainText = '\n'.join([text_log_tafel[each] for each in text_log_tafel])
        self.plainTextEdit_cv_summary.setPlainText(self.plainTextEdit_cv_summary.toPlainText() + '\n\n' + plainText)

        self._format_axis(ax_tafel)
        self._format_axis(ax_order)
        #set item tag e) and f) for tafel and order
        # ax_tafel.text(1.5, 4.4, 'e)',weight = 'bold', fontsize = 12)
        # ax_order.text(4.8, 1.81, 'f)',weight = 'bold', fontsize = 12)

        tafel_bounds_pot, tick_locs_tafel_pot, padding_tafel_pot, num_tick_marks_tafel_pot, fmt_tafel_pot, func_tafel_pot = self.cv_tool.info['tafel_bounds_pot'].rsplit('+')
        tafel_bounds_current, tick_locs_tafel_current, padding_tafel_current, num_tick_marks_tafel_current, fmt_tafel_current, func_tafel_current = self.cv_tool.info['tafel_bounds_current'].rsplit('+')
        self._format_ax_tick_labels(ax = ax_tafel,
                fun_set_bounds = func_tafel_pot,
                bounds = eval(tafel_bounds_pot),
                bound_padding = float(padding_tafel_pot),
                major_tick_location =eval(tick_locs_tafel_pot),
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks= int(num_tick_marks_tafel_pot),
                fmt_str = fmt_tafel_pot)
        self._format_ax_tick_labels(ax = ax_tafel,
                fun_set_bounds = func_tafel_current,
                bounds = eval(tafel_bounds_current),
                bound_padding = float(padding_tafel_current),
                major_tick_location =eval(tick_locs_tafel_current),
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks= int(num_tick_marks_tafel_current),
                fmt_str = fmt_tafel_current)

        order_bounds_ph, tick_locs_order_ph, padding_order_ph, num_tick_marks_order_ph, fmt_order_ph, func_order_ph = self.cv_tool.info['order_bounds_ph'].rsplit('+')
        order_bounds_y, tick_locs_order_y, padding_order_y, num_tick_marks_order_y, fmt_order_y, func_order_y = self.cv_tool.info['order_bounds_y'].rsplit('+')
        self._format_ax_tick_labels(ax = ax_order,
                                    fun_set_bounds = func_order_ph,
                                    bounds = eval(order_bounds_ph),
                                    bound_padding = eval(padding_order_ph),
                                    major_tick_location =eval(tick_locs_order_ph),
                                    show_major_tick_label = True, #show major tick label for the first scan
                                    num_of_minor_tick_marks= int(num_tick_marks_order_ph),
                                    fmt_str = fmt_order_ph)
        self._format_ax_tick_labels(ax = ax_order,
                                    fun_set_bounds = func_order_y,
                                    bounds = eval(order_bounds_y),
                                    bound_padding = eval(padding_order_y),
                                    major_tick_location =eval(tick_locs_order_y),
                                    show_major_tick_label = True, #show major tick label for the first scan
                                    num_of_minor_tick_marks= int(num_tick_marks_order_y),
                                    fmt_str = fmt_order_y)

        coord_top_left = [eval(tafel_bounds_pot)[0]-float(padding_tafel_pot),eval(tafel_bounds_current)[1]+float(padding_tafel_current)]
        offset = np.array(self.cv_tool.info['index_header_pos_offset_tafel'])
        coord_top_index_marker = coord_top_left+offset
        label_map = dict(zip(range(15),list('abcdefghijklmno')))
        cvs_total_num = len([self.cv_tool.info['selected_scan'],self.cv_tool.info['sequence_id']][int(self.checkBox_use_all.isChecked())])
        ax_tafel.text(*coord_top_index_marker, '{})'.format(label_map[cvs_total_num]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))

        coord_top_left = [eval(order_bounds_ph)[0]-float(padding_order_ph),eval(order_bounds_y)[1]+float(padding_order_y)]
        offset = np.array(self.cv_tool.info['index_header_pos_offset_order'])
        coord_top_index_marker = coord_top_left+offset
        label_map = dict(zip(range(15),list('abcdefghijklmno')))
        #cvs_total_num = len([self.cv_tool.info['selected_scan'],self.cv_tool.info['sequence_id']][int(self.checkBox_use_all.isChecked())])
        ax_order.text(*coord_top_index_marker, '{})'.format(label_map[cvs_total_num+1]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))
        for each in [ax_tafel,ax_order]:
            for tick in each.xaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))
            for tick in each.yaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))

        #move labels to right side of the plot

        '''
        ax_tafel.yaxis.set_label_position("right")
        ax_tafel.yaxis.tick_right()
        ax_order.yaxis.set_label_position("right")
        ax_order.yaxis.tick_right()
        '''
        self.widget_cv_view.canvas.draw()

    def plot_cv_data(self):
        self.widget_cv_view.canvas.figure.clear()
        '''
        if self.checkBox_default.isChecked():
            col_num = 2
            row_num = len(self.cv_tool.cv_info)
        else:
            col_num = max([2,self.spinBox_cols.value()])
            row_num = max([len(self.cv_tool.cv_info),self.spinBox_rows.value()])
        '''
        if self.checkBox_default.isChecked():
            col_num = 2
            row_num = max([3,len(self.cv_tool.cv_info)])
        else:
            col_num = max([2,int(self.widget_par_tree.par[('Figure_Layout_settings','total_columns')])])
            row_num = max([len(self.cv_tool.cv_info),int(self.widget_par_tree.par[('Figure_Layout_settings','total_rows')])])
        
        if not self.checkBox_use_all.isChecked():
            row_num = max([3,len(self.cv_tool.info['selected_scan'])])
        gs_left = plt.GridSpec(row_num,col_num,hspace=self.cv_tool.info['hspace'][0],wspace=self.cv_tool.info['wspace'][0])
        gs_right = plt.GridSpec(row_num,col_num, hspace=self.cv_tool.info['hspace'][1],wspace=self.cv_tool.info['wspace'][1])
        if self.checkBox_use_all.isChecked():
            # axs = [self.widget_cv_view.canvas.figure.add_subplot(len(self.cv_tool.cv_info), col_num, 1 + col_num*(i-1) ) for i in range(1,len(self.cv_tool.cv_info)+1)]
            axs = [self.widget_cv_view.canvas.figure.add_subplot(gs_left[i, 0]) for i in range(0,len(self.cv_tool.cv_info))]
            #self.cv_tool.plot_cv_files(axs = axs)
            self.cv_tool.plot_cv_files(axs = axs)
        else:
            # axs = [self.widget_cv_view.canvas.figure.add_subplot(len(self.cv_tool.info['selected_scan']), col_num, 1 + col_num*(i-1) ) for i in range(1,len(self.cv_tool.info['selected_scan'])+1)]
            axs = [self.widget_cv_view.canvas.figure.add_subplot(gs_left[i, 0]) for i in range(0,len(self.cv_tool.info['selected_scan']))]
            #self.cv_tool.plot_cv_files_selected_scans(axs = axs, scans = self.cv_tool.info['selected_scan'])
            self.cv_tool.plot_cv_files_selected_scans(axs = axs, scans = self.cv_tool.info['selected_scan'])
        labels = []
        for scan, each in zip([self.cv_tool.info['selected_scan'],self.cv_tool.info['sequence_id']][int(self.checkBox_use_all.isChecked())],axs):
            #index in the selected scan, if use all scans, then i=i_full
            i = axs.index(each)
            #index in the full sequence
            i_full = self.cv_tool.info['sequence_id'].index(scan)

            bounds_pot, tick_locs_pot, padding_pot, num_tick_marks_pot, fmt_pot, func_pot = self.cv_tool.info['cv_bounds_pot'].rsplit('+')
            bounds_current, tick_locs_current, padding_current, num_tick_marks_current, fmt_current, func_current = self.cv_tool.info['cv_bounds_current'].rsplit('+')
            show_tick_label_pot = self.cv_tool.info['cv_show_tick_label_x'][i_full]
            show_tick_label_current = self.cv_tool.info['cv_show_tick_label_y'][i_full]

            self._format_axis(each)
            self._format_ax_tick_labels(ax = each,
                                        fun_set_bounds = func_pot,
                                        bounds = eval(bounds_pot),
                                        bound_padding = float(padding_pot),
                                        major_tick_location = eval(tick_locs_pot),
                                        show_major_tick_label = show_tick_label_pot, #show major tick label for the first scan
                                        num_of_minor_tick_marks=int(num_tick_marks_pot),
                                        fmt_str = fmt_pot)
            self._format_ax_tick_labels(ax = each,
                                        fun_set_bounds = func_current,
                                        bounds = eval(bounds_current),
                                        bound_padding = float(padding_current),
                                        major_tick_location = eval(tick_locs_current),
                                        show_major_tick_label = show_tick_label_current, #show major tick label for the first scan
                                        num_of_minor_tick_marks=int(num_tick_marks_current),
                                        fmt_str = fmt_current)

            #set the index text marker for figure (eg. a), b) and so on ... )
            coord_top_left = np.array([eval(bounds_pot)[0]-float(padding_pot),eval(bounds_current)[1]+float(padding_current)])
            offset = np.array(self.cv_tool.info['index_header_pos_offset_cv'])
            coord_top_index_marker = coord_top_left+offset
            label_map = dict(zip(range(26),list('abcdefghijklmnopqrstuvwxyz')))
            each.text(*coord_top_index_marker, '{})'.format(label_map[i]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))
            #set pH label as title
            pH_text = 'pH {}'.format(self.cv_tool.info['ph'][i_full])
            which_pH13 = 0
            if self.cv_tool.info['ph'][i_full]==13:
                for each_scan in self.cv_tool.info['sequence_id']:
                    if self.cv_tool.info['ph'][self.cv_tool.info['sequence_id'].index(each_scan)]==13:
                        which_pH13 = which_pH13+1
                        if each_scan==scan:
                            pH_text = pH_text+'({})'.format(which_pH13)
                            break
            labels.append(pH_text)
            # ph_marker_pos = coord_top_left-[-0.1,eval(bounds_current)[1]*0.3]
            ph_marker_pos = coord_top_left-[-abs(eval(bounds_pot)[0]-eval(bounds_pot)[1])*0.2,eval(bounds_current)[1]*0.35]
            each.text(*ph_marker_pos, pH_text, fontsize = int(self.cv_tool.info['fontsize_text_marker']),color = self.cv_tool.info['color'][i_full])
            #set axis label
            if len(axs)<7:#show all y labels when the total num of ax is fewer than 7
                each.set_ylabel(r'j / mAcm$^{-2}$',fontsize = int(self.cv_tool.info['fontsize_axis_label']))
            else:#only show the y label for the middle panel
                if i== int((len(axs)-1)/2):
                    each.set_ylabel(r'j / mAcm$^{-2}$',fontsize = int(self.cv_tool.info['fontsize_axis_label']))

            if each == axs[-1]:
                each.set_xlabel(r'E / V$_{RHE}$',fontsize = int(self.cv_tool.info['fontsize_axis_label']))
            #now set the fontsize for tick marker
            for tick in each.xaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))
            for tick in each.yaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))

            #add scalling factor marker on each panel
            text_pos = (1,3)
            if 'scale_factor_text_pos' in self.cv_tool.info:
                if i_full < len(self.cv_tool.info['scale_factor_text_pos']):
                    text_pos = self.cv_tool.info['scale_factor_text_pos'][i_full]
                elif len(self.cv_tool.info['scale_factor_text_pos'])==0:
                    print('The length of text_pos doesnot match the lenght of scans, use default pos (1,3) instead!')
                else:
                    text_pos = self.cv_info.info['scale_factor_text_pos'][-1]
                    print('The length of text_pos doesnot match the lenght of scans, use the last item instead!')
            else:
                print('scale_factor_text_pos NOT existing in the config file, use default pos (1,3) instead!')
            each.text(*text_pos,'x{}'.format(self.cv_tool.info['cv_scale_factor'][i_full]),color=self.cv_tool.info['color'][i_full], fontsize = int(self.cv_tool.info['fontsize_text_marker']))
        tafel_row_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','tafel_row_range')])
        tafel_col_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','tafel_col_range')])
        tafel_row_lf = [tafel_row_range[0],0][int(self.checkBox_default.isChecked())]
        tafel_row_rt = [tafel_row_range[1],1][int(self.checkBox_default.isChecked())]
        tafel_col_lf = [tafel_col_range[0],1][int(self.checkBox_default.isChecked())]
        tafel_col_rt = [tafel_col_range[1],2][int(self.checkBox_default.isChecked())]

        order_row_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','rxn_order_row_range')])
        order_col_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','rxn_order_col_range')])
        order_row_lf = [order_row_range[0],1][int(self.checkBox_default.isChecked())]
        order_row_rt = [order_row_range[1],2][int(self.checkBox_default.isChecked())]
        order_col_lf = [order_col_range[0],1][int(self.checkBox_default.isChecked())]
        order_col_rt = [order_col_range[1],2][int(self.checkBox_default.isChecked())]

        charge_row_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','charge_row_range')])
        charge_col_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','charge_col_range')])
        charge_row_lf = [charge_row_range[0],2][int(self.checkBox_default.isChecked())]
        charge_row_rt = [charge_row_range[1],3][int(self.checkBox_default.isChecked())]
        charge_col_lf = [charge_col_range[0],1][int(self.checkBox_default.isChecked())]
        charge_col_rt = [charge_col_range[1],2][int(self.checkBox_default.isChecked())]
        #charge_col = [int(self.lineEdit_col_charge.text()),1][int(self.checkBox_default.isChecked())]
        # axs_2 = [self.widget_cv_view.canvas.figure.add_subplot(gs_right[0:1,1]),self.widget_cv_view.canvas.figure.add_subplot(gs_right[1:3,1])]
        axs_2 = [self.widget_cv_view.canvas.figure.add_subplot(gs_right[tafel_row_lf:tafel_row_rt,tafel_col_lf:tafel_col_rt]),self.widget_cv_view.canvas.figure.add_subplot(gs_right[order_row_lf:order_row_rt,order_col_lf:order_col_rt])]

        self.plot_reaction_order_and_tafel(axs = axs_2)
        # ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[3:,1])
        ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[charge_row_lf:charge_row_rt,charge_col_lf:charge_col_rt])
        self._format_axis(ax_3)
        if self.checkBox_use_all.isChecked():
            bar_list = ax_3.bar(range(len(self.cv_tool.info['charge'])),self.cv_tool.info['charge'],0.5)
            bar_colors = self.cv_tool.info['color']
        else:
            index_ = [i for i in range(len(self.cv_tool.info['sequence_id'])) if self.cv_tool.info['sequence_id'][i] in self.cv_tool.info['selected_scan']]
            bar_list = ax_3.bar(range(len(self.cv_tool.info['selected_scan'])),[self.cv_tool.info['charge'][i] for i in index_],0.5)
            bar_colors = [self.cv_tool.info['color'][i] for i in index_]

        ax_3.set_ylabel(r'q / mCcm$^{-2}$')
        ax_3.set_xlabel(r'Measurement sequence')
        ax_3.set_ylim(0,max(self.cv_tool.info['charge'])*1.3)
        ax_3.set_xticks(range(len(bar_list)))
        ax_3.set_xticklabels(range(1,1+len(bar_list)))
        # ax_3.set_xticklabels(labels[0:len(self.cv_tool.info['charge'])])

        coord_top_left = np.array([len(bar_list)-1-0.1, max(self.cv_tool.info['charge'])*1.1])
        offset = np.array([0,0])
        coord_top_index_marker = coord_top_left+offset
        label_map = dict(zip(range(26),list('abcdefghijklmnopqrstuvwxyz')))
        ax_3.text(*coord_top_index_marker, '{})'.format(label_map[len(axs)+2]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))

        for i, bar_ in enumerate(bar_list):
            # bar_.set_color(self.cv_tool.info['color'][i])
            bar_.set_color(bar_colors[i])
        # try:
        #     self.plot_reaction_order_and_tafel(axs = axs_2)
        #     # ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[3:,1])
        #     ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[charge_row_lf:charge_row_rt,charge_col_lf:charge_col_rt])
        #     bar_list = ax_3.bar(range(len(self.cv_tool.info['charge'])),self.cv_tool.info['charge'],0.5)
        #     ax_3.set_ylabel(r'q / mCcm$^{-2}$')
        #     #ax_3.set_title('Comparison of charge')
        #     ax_3.set_xticks(range(len(self.cv_tool.info['charge'])))
        #     #labels = ['HM1','HM2', 'HM3', 'PEEK1', 'PEEK2']
        #     ax_3.set_xticklabels(labels[0:len(self.cv_tool.info['charge'])])
        #     for i, bar_ in enumerate(bar_list):
        #         bar_.set_color(self.cv_tool.info['color'][i])
        # except:
        #     pass

        #ax_3.legend()
        #self.widget_cv_view.fig.subplots_adjust(wspace=0.31,hspace=0.15)
        self.widget_cv_view.canvas.figure.set_size_inches(self.cv_tool.info['figsize'])
        self.widget_cv_view.canvas.draw()

    def _setup_matplotlib_fig(self,plot_dim):
        self.mplwidget.fig.clear()
        # #[rows, columns]
        for scan in self.scans:
            setattr(self,'plot_axis_scan{}'.format(scan),[])
            j = self.scans.index(scan) + 1
            for i in range(plot_dim[0]):
                getattr(self,'plot_axis_scan{}'.format(scan)).append(self.mplwidget.canvas.figure.add_subplot(plot_dim[0], plot_dim[1],j+plot_dim[1]*i))
                self._format_axis(getattr(self,'plot_axis_scan{}'.format(scan))[-1])

    def _prepare_data_range_and_pot_range(self):
        '''
        data_range = self.lineEdit_data_range.text().rsplit(',')
        if len(data_range) == 1:
            data_range = [list(map(int,data_range[0].rsplit('-')))]*len(self.scans)
        else:
            assert len(data_range) == len(self.scans)
            data_range = [list(map(int,each.rsplit('-'))) for each in data_range]
        self.data_range = data_range
        '''
        # pot_range is a partial set from the specified data_ranges
        # which this, it is more intuitive to pick the data points for variantion calculation (bar chart)
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

        temp_pot = np.array(self.pot_range).flatten()
        pot_range_min_max = [min(temp_pot), max(temp_pot)]
        data_range = []
        for scan in self.scans:
            data_range_ = self._get_data_range_auto(scan = scan,
                                                    ref_pot_low = pot_range_min_max[0],
                                                    ref_pot_high = pot_range_min_max[1],
                                                    cycle = self.spinBox_cycle.value(),
                                                    sweep_direction = self.comboBox_scan_dir.currentText(),
                                                    threshold = 10)
            data_range.append(sorted(data_range_))
        self.data_range = data_range


    #given the scan number (scan), automatically locate the data point range corresponding to potential range from
    #ref_pot_low to ref_pot_high at scan cycle of (cycle) with potential sweep direction defined by (sweep_direction)
    #threshold is internally used to seperate potential groups
    #the result could be empty sometimes, especially when the given potential range is larger than the real potential limits in the data
    def _get_data_range_auto(self, scan, ref_pot_low, ref_pot_high,  cycle = -1, sweep_direction = 'down', threshold = 10):
        pot = self.data_to_plot[scan]['potential']
        def _locate_unique_index(ref_pot):
            idxs = sorted(list(np.argpartition(abs(pot-ref_pot), 18)[0:18]))
            sep_index = [0]
            for i, each in enumerate(idxs):
                if i>0 and each-idxs[i-1]>threshold:
                    sep_index.append(i)
            sep_index.append(len(idxs))
            group_idx = {}
            for i in range(len(sep_index)-1):
                group_idx[f'group {i}'] = idxs[sep_index[i]:sep_index[i+1]]
            group_idx_single_rep = []
            for each in group_idx:
                group_idx_single_rep.append(group_idx[each][int(len(group_idx[each])/2.)])
            # print(group_idx_single_rep)
            return group_idx_single_rep
        max_pot_idx = _locate_unique_index(max(pot))
        min_pot_idx = _locate_unique_index(min(pot))
        #note the len of each idx must be >2 for at least one case, not work if both have only one item.
        #let's add one item to either max or min idx to make the following logic work
        if len(max_pot_idx)==1 and len(min_pot_idx)==1:
            if max_pot_idx[0]>min_pot_idx[0]:
                max_pot_idx = [0] + max_pot_idx
                min_pot_idx = min_pot_idx + [len(pot)-1]
            else:
                min_pot_idx = [0] + min_pot_idx
                max_pot_idx = max_pot_idx + [len(pot)-1]

        if ref_pot_high>max(pot):
            target_pot_high_idx = max_pot_idx
        else:
            target_pot_high_idx = _locate_unique_index(ref_pot_high)
        if ref_pot_low<min(pot):
            target_pot_low_idx = min_pot_idx
        else:
            target_pot_low_idx = _locate_unique_index(ref_pot_low)

        #you may have cases where the located idx is too close to the adjacent max(min)_idx
        #in these cases set the idx to the associated max or min idx
        for i, each_ in enumerate(target_pot_high_idx):
            indx = np.argmin(abs(np.array(max_pot_idx)-each_))
            if abs(each_-max_pot_idx[indx])<=threshold:
                target_pot_high_idx[i] = max_pot_idx[indx]
        for i, each_ in enumerate(target_pot_low_idx):
            indx = np.argmin(abs(np.array(min_pot_idx)-each_))
            if abs(each_-min_pot_idx[indx])<=threshold:
                target_pot_low_idx[i] = min_pot_idx[indx]
        cases_map_low_idx = {}
        cases_map_high_idx = {}
        # print(ref_pot_high, max(pot), target_pot_high_idx)
        for each in target_pot_high_idx:
            max_idx = np.argmin(abs(np.array(max_pot_idx) - each))
            min_idx = np.argmin(abs(np.array(min_pot_idx) - each))
            if max_pot_idx[max_idx]>=each>=min_pot_idx[min_idx]:
                cases_map_high_idx[(min_pot_idx[min_idx],max_pot_idx[max_idx])] = each
            elif min_pot_idx[min_idx]>=each>=max_pot_idx[max_idx]:
                cases_map_high_idx[(max_pot_idx[max_idx],min_pot_idx[min_idx])] = each
            if each in max_pot_idx:#need to make up the other side if each is right on one max idx
                if min_pot_idx[min_idx]>each and (min_idx-1)>=0:
                    cases_map_high_idx[(min_pot_idx[min_idx-1],each)] = each
                elif min_pot_idx[min_idx]<each and (min_idx+1)<len(min_pot_idx):
                    cases_map_high_idx[(each,min_pot_idx[min_idx+1])] = each

            # if each>=min_pot_idx[min_idx] and each<=max_pot_idx[max_idx]:
                # cases_map_high_idx[(min_pot_idx[min_idx],max_pot_idx[max_idx])] = each
        for each in target_pot_low_idx:
            max_idx = np.argmin(abs(np.array(max_pot_idx) - each))
            min_idx = np.argmin(abs(np.array(min_pot_idx) - each))
            if max_pot_idx[max_idx]>=each>=min_pot_idx[min_idx]:
                cases_map_low_idx[(min_pot_idx[min_idx],max_pot_idx[max_idx])] = each
            elif min_pot_idx[min_idx]>=each>=max_pot_idx[max_idx]:
                cases_map_low_idx[(max_pot_idx[max_idx],min_pot_idx[min_idx])] = each
            if each in min_pot_idx:#need to make up the other side
                if max_pot_idx[max_idx]>each and (max_idx-1)>=0:
                    cases_map_low_idx[(max_pot_idx[max_idx-1],each)] = each
                elif max_pot_idx[max_idx]<each and (max_idx+1)<len(max_pot_idx):
                    cases_map_low_idx[(each,max_pot_idx[max_idx+1])] = each
        print(scan)
        print(min_pot_idx, max_pot_idx)
        print(cases_map_high_idx)
        print(cases_map_low_idx)
        if ref_pot_low == ref_pot_high:
            cases_map_high_idx.update(cases_map_low_idx)
            cases_map_low_idx = cases_map_high_idx

        unique_keys = [each for each in cases_map_low_idx if each in cases_map_high_idx]
        final_group = {'up':[],'down':[]}
        for each in unique_keys:
            temp = [cases_map_low_idx[each], cases_map_high_idx[each]]
            if temp[0]>temp[1]:
                final_group['down'].append(temp)
            elif temp[0]<temp[1]:
                final_group['up'].append(temp)
            else:
                final_group['down'].append(temp)
                final_group['up'].append(temp)

        # print(scan, final_group)
        if ref_pot_low == ref_pot_high:
            #now lets get the scan direction right, at this moment it is a inclusive list of possibility.
            idx_unique = [each_[0] for each_ in final_group['up']]
            #now merge this in the max and min idx container
            full_list = sorted(idx_unique + min_pot_idx + max_pot_idx)
            up_list = []
            down_list = []
            for i, each_ in enumerate(idx_unique):
                idx_in_full_list = full_list.index(each_)
                if idx_in_full_list==0:
                    if each_ in min_pot_idx:
                        up_list.append([each_]*2)
                    elif each_ in max_pot_idx:
                        down_list.append([each_]*2)
                elif idx_in_full_list==len(full_list)-1:
                    if idx_in_full_list==0:
                        if each_ in min_pot_idx:
                            down_list.append([each_]*2)
                        elif each_ in max_pot_idx:
                            up_list.append([each_]*2)
                else:
                    if full_list[idx_in_full_list-1] in min_pot_idx:
                        up_list.append([each_]*2)
                    else:
                        down_list.append([each_]*2)
            final_group = {'up':up_list, 'down': down_list}

        pot_ranges = final_group[sweep_direction]
        assert len(pot_ranges)>0, print('empty list in pot_ranges')
        try:
            return pot_ranges[cycle]
        except:
            print(f'pot_ranges for scan {scan} could not be queried with the cycle index {cycle}, use last cycle instead')
            return pot_ranges[-1]
        '''
        index_list = None
        if type(cycle)==int:
            try:
                pot_range = pot_ranges[cycle]
            except:
                print(f'pot_ranges could not be queried with the cycle index {cycle}, use last cycle instead')
                pot_range = pot_ranges[-1]
            index_list = pot_range
            if ref_pot_low == ref_pot_high:#in this case, we extract only absolute value. And note pot_range[0] = pot_range[1]
                total_change = y_values[pot_range[0]]
                std_val_norm = std_val
            else:
                total_change = abs((y_values[pot_range[1]] - y_values[pot_range[0]])/(pot[pot_range[1]]-pot[pot_range[0]]))
                std_val_norm = std_val/(pot[pot_range[1]]-pot[pot_range[0]])
        else:#here we do average over all cases
            total_change = 0
            for i in range(len(pot_ranges)):
                pot_range = pot_ranges[i]
                if ref_pot_low == ref_pot_high:
                    total_change += y_values[pot_range[0]]
                else:
                    total_change += abs((y_values[pot_range[1]] - y_values[pot_range[0]])/(pot[pot_range[1]]-pot[pot_range[0]]))
            total_change = total_change/len(pot_ranges)
            std_val_norm = std_val/(pot[pot_ranges[0][1]]-pot[pot_ranges[0][0]]) if ref_pot_low != ref_pot_high else std_val
            index_list = pot_ranges[0]#only take the first range
        return total_change, std_val_norm, index_list
        '''

    #cal std for channel centering at point_index with left and right boundary defined such that the span potential_range is reached
    def _cal_std_at_pt(self, scan, channel, point_index, potential_range = 0.02):
        data = self.data_to_plot[scan][channel]
        pot = self.data_to_plot[scan]['potential']
        index_left = point_index
        index_right = point_index
        while True:
            if index_left == 0:
                break
            else:
                if abs(pot[index_left]-pot[point_index])>=potential_range:
                    break
                else:
                    index_left = index_left - 1

        while True:
            if index_right == len(pot)-1:
                break
            else:
                if abs(pot[index_right]-pot[point_index])>=potential_range:
                    break
                else:
                    index_right = index_right + 1

        return np.std(data[index_left:index_right])

    def _cal_structural_change_rate(self,scan, channel,y_values, std_val, data_range, pot_range, marker_index_container, cal_std = True):
        assert 'potential' in list(self.data_to_plot[scan].keys())
        if data_range[0]==data_range[1]:
            index_left = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]+1] - pot_range[0])) + data_range[0]
            index_right = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]+1] - pot_range[1])) + data_range[0]
        else:            
            index_left = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]] - pot_range[0])) + data_range[0]
            index_right = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]] - pot_range[1])) + data_range[0]
        marker_index_container.append(index_left)
        marker_index_container.append(index_right)
        pot_offset = abs(self.data_to_plot[scan]['potential'][index_left]-self.data_to_plot[scan]['potential'][index_right])
        if cal_std:#use error propogation rule here
            # std_val = max([self._cal_std_at_pt(scan, channel, index_left),self._cal_std_at_pt(scan, channel, index_right)])
            std_val = (self._cal_std_at_pt(scan, channel, index_left)**2+self._cal_std_at_pt(scan, channel, index_right)**2)**0.5
        if pot_offset==0:
            if channel == 'current':
                self.data_summary[scan][channel].append(y_values[index_left])
            else:
                self.data_summary[scan][channel].append(y_values[index_left] + self.data_to_plot[scan][channel+'_max'])
            self.data_summary[scan][channel].append(std_val/(2**0.5))
        else:#calculate the slope here
            self.data_summary[scan][channel].append((y_values[index_right] - y_values[index_left])/pot_offset)
            self.data_summary[scan][channel].append(std_val/pot_offset)
        return marker_index_container

    def _plot_one_panel_x_is_potential(self, scan, channel, channel_index, y_values, y_values_smooth, fmt, marker_index_container, slope_info):
        if self.checkBox_use_external_slope.isChecked():
            seperators = self.return_seperator_values(scan)
        else:
            seperators = list(set(marker_index_container))
        if channel!='current':
            #plot the channel values now
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_merge.isChecked():
                if scan!=self.scans[0]:
                    getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_show_smoothed_curve.isChecked():
                # x, y = self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]], y_values_smooth
                # z = np.linspace(0, 1, len(x))
                # line_segment = colorline(x, y, z, cmap=plt.get_cmap('binary'), linewidth=3)
                # getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].add_collection(line_segment)
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values_smooth,'-', color = '0.4')
            #plot the slope line segments
            cases = self._plot_slope_segment(scan = scan, channel = channel, channel_index = channel_index, y_values_smooth = y_values_smooth,
                                        slope_info = slope_info, seperators = seperators,  marker_index_container = marker_index_container)
            #store the calculated strain/size change
            if 'ip' in channel and len(cases)!=0:
                if 'grain' in channel:
                    self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_ranges[scan],'horizontal',cases)
                elif 'strain' in channel:
                    self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_ranges[scan],'horizontal',cases)
            elif 'oop' in channel and len(cases)!=0:
                if 'grain' in channel:
                    self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_ranges[scan],'vertical',cases)
                elif 'strain' in channel:
                    self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_ranges[scan],'vertical',cases)
        #now plot current channel
        else:
            #extract seperators for displaying vertical line segments
            _seperators = []
            if self.checkBox_use_external_slope.isChecked():
                _seperators = seperators[scan][channel]
            else:
                _seperators = [[self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]][each_index]] for each_index in seperators]
            if self.checkBox_use_external_cv.isChecked():
                #plot cv profile from external files
                lim_y = self.plot_cv_from_external(getattr(self,'plot_axis_scan{}'.format(scan))[channel_index],scan,_seperators)
                if self.checkBox_merge.isChecked() and scan!=self.scans[0]:
                    self.plot_cv_from_external(getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index],scan,_seperators)
                #overplot the internal cv data if you want
                #here y is already scaled by 8 considerring the current density to be shown
                if self.checkBox_use_internal_cv.isChecked():
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt, ls = '-', marker = None)
            else:
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt,ls = '-', marker = None)
                #show marker and vert line segments
                if self.checkBox_show_marker.isChecked():
                    [getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([each_seperator, each_seperator],[-100,100],'k:') for each_seperator in _seperators]

    def _plot_one_panel(self, scan, channel, channel_index, y_values, y_values_smooth, x_values, fmt, marker_index_container):
        current_channel = channel == 'current'
        if current_channel:
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(x_values,y_values,fmt, lw=2.5, marker = None, ls='-')
            if self.checkBox_merge.isChecked() and scan!=self.scans[0]:
                getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index].plot(x_values,y_values,fmt, lw=2.5, marker = None, ls='-')
        else:
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(x_values,y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_merge.isChecked() and scan!=self.scans[0]:
                #if merged than plot the profile also at column 0 corresponding to self.scans[0]
                getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index].plot(x_values,y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_show_smoothed_curve.isChecked() and (channel!='potential'):#not show smooth line for grain size channels
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(x_values,y_values_smooth, fmt, color = 'red', lw=2.5, marker = None, ls='-')
            if self.checkBox_show_marker.isChecked():#also display the bounds for specified pot_ranges
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([x_values[iii] for iii in marker_index_container],[y_values_smooth[iii] for iii in marker_index_container],'k*')
                for iii in marker_index_container:
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([x_values[iii]]*2,[-100,100],':k')

    def _plot_slope_segment(self, scan, channel, channel_index, y_values_smooth, slope_info, seperators, marker_index_container):
        if self.checkBox_plot_slope.isChecked() and self.checkBox_use_external_slope.isChecked():
            cases = []
            if slope_info[scan][channel]!=None:
                #one known point coords (the cross point): (p1, y1)
                #slopes are: a1 and a2
                #the other two points coords are: (p0,y0) and (p2, y2)
                p0,p1,p2,y1,a1,a2 = slope_info[scan][channel]
                y0 = a1*(p0-p1)+y1
                y2 = a2*(p2-p1)+y1
                cases = [self.calculate_size_strain_change(p0,p1,p2,y1,a1,a2,pot_range = each_pot) for each_pot in self.pot_ranges[scan]]
                #slope line segments
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([p0,p1,p2],np.array([y0,y1,y2])-self.data_to_plot[scan][channel+"_max"],'k--')
                #vertical line segments
                for pot in seperators[scan][channel]:
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([pot,pot],[-100,100],'k:')
        else:
            cases = [self.calculate_size_strain_change_from_plot_data(scan, channel, self.data_range[self.scans.index(scan)], each_pot) for each_pot in self.pot_range]
            #vertical line segments only
            if self.checkBox_show_marker.isChecked():
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]][iii] for iii in marker_index_container],[y_values_smooth[iii] for iii in marker_index_container],'k*')
                for each_index in seperators:
                    pot = self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]][each_index]
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([pot,pot],[-100,100],'k:')
        return cases

    def _extract_y_values(self, scan, channel):
        y = self.data_to_plot[scan][channel]
        if channel == 'current':#current --> current density
            y = y*8
        #apply offset, very useful if you want to slightly tweak the channel values for setting a common 0 reference point
        #the offset can be specified from GUI
        y_offset = 0
        if hasattr(self, f'{channel}_offset_{scan}'):
            y_offset = getattr(self, f'{channel}_offset_{scan}')
        #apply the offset to y channel
        y = np.array([each + y_offset for each in y])
        y_smooth_temp = signal.savgol_filter(y,41,2)
        #std is calculated this way for estimation of error bar values
        std_val = np.sum(np.abs(y_smooth_temp - y))/len(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]])
        # std_val = (np.sum((y_smooth_temp - y)**2)/len(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]]-1))**0.5
        return y, y_smooth_temp, std_val

    def _get_fmt_style(self, scan, channel):
        #fmt here is a list of one or two format strings (eg. -b;-r,-b;-r will destructure to ['-b','-r'] and ['-b', '-r'])
        #you may want to show lines only for strain channels, but only show symbols for grain size channels due to the large error bar
        #if two items: second one is for grain size channels, first is for the other channels
        #if only one item: all channel share the same fmt style
        try:
            fmt = self.lineEdit_fmt.text().rsplit(',')[self.scans.index(scan)].rsplit(";")
        except:
            fmt = ['b-']
        #extract the fmt tag
        if len(fmt)==2:
            fmt = fmt[int('size' in channel)]
        else:
            fmt = fmt[0]
        return fmt

    def _update_bounds_xy(self, scan, y_values, channel_index, x_min_value, x_max_value, y_min_values, y_max_values):
        temp_max, temp_min = max(y_values), min(y_values)
        temp_max_x, temp_min_x = max(list(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]])), min(list(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]]))
        if temp_max_x > x_max_value:
            x_max_value = temp_max_x
        if temp_min_x < x_min_value:
            x_min_value = temp_min_x
        if y_max_values[channel_index]<temp_max:
            y_max_values[channel_index] = temp_max
        if y_min_values[channel_index]>temp_min:
            y_min_values[channel_index] = temp_min
        return x_min_value, x_max_value, y_min_values, y_max_values

    def _set_xy_tick_labels(self, scan, channel, channel_index, channel_length):
        ##set x tick labels
        #the x tick lable only shown for the last panel, will be hidden for the others
        if channel_index!=(channel_length-1):
            ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index]
            ax.set_xticklabels([])
        else:#show x tick label for last panel, either potential or image_no
            ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index]
            x_label = [r'Time (s)','E / V$_{RHE}$'][self.plot_label_x[self.scans.index(scan)]=='potential']
            ax.set_xlabel(x_label, fontsize = 13)
        ##set y tick labels
        #the y tick label only shown for the first column panel
        if scan!=self.scans[0]:
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].set_yticklabels([])
        else:
            #according to relative scale
            y_label_map = {'potential':'E / V$_{RHE}$',
                            'current':r'j / mAcm$^{-2}$',
                            'strain_ip':r'$\Delta\varepsilon_\parallel$  (%)',
                            'strain_oop':r'$\Delta\varepsilon_\perp$  (%)',
                            'grain_size_oop':r'$\Delta d_\perp$ / nm',
                            'grain_size_ip':r'$\Delta d_\parallel$ / nm',
                            'peak_intensity':r'Intensity / a.u.'}
            #based on absolute values
            y_label_map_abs = {'potential':'E / V$_{RHE}$',
                                'current':r'j / mAcm$^{-2}$',
                                'strain_ip':r'$\varepsilon_\parallel$  (%)',
                                'strain_oop':r'$\varepsilon_\perp$  (%)',
                                'grain_size_oop':r'$ d_\perp$ / nm',
                                'grain_size_ip':r'$ d_\parallel$ / nm',
                                'peak_intensity':r'Intensity / a.u.'}
            if not self.checkBox_max.isChecked():
                y_label_map = y_label_map_abs
            if channel in y_label_map:
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].set_ylabel(y_label_map[channel], fontsize = 13)

    def _cal_pseudcap_charge(self, scan):
        for each_pot_range in self.pot_ranges[scan]:
            try:
                horizontal = self.grain_size_info_all_scans[scan][each_pot_range]['horizontal']
                vertical = self.grain_size_info_all_scans[scan][each_pot_range]['vertical']
                q_skin,q_film = self.estimate_charge_from_skin_layer_thickness_philippe_algorithm({"horizontal":horizontal,"vertical":vertical})
                if scan not in self.charge_info:
                    self.charge_info[scan] = {}
                    self.charge_info[scan][each_pot_range] = {'skin_charge':q_skin,'film_charge':q_film,'total_charge':0}
                else:
                    self.charge_info[scan][each_pot_range]['skin_charge'] = q_skin
                    self.charge_info[scan][each_pot_range]['film_charge'] = q_film
            except:
                print('Fail to cal charge info. Check!')

    def _do_text_label(self, scan, count_pH13, x_min_value, y_max_values):
        #overwrite max_y using the format setting
        if 'master' in self.tick_label_settings:
            if 'current' in self.tick_label_settings['master']:
                y_max_values[0] = self.tick_label_settings['master']['current']['locator'][-1] + float(self.tick_label_settings['master']['current']['padding'])

        #extract color
        try:#from cv settings
            _,_,_,_,_,color, _, _ = self.plot_lib[scan]
        except:#specified in gui
            color = self.comboBox_color.currentText()

        # pH labeling
        text = r'pH {}'.format(self.phs[self.scans.index(scan)])
        tag = ''
        if self.radioButton_pH.isChecked():
            if self.phs[self.scans.index(scan)]==13:
                text = r'pH {} ({})'.format(self.phs[self.scans.index(scan)],count_pH13)
                count_pH13 += 1
            else:
                text = r'pH {}'.format(self.phs[self.scans.index(scan)])
        # scan number label
        elif self.radioButton_scan.isChecked():
            text = f'scan{scan}'
        elif self.radioButton_custom.isChecked():
            labels = self.lineEdit_custom_label.text().rstrip().rsplit(',')
            index = self.scans.index(scan)
            if len(labels)>index:
                text = labels[index]
            else:
                print('The dimention of text label does not match the total number of scans!')
                text = f'{scan}'
        # without label
        else:
            text = ''
        #set label here
        text_obj = getattr(self,'plot_axis_scan{}'.format(scan))[0].text(x_min_value, y_max_values[0]*0.8,text,color = color,fontsize=11)
        setattr(self, f'text_obj_{scan}', text_obj)
        return count_pH13

    def _decorate_axis_tick_labels(self, scan, channel, channel_index, x_min_value, x_max_value, y_min_values, y_max_values):

        if self.plot_label_x[self.scans.index(scan)] == 'potential':
            if 'master' in self.tick_label_settings:
                if 'potential' in self.tick_label_settings['master']:
                    if self.checkBox_use.isChecked():
                        self._format_ax_tick_labels(ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index],
                                                    fun_set_bounds = self.tick_label_settings['master']['potential']['func'],#'set_xlim',
                                                    bounds = [x_min_value,x_max_value],#[0.4,2.1],#[0.95,1.95],
                                                    bound_padding = float(self.tick_label_settings['master']['potential']['padding']),
                                                    major_tick_location = self.tick_label_settings['master']['potential']['locator'], #x_locator
                                                    show_major_tick_label = (len(self.plot_labels_y)-1)==channel_index, #show major tick label for the first scan
                                                    num_of_minor_tick_marks=self.tick_label_settings['master']['potential']['tick_num'], #4
                                                    fmt_str = self.tick_label_settings['master']['potential']['fmt'])#'{:3.1f}'

        #y axis
        if 'master' in self.tick_label_settings:
            if channel in self.tick_label_settings['master']:
                if self.checkBox_use.isChecked():
                    self._format_ax_tick_labels(ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index],
                                                fun_set_bounds = self.tick_label_settings['master'][channel]['func'],#'set_xlim',
                                                bounds = [y_min_values[channel_index],y_max_values[channel_index]],#[0.4,2.1],#[0.95,1.95],
                                                bound_padding = float(self.tick_label_settings['master'][channel]['padding']),
                                                major_tick_location = self.tick_label_settings['master'][channel]['locator'], #x_locator
                                                show_major_tick_label = self.scans.index(scan)==0, #show major tick label for the first scan
                                                num_of_minor_tick_marks=self.tick_label_settings['master'][channel]['tick_num'], #4
                                                fmt_str = self.tick_label_settings['master'][channel]['fmt'])#'{:3.1f}'

    #plot the master figure
    def plot_figure_xrv(self):
        #update state and reset meta data
        self.make_plot_lib()#external cv files
        self.reset_meta_data()#calculated values (eg strain and grain size) and tick label setting reset to empty {}
        self.extract_tick_label_settings()#extract the latest tick label setting

        #extract slope info if any
        slope_info_temp = None
        if self.checkBox_use_external_slope.isChecked():
            slope_info_temp = self.return_slope_values()

        #init plot settings, create figure axis, and init the bounds of x and y axis
        self._setup_matplotlib_fig(plot_dim = [len(self.plot_labels_y), len(self.scans)])
        #these are extreme values, these values will be updated
        y_max_values,y_min_values = [-100000000]*len(self.plot_labels_y),[100000000]*len(self.plot_labels_y) #multiple sets of ylim
        x_min_value, x_max_value = [1000000000,-10000000000] # one set of xlim

        #prepare ranges for viewing datasummary, which summarize the variance of structural pars in each specified range
        #this is a way to remove duplicate data points if there are multiple cycles
        self._prepare_data_range_and_pot_range()

        #the main loop starts from here
        for scan in self.scans:
            self.cal_potential_ranges(scan)
            #data_summary, summarizing values of structural changes, is used to plot bar char afterwards
            self.data_summary[scan] = {}
            if 'potential' in self.plot_labels_y and self.plot_label_x[self.scans.index(scan)] == 'potential':
                plot_labels_y = [each for each in self.plot_labels_y if each!='potential'] # remove potential in y lables if potential is set as x channel already
            else:
                plot_labels_y = self.plot_labels_y
            #plot each y channel from here
            for each in plot_labels_y:
                #each is the y channel string tag
                self.data_summary[scan][each] = []
                #i is the channel index
                i = plot_labels_y.index(each)
                #is this the current channel
                current_channel = each == 'current'
                #y vs image_no?
                x_is_frame_no = self.plot_label_x[self.scans.index(scan)] == 'image_no'
                #extract fmt style
                fmt = self._get_fmt_style(scan = scan, channel = each)
                #extract the channel values
                y, y_smooth_temp, std_val = self._extract_y_values(scan = scan, channel = each)
                #this marker container will contain the positions of potentials bounds according to the specified potential ranges
                marker_index_container = []
                for ii in range(len(self.pot_range)):
                    marker_index_container = self._cal_structural_change_rate(scan = scan, channel =each,
                                                                              y_values = y_smooth_temp,
                                                                              std_val = std_val,
                                                                              data_range = self.data_range[self.scans.index(scan)],
                                                                              pot_range = self.pot_range[ii],
                                                                              marker_index_container = marker_index_container)
                # print(marker_index_container)
                # if the plot channel is versus time (image_no)
                if x_is_frame_no:
                    #x offset
                    offset_ = 0
                    if hasattr(self,f'image_no_offset_{scan}'):
                        offset_ = getattr(self, f'image_no_offset_{scan}')
                    #here two situations, plot current density or plot other channels
                    #current density: you can either use internal data points or extract values from external files
                    #NOTE: the sampling rate of internal data is way smaller than that of external files, so we don't want to use external file for plotting current
                    self._plot_one_panel(scan = scan, channel = each, channel_index = i, y_values = y,
                                         y_values_smooth = y_smooth_temp, x_values = np.arange(len(y))+offset_,
                                         fmt = fmt, marker_index_container = marker_index_container)
                #if the plot channel is versus potential
                else:
                    self._plot_one_panel_x_is_potential(scan = scan, channel = each, channel_index = i, y_values = y,
                                                        y_values_smooth = y_smooth_temp, fmt = fmt, marker_index_container = marker_index_container,
                                                        slope_info = slope_info_temp)
                #update the x and y bounds
                x_min_value, x_max_value, y_min_values, y_max_values  = self._update_bounds_xy(scan = scan, y_values = y, channel_index = i,
                                                                                              x_min_value = x_min_value, x_max_value = x_max_value,
                                                                                              y_min_values = y_min_values, y_max_values = y_max_values)
                #set xy tick labels
                self._set_xy_tick_labels(scan = scan, channel = each, channel_index = i, channel_length = len(plot_labels_y))
            #now calculate the pseudocapacitative charge values
            self._cal_pseudcap_charge(scan = scan)
        count_pH13_temp = 1#count the times of dataset for pH 13
        #text labeling on the master figure
        for scan in self.scans:
            # label display only on the first row
            count_pH13_temp = self._do_text_label(scan = scan, count_pH13 = count_pH13_temp, x_min_value = x_min_value, y_max_values = y_max_values)
            for each in self.plot_labels_y:
                i = self.plot_labels_y.index(each)
                getattr(self,'plot_axis_scan{}'.format(scan))[i].set_xlim(x_min_value, x_max_value)
                getattr(self,'plot_axis_scan{}'.format(scan))[i].set_ylim(*[y_min_values[i],y_max_values[i]])
                ####The following lines are customized for axis formating (tick locations, padding, ax bounds)
                #decorate axis tick labels
                self._decorate_axis_tick_labels(scan = scan, channel = each, channel_index = i, x_min_value = x_min_value, x_max_value = x_max_value,
                                                y_min_values = y_min_values, y_max_values = y_max_values)
        self.mplwidget.fig.tight_layout()
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
        #redefine the bounds using major tick locations
        major_tick_values = [float(each) for each in major_tick_location]
        bounds = [min(major_tick_values), max(major_tick_values)]
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

    def _find_reference_at_potential(self, target_pot, pot_array, y_array):
        offset_abs = np.abs(np.array(pot_array)-target_pot)
        #only select three smaller points
        idx = np.argpartition(offset_abs, 4)[0:4]
        y_sub = [y_array[each] for each in idx]
        index_found = idx[y_sub.index(max(y_sub))]
        return y_array[index_found]

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
                    y_smooth_temp = signal.savgol_filter(temp_data,41,2)
                    # if scan_number==24000:
                        # y_smooth_temp = temp_data
                    # self.data_to_plot[scan_number][each] = list(temp_data-max(temp_data))
                    #if time_scan, then target_pots are actually image_no
                    target_pots = [float(each) for each in self.lineEdit_reference_potential.text().rstrip().rsplit(',')]
                    if len(target_pots)==1:
                        target_pots = target_pots*2
                    target_pot = target_pots[int('size' in each)]
                    channel = ['potential','image_no'][int(self.checkBox_time_scan.isChecked())]
                    temp_max = self._find_reference_at_potential(target_pot = target_pot, pot_array = self.data_to_plot[scan_number][channel], y_array = y_smooth_temp)
                    if self.checkBox_max.isChecked():
                        self.data_to_plot[scan_number][each] = list(temp_data-temp_max)
                        self.data_to_plot[scan_number][each+'_max'] = temp_max
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
        self._init_meta_data_for_scans(scans)
        self.phs = [self.phs_all[self.scans_all.index(each)] for each in scans]

        #self.plot_label_x = self.lineEdit_x.text()
        self.plot_label_x = self.lineEdit_x.text().rstrip().rsplit(',')
        if len(self.plot_label_x)==1:
            self.plot_label_x = self.plot_label_x*len(scans)
        else:
            if len(self.plot_label_x)<len(scans):
                self.plot_label_x = self.plot_label_x + [self.plot_label_x[-1]]*(len(scans) - len(self.plot_label_x))
        self.plot_labels_y = self.lineEdit_y.text().rstrip().rsplit(',')
        self.comboBox_scans.clear()
        self.comboBox_scans.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_2.clear()
        self.comboBox_scans_2.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_3.clear()
        self.comboBox_scans_3.addItems([str(each) for each in sorted(scans)])

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())
