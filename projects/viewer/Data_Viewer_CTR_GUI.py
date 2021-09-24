import sys,os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
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
from PlotSetup import data_viewer_plot_cv
import pandas as pd
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.ticker import AutoMinorLocator
from scipy import signal

# import scipy.signal.savgol_filter as savgol_filter

#from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

class MyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        # pg.setConfigOptions(imageAxisOrder='row-major')
        # pg.mkQApp()
        uic.loadUi(os.path.join(DaFy_path,'projects','viewer','data_viewer_CTR_new.ui'),self)
        # self.setupUi(self)
        # plt.style.use('ggplot')
        self.widget_terminal.update_name_space('main_gui',self)
        self.setWindowTitle('CTR data Viewer')
        self.set_plot_channels()
        self.data_to_save = {}
        #pot_offset is the difference between the spock value and those recorded by potentiostat
        #you need some calibration step to figure out this value not necessarily always 0.055 V
        #the correction will be real_pot = spock_value + pot_offset
        self.potential_offset = 0.055
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        plt.rcParams.update({'axes.labelsize': 10})
        plt.rc('font',size = 10)
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = False
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['mathtext.default']='regular'
        plt.rcParams['savefig.dpi'] = 300
        self.open.clicked.connect(self.load_file)
        self.plot.clicked.connect(self.plot_figure)
        # self.apply.clicked.connect(self.replot_figure)
        self.PushButton_append_scans.clicked.connect(self.append_scans)
        self.pushButton_filePath.clicked.connect(self.locate_data_folder)
        self.PushButton_fold_or_unfold.clicked.connect(self.fold_or_unfold)
        self.checkBox_time_scan.clicked.connect(self.set_plot_channels)
        self.radioButton_ctr.clicked.connect(self.set_plot_channels)
        # self.radioButton_xrv.clicked.connect(self.set_plot_channels)
        self.checkBox_mask.clicked.connect(self.append_scans)
        self.pushButton_load_config.clicked.connect(self.load_config)
        self.pushButton_save_config.clicked.connect(self.save_config_ctr)
        self.pushButton_save_data.clicked.connect(self.save_data_method)

        self.pushButton_auto_plot.clicked.connect(self.generate_plot_settings)
        self.pushButton_reset.clicked.connect(self.reset)
        # self.pushButton_save_xrv_data.clicked.connect(self.save_xrv_data)
        # self.pushButton_plot_datasummary.clicked.connect(self.plot_data_summary_xrv)
        self.data = None
        self.data_summary = {} 
        self.data_range = None
        self.pot_range = None
        self.potential = []
       
    def reset(self):
        self.data = None
        self.data_to_plot = {}

    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName)) 

    def save_data_method(self):
        # print(self.data_to_save.keys())
        for each in self.data_to_save:
            # print(self.data_to_save[each])
            self.data_to_save[each].to_csv(os.path.join(self.lineEdit_data_file_path.text(), self.lineEdit_data_file_name.text()+'_{}.csv'.format(each)),header = False, sep =' ',index=False)

    def uniform_symmetry_rods(self,hk=[1,3]):
        if (hk[0]>=0) and (hk[1]>=0):#[1,1]-->[1,1]
            return hk
        elif (hk[0]<=0) and (hk[1]<=0):#[-1,-3]-->[3,1]
            return [abs(each) for each in hk]
        else:
            return [abs(each) for each in hk[::-1]]

    def generate_plot_settings(self):
        self.data_to_plot = {}
        set_of_NhkE = list(set(zip(self.data['scan_no'],self.data['H'].round(0),self.data['K'].round(0),self.data['potential'].round(1))))
        set_of_potential = sorted(list(set(self.data['potential'].round(1))))
        set_of_hk = list(set(zip(self.data['H'].round(0),self.data['K'].round(0))))
        all_scans = []
        plot_styles = '-*r;-*y;-*b;-*g;-*m;-*k;:*r;:*y;:*b;:*g;:*m;:*k;--*r;--*y;--*b;--*g;--*m;--*k'
        for each_hk in set_of_hk:
            temp_scans = []
            for each_potential in set_of_potential:
                sub_scans = []
                for each_set in set_of_NhkE:
                    if ((each_hk[0],each_hk[1],each_potential)==each_set[1:]):
                        sub_scans.append(str(each_set[0]))
                temp_scans.append(sorted(sub_scans))
            all_scans.append(temp_scans)
        self.lineEdit_fmt.setText(';'.join(plot_styles.rsplit(';')[0:len(set_of_potential)]))
        self.lineEdit_labels.setText(';'.join(['{} V'.format(each) for each in set_of_potential]))
        all_scans_plot = []
        for each in all_scans:
            temp = []
            for each_sub in each:
                temp.append(','.join(each_sub))
            all_scans_plot.append(';'.join(temp))
        all_scans_plot = ['[{}]'.format(each) for each in all_scans_plot]
        self.scan_numbers_append.setText('+'.join(all_scans_plot))
        self.append_scans_ctr()
        return all_scans_plot

    #save a segment of data to be formated for loading in superrod
    def save_xrv_data(self):
        key_map_lib = {
                       'peak_intensity':1,
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
                         'I':[],
                         'eI':[],
                         'e1':[],
                         'e2':[]}
            for key in key_map_lib:
                temp_data['potential'] = temp_data['potential'] + list(data_['potential'][data_range_[0]:])
                temp_data['eI'] = temp_data['eI'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['Y'] = temp_data['Y'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['e1'] = temp_data['e1'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['e2'] = temp_data['e2'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['items'] = temp_data['items'] + [key_map_lib[key]]*len(data_['potential'][data_range_[0]:])
                temp_data['scan_no'] = temp_data['scan_no'] + [scan_]*len(data_['potential'][data_range_[0]:])
                temp_data['I'] = temp_data['I'] + list(data_[key][data_range_[0]:])
            df = pd.DataFrame(temp_data)
            df.to_csv(os.path.join(self.lineEdit_data_file_path.text(), self.lineEdit_data_file_name.text()+'_{}.csv'.format(scan_)),\
                      header = False, sep =' ',columns = list(temp_data.keys()), index=False)

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
                    getattr(self,channel).setText(value)
        self._load_file()
        self.append_scans()

    def save_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","config file (*.ini);Text Files (*.txt);all files(*.*)", options=options)
        with open(fileName,'w') as f:
            channels = ['lineEdit_data_file','radioButton_ctr','radioButton_xrv','checkBox_time_scan','checkBox_mask','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt','lineEdit_labels',\
                        'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar']
            for channel in channels:
                try:
                    f.write(channel+':'+str(getattr(self,channel).isChecked())+'\n')
                except:
                    f.write(channel+':'+getattr(self,channel).text()+'\n')

    def save_config_ctr(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","config file (*.ini);Text Files (*.txt);all files(*.*)", options=options)
        with open(fileName,'w') as f:
            channels = ['lineEdit_data_file','radioButton_ctr','checkBox_time_scan','checkBox_mask','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt','lineEdit_labels']
            for channel in channels:
                try:
                    f.write(channel+':'+str(getattr(self,channel).isChecked())+'\n')
                except:
                    f.write(channel+':'+getattr(self,channel).text()+'\n')  

    def set_plot_channels(self):
        time_scan = self.checkBox_time_scan.isChecked()
        if time_scan:
            self.lineEdit_x.setText('image_no')
            self.lineEdit_y.setText('peak_intensity,potential')
        else:
            self.lineEdit_x.setText('L')
            self.lineEdit_y.setText('peak_intensity,peak_intensity_error,H,K') 

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
        if self.radioButton_xrv.isChecked():
            self.phs_all = [list(self.data[self.data['scan_no']==scan]['phs'])[0] for scan in scans]
            phs = 'pHs\n'+str(self.phs_all)+'\n'
        else:
            phs = ''
        self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers,phs]))

    def load_file(self):
        if self.radioButton_ctr_model.isChecked():
            self.load_file_ctr_model()
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;All Files (*.csv)", options=options)
            if fileName:
                self.lineEdit_data_file.setText(fileName)
                if type(self.data)==type(None):
                    self.data = pd.read_excel(fileName)
                else:
                    self.data = self.data.append(pd.read_excel(fileName))
            col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
            self.data['H'] = self.data['H'].round(0)
            self.data['K'] = self.data['K'].round(0)
            #uniform HK
            for i in range(len(self.data['H'])):
                HK = self.uniform_symmetry_rods([self.data['H'][i],self.data['K'][i]])
                self.data.iloc[i,self.data.columns.get_loc('H')] = HK[0]
                self.data.iloc[i,self.data.columns.get_loc('K')] = HK[1]
            scans = list(set(list(self.data['scan_no'])))
            self.scans_all = scans
            scans.sort()
            scan_numbers = 'scan_nos\n'+str(scans)+'\n'
            # print(list(self.data[self.data['scan_no']==scans[0]]['phs'])[0])
            self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers]))


    def load_file_ctr_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;All Files (*.csv)", options=options)
        if fileName:
            self.lineEdit_data_file.setText(fileName)
            self.data = pd.read_excel(fileName)
        col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
        self.potentials = sorted(list(set(list(self.data['potential']))))
        self.hk_list = list(set(list(zip(list(self.data['H']),list(self.data['K'])))))

        self.textEdit_summary_data.setText('\n'.join([col_labels,str(self.hk_list),str(self.potentials)]))

    #to fold or unfold the config file editor
    def fold_or_unfold(self):
        text = self.PushButton_fold_or_unfold.text()
        if text == "<":
            self.frame.setVisible(False)
            self.PushButton_fold_or_unfold.setText(">")
        elif text == ">":
            self.frame.setVisible(True)
            self.PushButton_fold_or_unfold.setText("<")

    def plot_figure(self):
        if self.radioButton_ctr.isChecked():
            self.plot_figure_ctr()
        elif self.radioButton_ctr_model.isChecked():
            self.plot_figure_ctr_model()

    def plot_figure_ctr_model(self):
        self.mplwidget.fig.clear()
        col_num=2#two columns only
        if len(self.hk_list) in [1,2]:
            col_num = 1
        else:
            pass

        plot_dim = [int(len(self.hk_list)/col_num)+int(len(self.hk_list)%col_num != 0), col_num]
        for i in range(len(self.hk_list)):
            setattr(self,'plot_axis_plot_set{}'.format(i+1),self.mplwidget.canvas.figure.add_subplot(plot_dim[0], plot_dim[1],i+1))
            current_hk = self.hk_list[i]
            current_maximum = 0
            offset_list = self.lineEdit_ctr_offset.text().rstrip().rsplit('+')[i]
            for each_potential in self.potentials:
                #current_offset = float(offset_list.rstrip().rsplit(',')[self.potentials.index(each_potential)])
                offset = np.multiply.accumulate([float(each) for each in offset_list.rstrip().rsplit(',')])[self.potentials.index(each_potential)]
                index_ = (self.data['potential']==each_potential)&(self.data['H']==current_hk[0])&(self.data['K']==current_hk[1])
                if len(index_)==0:
                    index_ = (self.data['potential']==each_potential)&(self.data['H']==current_hk[1])&(self.data['K']==current_hk[0])
                x = self.data[index_]['L']
                y_data=self.data[index_]['I']
                y_model = self.data[index_]['I_model']
                y_ideal = np.array(self.data[index_]['I_bulk'])
                
                if each_potential == self.potentials[0]:
                    # current_minimum = np.min(y_ideal)
                    current_maximum = np.max(y_ideal)
                    # getattr(self,'plot_axis_plot_set{}'.format(i+1)).plot(x,y_ideal,color ='k',linestyle ='--', label = 'Unrelaxed structure')
                else:
                    pass
                error = self.data[index_]['error']
                use = list(self.data[index_]['use'])[0]
                fmt = self.lineEdit_fmt.text().rstrip().rsplit(',')[self.potentials.index(each_potential)]
                
                #np.min(y_ideal)
                # scale_factor = offset*current_minimum/np.min(y_data)
                scale_factor = offset*current_maximum/np.max(y_ideal)
                #if each_potential == self.potentials[0]:
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).plot(x,y_ideal*scale_factor,color ='0.5',linestyle ='-', label = [None,'Unrelaxed structure'][int(each_potential == self.potentials[0])])
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).scatter(x,y_data*scale_factor,s = 8, marker = 'o',c=fmt[-1], label = ['Data'+str(each_potential)+'V w.r.t Ag/AgCl',None][int(use)])
                if use:
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).plot(x,y_model*scale_factor,fmt, label = 'Fit '+str(each_potential)+'V w.r.t Ag/AgCl')
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).fill_between(x,y_ideal*scale_factor,y_model*scale_factor,color = fmt[-1],alpha = 0.5)
                if i in [0,2]:
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_ylabel('F',fontsize=10)
                if i in [2,3]:
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_xlabel('L(r.l.u)',fontsize=10)
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_title('{}{}L'.format(*current_hk))
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_yscale('log')
                if i in [0,2]:#manually set this accordingly
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).legend()
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).autoscale()
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).tick_params(axis='both',which='major',labelsize = 10)
        self.mplwidget.fig.tight_layout()
        # self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

    def plot_figure_ctr(self):
        self.mplwidget.fig.clear()
        col_num=2#two columns only
        self.data_to_save = {}
        if len(self.scan_numbers_all.text().rsplit('+')) in [1,2]:
            col_num = 1
        else:
            pass

        plot_dim = [int(len(self.scan_numbers_all.text().rsplit('+'))/col_num)+int(len(self.scan_numbers_all.text().rsplit('+'))%col_num != 0), col_num]
        for i in range(len(self.scan_numbers_all.text().rsplit('+'))):
            setattr(self,'plot_axis_plot_set{}'.format(i+1),self.mplwidget.canvas.figure.add_subplot(plot_dim[0], plot_dim[1],i+1))
            each = self.scan_numbers_all.text().rsplit('+')[i]
            each = each[1:-1]#remove []
            scan_list_temp = each.rsplit(';')
            for each_set in scan_list_temp:
                j = scan_list_temp.index(each_set)
                sub_scan_list = each_set.rsplit(',')
                scans_temp= [int(i) for i in sub_scan_list]
                for scan in scans_temp:
                    # fmt = self.lineEdit_fmt.text().rsplit('+')[i].rsplit(';')[j]
                    fmt = self.lineEdit_fmt.text().rsplit(';')[j]
                    if scan == scans_temp[0]:
                        # label = self.lineEdit_labels.text().rsplit('+')[i].rsplit(';')[j]
                        label = self.lineEdit_labels.text().rsplit(';')[j]
                    else:
                        label = None
                    #append data to save
                    map_BL = {'00':0,'20':0,'11':1,'13':1,'31':1}
                    try:
                        BL=map_BL['{}{}'.format(int(round(self.data_to_plot[scan]['H'][0],0)),int(round(self.data_to_plot[scan]['K'][0],0)))]
                    except:
                        BL = 0
                    temp_key = self.lineEdit_labels.text().rsplit(';')[j]
                    if '[' in temp_key:
                        temp_key = temp_key[1:]
                    if ']' in temp_key:
                        temp_key = temp_key[0:len(temp_key)-1]
                    if temp_key not in self.data_to_save.keys():
                        self.data_to_save[temp_key] = pd.DataFrame(np.zeros([1,8])[0:0],columns=["L","H","K","na","I","I_err","BL","dL"])
                    else:
                        pass
                    len_data = len(self.data_to_plot[scan]['L'])
                    #Lorentz factor calculation, refer to Vlieg 1997, J.Appl.Cryst. Equation 16
                    lorentz_ft = np.sin(np.deg2rad(self.data_to_plot[scan]['delta']))*np.cos(np.deg2rad(self.data_to_plot[scan]['omega']))*np.cos(np.deg2rad(self.data_to_plot[scan]['gamma']))
                    #footprint area correction factor (take effect only for specular rod)
                    area_ft = np.sin(np.deg2rad(self.data_to_plot[scan]['omega']))
                    self.data_to_save[temp_key] = self.data_to_save[temp_key].append(pd.DataFrame({"L":self.data_to_plot[scan]['L'],"H":self.data_to_plot[scan]['H'],\
                                                                                     "K":self.data_to_plot[scan]['K'],"na":[0]*len_data,"I":self.data_to_plot[scan]['peak_intensity']*lorentz_ft*area_ft,\
                                                                                     "I_err":self.data_to_plot[scan]['peak_intensity_error']*lorentz_ft*area_ft,"BL":[BL]*len_data ,"dL":[2]*len_data}))
                    #remove [ or ] in the fmt and label
                    if ('[' in fmt) and (']' in fmt):
                        fmt = fmt[1:-1]
                    elif '[' in fmt:
                        fmt = fmt[1:]
                    elif ']' in fmt:
                        fmt = fmt[0:-1]
                    else:
                        pass
                    if label != None:
                        if ('[' in label) and (']' in label):
                            label = label[1:-1]
                        elif '[' in label:
                            label = label[1:]
                        elif ']' in label:
                            label = label[0:-1]
                        else:
                            pass
                    if self.checkBox_time_scan.isChecked():
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).plot(self.data_to_plot[scan][self.plot_label_x],self.data_to_plot[scan][self.plot_labels_y[0]]*lorentz_ft*area_ft,fmt,label =label)
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_xlabel(self.plot_label_x)
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_ylabel('Intensity')
                        pot_ax = getattr(self,'plot_axis_plot_set{}'.format(i+1)).twinx()
                        pot_ax.plot(self.data_to_plot[scan][self.plot_label_x],self.data_to_plot[scan][self.plot_labels_y[1]]*lorentz_ft*area_ft,'b-',label = None)
                        pot_ax.set_ylabel(self.plot_labels_y[1],color = 'b')
                        pot_ax.tick_params(axis = 'y', labelcolor = 'b')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).legend()
                    else:
                        # getattr(self,'plot_axis_plot_set{}'.format(i+1)).plot(self.data_to_plot[scan][self.plot_label_x],self.data_to_plot[scan][self.plot_labels_y[0]]*lorentz_ft*area_ft,fmt,label =label)
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).errorbar(self.data_to_plot[scan][self.plot_label_x],self.data_to_plot[scan]['peak_intensity']*lorentz_ft*area_ft,yerr=self.data_to_plot[scan]['peak_intensity_error']*lorentz_ft*area_ft,fmt=fmt,label =label)
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_ylabel('Intensity')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_xlabel('L')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_title('{}{}L'.format(int(round(list(self.data[self.data['scan_no']==scan]['H'])[0],0)),int(round(list(self.data[self.data['scan_no']==scan]['K'])[0],0))))
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_yscale('log')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).legend()
        self.mplwidget.fig.tight_layout()
        self.mplwidget.canvas.draw()

    #this is temporary func to plot the l positions for different Bragg peaks (Co3O4 and CoOOH peaks on 01L of Au_hex)
    #eg for info info = {'Au_hex': [[1.9999999999999998, '(0, 1, 2)'], [5.0, '(0, 1, 5)'], [7.999999999999999, '(0, 1, 8)']], 
    # 'Co3O4': [[1.0089863209787204, '(2, -2, 2)'], [2.5224658024468014, '(3, -1, 3)'], [4.035945283914882, '(4, 0, 4)'], 
    # [5.5494247653829625, '(5, 1, 5)'], [7.062904246851043, '(6, 2, 6)'], [8.576383728319124, '(7, 3, 7)']], 
    # 'Co3O4_R60': [[0.5044931604893608, '(3, -1, -1)'], [2.0179726419574417, '(4, 0, 0)'], [3.5314521234255225, '(5, 1, 1)'], 
    # [5.044931604893603, '(6, 2, 2)'], [6.558411086361683, '(7, 3, 3)'], [8.071890567829763, '(8, 4, 4)']], 
    # 'CoOOH': [[1.0743117870722434, '(0, 1, 2)'], [2.685779467680608, '(0, 1, 5)'], [4.297247148288973, '(0, 1, 8)']], 
    # 'CoOOH_R60': [[0.5371558935361217, '(1, 0, 1)'], [2.148623574144487, '(1, 0, 4)'], [3.7600912547528518, '(1, 0, 7)'], 
    # [5.371558935361216, '(1, 0, 10)']]}  
    #each item contains l value wrt Au_hex, hkl str for that peak
    def plot_miller_index_temp(self, key = '0.4 V', ax_name = 'plot_axis_plot_set1', color_map = {'CoOOH':'blue','CoOOH_R60':'blue','Co3O4':'red','Co3O4_R60':'red','Au_hex':'m'}):
        info = {'Au_hex': [[1.9999999999999998, '(0, 1, 2)'], [5.0, '(0, 1, 5)'], [7.999999999999999, '(0, 1, 8)']], 
            'Co3O4': [[1.0089863209787204, '(2, -2, 2)'], [2.5224658024468014, '(3, -1, 3)'], [4.035945283914882, '(4, 0, 4)'], 
            [5.5494247653829625, '(5, 1, 5)'], [7.062904246851043, '(6, 2, 6)'], [8.576383728319124, '(7, 3, 7)']], 
            'Co3O4_R60': [[0.5044931604893608, '(3, -1, -1)'], [2.0179726419574417, '(4, 0, 0)'], [3.5314521234255225, '(5, 1, 1)'], 
            [5.044931604893603, '(6, 2, 2)'], [6.558411086361683, '(7, 3, 3)'], [8.071890567829763, '(8, 4, 4)']], 
            'CoOOH': [[1.0743117870722434, '(0, 1, 2)'], [2.685779467680608, '(0, 1, 5)'], [4.297247148288973, '(0, 1, 8)']], 
            'CoOOH_R60': [[0.5371558935361217, '(1, 0, 1)'], [2.148623574144487, '(1, 0, 4)'], [3.7600912547528518, '(1, 0, 7)'], 
            [5.371558935361216, '(1, 0, 10)']]}  
        # set the font size of tick lable first
        getattr(self,ax_name).xaxis.label.set_size(20)
        getattr(self,ax_name).yaxis.label.set_size(20)
        for tick in getattr(self,ax_name).xaxis.get_major_ticks():
            tick.label.set_fontsize(20)  
        for tick in getattr(self,ax_name).yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        def _format_hkl(hkl):
            h,k,l = hkl
            if h>=0:
                h_str = str(h)
            else:
                h_str = r'$\overline{{{}}}$'.format(abs(h))
            if k>=0:
                k_str = str(k)
            else:
                k_str = r'$\overline{{{}}}$'.format(abs(k))
            if l>=0:
                l_str = str(l)
            else:
                l_str = r'$\overline{{{}}}$'.format(abs(l))
            if int(l)>=10:
                l_str = ' '+l_str
            return ' ('+h_str+k_str+l_str+')'

        for each in info:
            peaks = info[each]
            name = each
            
            for each_peak in peaks:
                l, hkl = each_peak
                if 'R60' in name:
                    name = name[0:5]
                    hkl = eval(hkl)
                    hkl = (hkl[1],hkl[0],hkl[2])
                else:
                    hkl = eval(hkl)
                int_high_end = self.data_to_save[key]['I'][np.argmin(abs(self.data_to_save[key]['L']-l))]
                int_low_end = 0.0005
                if name!='Au_hex' and l<5.6:
                # if l<5.6:
                    if 'CoOOH' in name:
                        getattr(self,ax_name).plot([l-0.045,l-0.045],[int_low_end,self.data_to_save[key]['I'][np.argmin(abs(self.data_to_save[key]['L']-(l-0.045)))]],color = color_map[each])
                        if hkl==(0,1,2):
                            getattr(self,ax_name).text(l-0.045,self.data_to_save[key]['I'][np.argmin(abs(self.data_to_save[key]['L']-(l-0.045)))]+0.06,'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
                        elif hkl==(0,1,1):
                            getattr(self,ax_name).text(l-0.045-0.12,self.data_to_save[key]['I'][np.argmin(abs(self.data_to_save[key]['L']-(l-0.045)))],'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
                        else:
                            getattr(self,ax_name).text(l-0.045,self.data_to_save[key]['I'][np.argmin(abs(self.data_to_save[key]['L']-(l-0.045)))],'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
                    else:
                        getattr(self,ax_name).plot([l,l],[int_low_end,int_high_end],color = color_map[each])
                        if hkl==(2,-2,2):
                            getattr(self,ax_name).text(l-0.22,int_high_end,'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
                        elif hkl==(3,-1,3):
                            getattr(self,ax_name).text(l-0.12,int_high_end,'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
                        elif hkl in [(4,0,0),(6,2,2)]:
                            getattr(self,ax_name).text(l+0.06,int_high_end,'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
                        else:
                            getattr(self,ax_name).text(l,int_high_end,'{}{}'.format(' ',_format_hkl(hkl)),rotation ='vertical',color = color_map[each],fontsize = 20)
        getattr(self,ax_name).text(2,696,'Au',fontsize = 20)
        getattr(self,ax_name).text(5,368,'Au', fontsize = 20)
        getattr(self,ax_name).set_ylim(0.0005,2000)
        getattr(self,ax_name).set_xlim(0.,5.8)
        getattr(self,ax_name).get_legend().remove()
        getattr(self,ax_name).tick_params(which = 'major', axis="x", direction="in")
        getattr(self,ax_name).tick_params(which = 'minor', axis="x", direction="in")
        getattr(self,ax_name).tick_params(which = 'major', axis="y", direction="in")
        getattr(self,ax_name).tick_params(which = 'minor', axis="y", direction="in")
        getattr(self,ax_name).tick_params(which = 'major', bottom=True, top=True, left=True, right=True)
        getattr(self,ax_name).tick_params(which = 'minor', bottom=True, top=True, left=True, right=True)
        getattr(self,ax_name).tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        getattr(self,ax_name).xaxis.set_minor_locator(AutoMinorLocator())
        getattr(self,ax_name).set_title('')
        self.mplwidget.canvas.draw()
        self.mplwidget.canvas.figure.savefig('/Users/canrong/Documents/Co oxide project/ctr_profile.png',dpi = 300)

    def prepare_data_to_plot_ctr(self,plot_label_list, scan_number):
        scans_ = [int(each) for each in self.lineEdit_scan_numbers.text().rsplit(',')]
        img_ranges_ = [[int(each_) for each_ in each.rsplit('-')] for each in self.lineEdit_point_ranges.text().rsplit(',')]
        
        if scan_number in scans_:
            which = scans_.index(scan_number)
            l , r = img_ranges_[which]
        else:
            l, r = 0, 100000000
        if hasattr(self,'data_to_plot'):
            self.data_to_plot[scan_number] = {}
        else:
            self.data_to_plot = {}
            self.data_to_plot[scan_number] = {}
        if self.checkBox_mask.isChecked():
            condition = (self.data['mask_ctr'] == True)&(self.data['scan_no'] == scan_number)
        else:
            condition = self.data['scan_no'] == scan_number
        if 'gamma' not in plot_label_list:
            plot_label_list.append('gamma')
        
        if 'delta' not in plot_label_list:
            plot_label_list.append('delta')

        if 'omega_t' not in plot_label_list:
            plot_label_list.append('omega_t')

        if 'omega' not in plot_label_list:
            plot_label_list.append('omega')

        if 'peak_intensity_error' not in plot_label_list:
            plot_label_list.append('peak_intensity_error')

        for each in plot_label_list:
            if each=='current':#RHE potential
                self.data_to_plot[scan_number][each] = -np.array(self.data[condition][each])[l:r]
                #self.data_to_plot[scan_number][each] = 0.205+np.array(self.data[condition][each])[l:r]+0.059*np.array(self.data[self.data['scan_no'] == scan_number]['phs'])[0]               
            else:
                self.data_to_plot[scan_number][each] = np.array(self.data[condition][each])[l:r]

    def append_scans(self):
        self.append_scans_ctr()

    #the text looks like: [1,2,3;4,5,6]+[7,8,9;10,11,12]+...
    def append_scans_ctr(self):
        text = self.scan_numbers_append.text()
        text_original = self.scan_numbers_all.text()
        if (text_original!='') and (text not in text_original):
            text_new = '+'.join([text_original, text])
        else:
            text_new = text
        scans = []
        for each in text_new.rstrip().rsplit('+'):
            each = each[1:-1]#remove []
            scan_list_temp = each.rstrip().rsplit(';')
            for each_set in scan_list_temp:
                sub_scan_list = each_set.rstrip().rsplit(',')
                scans+= [int(i) for i in sub_scan_list]

        # scans = list(set([int(each) for each in text_new.rstrip().rsplit('+')]))
        #scans.sort()
        self.scan_numbers_all.setText(text_new)
        assert (self.lineEdit_x.text()!='' and self.lineEdit_y.text()!=''), 'No channels for plotting have been selected!'
        assert self.scan_numbers_all.text()!='', 'No scans have been selected!'
        plot_labels = self.lineEdit_x.text() + ',' + self.lineEdit_y.text()
        plot_labels = plot_labels.rstrip().rsplit(',')
        for scan in scans:
            self.prepare_data_to_plot_ctr(plot_labels,scan)
            print('Prepare data for scan {} now!'.format(scan))
        self.scans = scans
        #self.phs = [self.phs_all[self.scans_all.index(each)] for each in scans]
        self.plot_label_x = self.lineEdit_x.text()
        self.plot_labels_y = self.lineEdit_y.text().rstrip().rsplit(',')


if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())