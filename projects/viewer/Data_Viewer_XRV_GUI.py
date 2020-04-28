import sys,os,qdarkstyle
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
matplotlib.use("Qt5Agg")
from scipy import signal
# import scipy.signal.savgol_filter as savgol_filter

#from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

class MyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        uic.loadUi(os.path.join(DaFy_path,'projects','viewer','data_viewer__xrv_new.ui'),self)
        # self.setupUi(self)
        # plt.style.use('ggplot')
        self.setWindowTitle('XRV data Viewer')
        self.data_to_save = {}
        self.image_range_info = {}
        self.lineEdit_data_file_name.setText(os.path.join(DaFy_path,'dump_files','temp_xrv.csv'))
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
        self.open.clicked.connect(self.load_file)
        self.actionPlotData.triggered.connect(self.plot_figure_xrv)
        self.actionPlotRate.triggered.connect(self.plot_data_summary_xrv)
        self.actionSaveData.triggered.connect(self.save_xrv_data)
        self.PushButton_append_scans.clicked.connect(self.append_scans_xrv)
        self.checkBox_time_scan.clicked.connect(self.set_plot_channels)
        self.checkBox_mask.clicked.connect(self.append_scans_xrv)
        self.pushButton_load_config.clicked.connect(self.load_config)
        self.pushButton_save_config.clicked.connect(self.save_config)
        self.pushButton_update.clicked.connect(self.update_plot_range)
        self.pushButton_update.clicked.connect(self.append_scans_xrv)
        self.pushButton_update.clicked.connect(self.plot_figure_xrv)
        #self.pushButton_save_data.clicked.connect(self.save_data_method)
        #self.pushButton_save_xrv_data.clicked.connect(self.save_xrv_data)
        #self.pushButton_plot_datasummary.clicked.connect(self.plot_data_summary_xrv)
        self.data = None
        self.data_summary = {} 
        self.data_range = None
        self.pot_range = None
        self.potential = []
       
    def update_plot_range(self):
        scan = int(self.comboBox_scans.currentText())
        l,r = int(self.lineEdit_img_range_left.text()),int(self.lineEdit_img_range_right.text())
        self.image_range_info[scan] = [l,r]
        all_info=[]
        for each in self.image_range_info:
            all_info.append('{}:{}'.format(each,self.image_range_info[each]))
        self.plainTextEdit_img_range.setPlainText('\n'.join(all_info))

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
            df.to_csv(self.lineEdit_data_file_path.text().replace('.csv','_{}.csv'.format(scan_)),\
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
            
    def set_plot_channels(self):
        time_scan = self.checkBox_time_scan.isChecked()
        if time_scan:
            self.lineEdit_x.setText('image_no')
            self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop')
        else:
            self.lineEdit_x.setText('potential')
            self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop') 

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
        scans.sort()
        scan_numbers = 'scan_nos\n'+str(scans)+'\n'
        # print(list(self.data[self.data['scan_no']==scans[0]]['phs'])[0])
        self.phs_all = [list(self.data[self.data['scan_no']==scan]['phs'])[0] for scan in scans]
        phs = 'pHs\n'+str(self.phs_all)+'\n'
        self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers,phs]))
        self.image_range_info = {}
        self.plainTextEdit_img_range.setPlainText("")

    def plot_data_summary_xrv_from_external_file(self):
        if self.data_summary!={}:
            self.mplwidget2.fig.clear()
            y_label_map = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\Delta\varepsilon_\parallel$  (%/V)',
                        'strain_oop':r'$\Delta\varepsilon_\perp$  (%/V)',
                        'grain_size_oop':r'$\Delta d_\perp$  (nm/V)',
                        'grain_size_ip':r'$\Delta d_\parallel$  (nm/V)',
                        'peak_intensity':r'Intensity / a.u.'}
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
            lim_y_temp = {'strain_ip':-0.18,'strain_oop':-0.4,'grain_size_ip':-1.2,'grain_size_oop':-1.4}
            for each in plot_y_labels:
                for i in range(int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)):
                    # plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)],self.data_summary[each_scan][each][-1]] for each_scan in self.scans])
                    plot_data_y = np.array([[self.data_summary[each_scan][each][i*2],self.data_summary[each_scan][each][i*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    labels = ['pH {}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), int(len(self.data_summary[self.scans[0]]['strain_ip'])/2), i+1+int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)*plot_y_labels.index(each))
                    ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
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
                        ax_temp.set_xticklabels(labels,fontsize=13)
                    if i!=0:
                        ax_temp.set_yticklabels([])
                        
                    # ax_temp.set_xticklabels(plot_data_x,labels)
            self.mplwidget2.fig.subplots_adjust(hspace=0.04)
            self.mplwidget2.canvas.draw()
        else:
            pass

    def make_data_summary_from_external_file(self,file = '/Users/cqiu/app/DaFy/dump_files/superrod_fit_results.csv'):
        data = pd.read_csv(file,sep='\t')
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

    def plot_data_summary_xrv(self,use_external = True):
        print("starting from here!")
        if use_external:
            self.make_data_summary_from_external_file()
            self.plot_data_summary_xrv_from_external_file()
            print("new_data summary is built!")
            return 
        if self.data_summary!={}:
            self.mplwidget2.fig.clear()
            y_label_map = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\Delta\varepsilon_\parallel$  (%/V)',
                        'strain_oop':r'$\Delta\varepsilon_\perp$  (%/V)',
                        'grain_size_oop':r'$\Delta d_\perp$  (nm/V)',
                        'grain_size_ip':r'$\Delta d_\parallel$  (nm/V)',
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
            #TODO this has to be changed to set the y_max automatically in different cases.
            lim_y_temp = {'strain_ip':-0.18,'strain_oop':-0.4,'grain_size_ip':-1.2,'grain_size_oop':-1.4}
            for each in plot_y_labels:
                for each_pot in self.pot_range:
                    # plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)],self.data_summary[each_scan][each][-1]] for each_scan in self.scans])
                    plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2],self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    labels = ['pH {}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), len(self.pot_range), self.pot_range.index(each_pot)+1+len(self.pot_range)*plot_y_labels.index(each))
                    ax_temp.bar(plot_data_x,-plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                    if each_pot == self.pot_range[0]:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=13)
                        ax_temp.set_ylim([lim_y_temp[each],0])
                    else:
                        ax_temp.set_ylim([lim_y_temp[each],0])
                    if each == plot_y_labels[0]:
                        ax_temp.set_title('E range:{:4.2f}-->{:4.2f} V'.format(*each_pot), fontsize=13)
                    if each != plot_y_labels[-1]:
                        ax_temp.set_xticklabels([])
                    else:
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=13)
                    if each_pot!=self.pot_range[0]:
                        ax_temp.set_yticklabels([])
                        
                    # ax_temp.set_xticklabels(plot_data_x,labels)
            self.mplwidget2.fig.subplots_adjust(hspace=0.04)
            self.mplwidget2.canvas.draw()
        else:
            pass

    def return_seperator_values(self,file = '/Users/cqiu/app/DaFy/dump_files/superrod_fit_results.csv'):
        data = pd.read_csv(file,sep='\t')
        summary = {}
        for each in self.scans:
            summary[each] = {}
            for_current =[]
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                summary[each][item] = [data['scan{}'.format(each)]['{}_p1'.format(item)]+self.potential_offset]
                for_current.append(summary[each][item])
            summary[each]['current'] = for_current
        return summary

    def return_slope_values(self,file = '/Users/cqiu/app/DaFy/dump_files/superrod_fit_results.csv'):
        data = pd.read_csv(file,sep='\t')
        summary = {}
        for each in self.scans:
            summary[each] = {}
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                summary[each][item] = [data['scan{}'.format(each)]['{}_p{}'.format(item,i)]+self.potential_offset for i in range(3)]
                summary[each][item] = summary[each][item] + [data['scan{}'.format(each)]['{}_y1'.format(item)]]
                summary[each][item] = summary[each][item] +  list(data['scan{}'.format(each)]['{}'.format(item)])[::-1]
        #each item = [p0,p1,p2,y1,a1,a2], as are slope values, (p1,y1) transition value, y0 and y2 are end points for potentials
        return summary

    def plot_figure_xrv(self):
        #slope_info_temp = self.return_slope_values()
        self.mplwidget.fig.clear()
        plot_dim = [len(self.plot_labels_y), len(self.scans)]
        for scan in self.scans:
            setattr(self,'plot_axis_scan{}'.format(scan),[])
            j = self.scans.index(scan) + 1
            for i in range(plot_dim[0]):
                getattr(self,'plot_axis_scan{}'.format(scan)).append(self.mplwidget.canvas.figure.add_subplot(plot_dim[0], plot_dim[1],j+plot_dim[1]*i))
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
        pot_range = [list(map(float,each.rsplit('-'))) for each in pot_range]
        self.pot_range = pot_range

        for scan in self.scans:
            self.data_summary[scan] = {}
            if 'potential' in self.plot_labels_y and self.plot_label_x == 'potential':
                plot_labels_y = [each for each in self.plot_labels_y if each!='potential']
            else:
                plot_labels_y = self.plot_labels_y
            for each in plot_labels_y:
                self.data_summary[scan][each] = []
                i = plot_labels_y.index(each)
                try:
                    fmt = self.lineEdit_fmt.text().rsplit(',')[self.scans.index(scan)]
                except:
                    fmt = 'b-'
                y = self.data_to_plot[scan][plot_labels_y[i]]
                y_smooth_temp = signal.savgol_filter(self.data_to_plot[scan][plot_labels_y[i]],41,2)
                std_val = np.sum(np.abs(y_smooth_temp - y))/len(self.data_to_plot[scan][self.plot_label_x])
                marker_index_container = []
                for ii in range(len(self.pot_range)):
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
                    self.data_summary[scan][each].append((y_smooth_temp[index_left] - y_smooth_temp[index_right])/pot_offset)
                    self.data_summary[scan][each].append(std_val/pot_offset)
                    #self.data_summary[scan][each].append((y_smooth_temp[index_left] - y_smooth_temp[index_right])/abs(self.data_to_plot[scan]['potential'][index_left]-self.data_to_plot[scan]['potential'][index_right]))
                    #last value is the standard deviation
                    #self.data_summary[scan][each].append(std_val/abs(self.data_to_plot[scan]['potential'][index_left]-self.data_to_plot[scan]['potential'][index_right]))
                # print(scan, each, 'standard deviation = ',std_val)
                
                #plot the results
                if self.plot_label_x == 'image_no':
                    if each != 'current':
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y,fmt,markersize = 3)
                        # getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y_smooth_temp,'-')
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([np.arange(len(y))[iii] for iii in marker_index_container],[y_smooth_temp[iii] for iii in marker_index_container],'k*')
                    else:
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y,fmt,markersize = 3)
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([np.arange(len(y))[iii] for iii in marker_index_container],[y[iii] for iii in marker_index_container],'k*')
                else:
                    try:
                        seperators = self.return_seperator_values()
                    except:
                        seperators = []
                    if each!='current':
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y,fmt,markersize = 2)
                        #p0,p1,p2,y1,a1,a2 = slope_info_temp[scan][each]
                        #y0 = a1*(p0-p1)+y1
                        #y2 = a2*(p2-p1)+y1
                        #getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([p0,p1,p2],np.array([y0,y1,y2])-self.data_to_plot[scan][each+"_max"],'k--')
                        # getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y_smooth_temp,'-')
                        #getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y_smooth_temp[iii] for iii in marker_index_container],'k*')
                        # for pot in np.arange(1.0,1.8,0.1):
                        if len(seperators)!=0:
                            try:
                                for pot in seperators[scan][each]:
                                    getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([pot,pot],[-100,100],'k:')
                            except:
                                pass
                    else:
                        # lim_y = data_viewer_plot_cv(getattr(self,'plot_axis_scan{}'.format(scan))[i],scan,seperators[scan][each])
                        try:
                            lim_y = data_viewer_plot_cv(getattr(self,'plot_axis_scan{}'.format(scan))[i],scan,np.arange(1.0,1.8,0.1))
                            #getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y*8,fmt+'-',markersize = 3)
                            #getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y[iii]*8 for iii in marker_index_container],'k*')
                        except:
                            getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y,fmt,markersize = 3)
                            getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y[iii] for iii in marker_index_container],'k*')
                if i==0:
                    # getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}_scan{}'.format(self.phs[self.scans.index(scan)],scan),fontsize=13)
                    getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}'.format(self.phs[self.scans.index(scan)],scan),fontsize=13)
                if each=='current':
                    try:
                        temp_min,temp_max = lim_y
                    except:
                        temp_max, temp_min = max(list(self.data_to_plot[scan][plot_labels_y[i]])),min(list(self.data_to_plot[scan][plot_labels_y[i]]))
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
        for scan in self.scans:
            for each in self.plot_labels_y:
                i = self.plot_labels_y.index(each)
                if 'current' not in self.plot_labels_y:
                    break
                else:
                    pass
                j = self.plot_labels_y.index('current')
                if each != 'current':#synchronize the x_lim of all non-current to that for CV
                    x_lim = getattr(self,'plot_axis_scan{}'.format(scan))[j].get_xlim()
                    getattr(self,'plot_axis_scan{}'.format(scan))[i].set_xlim(*x_lim)
                else:
                    pass
                getattr(self,'plot_axis_scan{}'.format(scan))[i].set_ylim(y_min_values[i],y_max_values[i])
        self.mplwidget.fig.tight_layout()
        # print(self.data_summary)
        self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

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

if __name__ == "__main__":
    QApplication.setStyle("fusion")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())