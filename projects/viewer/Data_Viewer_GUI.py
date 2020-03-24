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
        uic.loadUi(os.path.join(DaFy_path,'scripts','Viewer','data_viewer_new.ui'),self)
        # self.setupUi(self)
        # plt.style.use('ggplot')
        self.setWindowTitle('XRV and CTR data Viewer')
        self.set_plot_channels()
        self.data_to_save = {}
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
        self.plot.clicked.connect(self.plot_figure)
        # self.apply.clicked.connect(self.replot_figure)
        self.PushButton_append_scans.clicked.connect(self.append_scans)
        self.pushButton_filePath.clicked.connect(self.locate_data_folder)
        self.PushButton_fold_or_unfold.clicked.connect(self.fold_or_unfold)
        self.checkBox_time_scan.clicked.connect(self.set_plot_channels)
        self.radioButton_ctr.clicked.connect(self.set_plot_channels)
        self.radioButton_xrv.clicked.connect(self.set_plot_channels)
        self.checkBox_mask.clicked.connect(self.append_scans)
        self.pushButton_load_config.clicked.connect(self.load_config)
        self.pushButton_save_config.clicked.connect(self.save_config)
        self.pushButton_save_data.clicked.connect(self.save_data_method)
        self.pushButton_plot_datasummary.clicked.connect(self.plot_data_summary_xrv)
        self.data = None
        self.data_summary = {} 
        self.data_range = None
        self.pot_range = None
        self.potential = []
       
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
        xrv = self.radioButton_xrv.isChecked()
        time_scan = self.checkBox_time_scan.isChecked()
        if xrv:
            if time_scan:
                self.lineEdit_x.setText('image_no')
                self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop')
            else:
                self.lineEdit_x.setText('potential')
                self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop') 
        else:
            if time_scan:
                self.lineEdit_x.setText('image_no')
                self.lineEdit_y.setText('peak_intensity,potential')
            else:
                self.lineEdit_x.setText('L')
                self.lineEdit_y.setText('peak_intensity') 

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

    def load_file_ctr_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;All Files (*.csv)", options=options)
        if fileName:
            self.lineEdit_data_file.setText(fileName)
            self.data = pd.read_excel(fileName)
        col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
        self.potentials = list(set(list(self.data['potential'])))
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
        if self.radioButton_xrv.isChecked():
            self.plot_figure_xrv()
        elif self.radioButton_ctr.isChecked():
            self.plot_figure_ctr()
        elif self.radioButton_ctr_model.isChecked():
            self.plot_figure_ctr_model()

    def plot_data_summary_xrv(self):
        if self.radioButton_xrv.isChecked() and self.plot_data_summary_xrv!={}:
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
                assert len(colors_bar) == len(self.scans)
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

    def plot_figure_xrv(self):
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
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y_smooth_temp,'-')
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([np.arange(len(y))[iii] for iii in marker_index_container],[y_smooth_temp[iii] for iii in marker_index_container],'k*')
                    else:
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(np.arange(len(y)),y,fmt+'-',markersize = 3)
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([np.arange(len(y))[iii] for iii in marker_index_container],[y[iii] for iii in marker_index_container],'k*')
                else:
                    if each!='current':
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y,fmt,markersize = 3)
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y_smooth_temp,'-')
                        # print([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container])
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y_smooth_temp[iii] for iii in marker_index_container],'k*')
                    else:
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot(self.data_to_plot[scan][self.plot_label_x],y,fmt+'-',markersize = 3)
                        getattr(self,'plot_axis_scan{}'.format(scan))[i].plot([self.data_to_plot[scan][self.plot_label_x][iii] for iii in marker_index_container],[y[iii] for iii in marker_index_container],'k*')
                if i==0:
                    # getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}_scan{}'.format(self.phs[self.scans.index(scan)],scan),fontsize=13)
                    getattr(self,'plot_axis_scan{}'.format(scan))[i].set_title(r'pH {}'.format(self.phs[self.scans.index(scan)],scan),fontsize=13)
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
                getattr(self,'plot_axis_scan{}'.format(scan))[i].set_ylim(y_min_values[i],y_max_values[i])
        self.mplwidget.fig.tight_layout()
        # print(self.data_summary)
        self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

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
                if i in [0,2]:
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_ylabel('F',fontsize=15)
                if i in [2,3]:
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_xlabel('L(r.l.u)',fontsize=15)
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_title('{}{}L'.format(*current_hk))
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_yscale('log')
                if i in [0,2]:#manually set this accordingly
                    getattr(self,'plot_axis_plot_set{}'.format(i+1)).legend()
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).autoscale()
                getattr(self,'plot_axis_plot_set{}'.format(i+1)).tick_params(axis='both',which='major',labelsize = 14)
        plt.tight_layout()



    def plot_figure_ctr(self):
        self.mplwidget.fig.clear()
        col_num=2#two columns only
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
                    fmt = self.lineEdit_fmt.text().rsplit('+')[i].rsplit(';')[j]
                    if scan == scans_temp[0]:
                        label = self.lineEdit_labels.text().rsplit('+')[i].rsplit(';')[j]
                    else:
                        label = None
                    #append data to save
                    map_BL = {'00':0,'20':0,'11':1,'13':1,'31':1}
                    try:
                        BL=map_BL['{}{}'.format(int(round(self.data_to_plot[scan]['H'][0],0)),int(round(self.data_to_plot[scan]['K'][0],0)))]
                    except:
                        BL = 0
                    temp_key = self.lineEdit_labels.text().rsplit('+')[i].rsplit(';')[j]
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
                    lorentz_ft = np.sin(np.deg2rad(self.data_to_plot[scan]['delta']))*np.cos(np.deg2rad(self.data_to_plot[scan]['omega_t']))*np.cos(np.deg2rad(self.data_to_plot[scan]['gamma']))
                    #footprint area correction factor (take effect only for specular rod)
                    area_ft = np.sin(np.deg2rad(self.data_to_plot[scan]['omega_t']))
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
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).plot(self.data_to_plot[scan][self.plot_label_x],self.data_to_plot[scan][self.plot_labels_y[0]]*lorentz_ft*area_ft,fmt,label =label)
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_ylabel('Intensity')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_xlabel('L')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_title('{}{}L'.format(int(round(list(self.data[self.data['scan_no']==scan]['H'])[0],0)),int(round(list(self.data[self.data['scan_no']==scan]['K'])[0],0))))
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).set_yscale('log')
                        getattr(self,'plot_axis_plot_set{}'.format(i+1)).legend()
        self.mplwidget.fig.tight_layout()
        self.mplwidget.canvas.draw()

    def prepare_data_to_plot_xrv(self,plot_label_list, scan_number):
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
            condition = (self.data['mask_cv_xrd'] == True)&(self.data['scan_no'] == scan_number)
        else:
            condition = self.data['scan_no'] == scan_number
        #RHE potential, potential always in the y dataset
        self.data_to_plot[scan_number]['potential'] = 0.205+np.array(self.data[condition]['potential'])[l:r]+0.059*np.array(self.data[self.data['scan_no'] == scan_number]['phs'])[0]
        for each in plot_label_list:
            if each == 'potential':
                pass
            elif each=='current':#RHE potential
                self.data_to_plot[scan_number][each] = -np.array(self.data[condition][each])[l:r]
            else:
                if each in ['peak_intensity','peak_intensity_error','strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                    temp_data = np.array(self.data[condition][each])[l:r]
                    self.data_to_plot[scan_number][each] = list(temp_data-max(temp_data))
                else:
                    self.data_to_plot[scan_number][each] = list(self.data[condition][each])[l:r]

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

        for each in plot_label_list:
            if each=='current':#RHE potential
                self.data_to_plot[scan_number][each] = -np.array(self.data[condition][each])[l:r]
                #self.data_to_plot[scan_number][each] = 0.205+np.array(self.data[condition][each])[l:r]+0.059*np.array(self.data[self.data['scan_no'] == scan_number]['phs'])[0]               
            else:
                self.data_to_plot[scan_number][each] = np.array(self.data[condition][each])[l:r]

    def append_scans(self):
        if self.radioButton_xrv.isChecked():
            self.append_scans_xrv()
        else:
            self.append_scans_ctr()

    def append_scans_xrv(self):
        text = self.scan_numbers_append.text()
        text_original = self.scan_numbers_all.text()
        if text_original!='':
            text_new = ','.join([text_original, text])
        else:
            text_new = text
        scans = list(set([int(each) for each in text_new.rstrip().rsplit(',')]))
        scans.sort()
        self.scan_numbers_all.setText(','.join([str(scan) for scan in scans]))
        assert (self.lineEdit_x.text()!='' and self.lineEdit_y.text()!=''), 'No channels for plotting have been selected!'
        assert self.scan_numbers_all.text()!='', 'No scans have been selected!'
        plot_labels = self.lineEdit_x.text() + ',' + self.lineEdit_y.text()
        plot_labels = plot_labels.rstrip().rsplit(',')
        for scan in scans:
            self.prepare_data_to_plot_xrv(plot_labels,scan)
            print('Prepare data for scan {} now!'.format(scan))
        self.scans = scans
        self.phs = [self.phs_all[self.scans_all.index(each)] for each in scans]
        self.plot_label_x = self.lineEdit_x.text()
        self.plot_labels_y = self.lineEdit_y.text().rstrip().rsplit(',')

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
            scan_list_temp = each.rsplit(';')
            for each_set in scan_list_temp:
                sub_scan_list = each_set.rsplit(',')
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