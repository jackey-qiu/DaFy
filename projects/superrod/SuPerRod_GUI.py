import sys,os, qdarkstyle
import traceback
from io import StringIO
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5 import uic, QtWidgets
import random
import numpy as np
import pandas as pd
import types,copy
import matplotlib.pyplot as plt
try:
    from . import locate_path
except:
    import locate_path
script_path = locate_path.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'dump_files'))
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
sys.path.append(os.path.join(DaFy_path,'projects'))
from UtilityFunctions import locate_tag
from UtilityFunctions import apply_modification_of_code_block as script_block_modifier
from models.structure_tools.sxrd_dafy import AtomGroup
from models.utils import UserVars
import diffev
from fom_funcs import *
import parameters
import data_superrod as data
import model
import solvergui
import time
import matplotlib
# matplotlib.use("TkAgg")
import _tkinter
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton, QTableWidgetItem, QHeaderView, QAbstractItemView, QInputDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTransform, QFont, QBrush, QColor, QIcon
from pyqtgraph.Qt import QtGui
import syntax_pars
from models.structure_tools import sorbate_tool
import logging

#redirect the error stream to qt widget
class QTextEditLogger(logging.Handler):
    def __init__(self, textbrowser_widget):
        super().__init__()
        self.textBrowser_error_msg = textbrowser_widget
        # self.widget.setReadOnly(True)

    def emit(self, record):
        error_msg = self.format(record)
        separator = '-' * 80
        notice = \
        """An unhandled exception occurred. Please report the problem\n"""\
        """using the error reporting dialog or via email to <%s>.\n"""%\
        ("crqiu2@gmail.com")
        self.textBrowser_error_msg.clear()
        cursor = self.textBrowser_error_msg.textCursor()
        cursor.insertHtml('''<p><span style="color: red;">{} <br></span>'''.format(" "))
        self.textBrowser_error_msg.setText(notice + '\n' +separator+'\n'+error_msg)

class RunFit(QtCore.QObject):
    """
    RunFit class to interface GUI to operate fit-engine, which is ran on a different thread
    ...
    Attributes
    ----------
    updateplot : pyqtSignal be emitted to be received by main GUI thread during fit
    solver: api for model fit using differential evolution algorithm

    Methods
    ----------
    run: start the fit
    stop: stop the fit

    """
    updateplot = QtCore.pyqtSignal(str,object,bool)
    fitended = QtCore.pyqtSignal(str)
    def __init__(self,solver):
        super(RunFit, self).__init__()
        self.solver = solver
        self.running = False

    def run(self):
        self.running = True
        self.solver.optimizer.stop = False
        self.solver.StartFit(self.updateplot,self.fitended)

    def stop(self):
        self.running = False
        self.solver.optimizer.stop = True

class RunBatch(QtCore.QObject):
    """
    RunFit class to interface GUI to operate fit-engine, which is ran on a different thread
    ...
    Attributes
    ----------
    updateplot : pyqtSignal be emitted to be received by main GUI thread during fit
    solver: api for model fit using differential evolution algorithm

    Methods
    ----------
    run: start the fit
    stop: stop the fit

    """
    updateplot = QtCore.pyqtSignal(str,object,bool)
    fitended = QtCore.pyqtSignal()
    def __init__(self,solver):
        super(RunBatch, self).__init__()
        self.solver = solver
        self.running = False

    def run(self):
        self.running = True
        self.solver.optimizer.stop = False
        self.solver.StartFit(self.updateplot,self.fitended)

    def stop(self):
        self.running = False
        self.solver.optimizer.stop = True

class MyMainWindow(QMainWindow):
    """
    GUI class for this app
    ....
    Attributes (selected)
    -----------
    <<widgets>>
    tableWidget_data: QTableWidget holding a list of datasets
    tableWidget_data_view: QTableWidget displaying each dataset
    widget_solver:pyqtgraph.parameter_tree_widget where you define
                  intrinsic parameters for undertaking DE optimization
    tableWidget_pars: QTableWidget displaying fit parameters
    widget_data: pyqtgraph.GraphicsLayoutWidget showing figures of
                 each ctr profile (data, fit, ideal and errorbars)
    widget_fom: pyqtgraph.GraphicsLayoutWidget showing evolution of
                figure of merit during fit
    widget_pars:pyqtgraph.GraphicsLayoutWidget showing best fit of
                each parameter at current generation and the search
                range in bar chart at this moment. longer bar means
                more aggressive searching during fit. If the bars 
                converge to one narrow line, fit quality cannot improved
                anymore. That means the fit is finished.
    widget_edp: GLViewWidget showing the 3d molecular structure of the
                current best fit model.
    widget_msv_top: GLViewWidget showing the top view of 3d molecular
                structure of the current best fit model. Only one sorbate
                and one layer of surface atoms are shown for clarity.
    plainTextEdit_script: QCodeEditor widget showing the model script
    widget_terminal:TerminalWidget, where you can do whatever you can in
                a normal python terminal. Three variables are loaded in 
                the namespace of the terminal:
                1) win: GUI main frame
                2) model: model that bridget script_module, pars and Fit engine
                you can explore the variables defined in your model script
                using model.script_module.vars (if vars is defined in script)
    <<others>>
    run_fit: Run_Fit instance to be launched to start/stop a fit. Refer to
             Run_Fit.solver to learn more about implementation of multi-processing
             programe method.
    model: model instance to coordinate script name space, dataset instance and par
           instance
    f_ideal: a list holding the structure factor values for unrelaxed structure
    data_profile: list of handles to plot ctr profiles including data and fit reuslts

    Methods (selected)
    -----------
    """
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        #pyqtgraph preference setting
        pg.setConfigOptions(imageAxisOrder='row-major', background = (50,50,100))
        pg.mkQApp()
        #load GUI ui file made by qt designer
        uic.loadUi(os.path.join(DaFy_path,'projects','SuperRod','superrod_gui.ui'),self)
        self.setWindowTitle('Data analysis factory: CTR data modeling')
        icon = QIcon(os.path.join(script_path,"icons","DAFY.png"))
        self.setWindowIcon(icon)

        #set redirection of error message to embeted text browser widget
        logTextBox = QTextEditLogger(self.textBrowser_error_msg)
        # You can format what is printed to text box
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        self.comboBox_all_motif.insertItems(0, sorbate_tool.ALL_MOTIF_COLLECTION)
        #self.stop = False
        self.show_checkBox_list = []
        self.domain_tag = 1
        #structure factor for ideal structure
        self.f_ideal=[]
        self.data_profiles = []
        self.model = model.Model()
        #init run_fit
        self.run_fit = RunFit(solvergui.SolverController(self.model))
        self.fit_thread = QtCore.QThread()
        self.run_fit.moveToThread(self.fit_thread)#move run_fit to a different thread
        #signal-slot connection
        self.run_fit.updateplot.connect(self.update_par_during_fit)
        self.run_fit.updateplot.connect(self.update_status)
        self.run_fit.fitended.connect(self.stop_model_slot)
        self.fit_thread.started.connect(self.run_fit.run)

        #init run_batch
        self.run_batch = RunBatch(solvergui.SolverController(self.model))
        self.batch_thread = QtCore.QThread()
        self.run_batch.moveToThread(self.batch_thread)
        #signal-slot connection
        self.run_batch.updateplot.connect(self.update_par_during_fit)
        self.run_batch.updateplot.connect(self.update_status_batch)
        self.run_batch.fitended.connect(self.stop_model_batch)
        self.batch_thread.started.connect(self.run_batch.run)

        #tool bar buttons to operate model
        self.actionNew.triggered.connect(self.init_new_model)
        self.actionOpen.triggered.connect(self.open_model)
        self.actionSaveas.triggered.connect(self.save_model_as)
        self.actionSave.triggered.connect(self.save_model)
        self.actionSimulate.triggered.connect(self.simulate_model)
        self.actionRun.triggered.connect(self.run_model)
        self.actionStop.triggered.connect(self.stop_model)
        self.actionCalculate.triggered.connect(self.calculate_error_bars)
        self.actionRun_batch_script.triggered.connect(self.run_model_batch)
        self.actionStopBatch.triggered.connect(self.terminate_model_batch)

        #menu items
        self.actionOpen_model.triggered.connect(self.open_model)
        self.actionSave_model.triggered.connect(self.save_model_as)
        self.actionSimulate_2.triggered.connect(self.simulate_model)
        self.actionStart_fit.triggered.connect(self.run_model)
        self.actionStop_fit.triggered.connect(self.stop_model)
        self.actionSave_table.triggered.connect(self.save_par)
        self.actionSave_script.triggered.connect(self.save_script)
        self.actionSave_data.triggered.connect(self.save_data)
        self.actionData.changed.connect(self.toggle_data_panel)
        self.actionPlot.changed.connect(self.toggle_plot_panel)
        self.actionScript.changed.connect(self.toggle_script_panel)

        #pushbuttons for data handeling
        self.pushButton_load_data.clicked.connect(self.load_data_ctr)
        self.pushButton_append_data.clicked.connect(self.append_data)
        self.pushButton_delete_data.clicked.connect(self.delete_data)
        self.pushButton_save_data.clicked.connect(self.save_data)
        self.pushButton_update_mask.clicked.connect(self.update_mask_info_in_data)
        self.pushButton_use_all.clicked.connect(self.use_all_data)
        self.pushButton_use_none.clicked.connect(self.use_none_data)
        self.pushButton_use_selected.clicked.connect(self.use_selected_data)
        self.pushButton_invert_use.clicked.connect(self.invert_use_data)

        #pushbuttons for structure view
        self.pushButton_azimuth_0.clicked.connect(self.azimuth_0)
        self.pushButton_azimuth_90.clicked.connect(self.azimuth_90)
        self.pushButton_elevation_0.clicked.connect(self.elevation_0)
        self.pushButton_elevation_90.clicked.connect(self.elevation_90)
        self.pushButton_parallel.clicked.connect(self.parallel_projection)
        self.pushButton_projective.clicked.connect(self.projective_projection)
        self.pushButton_pan.clicked.connect(self.pan_msv_view)
        self.pushButton_start_spin.clicked.connect(self.start_spin)
        self.pushButton_stop_spin.clicked.connect(self.stop_spin)
        self.pushButton_xyz.clicked.connect(self.save_structure_file)

        #spinBox to save the domain_tag
        self.spinBox_domain.valueChanged.connect(self.update_domain_index)

        #pushbutton to load/save script
        self.pushButton_load_script.clicked.connect(self.load_script)
        self.pushButton_save_script.clicked.connect(self.save_script)
        self.pushButton_modify_script.clicked.connect(self.modify_script)

        #pushbutton to load/save parameter file
        self.pushButton_load_table.clicked.connect(self.load_par)
        self.pushButton_save_table.clicked.connect(self.save_par)
        self.pushButton_remove_rows.clicked.connect(self.remove_selected_rows)
        self.pushButton_add_one_row.clicked.connect(self.append_one_row)
        self.pushButton_add_par_set.clicked.connect(self.append_par_set)
        self.pushButton_add_all_pars.clicked.connect(self.append_all_par_sets)
        self.pushButton_fit_all.clicked.connect(self.fit_all)
        self.pushButton_fit_none.clicked.connect(self.fit_none)
        self.pushButton_fit_selected.clicked.connect(self.fit_selected)
        self.pushButton_fit_next_5.clicked.connect(self.fit_next_5)
        self.pushButton_invert_fit.clicked.connect(self.invert_fit)
        self.pushButton_update_pars.clicked.connect(self.update_model_parameter)
        self.horizontalSlider_par.valueChanged.connect(self.play_with_one_par)

        #pushButton to operate plots
        self.pushButton_update_plot.clicked.connect(self.update_structure_view)
        self.pushButton_update_plot.clicked.connect(self.update_plot_data_view_upon_simulation)
        self.pushButton_update_plot.clicked.connect(self.update_par_bar_during_fit)
        self.pushButton_update_plot.clicked.connect(self.update_electron_density_profile)
        self.pushButton_previous_screen.clicked.connect(self.show_plots_on_previous_screen)
        self.pushButton_next_screen.clicked.connect(self.show_plots_on_next_screen)
        #select dataset in the viewer
        self.comboBox_dataset.activated.connect(self.update_data_view)

        #syntax highlight for script
        self.plainTextEdit_script.setStyleSheet("""QPlainTextEdit{
                                font-family:'Consolas';
                                font-size:14pt;
                                color: #ccc;
                                background-color: #2b2b2b;}""")
        self.plainTextEdit_script.setTabStopWidth(self.plainTextEdit_script.fontMetrics().width(' ')*4)

        #table view for parameters set to selecting row basis
        self.tableWidget_pars.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget_data.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.timer_update_structure = QtCore.QTimer(self)
        self.timer_update_structure.timeout.connect(self.pushButton_update_plot.click)
        self.timer_spin_msv = QtCore.QTimer(self)
        self.timer_spin_msv.timeout.connect(self.spin_msv)
        self.azimuth_angle = 0
        self.setup_plot()
        self._load_par()

    def show_plots_on_next_screen(self):
        """
        show plots on next screen, if one screen is not enough to fill all plots
        """
        if not hasattr(self,"num_screens_plot"):
            return

        if self.num_screens_plot>1:
            if self.current_index_plot_screen<(self.num_screens_plot-1):
                self.update_plot_dimension(self.current_index_plot_screen+1)
                self.update_plot_data_view()
            else:
                pass
        else:
            pass

    def show_plots_on_previous_screen(self):
        """
        show plots on previous screen
        """

        if not hasattr(self,"num_screens_plot"):
            return

        if self.num_screens_plot>1:
            if self.current_index_plot_screen>0:
                self.update_plot_dimension(self.current_index_plot_screen-1)
                self.update_plot_data_view()
            else:
                pass
        else:
            pass

    def toggle_data_panel(self):
        """data panel on the left side of GUI main frame"""
        self.tabWidget_data.setVisible(self.actionData.isChecked())

    def toggle_plot_panel(self):
        """plot panel on the top right side of main GUI frame"""
        self.tabWidget.setVisible(self.actionPlot.isChecked())

    def toggle_script_panel(self):
        """script panel on the bottom right side of main GUI frame"""
        self.tabWidget_2.setVisible(self.actionScript.isChecked())

    def update_domain_index(self):
        """update domain index, triggering the associated structure to show"""
        self.domain_tag = int(self.spinBox_domain.text())
        if self.model.compiled:
            self.widget_edp.items = []
            # self.widget_msv_top.items = []
            self.init_structure_view()
        else:
            pass

    def parallel_projection(self):
        self.widget_edp.opts['distance'] = 2000
        self.widget_edp.opts['fov'] = 1
        # self.widget_msv_top.opts['distance'] = 2000
        # self.widget_msv_top.opts['fov'] = 1
        self.update_structure_view()

    def projective_projection(self):
        self.widget_edp.opts['distance'] = 25
        self.widget_edp.opts['fov'] = 60
        # self.widget_msv_top.opts['distance'] = 25
        # self.widget_msv_top.opts['fov'] = 60
        self.update_structure_view()

    def pan_msv_view(self):
        value = int(self.spinBox_pan_pixel.text())
        self.widget_edp.pan(value*int(self.checkBox_x.isChecked()),value*int(self.checkBox_y.isChecked()),value*int(self.checkBox_z.isChecked()))

    def update_camera_position(self,widget_name = 'widget_edp', angle_type="azimuth", angle=0):
        getattr(self,widget_name).setCameraPosition(pos=None, distance=None, \
            elevation=[None,angle][int(angle_type=="elevation")], \
                azimuth=[None,angle][int(angle_type=="azimuth")])

    def azimuth_0(self):
        self.update_camera_position(angle_type="azimuth", angle=0)

    def azimuth_90(self):
        self.update_camera_position(angle_type="azimuth", angle=90)

    def start_spin(self):
        self.timer_spin_msv.start(100)

    def stop_spin(self):
        self.timer_spin_msv.stop()

    def spin_msv(self):
        #if self.azimuth > 360:
            
        self.update_camera_position(angle_type="azimuth", angle=self.azimuth_angle)
        self.azimuth_angle = self.azimuth_angle + 1


    def elevation_0(self):
        self.update_camera_position(angle_type="elevation", angle=0)

    def elevation_90(self):
        self.update_camera_position(angle_type="elevation", angle=90)

    #do this after model is loaded, so that you know len(data)
    def update_plot_dimension(self, current_index_plot_screen = 0):
        """Setting the layout of data profiles"""
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        height, width = sizeObject.height()*25.4/dpi,sizeObject.width()*25.4/dpi
        #maximum number of plots allowd to be fit in one screen
        #assuming the minimum plot panel has a size of (w:50mm * h:40mm)
        plot_cols, plot_rows = int(width/50), int(height/40)
        self.max_num_plots_per_screen = plot_cols*plot_rows

        self.widget_data.clear()
        self.widget_data.ci.currentRow = 0
        self.widget_data.ci.currentCol = 0

        self.data_profiles = []
        total_datasets = len(self.model.data)

        if total_datasets<self.max_num_plots_per_screen:
            self.num_screens_plot = 1
        else:
            self.num_screens_plot = int(total_datasets/self.max_num_plots_per_screen)+[0,1][int((total_datasets%self.max_num_plots_per_screen)>0)]
        self.current_index_plot_screen = current_index_plot_screen
        if self.num_screens_plot>1:#more than one screen
            if self.current_index_plot_screen<(self.num_screens_plot-1):
                columns = plot_cols#should be occupied in maximum
                num_plots_on_current_screen = self.max_num_plots_per_screen
            else:#last screen
                num_plots_ = total_datasets%self.max_num_plots_per_screen
                if num_plots_ == 0:
                    columns = plot_cols
                    num_plots_on_current_screen = self.max_num_plots_per_screen
                else:
                    num_plots_on_current_screen = num_plots_
                    if num_plots_>10:
                        columns = 4
                    else:
                        columns = 2
        elif self.num_screens_plot==1:#only one screen
            if total_datasets==self.max_num_plots_per_screen:
                num_plots_on_current_screen = self.max_num_plots_per_screen
                columns = plot_cols
            else:
                num_plots_on_current_screen = total_datasets
                if total_datasets>10:
                    columns = 4
                else:
                    columns = 2

        #current list of ax handle
        self.num_plots_on_current_screen = num_plots_on_current_screen
        offset = self.current_index_plot_screen*self.max_num_plots_per_screen
        for i in range(num_plots_on_current_screen):
            if 1:
                hk_label = '{}{}_{}'.format(str(int(self.model.data[i+offset].extra_data['h'][0])),str(int(self.model.data[i+offset].extra_data['k'][0])),str(self.model.data[i+offset].extra_data['Y'][0]))
                if (i%columns)==0 and (i!=0):
                    self.widget_data.nextRow()
                    self.data_profiles.append(self.widget_data.addPlot(title=hk_label))
                else:
                    self.data_profiles.append(self.widget_data.addPlot(title=hk_label))

    def setup_plot(self):
        self.fom_evolution_profile = self.widget_fom.addPlot()
        self.par_profile = self.widget_pars.addPlot()
        self.fom_scan_profile = self.widget_fom_scan.addPlot()

    def update_data_check_attr(self):
        """update the checkable attr of each dataset: use, show, showerror"""
        re_simulate = False
        for i in range(len(self.model.data)):
            #model.data: masked data
            self.model.data[i].show = self.tableWidget_data.cellWidget(i,1).isChecked()
            self.model.data[i].use_error = self.tableWidget_data.cellWidget(i,3).isChecked()
            #model.data_original: unmasked data for model saving later
            self.model.data_original[i].show = self.tableWidget_data.cellWidget(i,1).isChecked()
            self.model.data_original[i].use_error = self.tableWidget_data.cellWidget(i,3).isChecked()
            if self.model.data[i].use!=self.tableWidget_data.cellWidget(i,2).isChecked():
                re_simulate = True
                self.model.data[i].use = self.tableWidget_data.cellWidget(i,2).isChecked()
                self.model.data_original[i].use = self.tableWidget_data.cellWidget(i,2).isChecked()
        if re_simulate:
            self.simulate_model()

    def calc_f_ideal(self):
        self.f_ideal = []
        for i in range(len(self.model.data)):
            each = self.model.data[i]
            self.f_ideal.append(self.model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.x)**2)

    def update_plot_data_view(self):
        """update views of all figures if script is compiled, while only plot data profiles if otherwise"""
        if self.model.compiled:
            self.update_data_check_attr()
            self.update_plot_data_view_upon_simulation()
            self.update_electron_density_profile()
        else:
            offset = self.max_num_plots_per_screen*self.current_index_plot_screen
            for i in range(self.num_plots_on_current_screen):
                fmt = self.tableWidget_data.item(i+offset,4).text()
                fmt_symbol = list(fmt.rstrip().rsplit(';')[0].rsplit(':')[1])
                self.data_profiles[i].plot(self.model.data[i+offset].x, self.model.data[i+offset].y,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2], clear = True)
            [each.setLogMode(x=False,y=self.tableWidget_data.cellWidget(self.data_profiles.index(each),1).isChecked()) for each in self.data_profiles]
            [each.autoRange() for each in self.data_profiles]

    def update_electron_density_profile(self):
        if self.lineEdit_z_min.text()!='':
            z_min = float(self.lineEdit_z_min.text())
        else:
            z_min = -20
        if self.lineEdit_z_max.text()!='':
            z_max = float(self.lineEdit_z_max.text())
        else:
            z_max = 100
        raxs_A_list, raxs_P_list = [], []
        num_raxs = len(self.model.data)-1
        if hasattr(self.model.script_module, "rgh_raxs"):
            for i in range(num_raxs):
                raxs_A_list.append(eval("self.model.script_module.rgh_raxs.getA{}_D1()".format(i+1)))
                raxs_P_list.append(eval("self.model.script_module.rgh_raxs.getP{}_D1()".format(i+1)))
        else:
            raxs_A_list.append(0)
            raxs_P_list.append(0)
        HKL_raxs_list = [[],[],[]]
        for each in self.model.data:
            if each.x[0]>100:
                HKL_raxs_list[0].append(each.extra_data['h'][0])
                HKL_raxs_list[1].append(each.extra_data['k'][0])
                HKL_raxs_list[2].append(each.extra_data['Y'][0])
        try:
            if self.run_fit.running or self.run_batch.running:
                # edf = self.model.script_module.sample.plot_electron_density_muscovite_new(z_min=z_min, z_max=z_max,N_layered_water=50,resolution =200, freeze=self.model.script_module.freeze)
                # z_plot,eden_plot,_=self.model.script_module.sample.fourier_synthesis(np.array(HKL_raxs_list),np.array(raxs_P_list).transpose(),np.array(raxs_A_list).transpose(),z_min=z_min,z_max=z_max,resonant_el=self.model.script_module.raxr_el,resolution=200,water_scaling=0.33)
                label,edf = self.model.script_module.sample.plot_electron_density_superrod(z_min=z_min, z_max=z_max,N_layered_water=50,resolution =200)
                #z_plot,eden_plot,_=self.model.script_module.sample.fourier_synthesis(np.array(HKL_raxs_list),np.array(raxs_P_list).transpose(),np.array(raxs_A_list).transpose(),z_min=z_min,z_max=z_max,resonant_el=self.model.script_module.raxr_el,resolution=200,water_scaling=0.33)
            else:
                # edf = self.model.script_module.sample.plot_electron_density_muscovite_new(z_min=z_min,z_max=z_max,N_layered_water=500,resolution = 1000, freeze=self.model.script_module.freeze)
                # z_plot,eden_plot,_=self.model.script_module.sample.fourier_synthesis(np.array(HKL_raxs_list),np.array(raxs_P_list).transpose(),np.array(raxs_A_list).transpose(),z_min=z_min,z_max=z_max,resonant_el=self.model.script_module.raxr_el,resolution=1000,water_scaling=0.33)
                #edf = self.model.script_module.sample.plot_electron_density_muscovite_new(z_min=z_min,z_max=z_max,N_layered_water=500,resolution = 1000, freeze=self.model.script_module.freeze)
                label,edf = self.model.script_module.sample.plot_electron_density_superrod(z_min=z_min, z_max=z_max,N_layered_water=500,resolution =1000)
                # z_plot,eden_plot,_=self.model.script_module.sample.fourier_synthesis(np.array(HKL_raxs_list),np.array(raxs_P_list).transpose(),np.array(raxs_A_list).transpose(),z_min=z_min,z_max=z_max,resonant_el=self.model.script_module.raxr_el,resolution=1000,water_scaling=0.33)
            #eden_plot = [each*int(each>0) for each in eden_plot]
            self.fom_scan_profile.plot(edf[-1][0],edf[-1][1],pen = {'color': "w", 'width': 1},clear = True)
            self.fom_scan_profile.plot(edf[-1][0],edf[-1][1],fillLevel=0, brush = (0,200,0,100),clear = False)
            #self.fom_scan_profile.plot(z_plot,eden_plot,fillLevel=0, brush = (200,0,0,100),clear = False)
            #self.fom_scan_profile.plot(edf['e_data'][-1][0],edf['e_data'][-1][3],fillLevel=0, brush = (0,0,200,100),clear = False)
            self.fom_scan_profile.autoRange()
        except:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to draw e density profile!')
            logging.getLogger().exception('Fatal error encountered during drawing e density profile!')
            self.tabWidget_data.setCurrentIndex(4)

    def update_plot_data_view_upon_simulation(self):
        offset = self.max_num_plots_per_screen*self.current_index_plot_screen
        for i in range(self.num_plots_on_current_screen):
            if 1:
                fmt = self.tableWidget_data.item(i+offset,4).text()
                fmt_symbol = list(fmt.rstrip().rsplit(';')[0].rsplit(':')[1])
                line_symbol = list(fmt.rstrip().rsplit(';')[1].rsplit(':')[1])
                self.data_profiles[i].plot(self.model.data[i+offset].x, self.model.data[i+offset].y,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2],clear = True)
                if self.tableWidget_data.cellWidget(i+offset,3).isChecked():
                    #create error bar data, graphiclayout widget doesn't have a handy api to plot lines along with error bars
                    #disable this while the model is running
                    if not self.run_fit.solver.optimizer.running:
                        x = np.append(self.model.data[i+offset].x[:,np.newaxis],self.model.data[i+offset].x[:,np.newaxis],axis=1)
                        y_d = self.model.data[i+offset].y[:,np.newaxis] - self.model.data[i+offset].error[:,np.newaxis]/2
                        y_u = self.model.data[i+offset].y[:,np.newaxis] + self.model.data[i+offset].error[:,np.newaxis]/2
                        y = np.append(y_d,y_u,axis = 1)
                        for ii in range(len(y)):
                            self.data_profiles[i].plot(x=x[ii],y=y[ii],pen={'color':'w', 'width':1},clear = False)
                
                #plot ideal structure factor
                try:
                    scale_factor = [self.model.script_module.rgh.scale_nonspecular_rods,self.model.script_module.rgh.scale_specular_rod][int("00L" in self.model.data[i].name)]
                    h_, k_ = int(round(self.model.data[i+offset].extra_data['h'][0],0)),int(round(self.model.data[i+offset].extra_data['k'][0],0))
                    extra_scale_factor = 'scale_factor_{}{}L'.format(h_,k_)
                    if hasattr(self.model.script_module.rgh,extra_scale_factor):
                        rod_factor = getattr(self.model.script_module.rgh, extra_scale_factor)
                    else:
                        rod_factor = 1
                    self.data_profiles[i].plot(self.model.data[i+offset].x, self.f_ideal[i+offset]*(scale_factor*rod_factor)**2,pen = {'color': "w", 'width': 1},clear = False)
                except:
                    pass
                #plot simulated results
                if self.tableWidget_data.cellWidget(i+offset,2).isChecked():
                    self.data_profiles[i].plot(self.model.data[i+offset].x, self.model.data[i+offset].y_sim,pen={'color': line_symbol[1], 'width': int(line_symbol[0])},  clear = False)
                else:
                    pass
        [each.setLogMode(x=False,y=self.tableWidget_data.cellWidget(self.data_profiles.index(each)+offset,1).isChecked()) for each in self.data_profiles]
        [each.autoRange() for each in self.data_profiles]
        fom_log = np.array(self.run_fit.solver.optimizer.fom_log)
        self.fom_evolution_profile.plot(fom_log[:,0],fom_log[:,1],pen={'color': 'r', 'width': 2}, clear = True)
        self.fom_evolution_profile.autoRange()
        
    def update_par_bar_during_fit(self):
        """update bar chart during fit, which tells the current best fit and the searching range of each fit parameter"""
        if self.run_fit.running or self.run_batch.running:
            if self.run_fit.running:
                par_max = self.run_fit.solver.optimizer.par_max
                par_min = self.run_fit.solver.optimizer.par_min
                vec_best = copy.deepcopy(self.run_fit.solver.optimizer.best_vec)
                vec_best = (vec_best-par_min)/(par_max-par_min)
                pop_vec = np.array(copy.deepcopy(self.run_fit.solver.optimizer.pop_vec))
            elif self.run_batch.running:
                par_max = self.run_batch.solver.optimizer.par_max
                par_min = self.run_batch.solver.optimizer.par_min
                vec_best = copy.deepcopy(self.run_batch.solver.optimizer.best_vec)
                vec_best = (vec_best-par_min)/(par_max-par_min)
                pop_vec = np.array(copy.deepcopy(self.run_batch.solver.optimizer.pop_vec))

            trial_vec_min =[]
            trial_vec_max =[]
            for i in range(len(par_max)):
                trial_vec_min.append((np.min(pop_vec[:,i])-par_min[i])/(par_max[i]-par_min[i]))
                trial_vec_max.append((np.max(pop_vec[:,i])-par_min[i])/(par_max[i]-par_min[i]))
            trial_vec_min = np.array(trial_vec_min)
            trial_vec_max = np.array(trial_vec_max)
            bg = pg.BarGraphItem(x=range(len(vec_best)), y=(trial_vec_max + trial_vec_min)/2, height=(trial_vec_max - trial_vec_min)/2, brush='b', width = 0.8)
            self.par_profile.clear()
            self.par_profile.addItem(bg)
            self.par_profile.plot(vec_best, pen=(0,0,0), symbolBrush=(255,0,0), symbolPen='w')
        else:
            pass

    def calculate_error_bars(self):
        """
        cal the error bar for each fit par after fit is completed
        note the error bar values are only estimated from all intermediate fit reuslts from all fit generations,
        and the error may not accutely represent the statistic errors. If you want to get statistical errors of 
        each fit parameter, you can run a further NLLS fit using the the best fit parameters, which is not implemented in the program.
        """
        try:
            error_bars = self.run_fit.solver.CalcErrorBars()
            total_num_par = len(self.model.parameters.data)
            index_list = [i for i in range(total_num_par) if self.model.parameters.data[i][2]]

            for i in range(len(error_bars)):
                self.model.parameters.data[index_list[i]][-1] = error_bars[i]
            
            self.update_par_upon_load()
        except diffev.ErrorBarError as e:

            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to calculate error bar!')
            logging.getLogger().exception('Fatal error encountered during error calculation!')
            self.tabWidget_data.setCurrentIndex(4)

            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)


    def init_new_model(self):
        reply = QMessageBox.question(self, 'Message', 'Would you like to save the current model first?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.save_model()
        try:
            self.model = model.Model()
            self.run_fit.solver.model = self.model
            self.tableWidget_data.setRowCount(0)
            self.tableWidget_pars.setRowCount(0)
            self.plainTextEdit_script.setPlainText('')
            self.comboBox_dataset.clear()
            self.tableWidget_data_view.setRowCount(0)
            # self.update_plot_data_view()
            self._load_par()
        except Exception:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to init a new model!')
            logging.getLogger().exception('Fatal error encountered during model initiation!')
            self.tabWidget_data.setCurrentIndex(4)

    def open_model(self):
        """open a saved model file(*.rod), which is a compressed file containing data, script and fit parameters in one place"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","rod file (*.rod);;zip Files (*.rar)", options=options)
        load_add_ = 'success'
        self.rod_file = fileName
        if fileName:
            try:
                self.setWindowTitle('Data analysis factory: CTR data modeling-->{}'.format(fileName))
                self.model.load(fileName)
                self.update_plot_dimension()
                try:
                    self.load_addition()
                except:
                    load_add_ = 'failure'
                #add a mask attribute to each dataset
                for each in self.model.data_original:
                    if not hasattr(each,'mask'):
                        each.mask = np.array([True]*len(each.x))
                for each in self.model.data:
                    if not hasattr(each,'mask'):
                        each.mask = np.array([True]*len(each.x))
                #add model space to terminal
                self.widget_terminal.update_name_space("model",self.model)
                self.widget_terminal.update_name_space("solver",self.run_fit.solver)
                self.widget_terminal.update_name_space("win",self)

                #remove items in the msv and re-initialize it
                self.widget_edp.items = []
                # self.widget_msv_top.items = []
                #update other pars
                self.update_table_widget_data()
                self.update_combo_box_dataset()
                self.update_plot_data_view()
                self.update_par_upon_load()
                self.update_script_upon_load()
                #model is simulated at the end of next step
                self.init_mask_info_in_data_upon_loading_model()
                #add name space for cal bond distance after simulation
                try:
                    self.widget_terminal.update_name_space("report_distance",self.model.script_module.sample.inter_atom_distance_report)
                except:
                    pass
                #now set the comboBox for par set
                self.update_combo_box_list_par_set()

                self.statusbar.clearMessage()
                self.statusbar.showMessage("Model is loaded, and {} in config loading".format(load_add_))
                # self.update_mask_info_in_data()
            except Exception:

                self.statusbar.clearMessage()
                self.statusbar.showMessage('Failure to open a model file!')
                logging.getLogger().exception('Fatal error encountered during openning a model file!')
                self.tabWidget_data.setCurrentIndex(4)

    def update_combo_box_list_par_set(self):
        """atomgroup and uservars instances defined in script will be colleced and displayed in this combo box"""
        attrs = self.model.script_module.__dir__()
        attr_wanted = [each for each in attrs if type(getattr(self.model.script_module, each)) in [AtomGroup, UserVars]]
        num_items = self.comboBox_register_par_set.count()
        for i in range(num_items):
            self.comboBox_register_par_set.removeItem(0)
        self.comboBox_register_par_set.insertItems(0,attr_wanted)

    def append_all_par_sets(self):
        """append fit parameters for all parset listed in the combo box, handy tool to save manual adding them in par table"""
        if "table_container" in self.model.script_module.__dir__():
            if len(self.model.script_module.table_container)!=0:
                table = self.model.script_module.table_container[::-1]
                rows = self.tableWidget_pars.selectionModel().selectedRows()
                if len(rows) == 0:
                    row_index = self.tableWidget_pars.rowCount()
                else:
                    row_index = rows[-1].row()
                for ii in range(len(table)):
                    self.tableWidget_pars.insertRow(row_index)
                    for i in range(6):
                        if i==2:
                            check_box = QCheckBox()
                            check_box.setChecked(eval(table[ii][i]))
                            self.tableWidget_pars.setCellWidget(row_index,2,check_box)
                        else:
                            if i == 0:
                                qtablewidget = QTableWidgetItem(table[ii][i])
                                qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                            elif i in [1]:
                                qtablewidget = QTableWidgetItem(table[ii][i])
                                qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                            elif i ==5:
                                qtablewidget = QTableWidgetItem('(0,0)')
                            else:
                                qtablewidget = QTableWidgetItem(table[ii][i])
                            self.tableWidget_pars.setItem(row_index,i,qtablewidget)
                self.append_one_row_at_the_end()
        else:
            par_all = [self.comboBox_register_par_set.itemText(i) for i in range(self.comboBox_register_par_set.count())]
            for par in par_all:
                self.append_par_set(par)

    def append_par_set(self, par_selected = None):
        #boundary mapping for quick setting the bounds of fit pars
        bounds_map = {"setR":[0.8,1.8],"setScale":[0,1],("setdx","sorbate"):[-0.5,0.5],\
                     ("setdy","sorbate"):[-0.5,0.5],("setdz","sorbate"):[-0.1,1],("setoc","sorbate"):[0.5,3],\
                     ("setdx","surface"):[-0.1,0.1],("setdy","surface"):[-0.1,0.1],("setdz","surface"):[-0.1,0.1],\
                     ("setoc","surface"):[0.6,1],"setDelta":[-20,60],"setGamma":[0,180],"setBeta":[0,0.1]}
        def _get_bounds(attr_head,attr_item):
            for key in bounds_map:
                if type(key)==str:
                    if key in attr_item:
                        return bounds_map[key]
                else:
                    if (key[0] in attr_item) and (key[1] in attr_head):
                        return bounds_map[key]
            return []
        if par_selected==None:
            par_selected = self.comboBox_register_par_set.currentText()
        else:
            pass
        attrs = eval("self.model.script_module.{}.__dir__()".format(par_selected))
        attrs_wanted = [each for each in attrs if each.startswith("set")]

        rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(rows) == 0:
            row_index = self.tableWidget_pars.rowCount()
        else:
            row_index = rows[-1].row()
        for ii in range(len(attrs_wanted)):
            self.tableWidget_pars.insertRow(row_index)
            current_value = eval("self.model.script_module."+par_selected+'.'+attrs_wanted[ii].replace('set','get')+"()")
            bounds_temp = _get_bounds(par_selected,attrs_wanted[ii])
            #update the bounds if the current value is out of the bound range
            if len(bounds_temp)==2:
                if current_value<bounds_temp[0]:
                    bounds_temp[0] = current_value
                if current_value>bounds_temp[1]:
                    bounds_temp[1] = current_value
            for i in range(6):
                if i==2:
                    check_box = QCheckBox()
                    check_box.setChecked(True)
                    self.tableWidget_pars.setCellWidget(row_index,2,check_box)
                else:
                    if i == 0:
                        qtablewidget = QTableWidgetItem(".".join([par_selected,attrs_wanted[ii]]))
                        qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                    elif i in [1]:
                        qtablewidget = QTableWidgetItem(str(round(current_value,4)))
                        qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                    elif i ==5:
                        qtablewidget = QTableWidgetItem('(0,0)')
                    elif i ==3:
                        #left boundary of fit parameter
                        if len(bounds_temp)!=0:
                            qtablewidget = QTableWidgetItem(str(round(bounds_temp[0],4)))
                        else:
                            qtablewidget = QTableWidgetItem(str(round(current_value*0.5,4)))
                    elif i ==4:
                        #right boundary of fit parameter
                        if len(bounds_temp)!=0:
                            qtablewidget = QTableWidgetItem(str(round(bounds_temp[1],4)))
                        else:
                            qtablewidget = QTableWidgetItem(str(round(current_value*1.5,4)))

                    self.tableWidget_pars.setItem(row_index,i,qtablewidget)
        self.append_one_row_at_the_end()

    def auto_save_model(self):
        """model will be saved automatically during fit, for which you need to set the interval generations for saving automatically"""
        #the model will be renamed this way
        path = self.rod_file.replace(".rod","_ran.rod")
        if path:
            self.model.script = (self.plainTextEdit_script.toPlainText())
            self.model.save(path)
            save_add_ = 'success'
            try:
                self.save_addition()
            except:
                save_add_ = "failure"
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Model is saved, and {} in config saving".format(save_add_))

    def save_model(self):
        """model will be saved automatically during fit, for which you need to set the interval generations for saving automatically"""
        #the model will be renamed this way
        try:
            path = self.rod_file
            self.model.script = (self.plainTextEdit_script.toPlainText())
            self.model.save(path)
            save_add_ = 'success'
            try:
                self.save_addition()
            except:
                save_add_ = "failure"
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Model is saved, and {} in config saving".format(save_add_))
        except Exception:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to save model!')
            logging.getLogger().exception('Fatal error encountered during model save!')
            self.tabWidget_data.setCurrentIndex(4)

    def save_model_as(self):
        """save model file, promting a dialog widget to ask the file name to save model"""
        path, _ = QFileDialog.getSaveFileName(self, "Save file as", "", "rod file (*.rod);;zip files (*.rar)")
        if path:
            #update the rod_file attribute
            self.rod_file = path
            self.model.script = (self.plainTextEdit_script.toPlainText())
            self.model.save(path)
            save_add_ = 'success'
            try:
                self.save_addition()
            except:
                save_add_ = "failure"
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Model is saved, and {} in config saving".format(save_add_))
            self.setWindowTitle('Data analysis factory: CTR data modeling-->{}'.format(path))

    #here save also the config pars for diffev solver
    def save_addition(self):
        """save solver parameters, pulling from pyqtgraphy.parameter_tree widget"""
        values=\
                [self.widget_solver.par.param('Diff.Ev.').param('k_m').value(),
                self.widget_solver.par.param('Diff.Ev.').param('k_r').value(),
                self.widget_solver.par.param('Diff.Ev.').param('Method').value(),
                self.widget_solver.par.param('FOM').param('Figure of merit').value(),
                self.widget_solver.par.param('FOM').param('Auto save, interval').value(),
                self.widget_solver.par.param('FOM').param('weighting factor').value(),
                self.widget_solver.par.param('FOM').param('weighting region').value(),
                self.widget_solver.par.param('Fitting').param('start guess').value(),
                self.widget_solver.par.param('Fitting').param('Generation size').value(),
                self.widget_solver.par.param('Fitting').param('Population size').value()]
        pars = ['k_m','k_r','Method','Figure of merit','Auto save, interval','weighting factor','weighting region','start guess','Generation size','Population size']
        for i in range(len(pars)):
            self.model.save_addition(pars[i],str(values[i]))
            # print(pars[i],str(values[i]))
    
    def load_addition(self):
            funcs=\
                [self.widget_solver.par.param('Diff.Ev.').param('k_m').setValue,
                self.widget_solver.par.param('Diff.Ev.').param('k_r').setValue,
                self.widget_solver.par.param('Diff.Ev.').param('Method').setValue,
                self.widget_solver.par.param('FOM').param('Figure of merit').setValue,
                self.widget_solver.par.param('FOM').param('Auto save, interval').setValue,
                self.widget_solver.par.param('FOM').param('weighting factor').setValue,
                self.widget_solver.par.param('FOM').param('weighting region').setValue,
                self.widget_solver.par.param('Fitting').param('start guess').setValue,
                self.widget_solver.par.param('Fitting').param('Generation size').setValue,
                self.widget_solver.par.param('Fitting').param('Population size').setValue]

            types= [float,float,str,str,int,float,str,bool,int,int]
            pars = ['k_m','k_r','Method','Figure of merit','Auto save, interval','weighting factor','weighting region','start guess','Generation size','Population size']
            for i in range(len(pars)):
                type_ = types[i]
                if type_ == float:
                    value = np.round(float(self.model.load_addition(pars[i])),2)
                elif type_==str:
                    value = self.model.load_addition(pars[i]).decode("utf-8")
                elif type_==bool:
                    value = (self.model.load_addition(pars[i]).decode("ASCII")=="True")
                else:
                    value = type_(self.model.load_addition(pars[i]))
                funcs[i](value)

    def simulate_model(self):
        """
        simulate the model
        script will be updated and compiled to make name spaces in script_module
        """
        self.update_data_check_attr()
        self.update_par_upon_change()
        self.model.script = (self.plainTextEdit_script.toPlainText())
        self.widget_solver.update_parameter_in_solver(self)
        try:
            self.model.simulate()
            self.update_structure_view()
            try:
                self.calc_f_ideal()
            except:
                pass
            self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,self.model.fom))
            self.update_plot_data_view_upon_simulation()
            self.update_electron_density_profile()
            if hasattr(self.model.script_module,'model_type'):
                if self.model.script_module.model_type=='ctr':
                    self.init_structure_view()
                else:
                    pass
            else:
                self.init_structure_view()
            self.statusbar.clearMessage()
            self.update_combo_box_list_par_set()
            self.statusbar.showMessage("Model is simulated successfully!")
        except model.ModelError as e:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to simulate model!')
            logging.getLogger().exception('Fatal error encountered during model simulation!')
            self.tabWidget_data.setCurrentIndex(4)
            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)

    #execution when you move the slide bar to change only one parameter
    def simulate_model_light(self):
        try:
            self.model.simulate()
            self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,self.model.fom))
            self.update_plot_data_view_upon_simulation()
            self.statusbar.showMessage("Model is simulated now!")
        except model.ModelError as e:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to simulate model!')
            logging.getLogger().exception('Fatal error encountered during model simulation!')
            self.tabWidget_data.setCurrentIndex(4)
            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)

    def play_with_one_par(self):
        selected_rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(selected_rows)>0:
            #only get the first selected item
            par_set = self.model.parameters.data[selected_rows[0].row()]
            par_min, par_max = par_set[-3], par_set[-2]
            value = (par_max - par_min)*self.horizontalSlider_par.value()/100 + par_min
            self.model.parameters.set_value(selected_rows[0].row(), 1, value)
            self.lineEdit_scan_par.setText('{}:{}'.format(par_set[0],value))
            self.simulate_model_light()
        else:
            print('Doing nothing!')
            pass

    def run_model(self):
        """start the model fit looping"""
        #button will be clicked every 2 second to update figures
        try:
            # self.stop_model()
            self.simulate_model()
            self.statusbar.showMessage("Initializing model running ...")
            self.timer_update_structure.start(2000)
            self.widget_solver.update_parameter_in_solver(self)
            self.fit_thread.start()
        except:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to launch a model fit!')
            logging.getLogger().exception('Fatal error encountered during init model fitting!')
            self.tabWidget_data.setCurrentIndex(4)

    def stop_model(self):
        self.run_fit.stop()
        self.fit_thread.terminate()
        self.timer_update_structure.stop()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Model run is aborted!")
        
    @QtCore.pyqtSlot(str)
    def stop_model_slot(self,message):
        self.stop_model()
        logging.getLogger().exception(message)
        self.tabWidget_data.setCurrentIndex(4)

    def _stop_model(self):
        self.run_batch.stop()
        self.batch_thread.terminate()
        self.timer_update_structure.stop()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Batch model run is aborted!")

    def run_model_batch(self):
        """start the model fit looping in a batch mode
        To speed up the structure and plots are not to be updated!
        """
        try:
            #self._stop_model()
            self.simulate_model()
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Initializing model running ...")
            self.widget_solver.update_parameter_in_solver_batch(self)
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Parameters in solver are updated!")
            self.batch_thread.start()
        except Exception:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to batch run a model!')
            logging.getLogger().exception('Fatal error encountered during batch run a model!')
            self.tabWidget_data.setCurrentIndex(4)


    def stop_model_batch(self):
        self.run_batch.stop()
        self.batch_thread.terminate()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Batch model run is aborted to work on next task!")
        if self.update_fit_setup_for_batch_run():
            self.run_model_batch()
        else:
            pass

    def terminate_model_batch(self):
        self.run_batch.stop()
        self.batch_thread.terminate()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Batch model run is aborted now!")

    def update_fit_setup_for_batch_run(self):
        """
        Update the fit parameters and the fit dataset for next batch job!
        
        Returns:
            [bool] -- move to the end of datasets or not?
        """
        first_checked_data_item, first_checked_par_item = None, None
        for i in range(self.tableWidget_data.rowCount()):
            if self.tableWidget_data.cellWidget(i,2).checkState()!=0:
                first_checked_data_item = i
                break
        for i in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.cellWidget(i,2)!=None:
                if self.tableWidget_pars.cellWidget(i,2).checkState()!=0:
                    first_checked_par_item = i
                    break
        self.use_none_data()
        self.fit_none()
        try:
            [self.tableWidget_pars.cellWidget(i+6+first_checked_par_item,2).setChecked(True) for i in range(5)]
            self.tableWidget_data.cellWidget(1+first_checked_data_item,2).setChecked(True)
            self.update_model_parameter()
            return True
        except:
            return False

    def load_data(self, loader = 'ctr'):
        self._empty_data_pool()
        exec('self.load_data_{}()'.format(loader))

    def append_data(self):
        self.load_data_ctr()

    def _empty_data_pool(self):
        #now empty the data pool
        self.model.data.items = [data.DataSet(name='Data 0')]
        self.model.data._counter = 1

    def load_data_ctr(self):
        """
        load data
        ------------
        if the data is ctr data, then you should stick to the dataformat as follows
        #8 columns in total
        #X, H, K, Y, I, eI, LB, dL
        #for CTR data, X column is L column, Y column all 0
        #for RAXR data, X column is energy column, Y column is L column
        #H, K, columns are holding H, K values
        #I column is holding background-subtraced intensity of ctr signal
        #LB, and dL are two values for roughness calculation
           LB: first Bragg peak L of one rod
           dL: interval L between two adjacent Bragg peak L's
        To get access to these columns:
            X column: data.x
            I column: data.y
            eI column: data.error
            H column: data.extra_data["h"]
            K column: data.extra_data["k"]
            Y column: data.extra_data["Y"]
            LB column: data.extra_data["LB"]
            dL column: data.extra_data["dL"]
        ---------------
        if the data you want to load is not in CTR format, to make successful loading, assure:
            1)your data file has 8 columns
            2)columns are space-separated (or tab-seperated)
            3)you can add comment lines heading with "#"
            4)if your data has <8 columns, then fill the other unused columns with 0
            5)to asscess your data column, you should use the naming rule described above, although
              the real meaning of each column, eg X column, could be arbitrary at your wishes
              For example, you put frequence values to the first column(X column), then to access this
              column, you use data.X

        Data file of 8 columns should be enough to encountpass many different situations.
        """
        self.model.compiled = False
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","csv Files (*.csv);;data Files (*.dat);txt Files (*.txt)", options=options)
        current_data_set_name = [self.tableWidget_data.item(i,0).text() for i in range(self.tableWidget_data.rowCount())]
        if fileName:
            with open(fileName,'r') as f:
                data_loaded = np.loadtxt(f,comments = '#',delimiter=None)
                data_loaded_pd = pd.DataFrame(data_loaded, columns = ['X','h','k','Y','I','eI','LB','dL'])
                data_loaded_pd['h'] = data_loaded_pd['h'].apply(lambda x:int(np.round(x)))
                data_loaded_pd['k'] = data_loaded_pd['k'].apply(lambda x:int(np.round(x)))
                data_loaded_pd.sort_values(by = ['h','k','Y'], inplace = True)
                hk_unique = list(set(zip(list(data_loaded_pd['h']), list(data_loaded_pd['k']), list(data_loaded_pd['Y']))))
                hk_unique.sort()
                h_unique = [each[0] for each in hk_unique]
                k_unique = [each[1] for each in hk_unique]
                Y_unique = [each[2] for each in hk_unique]

                for i in range(len(h_unique)):
                    h_temp, k_temp, Y_temp = h_unique[i], k_unique[i], Y_unique[i]
                    if Y_temp==0:#CTR data
                        name = 'Data-{}{}L'.format(h_temp, k_temp)
                    else:#RAXR data
                        name = 'Data-{}{}_L={}'.format(h_temp, k_temp, Y_temp)
                    tag = sum([int(name in each) for each in current_data_set_name])+1
                    #if name in current_data_set_name:
                    name = name + '_{}'.format(tag)
                    self.model.data.add_new(name = name)
                    sub_data = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)& (data_loaded_pd['Y']==Y_temp)]
                    sub_data.sort_values(by='X',inplace =True)
                    self.model.data.items[-1].x = sub_data['X'].to_numpy()
                    self.model.data.items[-1].y = sub_data['I'].to_numpy()
                    self.model.data.items[-1].error = sub_data['eI'].to_numpy()
                    self.model.data.items[-1].x_raw = sub_data['X'].to_numpy()
                    self.model.data.items[-1].y_raw = sub_data['I'].to_numpy()
                    self.model.data.items[-1].error_raw = sub_data['eI'].to_numpy()
                    self.model.data.items[-1].set_extra_data(name = 'h', value = sub_data['h'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'k', value = sub_data['k'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'Y', value = sub_data['Y'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'LB', value = sub_data['LB'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'dL', value = sub_data['dL'].to_numpy())
                    self.model.data.items[-1].mask = np.array([True]*len(self.model.data.items[-1].x))
        #now remove the empty datasets
        empty_data_index = []
        i=0
        for each in self.model.data.items:
            if len(each.x_raw) == 0:
                empty_data_index.append(i)
            i += 1
        for i in range(len(empty_data_index)):
            self.model.data.delete_item(empty_data_index[i])
            for ii in range(len(empty_data_index)):
                if empty_data_index[ii]>empty_data_index[i]:
                    empty_data_index[ii] = empty_data_index[ii]-1
                else:
                    pass
        self.model.data_original = copy.deepcopy(self.model.data)
        #update the view
        self.update_table_widget_data()
        self.update_combo_box_dataset()
        self.update_plot_dimension()
        self.update_plot_data_view()

    def delete_data(self):
        self.model.compiled = False
        # Delete the selected mytable lines
        row_index = [each.row() for each in self.tableWidget_data.selectionModel().selectedRows()]
        row_index = sorted(row_index, reverse=True)
        for each in row_index:
            self.model.data.delete_item(each)
            self.model.data_original.delete_item(each)
        self.update_table_widget_data()
        self.update_combo_box_dataset()
        self.update_plot_dimension()
        self.update_plot_data_view()

    def update_table_widget_data(self):
        self.tableWidget_data.clear()
        self.tableWidget_data.setRowCount(len(self.model.data))
        self.tableWidget_data.setColumnCount(5)
        self.tableWidget_data.setHorizontalHeaderLabels(['DataID','logY','Use','Errors','fmt'])
        for i in range(len(self.model.data)):
            current_data = self.model.data[i]
            name = current_data.name
            for j in range(5):
                if j == 0:
                    qtablewidget = QTableWidgetItem(name)
                    self.tableWidget_data.setItem(i,j,qtablewidget)
                elif j == 4:
                    qtablewidget = QTableWidgetItem('sym:6bw;l:3r')
                    self.tableWidget_data.setItem(i,j,qtablewidget)
                else:
                    #note j=1 to j=3 corresponds to data.show, data.use, data.use_error
                    #data.show is not used for judging showing or not(all datasets are shown)
                    #It is instead used to specify the scale of Y(log or not)
                    check = getattr(current_data, ['show', 'use', 'use_error'][j-1])
                    check_box = QCheckBox()
                    check_box.setChecked(check)
                    #check_box.stateChanged.connect(self.update_plot_data_view)
                    self.tableWidget_data.setCellWidget(i,j,check_box)
        
        # self.tableWidget_data.resizeColumnsToContents()
        # self.tableWidget_data.resizeRowsToContents()

    def use_all_data(self):
        """fit all datasets
        """
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            self.tableWidget_data.cellWidget(i,2).setChecked(True)
        self.simulate_model()

    def use_none_data(self):
        """fit none of those datasets
        """
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            self.tableWidget_data.cellWidget(i,2).setChecked(False)
        self.simulate_model()

    def use_selected_data(self):
        """fit those that have been selected
        """
        selected_row_index = [each.row() for each in self.tableWidget_data.selectionModel().selectedRows()]
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            if i in selected_row_index:
                self.tableWidget_data.cellWidget(i,2).setChecked(True)
            else:
                self.tableWidget_data.cellWidget(i,2).setChecked(False)
        self.simulate_model()

    def invert_use_data(self):
        """invert the selection of to-be-fit datasets
        """
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            checkstate = self.tableWidget_data.cellWidget(i,2).checkState()
            if checkstate == 0:
                self.tableWidget_data.cellWidget(i,2).setChecked(True)
            else:
                self.tableWidget_data.cellWidget(i,2).setChecked(False)
        self.simulate_model()

    def update_combo_box_dataset(self):
        new_items = [each.name for each in self.model.data]
        self.comboBox_dataset.clear()
        self.comboBox_dataset.addItems(new_items)

    def update_data_view(self):
        """update the data view widget to show data values as table"""
        dataset_name = self.comboBox_dataset.currentText()
        dataset = None
        for each in self.model.data_original:
            if each.name == dataset_name:
                dataset = each
                break
            else:
                pass
        column_labels_main = ['x','y','error','mask']
        extra_labels = ['h', 'k', 'dL', 'LB']
        all_labels = ['x','y','error','h','k','dL','LB','mask']
        self.tableWidget_data_view.setRowCount(len(dataset.x))
        self.tableWidget_data_view.setColumnCount(len(all_labels))
        self.tableWidget_data_view.setHorizontalHeaderLabels(all_labels)
        for i in range(len(dataset.x)):
            for j in range(len(all_labels)):
                if all_labels[j] in column_labels_main:
                    item_ = getattr(dataset,all_labels[j])[i]
                    if all_labels[j] == 'mask':
                        qtablewidget = QTableWidgetItem(str(item_))
                    else:
                        qtablewidget = QTableWidgetItem(str(round(item_,4)))
                elif all_labels[j] in extra_labels:
                    qtablewidget = QTableWidgetItem(str(dataset.get_extra_data(all_labels[j])[i]))
                else:
                    qtablewidget = QTableWidgetItem('True')
                self.tableWidget_data_view.setItem(i,j,qtablewidget)

    def update_mask_info_in_data(self):
        """if the mask value is False, the associated data point wont be shown and wont be fitted as well"""
        dataset_name = self.comboBox_dataset.currentText()
        dataset = None
        for each in self.model.data_original:
            if each.name == dataset_name:
                dataset = each
                break
            else:
                pass
        for i in range(len(dataset.x)):
            dataset.mask[i] = (self.tableWidget_data_view.item(i,7).text() == 'True')
        self.model.data = copy.deepcopy(self.model.data_original)
        [each.apply_mask() for each in self.model.data]
        self.simulate_model()

    def init_mask_info_in_data_upon_loading_model(self):
        """apply mask values to each dataset"""
        self.model.data = copy.deepcopy(self.model.data_original)
        [each.apply_mask() for each in self.model.data]
        self.simulate_model()

    def init_structure_view(self):
        try:
            domain_tag = int(self.spinBox_domain.text())
        except:
            domain_tag = 0
        size_domain = len(self.model.script_module.sample.domain)
        if size_domain<(1+domain_tag):
            domain_tag = size_domain -1
        else:
            pass
        # self.widget_edp.items = []
        # self.widget_msv_top.items = []
        self.widget_edp.abc = [self.model.script_module.sample.unit_cell.a,self.model.script_module.sample.unit_cell.b,self.model.script_module.sample.unit_cell.c]
        # self.widget_msv_top.abc = self.widget_edp.abc
        xyz,_ = self.model.script_module.sample.extract_xyz_top(domain_tag)
        self.widget_edp.show_structure(xyz)
        self.update_camera_position(widget_name = 'widget_edp', angle_type="azimuth", angle=0)
        self.update_camera_position(widget_name = 'widget_edp', angle_type = 'elevation', angle = 0)

        # xyz,_ = self.model.script_module.sample.extract_xyz_top(domain_tag)
        # self.widget_msv_top.show_structure(xyz)
        # self.update_camera_position(widget_name = 'widget_msv_top', angle_type="azimuth", angle=0)
        # self.update_camera_position(widget_name = 'widget_msv_top', angle_type = 'elevation', angle = 90)
        """
        try:
            xyz,_ = self.model.script_module.sample.extract_xyz_top(domain_tag)
            self.widget_msv_top.show_structure(xyz)
            self.update_camera_position(widget_name = 'widget_msv_top', angle_type="azimuth", angle=0)
            self.update_camera_position(widget_name = 'widget_msv_top', angle_type = 'elevation', angle = 90)
        except:
            pass
        """

    def update_structure_view(self):
        if hasattr(self.model.script_module,"model_type"):
            if getattr(self.model.script_module,"model_type")=="ctr":
                pass
            else:
                return
        else:
            pass
        try:
            if self.spinBox_domain.text()=="":
                domain_tag = 0
            else:
                domain_tag = int(self.spinBox_domain.text())
            size_domain = len(self.model.script_module.sample.domain)
            if size_domain<(1+domain_tag):
                domain_tag = size_domain -1
            else:
                pass        
            xyz, _ = self.model.script_module.sample.extract_xyz_top(domain_tag)
            if self.run_fit.running: 
                self.widget_edp.update_structure(xyz)
            else:
                self.widget_edp.clear()
                #self.widget_edp.items = []
                self.widget_edp.abc = [self.model.script_module.sample.unit_cell.a,self.model.script_module.sample.unit_cell.b,self.model.script_module.sample.unit_cell.c]
                self.widget_edp.show_structure(xyz)

            """
            try:
                xyz, _ = self.model.script_module.sample.extract_xyz_top(domain_tag)
                self.widget_msv_top.update_structure(xyz)
            except:
                pass
            """
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            _ = QMessageBox.question(self, "",'Runtime error message:\n{}'.format(str(val)), QMessageBox.Ok)

    def save_structure_file(self):
        domain_tag, done = QInputDialog.getInt(self, 'Domain tag', 'Enter the domain index for the structure you want to save eg 0:')
        if not done:
            domain_tag = 0
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "xyz file (*.xyz)")
        self.model.script_module.sample.make_xyz_file(which_domain = int(domain_tag), save_file = path)
        self.statusbar.clearMessage()
        self.statusbar.showMessage('The data file is saved at {}'.format(path))

    #save data plus best fit profile
    def save_data(self):
        potential, done = QInputDialog.getDouble(self, 'Potential_info', 'Enter the potential for this dataset (in V):')
        if not done:
            potential = None
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "data file (*.*)")
        if path!="":
            keys_attri = ['x','y','y_sim','error']
            keys_extra = ['h','k','Y','dL','LB']
            lib_map = {'x': 'L', 'y':'I','y_sim':'I_model','error':'error','h':'H','k':'K','Y':'Y','dL':'dL','LB':'LB'}
            export_data = {}
            for key in ['x','h','k','y','y_sim','error','Y','dL','LB']:
                export_data[lib_map[key]] = []
            export_data['use'] = []
            export_data['I_bulk'] = []
            export_data['potential'] = []
            for each in self.model.data:
                if each.use:
                    for key in ['x','h','k','y','y_sim','error','Y','dL','LB']:
                        if key in keys_attri:
                            export_data[lib_map[key]] = np.append(export_data[lib_map[key]], getattr(each,key))
                        elif key in keys_extra:
                            export_data[lib_map[key]] = np.append(export_data[lib_map[key]], each.extra_data[key])
                    export_data['use'] = np.append(export_data['use'],[True]*len(each.x))
                else:
                    for key in ['x','h','k','y','y_sim','error','Y','dL','LB']:
                        if key in keys_attri:
                            if key=='y_sim':
                                export_data[lib_map[key]] = np.append(export_data[lib_map[key]], [0]*len(getattr(each,'x')))
                            else:
                                export_data[lib_map[key]] = np.append(export_data[lib_map[key]], getattr(each,key))
                        elif key in keys_extra:
                            export_data[lib_map[key]] = np.append(export_data[lib_map[key]], each.extra_data[key])
                    export_data['use'] = np.append(export_data['use'],[False]*len(each.x))
                export_data['potential'] = np.append(export_data['potential'],[float(potential)]*len(each.x))
                beta = self.model.script_module.rgh.beta
                #rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(each.x-each.extra_data['LB'])/each.extra_data['dL'])**2)**0.5
                scale_factor = [self.model.script_module.rgh.scale_nonspecular_rods,self.model.script_module.rgh.scale_specular_rod][int("00L" in each.name)]
                rough = 1
                export_data['I_bulk'] = np.append(export_data['I_bulk'],rough**2*np.array(self.model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.x)**2*scale_factor**2))
            df_export_data = pd.DataFrame(export_data)
            writer_temp = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))])
            df_export_data.to_excel(writer_temp, columns =['potential']+[lib_map[each_] for each_ in ['x','h','k','y','y_sim','error']]+['I_bulk','use'])
            writer_temp.save()
            writer_temp.close()
            #also save loadable csv file
            df_export_data.to_csv([path+'.csv',path][int(path.endswith('.csv'))],sep="\t",columns=['L','H','K','Y','I','error','LB','dL'],\
                                 index=False, header=['#L','H','K','Y','I','error','LB','dL'])

    #not implemented!
    def change_plot_style(self):
        if self.background_color == 'w':
            self.widget_data.getViewBox().setBackgroundColor('k')
            self.widget_edp.getViewBox().setBackgroundColor('k')
            # self.widget_msv_top.getViewBox().setBackgroundColor('k')
            self.background_color = 'k'
        else:
            self.widget_data.getViewBox().setBackgroundColor('w')
            self.widget_edp.getViewBox().setBackgroundColor('w')
            # self.widget_msv_top.getViewBox().setBackgroundColor('w')
            self.background_color = 'w'

    def load_script(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","script Files (*.py);;text Files (*.txt)", options=options)
        if fileName:
            with open(fileName,'r') as f:
                self.plainTextEdit_script.setPlainText(f.read())
        self.model.script = (self.plainTextEdit_script.toPlainText())

    def update_script_upon_load(self):
        self.plainTextEdit_script.setPlainText(self.model.script)

    def save_script(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save script file", "", "script file (*.py)")
        with open(path,'w') as f:
            f.write(self.model.script)

    def modify_script(self):
        """
        Modify script based on the specified sorbates and total domain numbers
        To use this function, your script file should be standadized to have 
        tags to specifpy the code block position where you define the sorbates.
        This func is customized to modify script_model_standard.py.
        """
        assert self.model.script!="","No script to work on, please load script first!"
        domain_num = int(self.lineEdit_domain_number.text().rstrip())
        motif_chain = self.lineEdit_sorbate_motif.text().strip().rsplit(",")

        assert domain_num == len(motif_chain), "Number of domain not match with the motif number. Fix it first!!"
        lines = script_block_modifier(self.model.script.rsplit("\n"), 'slabnumber',["num_surface_slabs"],[domain_num])

        els_sorbate = []
        anchor_index_list = []
        flat_down_index = []
        xyzu_oc_m = []
        structure = []
        for each in motif_chain:
            each = each.strip()
            properties_temp = getattr(sorbate_tool,each)
            for each_key in properties_temp:
                if each_key == "els_sorbate":
                    els_sorbate.append(properties_temp[each_key])
                elif each_key == "anchor_index_list":
                    anchor_index_list.append(properties_temp[each_key])
                elif each_key == "flat_down_index":
                    flat_down_index.append(properties_temp[each_key])
                elif each_key == "structure":
                    structure.append("#"+each+properties_temp[each_key])
        xyzu_oc_m = [[0.5, 0.5, 1.5, 0.1, 1, 1]]*len(els_sorbate)
        tag_list = ['els_sorbate', 'anchor_index_list', 'flat_down_index', 'xyzu_oc_m']
        tag_value_list = [els_sorbate, anchor_index_list, flat_down_index, xyzu_oc_m]
        lines = script_block_modifier(lines, 'sorbateproperties',tag_list, tag_value_list)
        left_, right_ = locate_tag(lines,'sorbatestructure')
        del(lines[left_:right_])
        if structure[-1][-1] == "\n":
            structure[-1] = structure[-1][0:-1]
        lines.insert(left_,"\n".join(structure))

        self.model.script = '\n'.join(lines)
        self.plainTextEdit_script.setPlainText(self.model.script)

    def remove_selected_rows(self):
        # Delete the selected mytable lines
        self._deleteRows(self.tableWidget_pars.selectionModel().selectedRows(), self.tableWidget_pars)
        self.update_model_parameter()

    # DeleteRows function
    def _deleteRows(self, rows, table):
            # Get all row index
            indexes = []
            for row in rows:
                indexes.append(row.row())

            # Reverse sort rows indexes
            indexes = sorted(indexes, reverse=True)

            # Delete rows
            for rowidx in indexes:
                table.removeRow(rowidx)

    def append_one_row(self):
        rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(rows) == 0:
            row_index = self.tableWidget_pars.rowCount()
        else:
            row_index = rows[-1].row()
        self.tableWidget_pars.insertRow(row_index+1)
        for i in range(6):
            if i==2:
                check_box = QCheckBox()
                check_box.setChecked(False)
                self.tableWidget_pars.setCellWidget(row_index+1,2,check_box)
            else:
                qtablewidget = QTableWidgetItem('')
                if i == 0:
                    qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                elif i == 1:
                    qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                self.tableWidget_pars.setItem(row_index+1,i,qtablewidget)
        self.update_model_parameter()

    def append_one_row_at_the_end(self):
        row_index = self.tableWidget_pars.rowCount()
        self.tableWidget_pars.insertRow(row_index)
        for i in range(6):
            if i==2:
                check_box = QCheckBox()
                check_box.setChecked(False)
                self.tableWidget_pars.setCellWidget(row_index,2,check_box)
            else:
                qtablewidget = QTableWidgetItem('')
                if i == 0:
                    qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                elif i == 1:
                    qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                self.tableWidget_pars.setItem(row_index,i,qtablewidget)
        self.update_model_parameter()

    def update_model_parameter(self):
        """After you made changes in the par table, this func is executed to update the par values in model"""
        self.model.parameters.data = []
        vertical_label = []
        label_tag=1
        for i in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.item(i,0)==None:
                items = ['',0,False,0,0,'-']
                vertical_label.append('')
            elif self.tableWidget_pars.item(i,0).text()=='':
                items = ['',0,False,0,0,'-']
                vertical_label.append('')
            else:
                items = [self.tableWidget_pars.item(i,0).text(),float(self.tableWidget_pars.item(i,1).text()),self.tableWidget_pars.cellWidget(i,2).isChecked(),\
                         float(self.tableWidget_pars.item(i,3).text()), float(self.tableWidget_pars.item(i,4).text()), self.tableWidget_pars.item(i,5).text()]
                self.model.parameters.data.append(items)
                vertical_label.append(str(label_tag))
                label_tag += 1
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_label)

    def fit_all(self):
        """fit all fit parameters
        """
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            try:
                self.tableWidget_pars.cellWidget(i,2).setChecked(True)
            except:
                pass
        self.update_model_parameter()

    def fit_next_5(self):
        """fit next 5 parameters starting from first selected row
        """
        num_rows_table = 5
        rows = self.tableWidget_pars.selectionModel().selectedRows()
        starting_row = 0
        if len(rows)!=0:
            starting_row = rows[0].row()

        for i in range(num_rows_table):
            try:
                self.tableWidget_pars.cellWidget(i+starting_row,2).setChecked(True)
            except:
                pass
        self.update_model_parameter()

    def fit_none(self):
        """fit none of parameters
        """
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            try:
                self.tableWidget_pars.cellWidget(i,2).setChecked(False)
            except:
                pass
        self.update_model_parameter()

    def fit_selected(self):
        """fit selected parameters
        """
        selected_row_index = [each.row() for each in self.tableWidget_pars.selectionModel().selectedRows()]
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            if i in selected_row_index:
                try:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(True)
                except:
                    pass
            else:
                try:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(False)
                except:
                    pass
        self.update_model_parameter()

    def invert_fit(self):
        """invert the selection of fit parameters
        """
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            try:
                checkstate = self.tableWidget_pars.cellWidget(i,2).checkState()
                if checkstate == 0:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(True)
                else:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(False)
            except:
                pass
        self.update_model_parameter()

    def update_par_upon_load(self):
        """upon loading model, the par table widget content will be updated with this func"""
        vertical_labels = []
        lines = self.model.parameters.data
        how_many_pars = len(lines)
        self.tableWidget_pars.clear()
        self.tableWidget_pars.setRowCount(how_many_pars)
        self.tableWidget_pars.setColumnCount(6)
        self.tableWidget_pars.setHorizontalHeaderLabels(['Parameter','Value','Fit','Min','Max','Error'])
        for i in range(len(lines)):
            items = lines[i]
            j = 0
            if items[0] == '':
                vertical_labels.append('')
                j += 1
            else:
                #add items to table view
                if len(vertical_labels)==0:
                    vertical_labels.append('1')
                else:
                    #if vertical_labels[-1] != '':
                    jj=0
                    while vertical_labels[-1-jj]=='':
                        jj = jj + 1
                    vertical_labels.append('{}'.format(int(vertical_labels[-1-jj])+1))

                    #vertical_labels.append('{}'.format(int(vertical_labels[-1])+1))
                    #else:
                    #    vertical_labels.append('{}'.format(int(vertical_labels[-2])+1))
                for item in items:
                    if j == 2:
                        check_box = QCheckBox()
                        check_box.setChecked(item==True)
                        self.tableWidget_pars.setCellWidget(i,2,check_box)
                    else:
                        if j == 1:
                            qtablewidget = QTableWidgetItem(str(round(item,5)))
                        else:
                            qtablewidget = QTableWidgetItem(str(item))
                        if j == 0:
                            qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                        elif j == 1:
                            qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                        self.tableWidget_pars.setItem(i,j,qtablewidget)
                    j += 1
        # self.tableWidget_pars.resizeColumnsToContents()
        # self.tableWidget_pars.resizeRowsToContents()
        """
        header = self.tableWidget_pars.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        """
        self.tableWidget_pars.setShowGrid(True)
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_labels)

    def _load_par(self):
        vertical_labels = []
        self.tableWidget_pars.setRowCount(1)
        self.tableWidget_pars.setColumnCount(6)
        self.tableWidget_pars.setHorizontalHeaderLabels(['Parameter','Value','Fit','Min','Max','Error'])
        items = ['par',0,'False',0,0,'-']
        for i in [0]:
            j = 0
            if items[0] == '':
                self.model.parameters.data.append([items[0],0,False,0, 0,'-'])
                vertical_labels.append('')
                j += 1
            else:
                #add items to parameter attr
                self.model.parameters.data.append([items[0],float(items[1]),items[2]=='True',float(items[3]), float(items[4]),items[5]])
                #add items to table view
                if len(vertical_labels)==0:
                    vertical_labels.append('1')
                else:
                    if vertical_labels[-1] != '':
                        vertical_labels.append('{}'.format(int(vertical_labels[-1])+1))
                    else:
                        vertical_labels.append('{}'.format(int(vertical_labels[-2])+1))
                for item in items:
                    if j == 2:
                        check_box = QCheckBox()
                        check_box.setChecked(item=='True')
                        self.tableWidget_pars.setCellWidget(i,2,check_box)
                    else:
                        qtablewidget = QTableWidgetItem(item)
                        # qtablewidget.setTextAlignment(Qt.AlignCenter)
                        if j == 0:
                            qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                        elif j == 1:
                            qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                        self.tableWidget_pars.setItem(i,j,qtablewidget)
                    j += 1
        # self.tableWidget_pars.resizeColumnsToContents()
        # self.tableWidget_pars.resizeRowsToContents()
        self.tableWidget_pars.setShowGrid(True)
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_labels)

    def load_par(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Table Files (*.tab);;text Files (*.txt)", options=options)
        vertical_labels = []
        if fileName:
            with open(fileName,'r') as f:
                lines = f.readlines()
                lines = [each for each in lines if not each.startswith('#')]
                how_many_pars = len(lines)
                self.tableWidget_pars.setRowCount(how_many_pars)
                self.tableWidget_pars.setColumnCount(6)
                self.tableWidget_pars.setHorizontalHeaderLabels(['Parameter','Value','Fit','Min','Max','Error'])
                for i in range(len(lines)):
                    line = lines[i]
                    items = line.rstrip().rsplit('\t')
                    j = 0
                    if items[0] == '':
                        self.model.parameters.data.append([items[0],0,False,0, 0,'-'])
                        vertical_labels.append('')
                        j += 1
                    else:
                        #add items to parameter attr
                        self.model.parameters.data.append([items[0],float(items[1]),items[2]=='True',float(items[3]), float(items[4]),items[5]])
                        #add items to table view
                        if len(vertical_labels)==0:
                            vertical_labels.append('1')
                        else:
                            if vertical_labels[-1] != '':
                                vertical_labels.append('{}'.format(int(vertical_labels[-1])+1))
                            else:
                                vertical_labels.append('{}'.format(int(vertical_labels[-2])+1))
                        for item in items:
                            if j == 2:
                                check_box = QCheckBox()
                                check_box.setChecked(item=='True')
                                self.tableWidget_pars.setCellWidget(i,2,check_box)
                            else:
                                qtablewidget = QTableWidgetItem(item)
                                # qtablewidget.setTextAlignment(Qt.AlignCenter)
                                if j == 0:
                                    qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                                elif j == 1:
                                    qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                                self.tableWidget_pars.setItem(i,j,qtablewidget)
                            j += 1
        # self.tableWidget_pars.resizeColumnsToContents()
        # self.tableWidget_pars.resizeRowsToContents()
        """
        header = self.tableWidget_pars.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        """
        self.tableWidget_pars.setShowGrid(True)
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_labels)

    def update_par_upon_change(self):
        """will be executed before simulation"""
        self.model.parameters.data = []
        for each_row in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.item(each_row,0)==None:
                items = ['',0,False,0,0,'-']
            elif self.tableWidget_pars.item(each_row,0).text()=='':
                items = ['',0,False,0,0,'-']
            else:
                items = [self.tableWidget_pars.item(each_row,0).text()] + [float(self.tableWidget_pars.item(each_row,i).text()) for i in [1,3,4]] + [self.tableWidget_pars.item(each_row,5).text()]
                items.insert(2, self.tableWidget_pars.cellWidget(each_row,2).isChecked())
            self.model.parameters.data.append(items)

    @QtCore.pyqtSlot(str,object,bool)
    def update_par_during_fit(self,string,model,save_tag):
        """slot func to update par table widgets during fit"""
        for i in range(len(model.parameters.data)):
            if model.parameters.data[i][0]!='':
                item_temp = self.tableWidget_pars.item(i,1)
                item_temp.setText(str(round(model.parameters.data[i][1],5)))
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()
        self.tableWidget_pars.setShowGrid(False)

    @QtCore.pyqtSlot(str,object,bool)
    def update_status(self,string,model,save_tag):
        """slot func to update status info displaying fit status"""
        self.statusbar.clearMessage()
        self.statusbar.showMessage(string)
        self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,round(self.run_fit.solver.optimizer.best_fom,5)))
        if save_tag:
            self.auto_save_model()

    @QtCore.pyqtSlot(str,object,bool)
    def update_status_batch(self,string,model,save_tag):
        """slot func to update status info displaying fit status"""
        self.statusbar.clearMessage()
        self.statusbar.showMessage(string)
        self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,round(self.run_batch.solver.optimizer.best_fom,5)))
        if save_tag:
            self.auto_save_model()

    def save_par(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save tab file", "", "table file (*.*)")
        with open(path,'w') as f:
            f.write(self.model.parameters.get_ascii_output())

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    #get dpi info: dots per inch
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    myWin = MyMainWindow()
    myWin.setWindowIcon(QtGui.QIcon('DAFY.png'))
    hightlight = syntax_pars.PythonHighlighter(myWin.plainTextEdit_script.document())
    myWin.plainTextEdit_script.show()
    myWin.plainTextEdit_script.setPlainText(myWin.plainTextEdit_script.toPlainText())
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())
