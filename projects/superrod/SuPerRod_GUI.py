import sys,os, qdarkstyle
import traceback
from io import StringIO
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5 import uic
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
matplotlib.use("Qt5Agg")
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton, QTableWidgetItem, QHeaderView, QAbstractItemView, QInputDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTransform, QFont, QBrush, QColor, QIcon
from pyqtgraph.Qt import QtGui
import syntax_pars
from models.structure_tools import sorbate_tool
# from chemlab.graphics.renderers import AtomRenderer
# from chemlab.db import ChemlabDB

#from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

class RunFit(QtCore.QObject):

    updateplot = QtCore.pyqtSignal(str,object,bool)
    def __init__(self,solver):
        super(RunFit, self).__init__()
        self.solver = solver
        self.running = False

    def run(self):
        self.running = True
        self.solver.optimizer.stop = False
        self.solver.StartFit(self.updateplot)

    def stop(self):
        self.running = False
        self.solver.optimizer.stop = True

class MyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major', background = 'k')
        pg.mkQApp()
        uic.loadUi(os.path.join(DaFy_path,'projects','SuperRod','superrod3.ui'),self)
        self.setWindowTitle('Data analysis factory: CTR data modeling')
        icon = QIcon(os.path.join(script_path,"DAFY.png"))
        self.setWindowIcon(icon)
        self.comboBox_all_motif.insertItems(0, sorbate_tool.ALL_MOTIF_COLLECTION)
        # self.show()
        self.stop = False
        self.show_checkBox_list = []
        self.domain_tag = 1
        self.data_profiles = []
        #set fom_func
        #self.fom_func = chi2bars_2
        #parameters
        #self.parameters = parameters.Parameters()
        #scripts
        #self.script = ''
        #script module
        #self.script_module = types.ModuleType('genx_script_module')
        self.model = model.Model()
        # self.solver = solvergui.SolverController(self)
        self.run_fit = RunFit(solvergui.SolverController(self.model))
        self.fit_thread = QtCore.QThread()
        # self.structure_view_thread = QtCore.QThread()
        # self.widget_edp.moveToThread(self.structure_view_thread)
        self.run_fit.moveToThread(self.fit_thread)
        #self.run_fit.updateplot.connect(self.update_plot_data_view_upon_simulation)
        self.run_fit.updateplot.connect(self.update_par_during_fit)
        self.run_fit.updateplot.connect(self.update_status)
        #self.run_fit.updateplot.connect(self.update_structure_view)
        # self.run_fit.updateplot.connect(self.start_timer_structure_view)


        self.fit_thread.started.connect(self.run_fit.run)

        #tool bar buttons to operate modeling
        self.actionNew.triggered.connect(self.init_new_model)
        self.actionOpen.triggered.connect(self.open_model)
        self.actionSave.triggered.connect(self.save_model)
        self.actionSimulate.triggered.connect(self.simulate_model)
        self.actionRun.triggered.connect(self.run_model)
        self.actionStop.triggered.connect(self.stop_model)
        self.actionCalculate.triggered.connect(self.calculate_error_bars)
        #menu items
        self.actionOpen_model.triggered.connect(self.open_model)
        self.actionSave_model.triggered.connect(self.save_model)
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
        # self.pushButton_calculate.clicked.connect(self.calculate)
        self.pushButton_update_mask.clicked.connect(self.update_mask_info_in_data)

        #pushbuttons for structure view
        self.pushButton_azimuth_0.clicked.connect(self.azimuth_0)
        self.pushButton_azimuth_90.clicked.connect(self.azimuth_90)
        self.pushButton_elevation_0.clicked.connect(self.elevation_0)
        self.pushButton_elevation_90.clicked.connect(self.elevation_90)
        self.pushButton_parallel.clicked.connect(self.parallel_projection)
        self.pushButton_projective.clicked.connect(self.projective_projection)

        #spinBox to save the domain_tag
        self.spinBox_domain.valueChanged.connect(self.update_domain_index)

        #pushbutton for changing plotting style
        # self.pushButton_toggle_bkg_color.clicked.connect(self.change_plot_style)
        #pushbutton to load/save script
        self.pushButton_load_script.clicked.connect(self.load_script)
        self.pushButton_save_script.clicked.connect(self.save_script)
        self.pushButton_modify_script.clicked.connect(self.modify_script)
        #pushbutton to load/save parameter file
        self.pushButton_load_table.clicked.connect(self.load_par)
        self.pushButton_save_table.clicked.connect(self.save_par)
        self.pushButton_remove_rows.clicked.connect(self.remove_selected_rows)
        self.pushButton_add_one_row.clicked.connect(self.append_one_row)
        self.pushButton_update_plot.clicked.connect(self.update_structure_view)
        self.pushButton_update_plot.clicked.connect(self.update_plot_data_view_upon_simulation)
        self.pushButton_update_plot.clicked.connect(self.update_par_bar_during_fit)
        self.pushButton_add_par_set.clicked.connect(self.append_par_set)
        self.pushButton_add_all_pars.clicked.connect(self.append_all_par_sets)
        #select dataset in the viewer
        self.comboBox_dataset.activated.connect(self.update_data_view)

        #syntax highlight
        self.plainTextEdit_script.setStyleSheet("""QPlainTextEdit{
	                            font-family:'Consolas';
                                font-size:14pt;
	                            color: #ccc;
	                            background-color: #2b2b2b;}""")
        self.plainTextEdit_script.setTabStopWidth(self.plainTextEdit_script.fontMetrics().width(' ')*4)
        #self.data = data.DataList()

        #table view for parameters set to selecting row basis
        # self.tableWidget_pars.itemChanged.connect(self.update_par_upon_change)
        self.tableWidget_pars.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget_data.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.timer_save_data = QtCore.QTimer(self)
        self.timer_save_data.timeout.connect(self.save_model)
        self.timer_update_structure = QtCore.QTimer(self)
        self.timer_update_structure.timeout.connect(self.pushButton_update_plot.click)
        self.setup_plot()

    def toggle_data_panel(self):
        self.tabWidget_data.setVisible(self.actionData.isChecked())

    def toggle_plot_panel(self):
        self.tabWidget.setVisible(self.actionPlot.isChecked())

    def toggle_script_panel(self):
        self.tabWidget_2.setVisible(self.actionScript.isChecked())

    def update_domain_index(self):
        self.domain_tag = int(self.spinBox_domain.text())
        if self.model.compiled:
            self.widget_edp.items = []
            self.widget_msv_top.items = []
            self.init_structure_view()
        else:
            pass

    def parallel_projection(self):
        self.widget_edp.opts['distance'] = 2000
        self.widget_edp.opts['fov'] = 1
        self.widget_msv_top.opts['distance'] = 2000
        self.widget_msv_top.opts['fov'] = 1
        self.update_structure_view()


    def projective_projection(self):
        self.widget_edp.opts['distance'] = 25
        self.widget_edp.opts['fov'] = 60
        self.widget_msv_top.opts['distance'] = 25
        self.widget_msv_top.opts['fov'] = 60
        self.update_structure_view()

    def update_camera_position(self,widget_name = 'widget_edp', angle_type="azimuth", angle=0):
        #getattr(self,widget_name)
        getattr(self,widget_name).setCameraPosition(pos=None, distance=None, \
            elevation=[None,angle][int(angle_type=="elevation")], \
                azimuth=[None,angle][int(angle_type=="azimuth")])

    def azimuth_0(self):
        self.update_camera_position(angle_type="azimuth", angle=0)

    def azimuth_90(self):
        self.update_camera_position(angle_type="azimuth", angle=90)

    def elevation_0(self):
        self.update_camera_position(angle_type="elevation", angle=0)

    def elevation_90(self):
        self.update_camera_position(angle_type="elevation", angle=90)

    #do this after model is loaded, so that you know len(data)
    def update_plot_dimension(self, columns = 2):
        self.widget_data.clear()
        self.widget_data.ci.currentRow = 0
        self.widget_data.ci.currentCol = 0

        self.data_profiles = []
        self.data_error_bars = []
        total_datasets = len(self.model.data)
        #current list of ax handle
        # ax_list_now = list(range(len(self.data_profiles)))
        for i in range(total_datasets):
            # if i not in ax_list_now:
            if 1:
                hk_label = '{}{}L'.format(str(int(self.model.data[i].extra_data['h'][0])),str(int(self.model.data[i].extra_data['k'][0])))
                if (i%columns)==0 and (i!=0):
                    self.widget_data.nextRow()
                    self.data_profiles.append(self.widget_data.addPlot(title=hk_label))
                else:
                    self.data_profiles.append(self.widget_data.addPlot(title=hk_label))
                #error bar item
                # err = pg.ErrorBarItem(x=np.array([0]),y=np.array([0]),height=np.array([0]))
                # self.data_profiles[i].addItem(err)
                # self.data_error_bars.append(err)

    def setup_plot(self):
        self.fom_evolution_profile = self.widget_fom.addPlot()
        self.par_profile = self.widget_pars.addPlot()
        self.fom_scan_profile = self.widget_fom_scan.addPlot()

    def update_data_check_attr(self):
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

    def update_plot_data_view(self):
        if self.model.compiled:
            self.update_data_check_attr()
            self.update_plot_data_view_upon_simulation()
        else:
            # plot_data_index = []
            for i in range(len(self.model.data)):
                # if self.tableWidget_data.cellWidget(i,1).isChecked():
                fmt = self.tableWidget_data.item(i,4).text()
                fmt_symbol = list(fmt.rstrip().rsplit(';')[0].rsplit(':')[1])
                # self.selected_data_profile.plot(self.model.data[i].x, self.model.data[i].y,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2], clear = (len(plot_data_index) == 0))
                self.data_profiles[i].plot(self.model.data[i].x, self.model.data[i].y,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2], clear = True)
                #plot_data_index.append(i)
            [each.setLogMode(x=False,y=True) for each in self.data_profiles]
            [each.autoRange() for each in self.data_profiles]
            #self.selected_data_profile.autoRange()

    def update_plot_data_view_upon_simulation(self):
        for i in range(len(self.model.data)):
            if 1:
                fmt = self.tableWidget_data.item(i,4).text()
                fmt_symbol = list(fmt.rstrip().rsplit(';')[0].rsplit(':')[1])
                line_symbol = list(fmt.rstrip().rsplit(';')[1].rsplit(':')[1])
                self.data_profiles[i].plot(self.model.data[i].x, self.model.data[i].y,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2],clear = True)
                if self.tableWidget_data.cellWidget(i,3).isChecked():
                    #create error bar data, graphiclayout widget doesn't have a handy api to plot lines along with error bars
                    #disable this while the model is running
                    if not self.run_fit.solver.optimizer.running:
                        x = np.append(self.model.data[i].x[:,np.newaxis],self.model.data[i].x[:,np.newaxis],axis=1)
                        y_d = self.model.data[i].y[:,np.newaxis] - self.model.data[i].error[:,np.newaxis]/2
                        y_u = self.model.data[i].y[:,np.newaxis] + self.model.data[i].error[:,np.newaxis]/2
                        y = np.append(y_d,y_u,axis = 1)
                        for ii in range(len(y)):
                            self.data_profiles[i].plot(x=x[ii],y=y[ii],pen={'color':'w', 'width':1},clear = False)
                if self.tableWidget_data.cellWidget(i,2).isChecked():
                    self.data_profiles[i].plot(self.model.data[i].x, self.model.data[i].y_sim,pen={'color': line_symbol[1], 'width': int(line_symbol[0])},  clear = False)
                else:
                    pass
                # plot_data_index.append(i)
        [each.setLogMode(x=False,y=True) for each in self.data_profiles]
        [each.autoRange() for each in self.data_profiles]
        # self.selected_data_profile.setLogMode(x=False,y=True)
        # self.selected_data_profile.autoRange()
        fom_log = np.array(self.run_fit.solver.optimizer.fom_log)
        #print(fom_log)
        self.fom_evolution_profile.plot(fom_log[:,0],fom_log[:,1],pen={'color': 'r', 'width': 2}, clear = True)
        self.fom_evolution_profile.autoRange()
        
    def update_par_bar_during_fit(self):
        if self.run_fit.running:
            par_max = self.run_fit.solver.optimizer.par_max
            par_min = self.run_fit.solver.optimizer.par_min
            vec_best = copy.deepcopy(self.run_fit.solver.optimizer.best_vec)
            vec_best = (vec_best-par_min)/(par_max-par_min)
            pop_vec = np.array(copy.deepcopy(self.run_fit.solver.optimizer.pop_vec))

            trial_vec_min =[]
            trial_vec_max =[]
            for i in range(len(par_max)):
                trial_vec_min.append((np.min(pop_vec[:,i])-par_min[i])/(par_max[i]-par_min[i]))
                trial_vec_max.append((np.max(pop_vec[:,i])-par_min[i])/(par_max[i]-par_min[i]))
            trial_vec_min = np.array(trial_vec_min)
            trial_vec_max = np.array(trial_vec_max)
            bg = pg.BarGraphItem(x=range(len(vec_best)), y=(trial_vec_max + trial_vec_min)/2, height=(trial_vec_max - trial_vec_min)/2, brush='b', width = 0.8)
            # best_ = pg.ScatterPlotItem(size=10, pen=(200,200,200), brush=pg.mkBrush(255, 255, 255, 120))
            # best_.addPoints([{'pos':range(len(vec_best)),'data':vec_best}])
            # print(trial_vec_min)
            # print(trial_vec_max)
            # print(par_min)
            # print(par_max)
            self.par_profile.clear()
            self.par_profile.addItem(bg)
            # self.par_profile.addItem(best_)
            # p1 = self.par_profile.addPlot()
            self.par_profile.plot(vec_best, pen=(0,0,0), symbolBrush=(255,0,0), symbolPen='w')
        else:
            pass

    def calculate_error_bars(self):
        try:
            error_bars = self.run_fit.solver.CalcErrorBars()
            total_num_par = len(self.model.parameters.data)
            index_list = [i for i in range(total_num_par) if self.model.parameters.data[i][2]]

            for i in range(len(error_bars)):
                self.model.parameters.data[index_list[i]][-1] = error_bars[i]
            
            self.update_par_upon_load()
        except diffev.ErrorBarError as e:
            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)


    def init_new_model(self):
        reply = QMessageBox.question(self, 'Message', 'Would you like to save the current model first?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.save_model()
        self.model = model.Model()
        self.run_fit.solver.model = self.model
        self.tableWidget_data.setRowCount(0)
        self.tableWidget_pars.setRowCount(0)
        self.plainTextEdit_script.setPlainText('')
        self.comboBox_dataset.clear()
        self.tabelWidget_data_view.setRowCount(0)
        self.update_plot_data_view()

    def open_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","rod file (*.rod);;zip Files (*.rar)", options=options)
        load_add_ = 'success'
        self.rod_file = fileName
        if fileName:
            self.setWindowTitle('Data analysis factory: CTR data modeling-->{}'.format(fileName))
            self.model.load(fileName)
            self.update_plot_dimension()
            # self.load_addition()
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
            self.widget_msv_top.items = []
            #update other pars
            self.update_table_widget_data()
            self.update_combo_box_dataset()
            self.update_plot_data_view()
            self.update_par_upon_load()
            self.update_script_upon_load()
            # self.init_structure_view()
            #model is simulated at the end of next step
            self.init_mask_info_in_data_upon_loading_model()
            #add name space for cal bond distance after simulation
            self.widget_terminal.update_name_space("report_distance",self.model.script_module.sample.inter_atom_distance_report)
            #now set the comboBox for par set
            self.update_combo_box_list_par_set()

            self.statusbar.clearMessage()
            self.statusbar.showMessage("Model is loaded, and {} in config loading".format(load_add_))
            # self.update_mask_info_in_data()

    def update_combo_box_list_par_set(self):
        attrs = self.model.script_module.__dir__()
        attr_wanted = [each for each in attrs if type(getattr(self.model.script_module, each)) in [AtomGroup, UserVars]]
        num_items = self.comboBox_register_par_set.count()
        for i in range(num_items):
            self.comboBox_register_par_set.removeItem(0)
        self.comboBox_register_par_set.insertItems(0,attr_wanted)

    def append_all_par_sets(self):
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
        #attrs = getattr(self.model.script_module, par_selected)
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
        #self.update_model_parameter()

    def auto_save_model(self):
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
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "rod file (*.rod);;zip files (*.rar)")
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

    #here save also the config pars for diffev solver
    def save_addition(self):
        values=\
                [self.widget_solver.par.param('Diff.Ev.').param('k_m').value(),
                self.widget_solver.par.param('Diff.Ev.').param('k_r').value(),
                self.widget_solver.par.param('Diff.Ev.').param('Method').value(),
                self.widget_solver.par.param('FOM').param('Figure of merit').value(),
                self.widget_solver.par.param('FOM').param('Auto save, interval').value(),
                self.widget_solver.par.param('Fitting').param('start guess').value(),
                self.widget_solver.par.param('Fitting').param('Generation size').value(),
                self.widget_solver.par.param('Fitting').param('Population size').value()]
        pars = ['k_m','k_r','Method','Figure of merit','Auto save, interval','start guess','Generation size','Population size']
        for i in range(len(pars)):
            self.model.save_addition(pars[i],str(values[i]))
    
    def load_addition(self):
            funcs=\
                [self.widget_solver.par.param('Diff.Ev.').param('k_m').setValue,
                self.widget_solver.par.param('Diff.Ev.').param('k_r').setValue,
                self.widget_solver.par.param('Diff.Ev.').param('Method').setValue,
                self.widget_solver.par.param('FOM').param('Figure of merit').setValue,
                self.widget_solver.par.param('FOM').param('Auto save, interval').setValue,
                self.widget_solver.par.param('Fitting').param('start guess').setValue,
                self.widget_solver.par.param('Fitting').param('Generation size').setValue,
                self.widget_solver.par.param('Fitting').param('Population size').setValue]

            types= [float,float,str,str,int,bool,int,int]
            pars = ['k_m','k_r','Method','Figure of merit','Auto save, interval','start guess','Generation size','Population size']
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
        self.update_par_upon_change()
        self.model.script = (self.plainTextEdit_script.toPlainText())
        self.widget_solver.update_parameter_in_solver(self)
        try:
            self.model.simulate()
            self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,self.model.fom))
            self.update_plot_data_view_upon_simulation()
            self.init_structure_view()
            self.statusbar.clearMessage()
            self.update_combo_box_list_par_set()
            self.statusbar.showMessage("Model is simulated successfully!")
        except model.ModelError as e:
            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)
            #print("test error message!")
            #print(str(e))

        '''
        self.compile_script()
        # self.update_pars()
        (funcs, vals) = self.get_sim_pars()
        # Set the parameter values in the model
        #[func(val) for func,val in zip(funcs, vals)]
        i = 0
        for func, val in zip(funcs,vals):
            try:
                func(val)
            except Exception as e:
                (sfuncs_tmp, vals_tmp) = self.parameters.get_sim_pars()
                raise ParameterError(sfuncs_tmp[i], i, str(e), 1)
            i += 1

        self.evaluate_sim_func()
        '''
        #print(self.widget_solver.par.param("Fitting").param("start guess").value())
        #print(self.widget_solver.par.param("Fitting").param("Population size").value())

    def run_model(self):
        # self.solver.StartFit()
        # self.start_timer_structure_view()
        # self.structure_view_thread.start()
        #button will be clicked every 2 second to update figures
        self.timer_update_structure.start(2000)
        self.widget_solver.update_parameter_in_solver(self)
        self.fit_thread.start()

    def stop_model(self):
        self.run_fit.stop()
        self.fit_thread.terminate()
        self.timer_update_structure.stop()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Model run is aborted!")
        # self.stop_timer_structure_view()

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
        #8 columns in total
        #X, H, K, Y, I, eI, LB, dL
        #for CTR data, X column is L column, Y column all 0
        #for RAXR data, X column is energy column, Y column is L column
        # self.data = data.DataList()
        #self.model.compiled = False
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
                data_loaded_pd.sort_values(by = ['h','k'], inplace = True)
                # print(data_loaded_pd)
                hk_unique = list(set(zip(list(data_loaded_pd['h']), list(data_loaded_pd['k']))))
                hk_unique.sort()
                h_unique = [each[0] for each in hk_unique]
                k_unique = [each[1] for each in hk_unique]

                for i in range(len(h_unique)):
                    h_temp, k_temp = h_unique[i], k_unique[i]
                    name = 'Data-{}{}L'.format(h_temp, k_temp)
                    tag = sum([int(name in each) for each in current_data_set_name])+1
                    #if name in current_data_set_name:
                    name = name + '_{}'.format(tag)
                    self.model.data.add_new(name = name)
                    self.model.data.items[-1].x = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['X'].to_numpy()
                    self.model.data.items[-1].y = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['I'].to_numpy()
                    self.model.data.items[-1].error = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['eI'].to_numpy()
                    self.model.data.items[-1].x_raw = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['X'].to_numpy()
                    self.model.data.items[-1].y_raw = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['I'].to_numpy()
                    self.model.data.items[-1].error_raw = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['eI'].to_numpy()
                    self.model.data.items[-1].set_extra_data(name = 'h', value = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['h'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'k', value = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['k'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'Y', value = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['Y'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'LB', value = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['LB'].to_numpy())
                    self.model.data.items[-1].set_extra_data(name = 'dL', value = data_loaded_pd[(data_loaded_pd['h']==h_temp) & (data_loaded_pd['k']==k_temp)]['dL'].to_numpy())
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
        #update script_module
        #self.model.script_module.__dict__['data'] = self.data
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
        #self._deleteRows(self.tableWidget_data.selectionModel().selectedRows(), self.tableWidget_data)
        self.update_table_widget_data()
        self.update_combo_box_dataset()
        self.update_plot_dimension()
        self.update_plot_data_view()

    def update_table_widget_data(self):
        self.tableWidget_data.clear()
        self.tableWidget_data.setRowCount(len(self.model.data))
        self.tableWidget_data.setColumnCount(5)
        self.tableWidget_data.setHorizontalHeaderLabels(['DataID','Show','Use','Errors','fmt'])
        # self.tableWidget_pars.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
                    check = getattr(current_data, ['show', 'use', 'use_error'][j-1])
                    check_box = QCheckBox()
                    #self.show_checkBox_list.append(check_box)
                    check_box.setChecked(check)
                    check_box.stateChanged.connect(self.update_plot_data_view)
                    self.tableWidget_data.setCellWidget(i,j,check_box)

    def update_combo_box_dataset(self):
        new_items = [each.name for each in self.model.data]
        self.comboBox_dataset.clear()
        self.comboBox_dataset.addItems(new_items)

    def update_data_view(self):
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
                    # print(getattr(dataset,'x')[i])
                    # qtablewidget = QTableWidgetItem(str(round(getattr(dataset,all_labels[j])[i],4)))
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
        # print(domain_tag)
        xyz = self.model.script_module.sample.extract_xyz(domain_tag)
        self.widget_edp.show_structure(xyz)
        self.update_camera_position(widget_name = 'widget_edp', angle_type="azimuth", angle=0)
        self.update_camera_position(widget_name = 'widget_edp', angle_type = 'elevation', angle = 0)
        xyz,bond_index = self.model.script_module.sample.extract_xyz_top(domain_tag)
        self.widget_msv_top.show_structure(xyz,bond_index)
        self.update_camera_position(widget_name = 'widget_msv_top', angle_type="azimuth", angle=0)
        self.update_camera_position(widget_name = 'widget_msv_top', angle_type = 'elevation', angle = 90)

    def update_structure_view(self):
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
            # print(size_domain,domain_tag)
            xyz = self.model.script_module.sample.extract_xyz(domain_tag)
            self.widget_edp.update_structure(xyz)
            xyz, bond_index = self.model.script_module.sample.extract_xyz_top(domain_tag)
            self.widget_msv_top.update_structure(xyz, bond_index)
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            _ = QMessageBox.question(self, "",'Runtime error message:\n{}'.format(str(val)), QMessageBox.Ok)


    def start_timer_structure_view(self):
        self.timer_update_structure.start(2000)

    def stop_timer_structure_view(self):
        self.timer_update_structure.stop()


    #save data plus best fit profile
    def save_data(self):
        #potential = input('The potential corresponding to this dataset is:')
        potential, done = QInputDialog.getDouble(self, 'Potential_info', 'Enter the potential for this dataset (in V):')
        if not done:
            potential = None
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "model file (*.*)")
        if path!="":
            keys_attri = ['x','y','y_sim','error']
            keys_extra = ['h','k']
            lib_map = {'x': 'L', 'y':'I','y_sim':'I_model','error':'error','h':'H','k':'K'}
            export_data = {}
            for key in ['x','h','k','y','y_sim','error']:
                export_data[lib_map[key]] = []
            export_data['use'] = []
            export_data['I_bulk'] = []
            export_data['potential'] = []
            for each in self.model.data:
                if each.use:
                    for key in ['x','h','k','y','y_sim','error']:
                        if key in keys_attri:
                            export_data[lib_map[key]] = np.append(export_data[lib_map[key]], getattr(each,key))
                        elif key in keys_extra:
                            export_data[lib_map[key]] = np.append(export_data[lib_map[key]], each.extra_data[key])
                    export_data['use'] = np.append(export_data['use'],[True]*len(each.x))
                else:
                    for key in ['x','h','k','y','y_sim','error']:
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
                rough = 1
                export_data['I_bulk'] = np.append(export_data['I_bulk'],rough**2*np.array(self.model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.x)**2))
            df_export_data = pd.DataFrame(export_data)
            writer_temp = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))])
            df_export_data.to_excel(writer_temp, columns =['potential']+[lib_map[each_] for each_ in ['x','h','k','y','y_sim','error']]+['I_bulk','use'])
            writer_temp.save()
            #self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='a')

    #not implemented!
    def change_plot_style(self):
        if self.background_color == 'w':
            self.widget_data.getViewBox().setBackgroundColor('k')
            self.widget_edp.getViewBox().setBackgroundColor('k')
            self.widget_msv_top.getViewBox().setBackgroundColor('k')
            self.background_color = 'k'
        else:
            self.widget_data.getViewBox().setBackgroundColor('w')
            self.widget_edp.getViewBox().setBackgroundColor('w')
            self.widget_msv_top.getViewBox().setBackgroundColor('w')
            self.background_color = 'w'

    def load_script(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","script Files (*.py);;text Files (*.txt)", options=options)
        if fileName:
            with open(fileName,'r') as f:
                self.plainTextEdit_script.setPlainText(f.read())
        self.model.script = (self.plainTextEdit_script.toPlainText())
        #self.compile_script()

    def update_script_upon_load(self):
        self.plainTextEdit_script.setPlainText(self.model.script)

    def save_script(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save script file", "", "script file (*.py)")
        with open(path,'w') as f:
            f.write(self.model.script)

    def modify_script(self):
        assert self.model.script!="","No script to work on, please load script first!"
        domain_num = int(self.lineEdit_domain_number.text().rstrip())
        motif_chain = self.lineEdit_sorbate_motif.text().strip().rsplit(",")
        #print(self.lineEdit_sorbate_motif.text().strip().rsplit(","))
        #print(self.lineEdit_sorbate_motif.text().strip().rsplit(","))

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

    def update_par_upon_load(self):

        vertical_labels = []
        lines = self.model.parameters.data
        how_many_pars = len(lines)
        self.tableWidget_pars.clear()
        self.tableWidget_pars.setRowCount(how_many_pars)
        self.tableWidget_pars.setColumnCount(6)
        self.tableWidget_pars.setHorizontalHeaderLabels(['Parameter','Value','Fit','Min','Max','Error'])
        # self.tableWidget_pars.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(len(lines)):
            # print("test{}".format(i))
            items = lines[i]
            #items = line.rstrip().rsplit('\t')
            j = 0
            if items[0] == '':
                #self.model.parameters.data.append([items[0],0,False,0, 0,'-'])
                vertical_labels.append('')
                j += 1
            else:
                #add items to parameter attr
                #self.model.parameters.data.append([items[0],float(items[1]),items[2]=='True',float(items[3]), float(items[4]),items[5]])
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
                        check_box.setChecked(item==True)
                        self.tableWidget_pars.setCellWidget(i,2,check_box)
                    else:
                        if j == 1:
                            qtablewidget = QTableWidgetItem(str(round(item,5)))
                        else:
                            qtablewidget = QTableWidgetItem(str(item))
                        # qtablewidget.setTextAlignment(Qt.AlignCenter)
                        if j == 0:
                            qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                        elif j == 1:
                            qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                        self.tableWidget_pars.setItem(i,j,qtablewidget)
                    j += 1
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()
        self.tableWidget_pars.setShowGrid(False)
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_labels)


    def load_par(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Table Files (*.tab);;text Files (*.txt)", options=options)
        vertical_labels = []
        if fileName:
            with open(fileName,'r') as f:
                lines = f.readlines()
                # self.parameters.set_ascii_input(f)
                lines = [each for each in lines if not each.startswith('#')]
                how_many_pars = len(lines)
                self.tableWidget_pars.setRowCount(how_many_pars)
                self.tableWidget_pars.setColumnCount(6)
                self.tableWidget_pars.setHorizontalHeaderLabels(['Parameter','Value','Fit','Min','Max','Error'])
                # self.tableWidget_pars.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()
        self.tableWidget_pars.setShowGrid(False)
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_labels)

    @QtCore.pyqtSlot(str,object,bool)
    def update_par_during_fit(self,string,model,save_tag):
        #labels = [data[0] for each in self.model.parameters.data]
        for i in range(len(model.parameters.data)):
            if model.parameters.data[i][0]!='':
                # print(self.model.parameters.data[i][0])
                #print(len(self.model.parameters.data))
                # print(model.parameters.data[i][0])
                item_temp = self.tableWidget_pars.item(i,1)
                #print(type(item_temp))
                item_temp.setText(str(round(model.parameters.data[i][1],5)))
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()
        self.tableWidget_pars.setShowGrid(False)
        # self.update_structure_view()

    def update_par_upon_change(self):
        #print("before update:{}".format(len(self.model.parameters.data)))
        self.model.parameters.data = []
        for each_row in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.item(each_row,0)==None:
                items = ['',0,False,0,0,'-']
            elif self.tableWidget_pars.item(each_row,0).text()=='':
                items = ['',0,False,0,0,'-']
            else:
                # print(each_row,type(self.tableWidget_pars.item(each_row,0)))
                items = [self.tableWidget_pars.item(each_row,0).text()] + [float(self.tableWidget_pars.item(each_row,i).text()) for i in [1,3,4]] + [self.tableWidget_pars.item(each_row,5).text()]
                items.insert(2, self.tableWidget_pars.cellWidget(each_row,2).isChecked())
            self.model.parameters.data.append(items)
        #print("after update:{}".format(len(self.model.parameters.data)))

    @QtCore.pyqtSlot(str,object,bool)
    def update_status(self,string,model,save_tag):
        self.statusbar.clearMessage()
        self.statusbar.showMessage(string)
        self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,round(self.run_fit.solver.optimizer.best_fom,5)))
        if save_tag:
            self.auto_save_model()

    def save_par(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save tab file", "", "table file (*.*)")
        with open(path,'w') as f:
            f.write(self.model.parameters.get_ascii_output())

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.setWindowIcon(QtGui.QIcon('DAFY.png'))
    hightlight = syntax_pars.PythonHighlighter(myWin.plainTextEdit_script.document())
    myWin.plainTextEdit_script.show()
    myWin.plainTextEdit_script.setPlainText(myWin.plainTextEdit_script.toPlainText())
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())
