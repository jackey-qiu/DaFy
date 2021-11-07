import sys,os,qdarkstyle,io
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog, QShortcut,QMessageBox
from PyQt5 import uic, QtWidgets
import PyQt5
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DaFy_CTR_BKG_class import run_app
try:
    from . import locate_path_ctr
except:
    import locate_path_ctr
#DaFy_path = locate_path.module_path_locator()
#script_path = os.path.join(DaFy_path,'projects','ctr')
script_path = locate_path_ctr.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
from VisualizationEnginePool import plot_bkg_fit_gui_pyqtgraph,replot_bkg_profile
from PlotSetup import overplot_ctr_temp
import time
import matplotlib
matplotlib.use("Qt5Agg")
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton
from PyQt5.QtGui import QTransform
from pyqtgraph.Qt import QtGui
import logging

# pg.setConfigOption('background', (50,50,100))
# pg.setConfigOption('foreground', 'k')

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
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        uic.loadUi(os.path.join(DaFy_path,'projects','ctr','CTR_bkg_pyqtgraph_new.ui'),self)
        #super().setupUi(self)
        self.widget_config.init_pars(data_type = self.comboBox_beamline.currentText())
        self.setWindowTitle('Data analysis factory: CTR data analasis')
        
        #set redirection of error message to embeted text browser widget
        logTextBox = QTextEditLogger(self.textBrowser_error_msg)
        # You can format what is printed to text box
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.DEBUG)

        self.app_ctr=run_app(beamline = self.comboBox_beamline.currentText())
        self.lineEdit.setText(os.path.join(self.app_ctr.data_path,'default.ini'))
        self.ref_data = None
        self.ref_fit_pars_current_point = {}
        #self.app_ctr.run()
        self.current_image_no = 0
        self.current_scan_number = None
        self.bkg_intensity = 0
        self.bkg_clip_image = None
        self.image_log_scale = False
        self.run_mode = False
        self.image_set_up = False
        self.tag_reprocess = False
        self.roi_pos = None
        self.roi_size = None

        #self.setupUi(self)
        self.stop = False
        self.open.clicked.connect(self.load_file)
        self.launch.clicked.connect(self.launch_file)
        #self.reload.clicked.connect(self.rload_file)
        #self.horizontalSlider.valueChanged.connect(self.change_peak_width)
        self.spinBox_peak_width.valueChanged.connect(self.change_peak_width)
        self.stopBtn.clicked.connect(self.stop_func)
        self.saveas.clicked.connect(self.save_file_as)
        self.save.clicked.connect(self.save_file)
        self.plot.clicked.connect(self.plot_figure)
        self.runstepwise.clicked.connect(self.set_tag_process_type)
        self.runstepwise.clicked.connect(self.plot_)
        self.pushButton_filePath.clicked.connect(self.locate_data_folder)
        self.pushButton_load_ref_data.clicked.connect(self.load_ref_data)
        self.lineEdit_data_file_name.setText('temp_data_ctr.xlsx')
        self.lineEdit_data_file_path.setText(self.app_ctr.data_path)
        self.actionOpenConfig.triggered.connect(self.load_file)
        self.actionSaveConfig.triggered.connect(self.save_file)
        self.actionRun.triggered.connect(self.set_tag_process_type)
        self.actionRun.triggered.connect(self.plot_)
        self.actionStop.triggered.connect(self.stop_func)
        self.actionSaveData.triggered.connect(self.save_data)
        self.pushButton_save_rod_data.clicked.connect(self.start_dL_BL_editor_dialog)
        setattr(self.app_ctr,'data_path',os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text()))
        for each in self.groupBox_2.findChildren(QCheckBox):
            each.released.connect(self.update_poly_order)
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            each.toggled.connect(self.update_cost_func)
        self.pushButton_remove_current_point.clicked.connect(self.remove_data_point)
        self.pushButton_left.clicked.connect(self.move_roi_left)
        self.pushButton_right.clicked.connect(self.move_roi_right)
        self.pushButton_up.clicked.connect(self.move_roi_up)
        self.pushButton_down.clicked.connect(self.move_roi_down)
        self.pushButton_set_roi.clicked.connect(self.set_roi)
        self.pushButton_go.clicked.connect(self.reprocess_previous_frame)
        self.comboBox_beamline.currentTextChanged.connect(self.change_config_layout)
        self.pushButton_track_peak.clicked.connect(self.track_peak)
        self.pushButton_set_peak.clicked.connect(self.set_peak)

        self.leftShort = QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.leftShort.activated.connect(self.move_roi_left)
        self.rightShort = QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.rightShort.activated.connect(self.move_roi_right)
        self.upShort = QShortcut(QtGui.QKeySequence("Ctrl+Up"), self)
        self.upShort.activated.connect(self.move_roi_up)
        self.downShort = QShortcut(QtGui.QKeySequence("Ctrl+Down"), self)
        self.downShort.activated.connect(self.move_roi_down)
        self.switch_roi_adjustment_type_short = QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.switch_roi_adjustment_type_short.activated.connect(self.switch_roi_adjustment_type)

        self.nextShort = QShortcut(QtGui.QKeySequence("Right"), self)
        self.nextShort.activated.connect(self.plot_)
        self.deleteShort = QShortcut(QtGui.QKeySequence("Down"), self)
        self.deleteShort.activated.connect(self.remove_data_point)

        # self.checkBox_use_log_scale.stateChanged.connect(self.set_log_image)
        self.radioButton_fixed_percent.clicked.connect(self.update_image)
        self.radioButton_fixed_between.clicked.connect(self.update_image)
        self.radioButton_automatic_set_hist.clicked.connect(self.update_image)
        self.lineEdit_scale_factor.returnPressed.connect(self.update_image)
        self.lineEdit_tailing_factor.returnPressed.connect(self.update_image)
        self.lineEdit_left.returnPressed.connect(self.update_image)
        self.lineEdit_right.returnPressed.connect(self.update_image)

        self.radioButton_traditional.toggled.connect(self.update_ss_factor)
        self.radioButton_vincent.toggled.connect(self.update_ss_factor)
        self.doubleSpinBox_ss_factor.valueChanged.connect(self.update_ss_factor)

        self.comboBox_p3.activated.connect(self.select_source_for_plot_p3)
        self.comboBox_p4.activated.connect(self.select_source_for_plot_p4)
        self.p3_data_source = self.comboBox_p3.currentText()
        self.p4_data_source = self.comboBox_p4.currentText()
        setattr(self.app_ctr,'p3_data_source',self.comboBox_p3.currentText())
        setattr(self.app_ctr,'p4_data_source',self.comboBox_p4.currentText())
        self.timer_save_data = QtCore.QTimer(self)

    def start_dL_BL_editor_dialog(self):
        dlg = start_editor_dialog(self)
        dlg.exec()

    def switch_roi_adjustment_type(self):
        if self.radioButton_roi_position.isChecked():
            self.radioButton_roi_size.setChecked(True)
        else:
            self.radioButton_roi_position.setChecked(True)

    def change_config_layout(self):
        self.widget_config.init_pars(data_type = self.comboBox_beamline.currentText())
        self.app_ctr.beamline = self.comboBox_beamline.currentText()

    def set_tag_process_type(self):
        self.tag_reprocess = False
        
    def reprocess_previous_frame(self):
        self.tag_reprocess = True
        frame_number = self.app_ctr.current_frame + 1 +int(self.lineEdit_frame_index_offset.text())
        if (frame_number < 0) or (frame_number > self.app_ctr.current_frame):
            self.tag_reprocess = False
            return
        img = self.app_ctr.img_loader.load_one_frame(frame_number = frame_number)
        img = self.app_ctr.create_mask_new.create_mask_new(img = img, img_q_ver = None,
                                  img_q_par = None, mon = self.app_ctr.img_loader.extract_transm_and_mon(frame_number))
        self.app_ctr.bkg_sub.img = img
        self.get_fit_pars_from_frame_index(int(self.lineEdit_frame_index_offset.text()))
        self.set_fit_pars_from_reference()
        self.reset_peak_center_and_width()
        self.update_image()
        
        selected = self.roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
        self.bkg_intensity = selected.mean()
        self.app_ctr.run_update_one_specific_frame(img, self.bkg_intensity, poly_func = ['Vincent','traditional'][int(self.radioButton_traditional.isChecked())], frame_offset = int(self.lineEdit_frame_index_offset.text()))
        # self.update_plot()
        self.update_ss_factor()
        #plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)

        self.statusbar.clearMessage()
        self.statusbar.showMessage('Working on scan{}: we are now at frame{} of {} frames in total!'.format(self.app_ctr.img_loader.scan_number,frame_number+1,self.app_ctr.img_loader.total_frame_number))
        self.progressBar.setValue((self.app_ctr.img_loader.frame_number+1)/float(self.app_ctr.img_loader.total_frame_number)*100)
        # self.lcdNumber_frame_number.display(self.app_ctr.img_loader.frame_number+1)
        try:
            self.lcdNumber_speed.display(int(1./(time.time()-t0)))
        except:
            pass
    
    def set_log_image(self):
        if self.checkBox_use_log_scale.isChecked():
            self.image_log_scale = True
            self.update_image()
        else:
            self.image_log_scale = False
            self.update_image()

    #to fold or unfold the config file editor
    def fold_or_unfold(self):
        text = self.pushButton_fold_or_unfold.text()
        if text == "<":
            self.frame.setVisible(False)
            self.pushButton_fold_or_unfold.setText(">")
        elif text == ">":
            self.frame.setVisible(True)
            self.pushButton_fold_or_unfold.setText("<")

    def change_peak_width(self):
        # self.lineEdit_peak_width.setText(str(self.horizontalSlider.value()))
        self.app_ctr.bkg_sub.peak_width = int(self.spinBox_peak_width.value())
        self.updatePlot()

    def save_data(self):
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        try:
            self.app_ctr.save_data_file(data_file)
            self.statusbar.showMessage('Data file is saved as {}!'.format(data_file))
        except:
            self.statusbar.showMessage('Failure to save data file!')
            logging.getLogger().exception('Fatal to save datafile:')

    def remove_data_point(self):
        self.app_ctr.data['mask_ctr'][-1]=False
        self.statusbar.showMessage('Current data point is masked!')
        self.updatePlot2()

    def select_source_for_plot_p3(self):
        self.app_ctr.p3_data_source = self.comboBox_p3.currentText()
        self.updatePlot()

    def select_source_for_plot_p4(self):
        self.app_ctr.p4_data_source = self.comboBox_p4.currentText()
        self.updatePlot()

    def update_poly_order(self, init_step = False):
        ord_total = 0
        i=1
        for each in self.groupBox_2.findChildren(QCheckBox):
            ord_total += int(bool(each.checkState()))*int(each.text())
            i+=i
        self.app_ctr.bkg_sub.update_integration_order(ord_total)
        #print(self.app_ctr.bkg_sub.ord_cus_s)
        
        if not init_step:
            self.updatePlot()

    def update_cost_func(self, init_step = False):
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            if each.isChecked():
                self.app_ctr.bkg_sub.update_integration_function(each.text())
                break
        try:
            self.updatePlot()
        except:
            pass

    def update_ss_factor(self, init_step = False):
        self.app_ctr.bkg_sub.update_ss_factor(self.doubleSpinBox_ss_factor.value())
        #print(self.app_ctr.bkg_sub.ss_factor)
        try:
            self.updatePlot()
        except:
            pass

    def _check_roi_boundary(self,pos,size):
        ver,hor = self.app_ctr.cen_clip
        pos_bound_x = hor*2
        pos_bound_y = ver*2
        pos_return = []
        size_return = []
        if pos[0]<0:
            pos_return.append(0)
        elif pos[0]>pos_bound_x:
            pos_return.append(pos_bound_x-10)
        else:
            pos_return.append(pos[0])

        if pos[1]<0:
            pos_return.append(0)
        elif pos[1]>pos_bound_y:
            pos_return.append(pos_bound_y-10)
        else:
            pos_return.append(pos[1]) 

        if size[0]<1:
            size_return.append(1)
        elif size[0]+pos_return[0]>pos_bound_x:
            size_return.append(pos_bound_x-pos_return[0])
        else:
            size_return.append(size[0])

        if size[1]<1:
            size_return.append(1)
        elif size[1]+pos_return[1]>pos_bound_y:
            size_return.append(pos_bound_y-pos_return[1])
        else:
            size_return.append(size[1])
        return pos_return,size_return

    def load_ref_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;text Files (*.csv)", options=options)
        self.lineEdit_ref_data_path.setText(fileName)
        #self.lineEdit_data_file.setText(fileName)
        if fileName != "":
            try:
                self.ref_data = pd.read_excel(fileName)
            except:
                print("Failure to load ref data file!!")

    def get_fit_pars_from_reference(self, match_scan = False):
        if type(self.ref_data) != pd.DataFrame:
            self.ref_fit_pars_current_point = {}
            return
        current_H, current_K, current_L = self.app_ctr.img_loader.hkl
        current_scan = self.app_ctr.img_loader.scan_number
        current_H = int(round(current_H,0))
        current_K = int(round(current_K,0))
        condition = (self.ref_data["H"] == current_H) & (self.ref_data["K"] == current_K)
        if hasattr(self,'checkBox_match_scan'):
            match_scan = self.checkBox_match_scan.isChecked()
        if match_scan:
            condition = (self.ref_data["H"] == current_H) & (self.ref_data["K"] == current_K) & (self.ref_data["scan_no"] == current_scan)
        data_sub = self.ref_data[condition]
        if len(data_sub)!=0:
            which_row = (data_sub['L']-current_L).abs().idxmin()
            f = lambda obj_, str_,row_:obj_[str_][row_]
            for each in ["H", "K", "roi_x", "roi_y", "roi_w", "roi_h", "ss_factor", "peak_width","peak_shift","poly_func", "poly_order", "poly_type"]:
                self.ref_fit_pars_current_point[each] = f(data_sub, each, which_row) 
        else:
            self.ref_fit_pars_current_point = {}
        # print(self.ref_fit_pars_current_point)

    def get_fit_pars_from_frame_index(self,frame_index):
        which_row = frame_index
        f = lambda obj_, str_,row_:obj_[str_][row_]
        for each in ["H", "K", "roi_x", "roi_y", "roi_w", "roi_h", "ss_factor", "peak_width","poly_func", "poly_order", "poly_type"]:
            if each == "peak_width":
                self.ref_fit_pars_current_point[each] = f(self.app_ctr.data, each, which_row)/2 
            else:
                self.ref_fit_pars_current_point[each] = f(self.app_ctr.data, each, which_row) 
        # print(self.ref_fit_pars_current_point)

    def set_fit_pars_from_reference(self):
        if len(self.ref_fit_pars_current_point)==0:
            return
        # print(self.app_ctr.bkg_sub.peak_width)
        # print(self.ref_fit_pars_current_point['peak_width'])
        self.roi.setPos(pos = [self.ref_fit_pars_current_point['roi_x'],self.ref_fit_pars_current_point['roi_y']])
        self.roi.setSize(size = [self.ref_fit_pars_current_point['roi_w'],self.ref_fit_pars_current_point['roi_h']])
        self.doubleSpinBox_ss_factor.setValue(self.ref_fit_pars_current_point['ss_factor'])
        self.spinBox_peak_width.setValue(self.ref_fit_pars_current_point['peak_width']/2)
        self.app_ctr.bkg_sub.peak_shift = self.ref_fit_pars_current_point['peak_shift']
        def _split_poly_order(order):
            if order in [1,2,3,4]:
                return [order]
            else:
                if order == 5:
                    return [1,4]
                elif order == 6:
                    return [2,4]
                elif order == 7:
                    return [2,5]
                elif order == 8:
                    return [1,3,4]
                elif order == 9:
                    return [2,3,4]
                elif order ==10:
                    return [1,2,3,4]
        try:
            eval("self.radioButton_{}.setChecked(True)".format(self.ref_fit_pars_current_point['poly_func']))
        except:
            print("No radioButton named radioButton_{}".format(self.ref_fit_pars_current_point['poly_func']))

        try:
            eval("self.radioButton_{}.setChecked(True)".format(self.ref_fit_pars_current_point['poly_type']))
        except:
            print("No radioButton named radioButton_{}".format(self.ref_fit_pars_current_point['poly_type']))
        
        poly_order_list = _split_poly_order(self.ref_fit_pars_current_point['poly_order'])
        for each in poly_order_list:
            try:
                eval("self.checkBox_order{}.setChecked(True)".format(each))
            except:
                print("No checkBox named checkBox_order{}".format(each))


    def move_roi_left(self):
        if not self.checkBox_big_roi.isChecked():
            self.app_ctr.bkg_sub.peak_shift = self.app_ctr.bkg_sub.peak_shift-int(self.lineEdit_roi_offset.text())
            # self.app_ctr.bkg_sub.peak_shift = -int(self.lineEdit_roi_offset.text())
            self.updatePlot()
        else:
            pos = [int(each) for each in self.roi.pos()] 
            size=[int(each) for each in self.roi.size()]
            if self.radioButton_roi_position.isChecked():
                pos_return,size_return = self._check_roi_boundary([pos[0]-int(self.lineEdit_roi_offset.text()),pos[1]],size)
                self.roi.setPos(pos = pos_return)
                self.roi.setSize(size = size_return)
            else:
                pos_return,size_return = self._check_roi_boundary(pos=[pos[0]-int(self.lineEdit_roi_offset.text()), pos[1]],size=[size[0]+int(self.lineEdit_roi_offset.text())*2, size[1]])
                self.roi.setSize(size=size_return)
                self.roi.setPos(pos = pos_return)

    def move_roi_right(self):
        if not self.checkBox_big_roi.isChecked():
            self.app_ctr.bkg_sub.peak_shift = self.app_ctr.bkg_sub.peak_shift + int(self.lineEdit_roi_offset.text())
            self.updatePlot()
        else:
            pos = [int(each) for each in self.roi.pos()] 
            size=[int(each) for each in self.roi.size()]
            if self.radioButton_roi_position.isChecked():
                pos_return,size_return = self._check_roi_boundary([pos[0]+int(self.lineEdit_roi_offset.text()),pos[1]],size)
                self.roi.setPos(pos = pos_return)
                self.roi.setSize(size = size_return)
            else:
                pos_return,size_return = self._check_roi_boundary(pos=[pos[0]+int(self.lineEdit_roi_offset.text()), pos[1]],size=[size[0]-int(self.lineEdit_roi_offset.text())*2, size[1]])
                self.roi.setSize(size=size_return)
                self.roi.setPos(pos = pos_return)

    def move_roi_down(self):
        pos = [int(each) for each in self.roi.pos()] 
        size=[int(each) for each in self.roi.size()]
        if self.radioButton_roi_position.isChecked():
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]-int(self.lineEdit_roi_offset.text())],size)
            self.roi.setPos(pos_return)
            self.roi.setSize(size_return)
        else:
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]+int(self.lineEdit_roi_offset.text())],[size[0],size[1]-int(self.lineEdit_roi_offset.text())*2])
            self.roi.setPos(pos = pos_return)
            self.roi.setSize(size=size_return)

    def move_roi_up(self):
        pos = [int(each) for each in self.roi.pos()] 
        size=[int(each) for each in self.roi.size()]
        if self.radioButton_roi_position.isChecked():
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]+int(self.lineEdit_roi_offset.text())],size)
            self.roi.setPos(pos_return)
            self.roi.setSize(size_return)
        else:
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]-int(self.lineEdit_roi_offset.text())],[size[0],size[1]+int(self.lineEdit_roi_offset.text())*2])
            self.roi.setPos(pos = pos_return)
            self.roi.setSize(size=size_return)

    def display_current_roi_info(self):
        pos = [int(each) for each in self.roi.pos()] 
        size=[int(each) for each in self.roi.size()]
        pos_return,size_return = self._check_roi_boundary(pos,size)
        if not self.checkBox_lock.isChecked():
            self.lineEdit_roi_info.setText(str(pos_return + size_return))
        
    def set_roi(self):
        if self.lineEdit_roi_info.text()=='':
            return
        else:
            roi = eval(self.lineEdit_roi_info.text())
            self.roi.setPos(pos = roi[0:2])
            self.roi.setSize(size = roi[2:])
        

    def find_bounds_of_hist(self):
        bins = 200
        hist, bin_edges = np.histogram(self.app_ctr.bkg_sub.img, bins=bins)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        max_index = np.argmax(hist)
        mean_value = hist.mean()
        index_right = max_index
        for i in range(max_index,bins):
            if abs(hist[i]-mean_value)<mean_value*0.8:
                index_right = i
                break
        index_left = [max_index - (index_right - max_index),0][int((max_index - (index_right - max_index))<0)]
        index_right = min([index_right + int((index_right - index_left)*float(self.lineEdit_tailing_factor.text())),bins-1])
        return bin_centers[index_left], bin_centers[index_right]

    def maximize_roi(self):
        self.roi_pos = self.roi.pos()#save the roi pos first before maximize it
        self.roi_size = self.roi.size()#save the roi pos first before maximize it
        try:
            self.roi.setPos(pos = [int(self.roi.pos()[0]),0])
            self.roi.setSize(size = [int(self.roi.size()[0]),self.app_ctr.bkg_sub.img.shape[0]])
        except:
            logging.getLogger().exception('Error during setting roi to maximum, check the dimension!')
            self.tabWidget.setCurrentIndex(2)

    def track_peak(self):
        self.maximize_roi()
        if self.radioButton_automatic_set_hist.isChecked():
            self.hist.setLevels(*self.find_bounds_of_hist())
        loop_steps = int(self.lineEdit_track_steps.text())
        hist_range = self.hist.region.getRegion()
        left, right = hist_range
        for i in range(loop_steps):
            iso_value_temp = ((right - left)/loop_steps)*i + left + (right - left)*0.3
            self.isoLine.setValue(iso_value_temp)
            self.iso.setLevel(iso_value_temp)
            isocurve_center_x, iso_curve_center_y = self.iso.boundingRect().center().x(), self.iso.boundingRect().center().y()
            isocurve_height, isocurve_width = self.iso.boundingRect().height(),self.iso.boundingRect().width()
            if isocurve_height == 0 or isocurve_width==0:
                pass
            else:
                if (isocurve_height<int(self.lineEdit_track_size.text())) and (isocurve_width<int(self.lineEdit_track_size.text())):
                    break
                else:
                    pass

    def set_peak(self):
        arbitrary_size_offset = 10
        arbitrary_recenter_cutoff = 50
        isocurve_center_x, iso_curve_center_y = self.iso.boundingRect().center().x(), self.iso.boundingRect().center().y()
        isocurve_height, isocurve_width = self.iso.boundingRect().height()+arbitrary_size_offset,self.iso.boundingRect().width()+arbitrary_size_offset
        roi_new = [self.roi.pos()[0] + isocurve_center_x - self.roi.size()[0]/2,self.roi.pos()[1]+(iso_curve_center_y-self.roi.size()[1]/2)+self.roi.size()[1]/2-isocurve_height/2]
        if abs(sum(roi_new) - sum(self.roi_pos))<arbitrary_recenter_cutoff or (self.app_ctr.img_loader.frame_number == 0):
            self.roi.setPos(pos = roi_new)
            self.roi.setSize(size = [self.roi.size()[0],isocurve_height])
        else:#if too far away, probably the peak tracking failed to track the right peak. Then reset the roi to what it is before the track!
            self.roi.setPos(pos = self.roi_pos)
            self.roi.setSize(size = self.roi_size)

    def setup_image(self):
        # Interpret image data as row-major instead of col-major
        global img, roi, roi_bkg, data, p2, isoLine, iso
        win = self.widget_image
        # print(dir(win))
        win.setWindowTitle('pyqtgraph example: Image Analysis')

        # A plot area (ViewBox + axes) for displaying the image
        p1 = win.addPlot()

        # Item for displaying image data
        img = pg.ImageItem()
        self.img_pyqtgraph = img
        p1.addItem(img)

        # Custom ROI for selecting an image region
        ver_width,hor_width = self.app_ctr.cen_clip

        roi = pg.ROI(pos = [hor_width*2*0.2, 0.], size = [hor_width*2*0.6, ver_width*2])
        #roi = pg.ROI([100, 100], [100, 100])
        self.roi = roi
        roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        roi.addRotateHandle([0., 0.], [0.5, 0.5])
        p1.addItem(roi)
        
        roi_peak = pg.ROI(pos = [hor_width-self.app_ctr.bkg_sub.peak_width, 0.], size = [self.app_ctr.bkg_sub.peak_width*2, ver_width*2], pen = 'g')
        #roi = pg.ROI([100, 100], [100, 100])
        self.roi_peak = roi_peak
        p1.addItem(roi_peak)

        # Custom ROI for monitoring bkg
        roi_bkg = pg.ROI(pos = [hor_width*2*0.2, 0.], size = [hor_width*2*0.1, ver_width*2],pen = 'r')
        # roi_bkg = pg.ROI([0, 100], [100, 100],pen = 'r')
        self.roi_bkg = roi_bkg
        # roi_bkg.addScaleHandle([0.5, 1], [0.5, 0.5])
        # roi_bkg.addScaleHandle([0, 0.5], [0.5, 0.5])
        p1.addItem(roi_bkg)
        #roi.setZValue(10)  # make sure ROI is drawn above image

        # Isocurve drawing
        iso = pg.IsocurveItem(level=0.8, pen='g')
        iso.setParentItem(img)
        self.iso = iso
        
        #iso.setZValue(5)

        # Contrast/color control
        hist = pg.HistogramLUTItem()
        self.hist = hist
        hist.setImageItem(img)
        win.addItem(hist)

        # Draggable line for setting isocurve level
        isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self.isoLine = isoLine
        hist.vb.addItem(isoLine)
        hist.vb.setMouseEnabled(y=True) # makes user interaction a little easier
        isoLine.setValue(0.8)
        isoLine.setZValue(100000) # bring iso line above contrast controls

        # Another plot area for displaying ROI data
        win.nextRow()
        p2 = win.addPlot(colspan=2, title = 'ROI image profile')
        p2.setMaximumHeight(200)
        p2.setLabel('left','Intensity', units='c/s')
        p2.setLabel('bottom','Pixel number')

        #p2.setLogMode(y = True)


        # plot to show intensity over time
        win.nextRow()
        p3 = win.addPlot(colspan=2)
        p3.setMaximumHeight(200)
        p3.setLabel('left','Integrated Intensity', units='c/s')

        # plot to show intensity over time
        win.nextRow()
        p4 = win.addPlot(colspan=2)
        p4.setMaximumHeight(200)
        p4.setLabel('bottom','frame number')

        region_roi = pg.LinearRegionItem()
        region_roi.setZValue(10)
        region_roi.setRegion([10, 15])

        # Generate image data
        #data = np.random.normal(size=(500, 600))
        #data[20:80, 20:80] += 2.
        #data = pg.gaussianFilter(data, (3, 3))
        #data += np.random.normal(size=(500, 600)) * 0.1
        #img.setImage(data)
        ##hist.setLevels(data.min(), data.max())

        # build isocurves from smoothed data
        ##iso.setData(pg.gaussianFilter(data, (2, 2)))

        # set position and scale of image
        #img.scale(0.2, 0.2)
        #img.translate(-50, 0)

        # zoom to fit imageo
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        p1.autoRange()  

        def update_bkg_signal():
            selected = roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
            self.bkg_intensity = selected.mean()
            #self.bkg_clip_image = selected
            #self.app_ctr.bkg_clip_image = selected

        def update_bkg_clip():
            selected = roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
            #self.bkg_intensity = selected.sum()
            #self.bkg_clip_image = selected
            self.app_ctr.bkg_clip_image = selected

        # Callbacks for handling user interaction
        def updatePlot(begin = False):
            # t0 = time.time()
            update_bkg_signal()
            #global data
            try:
                selected = roi.getArrayRegion(self.app_ctr.bkg_sub.img, self.img_pyqtgraph)
            except:
                #selected = roi.getArrayRegion(data, self.img_pyqtgraph)
                pass

            self.p3.setLabel('left',self.comboBox_p3.currentText())
            self.p4.setLabel('left',self.comboBox_p4.currentText())
            
            # p2.plot(selected.sum(axis=int(self.app_ctr.bkg_sub.int_direct=='y')), clear=True)
            self.reset_peak_center_and_width()
            if self.tag_reprocess:
                self.app_ctr.run_update_one_specific_frame(self.app_ctr.bkg_sub.img, self.bkg_intensity, poly_func = ['Vincent','traditional'][int(self.radioButton_traditional.isChecked())], frame_offset = int(self.lineEdit_frame_index_offset.text()))
            else:
                self.app_ctr.run_update(bkg_intensity=self.bkg_intensity,begin = begin,poly_func=['Vincent','traditional'][int(self.radioButton_traditional.isChecked())])
            # t1 = time.time()
            ##update iso curves
            x, y = [int(each) for each in self.roi.pos()]
            w, h = [int(each) for each in self.roi.size()]
            self.iso.setData(pg.gaussianFilter(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)], (2, 2)))
            self.iso.setPos(x,y)
            #update peak roi
            self.roi_peak.setSize([self.app_ctr.bkg_sub.peak_width*2,h])
            self.roi_peak.setPos([x+w/2.-self.app_ctr.bkg_sub.peak_width+self.app_ctr.bkg_sub.peak_shift,y])
            #update bkg roi
            self.roi_bkg.setSize([w/2-self.app_ctr.bkg_sub.peak_width+self.app_ctr.bkg_sub.peak_shift,h])
            self.roi_bkg.setPos([x,y])
            self.display_current_roi_info()
            # t2 = time.time()
            # print(t1-t0,t2-t1)
            if self.app_ctr.img_loader.frame_number ==0:
                isoLine.setValue(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)].mean())
            else:
                pass
            #print(isoLine.value(),self.current_image_no)
            #plot others
            #plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
            if self.tag_reprocess:
                index_frame = int(self.lineEdit_frame_index_offset.text())
                plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr,index_frame)
                self.lcdNumber_frame_number.display(self.app_ctr.img_loader.frame_number+1+index_frame+1)
                try:
                    self.lcdNumber_potential.display(self.app_ctr.data['potential'][index_frame])
                    self.lcdNumber_current.display(self.app_ctr.data['current'][index_frame])
                except:
                    pass
                self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][index_frame])
                self.lcdNumber_signal_noise_ratio.display(self.app_ctr.data['peak_intensity'][index_frame]/self.app_ctr.data['noise'][index_frame])
            else:
                plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
                try:
                    self.lcdNumber_potential.display(self.app_ctr.data['potential'][-1])
                    self.lcdNumber_current.display(self.app_ctr.data['current'][-1])
                except:
                    pass
                self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][-1])
                self.lcdNumber_signal_noise_ratio.display(self.app_ctr.data['peak_intensity'][-1]/self.app_ctr.data['noise'][-1])
            self.lcdNumber_iso.display(isoLine.value())
            # if self.run_mode and ((self.app_ctr.data['peak_intensity'][-1]/self.app_ctr.data['peak_intensity_error'][-1])<1.5):
            if self.run_mode and ((self.app_ctr.data['peak_intensity'][-1]/self.app_ctr.data['noise'][-1])<self.doubleSpinBox_SN_cutoff.value()):
                self.pushButton_remove_current_point.click()

        def updatePlot_after_remove_point():
            #global data
            try:
                selected = roi.getArrayRegion(self.app_ctr.bkg_sub.img, self.img_pyqtgraph)
            except:
                #selected = roi.getArrayRegion(data, self.img_pyqtgraph)
                pass
            p2.plot(selected.sum(axis=0), clear=True)
            self.reset_peak_center_and_width()
            #self.app_ctr.run_update()
            ##update iso curves
            x, y = [int(each) for each in self.roi.pos()]
            w, h = [int(each) for each in self.roi.size()]
            self.iso.setData(pg.gaussianFilter(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)], (2, 2)))
            self.iso.setPos(x,y)
            if self.app_ctr.img_loader.frame_number ==0:
                isoLine.setValue(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)].mean())
            else:
                pass
            #print(isoLine.value(),self.current_image_no)
            #plot others
            plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
            try:
                self.lcdNumber_potential.display(self.app_ctr.data['potential'][-2])
                self.lcdNumber_current.display(self.app_ctr.data['current'][-2])
            except:
                pass
            self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][-2])
            self.lcdNumber_signal_noise_ratio.display(self.app_ctr.data['peak_intensity'][-2]/self.app_ctr.data['noise'][-2])
            self.lcdNumber_iso.display(isoLine.value())

        self.updatePlot = updatePlot
        self.updatePlot2 = updatePlot_after_remove_point
        self.update_bkg_clip = update_bkg_clip
        #roi.sigRegionChanged.connect(updatePlot)
        roi.sigRegionChanged.connect(self.update_ss_factor)

        def updateIsocurve():
            global isoLine, iso
            iso.setLevel(isoLine.value())
            self.lcdNumber_iso.display(isoLine.value())

        self.updateIsocurve = updateIsocurve

        isoLine.sigDragged.connect(updateIsocurve)

    def stop_func(self):
        if not self.stop:
            self.stop = True
            self.stopBtn.setText('Resume')
        else:
            self.stop = False
            self.stopBtn.setText('Stop')
        
    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Conf Files (*.ini);;text Files (*.txt)", options=options)
        if fileName:
            self.lineEdit.setText(fileName)
            error_msg = self.widget_config.update_parameter(fileName)
            if error_msg!=None:
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Error to load config file!')
                logging.getLogger().exception(error_msg)
                self.tabWidget.setCurrentIndex(2)

    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName))

    def rload_file(self):#to be deleted
        self.save_file()
        #self.region_bounds = [0,1]
        try:
            self.app_ctr.run(self.lineEdit.text())
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000)
            self.plot_()
            self.statusbar.showMessage('Initialization succeed!')
        except:
            self.statusbar.showMessage('Initialization failed!')

    def launch_file(self):
        if not self.lineEdit.text().endswith('_temp.ini'):
            self.lineEdit.setText(self.lineEdit.text().replace('.ini','_temp.ini'))
        self.save_file()
        self.timer_save_data.timeout.connect(self.save_data)
        self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        #update the path to save data
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        self.app_ctr.data_path = data_file

        try:
            self.app_ctr.run(self.lineEdit.text())
            self.update_poly_order(init_step=True)
            self.update_cost_func(init_step=True)
            if self.launch.text()=='Launch':
                self.setup_image()
            else:
                pass
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
            self.plot_()
            self.update_ss_factor()
            self.image_set_up = False
            self.launch.setText("Relaunch")
            self.statusbar.showMessage('Initialization succeed!')
            self.image_set_up = True

            self.widget_terminal.update_name_space('data',self.app_ctr.data)
            self.widget_terminal.update_name_space('bkg_sub',self.app_ctr.bkg_sub)
            self.widget_terminal.update_name_space('img_loader',self.app_ctr.img_loader)
            self.widget_terminal.update_name_space('main_win',self)
            self.widget_terminal.update_name_space('overplot_ctr',overplot_ctr_temp)
            self.hist.sigLevelsChanged.connect(self.update_hist_levels)

        except Exception:
            self.image_set_up = False
            try:
                self.timer_save_data.stop()
            except:
                pass
            self.statusbar.showMessage('Initialization failed!')
            logging.getLogger().exception('Fatal error encounter during lauching config file! Check the config file for possible errors.')
            self.tabWidget.setCurrentIndex(2)

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Text documents (*.txt);All files (*.*)")
        #text = self.textEdit.toPlainText()
        #with open(path, 'w') as f:
        #    f.write(text)
        #self.statusbar.showMessage('Config file is saved as {}!'.format(path))
        if path != '':
            self.widget_config.save_parameter(path)
        else:
            self.statusbar.showMessage('Failure to save Config file with the file name of {}!'.format(path))

    def save_file(self):
        #text = self.textEdit.toPlainText()
        #if text=='':
        #    self.statusbar.showMessage('Text editor is empty. Config file is not saved!')
        #else:
        if self.lineEdit.text()!='':
            self.widget_config.save_parameter(self.lineEdit.text())
            self.statusbar.showMessage('Config file is saved with the same file name!')
        else:
            self.statusbar.showMessage('Failure to save Config file with the file name of {}!'.format(self.lineEdit.text()))

    def plot_figure(self):
        self.tag_reprocess = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.plot_)
        self.run_mode = True
        self.timer.start(100)

    def update_image(self):
        int_max,int_min = np.max(self.app_ctr.bkg_sub.img),np.min(self.app_ctr.bkg_sub.img)
        if self.image_log_scale:
            self.img_pyqtgraph.setImage(np.log10(self.app_ctr.bkg_sub.img))
            int_max,int_min = np.log10(int_max),np.log10(int_min)
        else:
            self.img_pyqtgraph.setImage(self.app_ctr.bkg_sub.img)
        self.p1.autoRange() 
        # self.hist.setImageItem(self.img_pyqtgraph)
        # self.hist.setLevels(self.app_ctr.bkg_sub.img.min(), self.app_ctr.bkg_sub.img.mean()*10)
        if self.radioButton_fixed_percent.isChecked():
            offset_ = float(self.lineEdit_scale_factor.text())/100*(int_max-int_min)
            # print(int_min,int_max,offset_)
            self.hist.setLevels(int_min, int_min+offset_)
        elif self.radioButton_fixed_between.isChecked():
            self.hist.setLevels(max([int_min,float(self.lineEdit_left.text())]), float(self.lineEdit_right.text()))
        elif self.radioButton_automatic_set_hist.isChecked(): 
            self.hist.setLevels(*self.find_bounds_of_hist())

    def update_hist_levels(self):
        left,right = self.hist.getLevels()
        self.lineEdit_left.setText(str(round(left,6)))
        self.lineEdit_right.setText(str(round(right,6)))

    def plot_(self):
        #self.app_ctr.set_fig(self.MplWidget.canvas.figure)
        t0 = time.time()
        if self.stop:
            self.timer.stop()
            self.run_mode = False
        else:
            try:
                return_value = self.app_ctr.run_script(poly_func=['Vincent','traditional'][int(self.radioButton_traditional.isChecked())])
                self.get_fit_pars_from_reference()
                self.set_fit_pars_from_reference()
                self.update_plot()
                if self.app_ctr.bkg_sub.img is not None:
                    self.lcdNumber_scan_number.display(self.app_ctr.img_loader.scan_number)
                    self.update_image()
                    if self.image_set_up:
                        self.updatePlot(begin = False)
                    else:
                        self.updatePlot(begin = True)
                if self.checkBox_auto_track.isChecked():
                    self.track_peak()
                    self.set_peak()
                if return_value:
                    self.statusbar.clearMessage()
                    self.statusbar.showMessage('Working on scan{}: we are now at frame{} of {} frames in total!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.frame_number+1,self.app_ctr.img_loader.total_frame_number))
                    self.progressBar.setValue(int((self.app_ctr.img_loader.frame_number+1)/self.app_ctr.img_loader.total_frame_number*100))
                    self.lcdNumber_frame_number.display(self.app_ctr.img_loader.frame_number+1)
                else:
                    self.timer.stop()
                    self.save_data()
                    self.stop = False
                    self.stopBtn.setText('Stop')
                    self.statusbar.clearMessage()
                    self.statusbar.showMessage('Run for scan{} is finished, {} frames in total have been processed!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.total_frame_number))
                # """
                    #if you want to save the images, then uncomment the following three lines
                    #QtGui.QApplication.processEvents()
                    #exporter = pg.exporters.ImageExporter(self.widget_image.scene())
                    #exporter.export(os.path.join(DaFy_path,'temp','temp_frames','scan{}_frame{}.png'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.frame_number+1)))
            except:
                logging.getLogger().exception('Fatal error encounter during data analysis.')
                self.tabWidget.setCurrentIndex(2)

            # """
        try:
            self.lcdNumber_speed.display(int(1./(time.time()-t0)))
        except:
            pass

    def update_plot(self):
        try:
            img = self.app_ctr.run_update(poly_func=['Vincent','traditional'][int(self.radioButton_traditional.isChecked())])
            if self.tag_reprocess:
                plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr, int(self.lineEdit_frame_index_offset.text()))
            else:
                plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
            # self.MplWidget.canvas.figure.tight_layout()
            # self.MplWidget.canvas.draw()
        except:
            logging.getLogger().exception('Fatal error encounter during data analysis.')
            self.tabWidget.setCurrentIndex(2)

    def reset_peak_center_and_width(self):
        roi_size = [int(each/2) for each in self.roi.size()][::-1]
        roi_pos = [int(each) for each in self.roi.pos()][::-1]
        #roi_pos[0] = self.app_ctr.cen_clip[0]*2-roi_pos[0]
        #new_center = [roi_pos[0]-roi_size[0],roi_pos[1]+roi_size[1]]
        #roi_pos[0] = self.app_ctr.cen_clip[0]*2-roi_pos[0]
        new_center = [roi_pos[0]+roi_size[0],roi_pos[1]+roi_size[1]]
        self.app_ctr.bkg_sub.center_pix = new_center
        self.app_ctr.bkg_sub.row_width = roi_size[1]
        self.app_ctr.bkg_sub.col_width = roi_size[0]

    def peak_cen_shift_hor(self):
        offset = int(self.spinBox_hor.value())
        self.app_ctr.bkg_sub.update_center_pix_left_and_right(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

    def peak_cen_shift_ver(self):
        offset = int(self.spinBox_ver.value())
        self.app_ctr.bkg_sub.update_center_pix_up_and_down(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

    def row_width_shift(self):
        offset = int(self.horizontalSlider.value())
        self.app_ctr.bkg_sub.update_integration_window_row_width(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

    def col_width_shift(self):
        offset = int(self.verticalSlider.value())
        self.app_ctr.bkg_sub.update_integration_window_column_width(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, tableviewer, main_gui, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self.tableviewer = tableviewer
        self.main_gui = main_gui

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role):
        if index.isValid():
            if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
                return str(self._data.iloc[index.row(), index.column()])
            if role == QtCore.Qt.BackgroundRole and index.row()%2 == 0:
                # return QtGui.QColor('DeepSkyBlue')
                return QtGui.QColor('green')
            if role == QtCore.Qt.BackgroundRole and index.row()%2 == 1:
                return QtGui.QColor('dark')
                # return QtGui.QColor('lightGreen')
            if role == QtCore.Qt.ForegroundRole and index.row()%2 == 1:
                return QtGui.QColor('white')
            if role == QtCore.Qt.CheckStateRole and (index.column() in [0, 6]):
                if self._data.iloc[index.row(),index.column()]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role == QtCore.Qt.CheckStateRole and index.column() == 0:
            if value == QtCore.Qt.Checked:
                self._data.iloc[index.row(),index.column()] = True
            else:
                self._data.iloc[index.row(),index.column()] = False
        elif role == QtCore.Qt.CheckStateRole and index.column() == 6:
            if value == QtCore.Qt.Checked:
                self._data.iloc[index.row(),index.column()] = True
            else:
                self._data.iloc[index.row(),index.column()] = False
        else:
            if str(value)!='':
                self._data.iloc[index.row(),index.column()] = str(value)
        #if self._data.columns.tolist()[index.column()] in ['select','archive_data','user_label','read_level']:
        #    self.main_gui.update_meta_info_paper(paper_id = self._data['paper_id'][index.row()])
        self.dataChanged.emit(index, index)
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()
        self.tableviewer.resizeColumnsToContents() 
        return True

    def update_view(self):
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[rowcol]         
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[rowcol]         
        return None

    def flags(self, index):
        if not index.isValid():
           return QtCore.Qt.NoItemFlags
        else:
            if index.column() == 0:
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
            elif index.column() == 6:
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
            else:
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        self.layoutAboutToBeChanged.emit()
        self._data = self._data.sort_values(self._data.columns.tolist()[Ncol],
                                        ascending=order == QtCore.Qt.AscendingOrder, ignore_index = True)
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

class start_editor_dialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.parent = parent
        uic.loadUi(os.path.join(script_path,"save_ctr_data_dialog.ui"), self)
        self.lineEdit_file_path.setText(os.path.join(self.parent.lineEdit_data_file_path.text(),'ctr_data_temp.csv'))
        self.pushButton_close.clicked.connect(lambda:self.close())
        self.pushButton_save.clicked.connect(self.save_data)
        self.display_table()

    def _generate_scan_info(self):
        info_list = list(set(zip(self.parent.app_ctr.data['scan_no'],self.parent.app_ctr.data['H'], self.parent.app_ctr.data['K'])))
        scan_no, H, K, dL, BL, save, escan = [],[],[],[],[],[], []
        for each in info_list:
            scan_no.append(each[0])
            H.append(each[1])
            K.append(each[2])
            dL.append(0)
            BL.append(0)
            save.append(True)
            escan.append(False)
        result = pd.DataFrame({'save': save,'scan_no':scan_no, 'H': H, 'K': K, 'dL': dL, 'BL': BL, 'escan': escan})
        return result

    def save_data(self):
        self.parent.app_ctr.save_rod_data(self.lineEdit_file_path.text(), self.pandas_model._data)
        error_pop_up('Successful to save rod data!','Information')

    def display_table(self):
        self.pandas_model = PandasModel(data = pd.DataFrame(self._generate_scan_info()), tableviewer = self.tableView_editor, main_gui = self)
        self.tableView_editor.setModel(self.pandas_model)
        self.tableView_editor.resizeColumnsToContents()
        self.tableView_editor.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
        self.parent.pandas_model_ = self.pandas_model

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    # QApplication.setGraphicsSystem("raster")
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())