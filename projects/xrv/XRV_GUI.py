import sys,os,qdarkstyle
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QShortcut
from PyQt5 import uic
import random
import numpy as np
import matplotlib.pyplot as plt
from CV_XRD_DaFy_class import run_app
try:
    from . import locate_path_xrv
except:
    import locate_path_xrv
script_path = locate_path_xrv.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
from VisualizationEnginePool import plot_xrv_gui_pyqtgraph,replot_bkg_profile
import time
import matplotlib
matplotlib.use("TkAgg")
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton
from pyqtgraph.Qt import QtGui

#customize the label of image axis (q basis instead of index basis)
class pixel_to_q(pg.AxisItem):
    def __init__(self, scale, shift, *args, **kwargs):
        super(pixel_to_q, self).__init__(*args, **kwargs)
        self.scale = scale
        self.shift = shift

    def tickStrings(self, values, scale, spacing):
        return [round(value*self.scale+self.shift,2) for value in values]

    def attachToPlotItem(self, plotItem):
        """Add this axis to the given PlotItem
        :param plotItem: (PlotItem)
        """
        self.setParentItem(plotItem)
        viewBox = plotItem.getViewBox()
        self.linkToView(viewBox)
        self._oldAxis = plotItem.axes[self.orientation]['item']
        self._oldAxis.hide()
        plotItem.axes[self.orientation]['item'] = self
        pos = plotItem.axes[self.orientation]['pos']
        plotItem.layout.addItem(self, *pos)
        self.setZValue(-1000)

class MyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        #load ui script
        uic.loadUi(os.path.join(DaFy_path,'projects','xrv','XRV_GUI.ui'),self)
        self.setWindowTitle('Data analysis factory: XRV data analasis')
        self.app_ctr=run_app()
        self.current_image_no = 0
        self.current_scan_number = None
        self.bkg_intensity = 0
        self.bkg_clip_image = None
        self.stop = False
        self.widget_terminal.update_name_space('xrv',self.app_ctr)
        self.widget_terminal.update_name_space('main_gui',self)

        #pre-set the data path and data name
        self.lineEdit_data_file_name.setText('temp_data_xrv.xlsx')
        self.lineEdit_data_file_path.setText(self.app_ctr.data_path)
        setattr(self.app_ctr,'data_path',os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text()))

        #signal-slot setting
        self.open.clicked.connect(self.load_file)
        self.launch.clicked.connect(self.launch_file)
        self.stopBtn.clicked.connect(self.stop_func)
        self.saveas.clicked.connect(self.save_file_as)
        self.save.clicked.connect(self.save_file)
        self.plot.clicked.connect(self.plot_figure)
        self.runstepwise.clicked.connect(self.plot_)
        self.nextShort = QShortcut(QtGui.QKeySequence("Right"), self)
        self.nextShort.activated.connect(self.plot_)
        self.pushButton_filePath.clicked.connect(self.locate_data_folder)
        # self.pushButton_fold_or_unfold.clicked.connect(self.fold_or_unfold)
        for each in self.groupBox_2.findChildren(QCheckBox):
            each.released.connect(self.update_poly_order)
        '''
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            each.toggled.connect(self.update_cost_func)
        '''
        self.pushButton_remove_current_point.clicked.connect(self.remove_data_point)
        self.pushButton_recenter.clicked.connect(self.recenter)
        #self.doubleSpinBox_ss_factor.valueChanged.connect(self.update_ss_factor)
        self.comboBox_p2.activated.connect(self.select_source_for_plot_p2)
        self.actionOpenConfig.triggered.connect(self.load_file)
        self.actionSaveConfig.triggered.connect(self.save_file)
        self.actionRun.triggered.connect(self.plot_)
        self.actionStop.triggered.connect(self.stop_func)
        self.actionSaveData.triggered.connect(self.save_data)
        self.p2_data_source = self.comboBox_p2.currentText()
        setattr(self.app_ctr,'p2_data_source',self.comboBox_p2.currentText())
        self.timer_save_data = QtCore.QTimer(self)
        self.timer_save_data.timeout.connect(self.save_data)
        
    #to fold or unfold the config file editor
    def fold_or_unfold(self):
        text = self.pushButton_fold_or_unfold.text()
        if text == "<":
            self.frame.setVisible(False)
            self.pushButton_fold_or_unfold.setText(">")
        elif text == ">":
            self.frame.setVisible(True)
            self.pushButton_fold_or_unfold.setText("<")

    #fun to save data
    def save_data(self):
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        self.app_ctr.save_data_file(data_file)
        try:
            self.app_ctr.save_data_file(data_file)
            self.statusbar.showMessage('Data file is saved as {}!'.format(data_file))
        except:
            self.statusbar.showMessage('Failure to save data file!')

    #fun to remove abnormal points (set marsk to false, data is saved acturally!)
    def remove_data_point(self):
        left,right = [int(each) for each in self.region_abnormal.getRegion()]
        self.lineEdit_abnormal_points.setText('Frame {} to Frame {}'.format(left,right))
        first_index_for_current_scan = np.where(np.array(self.app_ctr.data['scan_no'])==self.app_ctr.img_loader.scan_number)[0][0]
        for each_index in range(first_index_for_current_scan+left,first_index_for_current_scan+right+1):
            self.app_ctr.data['mask_cv_xrd'][each_index] = False
            self.app_ctr.data['mask_ctr'][each_index] = False
        self.updatePlot()

    def select_source_for_plot_p2(self):
        self.app_ctr.p2_data_source = self.comboBox_p2.currentText()
        self.updatePlot()

    def update_poly_order(self, init_step = False):
        ord_total = 0
        for each in self.groupBox_2.findChildren(QCheckBox):
            ord_total += int(bool(each.checkState()))*int(each.text())
        self.app_ctr.bkg_sub.update_integration_order(ord_total)
        print('ord_total:',ord_total)
        if not init_step:
            self.updatePlot()

    def update_cost_func(self, init_step = False):
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            if each.isChecked():
                self.app_ctr.bkg_sub.update_integration_function(each.text())
                if not init_step:
                    self.updatePlot()
                break
        '''
        try:
        except:
            pass
        '''

    def update_ss_factor(self, init_step = False):
        self.app_ctr.bkg_sub.update_ss_factor(self.doubleSpinBox_ss_factor.value())
        self.updatePlot()
        '''
        try:
            self.updatePlot()
        except:
            pass
        '''

    #set up the image zone (base on pyqtgraph engine)
    def setup_image(self):
        # global img, roi, roi_bkg, p2, p2_r, p3_r, p3, p4, p4_r, isoLine, iso

        win = self.widget_image

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        win.addItem(self.hist,row=0,col=0,rowspan=1,colspan=2)

        # A plot area (ViewBox + axes) for displaying the image
        p1 = win.addPlot(row=0,col=2,rowspan=1,colspan=2)
        # Item for displaying image data
        img = pg.ImageItem()
        p1.getViewBox().invertY(False)
        self.img_pyqtgraph = img
        p1.addItem(img)
        self.hist.setImageItem(img)

        # Custom ROI for selecting an image region
        roi = pg.ROI([100, 100], [200, 200])
        self.roi = roi
        roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        p1.addItem(roi)

        # Custom ROI for monitoring bkg
        roi_bkg = pg.ROI([0, 100], [200, 200],pen = 'r')
        self.roi_bkg = roi_bkg
        roi_bkg.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi_bkg.addScaleHandle([0, 0.5], [0.5, 0.5])
        p1.addItem(roi_bkg)

        # Isocurve drawing
        iso = pg.IsocurveItem(level=0.8, pen='g')
        iso.setParentItem(img)
        self.iso = iso
        
        # Draggable line for setting isocurve level
        isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self.isoLine = isoLine
        self.hist.vb.addItem(isoLine)
        self.hist.vb.setMouseEnabled(y=True) # makes user interaction a little easier
        isoLine.setValue(0.8)
        isoLine.setZValue(100000) # bring iso line above contrast controls

        #set up the region selector for peak fitting
        self.region_cut_hor = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal)
        self.region_cut_ver = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical)
        self.region_cut_hor.setRegion([180,220])
        self.region_cut_ver.setRegion([180,220])
        p1.addItem(self.region_cut_hor, ignoreBounds = True)
        p1.addItem(self.region_cut_ver, ignoreBounds = True)

        # double-y axis plot for structure data (strain and size)
        p2 = win.addPlot(row=1,col=1,colspan=2,rowspan=1, title = 'Strain (left,white) and size (right,blue)')
        p2.setLabel('bottom','x channel')
        # p2.showAxis('right')
        # p2.setLabel('right','grain_size', pen = "b")
        #p2.setLogMode(y = True)
        p2_r = pg.ViewBox()
        p2.showAxis('right')
        p2.scene().addItem(p2_r)
        p2.getAxis('right').linkToView(p2_r)
        p2_r.setXLink(p2)
        # p2.getAxis('right').setLabel('grain_size', color='b')
        ## Handle view resizing
        def updateViews_p2():
            ## view has resized; update auxiliary views to match
            p2_r.setGeometry(p2.vb.sceneBoundingRect())
            ## need to re-update linked axes since this was called
            ## incorrectly while views had different shapes.
            ## (probably this should be handled in ViewBox.resizeEvent)
            p2_r.linkedViewChanged(p2.vb, p2_r.XAxis)
        updateViews_p2()
        p2.vb.sigResized.connect(updateViews_p2)

        # plot to show intensity(sig and bkg) over time
        p3 = win.addPlot(row=2,col=1,colspan=2,rowspan=1,title = 'Peak intensity(left,white) and bkg intensity (right,blue)')
        #p3.setLabel('left','Integrated Intensity', units='c/s')
        p3_r = pg.ViewBox()
        p3.showAxis('right')
        p3.scene().addItem(p3_r)
        p3.getAxis('right').linkToView(p3_r)
        p3_r.setXLink(p3)
        #p3.getAxis('right').setLabel('bkg', color='b')
        ## Handle view resizing
        def updateViews_p3():
            ## view has resized; update auxiliary views to match
            p3_r.setGeometry(p3.vb.sceneBoundingRect())
            ## need to re-update linked axes since this was called
            ## incorrectly while views had different shapes.
            ## (probably this should be handled in ViewBox.resizeEvent)
            p3_r.linkedViewChanged(p3.vb, p3_r.XAxis)
        updateViews_p3()
        p3.vb.sigResized.connect(updateViews_p3)

        # plot to show current/potential over time
        p4 = win.addPlot(row=3,col=1,colspan=2,rowspan=1, title = 'Potential (left,white) and current(right,blue)')
        #p4.setMaximumHeight(200)
        p4.setLabel('bottom','x channel')
        p4_r = pg.ViewBox()
        p4.showAxis('right')
        p4.scene().addItem(p4_r)
        p4.getAxis('right').linkToView(p4_r)
        p4_r.setXLink(p4)
        #p4.getAxis('right').setLabel('bkg', color='b')
        ## Handle view resizing
        def updateViews_p4():
            ## view has resized; update auxiliary views to match
            p4_r.setGeometry(p4.vb.sceneBoundingRect())
            ## need to re-update linked axes since this was called
            ## incorrectly while views had different shapes.
            ## (probably this should be handled in ViewBox.resizeEvent)
            p4_r.linkedViewChanged(p4.vb, p4_r.XAxis)
        updateViews_p4()
        p4.vb.sigResized.connect(updateViews_p4)

        #plot the peak fit results(horizontally)
        p5 = win.addPlot(row=1,col=0,colspan=1,rowspan=1,title = 'peak fit result_horz')
        p6 = win.addPlot(row=2,col=0,colspan=1,rowspan=1,title = 'peak fit result_vert')
        p6.setLabel('bottom','q')
        p7 = win.addPlot(row=3,col=0,colspan=1,rowspan=1,title = 'Peak intensity')
        p7.setLabel('bottom','pixel index')

        #add slider to p2 to exclude the abnormal points
        self.region_abnormal = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical)
        self.region_abnormal.setZValue(100)
        self.region_abnormal.setRegion([0, 5])
        p2.addItem(self.region_abnormal, ignoreBounds = True)

        self.p1 = p1
        self.p2 = p2
        self.p2_r = p2_r
        self.p3 = p3
        self.p3_r = p3_r
        self.p4 = p4
        self.p4_r = p4_r
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7

        # zoom to fit image
        p1.autoRange()  

        def update_bkg_signal():
            selected = self.roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
            self.bkg_intensity = selected.sum()
            #self.bkg_clip_image = selected
            #self.app_ctr.bkg_clip_image = selected

        self.update_bkg_signal = update_bkg_signal

        # Callbacks for handling user interaction
        def updatePlot():
            #global data
            try:
                selected = self.roi.getArrayRegion(self.app_ctr.bkg_sub.img, self.img_pyqtgraph)
            except:
                pass

            self.reset_peak_center_and_width()
            self.update_bkg_signal()
            self.app_ctr.run_update(bkg_intensity=self.bkg_intensity)
            '''
            if self.stop:
                self.update_bkg_signal()
                self.app_ctr.run_update(bkg_intensity=self.bkg_intensity)
            else:
                pass
            '''
            ##update iso curves
            x, y = [int(each) for each in self.roi.pos()]
            w, h = [int(each) for each in self.roi.size()]
            self.iso.setData(pg.gaussianFilter(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)], (2, 2)))
            self.iso.setPos(x,y)
            #update bkg roi
            self.roi_bkg.setSize([w,h])
            self.roi_bkg.setPos([x-w,y])

            if self.app_ctr.img_loader.frame_number ==0:
                isoLine.setValue(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)].mean())
            else:
                pass

            #plot others
            plot_xrv_gui_pyqtgraph(self.p1,[self.p2,self.p2_r], [self.p3,self.p3_r], [self.p4,self.p4_r],self.p5, self.p6, self.p7,self.app_ctr, self.checkBox_x_channel.isChecked())
            # add slider afterwards to have it show up on th plot
            self.p2.addItem(self.region_abnormal, ignoreBounds = True)

            #show values for current status
            self.lcdNumber_potential.display(self.app_ctr.data['potential'][-1])
            self.lcdNumber_current.display(self.app_ctr.data['current'][-1])
            self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][-1])
            self.lcdNumber_strain_par.display(self.app_ctr.data['strain_ip'][-1])
            self.lcdNumber_strain_ver.display(self.app_ctr.data['strain_oop'][-1])
            self.lcdNumber_size_par.display(self.app_ctr.data['grain_size_ip'][-1])
            self.lcdNumber_size_ver.display(self.app_ctr.data['grain_size_oop'][-1])
            self.lineEdit_peak_center.setText(str(self.app_ctr.peak_fitting_instance.peak_center))
            self.lineEdit_previous_center.setText(str(self.app_ctr.peak_fitting_instance.previous_peak_center))
            self.lcdNumber_iso.display(isoLine.value())

        roi.sigRegionChanged.connect(updatePlot)
        self.updatePlot = updatePlot

        def updateIsocurve():
            # global isoLine, iso
            self.iso.setLevel(isoLine.value())
            self.lcdNumber_iso.display(isoLine.value())
        self.updateIsocurve = updateIsocurve
        isoLine.sigDragged.connect(updateIsocurve)

    def recenter(self):
        #self.app_ctr.peak_fitting_instance.previous_peak_center = self.app_ctr.peak_fitting_instance.peak_center
        self.app_ctr.peak_fitting_instance.recenter = True
        self.updatePlot()

    def stop_func(self):
        if not self.stop:
            self.stop = True
            self.stopBtn.setText('Resume')
        else:
            self.stop = False
            self.stopBtn.setText('Stop')
    
    #load config file
    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Config Files (*.ini)", os.path.join(DaFy_path,'scripts','XRV'),options = options)
        if fileName:
            self.lineEdit.setText(fileName)
            #self.timer_save_data.start(self.spinBox_save_frequency.value()*1000)
            self.widget_config.update_parameter(fileName)
            #with open(fileName,'r') as f:
            #    self.textEdit.setText(f.read())

    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName))

    def launch_file(self):
        self.save_file() 
        #update the path to save data
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        self.app_ctr.data_path = data_file
        self.app_ctr.run(self.lineEdit.text())
        self.update_poly_order(init_step=True)
        # self.update_cost_func(init_step=True)
        if self.launch.text()=='Launch':
            self.setup_image()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        else:
            pass
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        #self.timer_save_data.stop()
        self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        self.plot_()
        self.launch.setText("Relaunch")
        self.statusbar.showMessage('Initialization succeed!') 
        try:
            self.app_ctr.run(self.lineEdit.text())
            self.update_poly_order(init_step=True)
            # self.update_cost_func(init_step=True)
            if self.launch.text()=='Launch':
                self.setup_image()
            else:
                pass
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
            self.plot_()
            self.launch.setText("Relaunch")
            self.statusbar.showMessage('Initialization succeed!')
        except:
            self.statusbar.showMessage('Initialization failed!')

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Text documents (*.txt);All files (*.*)")
        text = self.textEdit.toPlainText()
        with open(path, 'w') as f:
            f.write(text)
        self.statusbar.showMessage('Config file is saved as {}!'.format(path))

    def save_file(self):
        # text = self.textEdit.toPlainText()
        if self.lineEdit.text()=='':
            self.statusbar.showMessage('Text editor is empty. Config file is not saved!')
        else:
            #with open(self.lineEdit.text(), 'w') as f:
            #    f.write(text)
            self.widget_config.save_parameter(self.lineEdit.text())
            self.statusbar.showMessage('Config file is saved with the same file name!')

    def plot_figure(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.plot_)
        self.timer.start(50)

    def plot_(self):
        #self.app_ctr.set_fig(self.MplWidget.canvas.figure)
        t0 = time.time()
        if self.stop:
            self.timer.stop()
        else:
            return_value = self.app_ctr.run_script()
            self.update_bkg_signal()
            self.app_ctr.data['bkg'][-1] = self.bkg_intensity
            if self.app_ctr.bkg_sub.img is not None:
                #if self.current_scan_number == None:
                #    self.current_scan_number = self.app_ctr.img_loader.scan_number
                self.lcdNumber_scan_number.display(self.app_ctr.img_loader.scan_number)
                #set image and cut on the image (NOTE:row-major index)
                cut_values_hoz=[self.app_ctr.peak_fitting_instance.peak_center[0]-self.app_ctr.peak_fitting_instance.cut_offset['hor'][-1],self.app_ctr.peak_fitting_instance.peak_center[0]+self.app_ctr.peak_fitting_instance.cut_offset['hor'][-1]]
                cut_values_ver=[self.app_ctr.peak_fitting_instance.peak_center[1]-self.app_ctr.peak_fitting_instance.cut_offset['ver'][-1],self.app_ctr.peak_fitting_instance.peak_center[1]+self.app_ctr.peak_fitting_instance.cut_offset['ver'][-1]]
                self.img_pyqtgraph.setImage(self.app_ctr.bkg_sub.img)
                self.region_cut_hor.setRegion(cut_values_hoz)
                self.region_cut_ver.setRegion(cut_values_ver)

                #set roi
                size_of_roi = self.roi.size()
                self.roi.setPos([self.app_ctr.peak_fitting_instance.peak_center[0]-size_of_roi[0]/2.,self.app_ctr.peak_fitting_instance.peak_center[1]-size_of_roi[1]/2.])
                # self.roi.setPos([self.app_ctr.peak_fitting_instance.peak_center[1]-size_of_roi[1]/2.,self.app_ctr.peak_fitting_instance.peak_center[0]-size_of_roi[0]/2.])
                #self.p1.plot([0,400],[200,200])
                if self.app_ctr.img_loader.frame_number == 0:
                    self.p1.autoRange() 
                    #relabel the axis
                    q_par = self.app_ctr.rsp_instance.q['grid_q_par'][0]
                    q_ver = self.app_ctr.rsp_instance.q['grid_q_perp'][:,0]
                    scale_ver = (max(q_ver)-min(q_ver))/(len(q_ver)-1)
                    shift_ver = min(q_ver)
                    scale_hor = (max(q_par)-min(q_par))/(len(q_par)-1)
                    shift_hor = min(q_par)
                    ax_item_img_hor = pixel_to_q(scale = scale_hor, shift = shift_hor, orientation = 'bottom')
                    ax_item_img_ver = pixel_to_q(scale = scale_ver, shift = shift_ver, orientation = 'left')
                    ax_item_img_hor.attachToPlotItem(self.p1)
                    ax_item_img_ver.attachToPlotItem(self.p1)
                self.hist.setLevels(self.app_ctr.bkg_sub.img.min(), self.app_ctr.bkg_sub.img.mean()*10)
                self.updatePlot()

            if return_value:
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Working on scan{}: we are now at frame{} of {} frames in total!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.frame_number+1,self.app_ctr.img_loader.total_frame_number))
                self.progressBar.setValue((self.app_ctr.img_loader.frame_number+1)/float(self.app_ctr.img_loader.total_frame_number)*100)
                self.lcdNumber_frame_number.display(self.app_ctr.img_loader.frame_number+1)
                #self.app_ctr.img_loader.frame_number
                #self.current_image_no += 1
            else:
                self.timer.stop()
                self.save_data()
                self.stop = False
                self.stopBtn.setText('Stop')
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Run for scan{} is finished, {} frames in total have been processed!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.total_frame_number))
        try:
            self.lcdNumber_speed.display(int(1./(time.time()-t0)))
        except:
            pass

    def reset_peak_center_and_width(self):
        roi_size = [int(each/2) for each in self.roi.size()][::-1]
        roi_pos = [int(each) for each in self.roi.pos()][::-1]
        new_center = [roi_pos[0]+roi_size[0],roi_pos[1]+roi_size[1]]
        self.app_ctr.bkg_sub.center_pix = new_center
        self.app_ctr.bkg_sub.row_width = roi_size[1]
        self.app_ctr.bkg_sub.col_width = roi_size[0]

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())