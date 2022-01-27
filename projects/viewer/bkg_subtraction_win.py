import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import sys, os
try:
    from . import locate_path_viewer
except:
    import locate_path_viewer
script_path = locate_path_viewer.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
from FitEnginePool import backcor_confined
from sklearn import metrics

class bkg_win(pg.GraphicsLayoutWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.init_plot()

    def init_plot(self):
        #p1: subset of current data
        #p2: full scale of current data
        #p3: full scale of potential data
        self.p1 = self.addPlot(row = 1, col = 0)
        self.p2 = self.addPlot(row = 2, col = 0)
        self.p3 = self.addPlot(row = 3, col = 0)
        self.p1.setAutoVisible(y = True)

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.update)

        self.region_peak = pg.LinearRegionItem()
        self.region_peak.setZValue(10)

        self.p2.addItem(self.region, ignoreBounds = True)
        self.p1.addItem(self.region_peak, ignoreBounds = True)

        data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
        data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
        self.p1_handle = self.p1.plot(data1, pen="r")
        self.p1_bkg_handle = self.p1.plot(data1*0, pen="w")
        self.p2_handle = self.p2.plot(data1, pen="w")
        self.p3_handle = self.p3.plot(data2, pen="y")

    def update(self):
        self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        minX, maxX = int(minX), int(maxX)
        self.p1.setXRange(minX, maxX, padding=0)    
        self.region_peak.setRegion([minX, maxX])
        self.fit_data = np.array(self.p1_handle.getData()[1][minX:maxX])
        self.fit_data_x = np.array(list(range(minX, maxX)))

    def set_data(self, pot, current):
        self.p1_handle.setData(list(range(len(current))), current)
        self.p1_bkg_handle.setData(list(range(len(current))), np.array(current)*0)
        self.p2_handle.setData(list(range(len(current))), current)
        self.p3_handle.setData(list(range(len(pot))), pot)

    def perform_bkg_fitting(self, ord_cus = 3,s = 1,fct = 'atq', scan_rate = 0.005):
        n = self.fit_data_x - self.fit_data_x[0]
        y = self.fit_data
        peak_area_index_original = list(map(int,list(self.region_peak.getRegion())))
        peak_area_index = [each - self.fit_data_x[0] for each in peak_area_index_original]
        z,*_ = backcor_confined(n,y,ord_cus,s,fct, peak_area_index)
        #update the background curve
        x_old, y_old = self.p1_bkg_handle.getData()
        y_new = list(y_old[0:self.fit_data_x[0]]) + list(z) + list(y_old[(self.fit_data_x[-1]+1):])
        self.p1_bkg_handle.setData(x_old, y_new)
        self._cal_charge(peak_area_index_original[0], peak_area_index_original[1], scan_rate)
        return self.charge

    def _cal_charge(self, index_left, index_right, scan_rate):
        pot = self.p3_handle.getData()[1]
        current = self.p2_handle.getData()[1]
        current_bkg = self.p1_bkg_handle.getData()[1]
        charge_top = metrics.auc(pot[index_left:index_right],current[index_left:index_right])
        charge_bottom = metrics.auc(pot[index_left:index_right],current_bkg[index_left:index_right])
        v_to_t = 1/scan_rate
        self.charge = round((charge_top-charge_bottom)*v_to_t/2,4)