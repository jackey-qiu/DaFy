import sys,os, qdarkstyle
from io import StringIO
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5 import uic, QtWidgets
import PyQt5
import random
import math
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
import model
import solvergui
import time
import datetime
os.environ["QT_MAC_WANTS_LAYER"] = "1"
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton, QTableWidgetItem, QHeaderView, QAbstractItemView, QInputDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTransform, QFont, QBrush, QColor, QIcon
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import*
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator, FixedFormatter

# Tools
def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))

def create_time_serie(size, time):
    """Generate a time serie of length size and dynamic with respect to time."""
    # Generating time-series
    support = np.arange(0, size)
    serie = np.cos(support + float(time))
    return(support, serie)

def compute_GAF(serie):
    """Compute the Gramian Angular Field of an image"""
    # Min-Max scaling
    serie = np.array(serie)
    min_ = np.amin(serie)
    max_ = np.amax(serie)
    scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

    # Floating point inaccuracy!
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

    # Polar encoding
    phi = np.arccos(scaled_serie)
    # Note! The computation of r is not necessary
    r = np.linspace(0, 1, len(scaled_serie))
    # GAF Computation (every term of the matrix)
    gaf = tabulate(phi, phi, cos_sum)

    return(gaf, phi, r, scaled_serie)

class GAF_Widget(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent) 
        self.parent = None
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig) 
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.navi_toolbar)
        # self.canvas.ax_img = self.canvas.figure.add_subplot(121)
        # self.canvas.ax_profile = self.canvas.figure.add_subplot(322)
        # self.canvas.ax_ctr = self.canvas.figure.add_subplot(324)
        # self.canvas.ax_pot = self.canvas.figure.add_subplot(326)
        #self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    def update_canvas(self, fig_size):
        self.canvas = FigureCanvas(Figure(fig_size)) 
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.navi_toolbar)
        # self.canvas.ax_profile = self.canvas.figure.add_subplot(322)
        # self.canvas.ax_ctr = self.canvas.figure.add_subplot(324)
        # self.canvas.ax_pot = self.canvas.figure.add_subplot(326)
        #self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    def reset(self):
        self.canvas.figure.clear()
        self.canvas.draw()
        self.data = {}
        self.ax_handle_ed = None
        self.ax_handles_ctr = None

    def clear_plot(self):
        self.canvas.figure.clear()
        self.canvas.draw()

    def make_gaf_data(self, serie_data = np.cos(np.arange(0,100))):
        return compute_GAF(serie_data)

    def create_plots(self, serie_data = np.cos(np.arange(0,100)), plot_type = 'GAF diagram'):
        gaf, phi, r, scaled_time_serie = self.make_gaf_data(serie_data)
        self.gaf, self.phi, self.r, self.scaled_time_serie = gaf, phi, r, scaled_time_serie
        # self.update_canvas(eval(self.parent.lineEdit_fig_size.text()))
        self.canvas.figure.clear()
        if plot_type == 'polar encoding':
            self.ax = self.canvas.figure.add_subplot(polar = True)
            self.ax.plot(phi, r)
            self.ax.set_title("Polar Encoding")
            self.ax.set_rticks([0, 1])
            self.ax.set_rlabel_position(-22.5)
            self.ax.grid(True)
        elif plot_type == 'GAF diagram':
            self.ax = self.canvas.figure.add_subplot()
            self.ax.matshow(gaf)
            self.ax.set_title("Gramian Angular Field")
            self.ax.set_yticklabels([])
            self.ax.set_xticklabels([])
        elif plot_type == 'CTR series':
            self.ax = self.canvas.figure.add_subplot()
            self.ax.plot(range(len(scaled_time_serie)), scaled_time_serie)
            self.ax.set_title("Scaled CTR Serie")
        self.canvas.draw()