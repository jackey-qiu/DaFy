import sys,os, qdarkstyle
from io import StringIO
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5 import uic, QtWidgets
import PyQt5
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
                return QtGui.QColor('DeepSkyBlue')
                # return QtGui.QColor('green')
            if role == QtCore.Qt.BackgroundRole and index.row()%2 == 1:
                return QtGui.QColor('aqua')
                # return QtGui.QColor('lightGreen')
            if role == QtCore.Qt.ForegroundRole and index.row()%2 == 1:
                return QtGui.QColor('black')
            '''
            if role == QtCore.Qt.CheckStateRole and index.column()==0:
                if self._data.iloc[index.row(),index.column()]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
            '''
        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        '''
        if role == QtCore.Qt.CheckStateRole and index.column() == 0:
            if value == QtCore.Qt.Checked:
                self._data.iloc[index.row(),index.column()] = True
            else:
                self._data.iloc[index.row(),index.column()] = False
        else:
        '''
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
            return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        self.layoutAboutToBeChanged.emit()
        self._data = self._data.sort_values(self._data.columns.tolist()[Ncol],
                                        ascending=order == QtCore.Qt.AscendingOrder, ignore_index = True)
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

class MplWidget2(QWidget):
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

    def update_canvas(self):
        self.canvas = FigureCanvas(Figure()) 
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

    def create_plots(self):
        self.canvas.figure.clear()
        #self.reset()
        self.generate_axis_handle(hspaces = eval(self.parent.lineEdit_hspaces.text()))
        #self.extract_data_all(self, z_min = -20, z_max = 100)
        self.extract_plot_pars()
        self.extract_format_pars()
        keys = list(self.data.keys())
        keys_rods = list(self.data[keys[0]].keys())
        offset = self.get_uniform_offset()

        def _calc_offset(min_value, offset, index):
            return min_value*(offset)**index

        #CTR data
        for i, ax in enumerate(self.ax_handles_ctr):
            ax.set_yscale('log')
            if i==0:
                ax.set_ylabel('Structure factor',fontsize = 10)
            ax.set_xlabel('L(r.l.u)', fontsize = 10)
            ax.set_title(keys_rods[i], fontsize = 10)
            for key in keys:
                l, I, Ierr, y_sim, I_ideal = self.data[key][keys_rods[i]]#ctr data [l, I, Ierr, y_sim, I_ideal]
                #if keys_rods[i]!='ed_profile':#indicate a ctr or RAXS data
                applied_offset = _calc_offset(min(y_sim),offset,keys.index(key))
                ax.plot(l, y_sim*applied_offset, color = self.plot_pars[key]['color'],linestyle =self.plot_pars[key]['ls'], label = self.plot_pars[key]['label'])
                ax.plot(l, I_ideal*applied_offset, color = '0.5')
                ax.scatter(l,I*applied_offset,s = self.plot_pars[key]['symbol_size'], marker = self.plot_pars[key]['symbol'],c = self.plot_pars[key]['symbol_color'])
            self._format_ax_tick_labels(ax, self.ax_format_pars_x[i])
            self._format_ax_tick_labels(ax, self.ax_format_pars_y[i])
        #now plot ed profile
        self.ax_handle_ed.set_xlabel('Height (Å)')
        self.ax_handle_ed.set_ylabel('Electron density (per water)')
        for i, key in enumerate(keys):
            x, y = self.data[key]['ed_profile']['Total electron density']
            # self.ax_handle_ed.plot(x, y+offset*i, color = 'white')
            self.ax_handle_ed.fill_between(x,y+offset*i,y*0+offset*i,color = self.plot_pars[key]['color'], alpha = 0.2)
        self._format_ax_tick_labels(self.ax_handle_ed, self.ax_format_pars_x[len(self.ax_handles_ctr)])
        self._format_ax_tick_labels(self.ax_handle_ed, self.ax_format_pars_y[len(self.ax_handles_ctr)])
        self.canvas.draw()

    def get_uniform_offset(self):
        return int(self.parent.lineEdit_vert_offset.text())

    def init_pandas_model(self):
        def _get_bounds_x(data):
            one_dataset = data[list(data.keys())[0]]
            bounds = []
            for each in one_dataset:
                if each!='ed_profile':
                    bounds.append([min(one_dataset[each][0]),max(one_dataset[each][0])])
                else:
                    bounds.append([float(self.parent.lineEdit_ed_min.text()), float(self.parent.lineEdit_ed_max.text())])
            return bounds

        #pnadas_model_format_pars_x
        data_ = {}
        num_axes = len(self.data[list(self.data.keys())[0]].keys())
        data_['bounds'] = _get_bounds_x(self.data)
        data_['bound_padding'] = [0.1]*num_axes
        data_['major_tick_location'] = [[]]*num_axes####to edit
        data_['show_major_tick_label'] = [True]*num_axes
        data_['num_of_minor_tick_marks'] = [4]*num_axes
        data_['fmt_str'] = ['{:4.2f}']*num_axes
        self.pandas_model_format_pars_x = PandasModel(data = pd.DataFrame(data_), tableviewer = self.parent.tableView_ax_format_x, main_gui = self.parent)
        self.parent.tableView_ax_format_x.setModel(self.pandas_model_format_pars_x)
        self.parent.tableView_ax_format_x.resizeColumnsToContents()
        self.parent.tableView_ax_format_x.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

        #pnadas_model_format_pars_y
        data_ = {}
        num_axes = len(self.data[list(self.data.keys())[0]].keys())
        data_['bounds'] = [[]]*num_axes####to edit
        data_['bound_padding'] = [0.]*num_axes
        data_['major_tick_location'] = [[]]*num_axes####to edit
        data_['show_major_tick_label'] = [True]+[False]*(num_axes-2)+[True]
        data_['num_of_minor_tick_marks'] = [4]*num_axes
        data_['fmt_str'] = ['{:.0e}']*(num_axes-1)+['{:4.0f}']
        self.pandas_model_format_pars_y = PandasModel(data = pd.DataFrame(data_), tableviewer = self.parent.tableView_ax_format_y, main_gui = self.parent)
        self.parent.tableView_ax_format_y.setModel(self.pandas_model_format_pars_y)
        self.parent.tableView_ax_format_y.resizeColumnsToContents()
        self.parent.tableView_ax_format_y.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

        #pnadas_model_format_plot_pars
        data_ = {}
        num_axes = len(list(self.data.keys()))
        data_['ls'] = ['-']*num_axes
        data_['lw'] = [1]*num_axes####to edit
        data_['color'] = ['r']*num_axes
        data_['symbol'] = ['o']*num_axes
        data_['symbol_size'] = [4]*num_axes
        data_['symbol_color'] = ['b']*num_axes
        data_['label'] = ['label']*num_axes
        self.pandas_model_plot_pars = PandasModel(data = pd.DataFrame(data_), tableviewer = self.parent.tableView_plot_pars, main_gui = self.parent)
        self.parent.tableView_plot_pars.setModel(self.pandas_model_plot_pars)
        self.parent.tableView_plot_pars.resizeColumnsToContents()
        self.parent.tableView_plot_pars.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def extract_plot_pars(self):
        #extract point/line styles for each dataset
        self.plot_pars = {}
        for i in range(len(self.pandas_model_plot_pars._data.index)):
            data_key = list(self.data.keys())[i]
            self.plot_pars[data_key] = {'ls':self.pandas_model_plot_pars._data.loc[i]['ls'],
                                        'lw':self.pandas_model_plot_pars._data.loc[i]['lw'],
                                        'color':self.pandas_model_plot_pars._data.loc[i]['color'],
                                        'symbol':self.pandas_model_plot_pars._data.loc[i]['symbol'],
                                        'symbol_size':self.pandas_model_plot_pars._data.loc[i]['symbol_size'],
                                        'symbol_color':self.pandas_model_plot_pars._data.loc[i]['symbol_color'],
                                        'label':self.pandas_model_plot_pars._data.loc[i]['label']}

    def extract_format_pars(self):
        #extract axis format pars for each axis
        self.ax_format_pars_x = {}
        self.ax_format_pars_y = {}
        for i in range(len(self.ax_handles_ctr)+1):#1 for the ed profile ax
            # ax_key = self.pandas_model_format_pars_x._data.loc[i]['ax_key']
            self.ax_format_pars_x[i] = {'fun_set_bounds':'set_xlim',
                                             'bounds':eval(str(self.pandas_model_format_pars_x._data.loc[i]['bounds'])),
                                             'bound_padding':eval(str(self.pandas_model_format_pars_x._data.loc[i]['bound_padding'])),
                                             'major_tick_location':eval(str(self.pandas_model_format_pars_x._data.loc[i]['major_tick_location'])),
                                             'show_major_tick_label':eval(str(self.pandas_model_format_pars_x._data.loc[i]['show_major_tick_label'])),
                                             'num_of_minor_tick_marks':int(self.pandas_model_format_pars_x._data.loc[i]['num_of_minor_tick_marks']),
                                             'fmt_str':self.pandas_model_format_pars_x._data.loc[i]['fmt_str']}
            # ax_key = self.pandas_model_format_pars_y._data.loc[i]['ax_key']
            self.ax_format_pars_y[i] = {'fun_set_bounds':'set_ylim',
                                             'bounds':eval(str(self.pandas_model_format_pars_y._data.loc[i]['bounds'])),
                                             'bound_padding':eval(str(self.pandas_model_format_pars_y._data.loc[i]['bound_padding'])),
                                             'major_tick_location':eval(str(self.pandas_model_format_pars_y._data.loc[i]['major_tick_location'])),
                                             'show_major_tick_label':eval(str(self.pandas_model_format_pars_y._data.loc[i]['show_major_tick_label'])),
                                             'num_of_minor_tick_marks':int(self.pandas_model_format_pars_y._data.loc[i]['num_of_minor_tick_marks']),
                                             'fmt_str':self.pandas_model_format_pars_y._data.loc[i]['fmt_str']}

    def generate_axis_handle(self, hspaces):
        col_num = len(self.data[list(self.data.keys())[0]].keys())
        gs_left = plt.GridSpec(1,col_num,hspace=hspaces[0],wspace = hspaces[0])
        gs_right = plt.GridSpec(1,col_num, hspace=hspaces[1],wspace = hspaces[1])
        ax_handles_ctr = [self.canvas.figure.add_subplot(gs_left[0, i]) for i in range(col_num-1)]
        ax_handle_ed = self.canvas.figure.add_subplot(gs_right[0, col_num-1])
        self.ax_handles_ctr = ax_handles_ctr
        self.ax_handle_ed = ax_handle_ed
        self._format_axis(self.ax_handles_ctr)
        self._format_axis(self.ax_handle_ed)

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

    def _format_ax_tick_labels(self,ax, pars):
        fun_set_bounds = pars['fun_set_bounds']
        bounds = pars['bounds']
        bound_padding=pars['bound_padding']
        major_tick_location = pars['major_tick_location']
        show_major_tick_label = pars['show_major_tick_label']
        num_of_minor_tick_marks = pars['num_of_minor_tick_marks']
        fmt_str = pars['fmt_str'] 
        mapping = {'set_ylim':'yaxis','set_xlim':'xaxis'}
        if len(bounds)==0:
            if fun_set_bounds=='set_ylim':
                bounds = ax.get_ylim()
            else:
                bounds = ax.get_xlim()
        if len(major_tick_location)==0:
            major_tick_location = getattr(ax,mapping[fun_set_bounds]).get_majorticklocs()
            # bound_padding = bounds[1]-major_tick_location[-1]
        which_axis = mapping[fun_set_bounds]
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

    def extract_data_all(self):
        self.data = {}
        z_min = float(self.parent.lineEdit_ed_min.text())
        z_max = float(self.parent.lineEdit_ed_max.text())
        folder = self.parent.lineEdit_folder_of_rod_files.text()
        files = [self.parent.listWidget_rod_files.item(i).text() for i in range(self.parent.listWidget_rod_files.count())]
        for i, file in enumerate(files):
            path = os.path.join(folder, file)
            self.data[file] = self.extract_data_for_one_file(path, z_min, z_max)

    def calc_f_ideal(self, model):
        f_ideal = []
        for i in range(len(model.data)):
            each = model.data[i]
            if each.x[0]>1000:#indicate energy column
                f_ideal.append(model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.extra_data['Y'])**2)
            else:
                f_ideal.append(model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.x)**2)
        return f_ideal

    def extract_data_for_one_file(self, file_path, z_min = -20, z_max = 100):
        def _make_key(dataset):
            ctr_or_not = dataset.x[0]<100
            h, k = int(round(dataset.extra_data['h'][0],0)), int(round(dataset.extra_data['k'][0],0))
            if not ctr_or_not:
                l = round(dataset.extra_data['Y'][0],2)
                return f"raxs_{h}_{k}_{l}"
            else:
                return f"ctr_{h}{k}L"

        model_ = model.Model()
        solver = solvergui.SolverController(model_)
        #set mask points
        model_.load(file_path)
        for each in model_.data_original:
            if not hasattr(each,'mask'):
                each.mask = np.array([True]*len(each.x))
        for each in model_.data:
            if not hasattr(each,'mask'):
                each.mask = np.array([True]*len(each.x))
        #update mask info
        model_.data = copy.deepcopy(model_.data_original)
        [each.apply_mask() for each in model_.data]
        model_.simulate()
        data = {}
        f_ideal = self.calc_f_ideal(model_)

        for i in range(len(model_.data)):
            each = model_.data[i]
            key = _make_key(each)
            data[key] = [each.x, each.y, each.error, each.y_sim, f_ideal[i]]
        #electron density
        label,edf = model_.script_module.sample.plot_electron_density_superrod(z_min=z_min, z_max=z_max,N_layered_water=500,resolution =1000, raxs_el = None, use_sym = True)
        ed_profile = {key:value for key, value in zip(label,edf)}
        data['ed_profile'] = ed_profile
        return data

