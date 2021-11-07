import sys,os, qdarkstyle
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5 import uic, QtWidgets
import PyQt5
try:
    from . import locate_path_dafy
except:
    import locate_path_dafy
DaFy_path = locate_path_dafy.module_path_locator()
print(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'projects','ctr'))
sys.path.append(os.path.join(DaFy_path,'projects','ubmate'))
sys.path.append(os.path.join(DaFy_path,'projects','superrod'))
sys.path.append(os.path.join(DaFy_path,'projects','xrv'))
sys.path.append(os.path.join(DaFy_path,'projects','viewer'))
sys.path.append(os.path.join(DaFy_path,'projects'))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'dump_files'))
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
# from superrod import SuPerRod_GUI
import SuPerRod_GUI
import UBMate_GUI
import CTR_bkg_GUI
import XRV_GUI
import Data_Viewer_XRV_GUI as Viewer_GUI
import matplotlib
# matplotlib.use("TkAgg")
os.environ["QT_MAC_WANTS_LAYER"] = "1"
#import _tkinter
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton, QTableWidgetItem, QHeaderView, QAbstractItemView, QInputDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTransform, QFont, QBrush, QColor, QIcon, QImage, QPixmap
from pyqtgraph.Qt import QtGui
import imageio, cv2

class MyMainWindow(QMainWindow):

    """
    GUI class for this app
    ....
    Methods (selected)
    -----------
    """
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        # self.setupUi(self)
        #pyqtgraph preference setting
        pg.setConfigOptions(imageAxisOrder='row-major', background = (50,50,100))
        pg.mkQApp()
        #load GUI ui file made by qt designer
        uic.loadUi(os.path.join(DaFy_path,'DaFy_master_GUI.ui'),self)
        self.setWindowTitle('Data analysis factory')
        image = imageio.imread(os.path.join(DaFy_path, 'projects', 'superrod','icons', 'DAFY.png'))
        #image = imageio.imread()        
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, dsize=(int(319), int(214)), interpolation=cv2.INTER_CUBIC)
        self.im = QImage(image,image.shape[1],image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)
        self.label.setPixmap(QPixmap(self.im))

        self.pushButton_ctr.setIcon(QtGui.QIcon(os.path.join(DaFy_path,'projects','ctr','icons','ctr.png')))
        self.pushButton_ctr.setIconSize(QtCore.QSize(100,100))
        self.pushButton_superrod.setIcon(QtGui.QIcon(os.path.join(DaFy_path,'projects','superrod','icons','superrod.png')))
        self.pushButton_superrod.setIconSize(QtCore.QSize(100,100))
        self.pushButton_ubmate.setIcon(QtGui.QIcon(os.path.join(DaFy_path,'projects','ubmate','pics','ubmate.png')))
        self.pushButton_ubmate.setIconSize(QtCore.QSize(100,100))
        self.pushButton_viewer.setIcon(QtGui.QIcon(os.path.join(DaFy_path,'projects','viewer','icons','viewer.png')))
        self.pushButton_viewer.setIconSize(QtCore.QSize(100,100))
        self.pushButton_xrv.setIcon(QtGui.QIcon(os.path.join(DaFy_path,'projects','xrv','icons','xrv.png')))
        self.pushButton_xrv.setIconSize(QtCore.QSize(100,100))

        self.pushButton_superrod.clicked.connect(self.launch_superrod)
        self.pushButton_ubmate.clicked.connect(self.launch_ubmate)
        self.pushButton_xrv.clicked.connect(self.launch_xrv)
        self.pushButton_viewer.clicked.connect(self.launch_viewer)
        self.pushButton_ctr.clicked.connect(lambda:ctrWin.show())

    def launch_superrod(self):
        matplotlib.rc('image', cmap='prism')
        pg.setConfigOption('foreground', 'w')
        superrodWin.dpi = dpi
        superrodWin.show()

    def launch_ubmate(self):
        matplotlib.rc('image', cmap='plasma')
        ubmateWin.show()

    def launch_xrv(self):
        xrvWin.show()

    def launch_viewer(self):
        viewerWin.show()

if __name__ == "__main__":
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    #get dpi info: dots per inch
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    myWin = MyMainWindow()
    #ctr main window
    ctrWin = CTR_bkg_GUI.MyMainWindow()
    #xrv main window
    xrvWin = XRV_GUI.MyMainWindow()
    #ubmate main window
    ubmateWin = UBMate_GUI.MyMainWindow()
    #viewer main window
    viewerWin = Viewer_GUI.MyMainWindow()
    #superrod main window
    superrodWin = SuPerRod_GUI.MyMainWindow()
    hightlight = SuPerRod_GUI.syntax_pars.PythonHighlighter(superrodWin.plainTextEdit_script.document())
    superrodWin.plainTextEdit_script.show()
    superrodWin.plainTextEdit_script.setPlainText(superrodWin.plainTextEdit_script.toPlainText())

    # myWin.setWindowIcon(QtGui.QIcon('DAFY.png')
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())