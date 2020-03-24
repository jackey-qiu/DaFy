import sys
import matplotlib
matplotlib.use("tkAgg")
from numpy import dtype
sys.path.append('./XRD_tools/')
from P23config import *
from Fio import Fiofile
import numpy as np
from scipy.interpolate import griddata
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
import p23_tools_debug as p23
import matplotlib.pyplot as plt
# from pyspec import spec
import subprocess
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FFMpegWriter

def make_empty_dictionary(keys=['potential','current_density','Time','pcov_ip',\
                                'pcov_oop','cen_ip','FWHM_ip', 'amp_ip', 'lfrac_ip',\
                                'bg_slope_ip','bg_offset_ip','cen_oop','FWHM_oop',\
                                'amp_oop','lfrac_oop','bg_slope_oop','bg_offset_oop']):
    container={}
    for key in keys:
        container[key] = []
    return container

def update_dictionary(dic, key, value):
    dic[key]=value
    return dic
