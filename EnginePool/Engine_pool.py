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

class fit_data():
    def __init__(self):
        self.potential = []
        self.current_density = []
        self.Time = []
        self.pcov_ip = []
        self.pcov_oop = []
        self.cen_ip = []
        self.FWHM_ip = []
        self.amp_ip = []
        self.lfrac_ip = []
        self.bg_slope_ip = []
        self.bg_offset_ip = []
        self.cen_oop = []
        self.FWHM_oop = []
        self.amp_oop = []
        self.lfrac_oop = []
        self.bg_slope_oop = []
        self.bg_offset_oop = []

def gauss(x, x0, sig, amp):
    return amp*np.exp(-(x-x0)**2/2./sig**2)

def lor(x, x0, FWHM, amp):
    return amp*FWHM/((x-x0)**2+FWHM**2/4)

def pvoigt2(x, x0, FWHM, amp, lorfact):
    w = FWHM/2.
    return amp*(lorfact/(1+((x-x0)/w)**2)+(1.-lorfact)*np.exp(-np.log(2)*((x-x0)/w)**2))

def pvoigt(x, x0, FWHM, area, lfrac):
    return area / FWHM / ( lfrac*np.pi/2 + (1-lfrac)*np.sqrt(np.pi/4/np.log(2)) ) * ( lfrac / (1 + 4*((x-x0)/FWHM)**2) + (1-lfrac)*np.exp(-4*np.log(2)*((x-x0)/FWHM)**2) )

def model2(x, x0, FWHM, amp, bg_slope, bg_offset):
    return lor(x, x0, FWHM, amp) + x*bg_slope*0 + bg_offset

def model3(x, x0, FWHM, amp, bg_slope, bg_offset):
    sig = FWHM/2.35482
    return gauss(x, x0, sig, amp) + x*bg_slope*0 + bg_offset

def model(x, x0, FWHM, area, lfrac, bg_slope, bg_offset):
    return pvoigt(x, x0, FWHM, area, lfrac) + x*bg_slope + bg_offset

def extract_potential(pt_no = 2000, time_step = [10, 50, 10], pot_step = [0.2, 0.5, 0.8]):
    potential_container= []
    frames_per_time = float(pt_no)/sum(time_step)
    for i in range(len(pot_step)):
        potential_container = potential_container + [pot_step[i]]*int(time_step[i]*frames_per_time)
    for i in range(abs(len(potential_container)-pt_no)):
        potential_container.pop()
    return potential_container

def peak_fit(model, x, y, par_init, bounds, max_nfev = 10000):
    return opt.curve_fit(model, x, y, par_init, bounds, max_nfev)

def calculate_UB_matrix_p23(lattice_constants, energy, or0_angles, or1_angles,or0_hkl,or1_hkl):
    return p23.cal_UB(lattice_constants, energy, or0_angles, or1_angles,or0_hkl, or1_hkl)

def normalize_img_intensity(img, q_grid,mask_img, mask_profile, cen, offset, direction = 'horizontal'):
    if direction == 'horizontal':
        cut_mask = np.sum(mask_img[cen-offset:cen+offset+1,:], axis=0)
        cut_img = np.sum(img[cen-offset:cen+offset+1,:],axis=0)
        cut_img = cut_img/cut_mask
        cut_img = cut_img[mask_profile]
        cut_q = q_grid[0,mask_profile]
        #now remove nan values
        cut_img_nan = cut_img == np.nan
        return cut_img[~cut_img_nan],cut_q[~cut_img_nan]
    elif direction == 'vertical':
        cut_mask = np.sum(mask_img[:,cen-offset:cen+offset+1], axis=1)
        cut_img = np.sum(img[:,cen-offset:cen+offset+1],axis=1)
        cut_img = cut_img/cut_mask
        cut_img = cut_img[mask_profile]
        cut_q = q_grid[mask_profile,0]
        #now remove nan values
        cut_img_nan = cut_img == np.nan
        return cut_img[~cut_img_nan],cut_q[~cut_img_nan]
