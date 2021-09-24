import numpy as np
from Reciprocal_space_tools.HKLVlieg import Crystal, printPos, UBCalculator, VliegAngles, printPos_prim, vliegDiffracAngles
from nexusformat.nexus import *
import h5py
import fnmatch
import re,os,sys
#from scipy import misc
import imageio as misc
from scipy.ndimage import gaussian_filter
from PyMca5.PyMcaIO import specfilewrapper
# from pyspec import spec
from PyMca5.PyMcaIO import EdfFile
from math import sqrt
# from skimage.feature import blob_log
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import pandas as pd
import collections
import time
from random import randint
from scipy.optimize import curve_fit
from itertools import count
try:
    import ConfigParser as configparser
except:
    import configparser
izip = zip
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DaFy_Path = os.path.dirname(currentdir)
sys.path.append(os.path.join(DaFy_Path,'EnginePool'))
from models.utils import UserVars
import FitEnginePool
import matplotlib.pyplot as pyplot
#make compatibility of py 2 and py 3#
if (sys.version_info > (3, 0)):
    raw_input = input

#place timer at different places and check the time it took at different blocks
#timer_placer.reset_timer()
#timer_placer.add_timer(name = 't0')
#timer_placer.add_timer(name = 't1')
#timer_placer.add_timer(name = 't3')
#timer_placer.report()
class timer_placer:
    def __init__(self):
        self.time_container = []
        self.place_container = []

    def reset_timer(self):
        self.time_container = []
        self.place_container = []

    def add_timer(self, name ='place'):
        self.time_container.append(time.time())
        self.place_container.append(name)

    def report(self):
        if len(self.time_container)==0:
            print('Timer placer not started!')
        else:
            print('Starting at {}'.format(self.place_container[0]))
            for i in range(1,len(self.place_container)):
                print('It took {} seconds to reach {} from {}'.format(self.time_container[i]-self.time_container[i-1], self.place_container[i],self.place_container[i-1]))

def q_correction_for_one_Bragg_peak(L,data, cell, lam, L_bragg = 1, scale=1,delta=0,c_off=0, R_tth = 1100):
    #angles are in rad unit not degree unit
    clat=cell[2]*np.sin(cell[4])
    tth=2*np.arcsin(L*lam/2./clat)
    offtth=2/R_tth*np.cos(tth/2)*(delta+c_off)-c_off/R_tth
    Q=4*np.pi/lam*np.sin((tth-offtth)/2)
    clat_corrt=clat/scale
    normcalc = (np.sin(Q*clat_corrt/4+np.pi/2*L_bragg))**2
    normdata = data*normcalc
    # y_max,y_min=normdata.max(),normdata.min()
    LL=Q*clat_corrt/2./np.pi
    return LL, normdata

def q_correction_for_one_rod(L, data, cell, lam, correction_factor_dict, R_tth = 1100):
    #angles in cell in rad unit not in degree unit
    clat=cell[2]*np.sin(cell[4])
    tth=2*np.arcsin(L*lam/2./clat)
    #delta,c_off,scale = fit_q_correction(lam,R_tth,L_Bragg_container,scale_container)
    delta = correction_factor_dict['delta']
    c_off = correction_factor_dict['c_off']
    scale = correction_factor_dict['scale']
    L_bragg = correction_factor_dict['L_bragg']
    offtth=2/R_tth*np.cos(tth/2)*(delta+c_off)-c_off/R_tth
    Q=4*np.pi/lam*np.sin((tth-offtth)/2)
    clat_corrt=clat/scale
    normcalc = (np.sin(Q*clat_corrt/4+np.pi/2*L_bragg))**2
    normdata = data*normcalc
    LL=Q*clat_corrt/2./np.pi
    return LL, normdata

def fit_q_correction(lam,R_tth,L_container,scale_container,clat=19.9490):
    tth = 2*np.arcsin(np.array(L_container)*lam/2/clat)
    data=4*np.pi/lam*np.sin( tth/2 )/np.array(scale_container)
    def _cal_q(tth,delta,c,s):
        del_tth = 2/R_tth*np.cos(tth/2)*(delta+c) - c/R_tth
        return 4*np.pi/lam*np.sin((tth-del_tth)/2)/s
    popt, pcov = curve_fit(_cal_q, tth, data, bounds=([-0.3,-0.3,0.98], [0.3, 0.3, 1.1]))
    print ('fit is completed!')
    print ('delta,c_off,scale=',popt)
    return popt

def setup_raxr_pars_new(NUMBER_SPECTRA,F1F2_FILE):
    if NUMBER_SPECTRA!=0:
        F1F2=np.loadtxt(os.path.join(F1F2_FILE))
        rgh_raxr=UserVars()
        for i in range(NUMBER_SPECTRA):
            rgh_raxr.new_var('a'+str(i+1),1.0)
            rgh_raxr.new_var('b'+str(i+1),1.0)
            rgh_raxr.new_var('c'+str(i+1),1.0)
            rgh_raxr.new_var('A_'+str(i+1),0.2)
            rgh_raxr.new_var('P_'+str(i+1),0.0) 
    else:
        rgh_raxr=None
        F1F2=None
    return rgh_raxr,F1F2

def config_file_parser_bv(config_file):
    return_lib = {}
    config = configparser.RawConfigParser()
    #preserve the case of items
    config.optionxform = str
    config.read(config_file)
    for section in config.sections():
        return_lib[section] = {}
        for each in config.items(section):
            each = list(each)
            if section in ['R0_BV', 'IDEAL_BOND_LENGTH']:
                each[0] = tuple([each_ for each_ in each[0].rsplit("_")])
            else:
                pass
            return_lib[section][each[0]] = eval(each[1])
    settings = return_lib['SETTINGS']
    del return_lib['SETTINGS']
    #extract items in SETTINGS tag
    return_lib.update(settings)
    return return_lib

#O_NUMBER_lib = {'O_NUMBER_HL':[[0,4]],'O_NUMBER_HL_EXTRA':[[0,3]]}
def update_O_NUMBER(sorbate,pickup_index,metal_valence):
    O_NUMBER_lib = {'O_NUMBER_HL':[],'O_NUMBER_HL_EXTRA':[],'O_NUMBER_FL':[],'O_NUMBER_FL_EXTRA':[]}
    key_map ={'O_NUMBER_HL':list(range(0,8)),'O_NUMBER_HL_EXTRA':list(range(8,15)),'O_NUMBER_FL':list(range(15,23)),'O_NUMBER_FL_EXTRA':list(range(23,30))} 
    match_table = {0:[6,7,21,22],1:[8,9,10,23,24,25],2:[0,1,2,3,11,12,13,15,16,17,18,26,27,28],3:[4,5,14,19,20,19]}
    for i in range(len(sorbate)):
        current_sorbate_list = sorbate[i]
        current_index_list = pickup_index[i]
        for j in range(len(current_sorbate_list)):
            current_sorbate = current_sorbate_list[j]
            current_index = current_index_list[j]
            current_key = None
            for key in key_map:
                if current_index in key_map[key]:
                    current_key = key
                    break
            anchor_o_number = 0 
            for key in match_table:
                if current_index in match_table[key]:
                    anchor_o_number = key
                    break
            CN = metal_valence[current_sorbate][1]
            temp = [int(current_index-key_map[current_key][0]),int(CN-anchor_o_number)]
            if temp not in O_NUMBER_lib[current_key]:
                O_NUMBER_lib[current_key].append(temp)
    O_NUMBER_lib_final = {}

    for key in O_NUMBER_lib:
        if len(O_NUMBER_lib[key])!=0:
            O_NUMBER_lib_final[key] = O_NUMBER_lib[key]
    return O_NUMBER_lib_final

def locate_tag(lines, tag='wavelength'):
    target_zone_begin = None
    target_zone_end = None
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('#/{}/begin#'.format(tag)):
            target_zone_begin = i+1
        elif line.startswith('#/{}/end#'.format(tag)):
            target_zone_end = i
            break
    return target_zone_begin, target_zone_end

def apply_modification_of_code_block(lines, tag, vars_labels = [], vars_values = []):
    target_begin, target_end = locate_tag(lines, tag)
    for i in range(target_begin, target_end):
        for var in vars_labels:
            if lines[i].startswith(var):
                lines[i] = '{} = {}\n'.format(var, vars_values[vars_labels.index(var)])
    return lines
    
def replace_block(lines, tag, new_item):
    target_begin, target_end = locate_tag(lines, tag)
    for i in range(target_end-target_begin):
        lines.pop(target_begin)
    lines.insert(target_begin, new_item)
    return lines

def show_status_bar(img_loader, column_size_offset = 22):
    finish_percent = (img_loader.frame_number+1)/float(img_loader.total_frame_number)
    column_size = int(get_console_size()[0])-column_size_offset
    left_abnormal = int((img_loader.abnormal_range[0]+1)/float(img_loader.total_frame_number)*column_size+8)
    right_abnormal = int((img_loader.abnormal_range[1]+1)/float(img_loader.total_frame_number)*column_size+8)
    output_text =list('{}{}{}{}{}'.format('BEGIN(0)','='*int(finish_percent*column_size),'==>',' '*int((1-finish_percent)*column_size),'>|END('+str(img_loader.total_frame_number)+')'))
    for index_text in range(len(output_text)):
        if output_text[index_text]!=' ' and index_text>left_abnormal and index_text<right_abnormal:
            output_text[index_text] = 'x'
        else:
            pass
    print(''.join(output_text),end="\r")
    time.sleep(0.003)

def show_status_bar_2(img_loader, column_size_offset = 22):
    finish_percent = (img_loader.frame_number+1)/float(img_loader.total_frame_number)
    column_size = int(get_console_size()[0])-column_size_offset
    output_text =list('Frame{}{}{}{}{}'.format(img_loader.frame_number,'='*int(finish_percent*column_size),'==>',' '*int((1-finish_percent)*column_size),'>|END('+str(img_loader.total_frame_number)+')'))
    print(''.join(output_text),end="\r")
    time.sleep(0.003)

def make_tweak_string():
    tweak_motion_str = raw_input(", splited stream of string\n\
                        UD:up or down for bkg\n\
                        LR:left or right for bkg\n\
                        CW:column width for bkg\n\
                        RW:row width for bkg\n\
                        ud:up or down for bkg\n\
                        lr:left or right\n\
                        cw:column width\n\
                        rw:row width\n\
                        pw:peak width\n\
                        ps:peak shift\n\
                        od:polynomial order\n\
                        sf:ss_factor, smaller lower bkg\n\
                        r:repeat last motion\n\
                        #r:repeat motion for rest points\n\
                        ft:fit function(ah, sh, stq, atq)\n\
                        qw:quit and write date\n\
                        rm:remove current date and quit\n\
                        Your input is:") or 'qw'
    return tweak_motion_str


def tweak_integration(integration_object, tweak_motion_str, pre_tweak_motion):
    repeat_last = ''
    tweak = ''
    process_through = False
    tweak_return = integration_object.update_var_for_tweak(tweak_motion_str)
    if tweak_return in ['process_through','qw','rm']:
        tweak = False
    else:
        tweak = True
    if tweak_return == 'tweak':
        repeat_last = False
    elif tweak_return == 'repeat_last':
        repeat_last = True
    if tweak_return != 'repeat_last':
        #all parameters are updated in previous step, so you just 'qw' to repeat.
        pre_tweak_motion = tweak_motion_str
    else:
        _ = integration_object.update_var_for_tweak(pre_tweak_motion)
    if tweak_return == 'process_through':
        process_through = True
        repeat_last = False
    return integration_object, tweak, tweak_return, repeat_last, pre_tweak_motion, process_through

#define generator funcs to hold scans and images
def scan_generator(scans):
    _scans = []
    for each in scans:
        if type(each)==type([]):
            _scans = _scans + list(range(each[0],each[1]+1))
        else:
            _scans = _scans + [each]
    for scan in _scans:
        yield scan

def image_generator(scans,img_loader,rsp_instance,peak_fitting_instance,mask_creator):
    for scan in scans:
        img_loader.update_scan_info(scan)
        current_image_no = 0
        for image in img_loader.load_frame(frame_number=0, flip=True):
            rsp_instance.update_img(image,motor_angles = img_loader.motor_angles, update_q = current_image_no == 0)
            if current_image_no == 0:
                peak_fitting_instance.q_ip = rsp_instance.q['grid_q_par']
                peak_fitting_instance.q_oop = rsp_instance.q['grid_q_perp']
                peak_fitting_instance.initiat_p0_and_bounds()
            yield gaussian_filter(mask_creator.create_mask_new(img = rsp_instance.grid_intensity, img_q_ver = rsp_instance.q['grid_q_perp'],
                                  img_q_par = rsp_instance.q['grid_q_par'], mon = img_loader.motor_angles['mon']*img_loader.motor_angles['transm']),sigma = 0)
            current_image_no +=1

def image_generator_bkg(scans,img_loader,mask_creator):
    for scan in scans:
        img_loader.update_scan_info(scan)
        current_image_no = 0
        img_index_ver, img_index_hor = None, None
        for image in img_loader.load_frame(frame_number=0, flip=True):
            if current_image_no==0:
                img_index_hor, img_index_ver = np.meshgrid(range(image.shape[1]),range(image.shape[0]))
            else:
                pass
            yield mask_creator.create_mask_new(img = image, img_q_ver = img_index_ver,
                                  img_q_par = img_index_hor, mon = img_loader.motor_angles['mon']*img_loader.motor_angles['transm'])
            current_image_no +=1

def image_generator_bkg_gsecars(scans,img_loader,mask_creator):
    for scan in scans:
        img_loader.update_scan_info(scan)
        current_image_no = 0
        img_index_ver, img_index_hor = None, None
        for image in img_loader.load_frame(frame_number=0, flip=True):
            if current_image_no==0:
                img_index_hor, img_index_ver = np.meshgrid(range(image.shape[1]),range(image.shape[0]))
            else:
                pass
            yield mask_creator.create_mask_new(img = image, img_q_ver = img_index_ver,
                                  img_q_par = img_index_hor, mon = img_loader.motor_angles['norm']*img_loader.motor_angles['transmission'])
            current_image_no +=1

def extract_vars_from_config(config_file, section_var):
    config = configparser.ConfigParser()
    config.read(config_file)
    kwarg = {}
    for each in config.items(section_var):
        try:
            kwarg[each[0]] = eval(each[1])
        except:
            kwarg[each[0]] = each[1]

    return kwarg

def get_console_size():
    if sys.platform == 'win32':
        from ctypes import windll, create_string_buffer
        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12

        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)

        if res:
            import struct
            (bufx, bufy, curx, cury, wattr,
            left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
        else:
            sizex, sizey = 80, 25 # can't determine actual size - return default values
    else:
        sizex, sizey = os.popen('stty size','r').read().split()[::-1]

    return sizex, sizey

def extract_global_vars_from_string(keys_in_order=[],job=[]):
    return_lib = {}
    for key in keys_in_order:
        return_lib[key] = []
    for each in job:
        items = each.rstrip().rsplit()
        for i in range(len(items)):
            try:
                item = eval(items[i])
            except:
                item = items[i]
            return_lib[keys_in_order[i]].append(item)
    return return_lib

#how many frames to be processed?
def find_boundary(n_process,n_jobs,rank):
    step_len=int(n_jobs/n_process)
    remainder=int(n_jobs%n_process)
    left,right=0,0
    if rank<=remainder-1:
        left=rank*(step_len+1)
        right=(rank+1)*(step_len+1)-1
    elif rank>remainder-1:
        left=remainder*(step_len+1)+(rank-remainder)*step_len
        right=remainder*(step_len+1)+(rank-remainder+1)*step_len-1
    return left,right

def remove_abnormality(mon, left_offset,right_offset):
    mon = np.array(mon)
    max_index = np.argmax(mon)
    print('kick of data points from {} to {}'.format(max_index-left_offset, max_index+right_offset))
    return [i for i in range(len(mon)) if (i<(max_index - left_offset) or i>(max_index+right_offset))]

def remove_abnormality_2(mon, left_offset,right_offset):
    mon = np.array(mon)
    max_index = np.argmax(mon)
    print('kick of data points from {} to {}'.format(max_index-left_offset, max_index+right_offset))
    return max([0,max_index - left_offset]), min([max_index+right_offset,len(mon)])

# 3. Lets define some use-case specific UDF(User Defined Functions)

def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution

    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def explain_anomalies(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using stationary standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation
    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                       index, y_i, avg_i in izip(count(), y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using rolling standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    # testing_std = pd.rolling_std(residual, window_size)
    testing_std = pd.rolling(residual, window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                       for index, y_i, avg_i, rs_i in izip(count(),
                                                                                           y, avg_list, rolling_std)
              if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}


# This function is repsonsible for displaying how the function performs on the given dataset.
def plot_results(x, y, window_size, sigma_value=1,
                 text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies.
        Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        text_xlabel (str): label for annotating the X Axis
        text_ylabel (str): label for annotatin the Y Axis
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    plt.figure(figsize=(15, 8))
    plt.plot(x, y, "k.")
    y_av = moving_average(y, window_size)
    plt.plot(x, y_av, color='green')
    plt.xlim(0, 1000)
    plt.xlabel(text_xlabel)
    plt.ylabel(text_ylabel)

    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

    x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
    y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float,
                                            count=len(events['anomalies_dict']))
    plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)

    # add grid and lines and enable the plot
    plt.grid(True)
    plt.show()

def check_abnormality(data, mask, tolerance_factor = 3):
    data, mask = np.array(data), np.array(mask)
    last_accpt_pt = np.where(mask==1)[0][-1]
    #intrinsic_offset = np.array(len(data)- last_accpt_pt)
    data_filtered = data[np.where(mask==1)]
    if np.abs(data[-1]-data_filtered[-1])>tolerance_factor:
        return False
    else:
        return True

def check_abnormality_old(data, mask,pot, tolerance_factor = 3):
    data, mask, pot = np.array(data), np.array(mask), np.array(pot)
    last_accpt_pt = np.where(mask==1)[0][-1]
    pot_diff = 1*(np.diff(pot)<0)
    intrinsic_offset = sum(pot_diff[last_accpt_pt-1:])
    #intrinsic_offset = np.array(len(data)- last_accpt_pt)
    data_filtered = data[np.where(mask==1)]
    diff_mean = np.abs(np.diff(data_filtered)).mean()
    if abs(np.abs(data[-1]-data_filtered[-1])+diff_mean*intrinsic_offset)>diff_mean*tolerance_factor:
        return False
    else:
        return True


def peak_checker(img_path='/media/qiu/JACKEY/1704_APS_13IDC/images/sb1_32mM_CaCl2_Zr_1/S013/sb1_32mM_CaCl2_Zr_1_S013_00207.tif',max_sigma= 30, num_sigma = 10, threshold = 0.1):
    img = misc.imread(img_path)
    img[np.isnan(img)] = 0
    img=img/img.max()
    blobs_log = blob_log(img, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    ax.imshow(img, interpolation='nearest')
    for blob in blobs_log:
        y, x, r = blob
        print(x,y,r)
        c = plt.Circle((x, y), r, color='g', linewidth=2, fill=False)
        ax.add_patch(c)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def gen_find_old(filepat, top):
    #find all filenames in a dic tree that match a shell wildcard pattern
    files = []
    for path, dirlist, filelist  in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            files.append(os.path.join(path,name))
    return files

def gen_find(filepat, top):
    #find all filenames in a dic tree that match a shell wildcard pattern
    files = [f for f in os.listdir(top) if re.match(filepat, f)]
    return files

def extract_arg(config, section, local_lib):
    arg_list=[]
    temp = dict(config.items(section))
    keys=temp.keys()
    keys.sort()
    for key in keys:
        try:
            arg_list.append(eval(temp[key],None,local_lib))
        except:
            arg_list.append(None)
    return arg_list

def collect_args(local_lib,tag):
    args_list = [None]*20
    for key in local_lib.keys():
        items_key = key.rsplit("_")
        if tag == items_key[-1]:
            args_list[int(items_key[-2][3:])-1]=local_lib[key]
    return [value for value in args_list if value!= None]

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def update_data(data,keys,values):
    for key, value in zip(keys,values):
        data[key].append(value)
    return data

def pop_last_item(data,keys=None):
    if keys == None:
        keys = data.keys()
    for key in keys:
        data[key].pop()
    return data

def extract_potential(pt_no = 2000, time_step = [10, 50, 10], pot_step = [0.2, 0.5, 0.8]):
    potential_container= []
    frames_per_time = float(pt_no)/sum(time_step)
    for i in range(len(pot_step)):
        potential_container = potential_container + [pot_step[i]]*int(time_step[i]*frames_per_time)
    for i in range(abs(len(potential_container)-pt_no)):
        potential_container.pop()
    return potential_container

def cal_UB_p23(lattice_constants=[2.8837,2.8837,7.0636,90,90,120],energy=18.739,or0_angles=[0.4,15.4,22.43,-30.9,0.,0.],or1_angles=[0.4,7.61,13.63,-38.,0.,0.],or0_hkl=[1.0009,1.-1.0009,4.0359],or1_hkl=[0.0,-0.5045,2.5225]):
    substrate=Crystal(lattice_constants[0:3],lattice_constants[3:])
    # or0_angles=[0.4,15.4,22.43,-30.9,0.,0.]
    # or1_angles=[0.4,7.61,13.63,-38.,0.,0.]
    # or0_hkl=[1.0009,1.-1.0009,4.0359]
    # or1_hkl=[0.0,-0.5045,2.5225]
    # energy=18.739
    ub_substrate=UBCalculator(substrate,energy)
    or0_angles=np.deg2rad(or0_angles)
    or1_angles=np.deg2rad(or1_angles)
    or0_angles = vliegDiffracAngles(or0_angles)
    or1_angles = vliegDiffracAngles(or1_angles)
    ub_substrate.setPrimaryReflection(or0_angles,or0_hkl)
    ub_substrate.setSecondayReflection(or1_angles,or1_hkl)
    ub_substrate.calculateU()
    return ub_substrate.getUB()

def cal_UB_p23_2(kwarg):
    # or0_angles=[0.4,15.4,22.43,-30.9,0.,0.]
    # or1_angles=[0.4,7.61,13.63,-38.,0.,0.]
    # or0_hkl=[1.0009,1.-1.0009,4.0359]
    # or1_hkl=[0.0,-0.5045,2.5225]
    # energy=18.739
    lattice_constants = kwarg['lattice_constants']
    substrate = Crystal(lattice_constants[0:3],lattice_constants[3:])
    energy = kwarg['energy']
    or0_angles = kwarg['or0_angles']
    or1_angles = kwarg['or1_angles']
    or0_hkl = kwarg['or0_hkl']
    or1_hkl = kwarg['or1_hkl']

    ub_substrate=UBCalculator(substrate,energy)
    or0_angles=np.deg2rad(or0_angles)
    or1_angles=np.deg2rad(or1_angles)
    or0_angles = vliegDiffracAngles(or0_angles)
    or1_angles = vliegDiffracAngles(or1_angles)
    ub_substrate.setPrimaryReflection(or0_angles,or0_hkl)
    ub_substrate.setSecondayReflection(or1_angles,or1_hkl)
    ub_substrate.calculateU()
    return ub_substrate.getUB()

def cal_UB_id03(spec_file_name, scan_no):
    scan = specfilewrapper.Specfile(spec_file_name).select('{}.1'.format(scan_no))
    return np.array(scan.header('G')[2].split(' ')[-9:],dtype = np.float)

def get_UB(name = 'P23'):
    if name in ['P23','DIFFABS']:
        return cal_UB_p23
    elif name =='ID03':
        return cal_UB_id03

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

def update_bounds(original_bounds, guess_values,offset=0.1, which=0):
    original_bounds_left, original_bounds_right = original_bounds
    guess_value = guess_values[which]
    if guess_value > original_bounds_right[which]:
        original_bounds_right[which] = guess_value + offset
    elif guess_value < original_bounds_left[which]:
        original_bounds_left[which] = guess_value - offset
    return [original_bounds_left,original-bounds_right]

def debug_output(local_lib,var_list,debug):
    if debug:
        for var in var_list:
            print(var,local_lib[var])
    return None

class nexus_image_loader_diffabs(object):
    def __init__(self,nexus_path='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/FirstTest_00666/lmbd',frame_prefix='FirstTest'):
        self.nexus_path=nexus_path
        self.frame_prefix=frame_prefix
        # self.get_frame_number()
        self.current_scan_no=None
        self.current_scan_img_no=None
        self.current_energy = None

    def get_frame_number(self, frame_prefix,scan_no):
        img_name='{}_{}.nxs'.format(frame_prefix,current_scan_no)
        img_path=os.path.join(self.nexus_path,img_name)
        # data=nxload(img_path)
        data=h5py.File(img_path,'r')
        access_path="scan_{}/scan_data/data_57".format(scan_number)
        total_img_number = data[access_path].shape[0]
        self.total_frame_number = total_img_number
        return self.total_frame_number

    def load_frame(self,scan_number,frame_number,offset={},flip=True):
        img_name='{}_{}.nxs'.format(self.frame_prefix,scan_number)
        self.current_scan_no, self.current_scan_img_no = scan_number, frame_number
        img_path=os.path.join(self.nexus_path,img_name)
        # data=nxload(img_path)
        data=h5py.File(img_path,'r')
        access_path="scan_{}/scan_data/data_57".format(scan_number)
        img=np.array(data[access_path][frame_number])
        self.total_frame_number = data[access_path].shape[0]
        self.extract_motor_angles(data, scan_number, frame_number,offset)
        self.update_energy(data, scan_number, frame_number)
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def extract_motor_angles(self, scan, scan_number,frame_number, offset ={'delta':0}):
        motors={}
        common_head_header = "scan_{}/DIFFABS/D13-1-CX1__EX__DIF.1-{}__#1/raw_value"
        common_head_counter ="scan_{}/scan_data/{}"
        motor_names = ['phi', 'chi', 'delta', 'gamma', 'mu', 'omega_t']
        motor_map_lib = {'phi':["header","PHI_E"], "chi":["header","CHI_E"], \
                         'delta': ["counter", "actuator_1_4"], "gamma":["header","GAMMA"],\
                         "mu":['header','MU'],'omega_t':['counter','actuator_1_3']}
        for key in motor_map_lib:
            pot, map_key = motor_map_lib[key]
            _offset = 0
            if key in offset.keys():
                _offset = offset[key]
            if pot == "header":
                # print(key,common_head_header.format(scan_number,map_key))
                motors[key] =np.array(scan[common_head_header.format(scan_number,map_key)])[0]+_offset
            elif pot == "counter":
                # print(key, map_key,common_head_counter.format(scan_number,map_key))
                motors[key] = np.array(scan[common_head_counter.format(scan_number,map_key)])[frame_number]+_offset
        self.motor_angles = motors

    def update_energy(self, scan, scan_number, frame_number):
        e_counter ="scan_{}/scan_data/actuator_1_1".format(scan_number)
        energy = np.array(scan[e_counter])[frame_number]
        self.current_energy = energy
        return energy

    def load_frame_from_path(self,img_path,frame_number = 0,flip=True):
        try:
            #if one frame one nxs file
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data.nxdata[0])
        except:
            #if all frames in one nxs file
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data[frame_number])
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def show_frame(self,scan_number,frame_number,one_frame_in_one_nxs=True,flip=True):
        img=self.load_frame(scan_number,frame_number,one_frame_in_one_nxs,flip)
        fig,ax=pyplot.subplots()
        pyplot.imshow(img,cmap='jet')
        if flip:
            pyplot.colorbar(extend='both',orientation='vertical')
        else:
            pyplot.colorbar(extend='both',orientation='horizontal')
        pyplot.clim(0,205)
        # pyplot.show()
        return img

    def find_dead_pix(self,scan_number=666,img_end=100):
        dead_pix_container=self.load_frame(scan_number,0)==self.load_frame(scan_number,1)
        dead_pix_container=np.where(dead_pix_container==True)
        dead_pix_container=zip(tuple(dead_pix_container[0]),tuple(dead_pix_container[1]))
        img0= self.load_frame(scan_number,0)
        print(len(dead_pix_container))
        for i in range(2,img_end):
            print('Processing img_',i)
            img = self.load_frame(scan_number,i)
            temp= img != img0
            temp= np.where(temp==True)
            temp= zip(tuple(temp[0]),tuple(temp[1]))
            for each in temp:
                if each in dead_pix_container:
                    dead_pix_container.remove(each)
        return dead_pix_container

class nexus_image_loader_old(object):
    def __init__(self, clip_boundary, FioFile, kwarg):
        # self.fio_path=fio_path
        # self.nexus_path=nexus_path
        # self.frame_prefix=frame_prefix
        # self.spec = Fio.FioFile(fio_path)
        # self.motor_angles = motor_angles
        # self.get_frame_number()
        self.abnormal_range = [0,0]
        self.FioFile = FioFile
        self.scan_number = None
        self.frame_number = None
        self.img_structure = 'one'#by default all images saved in one nexus file
        self.potential = None
        self.current = None
        self.hkl = None
        self.clip_boundary = clip_boundary
        self.potential_profile_cal = None
        self.potential_profile = None
        self.potential_cal = None
        for key in kwarg:
            setattr(self, key, kwarg[key])

    def get_frame_number(self):
        image_path = os.path.join(self.nexus_path,'{}_{:0>5}/lmbd'.format(self.frame_prefix,self.scan_number))
        total_img_number = len(os.listdir(image_path))
        if total_img_number ==1:
            img_name = os.listdir(self.nexus_path)[0]
            img_path=os.path.join(self.nexus_path,img_name)
            data=nxload(img_path)
            total_img_number = len(np.array(data.entry.instrument.detector.data))
        self.total_frame_number = total_img_number
        return self.total_frame_number

    def update_scan_info(self,scan_number):
        self.fio_path = os.path.join(self.nexus_path,'startup','{}_{:0>5}.fio'.format(self.frame_prefix,scan_number))
        self.spec = self.FioFile(self.fio_path)
        img_path = os.path.join(self.nexus_path,'{}_{:0>5}/lmbd'.format(self.frame_prefix,scan_number))
        self.scan_number = scan_number
        print('\nRunning scan {} now...'.format(scan_number))
        img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        img_name_1='{}_{:0>5}_00000.nxs'.format(self.frame_prefix,scan_number)
        # img_path=os.path.join(self.nexus_path,img_name)
        img_path_1=os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd',img_name_1)
        # self.nexus_data = nxload(img_path)
        self.nexus_data_1 = nxload(img_path_1)
        self.get_frame_number()
        if abs(self.total_frame_number - self.nexus_data_1.entry.instrument.detector.data.shape[0])>2:
            self.img_structure = 'multiple'#means each image correspond to one nexus fle
        self.extract_pot_profile()

    def extract_pot_profile(self):
        results = self.spec.extract_pot_profile(self.scan_number)
        self.potential_profile = results
        self.potential_profile_cal = FitEnginePool.fit_pot_profile(list(range(len(results))),results, show_fig = False)

    def extract_motor_angles(self, frame_number):
        self.motor_angles = self.spec.extract_motor_angle(self.motor_angles,self.scan_number,frame_number,['mu','delta','gamma','omega_t'])
        self.motor_angles['mon'] = 1
        self.motor_angles['transm'] = 1
        self.motor_angles['time'] = self.spec.get_col('timestamp')[frame_number] 
        return self.motor_angles

    def extract_pot_current(self, frame_number):
        results = self.spec.extract_pot_current(self.scan_number, frame_number)
        self.potential, self.current = results
        try:
            self.potential_cal = self.potential_profile_cal[frame_number]
        except:
            self.potential_cal = results[0]
            print('Use real potential for the potential_cal')
        return results

    def extract_HKL(self, frame_number):
        self.hkl = (0,0,0)
        return 0,0,0

    def load_frame(self,frame_number,flip=True):
        while frame_number < self.total_frame_number:
            if self.img_structure == 'multiple':
                #if one frame one nxs file
                folder_name = '{}_{:0>5}'.format(self.frame_prefix,self.scan_number)
                img_name='{}_{:0>5}_{:0>5}.nxs'.format(self.frame_prefix,self.scan_number,frame_number)
                img_path=os.path.join(self.nexus_path,folder_name, 'lmbd', img_name)
                data=nxload(img_path)
                img=np.array(data.entry.instrument.detector.data.nxdata[0])
            else:
                #if all frames in one nxs file
                folder_name = '{}_{:0>5}'.format(self.frame_prefix,self.scan_number)
                img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,self.scan_number)
                img_path=os.path.join(self.nexus_path,folder_name, 'lmbd', img_name)
                data=nxload(img_path)
                img=np.array(data.entry.instrument.detector.data[frame_number])
            self.extract_motor_angles(frame_number)
            self.extract_pot_current(frame_number)
            self.extract_HKL(frame_number)
            self.frame_number = frame_number
            if flip:
                img = np.flip(img.T,1)
            img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                    self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
            #yield img/self.motor_angles['mon']/self.motor_angles['transm']
            yield img
            frame_number +=1

    def load_frame_from_path(self,img_path,frame_number = 0,flip=True):
        try:
            #if one frame one nxs file
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data.nxdata[0])
        except:
            #if all frames in one nxs file
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data[frame_number])
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def show_frame(self,scan_number,frame_number,one_frame_in_one_nxs=True,flip=True):
        img=self.load_frame(scan_number,frame_number,one_frame_in_one_nxs,flip)
        fig,ax=pyplot.subplots()
        pyplot.imshow(img,cmap='jet')
        if flip:
            pyplot.colorbar(extend='both',orientation='vertical')
        else:
            pyplot.colorbar(extend='both',orientation='horizontal')
        pyplot.clim(0,205)
        # pyplot.show()
        return img

    def find_dead_pix(self,scan_number=666,img_end=100):
        dead_pix_container=self.load_frame(scan_number,0)==self.load_frame(scan_number,1)
        dead_pix_container=np.where(dead_pix_container==True)
        dead_pix_container=zip(tuple(dead_pix_container[0]),tuple(dead_pix_container[1]))
        img0= self.load_frame(scan_number,0)
        print(len(dead_pix_container))
        for i in range(2,img_end):
            print('Processing img_',i)
            img = self.load_frame(scan_number,i)
            temp= img != img0
            temp= np.where(temp==True)
            temp= zip(tuple(temp[0]),tuple(temp[1]))
            for each in temp:
                if each in dead_pix_container:
                    dead_pix_container.remove(each)
        return dead_pix_container

class nexus_image_loader(object):
    def __init__(self,clip_boundary,kwarg):
        # self.nexus_path=nexus_path
        # self.frame_prefix=frame_prefix
        self.scan_number = None
        self.frame_number = None
        self.img_structure = 'one'#by default all images saved in one nexus file
        self.potential = None
        self.current = None
        self.hkl = None
        self.clip_boundary = clip_boundary
        self.potential_profile_cal = None
        self.potential_profile = None
        self.potential_cal = None
        for key in kwarg:
            setattr(self, key, kwarg[key])
        # self.constant_motors = constant_motors
        #load nexus data only once here
        #img_name='{}_{:0>5}.nxs'.format(frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #self.nexus_data = nxload(img_path)
        #self.get_frame_number()

    def update_scan_info_old(self,scan_number):
        self.scan_number = scan_number
        print('\nRunning scan {} now...'.format(scan_number))
        img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        img_name_1='{}_{:0>5}_00000.nxs'.format(self.frame_prefix,scan_number)
        img_path=os.path.join(self.nexus_path,img_name)
        img_path_1=os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd',img_name_1)
        self.nexus_data = nxload(img_path)
        self.nexus_data_1 = nxload(img_path_1)
        self.get_frame_number()
        self.extract_pot_profile()
        if self.check_abnormality:
            self.abnormal_range = remove_abnormality_2(mon = self.extract_beam_mon_ct(),left_offset = self.left_offset, right_offset = self.right_offset)
        else:
            self.abnormal_range = [-10,-1]

    def update_scan_info(self,scan_number):
        self.scan_number = scan_number
        print('\nRunning scan {} now...'.format(scan_number))
        img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        img_name_1='{}_{:0>5}_00000.nxs'.format(self.frame_prefix,scan_number)
        img_path=os.path.join(self.nexus_path,img_name)
        img_path_1=os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd',img_name_1)
        self.nexus_data = nxload(img_path)
        self.nexus_data_1 = nxload(img_path_1)
        self.get_frame_number()
        if len(os.listdir(os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd'))) == 1:
            self.img_structure = 'one'
            #update the total frame_number to whichever smaller
            self.total_frame_number =min(self.total_frame_number, self.nexus_data_1.entry.instrument.detector.data.shape[0])
        else:
            self.img_structure = 'multiple'
            #update the total frame_number to whichever smaller
            self.total_frame_number = min(self.total_frame_number, len(os.listdir(os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd'))))
        # if abs(self.total_frame_number - self.nexus_data_1.entry.instrument.detector.data.shape[0])>2:
            # self.img_structure = 'multiple'#means each image correspond to one nexus fle
        self.extract_pot_profile()
        if self.check_abnormality:
            self.abnormal_range = remove_abnormality_2(mon = self.extract_beam_mon_ct(),left_offset = self.left_offset, right_offset = self.right_offset)
        else:
            self.abnormal_range = [-10,-1]

    def get_frame_number(self):
        #total_img_number = len(os.listdir(self.nexus_path))
        #if total_img_number ==1:
            #img_name = os.listdir(self.nexus_path)[0]
            #img_path=os.path.join(self.nexus_path,img_name)
            #data=nxload(img_path)
            #try:
            #    total_img_number = len(np.array(self.nexus_data.entry.instrument.detector.data))
            #except:
            #    total_img_number = len(np.array(self.nexus_data.scan.data.atten))
            #total_img_number = len(np.array(self.nexus_data.scan.data.atten))
            #print(total_img_number)
        total_img_number = len(np.array(self.nexus_data.scan.data.eh_c01))
        self.total_frame_number = total_img_number
        return total_img_number

    def load_frame_old(self,scan_number,frame_number,flip=True):
        try:
            #if one frame one nxs file
            img_name='{}_{:0>5}_{:0>5}.nxs'.format(self.frame_prefix,scan_number,frame_number)
            img_path=os.path.join(self.nexus_path,img_name)
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data.nxdata[0])
        except:
            #if all frames in one nxs file
            img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
            img_path=os.path.join(self.nexus_path,img_name)
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data[frame_number])
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def load_frame_new(self,frame_number,flip=True, clip_boundary = {'ver':[0,10000],'hor':[0,10000]}):
        #if one frame one nxs file
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        #img=np.array(data.entry.instrument.detector.data.nxdata[0])
        img=np.array(self.nexus_data.scan.data.lmbd)[frame_number]
        if flip:
            img = np.flip(img.T,1)
        img = img[clip_boundary['ver'][0]:clip_boundary['ver'][1],clip_boundary['hor'][0]:clip_boundary['hor'][1]]
        return img

    def load_one_frame(self,frame_number,flip = True):
        #if one frame one nxs file
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        #img=np.array(data.entry.instrument.detector.data.nxdata[0])
        if self.img_structure=='one':
            if (frame_number < self.total_frame_number) and (frame_number >= 0):
                img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(frame_number)
                #print(self.nexus_data_1.entry.instrument.detector.data.shape)
                #print(img)
                if img is None:
                    img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(frame_number-1)
                if flip:
                    img = np.flip(img.T,1)
                img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                        self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
                #normalized the intensity by the monitor and trams counters
                self.extract_motor_angles(frame_number)
                self.extract_pot_current(frame_number)
                self.extract_HKL(frame_number)
                return img/self.extract_transm_and_mon(frame_number)

        elif self.img_structure == 'multiple':
            if (frame_number < self.total_frame_number) and (frame_number >= 0):
                img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,self.scan_number)
                img_name_1='{}_{:0>5}_{:0>5}.nxs'.format(self.frame_prefix,self.scan_number,frame_number)
                #print(img_name_1,img_name)
                img_path_1=os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd',img_name_1)
                self.nexus_data_1 = nxload(img_path_1)
                img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(0)
                
                if flip:
                    img = np.flip(img.T,1)
                img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                        self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
                #normalized the intensity by the monitor and trams counters
                self.extract_motor_angles(frame_number)
                self.extract_pot_current(frame_number)
                self.extract_HKL(frame_number)
                return img/self.extract_transm_and_mon(frame_number)

    def load_frame(self,frame_number,flip=True):
        #if one frame one nxs file
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        #img=np.array(data.entry.instrument.detector.data.nxdata[0])
        if self.img_structure=='one':
            while frame_number < self.total_frame_number:
                img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(frame_number)
                #print(self.nexus_data_1.entry.instrument.detector.data.shape)
                #print(img)
                if img is None:
                    img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(frame_number-1)
                self.extract_motor_angles(frame_number)
                self.extract_pot_current(frame_number)
                self.extract_HKL(frame_number)
                self.frame_number = frame_number
                if flip:
                    img = np.flip(img.T,1)
                img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                        self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
                #normalized the intensity by the monitor and trams counters
                yield img/self.motor_angles['mon']/self.motor_angles['transm']
                #yield img/self.motor_angles['transm']/200000
                frame_number +=1
        elif self.img_structure == 'multiple':
            while frame_number < self.total_frame_number:
                img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,self.scan_number)
                img_name_1='{}_{:0>5}_{:0>5}.nxs'.format(self.frame_prefix,self.scan_number,frame_number)
                #print(img_name_1,img_name)
                img_path_1=os.path.join(self.nexus_path,img_name.replace(".nxs",""),'lmbd',img_name_1)
                self.nexus_data_1 = nxload(img_path_1)
                img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(0)
                #print(self.nexus_data_1.entry.instrument.detector.data.shape)
                #if img is None:
                #    img=self.nexus_data_1.entry.instrument.detector.data._get_filedata(frame_number-1)
                self.extract_motor_angles(frame_number)
                self.extract_pot_current(frame_number)
                self.extract_HKL(frame_number)
                self.frame_number = frame_number
                
                if flip:
                    img = np.flip(img.T,1)
                img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                        self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
                #normalized the intensity by the monitor and trams counters
                yield img/self.motor_angles['mon']/self.motor_angles['transm']
                #yield img/self.motor_angles['transm']/200000
                frame_number +=1

    def extract_beam_mon_ct(self,mon_path = 'scan/data/eh_c01'):
        return np.array(self.nexus_data['scan/data/eh_c01'])

    def extract_motor_angles(self, frame_number):
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        motors={}
        motor_names = ['phi', 'chi', 'delta', 'gamma', 'mu', 'omega_t','omega']
        #for motor in self.constant_motors:
        #    motors[motor] = self.constant_motors[motor]
        for motor in motor_names:
            #if motor not in motors.keys():
            try:#use those from nexus file if it is presence
                fetch_path = 'scan/data/{}'.format(motor)
                motors[motor] = np.array(self.nexus_data[fetch_path])[frame_number]
            except:#if not then use the constant motor angles
                motors[motor] = self.constant_motors[motor]
        motors['mon'] = np.array(self.nexus_data['scan/data/eh_c01'])[frame_number]
        try:
            motors['transm']=1./np.array(self.nexus_data['scan/data/atten'])[frame_number]
        except:
            #motors['transm']=np.array(self.nexus_data['scan/data/lmbd_countsroi1'])[frame_number]/np.array(self.nexus_data['scan/data/lmbd_countsroi1_atten'])[frame_number]
            motors['transm']= 1
        try:
            motors['time'] = np.array(self.nexus_data['scan/data/timestamp'])[frame_number]
        except:
            motors['time'] = sum(np.array(self.nexus_data['scan/data/eh_t01'])[0:frame_number])
        self.motor_angles = motors
        #self.motor_angles['transm'] = 1
        #self.motor_angles['mon'] =1
        return motors

    def extract_transm_and_mon(self,frame_number):
        mon = np.array(self.nexus_data['scan/data/eh_c01'])[frame_number]
        try:
            transm=1./np.array(self.nexus_data['scan/data/atten'])[frame_number]
        except:
            #motors['transm']=np.array(self.nexus_data['scan/data/lmbd_countsroi1'])[frame_number]/np.array(self.nexus_data['scan/data/lmbd_countsroi1_atten'])[frame_number]
            transm= 1
        return mon*transm

    def extract_delta_angles(self):
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        try:
            return np.array(self.nexus_data['scan/data/{}'.format('delta')])
        except:
            return [self.constant_motors['delta']]

    def update_motor_angles_in_data(self,data):
        for motor in self.motor_angles:
            data[motor].append(self.motor_angles[motor])
        return data

    def extract_pot_current(self, frame_number):
        try:
            try:
                pot = np.array(self.nexus_data['scan/data/voltage2'])[frame_number]
            except:
                pot = np.array(self.nexus_data['scan/data/pilc_voltage2'])[frame_number]
        except:
            print('Potential channel not saved in the nexus file!')
            pot = np.zeros(self.total_frame_number)[frame_number]
        try:
            try:
                cur = np.array(self.nexus_data['scan/data/voltage1'])[frame_number]
            except:
                cur = np.array(self.nexus_data['scan/data/pilc_voltage1'])[frame_number]
        except:
            print('Current channel not saved in the nexus file!')
            cur = np.zeros(self.total_frame_number)[frame_number]
        self.potential = pot
        self.current = cur
        try:
            self.potential_cal = self.potential_profile_cal[frame_number]
        except:
            self.potential_cal = pot
            print('Use real potential for the potential_cal')
        return pot, cur

    def extract_pot_profile(self):
        try:
            pot_profile = np.array(self.nexus_data['scan/data/voltage2'])
        except:
            pot_profile = np.array(self.nexus_data['scan/data/pilc_voltage2'])

        self.potential_profile = pot_profile
        self.potential_profile_cal = FitEnginePool.fit_pot_profile(list(range(len(pot_profile))),pot_profile, show_fig = False)
        return pot_profile

    def extract_HKL(self, frame_number):
        try:
            H = np.array(self.nexus_data['scan/data/diffractometer_h'])[frame_number]
            K = np.array(self.nexus_data['scan/data/diffractometer_k'])[frame_number]
            L = np.array(self.nexus_data['scan/data/diffractometer_l'])[frame_number]
            self.hkl =(H,K,L)
            #cur = np.array(self.nexus_data['scan/data/voltage1'])[frame_number]
            return H, K, L
        except:
            self.hkl=(0,0,0)
            return 0,0,0

    def load_frame_from_path(self,img_path,frame_number = 0,flip=True):
        try:
            #if one frame one nxs file
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data.nxdata[0])
        except:
            #if all frames in one nxs file
            data=nxload(img_path)
            img=np.array(data.entry.instrument.detector.data[frame_number])
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def show_frame(self,scan_number,frame_number,one_frame_in_one_nxs=True,flip=True):
        img=self.load_frame(scan_number,frame_number,one_frame_in_one_nxs,flip)
        fig,ax=pyplot.subplots()
        pyplot.imshow(img,cmap='jet')
        if flip:
            pyplot.colorbar(extend='both',orientation='vertical')
        else:
            pyplot.colorbar(extend='both',orientation='horizontal')
        pyplot.clim(0,205)
        # pyplot.show()
        return img

    def find_dead_pix(self,scan_number=666,img_end=100):
        dead_pix_container=self.load_frame(scan_number,0)==self.load_frame(scan_number,1)
        dead_pix_container=np.where(dead_pix_container==True)
        dead_pix_container=zip(tuple(dead_pix_container[0]),tuple(dead_pix_container[1]))
        img0= self.load_frame(scan_number,0)
        print(len(dead_pix_container))
        for i in range(2,img_end):
            print('Processing img_',i)
            img = self.load_frame(scan_number,i)
            temp= img != img0
            temp= np.where(temp==True)
            temp= zip(tuple(temp[0]),tuple(temp[1]))
            for each in temp:
                if each in dead_pix_container:
                    dead_pix_container.remove(each)
        return dead_pix_container

class gsecars_image_loader(object):
    def __init__(self,clip_boundary,kwarg,scan_numbers):
        # self.nexus_path=nexus_path
        # self.frame_prefix=frame_prefix
        self.spec_info = None
        self.scan_number = None
        self.frame_number = None
        self.scan_numbers = scan_numbers
        self.hkl = None
        self.clip_boundary = clip_boundary
        for key in kwarg:
            setattr(self, key, kwarg[key])
        self.sort_spec_file()
        # self.constant_motors = constant_motors
        #load nexus data only once here
        #img_name='{}_{:0>5}.nxs'.format(frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #self.nexus_data = nxload(img_path)
        #self.get_frame_number()

    def _get_col_from_file(self,lines,start_row,end_row,col,type=float):
        numbers=[]
        for i in range(start_row,end_row):
            numbers.append(type(lines[i].rstrip().rsplit()[col]))
        return numbers

    #extract info from spec file
    #def sort_spec_file(self,spec_path='.',spec_name='mica-zr_s2_longt_1.spec',scan_number=[16,17,19],\
    #               general_labels={'H':'H','K':'K','L':'L','E':'Energy'},correction_labels={'time':'Seconds','norm':'io','transmision':'trans'},\
    #                angle_labels={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'},\
    #                angle_labels_escan={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'},\
    #                G_labels={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[31]],'lambda':['G4',range(3,4)]}):
    def sort_spec_file(self):
        G_labels = self.g_labels
        data_info,col_label={},{}
        data_info['scan_type']=[]
        data_info['scan_number']=self.scan_numbers
        data_info['row_number_range']=[]

        for key in self.general_labels.keys():
            data_info[key]=[]

        for key in self.correction_labels.keys():
            data_info[key]=[]

        for key in self.angle_labels.keys():
            data_info[key]=[]

        for key in G_labels.keys():
            data_info[key]=[]

        f_spec=open(os.path.join(self.spec_path,self.spec_name))
        spec_lines=f_spec.readlines()
        scan_rows=[]
        data_rows=[]
        G0_rows=[]
        G1_rows=[]
        G3_rows=[]
        G4_rows=[]
        for i in range(len(spec_lines)):
            if spec_lines[i].startswith("#S"):
                scan_rows.append([i,int(spec_lines[i].rsplit()[1])])
            elif spec_lines[i].startswith("#L"):
                data_rows.append(i+1)
            elif spec_lines[i].startswith("#G0"):
                G0_rows.append(i)
            elif spec_lines[i].startswith("#G1"):
                G1_rows.append(i)
            elif spec_lines[i].startswith("#G3"):
                G3_rows.append(i)
            elif spec_lines[i].startswith("#G4"):
                G4_rows.append(i)

        if self.scan_numbers==None:#if None, then take all rodscan and Escan existing in the spec file
            data_info['scan_number']=[]
            for i in range(len(scan_rows)):
                scan=scan_rows[i]
                data_start=data_rows[i]
                r_index_temp,scan_number_temp=scan
                scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
                if scan_type_temp=="raxr_ascan":
                    scan_type_temp="Escan"
                if scan_type_temp=='rodscan' or scan_type_temp=='Escan':
                    j=0

                    while (not spec_lines[data_start+j].startswith("#")) and (spec_lines[data_start+j]!="\n"):
                        j+=1
                    #print (scan_number_temp,j)
                    row_number_range=[data_start,data_start+j]
                    data_info['scan_type'].append(scan_type_temp)
                    data_info['scan_number'].append(scan_number_temp)
                    data_info['row_number_range'].append(row_number_range)
                    data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                    for key in general_labels.keys():
                        try:
                            data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                        except:
                            data_info[key].append([np.NaN]*j)#there is no energy column in rodscan data

                    for key in correction_labels.keys():
                        if key=="transmision" and self.beamline=="ESRF":
                            data_info[key].append([1]*j)
                        else:
                            try:
                                data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))
                            except:
                                data_info[key].append([1]*j)
                    for key in angle_labels.keys():#we only extract rodscan and Escan data info and skip ascan
                        if scan_type_temp=='rodscan':
                            data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                        if scan_type_temp=='Escan':
                            data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))

                    for key in G_labels.keys():
                        G_type=G_labels[key][0]
                        inxes=G_labels[key][1]
                        #ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                        ff=lambda items,inxes:[float(items[i]) for i in inxes]
                        if G_type=='G0':
                            data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                        if G_type=='G1':
                            data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                        if G_type=='G3':
                            data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                        if G_type=='G4':
                            data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

                    if scan_type_temp in col_label.keys():
                        pass
                    else:
                        col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()[1:]
                else:
                    pass

        elif self.scan_numbers==[]:#if no spec number is specfied, then do nothing
            pass
        else:
            for ii in range(len(self.scan_numbers)):
                _scan=self.scan_numbers[ii]
                i=np.where(np.array(scan_rows)[:,1]==_scan)[0][0]
                scan=scan_rows[i]
                data_start=data_rows[i]
                r_index_temp,scan_number_temp=scan
                scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
                if scan_type_temp=="raxr_ascan":
                    scan_type_temp="Escan"
                j=0
                while (not spec_lines[data_start+j].startswith("#")) and (spec_lines[data_start+j]!="\n"):#continue until you hit a blank line or '#'
                    #print spec_lines[data_start+j].startswith(""),j,j+1
                    j+=1
                row_number_range=[data_start,data_start+j]
                data_info['scan_type'].append(scan_type_temp)
                #data_info['scan_number'].append(scan)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in self.general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(self.general_labels[key]),type=float))
                    except:
                        data_info[key].append([np.NaN]*j)
                        #data_info[key].append([])
                for key in self.correction_labels.keys():
                    if key=="transmision" and beamline=="ESRF":
                        data_info[key].append([1]*j)
                    else:
                        try:
                            data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(self.correction_labels[key]),type=float))
                        except:
                            data_info[key].append([1]*j)
                for key in self.angle_labels.keys():
                    if scan_type_temp=='rodscan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(self.angle_labels[key]),type=float))
                    if scan_type_temp=='Escan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(self.angle_labels_escan[key]),type=float))
                for key in G_labels.keys():
                    G_type=G_labels[key][0]
                    inxes=G_labels[key][1]
                    #ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                    ff=lambda items,inxes:[float(items[i]) for i in inxes]
                    if G_type=='G0':
                        data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G1':
                        data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G3':
                        data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G4':
                        data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

                #data_info['scan_type'].append(scan_type_temp)
                #data_info['scan_number'].append(_scan)
                #data_info['row_number_range'].append(row_number_range)
                if scan_type_temp in col_label.keys():
                    pass
                else:
                    col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()

        data_info['col_label']=col_label
        #print data_info['scan_number']
        f_spec.close()
        self.spec_info = data_info
        return None

    def update_scan_info(self,scan_number):
        if scan_number not in self.scan_numbers:
            return

        self.scan_number = scan_number
        print('\nRunning scan {} now...'.format(scan_number))
        self.total_frame_number = len(self.spec_info['H'][self.scan_numbers.index(scan_number)])

    def load_one_frame(self,frame_number,flip = True):
        #if one frame one nxs file
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        #img=np.array(data.entry.instrument.detector.data.nxdata[0])
        self.frame_number = frame_number
        prefix_image = self.spec_name[0:self.spec_name.index('.')]
        scan_folder = 'S{:0>3}'.format(self.scan_number)
        img_name = '{}_S{:0>3}_{:0>5}.{}'.format(prefix_image,self.scan_number,frame_number,self.img_extention)
        img_path = os.path.join(self.spec_path,'images',prefix_image,scan_folder,img_name)
        img=misc.imread(img_path)
        if flip:
            # img = np.flip(img.T,1)
            img = img.T
        img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
        #normalized the intensity by the monitor and trams counters
        self.extract_motor_angles(frame_number)
        self.extract_HKL(frame_number)
        return img/self.extract_transm_and_mon(frame_number)

    def load_frame(self,frame_number,flip=True):
        #if one frame one nxs file
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        #img=np.array(data.entry.instrument.detector.data.nxdata[0])
        while frame_number<self.total_frame_number:
            self.frame_number = frame_number
            prefix_image = self.spec_name[0:self.spec_name.index('.')]
            scan_folder = 'S{:0>3}'.format(self.scan_number)
            img_name = '{}_S{:0>3}_{:0>5}.{}'.format(prefix_image,self.scan_number,frame_number,self.img_extention)
            img_path = os.path.join(self.spec_path,'images',prefix_image,scan_folder,img_name)
            img=misc.imread(img_path)
            if flip:
                img = img.T
                #img = np.flip(img.T,1)
            img = img[self.clip_boundary['ver'][0]:self.clip_boundary['ver'][1],
                    self.clip_boundary['hor'][0]:self.clip_boundary['hor'][1]]
            #normalized the intensity by the monitor and trams counters
            self.extract_motor_angles(frame_number)
            # self.extract_pot_current(frame_number)
            self.extract_HKL(frame_number)
            yield img/self.extract_transm_and_mon(frame_number)
            frame_number += 1

    def extract_beam_mon_ct(self,mon_path = 'scan/data/eh_c01'):
        return np.array(self.nexus_data['scan/data/eh_c01'])

    def extract_motor_angles(self, frame_number):
        #img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
        #img_path=os.path.join(self.nexus_path,img_name)
        #data=nxload(img_path)
        motors={}
        motor_names = ['phi', 'chi', 'del', 'nu', 'mu', 'eta','norm','transmission']
        if 'E' in self.spec_info:
            motor_names.append('E')
        #for motor in self.constant_motors:
        #    motors[motor] = self.constant_motors[motor]
        scan_index = self.scan_numbers.index(self.scan_number)
        for motor in motor_names:
            motors[motor] = self.spec_info[motor][scan_index][frame_number]
        self.motor_angles = motors
        #self.motor_angles['transm'] = 1
        #self.motor_angles['mon'] =1
        return motors

    def extract_transm_and_mon(self,frame_number):
        scan_index = self.scan_numbers.index(self.scan_number)
        return self.spec_info['norm'][scan_index][frame_number]*self.spec_info['transmission'][scan_index][frame_number]

    def update_motor_angles_in_data(self,data):
        for motor in self.motor_angles:
            data[motor].append(self.motor_angles[motor])
        return data

    def extract_HKL(self, frame_number):
        scan_index = self.scan_numbers.index(self.scan_number)
        H,K,L = (self.spec_info['H'][scan_index][frame_number], self.spec_info['K'][scan_index][frame_number], self.spec_info['L'][scan_index][frame_number])
        self.hkl = (H,K,L)
        return H, K, L

    def load_frame_from_path(self,img_path,flip=True):
        img=misc.imread(img_path)
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def show_frame(self,frame_number,flip=True):
        img=self.load_frame(frame_number,flip)
        fig,ax=pyplot.subplots()
        pyplot.imshow(img,cmap='jet')
        if flip:
            pyplot.colorbar(extend='both',orientation='vertical')
        else:
            pyplot.colorbar(extend='both',orientation='horizontal')
        pyplot.clim(0,205)
        # pyplot.show()
        return img

class DetImage:
    def __init__(self, img, motors, counters, header=None):
        self.img = img
        self.motors = motors
        self.counters = counters
        self.header = header

#extract info from spec file
class pyspec(object):
    def __init__(self, spec_name = "/home/qiu/data/CH5314/ch5314_sixcvertical.spec"):
        self.spec_file = spec_name
        self.data= {}
        self.headers = {}
        self.comments = {}
        self.data_label = {}
        self.extract_info()

    def extract_info(self, ):
        with open(self.spec_file) as f_spec:
            spec_lines=f_spec.readlines()
            current_scan = " "
            for i in range(len(spec_lines)):
                if spec_lines[i].startswith("#S"):
                    current_scan = int(spec_lines[i].rsplit()[1])
                    self.data[current_scan] = []
                    self.headers[current_scan] = [spec_lines[i]]
                    self.comments[current_scan] = []
                elif spec_lines[i].startswith("#C"):
                    if current_scan!=" ":
                        self.comments[current_scan].append(spec_lines[i])
                elif spec_lines[i].startswith("#L"):
                    if current_scan!=" ":
                        self.data_label[current_scan] = spec_lines[i].rstrip().rsplit()[1:]
                elif spec_lines[i].startswith("#"):
                    if current_scan!=" ":
                        self.headers[current_scan].append(spec_lines[i])
                elif spec_lines[i] == "\n":
                    pass
                else:
                    if current_scan!=" ":
                        try:
                            self.data[current_scan].append([float(k) for k in spec_lines[i].rstrip().rsplit()])
                        except:
                            print('Check scan_{} in the spec file'.format(current_scan))
'''
    Loads the images from a spec scan.
'''
class edf_image_loader_old:
    def __init__(self, spec_filename, image_foldername,is_zap_scan):
        self.spec_file = spec.SpecDataFile(spec_filename)
        self.scan_selector = specfilewrapper.Specfile(spec_filename)
        self.spec_filename = spec_filename
        self.image_foldername = image_foldername
        self.total_frame_number = None
        self.is_zap_scan = is_zap_scan

    def get_frame_number(self, scan_no):
        return self.spec_file[scan_no].data.shape[0]

    def get_trans_mon_factor(self,scan,frame_no):
        mon, trans = None, None
        if self.is_zap_scan:
            mon = scan.datacol('zap_mon')[frame_no]
            trans = scan.datacol('zap_transm')[frame_no]
        else:
            counter_prefix = ''
            if 'ccoscan' in scan.header('S')[0]:
                counter_prefix = 'zap_'
            mon = scan.datacol('%smon'%(counter_prefix))[frame_no]
            trans = scan.datacol('%stransm'%(counter_prefix))[frame_no]
        self.transm = trans
        self.mon = mon
        return mon*trans

    def load_frame(self, scan_no, frame_no, gz_compressed=True, normalize=False, monitor_name=None, monitor_names=None, remove_rows=None, remove_cols=None):
        self.total_frame_number = self.get_frame_number(scan_no)
        mon_trans_factor = self.get_trans_mon_factor(self.scan_selector.select('{}.1'.format(scan_no)),frame_no)
        header = dict()
        for line in self.spec_file[scan_no].header.split('\n'):
            header[line.split(' ')[0]] = line[len(line.split(' ')[0]):]
        comments = dict()
        for line in self.spec_file[scan_no].comments.split('\n'):
            comments[line.split(':')[0].strip()] = line[len(line.split(':')[0])+1:].strip()
        frames = self.total_frame_number

        if('ccoscan' in header['#S'] or 'zapline' in header['#S']):
            img_folder = comments['#C DIRECTORY'].split('/')[-1]+'/'
            if(img_folder == '/'):
                img_folder = comments['#C DIRECTORY'].split('/')[-2]+'/'
            zap_scan_no = int(comments['#C ZAP SCAN NUMBER'])
            radix = comments['#C RADIX']
            filename = radix + '_mpx-x4_%s_0000_0000.edf'%(str(zap_scan_no).zfill(4))
            multiframe_edf_frame_no = frame_no
        else:
            first_frame = int(header['#UCCD'].split('#r')[-1].split('.')[0])
            img_folder = header['#UCCD'].replace('//','/').split('/')[-2]+'/' #in the beginning the path of MA3886 had '//' for some reason
            filename_template = header['#UCCD'].split('/')[-1].split('#r')[0] + '#r.' + header['#UCCD'].split('/')[-1].split('.')[-1]
            filename = filename_template.replace('#n', str(scan_no).zfill(3)).replace('#p', str(frame_no).zfill(3)).replace('#r', str(first_frame+frame_no).zfill(3))
            multiframe_edf_frame_no = 0

        # automatically detect if frame is compressed or not
        if(not os.path.exists(os.path.join(self.image_foldername, img_folder, filename))):
            filename = filename.replace('.edf', '.edf.gz')

        #if(gz_compressed):
        #    filename = filename.replace('.edf', '.edf.gz')

        if(frame_no >= frames):
            raise IndexError("Frame number does not exist.")

        img_filename = os.path.join(self.image_foldername, img_folder, filename)
        edf = EdfFile.EdfFile(img_filename, 'r')


        edf_header = edf.GetHeader(multiframe_edf_frame_no)
        motors = dict()
        motor_mne = edf_header['motor_mne'].split()
        motor_pos = edf_header['motor_pos'].split()
        for i in range(len(motor_mne)):
            motors[motor_mne[i]] = float(motor_pos[i])
        counters = dict()
        if(not ('ccoscan' in header['#S'] or 'zapline' in header['#S'])):
            counter_mne = edf_header['counter_mne'].split()
            counter_pos = edf_header['counter_pos'].split()
            for i in range(len(counter_mne)):
                counters[counter_mne[i]] = float(counter_pos[i])

        the_img = np.array(edf.GetData(multiframe_edf_frame_no), dtype='float')
        if(remove_rows != None):
            the_img = np.delete(the_img, remove_rows, axis=0)
        if(remove_cols != None):
            the_img = np.delete(the_img, remove_cols, axis=1)

        if(normalize):
            if(monitor_name):
                mon_count = float(getattr(self.spec_file[scan_no], monitor_name)[frame_no])
                the_img /= mon_count
            if(monitor_names):
                for mon_name in monitor_names:
                    mon_count = float(getattr(self.spec_file[scan_no], mon_name)[frame_no])
                    the_img /= mon_count

        return DetImage(the_img/mon_trans_factor, motors, counters, header)

    def load_all_frames(self, scan_no, gz_compressed=True, normalize=False, monitor_name=None, remove_rows=None, remove_cols=None):
        frame_no = self.get_no_frames(scan_no)
        frames = np.zeros(frame_no)
        for i in range(frame_no):
            frames[i] = self.load_frame(scan_no, i, gz_compressed, normalize, monitor_name, remove_rows, remove_cols)
        return frame

class tiff_image_loader:
    def __init__(self, spec_filename, image_foldername):
        self.scan_selector = specfilewrapper.Specfile(spec_filename)
        self.spec_filename = spec_filename
        self.image_foldername = image_foldername
        self.total_frame_number = None
        # print(spec_filename)

    def get_frame_number(self, scan_no):
        return self.scan_selector.select('{}.1'.format(scan_no)).data().shape[1]

    def get_trans_mon_factor(self,scan,frame_no):
        mon, trans = None, None

        mon = scan.datacol('io')[frame_no]
        trans = scan.datacol('transm')[frame_no]
        # self.transm = trans
        self.transm = 1
        self.mon = mon
        return mon*trans

    def load_frame(self, scan_no, frame_no):
        self.total_frame_number = self.get_frame_number(scan_no)
        mon_trans_factor = self.get_trans_mon_factor(self.scan_selector.select('{}.1'.format(scan_no)),frame_no)
        scan_folder = 'S{:0>3}'.format(scan_no)
        img_name = '{}_{}_{:0>5}.tif'.format(os.path.basename(self.image_foldername),\
                                         scan_folder, frame_no)
        img_path = os.path.join(self.image_foldername, scan_folder, img_name)
        img=misc.imread(img_path)
        return img/mon_trans_factor

class edf_image_loader:
    def __init__(self, spec_filename, image_foldername,is_zap_scan):
        self.spec_file = pyspec(spec_filename)
        self.scan_selector = specfilewrapper.Specfile(spec_filename)
        self.spec_filename = spec_filename
        self.image_foldername = image_foldername
        self.total_frame_number = None
        self.is_zap_scan = is_zap_scan

    def get_frame_number(self, scan_no):
        return np.array(self.spec_file.data[scan_no]).shape[0]

    def get_trans_mon_factor(self,scan,frame_no):
        mon, trans = None, None
        if self.is_zap_scan:
            mon = scan.datacol('zap_mon')[frame_no]
            trans = scan.datacol('zap_transm')[frame_no]
        else:
            counter_prefix = ''
            if 'ccoscan' in scan.header('S')[0]:
                counter_prefix = 'zap_'
            try:
                mon = scan.datacol('%smon'%(counter_prefix))[frame_no]
            except:
                # mon = scan.datacol('%sattn'%(counter_prefix))[frame_no]
                mon = 1
            trans = scan.datacol('%stransm'%(counter_prefix))[frame_no]
        self.transm = trans
        self.mon = mon
        return mon*trans

    def load_frame(self, scan_no, frame_no, gz_compressed=False, normalize=False, monitor_name=None, monitor_names=None, remove_rows=None, remove_cols=None):
        self.total_frame_number = self.get_frame_number(scan_no)
        mon_trans_factor = self.get_trans_mon_factor(self.scan_selector.select('{}.1'.format(scan_no)),frame_no)
        header = dict()
        for line in self.spec_file.headers[scan_no]:
            # print(self.spec_file.headers[scan_no])
            # print(line.split()[0])
            header[line.split()[0]] = line[len(line.split()[0]):]
        comments = dict()
        for line in self.spec_file.comments[scan_no]:
            comments[line.split(':')[0].strip()] = line[len(line.split(':')[0])+1:].strip()
        frames = self.total_frame_number

        if('ccoscan' in header['#S'] or 'zapline' in header['#S']):
            img_folder = comments['#C DIRECTORY'].split('/')[-1]
            if(img_folder in ['','/']):
                img_folder = comments['#C DIRECTORY'].split('/')[-2]
            zap_scan_no = int(comments['#C ZAP SCAN NUMBER'])
            radix = comments['#C RADIX']
            filename = radix + '_mpx-x4_%s_0000_0000.edf'%(str(zap_scan_no).zfill(4))
            multiframe_edf_frame_no = frame_no
        else:
            first_frame = int(header['#UCCD'].split('#r')[-1].split('.')[0])
            img_folder = header['#UCCD'].replace('//','/').split('/')[-2]+'/' #in the beginning the path of MA3886 had '//' for some reason
            filename_template = header['#UCCD'].split('/')[-1].split('#r')[0] + '#r.' + header['#UCCD'].split('/')[-1].split('.')[-1]
            filename = filename_template.replace('#n', str(scan_no).zfill(3)).replace('#p', str(frame_no).zfill(3)).replace('#r', str(first_frame+frame_no).zfill(3))
            multiframe_edf_frame_no = 0

        # automatically detect if frame is compressed or not
        # if(not os.path.exists(os.path.join(self.image_foldername, img_folder, filename))):
            # filename = filename.replace('.edf', '.edf.gz')

        #if(gz_compressed):
        #    filename = filename.replace('.edf', '.edf.gz')

        if(frame_no >= frames):
            raise IndexError("Frame number does not exist.")

        img_filename = os.path.join(self.image_foldername, img_folder, filename.rstrip())
        edf = EdfFile.EdfFile(img_filename, 'r')
        # print(self.image_foldername,img_folder,filename)
        # edf = EdfFile.EdfFile("/Users/cqiu/app/ma4171/ma4171_img/ma4171_mpx02/ma4171_mpx_247_000_1977.edf", 'r')


        edf_header = edf.GetHeader(multiframe_edf_frame_no)
        motors = dict()
        motor_mne = edf_header['motor_mne'].split()
        motor_pos = edf_header['motor_pos'].split()
        for i in range(len(motor_mne)):
            motors[motor_mne[i]] = float(motor_pos[i])
        counters = dict()
        if(not ('ccoscan' in header['#S'] or 'zapline' in header['#S'])):
            counter_mne = edf_header['counter_mne'].split()
            counter_pos = edf_header['counter_pos'].split()
            for i in range(len(counter_mne)):
                counters[counter_mne[i]] = float(counter_pos[i])

        the_img = np.array(edf.GetData(multiframe_edf_frame_no), dtype='float')
        if(remove_rows != None):
            the_img = np.delete(the_img, remove_rows, axis=0)
        if(remove_cols != None):
            the_img = np.delete(the_img, remove_cols, axis=1)

        if(normalize):
            if(monitor_name):
                mon_count = self.spec_file.data[scan_no][frame_no][self.spec_file.data_label.index(monitor_name)]
                the_img /= mon_count
            if(monitor_names):
                for mon_name in monitor_names:
                    mon_count = self.spec_file.data[scan_no][frame_no][self.spec_file.data_label.index(mon_name)]
                    the_img /= mon_count

        return DetImage(the_img/mon_trans_factor, motors, counters, header)

    def load_all_frames(self, scan_no, gz_compressed=True, normalize=False, monitor_name=None, remove_rows=None, remove_cols=None):
        frame_no = self.get_no_frames(scan_no)
        frames = np.zeros(frame_no)
        for i in range(frame_no):
            frames[i] = self.load_frame(scan_no, i, gz_compressed, normalize, monitor_name, remove_rows, remove_cols)
        return frame

