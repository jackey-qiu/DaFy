import numpy as np
import sys, os, locate_path
script_path = locate_path.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(os.path.join(DaFy_path,'util', 'XRD_tools'))
sys.path.append(DaFy_path)
import reciprocal_space_v3 as rsp
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import sys,os
import configparser
from scipy.ndimage import gaussian_filter
from pylab import MultipleLocator, LogLocator, FormatStrFormatter
from util.PlotSetup import *
from scipy import stats
from collections import namedtuple

which_scans_to_plot = [221,229,231,236,243,244]
#which_scans_to_plot = [724,732,805,807]
config_file_name = 'P23_I20180835_plot.ini'
#config_file_name = 'CV_XRD_plot_I20180114_final.ini'
config_file = os.path.join(DaFy_path, 'config', config_file_name)
#use default data path?
data_path_default = os.path.join(DaFy_path,'data')
use_default_data_path = True

#use default ids file path?
ids_file_path_default = os.path.join(DaFy_path,'data','ids')
use_default_ids_path = True

#do you want to set the max to 0
ref_max_eq_0 = {'strain':1,'size':1,'intensity':1}

#ir drop correction(check the func of ir_drop_analysis in PlotSetup.py
IR = {'DaFy_231':0., 'DaFy_243':0.,'DaFy_221':0.,'DaFy_229':0.}
cv_scale_factor = 25#scaling factor to double layer region, change accordingly
cv_spike_cut = 0.07#smallest spike you want to filter out from CV profile, do this in a trail_and_error way

plot_intensity = True

#select_cycle
which_select_cycle = 'new'

#specify this for pot step scan
scan_time = 100 #in seconds

#do you want to bin your datapoints
#debug is required to set bin_level>1
bin_level = 1

#specify current density limit, other limits are set automatically
ylim_current_density = [-2., 4.2]
#crystal reciprocal lattice instance

crystals = ['Co3O4','CoHO2_9009884']
HKL_normals =[[1,1,1],[0,0,1]]
HKL_para_xs= [[1,1,-2],[1,0,0]]
offset_angles = [0,0]
for each in crystals:
    i = crystals.index(each)
    globals()[each] = rsp.lattice.from_cif(os.path.join(DaFy_path, 'util','cif',"{}.cif".format(each)), HKL_normal=HKL_normals[i],HKL_para_x=HKL_para_xs[i], offset_angle =offset_angles[i])


#################you seldom need to touch the following code lines###############
#extract info from config file
config = configparser.ConfigParser()
config.read(config_file)
global_vals = ['phs', 'scan_ids', 'scan_labels', 'ids_files', 'data_files', 'hkls', 'scan_direction_ranges','colors', 'xtal_lattices', 'plot_pot_steps']
for each in global_vals:
    globals()[each] = []

for section in config.sections():
    if section == 'beamtime':
        beamtime =  eval(config.get(section,'beamtime'))
    else:
        scan_number_temp = eval(config.get(section,'scan_number'))
        for each_scan in which_scans_to_plot:
            if each_scan in scan_number_temp:
                which_one = scan_number_temp.index(each_scan)
                for each_global_val in global_vals:
                    if each_global_val == 'ids_files':
                        if use_default_ids_path:
                            globals()['ids_files'].append(os.path.join(ids_file_path_default,\
                                                                       eval(config.get(section,'ids_files'))[which_one]))
                        else:
                            globals()['ids_files'].append(os.path.join(eval(config.get(section,'ids_file_header')),\
                                                                       eval(config.get(section,'ids_files'))[which_one]))
                    elif each_global_val == 'data_files':
                        if use_default_data_path:
                            globals()['data_files'].append(os.path.join(data_path_default,\
                                                                    eval(config.get(section,'data_files'))[which_one]))
                        else:
                            globals()['data_files'].append(os.path.join(eval(config.get(section,'data_file_header')),\
                                                                    eval(config.get(section,'data_files'))[which_one]))
                    elif each_global_val == 'xtal_lattices':
                        globals()['xtal_lattices'].append(globals()[eval(config.get(section,'xtal_lattices'))[which_one]])
                    else:
                        globals()[each_global_val].append(eval(config.get(section,each_global_val))[which_one])
scan_ids_reordered = ["Scan_{}".format(each) for each in which_scans_to_plot]
index_reordered =[scan_ids.index(each) for each in scan_ids_reordered]
for each_global in global_vals:
    globals()[each_global]=[globals()[each_global][i] for i in index_reordered]

#matplotlib global setting
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
plt.rcParams.update({'axes.labelsize': 10})
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['mathtext.default']='regular'
plt.style.use('ggplot')
print(os.path.dirname(os.path.abspath(__file__)))
overplot = 1
plot_vs_RHE = 1
create_ASCII = 0
plot_pot_step = plot_pot_steps[0]
if plot_pot_step:
    xlim = [0, scan_time+10]

scan_info = ScanInfoContainer()
for i in range(len(phs)):
    scan_info.add(scan_ids[i], np.load(data_files[i]), xtal_lattices[i], hkls[i], scan_direction_ranges[i], colors[i], \
                  scan_label=scan_labels[i],pH = phs[i],ids_filename =ids_files[i])
#####################################################################################################################################

x_label = ['E / V$_{RHE}$','Time(s)'][bool(plot_pot_step)]
y_lable_key = ['ip_strain','oop_strain','ip_sigma','oop_sigma','intensity','ip_size','oop_size','opt_ref','current_den']
ylabels = [[r'$\varepsilon_\parallel$  (%)',r'$\Delta\varepsilon_\parallel$  (%)'][ref_max_eq_0['strain']],\
           [r'$\varepsilon_\perp$  (%)',r'$\Delta\varepsilon_\perp$  (%)'][ref_max_eq_0['strain']],\
           r'FWHM$_\parallel$ / $\AA^{-1}$',\
           r'FWHM$_\perp$ / $\AA^{-1}$',\
           r'Intensity / a.u.',\
           [r'$ d_\parallel$ / nm',r'$\Delta d_\parallel$ / nm'][ref_max_eq_0['size']],\
           [r'$ d_\perp$ / nm',r'$\Delta d_\perp$ / nm'][ref_max_eq_0['size']],\
           r'Optical Reflectivity / %',\
           r'j / mAcm$^{-2}$']
y_labels_lib = dict(zip(y_lable_key,ylabels))

fig_ax_container = {'ip_strain':['fig1','ax1a'],\
                    'oop_strain':['fig2','ax2a'],\
                    'ip_sigma': ['fig3','ax3a'],\
                    'oop_sigma':['fig4','ax4a'],\
                    'intensity':['fig5','ax5a'],\
                    'ip_size':['fig6','ax6a'],\
                    'oop_size':['fig7','ax7a'],\
                    'opt_ref':['fig8','ax8a'],\
                    'current_den':['fig9','ax9a'],\
                    'strain_all':['fig_ALL_strain','ax_fig_ALL_strain_ip','ax_fig_ALL_strain_oop'],\
                    'size_all':['fig_ALL_size','ax_fig_ALL_size_ip','ax_fig_ALL_size_oop'],\
                    'all_in_one':['fig_all','ax1_all','ax2_all','ax3_all','ax4_all','ax5_all','ax6_all'][0:6+int(plot_intensity)]}

#use this the setup globally the limits
ax_can_ip_strain, ip_strain_min, ip_strain_max = [],1e10, -1e10
ax_can_oop_strain, oop_strain_min, oop_strain_max = [],1e10, -1e10
ax_can_ip_size, ip_size_min, ip_size_max = [],1e10, -1e10
ax_can_oop_size, oop_size_min, oop_size_max = [],1e10, -1e10
ax_can_current, current_min, current_max = [],1e10, -1e10
ax_can_intensity, intensity_min, intensity_max = [],1e10, -1e10
ax_can_set_yticks_ip_strain,ax_can_set_yticks_current, ax_can_set_yticks_oop_strain,  ax_can_set_yticks_ip_size,  ax_can_set_yticks_oop_size, ax_can_set_yticks_intensity  = [], [], [], [], [], []

#containers
size_container={}
strain_container={}

#build figures
for each in fig_ax_container:
    if each in ['strain_all','size_all']:
        fig_ax_container[each][0] = plt.figure(figsize=(3*len(scan_ids),5))
        fig_ax_container[each][0].subplots_adjust(wspace=0.02,hspace=0.02)
    elif each == 'all_in_one':
        if len(which_scans_to_plot)>1:
            fig_ax_container[each][0] = plt.figure(figsize=(3*len(scan_ids),12))
            fig_ax_container[each][0].subplots_adjust(wspace=0.02,hspace=0.02)
        else:
            fig_ax_container[each][0] = plt.figure(figsize=(9,5))
    else:
        fig_ax_container[each][0] = plt.figure(figsize=(6,6))
    #fig_ax_container[each][0].tight_layout()

#loop through the datasets
num_datasets = len(scan_ids)

for scan_id in scan_ids:
    scan_no = int(scan_id.split('_')[-1])
    size_container[scan_no], strain_container[scan_no] = {},{}
    #get high resolution cv data from ids file
    #cv_data = extract_ids_file(scan_info[scan_id].ids_filename)
    #data file saved from DaFy program
    #R = IR[scan_id]
    R = 0
    try:#different format if the data is manually filtered afterwards
        data = scan_info[scan_id].data.f.data.tolist()
        data = namedtuple('Struct',data.keys())(*data.values())
    except:
        data = scan_info[scan_id].data.f
    # print(list(data.keys))
    pot, current_density = data.potential, data.current_density
    if which_select_cycle == 'new':
        mask = data.mask_cv_xrd
    pot = pot-abs(current_density)*R
    try:
        pot_cal = data.potential_cal
    except:
        pot_cal = pot
    try:
        frame_number = data.frame_number
    except:
        frame_number = range(len(pot))
    # cv_data = pot, current_density
    if which_select_cycle == 'old':
        #cv_data = pot, current_density
        cv_data = list(extract_ids_file(scan_info[scan_id].ids_filename,1))
    else:
        cv_data = list(extract_cv_data(scan_info[scan_id].ids_filename,1))
    cv_data[0] = cv_data[0]-abs(cv_data[1])*R
    Time = data.Time
    if len(Time)==0:
        Time = np.array(range(len(pot)))*float(scan_time/len(pot))
    # print(Time)
    cen_ip, cen_oop = data.cen_ip, data.cen_oop
    strain_ip, sigma_strain_ip = strain_ip_with_uncertainty(cen_ip, \
                                                            scan_info[scan_id].HKL_position,\
                                                            scan_info[scan_id].structure_lattice, 0)

    strain_oop, sigma_strain_oop = strain_oop_with_uncertainty(cen_oop, \
                                                            scan_info[scan_id].HKL_position,\
                                                            scan_info[scan_id].structure_lattice, 0)
    #if scan_id == "DaFy_216":
    #    strain_oop=list(np.array(strain_oop)+0.91)
    FWHM_ip, FWHM_oop = np.array(data.FWHM_ip), np.array(data.FWHM_oop)
    amp_oop, amp_ip = data.amp_oop, data.amp_ip
    pcov_ip, pcov_oop = data.pcov_ip, data.pcov_oop
    try:
        intensity = data.peak_intensity
        if sum(intensity)==0:
            intensity = amp_ip* FWHM_oop
    except:
        intensity = amp_ip*amp_oop
    # intensity = data.peak_intensity
    I = amp_ip * FWHM_oop * FWHM_ip
    # if plot_pot_step:
        # scan_direction_ranges = []
    #in new select mode, the return cycle will be calculated based on mask value
    return_cycle = 0
    bin_mode = 'select'
    if which_select_cycle=='old':
        if plot_pot_step:
            scan_direction_ranges, _, _ = select_cycle((pot_cal,pot_cal),plot_mode = 'pot_step')
        else:
            scan_direction_ranges, _, _ = select_cycle((pot_cal,pot_cal),plot_mode = 'CV')
    else:
        scan_direction_ranges, _, _ = select_cycle_new2((pot_cal,pot_cal,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
    print(scan_direction_ranges)
    #first point couple from the positive sweep to make the plot a complete circle
    current_axs = []
    point_gap_ip_strain = []
    point_gap_oop_strain = []
    point_gap_ip_size = []
    point_gap_oop_size = []

    # print(scan_direction_ranges)

    x_size =[]
    y_size_hor =[]
    y_size_ver =[]

    x_strain =[]
    y_strain_hor =[]
    y_strain_ver =[]

    for pos in range(len(scan_direction_ranges)):
        if pos == len(scan_direction_ranges) -1:
            break
        elif pos == 1 and plot_pot_step:
            break
            # indx1, indx2 = scan_direction_ranges[pos],scan_direction_ranges[pos]*2
        else:
            indx1, indx2 = scan_direction_ranges[pos:pos+2]

            indx1 += 1
        fillcolor = 'w' if pos%2 ==1 else scan_info[scan_id].color
        label = scan_info[scan_id].scan_label+[' negative scan',' positive_scan'][int(pos==0)]
        marker = 'o'
        if which_select_cycle=='old':
            plot_mode =['CV','pot_step'][int(plot_pot_step)]
            if plot_pot_step:
                marker = ''
            #strain data
            _, pot_temp, Time_temp =  select_cycle((pot_cal,Time),plot_mode=plot_mode)
            _, pot_temp, strain_ip_temp =  select_cycle((pot_cal,strain_ip),plot_mode=plot_mode)
            _, pot_temp, sigma_strain_ip_temp =  select_cycle((pot_cal,sigma_strain_ip),plot_mode=plot_mode)
            _, pot_temp, strain_oop_temp =  select_cycle((pot_cal,strain_oop),plot_mode=plot_mode)
            _, pot_temp, sigma_strain_oop_temp =  select_cycle((pot_cal,sigma_strain_oop),plot_mode=plot_mode)

            #grain size
            _, pot_temp, FWHM_ip_temp =  select_cycle((pot_cal,FWHM_ip),plot_mode=plot_mode)
            _, pot_temp, FWHM_oop_temp =  select_cycle((pot_cal,FWHM_oop),plot_mode=plot_mode)
            size_ip_temp, size_oop_temp = 0.2*np.pi/np.array(FWHM_ip_temp), 0.2*np.pi/np.array(FWHM_oop_temp)
            #intensity
            _, pot_temp, intensity_temp =  select_cycle((pot_cal,intensity),plot_mode=plot_mode)
            #x: either potential or time
            x = POT(pot_temp,plot_vs_RHE, scan_info[scan_id].pH) if not plot_pot_step else range(len(Time_temp))
            # print(x)
        else:
            #strain data
            _, pot_temp, Time_temp =  select_cycle_new2((pot_cal,Time,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            _, pot_temp, strain_ip_temp =  select_cycle_new2((pot_cal,strain_ip,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            _, pot_temp, sigma_strain_ip_temp =  select_cycle_new2((pot_cal,sigma_strain_ip,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            _, pot_temp, strain_oop_temp =  select_cycle_new2((pot_cal,strain_oop,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            _, pot_temp, sigma_strain_oop_temp =  select_cycle_new2((pot_cal,sigma_strain_oop,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)

            #grain size
            _, pot_temp, FWHM_ip_temp =  select_cycle_new2((pot_cal,FWHM_ip,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            _, pot_temp, FWHM_oop_temp =  select_cycle_new2((pot_cal,FWHM_oop,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            size_ip_temp, size_oop_temp = 0.2*np.pi/np.array(FWHM_ip_temp), 0.2*np.pi/np.array(FWHM_oop_temp)
            #intensity
            _, pot_temp, intensity_temp =  select_cycle_new2((pot_cal,intensity,frame_number,mask),bin_mode=bin_mode,return_cycle=return_cycle)
            #x: either potential or time
            x = POT(pot_temp,plot_vs_RHE, scan_info[scan_id].pH) if not plot_pot_step else range(len(Time_temp))
            # print(x)
        #init the axis handle
        if type(fig_ax_container['ip_strain'][1])==str:
            fig_ax_container['ip_strain'][1] = fig_ax_container['ip_strain'][0].add_subplot(111)
        if type(fig_ax_container['oop_strain'][1])==str:
            fig_ax_container['oop_strain'][1] = fig_ax_container['oop_strain'][0].add_subplot(111)
        if type(fig_ax_container['ip_size'][1])==str:
            fig_ax_container['ip_size'][1] = fig_ax_container['ip_size'][0].add_subplot(111)
        if type(fig_ax_container['oop_size'][1])==str:
            fig_ax_container['oop_size'][1] = fig_ax_container['oop_size'][0].add_subplot(111)
        if type(fig_ax_container['intensity'][1])==str:
            fig_ax_container['intensity'][1] = fig_ax_container['intensity'][0].add_subplot(111)
        if type(fig_ax_container['current_den'][1])==str:
            fig_ax_container['current_den'][1] = fig_ax_container['current_den'][0].add_subplot(111)
        if num_datasets == 1:# the only dataset occupy the whole figure
            for i in range(1,7):
                fig_ax_container['all_in_one'][i] = fig_ax_container['all_in_one'][0].add_subplot(2,3,i)
        else:#each dataset occupy one column
            for i in range(5+int(plot_intensity)):
                fig_ax_container['all_in_one'][i+1] = fig_ax_container['all_in_one'][0].add_subplot(5+int(plot_intensity),num_datasets, scan_ids.index(scan_id)+1+i*num_datasets)
            for i in range(2):
                fig_ax_container['strain_all'][i+1] = fig_ax_container['strain_all'][0].add_subplot(2,num_datasets, scan_ids.index(scan_id)+1+i*num_datasets)
            for i in range(2):
                fig_ax_container['size_all'][i+1] = fig_ax_container['size_all'][0].add_subplot(2,num_datasets, scan_ids.index(scan_id)+1+i*num_datasets)

        ip_strain_axs = [fig_ax_container['ip_strain'][1]]
        oop_strain_axs = [fig_ax_container['oop_strain'][1]]
        ip_size_axs = [fig_ax_container['ip_size'][1]]
        oop_size_axs = [fig_ax_container['oop_size'][1]]
        intensity_axs = [fig_ax_container['intensity'][1]]
        current_axs.append(fig_ax_container['current_den'][1])

        if num_datasets ==1:
            ip_strain_axs.append(fig_ax_container['all_in_one'][1])
            oop_strain_axs.append(fig_ax_container['all_in_one'][2])
            ip_size_axs.append(fig_ax_container['all_in_one'][4])
            oop_size_axs.append(fig_ax_container['all_in_one'][5])
            intensity_axs.append(fig_ax_container['all_in_one'][6])
            current_axs.append(fig_ax_container['all_in_one'][3])
        else:
            ip_strain_axs.append(fig_ax_container['all_in_one'][2])
            oop_strain_axs.append(fig_ax_container['all_in_one'][3])
            ip_strain_axs.append(fig_ax_container['strain_all'][1])
            oop_strain_axs.append(fig_ax_container['strain_all'][2])
            ip_size_axs.append(fig_ax_container['all_in_one'][4])
            oop_size_axs.append(fig_ax_container['all_in_one'][5])
            ip_size_axs.append(fig_ax_container['size_all'][1])
            oop_size_axs.append(fig_ax_container['size_all'][2])
            current_axs.append(fig_ax_container['all_in_one'][1])
            if plot_intensity:
                intensity_axs.append(fig_ax_container['all_in_one'][6])
        #get the first point from first pos to feed in second pos
        if pos==0:
           point_gap_ip_strain = [x[0],strain_ip_temp[0]]
           point_gap_oop_strain = [x[0],strain_oop_temp[0]]
           point_gap_ip_size = [x[0],size_ip_temp[0]]
           point_gap_oop_size = [x[0],size_oop_temp[0]]
        else:
        ##update the data in second pos
           strain_ip_temp = np.append(strain_ip_temp,point_gap_ip_strain[1])
           strain_oop_temp = np.append(strain_oop_temp,point_gap_oop_strain[1])
           size_ip_temp = np.append(size_ip_temp,point_gap_ip_size[1])
           size_oop_temp = np.append(size_oop_temp,point_gap_oop_size[1])
           x = np.append(x,point_gap_ip_strain[0])
        # print(x[indx1],x[indx2-1])

        ip_strain_data_all = [strain_ip_temp]*len(ip_strain_axs)
        oop_strain_data_all = [strain_oop_temp]*len(oop_strain_axs)
        ip_size_data_all = [size_ip_temp]*len(ip_size_axs)
        oop_size_data_all = [size_oop_temp]*len(oop_size_axs)
        # if plot_intensity:
        intensity_data_all = [intensity_temp]*len(intensity_axs)
        if 1:
            y_size_hor = y_size_hor + list(size_ip_temp[indx1:indx2])
            y_size_ver = y_size_ver + list(size_oop_temp[indx1:indx2])
            x_size = x_size +list(x[indx1:indx2])

            y_strain_hor = y_strain_hor + list(strain_ip_temp[indx1:indx2])
            y_strain_ver = y_strain_ver + list(strain_oop_temp[indx1:indx2])
            x_strain = x_strain +list(x[indx1:indx2])
        if pos==0:
            strain_container[scan_no]["ip"]= [np.mean(strain_ip_temp[0:4]),np.mean(strain_ip_temp[-5:-1])]
            strain_container[scan_no]["oop"]= [np.mean(strain_oop_temp[0:4]),np.mean(strain_oop_temp[-5:-1])]
            size_container[scan_no]["ip"]= [np.mean(size_ip_temp[0:10]),np.mean(size_ip_temp[-10:-1])]
            size_container[scan_no]["oop"]= [np.mean(size_oop_temp[0:10]),np.mean(size_oop_temp[-10:-1])]


        #check the limits for current dataset and update it if necessary
        #here reference point is 0(max value)
        def _update_min_or_not(current_min, current_max,data):
            new_min, new_max = 0, 0
            data = np.array(data)
            if current_min > (data.min() - data.max()):
                new_min = data.min() - data.max()
            else:
                new_min = current_min
            return new_min, new_max
        #here we don't set the reference point (max and min is the original values)
        def _update_min_max_or_not(current_min, current_max, data):
            new_min, new_max = 0, 0
            data = np.array(data)
            if current_min > data.min():
                new_min = data.min()
            else:
                new_min = current_min
            if current_max < data.max():
                new_max = data.max()
            else:
                new_max = current_max
            return new_min, new_max
        update_max_min = [_update_min_max_or_not,_update_min_or_not]
        ip_strain_min, ip_strain_max = update_max_min[ref_max_eq_0['strain']](ip_strain_min,ip_strain_max, strain_ip_temp)
        oop_strain_min, oop_strain_max = update_max_min[ref_max_eq_0['strain']](oop_strain_min,oop_strain_max, strain_oop_temp)
        ip_size_min, ip_size_max = update_max_min[ref_max_eq_0['size']](ip_size_min,ip_size_max, size_ip_temp)
        oop_size_min, oop_size_max = update_max_min[ref_max_eq_0['size']](oop_size_min,oop_size_max, size_oop_temp)
        intensity_min, intensity_max = update_max_min[ref_max_eq_0['intensity']](intensity_min,intensity_max, intensity_temp)
        #now collect all axs for different data (exclude the first ax which is the single plot)
        ax_can_ip_strain = ax_can_ip_strain + ip_strain_axs[1:]
        ax_can_oop_strain = ax_can_oop_strain + oop_strain_axs[1:]
        ax_can_ip_size = ax_can_ip_size + ip_size_axs[1:]
        ax_can_oop_size = ax_can_oop_size + oop_size_axs[1:]
        ax_can_intensity = ax_can_intensity + intensity_axs[1:]

        y_labels_all =[y_labels_lib['ip_strain']]* len(ip_strain_axs)+\
                      [y_labels_lib['oop_strain']]* len(oop_strain_axs)+\
                      [y_labels_lib['ip_size']]* len(ip_size_axs)+\
                      [y_labels_lib['oop_size']]* len(oop_size_axs)+\
                      [y_labels_lib['intensity']]* len(intensity_axs)

        def plot_on_ax(ax,x,data,y_label,ref_max,fit=False):
            # print(ax)
            #you need this to match the size of data other than strain and size
            if len(x)!=len(data):
                x = x[0:-1]
            ax.plot(x[indx1:indx2],set_max_to_0(data,[indx1,indx2],ref_max),linestyle = '-', linewidth =1,\
                    color = scan_info[scan_id].color, markerfacecolor = fillcolor,\
                    markeredgecolor = scan_info[scan_id].color,marker = marker, markersize=4,label = label)
            if fit:
                fit_index = np.logical_and(x[indx1:indx2]>0.2, x[indx1:indx2]<2.6)
                slope,intercept,*others =stats.linregress(x[indx1:indx2][fit_index],set_max_to_0(data,[indx1,indx2],ref_max)[fit_index])
                ax.plot(x[indx1:indx2],np.array(x[indx1:indx2])*slope+intercept,color = scan_info[scan_id].color)
                # print(slope)
            ax.set_ylabel(y_label)
            return None
        for ax, data, y_label,ref, fit in zip(ip_strain_axs+oop_strain_axs+ip_size_axs+oop_size_axs+intensity_axs,\
                            ip_strain_data_all+oop_strain_data_all+ip_size_data_all+oop_size_data_all+intensity_data_all,\
                            y_labels_all,\
                            [ref_max_eq_0['strain']]*(len(ip_strain_axs)*2)+[ref_max_eq_0['size']]*(len(ip_strain_axs)*2)+[ref_max_eq_0['intensity']]*(len(ip_strain_axs)*1),\
                            [0]*(len(ip_strain_axs)*2)+[0]*(len(ip_strain_axs)*2)+[0]*(len(ip_strain_axs)*1)):
            plot_on_ax(ax,x,data,y_label,ref,fit)
            ax.set_xlabel(x_label)

        #now plot current density
        if pos==0:
            for ax in current_axs:
                colors_lib = {0:'sienna',1:'red',2:'green',3:'blue',4:'m',5:'black'}
                #ax.plot(POT(cv_data[0], plot_vs_RHE, scan_info[scan_id].pH), cv_data[1]*1000*8*50, '-',\
                #        color=scan_info[scan_id].color,linewidth=2,label=scan_info[scan_id].scan_label)
                if plot_pot_step:
                    ax.plot(range(len(cv_data[0])), cv_data[1]*(-8), '-',\
                            color=scan_info[scan_id].color,linewidth=2,label=scan_info[scan_id].scan_label)
                else:
                    #first remove some spikes
                    filter_index =np.where(abs(np.diff(cv_data[1]*8*cv_scale_factor))<cv_spike_cut)[0]
                    filter_index = filter_index+1#index offset by 1
                    pot_filtered = cv_data[0][(filter_index,)]
                    current_filtered = cv_data[1][(filter_index,)]
                    #do this for another twice
                    for ii in range(2):
                        filter_index =np.where(abs(np.diff(current_filtered*8*cv_scale_factor))<cv_spike_cut)[0]
                        filter_index = filter_index+1
                        pot_filtered = pot_filtered[(filter_index,)]
                        current_filtered = current_filtered[(filter_index,)]
                    pot_filtered = pot_filtered - current_filtered * R
                    ax.plot(POT(pot_filtered, plot_vs_RHE, scan_info[scan_id].pH), current_filtered*(8)*cv_scale_factor, linestyle='-',marker=None,\
                            color=scan_info[scan_id].color,linewidth=1,label=scan_info[scan_id].scan_label)
                    ax.plot(POT(cv_data[0], plot_vs_RHE, scan_info[scan_id].pH), (cv_data[1]*8), ':',\
                            color=scan_info[scan_id].color,linewidth=1,label=scan_info[scan_id].scan_label)
                    ax.text(1.2,ylim_current_density[1]/2,'x{}'.format(cv_scale_factor),color=scan_info[scan_id].color)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_labels_lib['current_den'])
                ax.set_ylim(ylim_current_density)
        #now set some x-y lables to '' for commen data range, also set titles
        x_ticks, x_tick_labels = find_tick_ticklables(x, num_ticks =4, endpoint = True, dec_place =1)
        if num_datasets == 1:
            [fig_ax_container['all_in_one'][i].set_xlabel('') for i in [1,2,3]]
            [fig_ax_container['all_in_one'][i].set_xticks(x_ticks) for i in [4,5,6]]
            [fig_ax_container['all_in_one'][i].set_xticklabels(x_tick_labels) for i in [4,5,6]]
            [fig_ax_container['all_in_one'][i].set_xticklabels([]) for i in [1,2,3]]
            ax_can_set_yticks_current = [fig_ax_container['all_in_one'][3]]

        else:
            fig_ax_container['all_in_one'][1].set_title(scan_info[scan_id].scan_label,fontsize = 10)
            fig_ax_container['size_all'][1].set_title(scan_info[scan_id].scan_label,fontsize = 10)
            fig_ax_container['strain_all'][1].set_title(scan_info[scan_id].scan_label,fontsize=10)
            [fig_ax_container['all_in_one'][i].set_xlabel('') for i in [1,2,3,4,5][0:4+int(plot_intensity)]]
            [fig_ax_container['size_all'][i].set_xlabel('') for i in [1]]
            [fig_ax_container['strain_all'][i].set_xlabel('') for i in [1]]
            #no tick labels
            [fig_ax_container['all_in_one'][i].set_xticklabels([]) for i in [1,2,3,4,5][0:4+int(plot_intensity)]]
            [fig_ax_container['size_all'][i].set_xticklabels([]) for i in [1]]
            [fig_ax_container['strain_all'][i].set_xticklabels([]) for i in [1]]
            #set the same tick labels
            fig_ax_container['all_in_one'][5+int(plot_intensity)].set_xticks(x_ticks)
            fig_ax_container['all_in_one'][5+int(plot_intensity)].set_xticklabels(x_tick_labels)
            fig_ax_container['size_all'][2].set_xticks(x_ticks)
            fig_ax_container['size_all'][2].set_xticklabels(x_tick_labels)
            fig_ax_container['strain_all'][2].set_xticks(x_ticks)
            fig_ax_container['strain_all'][2].set_xticklabels(x_tick_labels)
            if scan_id != scan_ids[0]:
                [fig_ax_container['all_in_one'][i].set_ylabel('') for i in [1,2,3,4,5,6][0:5+int(plot_intensity)]]
                [fig_ax_container['size_all'][i].set_ylabel('') for i in [1,2]]
                [fig_ax_container['strain_all'][i].set_ylabel('') for i in [1,2]]
                [fig_ax_container['all_in_one'][i].set_yticklabels([]) for i in [1,2,3,4,5,6][0:5+int(plot_intensity)]]
                [fig_ax_container['size_all'][i].set_yticklabels([]) for i in [1,2]]
                [fig_ax_container['strain_all'][i].set_yticklabels([]) for i in [1,2]]
            else:
                ax_can_set_yticks_ip_strain = [fig_ax_container['all_in_one'][2], fig_ax_container['strain_all'][1]]
                ax_can_set_yticks_oop_strain = [fig_ax_container['all_in_one'][3], fig_ax_container['strain_all'][2]]
                ax_can_set_yticks_ip_size = [fig_ax_container['all_in_one'][4], fig_ax_container['size_all'][1]]
                ax_can_set_yticks_oop_size = [fig_ax_container['all_in_one'][5], fig_ax_container['size_all'][2]]
                ax_can_set_yticks_current = [fig_ax_container['all_in_one'][1]]
                if plot_intensity:
                    ax_can_set_yticks_intensity = [fig_ax_container['all_in_one'][6]]

    #save ascii files
    header = '%s #%d CV with XRD\r\nPotential / V, Current Density / mA/cm^2, Strain ip / %%, Strain oop / %%, d ip / nm, d oop / nm, Intensity (area ip)'%(beamtime, scan_no)
    X = np.array([pot, current_density, strain_ip, strain_oop, (0.2*np.pi/FWHM_ip), (0.2*np.pi/FWHM_oop), amp_ip]).T
    filename = 'data/ascii/%s_%d_CV_XRD.dat'%(beamtime, scan_no)
    np.savetxt(filename, X, newline='\r\n', header=header)

    # print(len(x_size),len(y_size_hor))
    x_size,y_size_hor,y_size_ver=np.array(x_size),np.array(y_size_hor),np.array(y_size_ver)
    fit_index = np.logical_and(x_size>1.1, x_size<1.5)
    slope_hor,intercept_hor,*others,std_hor =stats.linregress(x_size[fit_index],y_size_hor[fit_index])
    slope_ver,intercept_ver,*others, std_ver =stats.linregress(x_size[fit_index],y_size_ver[fit_index])
    print('{},size analysis results:slope_hor={}({}),slope_ver={}({})'.format(scan_id,slope_hor,std_hor,slope_ver,std_ver))

     # print(len(x_size),len(y_size_hor))
    x_strain,y_strain_hor,y_strain_ver=np.array(x_strain),np.array(y_strain_hor),np.array(y_strain_ver)
    fit_index = np.logical_and(x_strain>1.1, x_strain<1.5)
    slope_hor,intercept_hor,*others,std_hor =stats.linregress(x_strain[fit_index],y_strain_hor[fit_index])
    slope_ver,intercept_ver,*others,std_ver =stats.linregress(x_strain[fit_index],y_strain_ver[fit_index])
    print('{},strain analysis results:slope_hor={}({}),slope_ver={}({})'.format(scan_id,slope_hor,std_hor,slope_ver,std_ver))
#now let us set the limits
offset_scale = 0.1
[each.set_ylim((ip_strain_min-(ip_strain_max-ip_strain_min)*offset_scale,ip_strain_max+(ip_strain_max-ip_strain_min)*offset_scale)) for each in ax_can_ip_strain]
[each.set_ylim((oop_strain_min-(oop_strain_max-oop_strain_min)*offset_scale,oop_strain_max+(oop_strain_max-oop_strain_min)*offset_scale)) for each in ax_can_oop_strain]
[each.set_ylim((ip_size_min-(ip_size_max-ip_size_min)*offset_scale,ip_size_max+(ip_size_max-ip_size_min)*offset_scale)) for each in ax_can_ip_size]
[each.set_ylim((oop_size_min-(oop_size_max-oop_size_min)*offset_scale,oop_size_max+(oop_size_max-oop_size_min)*offset_scale)) for each in ax_can_oop_size]
[each.set_ylim((intensity_min-(intensity_max-intensity_min)*offset_scale,intensity_max+(intensity_max-intensity_min)*offset_scale)) for each in ax_can_intensity]
# [each.set_ylim((ip_strain_min-(ip_strain_max-ip_strain_min)*offset_scale,ip_strain_max+(ip_strain_max-ip_strain_min)*offset_scale)) for each in ax_can_ip_strain]
# [each.set_ylim((oop_strain_min+oop_strain_min*offset_scale,-oop_strain_min*offset_scale)) for each in ax_can_oop_strain]
# [each.set_ylim((ip_size_min+ip_size_min*offset_scale,-ip_size_min*offset_scale)) for each in ax_can_ip_size]
# [each.set_ylim((oop_size_min+oop_size_min*offset_scale,-oop_size_min*offset_scale)) for each in ax_can_oop_size]
# [each.set_ylim((intensity_min+intensity_min*offset_scale,-intensity_min*offset_scale)) for each in ax_can_intensity]
#for each in ax_can_set_yticks_current:
#    y_ticks, y_tick_labels = find_tick_ticklables(ylim_current_density, num_ticks =6, endpoint = False, dec_place =0)
#    each.set_yticks(y_ticks)
#    each.set_yticklabels(['']+y_tick_labels[1:])

for each in ax_can_set_yticks_ip_strain:
    # y_ticks, y_tick_labels = find_tick_ticklables([ip_strain_min+ip_strain_min*offset_scale,-ip_strain_min*offset_scale], num_ticks =5, endpoint = False, dec_place =3)
    y_ticks, y_tick_labels = find_tick_ticklables([ip_strain_min-(ip_strain_max-ip_strain_min)*offset_scale,ip_strain_max+(ip_strain_max-ip_strain_min)*offset_scale], num_ticks =5, endpoint = False, dec_place =3)
    each.set_yticks(y_ticks)
    each.set_yticklabels(y_tick_labels)
    # print(ip_strain_min,y_ticks,y_tick_labels)

for each in ax_can_set_yticks_oop_strain:
    y_ticks, y_tick_labels = find_tick_ticklables([oop_strain_min-(oop_strain_max-oop_strain_min)*offset_scale,oop_strain_max+(oop_strain_max-oop_strain_min)*offset_scale], num_ticks =5, endpoint = False, dec_place =3)
    each.set_yticks(y_ticks)
    each.set_yticklabels(y_tick_labels)

for each in ax_can_set_yticks_ip_size:
    y_ticks, y_tick_labels = find_tick_ticklables([ip_size_min-(ip_size_max-ip_size_min)*offset_scale,ip_size_max+(ip_size_max-ip_size_min)*offset_scale], num_ticks =5, endpoint = False, dec_place =2)
    each.set_yticks(y_ticks)
    each.set_yticklabels(y_tick_labels)

for each in ax_can_set_yticks_oop_size:
    y_ticks, y_tick_labels = find_tick_ticklables([oop_size_min-(oop_size_max-oop_size_min)*offset_scale,oop_size_max+(oop_size_max-oop_size_min)*offset_scale], num_ticks =5, endpoint = False, dec_place =2)
    each.set_yticks(y_ticks)
    each.set_yticklabels(y_tick_labels)

for each in ax_can_set_yticks_intensity:
    y_ticks, y_tick_labels = find_tick_ticklables([intensity_min-(intensity_max-intensity_min)*offset_scale,intensity_max+(intensity_max-intensity_min)*offset_scale], num_ticks =5, endpoint = False, dec_place =2)
    each.set_yticks(y_ticks)
    each.set_yticklabels(y_tick_labels)
#tight layout for figures
if len(scan_ids)==1:
    fig_ax_container['all_in_one'][0].tight_layout()
if len(which_scans_to_plot)>1:
    f=lambda x:abs(x[0]-x[1])
    f2 = lambda x,y:x*y[0]+y[1]
    scatter_plot_x = list(range(1,len(which_scans_to_plot)+1))
    #print(scatter_plot_x)
    scatter_plot_x_pair = [[x,x] for x in scatter_plot_x]
    #print(scatter_plot_x_pair)
    fig_seq = plt.figure(figsize=(8,6))
    ax_strain_seq = fig_seq.add_subplot(221)
    ax_strain_ver_seq = fig_seq.add_subplot(223)
    ax_size_seq = fig_seq.add_subplot(222)
    ax_size_ver_seq = fig_seq.add_subplot(224)
    strain_seq_ip = np.array([strain_container[scan_no]["ip"] for scan_no in which_scans_to_plot])
    d_strain_seq_ip = np.array([f(strain_container[scan_no]["ip"]) for scan_no in which_scans_to_plot])
    strain_seq_oop = np.array([strain_container[scan_no]["oop"] for scan_no in which_scans_to_plot])
    d_strain_seq_oop = np.array([f(strain_container[scan_no]["oop"]) for scan_no in which_scans_to_plot])

    size_seq_ip = np.array([size_container[scan_no]["ip"] for scan_no in which_scans_to_plot])
    d_size_seq_ip = np.array([f(size_container[scan_no]["ip"]) for scan_no in which_scans_to_plot])
    size_seq_oop = np.array([size_container[scan_no]["oop"] for scan_no in which_scans_to_plot])
    d_size_seq_oop = np.array([f(size_container[scan_no]["oop"]) for scan_no in which_scans_to_plot])

    ax_strain_seq.scatter(np.array(scatter_plot_x_pair).T,strain_seq_ip.T,color=np.array([['blue','r']]*len(which_scans_to_plot)).flatten())
    ax_strain_seq.plot(np.array(scatter_plot_x_pair),strain_seq_ip,ls='-',color='green')
    ax_strain_seq.bar(scatter_plot_x[0::2],d_strain_seq_ip[0::2]*-1,color='blue')
    ax_strain_seq.bar(scatter_plot_x[1::2],d_strain_seq_ip[1::2]*-1,color='red')
    ax_strain_seq.set_xticks(scatter_plot_x)
    ax_strain_seq.set_xticklabels([""]*len(which_scans_to_plot))
    ax_strain_seq.set_ylabel(ylabels[0])

    if np.abs(size_seq_ip).max()<np.abs(d_size_seq_ip).max()*5:
        size_seq_offset = 0
    else:
        size_seq_offset = np.abs(size_seq_ip).max() - np.abs(d_size_seq_ip).max()*2
    ax_size_seq.scatter(np.array(scatter_plot_x_pair).T,size_seq_ip.T-size_seq_offset,color=np.array([['blue','r']]*len(which_scans_to_plot)).flatten())
    ax_size_seq.plot(np.array(scatter_plot_x_pair),size_seq_ip-size_seq_offset,ls='-',color='green')
    ax_size_seq.bar(scatter_plot_x[0::2],d_size_seq_ip[0::2]*-1,color='blue')
    ax_size_seq.bar(scatter_plot_x[1::2],d_size_seq_ip[1::2]*-1,color='red')

    ax_size_seq.set_ylabel(ylabels[5]+' offset by '+str(round(size_seq_offset,1)))
    ax_size_seq.set_xticks(scatter_plot_x)
    ax_size_seq.set_xticklabels([""]*len(which_scans_to_plot))

    ax_strain_ver_seq.scatter(np.array(scatter_plot_x_pair).T,strain_seq_oop.T,color=np.array([['blue','r']]*len(which_scans_to_plot)).flatten())
    ax_strain_ver_seq.plot(np.array(scatter_plot_x_pair),strain_seq_oop,ls='-',color='green')
    ax_strain_ver_seq.bar(scatter_plot_x[0::2],d_strain_seq_oop[0::2]*-1,color='blue')
    ax_strain_ver_seq.bar(scatter_plot_x[1::2],d_strain_seq_oop[1::2]*-1,color='red')

    ax_strain_ver_seq.set_xticks(scatter_plot_x)
    ax_strain_ver_seq.set_xticklabels(phs)
    ax_strain_ver_seq.set_ylabel(ylabels[1])
    ax_strain_ver_seq.set_xlabel("pH")

    if np.abs(size_seq_oop).max()<np.abs(d_size_seq_oop).max()*5:
        size_seq_oop_offset = 0
    else:
        size_seq_oop_offset = np.abs(size_seq_oop).max() - np.abs(d_size_seq_oop).max()*2
    ax_size_ver_seq.scatter(np.array(scatter_plot_x_pair).T,size_seq_oop.T-size_seq_oop_offset,color=np.array([['blue','r']]*len(which_scans_to_plot)).flatten())
    ax_size_ver_seq.plot(np.array(scatter_plot_x_pair),size_seq_oop-size_seq_oop_offset,ls='-',color='green')
    ax_size_ver_seq.bar(scatter_plot_x[0::2],d_size_seq_oop[1::2]*-1,color='blue')
    ax_size_ver_seq.bar(scatter_plot_x[1::2],d_size_seq_oop[1::2]*-1,color='red')

    ax_size_ver_seq.set_xticks(scatter_plot_x)
    ax_size_ver_seq.set_xticklabels(phs)
    ax_size_ver_seq.set_ylabel(ylabels[6]+' offset by '+str(round(size_seq_oop_offset,1)))
    ax_size_ver_seq.set_xlabel("pH")
    fig_seq.tight_layout()

#save figures
if len(scan_ids) > 1:
    path = 'plots/many/'
else:
    path =  'plots/' + scan_id + '/'
if not os.path.exists(path):
    os.makedirs(path)
fig_ax_container['ip_strain'][0].savefig(path+'strain_ip.png', dpi=300, bbox_inches='tight')
fig_ax_container['oop_strain'][0].savefig(path+'strain_oop.png', dpi=300, bbox_inches='tight')
fig_ax_container['ip_sigma'][0].savefig(path+'FWHM_ip.png', dpi=300, bbox_inches='tight')
fig_ax_container['oop_sigma'][0].savefig(path+'FWHM_oop.png', dpi=300, bbox_inches='tight')
fig_ax_container['intensity'][0].savefig(path+'Peak_Intensity.png', dpi=300, bbox_inches='tight')
fig_ax_container['ip_size'][0].savefig(path+'Grainsize_ip.png', dpi=300, bbox_inches='tight')
fig_ax_container['oop_size'][0].savefig(path+'Grainsize_oop.png', dpi=300, bbox_inches='tight')
fig_ax_container['current_den'][0].savefig(path+'Pot_CurrentDensity.png', dpi=300, bbox_inches='tight')
fig_ax_container['all_in_one'][0].savefig(path+'Pot_all_together.png', dpi=300, bbox_inches='tight')
fig_ax_container['strain_all'][0].savefig(path+'Pot_all_strain_together.png', dpi=300, bbox_inches='tight')
fig_ax_container['size_all'][0].savefig(path+'Pot_all_size_together.png', dpi=300, bbox_inches='tight')
try:
    fig_seq.savefig(path+'com_sequence.png',dpi=300,bbox_inches='tight')
except:
    pass
plt.legend()
plt.show()

