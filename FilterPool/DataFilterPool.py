import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import pandas as pd

if (sys.version_info > (3, 0)):
    raw_input = input

data_keys=['scan_no','image_no','potential','current','peak_intensity','peak_intensity_error', \
           'pcov_ip', 'cen_ip','FWHM_ip', 'amp_ip', 'lfrac_ip','bg_slope_ip','bg_offset_ip',\
           'pcov_oop','cen_oop','FWHM_oop','amp_oop','lfrac_oop','bg_slope_oop','bg_offset_oop',\
           'H','K','L','phi','chi','mu', 'delta', 'gamma','omega_t','mon','transm',\
           'mask_cv_xrd','mask_ctr']

def strain_ip(q_ip, HKL, lattice):
    q_bulk = lattice.q(HKL)
    q_ip_bulk = np.sqrt(q_bulk[0]**2 + q_bulk[1]**2)
    return (q_ip_bulk/q_ip - 1.0)*100.

def strain_ip_with_uncertainty(q_ip, HKL, lattice, uncertainty_q_ip):
    q_bulk = lattice.q(HKL)
    q_ip_bulk = np.sqrt(q_bulk[0]**2 + q_bulk[1]**2)
    _strain_ip = (q_ip_bulk/q_ip - 1.0)*100.
    uncedrtainty_strain_ip = np.abs(q_ip_bulk/q_ip**2*100*uncertainty_q_ip)
    return (_strain_ip, uncedrtainty_strain_ip)

def strain_oop(q_oop, HKL, lattice):
    return (lattice.q(HKL)[2]/q_oop - 1.0)*100.

def strain_oop_with_uncertainty(q_oop, HKL, lattice, uncertainty_q_oop):
    q_oop_bulk = lattice.q(HKL)[2]
    _strain_oop = (q_oop_bulk/q_oop - 1.0)*100.
    uncertainty_strain_oop = np.abs(q_oop_bulk/q_oop**2*100*uncertainty_q_oop)
    return  (_strain_oop, uncertainty_strain_oop)

    make_data_config_file(data_pd = df,
                          film_material_cif=kwarg_film['film_material_cif'],
                          hkls=kwarg_film['film_hkl_bragg_peak'],
                          pot_step=kwarg_visulization['pot_step'],
                          beamline=name,
                          beamtime=beamtime,
                          kwarg = kwarg_data
                          )

def save_data_pxrd(data, scan_number, path, time_scan = False):
    if not time_scan:
        int_intensity_pd = pd.DataFrame(data).sort_values(by = '2theta')
        #smooth the intensity
        inten_smoothed = scipy.signal.savgol_filter(int_intensity_pd['intensity'],21,2)
        int_intensity_pd['intensity_smoothed']=inten_smoothed
        int_intensity_pd.to_excel(os.path.join(path,'data','pxrd_ascan_scan{}.xlsx'.format(scan_number)),columns = ['2theta','intensity','intensity_smoothed'])
        int_intensity_pd.to_csv(os.path.join(path,'data','pxrd_ascan_scan{}.csv'.format(scan_number)),columns = ['2theta','intensity','intensity_smoothed'])
    else:
        int_intensity_pd = pd.DataFrame(dict((key,data[key]) for key in data.keys() if key not in ['2theta','intensity'])).sort_values(by='frame_number')
        int_intensity_pd.to_excel(os.path.join(path,'data','pxrd_time_scan_scan{}.xlsx'.format(scan_number)))
        int_intensity_pd.to_csv(os.path.join(path,'data','pxrd_time_scan_scan{}.csv'.format(scan_number)))

def make_data_config_file(DaFy_path,data_folder,data, film_material_cif, hkls,pot_step,beamline,beamtime,kwarg):
    #Now append info to the plotting config file
    #scan number are unique
    scans = list(set(data['scan_no']))
    #ph values are not necessary unique
    ph_index = [list(data['scan_no']).index(scan) for scan in scans]
    phs = [data['phs'][i] for i in ph_index]
    for key in kwarg:
        globals()[key] = kwarg[key]
    config_file_plot = os.path.join(DaFy_path,'config','{}_{}_plot.ini'.format(beamline,beamtime))
    for scan in scans:
        scan_id =  '{}_{}_scan{}_{}'.format(beamline, beamtime, scan, time.strftime("%Y%m%d-%H%M%S"))
        index_scan = data['scan_no']==scan
        #save data
        np.savez(os.path.join(data_folder,'%s.npz'%(scan_id)),frame_number = data['image_no'][index_scan], potential=data['potential'][index_scan], potential_cal = data['potential_cal'][index_scan],\
                current_density=data['current'][index_scan], Time=data['image_no'][index_scan], pcov_ip=data['pcov_ip'][index_scan], pcov_oop=data['pcov_oop'][index_scan], mask_cv_xrd =data['mask_cv_xrd'][index_scan],\
                cen_ip=data['cen_ip'][index_scan], FWHM_ip=data['FWHM_ip'][index_scan], amp_ip=data['amp_ip'][index_scan], lorfact_ip=data['lfrac_ip'][index_scan],\
                bg_slope_ip=data['bg_slope_ip'][index_scan], bg_offset_ip=data['bg_offset_ip'][index_scan], cen_oop=data['cen_oop'][index_scan], \
                FWHM_oop=data['FWHM_oop'][index_scan], amp_oop=data['amp_oop'][index_scan], lorfact_oop=data['lfrac_oop'][index_scan], \
                bg_slope_oop=data['bg_slope_oop'][index_scan], bg_offset_oop=data['bg_offset_oop'][index_scan],peak_intensity = data['peak_intensity'][index_scan],\
                peak_intensity_error = data['peak_intensity_error'][index_scan])
        config_test = os.path.exists(config_file_plot)
        with open(config_file_plot, 'a+') as f:
            if not config_test:
                f.write("[beamtime]\nbeamtime = '{}'\n".format(beamtime))
            f.write('\n[{}]\n'.format(scan_id))
            f.write("scan_number = [{}]\n".format(scan))
            f.write("phs = [{}]\n".format(phs[scans.index(scan)]))
            f.write("colors = ['r']\n")
            f.write("xtal_lattices = ['{}']\n".format(film_material_cif.split('.')[0]))
            f.write("scan_ids = ['Scan_{}']\n".format(scan))
            f.write("scan_labels = ['Scan{}']\n".format(scan))
            f.write("ids_file_header ='{}'\n".format(os.path.join(data_folder,ids_file_head)))
            f.write("ids_files = ['{}']\n".format(ids_files[scans.index(scan)]))
            f.write("data_file_header = '{}'\n".format(data_folder))
            f.write("data_files= ['{}.npz']\n".format(scan_id))
            f.write("hkls = [{}]\n".format(hkls[0]))
            f.write("scan_direction_ranges =[[0,111,-2]]\n")
            f.write("plot_pot_steps = [{}]\n".format(int(pot_step)))

def merge_data_image_loader(data, object_image_loader):
    key_map_rules = {'scan_no':object_image_loader.scan_number,
                     'image_no':object_image_loader.frame_number,
                     'potential':object_image_loader.potential,
                     'current':object_image_loader.current,
                     'H':int(round(object_image_loader.hkl[0],0)),
                     'K':int(round(object_image_loader.hkl[1],0)),
                     'L':object_image_loader.hkl[2],
                     'phi':object_image_loader.motor_angles['phi'],
                     'chi':object_image_loader.motor_angles['chi'],
                     'mu':object_image_loader.motor_angles['mu'],
                     'delta':object_image_loader.motor_angles['delta'],
                     'gamma':object_image_loader.motor_angles['gamma'],
                     'omega_t':object_image_loader.motor_angles['omega_t'],
                     'omega':object_image_loader.motor_angles['omega'],
                     'mon':object_image_loader.motor_angles['mon'],
                     'transm':object_image_loader.motor_angles['transm']
                     }
    for key in key_map_rules:
        data[key].append(key_map_rules[key])
    return data

def cal_ctot_stationary(incidence_ang, det_ang_ver, det_ang_hor):
    """
    correction factors for stationary measurements (e.g. images)
    all angels in degree
    refer to Vlieg 1997, J.Appl.Cryst. for details
    """
    lorentz_factor = np.cos(np.deg2rad(incidence_ang)) * np.sin(np.deg2rad(det_ang_ver)) - np.sin(np.deg2rad(incidence_ang)) * np.cos(np.deg2rad(det_ang_ver)) * np.cos(np.deg2rad(det_ang_hor))
    area_factor = np.sin(np.deg2rad(incidence_ang))
    polarization_factor = 1/(1 - np.sin(np.deg2rad(det_ang_hor))**2 * np.cos(np.deg2rad(det_ang_ver))**2)
    # print(incidence_ang, det_ang_ver, det_ang_hor)
    # print(lorentz_factor, area_factor, polarization_factor)
    return lorentz_factor * area_factor * polarization_factor

def merge_data_image_loader_gsecars(data, object_image_loader):
    key_map_rules = {'scan_no':object_image_loader.scan_number,
                     'image_no':object_image_loader.frame_number,
                     'H':int(round(object_image_loader.hkl[0],0)),
                     'K':int(round(object_image_loader.hkl[1],0)),
                     'L':object_image_loader.hkl[2],
                     'phi':object_image_loader.motor_angles['phi'],
                     'chi':object_image_loader.motor_angles['chi'],
                     'mu':object_image_loader.motor_angles['mu'],
                     'del':object_image_loader.motor_angles['del'],
                     'nu':object_image_loader.motor_angles['nu'],
                     'eta':object_image_loader.motor_angles['eta'],
                     'norm':object_image_loader.motor_angles['norm'],
                     'transmission':object_image_loader.motor_angles['transmission']
                     }
    for key in key_map_rules:
        data[key].append(key_map_rules[key])
    if 'E' in object_image_loader.motor_angles and (not np.isnan(object_image_loader.motor_angles['E'])):
        data['E'].append(object_image_loader.motor_angles['E'])
    else:
        pass
    return data

def merge_data_bkg(data, object_bkg, correction_factor = 1):
    #correction_factor = lorenz_factor * polorization_factor * area_factor
    key_map_rules = {
                     'peak_intensity':object_bkg.fit_results['I'] * correction_factor,
                     'peak_intensity_error':object_bkg.fit_results['Ierr'] * correction_factor,
                     'noise':object_bkg.fit_results['noise'],
                     'mask_ctr':object_bkg.fit_status,
                     "roi_x": object_bkg.opt_values["cen"][1]-object_bkg.opt_values["row_width"], 
                     "roi_y":object_bkg.opt_values["cen"][0]-object_bkg.opt_values["col_width"], 
                     "roi_w":object_bkg.opt_values["row_width"]*2, 
                     "roi_h":object_bkg.opt_values["col_width"]*2, 
                     "ss_factor":object_bkg.ss_factor, 
                     "poly_func":object_bkg.fct, 
                     "poly_order":object_bkg.opt_values['int_power'], 
                     "poly_type":object_bkg.opt_values['poly_type'],
                     "peak_width":object_bkg.opt_values['peak_width'],
                     "peak_shift":object_bkg.peak_shift,
                     }
    if object_bkg.fit_status:
        for key in key_map_rules:
            data[key].append(key_map_rules[key])
    return data

def update_data_bkg(data, object_bkg):
    key_map_rules = {
                     'peak_intensity':object_bkg.fit_results['I'],
                     'peak_intensity_error':object_bkg.fit_results['Ierr'],
                     'noise':object_bkg.fit_results['noise'],
                     'mask_ctr':object_bkg.fit_status,
                     "roi_x": object_bkg.opt_values["cen"][1]-object_bkg.opt_values["row_width"], 
                     "roi_y":object_bkg.opt_values["cen"][0]-object_bkg.opt_values["col_width"], 
                     "roi_w":object_bkg.opt_values["row_width"]*2, 
                     "roi_h":object_bkg.opt_values["col_width"]*2, 
                     "ss_factor":object_bkg.ss_factor, 
                     "poly_func":object_bkg.fct, 
                     "poly_order":object_bkg.opt_values['int_power'], 
                     "poly_type":object_bkg.opt_values['poly_type'],
                     "peak_width":object_bkg.opt_values['peak_width']
                     }
    for key in key_map_rules:
        data[key][-1] = key_map_rules[key]
    # data['peak_intensity_error'][-1] = data['peak_intensity_error'][-1]/(data['mon'][-1]*data['transm'][-1])**0.5
    return data

def update_data_bkg_previous_frame(data, object_bkg, frame_index_offset=-1):
    key_map_rules = {
                     'peak_intensity':object_bkg.fit_results['I'],
                     'peak_intensity_error':object_bkg.fit_results['Ierr'],
                     'noise':object_bkg.fit_results['noise'],
                     'mask_ctr':object_bkg.fit_status,
                     "roi_x": object_bkg.opt_values["cen"][1]-object_bkg.opt_values["row_width"], 
                     "roi_y":object_bkg.opt_values["cen"][0]-object_bkg.opt_values["col_width"], 
                     "roi_w":object_bkg.opt_values["row_width"]*2, 
                     "roi_h":object_bkg.opt_values["col_width"]*2, 
                     "ss_factor":object_bkg.ss_factor, 
                     "poly_func":object_bkg.fct, 
                     "poly_order":object_bkg.opt_values['int_power'], 
                     "poly_type":object_bkg.opt_values['poly_type'],
                     "peak_width":object_bkg.opt_values['peak_width']
                     }
    for key in key_map_rules:
        data[key][frame_index_offset] = key_map_rules[key]
        # if key=='peak_intensity':
            # print(data[key][frame_index_offset],key_map_rules[key])
    # data['peak_intensity_error'][-1] = data['peak_intensity_error'][-1]/(data['mon'][-1]*data['transm'][-1])**0.5
    return data

def merge_data(data, object_image_loader, object_peak_fit, object_bkg, global_kwarg, tweak = False):
    key_map_rules = {'scan_no':object_image_loader.scan_number,
                     'phs': global_kwarg['phs'][global_kwarg['scan_nos'].index(object_image_loader.scan_number)],
                     'image_no':object_image_loader.frame_number,
                     'potential':object_image_loader.potential,
                     'potential_cal':object_image_loader.potential_cal,
                     'current':object_image_loader.current,
                     'peak_intensity':object_bkg.fit_results['I'],
                     'peak_intensity_error':object_bkg.fit_results['Ierr'],
                     'pcov_ip':object_peak_fit.fit_results['hor'][1],
                     'cen_ip':object_peak_fit.fit_results['hor'][0][0],
                     'strain_ip':0,
                     'grain_size_ip':0,
                     'FWHM_ip':object_peak_fit.fit_results['hor'][0][1],
                     'amp_ip':object_peak_fit.fit_results['hor'][0][2],
                     'lfrac_ip':object_peak_fit.fit_results['hor'][0][3],
                     'bg_slope_ip':object_peak_fit.fit_results['hor'][0][4],
                     'bg_offset_ip':object_peak_fit.fit_results['hor'][0][5],
                     'pcov_oop':object_peak_fit.fit_results['ver'][1],
                     'cen_oop':object_peak_fit.fit_results['ver'][0][0],
                     'FWHM_oop':object_peak_fit.fit_results['ver'][0][1],
                     'strain_oop':0,
                     'grain_size_oop':0,
                     'amp_oop':object_peak_fit.fit_results['ver'][0][2],
                     'lfrac_oop':object_peak_fit.fit_results['ver'][0][3],
                     'bg_slope_oop':object_peak_fit.fit_results['ver'][0][4],
                     'bg_offset_oop':object_peak_fit.fit_results['ver'][0][5],
                     'mask_cv_xrd':object_peak_fit.fit_status & (object_image_loader.frame_number not in range(*object_image_loader.abnormal_range)),
                     'mask_ctr':object_bkg.fit_status,
                     'H':object_image_loader.hkl[0],
                     'K':object_image_loader.hkl[1],
                     'L':object_image_loader.hkl[2],
                     'phi':object_image_loader.motor_angles['phi'],
                     'chi':object_image_loader.motor_angles['chi'],
                     'mu':object_image_loader.motor_angles['mu'],
                     'delta':object_image_loader.motor_angles['delta'],
                     'gamma':object_image_loader.motor_angles['gamma'],
                     'omega_t':object_image_loader.motor_angles['omega_t'],
                     'mon':object_image_loader.motor_angles['mon'],
                     'transm':object_image_loader.motor_angles['transm']
                     }
    if tweak:
        for key in data:
            if key!='bkg':
                data[key][-1] = key_map_rules[key]
    else:
        for key in data:
            if key!='bkg':
                data[key].append(key_map_rules[key])
    data['peak_intensity_error'][-1] = data['peak_intensity_error'][-1]/(data['mon'][-1]*data['transm'][-1])**0.5
    return data

def cut_profile_from_2D_img(img, cut_range, cut_direction, sum_result=True):
    if cut_direction=='horizontal':
        if sum_result:
            return np.sum(img[cut_range[0]:cut_range[1],:],axis = 0)
        else:
            return np.average(img[cut_range[0]:cut_range[1],:],axis = 0)

    elif cut_direction=='vertical':
        if sum_result:
            return np.sum(img[:,cut_range[0]:cut_range[1]],axis = 1)
        else:
            return np.average(img[:,cut_range[0]:cut_range[1]],axis = 1)

def extract_subset_of_zap_scan(n_data=0, l_boundary=[0, 1], bragg_ls = [], skip_bragg_l_offset = 0.05, delta_l_norm = 0.05, min_delta_l =0.01):
    def _point_density_calculator(l, l_bragg_adjacent,delta_l_bt_adj_bragg_peak):
        density_factor = 1/np.sin((l-l_bragg_adjacent)/delta_l_bt_adj_bragg_peak*np.pi)**2
        delta_l_current_point = delta_l_norm/density_factor
        if delta_l_current_point < min_delta_l:
            delta_l_current_point = min_delta_l
        # print(density_factor,delta_l_current_point)
        return delta_l_current_point
    l_all = np.linspace(l_boundary[0],l_boundary[1],n_data)
    l_new = []
    current_l = l_boundary[0]
    l_new.append(current_l)
    partial_index = []
    while current_l<l_boundary[1]:
        adjacent_bragg_l_index = np.argmin(np.abs(np.array(bragg_ls)-current_l))
        adjacent_bragg_l = bragg_ls[adjacent_bragg_l_index]
        if current_l > adjacent_bragg_l:
            right = True
        else:
            right = False
        if right:
            next_bragg_l = bragg_ls[adjacent_bragg_l_index +1]
        else:
            next_bragg_l = bragg_ls[adjacent_bragg_l_index -1]
        delta_l_current_point = _point_density_calculator(current_l, adjacent_bragg_l, abs(next_bragg_l-adjacent_bragg_l))
        if abs(current_l+delta_l_current_point-adjacent_bragg_l)>skip_bragg_l_offset:
            l_new.append(current_l+delta_l_current_point)
        else:
            pass
        current_l+= delta_l_current_point
    for each_l in l_new:
        partial_index.append(np.argmin(np.abs(np.array(l_all)-each_l)))
    # print(len(l_new),l_new)
    return partial_index


def cut_profile_from_2D_img_around_center(img, cut_offset = {'hor':10, 'ver':20}, data_range_offset = {'hor':50, 'ver':50}, center_index = None, sum_result = True):
    size = img.shape
    f = lambda x, y: [x-y, x+y]
    if center_index == None:
        center_index = [int(each/2) for each in size]
    data_range = {'hor':f(center_index[1],data_range_offset['hor']),'ver':f(center_index[0], data_range_offset['ver'])}
    cut_range = {'hor': f(center_index[0],cut_offset['hor']),'ver': f(center_index[1], cut_offset['ver'])}
    cut = {'hor': cut_profile_from_2D_img(img, cut_range['hor'], cut_direction ='horizontal', sum_result = sum_result)[data_range['hor'][0]:data_range['hor'][1]],
            'ver': cut_profile_from_2D_img(img, cut_range['ver'], cut_direction ='vertical', sum_result = sum_result)[data_range['ver'][0]:data_range['ver'][1]]}
    return cut

def create_mask_(img, img_q_par, img_q_ver, threshold = 10000, compare_method ='larger',remove_columns = [], \
                remove_rows = [], remove_pix = None, remove_q_range = {'par':[], 'ver':[]}, \
                remove_partial_range = {'point_couple':[{'p1':[2.4,3.2],'p2':[2.5,3.0]},{'p1':[2.55,3.3],'p2':[2.4,3.0]}],'pixel_width':[]}):
    #remove_partial_range, each point is of form [q_hor, q_ver]; two point couple will be used to calculate a line equation, which will be
    #coupled with pixel width to get those pixel index points to be excluded.
    #point group item must be a list of lib with only two items with keys of p1 and p2.
    mask = np.ones(np.array(img).shape)
    mask_ip_q = np.ones(np.array(img_q_par).shape)
    mask_oop_q = np.ones(np.array(img_q_ver).shape)

    def _find_pixel_index_from_q(grid_q_par, grid_q_ver, point):
        q_par_one_row = grid_q_par[0,:]
        q_ver_one_col = grid_q_ver[:,0]
        qx,qy = point
        index_point = [np.argmin(abs(q_ver_one_col - qy)),np.argmin(abs(q_par_one_row - qx))]
        return index_point

    if remove_q_range['par']!=[]:
        for each_range in remove_q_range['par']:
            mask_ip_q[np.where(np.logical_and(img_q_par>each_range[0], img_q_par<each_range[1]))] = 0
            # remove_rows = remove_rows + range(np.argmin(abs(img_q_par[:,0]- each_range[1])), np.argmin(abs(img_q_par[:,0]- each_range[0])))
            # print(np.argmin(abs(img_q_ver[:,0]- each_range[1])), np.argmin(abs(img_q_ver[:,0]- each_range[0])))
            # print img_q_ver.max(), img_q_par.max()
    if remove_q_range['ver']!=[]:
        for each_range in remove_q_range['ver']:
            mask_oop_q[np.where(np.logical_and(img_q_ver>each_range[0], img_q_ver<each_range[1]))] = 0
            # mask_oop_q[img_q_ver>each_range[0] && img_q_ver<each_range[1]] = 0
            # remove_columns = remove_columns + range(np.argmin(abs(img_q_ver[:,0]- each_range[0])), np.argmin(abs(img_q_ver[:,0]- each_range[1])))
            # print(np.argmin(abs(img_q_par[:,0]- each_range[0])), np.argmin(abs(img_q_par[:,0]- each_range[1])))
    if remove_partial_range['pixel_width']!=[]:
        for i in range(len(remove_partial_range['pixel_width'])):
            p1, p2 = remove_partial_range['point_couple'][i]['p1'], remove_partial_range['point_couple'][i]['p2']
            p1_index = _find_pixel_index_from_q(img_q_ver, img_q_par, p1)
            p2_index = _find_pixel_index_from_q(img_q_ver, img_q_par, p2)
            slope = float(p2_index[1]-p1_index[1])/float(p2_index[0]-p1_index[0])
            offset = p1_index[1] - slope*p1_index[0]
            h, w = img.shape[:2]
            j, k = np.ogrid[:h,:w]
            temp_mask = abs(j*slope+offset-k) < remove_partial_range['pixel_width'][i]
            remove_pix.extend([each for each in np.argwhere(temp_mask == True) if (each[0]>min(p1_index[0],p2_index[0])) and (each[0]<max(p1_index[0],p2_index[0]))])
    if compare_method =='larger':
        mask[img>threshold]=0
    elif compare_method =='smaller':
        mask[img<threshold]=0
    elif compare_method =='equal':
        mask[img == threshold] =0
    mask[:,remove_columns] = 0
    mask[remove_rows,:] = 0
    if remove_pix!=None:
        for each in remove_pix:
            mask[tuple(each)] = 0
    return mask*mask_ip_q*mask_oop_q

class create_mask():
    def __init__(self,kwarg):
        self.kwarg = kwarg

    def create_mask_new(self,img, img_q_par, img_q_ver, mon):
        #remove_partial_range, each point is of form [q_hor, q_ver]; two point couple will be used to calculate a line equation, which will be
        #coupled with pixel width to get those pixel index points to be excluded.
        #point group item must be a list of lib with only two items with keys of p1 and p2.
        kwarg = self.kwarg
        mask = np.ones(np.array(img).shape)
        mask_ip_q = np.ones(np.array(img_q_par).shape)
        mask_oop_q = np.ones(np.array(img_q_ver).shape)

        for key in kwarg:
            globals()[key] = kwarg[key]

        #threshold should be normalized to the mon = mon*transm
        #threshold = threshold/mon
        remove_q_range = {'par':remove_q_par,'ver':remove_q_ver}
        remove_partial_range = {'point_couple':line_strike_segments, 'pixel_width':line_strike_width}

        def _find_pixel_index_from_q(grid_q_ver, grid_q_par, point):
            q_par_one_row = grid_q_par[0,:]
            q_ver_one_col = grid_q_ver[:,0]
            qx,qy = point
            index_point = [np.argmin(abs(q_ver_one_col - qy)),np.argmin(abs(q_par_one_row - qx))]
            return index_point

        if remove_q_range['par']!=[]:
            for each_range in remove_q_range['par']:
                mask_ip_q[np.where(np.logical_and(img_q_par>each_range[0], img_q_par<each_range[1]))] = 0
                # remove_rows = remove_rows + range(np.argmin(abs(img_q_par[:,0]- each_range[1])), np.argmin(abs(img_q_par[:,0]- each_range[0])))
                # print(np.argmin(abs(img_q_ver[:,0]- each_range[1])), np.argmin(abs(img_q_ver[:,0]- each_range[0])))
                # print img_q_ver.max(), img_q_par.max()
        if remove_q_range['ver']!=[]:
            for each_range in remove_q_range['ver']:
                mask_oop_q[np.where(np.logical_and(img_q_ver>each_range[0], img_q_ver<each_range[1]))] = 0
                # mask_oop_q[img_q_ver>each_range[0] && img_q_ver<each_range[1]] = 0
                # remove_columns = remove_columns + range(np.argmin(abs(img_q_ver[:,0]- each_range[0])), np.argmin(abs(img_q_ver[:,0]- each_range[1])))
                # print(np.argmin(abs(img_q_par[:,0]- each_range[0])), np.argmin(abs(img_q_par[:,0]- each_range[1])))
        if remove_partial_range['pixel_width']!=[]:
            for i in range(len(remove_partial_range['pixel_width'])):
                p1, p2 = remove_partial_range['point_couple'][i]['p1'], remove_partial_range['point_couple'][i]['p2']
                p1_index = _find_pixel_index_from_q(img_q_ver, img_q_par, p1)
                p2_index = _find_pixel_index_from_q(img_q_ver, img_q_par, p2)
                slope = float(p2_index[1]-p1_index[1])/float(p2_index[0]-p1_index[0])
                offset = p1_index[1] - slope*p1_index[0]
                h, w = img.shape[:2]
                j, k = np.ogrid[:h,:w]
                temp_mask = abs(j*slope+offset-k) < remove_partial_range['pixel_width'][i]
                remove_pix.extend([each for each in np.argwhere(temp_mask == True) if (each[0]>min(p1_index[0],p2_index[0])) and (each[0]<max(p1_index[0],p2_index[0]))])
        mon = 1 #More pratical to only compare the scaled image pixel value to the specified threshold
        if compare_method =='larger':
            mask[img>(threshold/mon)]=0
        elif compare_method =='smaller':
            mask[img<(threshold/mon)]=0
        elif compare_method =='equal':
            maks[img == (threshold/mon)] =0
        #mask[:,remove_columns] = 0
        #mask[remove_rows,:] = 0
        if remove_pix!=None:
            for each in remove_pix:
                mask[tuple(each)] = 0
                # print('remove pix at',each)
        new_img = mask*mask_ip_q*mask_oop_q*img
        #now remove the masked columns and rows
        new_img = np.delete(new_img, remove_columns, axis =1)
        new_img = np.delete(new_img, remove_rows, axis =0)

        return new_img



def create_mask_bkg(img, threshold = 10000, compare_method ='larger',remove_columns = [], \
                remove_rows = [], remove_pix = None, remove_xy_range = {'par':[], 'ver':[]}, \
                remove_partial_range = {'point_couple':[{'p1':[10,20],'p2':[20,30]},{'p1':[20,30],'p2':[24,35]}],'pixel_width':[]}):
    #remove_partial_range, each point is of form [q_hor, q_ver]; two point couple will be used to calculate a line equation, which will be
    #coupled with pixel width to get those pixel index points to be excluded.
    #point group item must be a list of lib with only two items with keys of p1 and p2.For each point,(x_hor_index, y_ver_index)
    mask = np.ones(np.array(img).shape)

    if remove_xy_range['par']!=[]:
        for each_range in remove_xy_range['par']:
            mask[each_range[0]:each_range[1],:] = 0
            # remove_rows = remove_rows + range(np.argmin(abs(img_q_par[:,0]- each_range[1])), np.argmin(abs(img_q_par[:,0]- each_range[0])))
            # print(np.argmin(abs(img_q_ver[:,0]- each_range[1])), np.argmin(abs(img_q_ver[:,0]- each_range[0])))
            # print img_q_ver.max(), img_q_par.max()
    if remove_xy_range['ver']!=[]:
        for each_range in remove_xy_range['ver']:
            mask[:,each_range[0]:each_range[1]] = 0
            # mask_oop_q[img_q_ver>each_range[0] && img_q_ver<each_range[1]] = 0
            # remove_columns = remove_columns + range(np.argmin(abs(img_q_ver[:,0]- each_range[0])), np.argmin(abs(img_q_ver[:,0]- each_range[1])))
            # print(np.argmin(abs(img_q_par[:,0]- each_range[0])), np.argmin(abs(img_q_par[:,0]- each_range[1])))
    if remove_partial_range['pixel_width']!=[]:
        for i in range(len(remove_partial_range['pixel_width'])):
            p1_index, p2_index = remove_partial_range['point_couple'][i]['p1'], remove_partial_range['point_couple'][i]['p2']
            slope = float(p2_index[1]-p1_index[1])/float(p2_index[0]-p1_index[0])
            offset = p1_index[1] - slope*p1_index[0]
            h, w = img.shape[:2]
            j, k = np.ogrid[:h,:w]
            temp_mask = abs(j*slope+offset-k) < remove_partial_range['pixel_width'][i]
            remove_pix.extend([each for each in np.argwhere(temp_mask == True) if (each[0]>min(p1_index[0],p2_index[0])) and (each[0]<max(p1_index[0],p2_index[0]))])
    if compare_method =='larger':
        mask[img>threshold]=0
    elif compare_method =='smaller':
        mask[img<threshold]=0
    elif compare_method =='equal':
        mask[img == threshold] =0
    mask[:,remove_columns] = 0
    mask[remove_rows,:] = 0
    if remove_pix!=None:
        for each in remove_pix:
            mask[tuple(each)] = 0
    return mask

#data format based on Fouad's potentiostat
def extract_cv_data(file_path='/home/qiu/apps/048_S221_CV', which_cycle=1):
    #return:pot(V), current (mA)
    skiprows = 0
    with open(file_path,'r') as f:
        for each in f.readlines():
            if each.startswith('Time(s)'):
                skiprows+=1
                break
            else:
                skiprows+=1
    data = np.loadtxt(file_path,skiprows = skiprows)
    #nodes index saving all the valley pot positions
    nodes =[0]
    for i in range(len(data[:,1])):
        if i!=0 and i!=len(data[:,1])-1:
            if data[i,1]<data[i+1,1] and data[i,1]<data[i-1,1]:
                nodes.append(i)
    nodes.append(len(data[:,1]))
    if which_cycle>len(nodes):
        print('Cycle number lager than the total cycles! Use the first cycle instead!')
        return data[nodes[1]:nodes[2],1],data[nodes[1]:nodes[2],2]
    else:
        return data[nodes[which_cycle]:nodes[which_cycle+1],1],data[nodes[which_cycle]:nodes[which_cycle+1],2]

def data_point_picker(data_file = '/home/qiu/apps/DaFy_P23/data/DataBank_231_20190718-153651.npz',x_key='potential', y_key='cen_ip'):
    plt.ion()
    fig = plt.figure()
    data = np.load(data_file)
    print(list(data.keys()))
    x = data[x_key]
    y = data[y_key]
    index_keep =[]
    index_remove = []
    data_new = {}
    for i in range(len(x)):
        fig.clear()
        plt.plot(x,y,':',color = 'blue')
        plt.scatter(x[i],y[i],color ='r')
        data_quality = raw_input('Is this point a good point?(y/n/q);q means quit the loop. Your input is:') or 'y'
        if data_quality == 'n':
            index_remove.append(i)
        elif data_quality == 'q':
            break
        else:
            pass
        plt.tight_layout()
        plt.pause(0.05)
        plt.show()
    index_keep = [i for i in range(len(x)) if i not in index_remove]
    plt.ioff()
    fig.clear()
    print('After outliner points being removed:')
    plt.plot(x[index_keep],y[index_keep])
    plt.show()
    for key in data:
        try:
            data_new[key] = data[key][index_keep]
        except:
            data_new[key] = data[key]
    np.savez(data_file.replace('.npz','_filtered.npz'), data = data_new)
#data_point_picker()
