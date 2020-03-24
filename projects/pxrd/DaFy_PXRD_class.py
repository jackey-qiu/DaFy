import sys,os,itertools
import numpy as np
import pandas as pd
import copy
import scipy
import scipy.optimize as opt
try:
    from . import locate_path
except:
    import locate_path
try:
    from mpi4py import MPI
except:
    print('mpi4py not installed, use single processor!')
try:
    import ConfigParser as configparser
except:
    import configparser
script_path = locate_path.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
import matplotlib,time
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from VisualizationEnginePool import plot_pxrd_profile,plot_pxrd_profile_time_scan
from DataFilterPool import create_mask, save_data_pxrd
from FitEnginePool import fit_pot_profile,backcor,model
from util.XRD_tools import reciprocal_space_v3 as rsp
from util.UtilityFunctions import image_generator_bkg
from util.UtilityFunctions import scan_generator
from util.UtilityFunctions import nexus_image_loader
from util.UtilityFunctions import extract_vars_from_config
from util.UtilityFunctions import show_status_bar_2
#make compatibility of py 2 and py 3#
if (sys.version_info > (3, 0)):
    raw_input = input

class run_app(object):
    def __init__(self):
        self.stop = True
        self.conf_file = None
        self.data = {}
        self.model = model
        self.int_range = []
        self.current_frame = 0
        self.int_range_bkg = []
        self.spikes_bounds = None
        self.data_path = os.path.join(DaFy_path,'dump_files')
        self.conf_path_temp = os.path.join(DaFy_path,'projects','pxrd','config_pxrd_standard.ini')

    def run(self, config = 'C:\\apps\\DaFy_P23\\config\\config_p23_pxrd_new.ini'):
        #extract global vars from config
        if config == None:
            #self.kwarg_global = extract_vars_from_config(self.conf_file, section_var ='Global')
            pass
        else:
            self.conf_file = config
            self.kwarg_global = extract_vars_from_config(self.conf_file, section_var ='Global')
        
        for each in self.kwarg_global:
            setattr(self,each,self.kwarg_global[each])
        if self.time_scan:
            self.plot_pxrd = plot_pxrd_profile_time_scan
        else:
            self.plot_pxrd = plot_pxrd_profile
        #set data to empty lib {}
        self.data = {}
        #pars lib for everything else
        self.kwarg_image = extract_vars_from_config(self.conf_file, section_var = 'Image_Loader')
        self.kwarg_mask = extract_vars_from_config(self.conf_file,section_var = 'Mask')
        try:
            self.kwarg_bkg = extract_vars_from_config(self.conf_file,section_var = 'Background_Subtraction')
        except:
            self.kwarg_bkg = {"ord_cus_s":8, "ss":5, "ss_factor":0.08, "fct":"ah"}
        self.kwarg_peak_fit = extract_vars_from_config(self.conf_file,section_var = 'Peak_Fit')

        #recal clip_boundary and cen(you need to remove the edges)
        self.ver_offset = self.clip_width['ver']
        self.hor_offset = self.clip_width['hor']
        self.clip_boundary = {"ver":[self.ver_offset,self.dim_detector[0]-self.ver_offset],"hor":[self.hor_offset,self.dim_detector[1]-self.hor_offset]}
        self.cen_clip = [self.cen[0]-self.ver_offset,self.cen[1]-self.hor_offset]

        #init peak fit, bkg subtraction and reciprocal space and image loader instance
        self.img_loader = nexus_image_loader(clip_boundary = self.clip_boundary, kwarg = self.kwarg_image)
        
        self.create_mask_new = create_mask(kwarg = self.kwarg_mask)

        #build generator funcs
        self._scans = scan_generator(scans = self.scan_nos)
        self._images = image_generator_bkg(self._scans,self.img_loader,self.create_mask_new)
        

    def run_script(self,bkg_intensity = 0):
        try:
            img = next(self._images)
            if hasattr(self,'current_scan_number'):
                if self.current_scan_number!=self.img_loader.scan_number:
                    self.save_data_file(self.data_path)
                    self.current_scan_number = self.img_loader.scan_number
            else:
                setattr(self,'current_scan_number',self.img_loader.scan_number)
            self.current_frame = self.img_loader.frame_number
            self.img = img
            self.merge_data_image_loader()
            self.fit_background()
            self._merge_data_bkg(tweak = False)
            if self.time_scan:
                self.fit_peak(tweak = False)
                self.data[self.img_loader.scan_number]['bkg'].append(bkg_intensity)
            return True

        except StopIteration:
            self.save_data_file(self.data_path)
            return False

    def run_update(self,bkg_intensity = 0):
        self.fit_background()
        self._merge_data_bkg(tweak = True)
        if self.time_scan:
            self.fit_peak(tweak = True)
            self.data[self.img_loader.scan_number]['bkg'][-1] = bkg_intensity

    def merge_data_image_loader(self):
        if self.img_loader.scan_number not in self.data:
            if not self.time_scan:
                self.data[self.img_loader.scan_number] = {'2theta':[],'intensity':[],'intensity_without_bkg':[],'potential':[],'current':[]}
            else:
                self.data[self.img_loader.scan_number] = {'time':[],'2theta':[],'intensity':[],'bkg':[],'frame_number':[],'potential':[],'current':[]}
                for i in range(len(self.kwarg_peak_fit['peak_ranges'])):
                    self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_intensity']=[]
                    if self.kwarg_peak_fit['peak_fit'][i]:
                        pars = ['_peak_pos','_FWHM','_amp','_lfrac','_bg_slope','_bg_offset','_pcov','_fit_status']
                        for each in pars:
                            self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+each]=[]
                    else:
                        pass
        else:
            pass

    def fit_background(self,fit_background=True):
        int_range = np.sum(self.img[:,:],axis = 1)
        if fit_background:
            #bkg subtraction
            n=np.array(range(len(int_range)))
            #by default the contamination rate is 25%
            #the algorithm may fail if the peak cover >40% of the cut profile
            bkg_n = int(len(n)/4)
            y_sorted = list(copy.deepcopy(int_range))
            y_sorted.sort()
            std_bkg =np.array(y_sorted[0:bkg_n*3]).std()/(max(y_sorted)-min(y_sorted))
            #int_range[np.argmin(int_range)] =  int_range[np.argmin(int_range)+1]#???what for???
            int_range_bkg, *discard = backcor(range(len(int_range)),int_range,\
                                                ord_cus=self.kwarg_bkg['ord_cus_s'],\
                                                s=std_bkg*self.kwarg_bkg['ss_factor'],fct=self.kwarg_bkg['fct'])
            self.int_range = (int_range-int_range_bkg)[::-1]
            self.int_range_bkg = list(int_range_bkg)[::-1]
            self.int_range_without_bkg = int_range[::-1]
        else:
            pass
            #self.int_range = int_range[::-1]
            #self.int_range_bkg = int_range*0

    def get_peak_fit_data(self,delta,data,bounds):
        spikes = self.spikes_bounds
        delta = np.array(delta)
        data = np.array(data)
        if spikes is not None:
            try:
                if spikes[0]>bounds[0] and spikes[1]<bounds[1]:
                    return data[np.where(((delta>=bounds[0]) & (delta<=spikes[0]))|((delta>=spikes[1]) & (delta<=bounds[1])))]
                else:
                    return data[np.where((delta>=bounds[0]) & (delta<=bounds[1]))]
            except:
                print('conflict of spike bounds compare to data range!')
        else:
            return data[np.where((delta>=bounds[0]) & (delta<=bounds[1]))]
        #return data[np.where(delta>=bounds[0 and delta<=bounds[1]])]
        #return data[np.where((delta>=bounds[0]) & (delta<=bounds[1]))]

    def fit_peak(self,tweak = False):
        self.peak_fit_results, self.peak_fit_fom, self.peak_fit_status= [], [], []
        for i in range(len(self.kwarg_peak_fit['peak_fit'])):
            if self.kwarg_peak_fit['peak_fit'][i]:
                fit_bounds = self.kwarg_peak_fit['peak_fit_bounds']
                fit_bounds[0][0], fit_bounds[1][0] = self.kwarg_peak_fit['peak_ranges'][i]
                fit_p0= [np.mean(self.kwarg_peak_fit['peak_ranges'][i])] + self.kwarg_peak_fit['peak_fit_p0'][1:]
                #print(fit_bounds)
                xdata = self.get_peak_fit_data(delta=self.data[self.img_loader.scan_number]['2theta'],\
                                               data = self.data[self.img_loader.scan_number]['2theta'],\
                                               bounds = self.kwarg_peak_fit['peak_ranges'][i])
                ydata = self.get_peak_fit_data(delta=self.data[self.img_loader.scan_number]['2theta'],\
                                               data = self.int_range,\
                                               bounds = self.kwarg_peak_fit['peak_ranges'][i])
                try:
                    fit_result,fom_result = opt.curve_fit(f=model, xdata=xdata, ydata=ydata, p0 = fit_p0, bounds = fit_bounds, max_nfev = 10000)
                except:
                    pass
                #    fit_result 
                try:
                    #fit_result,fom_result = opt.curve_fit(f=model, xdata=xdata, ydata=ydata, p0 = fit_p0, bounds = fit_bounds, max_nfev = 10000)
                    #print(fit_result)
                    if not tweak:
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_peak_pos'].append(fit_result[0])
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_FWHM'].append(fit_result[1])
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_amp'].append(fit_result[2])
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_lfrac'].append(fit_result[3])
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_slope'].append(fit_result[4])
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_offset'].append(fit_result[5])
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_fit_status'].append(True)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_pcov'].append(fom_result)
                    else:
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_peak_pos'][-1]=fit_result[0]
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_FWHM'][-1]=fit_result[1]
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_amp'][-1]=fit_result[2]
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_lfrac'][-1]=fit_result[3]
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_slope'][-1]=fit_result[4]
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_offset'][-1]=fit_result[5]
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_fit_status'][-1]=True
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_pcov'][-1]=fom_result
                except:
                    if not tweak:
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_peak_pos'].append(0)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_FWHM'].append(0)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_amp'].append(0)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_lfrac'].append(0)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_slope'].append(0)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_offset'].append(0)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_fit_status'].append(False)
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_pcov'].append(None)
                    else:
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_peak_pos'][-1]=0
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_FWHM'][-1]=0
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_amp'][-1]=0
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_lfrac'][-1]=0
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_slope'][-1]=0
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_bg_offset'][-1]=0
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_fit_status'][-1]=False
                        self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][i]+'_pcov'][-1]=None
                
    def _merge_data_bkg(self, tweak = False):
        if hasattr(self,'delta_motor'):
            pass
        else:
            self.delta_motor = list(self.img_loader.extract_delta_angles())
        #run this after fit_background
        t0=time.time()
        #data_temp = pd.DataFrame(self.data[self.img_loader.scan_number])
        if not self.time_scan:
            delta = self.img_loader.motor_angles['delta']
            delta_range = np.round(delta + np.arctan((self.cen_clip[0] - np.array(range(self.dim_detector[0]-self.ver_offset*2)))*self.ps/self.sd)/np.pi*180, 4)[::-1]
            self.delta_range = delta_range
             #append results
            if self.delta_motor[0]==delta:
                where_is_delta = 'head'
            else:
                where_is_delta = self.delta_motor.index(delta)           
            append_index_lf = None
            append_index_rg = None
            if where_is_delta=='head':
                append_index_lf = 0
                append_index_rg = len(delta_range)
            else:
                append_index_lf=np.argmin(abs(delta_range-self.data[self.img_loader.scan_number]['2theta'][-1]))
                append_index_rg = len(delta_range)
                if self.data[self.img_loader.scan_number]['2theta'][-1] == delta_range[append_index_lf]:
                    append_index_lf = append_index_lf +1
            if tweak:
                current_length = len(self.data[self.img_loader.scan_number]['2theta'])
                append_length =append_index_rg - append_index_lf
                if append_length<0:
                    append_length=0
                self.data[self.img_loader.scan_number]['intensity'][current_length-append_length:current_length] = list(self.int_range[append_index_lf:append_index_rg])
                self.data[self.img_loader.scan_number]['intensity_without_bkg'][current_length-append_length:current_length] = list(self.int_range_without_bkg[append_index_lf:append_index_rg])
            else:
                self.data[self.img_loader.scan_number]['2theta'] += list(delta_range[append_index_lf:append_index_rg])
                self.data[self.img_loader.scan_number]['intensity'] += list(self.int_range[append_index_lf:append_index_rg])
                self.data[self.img_loader.scan_number]['intensity_without_bkg'] += list(self.int_range_without_bkg[append_index_lf:append_index_rg])
                self.data[self.img_loader.scan_number]['potential']+=[self.img_loader.potential]*len(range(append_index_lf,append_index_rg))
                self.data[self.img_loader.scan_number]['current']+=[self.img_loader.current]*len(range(append_index_lf,append_index_rg))
        else:
            if len(self.data[self.img_loader.scan_number]['2theta']) == 0:
                delta = self.img_loader.motor_angles['delta']
                delta_range = np.round(delta + np.arctan((self.cen_clip[0] - np.array(range(self.dim_detector[0]-self.ver_offset*2)))*self.ps/self.sd)/np.pi*180, 4)[::-1]
                
                self.data[self.img_loader.scan_number]['2theta'] = delta_range
                self.data[self.img_loader.scan_number]['intensity'] = self.int_range
            else:#time scan: same 2theta value for each frame
                self.data[self.img_loader.scan_number]['intensity'] = np.array(self.int_range)
            k=0
            for each_segment in self.kwarg_peak_fit['peak_ranges']:
                index_left = np.argmin(np.abs(np.array(self.data[self.img_loader.scan_number]['2theta'])-each_segment[0]))
                index_right = np.argmin(np.abs(np.array(self.data[self.img_loader.scan_number]['2theta'])-each_segment[1]))
                if not tweak:
                    self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][k]+'_intensity'].append(np.array(self.int_range)[min([index_left,index_right]):max([index_left,index_right])].sum())
                else:
                    self.data[self.img_loader.scan_number][self.kwarg_peak_fit['peak_ids'][k]+'_intensity'][-1] = np.array(self.int_range)[min([index_left,index_right]):max([index_left,index_right])].sum()
                k = k+1
            if not tweak:
                self.data[self.img_loader.scan_number]['frame_number'].append(self.img_loader.frame_number)
                self.data[self.img_loader.scan_number]['potential'].append(self.img_loader.potential)
                self.data[self.img_loader.scan_number]['current'].append(self.img_loader.current)
                self.data[self.img_loader.scan_number]['time'].append(self.img_loader.motor_angles['time'])

    def _merge_data_bkg_old(self, tweak = False):
        if hasattr(self,'delta_motor'):
            pass
        else:
            self.delta_motor = list(self.img_loader.extract_delta_angles())
        #run this after fit_background
        #data_temp = pd.DataFrame(self.data[self.img_loader.scan_number])
        if not self.time_scan:
            delta = self.img_loader.motor_angles['delta']
            delta_range = list(np.round(delta + np.arctan((self.cen_clip[0] - np.array(range(self.dim_detector[0]-self.ver_offset*2)))*self.ps/self.sd)/np.pi*180, 4))
            if not tweak:
                for j in delta_range:
                    if j in self.data[self.img_loader.scan_number]['2theta']:
                        jj = self.data[self.img_loader.scan_number]['2theta'].index(j)
                        self.data[self.img_loader.scan_number]['intensity'][jj] = 0.5*self.data[self.img_loader.scan_number]['intensity'][jj] + 0.5*self.int_range[delta_range.index(j)]
                    else:
                        self.data[self.img_loader.scan_number]['2theta'].append(j)
                        self.data[self.img_loader.scan_number]['intensity'].append(self.int_range[delta_range.index(j)])

                self.data[self.img_loader.scan_number]['2theta_previous'] = copy.deepcopy(self.data[self.img_loader.scan_number]['2theta'])
                self.data[self.img_loader.scan_number]['intensity_previous'] = copy.deepcopy(self.data[self.img_loader.scan_number]['intensity'])
            else:
                for j in delta_range:
                    if j in self.data[self.img_loader.scan_number]['2theta_previous']:
                        jj = self.data[self.img_loader.scan_number]['2theta_previous'].index(j)
                        self.data[self.img_loader.scan_number]['intensity'][jj] = 0.5*self.data[self.img_loader.scan_number]['intensity_previous'][jj] + 0.5*self.int_range[delta_range.index(j)]
                    else:
                        #self.data[self.img_loader.scan_number]['2theta'].append(j)
                        self.data[self.img_loader.scan_number]['intensity'].append(self.int_range[delta_range.index(j)])
            #self.data[self.img_loader.scan_number]=data_temp
            
        else:
            if len(self.data[self.img_loader.scan_number]['2theta']) == 0:
                delta = self.img_loader.motor_angles['delta']
                delta_range = np.round(delta + np.arctan((self.cen[0] - np.array(range(self.dim_detector[0]-self.ver_offset*2)))*self.ps/self.sd)/np.pi*180, 3)
                self.data[self.img_loader.scan_number]['2theta'] = delta_range
                self.data[self.img_loader.scan_number]['intensity'] = self.int_range
            else:#time scan: same 2theta value for each frame
                self.data[self.img_loader.scan_number]['intensity'] = np.array(self.int_range)
            k=0
            for each_segment in self.delta_segment_time_scan:
                k = k+1
                index_left = np.argmin(np.abs(np.array(self.data[self.img_loader.scan_number]['2theta'])-each_segment[0]))
                index_right = np.argmin(np.abs(np.array(self.data[self.img_loader.scan_number]['2theta'])-each_segment[1]))
                if not tweak:
                    self.data[self.img_loader.scan_number]['intensity_peak{}'.format(k)].append(np.array(self.int_range)[min([index_left,index_right]):max([index_left,index_right])].sum())
                else:
                    self.data[self.img_loader.scan_number]['intensity_peak{}'.format(k)][-1] = np.array(self.int_range)[min([index_left,index_right]):max([index_left,index_right])].sum()

    def save_data_file(self,path,one_sheet = True):
        #to be finished
        if self.time_scan:
            keys = ['time','frame_number','potential','current','bkg']+[each+'_intensity' for each in self.kwarg_peak_fit['peak_ids']]
            for i in range(len(self.kwarg_peak_fit['peak_ranges'])):
                if self.kwarg_peak_fit['peak_fit'][i]:
                    pars = ['_peak_pos','_FWHM','_amp','_lfrac','_bg_slope','_bg_offset','_pcov','_fit_status']
                    for each in pars:
                        keys.append(self.kwarg_peak_fit['peak_ids'][i]+each)
        else:
            keys = ['2theta','intensity','intensity_without_bkg','potential','current']
        #print(keys)
        self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
        # if hasattr(self,'writer'):
            # pass
        # else:
            # try:
                # self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
            # except:
                # df_temp = pd.DataFrame({})
                # writer_temp = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))])
                # df_temp.to_excel(writer_temp)
                # writer_temp.save()
                # self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
        #data = {key:self.data[self.img_loader.scan_number][key] for key in keys}
        if not one_sheet:
            data = {key:self.data[self.current_scan_number][key] for key in keys}
            data['scan_number'] = [self.current_scan_number]*len(data['current'])
            df = pd.DataFrame(data)
            df.to_excel(self.writer,sheet_name='scan{}'.format(self.current_scan_number),columns = ['scan_number']+keys)
            self.writer.save()
        else:#all data saved in one sheet
            data = {}
            scan_numbers = list(self.data.keys())
            scan_numbers.sort()
            scan_numbers = scan_numbers
            for key in scan_numbers:
                data_temp = copy.deepcopy(self.data[key])
                data_temp = {k:data_temp[k] for k in keys}
                data_temp['scan_number'] = [key]*len(data_temp['current'])
                if data == {}:
                    data.update(data_temp)
                else:
                    for k in ['scan_number']+keys:
                        data[k] = data[k] + data_temp[k]
            #data.update(self.data[key]) for key in scan_numbers
            #data['scan_number'] = [self.current_scan_number]*len(data['current'])
            df = pd.DataFrame(data)
            df.to_excel(self.writer,sheet_name='Sheet1',columns = ['scan_number']+keys)
            self.writer.save()




if __name__ == "__main__":
    run_app()
