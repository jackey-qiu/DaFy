import sys,os,itertools
import numpy as np
import pandas as pd
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
from GrainAnalysisEnginePool import cal_strain_and_grain
from VisualizationEnginePool import plot_bkg_fit
from DataFilterPool import create_mask, merge_data_bkg, update_data_bkg, update_data_bkg_previous_frame, merge_data_image_loader, merge_data_image_loader_gsecars,make_data_config_file
from FitEnginePool import fit_pot_profile
from FitEnginePool import Reciprocal_Space_Mapping
from FitEnginePool import XRD_Peak_Fitting
from FitEnginePool import background_subtraction_single_img
from util.XRD_tools import reciprocal_space_v3 as rsp
from util.UtilityFunctions import pop_last_item
from util.UtilityFunctions import image_generator_bkg, image_generator_bkg_gsecars
from util.UtilityFunctions import scan_generator
from util.UtilityFunctions import nexus_image_loader, gsecars_image_loader
from util.UtilityFunctions import find_boundary
from util.UtilityFunctions import extract_global_vars_from_string
from util.UtilityFunctions import extract_vars_from_config
from util.UtilityFunctions import get_console_size
from util.UtilityFunctions import make_tweak_string
from util.UtilityFunctions import tweak_integration
import time
#make compatibility of py 2 and py 3#
if (sys.version_info > (3, 0)):
    raw_input = input

class run_app(object):
    def __init__(self, beamline = 'PETRA3_P23'):
        self.stop = True
        self.current_frame = 0
        self.conf_file = None
        self.bkg_clip_image = None
        self.beamline = beamline
        self.data_path = os.path.join(DaFy_path,'dump_files')
        self.conf_path_temp = os.path.join(DaFy_path,"projects",'ctr','config_ctr_analysis_standard.ini')

    def run(self, config = None):
        #extract global vars from config
        if config == None:
            pass
        else:
            self.conf_file = config
            self.kwarg_global = extract_vars_from_config(self.conf_file, section_var ='Global')
        for each in self.kwarg_global:
            #flatten [[1,5],15] to [1,2,3,4,5,15] if necessarily
            if each=='scan_nos':
                temp_scans = []
                for each_item in self.kwarg_global['scan_nos']:
                    if type(each_item)==list:
                        temp_scans = temp_scans + list(range(each_item[0],each_item[1]+1))
                    else:
                        temp_scans.append(each_item)
                setattr(self,'scan_nos',temp_scans)
            else:
                setattr(self,each,self.kwarg_global[each])

        #pars lib for everything else
        self.kwarg_image = extract_vars_from_config(self.conf_file, section_var = 'Image_Loader')
        self.kwarg_mask = extract_vars_from_config(self.conf_file,section_var = 'Mask')

        #recal clip_boundary and cen
        self.clip_boundary = {"ver":[self.cen[0]-self.clip_width['ver'],self.cen[0]+self.clip_width['ver']+1],
                        "hor":[self.cen[1]-self.clip_width['hor'],self.cen[1]+self.clip_width['hor']+1]}     
        self.cen_clip = [self.clip_width['ver'],self.clip_width['hor']]

        self.img = None
        #data file
        self.data = {}
        if 'noise' not in self.data_keys:
            self.data_keys.append('noise')
        for key in self.data_keys:
            self.data[key]=[]
        # print(data)
        #init peak fit, bkg subtraction and reciprocal space and image loader instance
        self.bkg_sub = background_subtraction_single_img(self.cen_clip, self.conf_file, sections = ['Background_Subtraction'])
        if self.beamline == 'PETRA3_P23':
            self.img_loader = nexus_image_loader(clip_boundary = self.clip_boundary, kwarg = self.kwarg_image)
        elif self.beamline == 'APS_13IDC':
            self.img_loader = gsecars_image_loader(clip_boundary = self.clip_boundary, kwarg = self.kwarg_image, scan_numbers= self.scan_nos)
        self.create_mask_new = create_mask(kwarg = self.kwarg_mask)
        self.setup_frames()


    def setup_frames(self):
        #build generator funcs
        self._scans = scan_generator(scans = self.scan_nos)
        if self.beamline == 'PETRA3_P23':
            self._images = image_generator_bkg(self._scans,self.img_loader,self.create_mask_new)
        elif self.beamline == 'APS_13IDC':
            self._images = image_generator_bkg_gsecars(self._scans,self.img_loader,self.create_mask_new)

    def run_script(self,bkg_intensity = 0,poly_func = 'Vincent'):
        try:
            # t0 = time.time()
            img = next(self._images)
            #img = img/self.bkg_clip_image
            if hasattr(self,'current_scan_number'):
                if self.current_scan_number!=self.img_loader.scan_number:
                    # self.save_data_file(self.data_path)
                    self.current_scan_number = self.img_loader.scan_number
            else:
                setattr(self,'current_scan_number',self.img_loader.scan_number)
            self.current_frame = self.img_loader.frame_number
            self.img = img
            if self.beamline == 'PETRA3_P23':
                self.data = merge_data_image_loader(self.data, self.img_loader)
            elif self.beamline == 'APS_13IDC':
                self.data = merge_data_image_loader_gsecars(self.data, self.img_loader)
            self.bkg_sub.fit_background(None, img, self.data, plot_live = True, freeze_sf = True,poly_func = poly_func)
            self.data = merge_data_bkg(self.data, self.bkg_sub)
            self.data['bkg'].append(bkg_intensity)
            # print(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4)
            return True
        except StopIteration:
            self.save_data_file(self.data_path)
            return False

    def run_update(self,bkg_intensity = 0,begin = False, poly_func = 'Vincent'):
        if not begin:
            self.bkg_sub.fit_background(None, self.img, self.data, plot_live = True, freeze_sf = True, poly_func = poly_func)
        self.data = update_data_bkg(self.data, self.bkg_sub)
        self.data['bkg'][-1] = bkg_intensity

    def run_update_one_specific_frame(self, img, bkg_intensity, poly_func = 'Vincent', frame_offset = -1):
        self.bkg_sub.fit_background(None, img, self.data, plot_live = True, freeze_sf = True, poly_func = poly_func)
        # print(self.data['peak_intensity'])
        self.data = update_data_bkg_previous_frame(self.data, self.bkg_sub, frame_offset)
        # print(self.data['peak_intensity'])
        self.data['bkg'][frame_offset] = bkg_intensity

    def save_data_file(self,path):
        #update path for saving data
        if path == self.data_path:
            pass
        else:
            self.data_path = path
        #path_ = path.replace('.xlsx','_.xlsx')
        #writer_ = pd.ExcelWriter(path_,engine = 'openpyxl',mode = 'w')
        self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
        with self.writer as writer:
            pd.DataFrame(self.data).to_excel(writer,sheet_name='CTR_data',columns = self.data_keys)
            writer.save()
        #now empty the data container
        #for key in self.data_keys:
        #    self.data[key]=[self.data[key][-1]]

if __name__ == "__main__":
    run_app()
