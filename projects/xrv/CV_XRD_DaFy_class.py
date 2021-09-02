import sys,os,itertools
import numpy as np
import pandas as pd
try:
    from . import locate_path
except:
    import locate_path
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
from GrainAnalysisEnginePool import cal_strain_and_grain
from VisualizationEnginePool import show_all_plots_new
from DataFilterPool import create_mask, merge_data, make_data_config_file
from FitEnginePool import fit_pot_profile, model
from FitEnginePool import Reciprocal_Space_Mapping
from FitEnginePool import XRD_Peak_Fitting
from FitEnginePool import background_subtraction_single_img
from util.XRD_tools import reciprocal_space_v3 as rsp
from util.UtilityFunctions import image_generator
from util.UtilityFunctions import scan_generator
from util.UtilityFunctions import nexus_image_loader
from util.UtilityFunctions import find_boundary
from util.UtilityFunctions import extract_global_vars_from_string
from util.UtilityFunctions import extract_vars_from_config
from util.UtilityFunctions import get_console_size
from functools import wraps

class run_app(object):
    def __init__(self):
        self.stop = True
        self.conf_file = None
        self.data = {}
        self.model = model
        self.data_path = os.path.join(DaFy_path, 'dump_files')


    def run(self, config):
        self.conf_file = config
        #extract global vars from config
        self.kwarg_global = extract_vars_from_config(self.conf_file, section_var ='Global')
        for each in self.kwarg_global:
            setattr(self,each,self.kwarg_global[each])

        #pars lib for everything else
        self.kwarg_visulization = extract_vars_from_config(self.conf_file, section_var ='Visulization')
        self.kwarg_film = extract_vars_from_config(self.conf_file, section_var ='Film_Lattice')
        self.kwarg_data = extract_vars_from_config(self.conf_file, section_var ='Data_Storage')
        self.kwarg_peak_fit = extract_vars_from_config(self.conf_file, section_var = 'Peak_Fit')
        self.kwarg_rsp = extract_vars_from_config(self.conf_file, section_var = 'Reciprocal_Mapping')
        self.kwarg_image = extract_vars_from_config(self.conf_file, section_var = 'Image_Loader')
        self.kwarg_mask = extract_vars_from_config(self.conf_file,section_var = 'Mask')

        #recal clip_boundary and cen
        self.clip_boundary = {"ver":[self.cen[0]-self.clip_width['ver'],self.cen[0]+self.clip_width['ver']+1],
                        "hor":[self.cen[1]-self.clip_width['hor'],self.cen[1]+self.clip_width['hor']+1]}
        self.cen_clip = [self.clip_width['ver'],self.clip_width['hor']]

        #data file
        for key in self.data_keys:
            self.data[key]=[]

        #init peak fit, bkg subtraction and reciprocal space and image loader instance
        self.bkg_sub = background_subtraction_single_img(self.cen_clip, self.conf_file, sections = ['Background_Subtraction'])
        self.peak_fitting_instance = XRD_Peak_Fitting(img = None, cen=self.cen_clip, kwarg = self.kwarg_peak_fit)
        self.rsp_instance = Reciprocal_Space_Mapping(img =None, cen=self.cen_clip, kwarg = self.kwarg_rsp)
        self.img_loader = nexus_image_loader(clip_boundary = self.clip_boundary, kwarg = self.kwarg_image)
        self.create_mask_new = create_mask(kwarg = self.kwarg_mask)
        self.lattice_skin = rsp.lattice.from_cif(os.path.join(DaFy_path, 'util','cif',"{}".format(self.kwarg_film['film_material_cif'])),
                                            HKL_normal=self.kwarg_film['film_hkl_normal'],\
                                            HKL_para_x=self.kwarg_film['film_hkl_x'],\
                                            E_keV=self.rsp_instance.e_kev, offset_angle=0)

        #build generator funcs
        self._scans = scan_generator(scans = self.scan_nos)
        self._images = image_generator(self._scans,self.img_loader,self.rsp_instance,self.peak_fitting_instance,self.create_mask_new)

    def run_script(self, bkg_intensity = 0):
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
            good_check = self.peak_fitting_instance.reset_fit(img, check = True)
            if good_check:
                self.bkg_sub.fit_background(None, img, plot_live = False, freeze_sf = True)
                self.data = merge_data(self.data, self.img_loader, self.peak_fitting_instance, self.bkg_sub, self.kwarg_global, tweak = False)
                self.data = cal_strain_and_grain(self.data,HKL = self.kwarg_film['film_hkl_bragg_peak'][0], lattice = self.lattice_skin)
                self.data['bkg'].append(bkg_intensity)
            return True
        except StopIteration:
            self.save_data_file(self.data_path)
            return False

    def run_update(self, bkg_intensity = 0):
        self.peak_fitting_instance.reset_fit(self.img, check = False)
        self.bkg_sub.fit_background(None, self.img, plot_live = False, freeze_sf = True)
        self.data = merge_data(self.data, self.img_loader, self.peak_fitting_instance, self.bkg_sub, self.kwarg_global, tweak = True)
        self.data = cal_strain_and_grain(self.data,HKL = self.kwarg_film['film_hkl_bragg_peak'][0], lattice = self.lattice_skin)
        self.data['bkg'][-1] = bkg_intensity

    def save_data_file(self, path):
        self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
        with self.writer as writer:
            pd.DataFrame(self.data).to_excel(writer,sheet_name='CV_XRD_data',columns = self.kwarg_global['data_keys'])
            writer.save()
        """
        make_data_config_file(
                            DaFy_path = DaFy_path,
                            data_folder = os.path.join(DaFy_path,'dump_files'),
                            data = pd.DataFrame(self.data),
                            film_material_cif=self.kwarg_film['film_material_cif'],
                            hkls=self.kwarg_film['film_hkl_bragg_peak'],
                            pot_step = self.kwarg_peak_fit['pot_step_scan'],
                            beamline=self.beamline,
                            beamtime=self.beamtime_id,
                            kwarg = self.kwarg_data
                            )
        """

if __name__ == "__main__":
    run_app()

