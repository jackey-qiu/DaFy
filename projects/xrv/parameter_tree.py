# -*- coding: utf-8 -*-
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
from util.UtilityFunctions import extract_vars_from_config
try:
    import ConfigParser as configparser
except:
    import configparser

#p.setReadonly(readonly=True)
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), type=typ, value=val, removable=True, renamable=True))


foms_label = ["diff",'log','sqrt','R1','R1_weighted','R1_weighted_2','chi2bars_2','R1_weighted_2b','R1_weighted_3','logR1','R2','R2_wighted','logR2','sintth4','Norm','chi2bars','chi2bars_w_trainor','chi2bars_weighted','chibars','logbars','R1bars','R2bars']
data_keys = ",".join(['phs','scan_no','image_no','potential','potential_cal','current','peak_intensity','peak_intensity_error','bkg', \
           'pcov_ip', 'strain_ip','grain_size_ip','cen_ip','FWHM_ip', 'amp_ip', 'lfrac_ip','bg_slope_ip','bg_offset_ip',\
           'pcov_oop','strain_oop','grain_size_oop','cen_oop','FWHM_oop','amp_oop','lfrac_oop','bg_slope_oop','bg_offset_oop',\
           'H','K','L','phi','chi','mu', 'delta', 'gamma','omega_t','mon','transm','mask_cv_xrd','mask_ctr'])
params = [
    {'name': 'Global', 'type': 'group', 'children': [
        {'name': 'beamline', 'type': 'str', 'value': "P23_PETRA3"},
        {'name': 'beamtime_id', 'type': 'str', 'value': "I20190574"},
        {'name': 'scan_nos', 'type': 'text', 'value': '82'},
        {'name': 'phs', 'type': 'text', 'value': '10,13,8,13,7,13'},
        {'name': 'data_keys', 'type': 'text', 'value': data_keys},
        {'name': 'cen', 'type': 'str', 'value': "637,328"},
        {'name': 'clip_width', 'type': 'str', 'value': "{'hor':200,'ver':300}"},
        {'name': 'dim_detector', 'type': 'str', 'value': "[1556,516]"},
    ]},
    {'name': 'Data_Storage', 'type': 'group', 'children': [
        {'name': 'ids_file_head', 'type': 'str', 'value': "ids"},
        {'name': 'ids_files', 'type': 'str', 'value': "['048_S221_CV','054_S229_CV','057_S231_CV','060_S236_CV','064_S243_CV','065_S244_CV']"},
    ]},
    {'name': 'Film_Lattice', 'type': 'group', 'children': [
        {'name': 'film_material_cif', 'type': 'str', 'value': "Co3O4.cif"},
        {'name': 'film_hkl_bragg_peak', 'type': 'str', 'value': "[[-1,1,3]]"},
        {'name': 'film_hkl_normal', 'type': 'str', 'value': '[1,1,1]'},
        {'name': 'film_hkl_x', 'type': 'str', 'value': '[1,1,-2]'},
    ]},
    {'name': 'Reciprocal_Mapping', 'type': 'group', 'children': [
        {'name': 'ub', 'type': 'text', 'value': "[-0.388904, -2.34687, 0.00424869, 0.000210428, -0.0104566, -0.889455, 2.48543, 0.905918, 0.00074013]"},
        {'name': 'sdd', 'type': 'str', 'value': "750"},
        {'name': 'e_kev', 'type': 'str', 'value': '22.5'},
        {'name': 'pixelsize', 'type': 'str', 'value': '[0.055,0.055]'},
        {'name': 'boost_mapping', 'type': 'str', 'value': 'False'},
    ]},
    {'name': 'Peak_Fit', 'type': 'group', 'children': [
        {'name': 'pot_step_scan', 'type': 'str', 'value': "False"},
        {'name': 'use_first_fit_for_pos', 'type': 'str', 'value': "True"},
        {'name': 'fit_bounds', 'type': 'text', 'value': "{'hor':[[1.0, 0.0050, 0, 0, -100000, -1e6],[1.3, 0.42, 1e9, 1, 100000, 1e6]], 'ver':[[1.2, 0.0050, 0, 0, -100000, -1e6],[1.6, 0.42, 1e9, 1, 100000, 1e6]]}"},
        {'name': 'fit_p0', 'type': 'str', 'value': "{'hor':[1.2, 0.21, 0.1, 0.5, 0, 0],'ver':[1.35, 0.21, 0.1, 0.5, 0, 0]}"},
        {'name': 'fit_p0_2', 'type': 'str', 'value': "{'hor':[1.2, 0.21, 0.1, 0.5, 0, 0],'ver':[1.35, 0.21, 0.1, 0.5, 0, 0]}"},
        {'name': 'cut_offset', 'type': 'str', 'value': "{'hor':[50,20],'ver':[50,20]}"},
        {'name': 'data_range_offset', 'type': 'str', 'value': "{'hor':[70,70],'ver':[70,70]}"},
    ]},
    {'name': 'Image_Loader', 'type': 'group', 'children': [
        {"name":"check_abnormality","type":"str","value":'False'},
        {"name":"left_offset","type":"str","value":'10'},
        {"name":"right_offset","type":"str","value":'100'},
        {'name': 'frame_prefix', 'type': 'str', 'value': 'i20180678_2'},
        {'name':'nexus_path','type':'str','value':'F://P23_I20180678/raw'},
        {'name':'constant_motors','type':'str','value':"{'omega_t':0.5, 'phi':0, 'chi':0, 'mu':0,'gamma':0,'delta':13.7}"},
    ]},
    {'name': 'Visulization', 'type': 'group', 'children': [
        {'name': 'vmax', 'type': 'str', 'value': "200"},
        {'name': 'vmin', 'type': 'str', 'value': "0"},
        {'name': 'cmap', 'type': 'str', 'value': 'jet'},
        {'name': 'pot_step', 'type': 'str', 'value': 'True'},
    ]},
    {'name': 'Mask', 'type': 'group', 'children': [
        {"name":"threshold","type":"str","value":'50000'},
        {"name":"compare_method","type":"list","value":'larger',"values":["larger","smaller"]},
        {"name":"remove_columns","type":"str","value":"10"},
        {"name":"remove_rows","type":"str","value":"10"},
        {"name":"remove_pix","type":"text","value":"[231,206]"},
        {"name":"remove_q_par","type":"str","value":"[]"},
        {"name":"remove_q_ver","type":"str","value":"[]"},
        {"name":"line_strike_segments","type":"str","value":"[]"},
        {"name":"line_strike_width","type":"str","value":"[]"},
    ]},
    {'name': 'Background_Subtraction', 'type': 'group', 'children': [
        {'name': 'rod_scan', 'type': 'str', 'value': "0"},
        {'name': 'check_level', 'type': 'str', 'value': "0.00000000001"},
        {'name': 'peak_shift', 'type': 'str', 'value': '0'},
        {'name': 'peak_width', 'type': 'str', 'value': '70'},
        {'name': 'update_width', 'type': 'str', 'value': "False"},
        {'name': 'row_width', 'type': 'str', 'value': "80"},
        {'name': 'col_width', 'type': 'str', 'value': "80"},
        {'name': 'bkg_row_width', 'type': 'str', 'value': "10"},
        {'name': 'bkg_col_width', 'type': 'str', 'value': "5"},
        {'name': 'bkg_win_cen_offset_lr', 'type': 'str', 'value': "10"},
        {'name': 'bkg_win_cen_offset_ud', 'type': 'str', 'value': "10"},
        {'name': 'int_direct', 'type': 'str', 'value': "x"},
        {'name': 'ord_cus_s', 'type': 'str', 'value': "[1]"},
        {'name': 'ss', 'type': 'str', 'value': "[1]"},
        {'name': 'ss_factor', 'type': 'str', 'value': "0.1"},
        {'name': 'fct', 'type': 'str', 'value': "atq"},
    ]},
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)
#print(p.names['Global'].names)
# print(len(p.children()))
# print(p.opts)
# p[('Global','beamline')]='Test'
# print(p.names['Global'].names)
## Create two ParameterTree widgets, both accessing the same data

# t.setWindowTitle('pyqtgraph example: Parameter Tree')
#from QtGui import QGridLayout

class SolverParameters(ParameterTree):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParameters(p, showTop=False)
        self.par = p

    def update_parameter(self,config_file, sections = ['Global','Data_Storage','Film_Lattice','Reciprocal_Mapping','Image_Loader','Mask','Peak_Fit','Visulization','Background_Subtraction']):
        for section in sections:
            kwarg_temp = extract_vars_from_config(config_file, section_var = section)
            for each in kwarg_temp:
                try:
                    self.par[(section,each)] = str(kwarg_temp[each])
                except:
                    pass

    def save_parameter(self, config_file):
        config = configparser.ConfigParser()
        sections = self.par.names.keys()
        for section in sections:
            sub_sections = self.par.names[section].names.keys()
            items = {}
            for each in sub_sections:
                items[each] = str(self.par[(section,each)])
            config[section] = items
        with open(config_file,'w') as config_file:
            config.write(config_file)

    def update_parameter_in_solver(self,parent):
        diffev_solver = parent.run_fit.solver.optimizer
        diffev_solver.set_km(self.par.param('Diff.Ev.').param('k_m').value())
        diffev_solver.set_kr(self.par.param('Diff.Ev.').param('k_r').value())
        diffev_solver.set_create_trial(self.par.param('Diff.Ev.').param('Method').value())
        parent.model.set_fom_func(self.par.param('FOM').param('Figure of merit').value())
        diffev_solver.set_autosave_interval(self.par.param('FOM').param('Auto save, interval').value())
        diffev_solver.set_use_start_guess(self.par.param('Fitting').param('start guess').value())
        diffev_solver.set_max_generations(self.par.param('Fitting').param('Generation size').value())
        diffev_solver.set_pop_size(self.par.param('Fitting').param('Population size').value())

