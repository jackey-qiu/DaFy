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
data_keys =",".join(['scan_no','image_no','potential','current','peak_intensity','peak_intensity_error', \
          'bkg', 'H','K','L','phi','chi','mu', 'delta', 'gamma','omega_t','mon','transm','mask_ctr'])
data_keys = ",".join(['scan_no', '2theta', 'peak_intensity'])
params = [
    {'name': 'Global', 'type': 'group', 'children': [
        {'name': 'beamline', 'type': 'str', 'value': "P23_PETRA3"},
        {'name': 'beamtime_id', 'type': 'str', 'value': "I20190574"},
        {'name': 'scan_nos', 'type': 'text', 'value': '82'},
        {'name': 'delta_range_plot', 'type': 'str', 'value': "[0,280]"},
        {'name': 'data_keys', 'type': 'text', 'value': data_keys},
        {'name': 'update_width', 'type': 'str', 'value': "False"},
        {'name': 'cen', 'type': 'str', 'value': "637,328"},
        {'name': 'clip_width', 'type': 'str', 'value': "{'hor':200,'ver':300}"},
        {'name': 'dim_detector', 'type': 'str', 'value': "[1556,516]"},
        {'name': 'sd', 'type': 'str', 'value': "930"},
        {'name': 'ps', 'type': 'str', 'value': "0.055"},
        {'name': 'time_scan', 'type': 'str', 'value': "True"},
    ]},
    {'name': 'Peak_Fit', 'type': 'group', 'children': [
        {'name': 'peak_ids', 'type': 'str', 'value': "['Cu2O(111)','Cu2O(200)','Cu(111)']"},
        {'name': 'colors', 'type': 'str', 'value': "['r','b','y']"},
        {'name': 'peak_ranges', 'type': 'text', 'value': "[[12.68,12.97],[14.73,14.97],[14.7,15.35]]"},
        {'name': 'peak_fit', 'type': 'str', 'value': "[False,False,True]"},
        {'name': 'peak_fit_bounds', 'type': 'text', 'value': "[[1.0, 0.1, 0, 0, -100000, -1e6],[1.3, 1.5, 1e9, 1, 100000, 1e6]]"},
        {'name': 'peak_fit_p0', 'type': 'str', 'value': "[None,0.2,0.01,0,0,0]"},
    ]},
    {'name': 'Image_Loader', 'type': 'group', 'children': [
        {"name":"check_abnormality","type":"str","value":'False'},
        {'name': 'frame_prefix', 'type': 'str', 'value': 'i20180678_2'},
        {'name':'nexus_path','type':'str','value':'F://P23_I20180678/raw'},
        {'name':'constant_motors','type':'str','value':"{'omega_t':0.5, 'phi':0, 'chi':0, 'mu':0,'gamma':0,'delta':13.7}"},
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
    ]}
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

    def update_parameter(self,config_file, sections = ['Global','Image_Loader','Mask','Peak_Fit']):
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

