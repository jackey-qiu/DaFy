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


foms_label = ["diff",'log','log_debug','sqrt','R1','R1_weighted','R1_weighted_2','chi2bars_2','R1_weighted_2b','R1_weighted_3','logR1','R2','R2_wighted','logR2','sintth4','Norm','chi2bars','chi2bars_w_trainor','chi2bars_weighted','chibars','logbars','R1bars','R2bars']

data_keys_petra3 =",".join(['scan_no','image_no','potential','current','peak_intensity','peak_intensity_error', \
          'bkg', 'H','K','L','phi','chi','mu', 'delta', 'gamma','omega_t','mon','transm','mask_ctr'])

params_petra3 = [
    {'name': 'Global', 'type': 'group', 'children': [
        {'name': 'beamline', 'type': 'str', 'value': "P23_PETRA3"},
        {'name': 'beamtime_id', 'type': 'str', 'value': "I20190574"},
        {'name': 'scan_nos', 'type': 'text', 'value': '82'},
        {'name': 'data_keys', 'type': 'text', 'value': data_keys_petra3},
        {'name': 'update_width', 'type': 'str', 'value': "False"},
        {'name': 'cen', 'type': 'str', 'value': "637,328"},
        {'name': 'clip_width', 'type': 'str', 'value': "{'hor':200,'ver':300}"},
        {'name': 'dim_detector', 'type': 'str', 'value': "[1556,516]"},
    ]},
    {'name': 'Image_Loader', 'type': 'group', 'children': [
        {"name":"check_abnormality","type":"str","value":'False'},
        {'name': 'frame_prefix', 'type': 'str', 'value': 'i20180678_2'},
        {'name':'nexus_path','type':'str','value':'F://P23_I20180678/raw'},
        {'name':'constant_motors','type':'str','value':"{'omega_t':2, 'phi':0, 'chi':0}"},
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
        {"name":"rod_scan","type":"str","value":"False"},
        {"name":"plot_x_channel","type":"str","value":"None"},
        {"name":"int_direct","type":"str","value":'x'},
    ]}
]

data_keys_aps =",".join(['scan_no','image_no','peak_intensity','peak_intensity_error', \
          'bkg', 'H','K','L','phi','chi','mu', 'del', 'nu','eta','norm','transmission','mask_ctr'])

params_aps = [
    {'name': 'Global', 'type': 'group', 'children': [
        {'name': 'beamline', 'type': 'str', 'value': "APS_13IDC"},
        {'name': 'beamtime_id', 'type': 'str', 'value': "I20190574"},
        {'name': 'scan_nos', 'type': 'text', 'value': '82'},
        {'name': 'data_keys', 'type': 'text', 'value': data_keys_aps},
        {'name': 'update_width', 'type': 'str', 'value': "False"},
        {'name': 'cen', 'type': 'str', 'value': "200,100"},
        {'name': 'clip_width', 'type': 'str', 'value': "{'hor':50,'ver':50}"},
        {'name': 'dim_detector', 'type': 'str', 'value': "[487,195]"},
    ]},
    {'name': 'Image_Loader', 'type': 'group', 'children': [
        {'name':'spec_path','type':'str','value':'F://P23_I20180678/raw'},
        {'name':'spec_name','type':'str','value':'spec_file.spec'},
        {'name':'img_extention','type':'str','value':'tif'},
        {'name':'general_labels','type':'str','value':"{'H':'H','K':'K','L':'L','E':'Energy'}"},
        {'name':'correction_labels','type':'str','value':"{'time':'Seconds','norm':'io','transmission':'trans'}"},
        {'name':'angle_labels','type':'str','value':"{'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'}"},
        {'name':'angle_labels_escan','type':'str','value':"{'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'}"},
        {"name":"G_labels","type":"text","value":"{'n_azt':['G0',list(range(3,6))],\n'cell':['G1',list(range(0,6))],\n'or0':['G1',list(range(12,15))+list(range(18,24))+[30]],\n'or1':['G1',list(range(15,18))+list(range(24,30))+[31]],\n'lambda':['G4',list(range(3,4))]}"},
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
        {"name":"rod_scan","type":"str","value":"False"},
        {"name":"plot_x_channel","type":"str","value":"None"},
        {"name":"int_direct","type":"str","value":'x'},
    ]}
]


## Create tree of Parameter objects
p_petra3 = Parameter.create(name='params', type='group', children=params_petra3)
p_aps = Parameter.create(name='params', type='group', children=params_aps)
p_esrf = Parameter.create(name='params', type='group', children=params_aps)
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
        self.data_type = None

    def init_pars(self,data_type = 'PETRA3_P23'):
        self.data_type = data_type
        if data_type == 'PETRA3_P23':
            self.setParameters(p_petra3, showTop=False)
            self.par = p_petra3
        if data_type == 'APS_13IDC':
            self.setParameters(p_aps, showTop=False)
            self.par = p_aps
        if data_type == 'ESRF_ROBL':
            self.setParameters(p_esrf, showTop=False)
            self.par = p_esrf

    def update_parameter(self,config_file, sections = ['Global','Image_Loader','Mask','Background_Subtraction']):
        beamline = extract_vars_from_config(config_file, section_var = 'Global')['beamline']
        if  beamline!= self.data_type:
            return 'Data formate does not match! Your par frame take {} format, while your config file has a {} format. Fix it first!'.format(self.data_type, beamline)
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

