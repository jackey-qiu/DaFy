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


params = [
    {'name': 'Plot', 'type': 'group', 'children': [
        {'name': 'energy_keV', 'type': 'float', 'value': 25.0},
        {'name': 'common_offset_angle', 'type': 'float', 'value': 0},
        {'name': 'q_inplane_lim', 'type': 'float', 'value': 6},
        {'name': 'q_mag_lim_low', 'type': 'float', 'value': 0.00001},
        {'name': 'q_mag_lim_high', 'type': 'float', 'value': 100},
        {'name': 'plot_axes', 'type': 'bool', 'value': True},
        {'name': 'qx_lim_high', 'type': 'str', 'value': "None"},
        {'name': 'qy_lim_high', 'type': 'str', 'value': "None"},
        {'name': 'qx_lim_low', 'type': 'str', 'value': "None"},
        {'name': 'qy_lim_low', 'type': 'str', 'value': "None"},
        {'name': 'qz_lim_low', 'type': 'float', 'value': -0.00001},
        {'name': 'qz_lim_high', 'type': 'float', 'value': 20},
    ]}
]


## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

class SolverParameters(ParameterTree):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_type = None

    def init_pars(self):
        self.setParameters(p, showTop=False)
        self.par = p

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

