# -*- coding: utf-8 -*-
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
#from util.UtilityFunctions import extract_vars_from_config
try:
    import ConfigParser as configparser
except:
    import configparser

def extract_vars_from_config(config_file, section_var = None):
    config = configparser.ConfigParser()
    config.read(config_file)
    if section_var == None:
        section_var = config.sections()
    kwarg = {}
    for item in section_var:
        for each in config.items((item)):
            kwarg[(item,each[0])] = str(each[1])
    return kwarg

#p.setReadonly(readonly=True)
fig_settings = {'name': 'Fig_settings', 'type': 'group', 'children': [
        {'name': 'figsize', 'type': 'str', 'value': "(5.845,6.845)"},
        {'name': 'hspace', 'type': 'str', 'value': "[0.02,0.6]"},
        {'name': 'wspace', 'type': 'str', 'value': '[0.6,0.3]'}]}

data_info = {'name': 'Data_Info', 'type': 'group', 'children': [
        {'name': 'sequence_id', 'type': 'str', 'value': "[221,229, 231, 236, 243, 244]"},
        {'name': 'selected_scan', 'type': 'str', 'value': "[229,221,231,243]"},
        {'name': 'cv_folder', 'type': 'str', 'value': "/Users/canrong/Documents/I20180835_Jul_2019_CVs"},
        {'name': 'path', 'type': 'str', 'value': "['048_S221_CV','054_S229_CV','057_S231_CV','060_S236_CV','064_S243_CV','065_S244_CV']"},
        {'name': 'ph', 'type': 'str', 'value': '[10,13,8,13,7,13]'}]}

general_format_settings = {'name': 'General_Format_Settings', 'type': 'group', 'children': [
        {'name': 'fontsize_tick_label', 'type': 'str', 'value': "10"},
        {'name': 'fontsize_axis_label', 'type': 'str', 'value': "10"},
        {'name': 'fontsize_text_marker', 'type': 'str', 'value': "9"},
        {'name': 'fontsize_index_header', 'type': 'str', 'value': "11"},
        {'name': 'fmt', 'type': 'str', 'value': "['-']*6"},
        {'name': 'color', 'type': 'str', 'value': "['g','r','blue','r','m','r']"},
        {'name': 'index_header_pos_offset_cv', 'type': 'str', 'value': "[-0.4,0.1]"},
        {'name': 'index_header_pos_offset_tafel', 'type': 'str', 'value': "[-0.09,0.]"},
        {'name': 'index_header_pos_offset_order', 'type': 'str', 'value': "[-2.5,0.0]"},
        {'name': 'tafel_show_tick_label_x_y', 'type': 'str', 'value': "[True,True]"},
        {'name': 'order_show_tick_label_x_y', 'type': 'str', 'value': '[True,True]'}]}

axis_format_settings = {'name': 'Axis_Format_Settings', 'type': 'group', 'children': [
        {'name': 'cv_bounds_pot', 'type': 'str', 'value': "[1,1.9]+[1, 1.2, 1.4, 1.6,1.8]+0.1+1+{: 4.1f}+set_xlim"},
        {'name': 'cv_bounds_current', 'type': 'str', 'value': "[-1.2,6.2]+[0, 2, 4, 6]+1+4+{: 4.2f}+set_ylim"},
        {'name': 'tafel_bounds_pot', 'type': 'str', 'value': "[1.55,1.85]+[1.6, 1.7,1.8]+0.01+4+{: 4.2f}+set_xlim"},
        {'name': 'tafel_bounds_current', 'type': 'str', 'value': "[0.1,6]+[0.1,1]+0.01+9+{: 4.1f}+set_ylim"},
        {'name': 'order_bounds_ph', 'type': 'str', 'value': "[7,13]+[7,8,10,13]+1+0+{: 4.0f}+set_xlim"},
        {'name': 'order_bounds_y', 'type': 'str', 'value': "[1.65,1.8]+[1.65,1.7,1.75,1.8]+0.03+4+{: 3.2f}+set_ylim"},
        {'name': 'cv_show_tick_label_x', 'type': 'str', 'value': "[False, False, False, False, True, False]"},
        {'name': 'cv_show_tick_label_y', 'type': 'str', 'value': "[True, True, True, True, True, True]"}]}

data_analysis_settings = {'name': 'Data_Analysis_settings', 'type': 'group', 'children': [
        {'name': 'cv_scale_factor', 'type': 'str', 'value': "[30,30,30,30,30,30]"},
        {'name': 'scale_factor_text_pos', 'type': 'str', 'value': "[(1.35,2.0),(1.3,3.2),(1.3,2.2),(1.4,3.2),(1.3,2.2),(1.4,3.2)]"},
        {'name': 'cv_spike_cut', 'type': 'str', 'value': "[0.002]*6"},
        {'name': 'scan_rate', 'type': 'str', 'value': "[0.005]*6"},
        {'name': 'resistance', 'type': 'str', 'value': "[80,70,40,50,90,50]"},
        {'name': 'which_cycle', 'type': 'str', 'value': "[1,2,2,2,2,2]"},
        {'name': 'method', 'type': 'str', 'value': "['extract_cv_file_fouad']*6"},
        {'name': 'pot_range', 'type': 'str', 'value': "[[1.05,1.57],[1.05,1.57],[1.05,1.57],[1.05,1.57],[1.05,1.57],[1.05,1.57]]"},
        {'name': 'pot_starts_tafel', 'type': 'str', 'value': "[1.6,1.58,1.7,1.58,1.7,1.59]"},
        {'name': 'pot_ends_tafel', 'type': 'str', 'value': "[1.75,1.67,1.85,1.668,1.83,1.67]"},
        {'name': 'potential_reaction_order', 'type': 'str', 'value': "1.65"},
        {'name': 'current_reaction_order', 'type': 'str', 'value': "1"},
        {'name': 'reaction_order_mode', 'type': 'str', 'value': "constant_current"},
        {'name': 'current_filter_length', 'type': 'str', 'value': "19"},
        {'name': 'current_filter_order', 'type': 'str', 'value': '0'}]}

params = [fig_settings, data_info, general_format_settings, axis_format_settings, data_analysis_settings]


## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

class Parameters(ParameterTree):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParameters(p, showTop=False)
        self.par = p

    def update_parameter(self,config_file):
        kwarg_temp = extract_vars_from_config(config_file)
        for each in kwarg_temp:
            assert type(each)==tuple and len(each) == 2, 'the keys has to be tuple of length 2'
            self.par[each] = str(kwarg_temp[each])

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

    def set_field(self, section_name, field_name, value):
        self.par.param(section_name).param(field_name).setValue(str(value))

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

