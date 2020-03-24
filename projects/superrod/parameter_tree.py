# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree

foms_label = ["diff",'log','sqrt','R1','R1_weighted','R1_weighted_2','chi2bars_2','R1_weighted_2b','R1_weighted_3','logR1','R2','R2_wighted','logR2','sintth4','Norm','chi2bars','chi2bars_w_trainor','chi2bars_weighted','chibars','logbars','R1bars','R2bars']
params = [
    {'name': 'Diff.Ev.', 'type': 'group', 'children': [
        {'name': 'k_m', 'type': 'float', 'value': 0.9,'step':0.1},
        {'name': 'k_r', 'type': 'float', 'value': 0.9, 'step': 0.1},
        {'name': 'Method', 'type': 'list', 'values':["best_1_bin","rand_1_bin","best_either_or","rand_eithor_or",'jade_best','simplex_best_1_bin'],'value': "best_1_bin"},
    ]},
    {'name': 'FOM', 'type': 'group', 'children': [
        {'name': 'Figure of merit', 'type': 'list', 'value': 'chi2bars_2', 'values':foms_label},
        {'name':'Error bar level','type':'float','value':1.05},
        {'name':'Auto save, interval','type':'int','value':50},
        {'name':'save evals, buffer','type':'int','value':100000},
        # {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
        # {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 'Hz'},
        
    ]},
    {'name': 'Fitting', 'type': 'group', 'children': [
        {"name":"start guess","type":"bool","value":False},
        {"name":"Use (Max, Min)","type":"bool","value":True},
        {"name":"Population size","type":"int","value":20},
        {"name":"Generation size","type":"int","value":2000000},
    ]}
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

## Create two ParameterTree widgets, both accessing the same data

# t.setWindowTitle('pyqtgraph example: Parameter Tree')
#from QtGui import QGridLayout

class SolverParameters(ParameterTree):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParameters(p, showTop=False)
        self.par = p

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

