import nlopt
import numpy as np
from threading import Thread
import pickle

class Parameter():
    def __init__(self, name, value, min, max, vary):
        self.name = name
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary

class Parameters(dict):
    def __init__(self):
        return
    def add(self, name, value, min, max, vary):
        self[name] = Parameter(name, value, min, max, vary)
    def get_str(self):
        _str = ""
        for key in self.keys():
            _str += self[key].name + " = " + str(self[key].value) + "  "
        return _str
    def get_str_2(self, sig_figs=5):
        min_len = sig_figs+8
        str_label = ""
        str_value = ""
        for key in sorted(self.keys()):
            len_key = len(key) + 2
            len_final = np.max([len_key, min_len])
            str_label += key.ljust(len_final)
            str_value += ('%.*e'%( sig_figs, self[key].value)).ljust(len_final)
        return str_label + '\n' + str_value
    def save(self, filename):
        with open(filename, 'wb+') as f:
            pickle.dump(self, f, 0)
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class data_fitter():
    def __init__(self, params, fitfunc, algorithm=nlopt.LN_COBYLA, xtol_rel=1e-4):
        self.params = params
        self.fitfunc = fitfunc
        self.algorithm = algorithm
        self.xtol_rel = xtol_rel
        self.thread = None
        self.abort_fit = False
        self.thread_running = False
        
    def set_params(self, params):
        self.params = params
        
    def get_params(self):
        return self.params
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        
    def set_tolerance(self, tolerance):
        self.xtol_rel = tolerance
        
    def start_fit(self):
        self.abort_fit = False
        self.thread = Thread(target=self.fit_in_bg)
        self.thread.run()
        
    def stop_fit(self):
        self.abort_fit = True
        
    def fit_in_bg(self):
        self.thread_running = True
        x = []
        bounds_low = []
        bounds_high = []
        for key in self.params:
            if self.params[key].vary:
                x.append(self.params[key].value)
                bounds_low.append(self.params[key].min)
                bounds_high.append(self.params[key].max)
                
        self.opt = nlopt.opt(self.algorithm, len(x))
        self.opt.set_lower_bounds(bounds_low)
        self.opt.set_upper_bounds(bounds_high)
        self.opt.set_min_objective(self.residuals)
        self.opt.set_xtol_rel(self.xtol_rel)
        self.opt.optimize(np.array(x))
        #minf = self.opt.last_optimum_value()
        self.thread_running = False
        
    def residuals(self, x, grad):
        if(self.abort_fit):
            self.opt.force_stop()
            self.thread_running = False
        i = 0
        for key in self.params:
            if self.params[key].vary:
                self.params[key].value = x[i]
                i += 1
        return self.fitfunc(self.params)    
    
    def is_working(self):
        return self.thread_running
        