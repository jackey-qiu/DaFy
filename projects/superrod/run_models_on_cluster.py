import numpy as np 
import sys, os
import copy 
try:
    from . import locate_path
except:
    import locate_path
script_path = locate_path.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'dump_files'))
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
sys.path.append(os.path.join(DaFy_path,'projects'))
import diffev
from diffev import fit_model_NLLS
from fom_funcs import *
import parameters
import data_superrod as data
import model
import solvergui

#uncomment the following lines if used with slurm bash script
'''
import ray
ray.shutdown()
diffev._cpu_count = int(sys.argv[2])
redis_password = sys.argv[1]
ray.init(address=os.environ["ip_head"], _redis_password=redis_password)
'''

#provide the folder where all model files (*.rod) are stored
folder_holding_model_files = os.path.join(DaFy_path,"examples/Cu100_CO2_EC/test_batch")
#folder_holding_model_files = "/User/cqiu/app/DaFy/examples/Cu100_CO2_EC/test_batch"
def obtain_rod_files(folder):
    '''
    load all rod files(*.rod) located in a selected folder
    '''
    files = []
    for file in os.listdir(folder):
        if file.endswith('.rod'):
            files.append(os.path.join(folder,file))
    return files

#partial set, add as many as you want
#key is the set funcs defined in /.../DaFy/EnginePool/diffev.py
#values are the associated value to be set
solver_settings = {
                   "set_pop_mult":False,
                   "set_pop_size":400,
                   "set_max_generations":200,
                   "set_autosave_interval":50
                  }


model = model.Model()
solver = solvergui.SolverController(model)

for each_file in obtain_rod_files(folder_holding_model_files):
    print(f"Loading file:{each_file}")
    model.load(each_file)
    model.apply_addition_to_optimizer(solver.optimizer)
    #set mask points
    for each in model.data_original:
        if not hasattr(each,'mask'):
            each.mask = np.array([True]*len(each.x))
    for each in model.data:
        if not hasattr(each,'mask'):
            each.mask = np.array([True]*len(each.x))
    #Update the following using solver_settings defined above
    for key, val in solver_settings.items():
        getattr(solver.optimizer,key)(val)
    #update mask info
    model.data = copy.deepcopy(model.data_original)
    [each.apply_mask() for each in model.data]
    #simulate the model first
    print("Simulating the model now ...")
    model.simulate()
    print("Start the fit...")
    solver.StartFit()
    
    
    
