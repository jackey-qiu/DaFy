import os,sys
sys.path.append('P://apps//genx_pc_qiu')
sys.path.append('P://apps//genx_pc_qiu//models')
sys.path.append('P://apps//genx_pc_qiu//lib')

import model,diffev
import filehandling as io

path='P:\\My stuff\\Models\\CTR models\\Zr models for final publication\\refit results\\GenX'
file_path_raxr_mi=os.path.join(path,'Good_MI_RAXR_refit_Zr_100mM_NaCl_6O_run2_Apr3combined_ran.gx')
file_path_raxr_md=os.path.join(path,'MD_RAXR_refit_Zr_100mM_NaCl_6O_run2_Apr3_best_1_bin_R1_weighted_2_kr0.90_km0.90_pf0.80_run1_ran.gx')

mod_raxr_mi = model.Model()
config_raxr_mi = io.Config()
opt_raxr_mi = diffev.DiffEv()

mod_raxr_md = model.Model()
config_raxr_md = io.Config()
opt_raxr_md = diffev.DiffEv()

io.load_gx(file_path_raxr_mi,mod_raxr_mi,opt_raxr_mi,config_raxr_mi)
io.load_gx(file_path_raxr_md,mod_raxr_md,opt_raxr_md,config_raxr_md)

first_grid_mi,first_grid_md=None,None

for i in range(len(mod_raxr_mi.parameters.data)):
    if mod_raxr_mi.parameters.data[i][0]=='rgh_raxs.setA1':
        first_grid_mi=i
        break
    else:
        print mod_raxr_mi.parameters.data[i][0]
for i in range(len(mod_raxr_md.parameters.data)):
    if mod_raxr_md.parameters.data[i][0]=='rgh_raxs.setA1':
        first_grid_md=i
        break
    else:
        print mod_raxr_md.parameters.data[i][0]
print first_grid_mi,first_grid_md
for i in range(first_grid_md,len(mod_raxr_md.parameters.data)):
    for k in range(6):
        mod_raxr_md.parameters.set_value(i,k,mod_raxr_mi.parameters.get_value(first_grid_mi+i-first_grid_md,k))
io.save_gx(file_path_raxr_md.replace('.gx','_merged.gx'),mod_raxr_md,opt_raxr_md,config_raxr_md)
