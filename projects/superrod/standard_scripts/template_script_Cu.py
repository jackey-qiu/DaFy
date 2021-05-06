import os
import models.structure_tools.sxrd_dafy as model
from models.utils import UserVars
import numpy as np
import batchfile.locate_path as batch_path
import dump_files.locate_path as output_path
import models.domain_creator as domain_creator
from models.structure_tools import tool_box
from models.structure_tools import sorbate_tool_beta as sorbate_tool

model_type = 'ctr'
#--global settings--#
#/globalsetting/begin#
#/path/begin#
batch_path_head=os.path.join(batch_path.module_path_locator(),'Cu100')
output_file_path=output_path.module_path_locator()
#/path/end#
#/wavelength/begin#
wal=0.551
#/wavelength/end#
#/slabnumber/begin#
num_surface_slabs = 1
num_sorbate_slabs = 2
#/slabnumber/end#

#--set unitcell--#
#/unitcell/begin#
lat_pars = [3.615, 3.615, 3.615, 90, 90, 90]
unitcell = model.UnitCell(*lat_pars)
#/unitcell/end#

#/expconstant/begin#
re = 2.818e-5#electron radius
kvect=2*np.pi/wal#k vector
Egam = 6.626*(10**-34)*3*(10**8)/wal*10**10/1.602*10**19#energy in ev
LAM=1.5233e-22*Egam**6 - 1.2061e-17*Egam**5 + 2.5484e-13*Egam**4 + 1.6593e-10*Egam**3 + 1.9332e-06*Egam**2 + 1.1043e-02*Egam
exp_const = 4*kvect/LAM
auc=unitcell.a*unitcell.b*np.sin(unitcell.gamma)
#/expconstant/end#
#globalsetting/end#

#--set instrument--#
#/instrument/begin#
inst = model.Instrument(wavel = wal, alpha = 2.0)
#/instrument/end#

#--set bulk slab--#
#/bulk/begin#
bulk = model.Slab()
bulk_file = 'Cu100_bulk.str'
tool_box.add_atom_in_slab(bulk,os.path.join(batch_path_head,bulk_file))
#/bulk/end#

#--set surface slabs--#
#/surfaceslab/begin#
surface_slab_head = 'Cu100_surface_'
use_same_tag = 1
for i in range(num_surface_slabs):
    globals()['surface_{}'.format(i+1)] = model.Slab(c = 1.0)
    if use_same_tag == None:
        tag = i+1
    else:
        tag = use_same_tag
    tool_box.add_atom_in_slab(globals()['surface_{}'.format(i+1)],os.path.join(batch_path_head,'{}{}.str'.format(surface_slab_head, tag)))
#/surfaceslab/end#

#--set sorbate properties--#
#/sorbateproperties/begin#
sorbate_instance_head = 'CO2_'
sorbate_rgh_head = 'rgh_CO2_'
els_sorbate = [['O', 'C', 'O', 'O'],['O']]
anchor_index_list = [[1, None, 1, 1],[None]]
flat_down_index = [[],[]]
xyzu_oc_m = [[0.5, 0, 2, 0.1, 1, 1],[0.5,0,2.2,0.2,1,1]]
for i in range(num_sorbate_slabs):
    globals()['{}{}'.format(sorbate_instance_head, i+1)] = sorbate_tool.CarbonOxygenMotif.build_instance(xyzu_oc_m = xyzu_oc_m[i],els=els_sorbate[i],flat_down_index = flat_down_index[i],anchor_index_list=anchor_index_list[i],lat_pars = lat_pars)
    globals()['{}{}'.format(sorbate_instance_head, i+1)].set_coordinate_all_rgh()
    globals()['sorbate_{}'.format(i+1)] = globals()['{}{}'.format(sorbate_instance_head, i+1)].domain
    globals()['{}{}'.format(sorbate_rgh_head,i+1)] = globals()['{}{}'.format(sorbate_instance_head, i+1)].rgh
    globals()['atm_gp_sorbate_{}'.format(i+1)] = globals()['{}{}'.format(sorbate_instance_head, i+1)].make_atom_group()
#/sorbateproperties/end#

#/sorbatestructure/begin#
#OCCO
#       O1    O2
#        \   /
#        C1-C2
#===================
#/sorbatestructure/end#

#--set rgh--#
#/rgh/begin#

#/rgh/global/begin#
rgh = UserVars()
rgh.new_var('beta', 0.0)#roughness factor
rgh.new_var('mu',1)#liquid film thickness
scales=['scale_nonspecular_rods','scale_specular_rod']
for scale in scales:
    rgh.new_var(scale,0.0005)
#/rgh/global/end#

#/rgh/layer_water/begin#
rgh_lw = UserVars()
pars=['u0','ubar','d_w','first_layer_height',  'density_w']
rgh_lw.new_var('u0', 2)
rgh_lw.new_var('ubar', 2)
rgh_lw.new_var('first_layer_height', 2)
rgh_lw.new_var('d_w', 2)
rgh_lw.new_var('density_w', 0.033)
#/rgh/layer_water/end#

#/rgh/domain_weight/begin#
rgh_wt = UserVars()
for i in range(num_surface_slabs):
    rgh_wt.new_var('wt_domain{}'.format(i+1),1)
#/rgh/domain_weight/end#
#/rgh/end#

#/atmgroup/begin#
#/substrate/start#
n_substrate_layers = 2
map_layer_index = {1:'1st',2:'2nd',3:'3rd',4:'4th',5:'5th',6:'6th',7:'7th',8:'8th',9:'9th',10:'10th'}
map_layer_atoms = {1:['Cu1_1','Cu2_1'],\
                   2:['Cu1_2','Cu2_2'],\
                   3:['Cu1_3','Cu2_3']}
for i in range(num_surface_slabs):
    for j in range(n_substrate_layers):
        globals()['atm_gp_surface_{}_layer_{}'.format(map_layer_index[j+1], i+1)] = model.AtomGroup()
        for id in map_layer_atoms[j+1]:
            globals()['atm_gp_surface_{}_layer_{}'.format(map_layer_index[j+1], i+1)].add_atom(globals()['surface_{}'.format(i+1)],id)
#/substrate/end#
#/atmgroup/end#

#/sorbatesym/begin#
n_sym = 16
for i in range(num_sorbate_slabs):
    globals()['sorbate_syms_{}'.format(i+1)] =  [model.SymTrans([[1,0],[0,1]]), model.SymTrans([[1,0],[0,1]],t=[0.5,0.5]),\
                                                 model.SymTrans([[0,1],[1,0]]), model.SymTrans([[0,1],[1,0]],t=[0.5,0.5]), \
                                                 model.SymTrans([[0,1],[-1,0]]), model.SymTrans([[0,1],[-1,0]],t=[0.5,0.5]),\
                                                 model.SymTrans([[-1,0],[0,1]]),model.SymTrans([[-1,0],[0,1]],t=[0.5,0.5]),\
                                                 model.SymTrans([[-1,0],[0,-1]]),model.SymTrans([[-1,0],[0,-1]],t=[0.5,0.5]),\
                                                 model.SymTrans([[0,-1],[-1,0]]),model.SymTrans([[0,-1],[-1,0]],t=[0.5,0.5]),\
                                                 model.SymTrans([[0,-1],[1,0]]),model.SymTrans([[0,-1],[1,0]],t=[0.5,0.5]),\
                                                 model.SymTrans([[1,0],[0,-1]]),model.SymTrans([[1,0],[0,-1]],t=[0.5,0.5])][0:n_sym]
#/sorbatesym/end#

#/sample/begin#
domains = {}
for i in range(num_surface_slabs):
    domains['domain{}'.format(i+1)] = {}
    domains['domain{}'.format(i+1)]['slab'] = globals()['surface_{}'.format(i+1)]
    domains['domain{}'.format(i+1)]['sorbate'] = [globals()['domain_sorbate_{}'.format(j+1)] for j in range(num_sorbate_slabs)]
    domains['domain{}'.format(i+1)]['wt'] = getattr(globals()['rgh_wt'],'wt_domain{}'.format(i+1))
    domains['domain{}'.format(i+1)]['sorbate_sym'] = globals()['sorbate_syms_{}'.format(i+1)]
    domains['domain{}'.format(i+1)]['layered_water'] = rgh_lw
sample = model.Sample(inst, bulk, domains, unitcell)
#/sample/end#

def Sim(data,VARS=vars()):
    F =[]
    fom_scaler=[]
    beta=rgh.beta

#/update_sorbate/begin#
    for i in range(num_sorbate_slabs):
        VARS['{}{}'.format(VARS['sorbate_instance_head'],i+1)].set_coordinate_all_rgh()
#/update_sorbate/end#

    #normalize the domain weight to make total = 1
    wt_list = [getattr(rgh_wt, 'wt_domain{}'.format(i+1)) for i in range(num_surface_slabs)]
    total_wt = sum(wt_list)
    for i in range(num_surface_slabs):
        sample.domain['domain{}'.format(i+1)]['wt']=wt_list[i]/total_wt

    #faster solution(a factor of two faster than using loop)
    h_, k_, x_,LB_,dL_ = data.ctr_data_all[:,0], data.ctr_data_all[:,1], data.ctr_data_all[:,2],data.ctr_data_all[:,4],data.ctr_data_all[:,5]
    rough_ = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(x_-LB_)/dL_)**2)**0.5
    f_ = rough_*sample.calc_f_all(h_, k_, x_)
    F_ = abs(f_*f_)
    #you need to edit the list of extra scaling factor accordingly
    scaling_factors = [[rgh.scale_nonspecular_rods, rgh.scale_specular_rod][int(each=='specular_rod')] for each in data.scaling_tag]
    F = data.split_fullset(F_,scaling_factors)
    fom_scaler = [1]*len(F)

    panelty_factor = [sample.bond_distance_constraint(which_domain=i, max_distance =1) for i in range(num_surface_slabs)]
    return F,1+sum(panelty_factor),fom_scaler
