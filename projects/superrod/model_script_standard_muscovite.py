import os
import models.structure_tools.sxrd_new1 as model
from models.utils import UserVars
import numpy as np
from numpy.linalg import inv
import batchfile.locate_path as batch_path
import dump_files.locate_path as output_path
import models.domain_creator as domain_creator
import models.domain_creator_sorbate as domain_creator_sorbate
from models.structure_tools import tool_box
from models.structure_tools import sorbate_tool
import accessory_functions.make_par_table.make_parameter_table_GenX_5_beta as make_grid

VERSION=1.2#version number to make easier code update to compatible with gx files based on old version scripts

#--global settings--#
#/globalsetting/begin#
#/fit_type/begin#
model_type = 'ctr'
raxr_fit_mode = 'MI'#'MI' for model-independent, and 'MD' for model-dependent
freeze=True#FREEZE=True will have resonant el make no influence on the non-resonant structure factor. And it will otherwise.
coherence = True
#/fit_type/end#
#/path/begin#
batch_path_head=os.path.join(batch_path.module_path_locator(),'Muscovite001')
output_file_path=output_path.module_path_locator()
#/path/end#
#/wavelength/begin#
wal=0.551
#/wavelength/end#

#/slab/begin#
num_surface_slabs = 1
height_offset = -2.6685
xy_offset = [0, 0]
group_scheme = [[1,0]]
#/slab/end#

#/raxr_setting/begin#
f1f2_file='Zr_K_edge.f1f2'
f1f2=np.loadtxt(os.path.join(batch_path_head,f1f2_file))#the energy column should NOT have duplicate values after rounding up to 0 digit. If so, cut off rows of duplicate energy!
raxr_el='Zr'#resonant element
E0=18007#Adsorption edge energy
number_raxs_spectra = 21#total number of RAXR spectra
electron_density_constraint = False
#/raxr_setting/end#

#--set unitcell--#
#/unitcell/begin#
c_lattice = 20.04156
lat_pars = [5.1988, 9.0266, c_lattice, 90, 95.782, 90]
unitcell = model.UnitCell(*lat_pars)
#/unitcell/end#

#/expconstant/begin#
L_max = 17.34
sig_eff=2*np.pi/(2*np.pi/(unitcell.c*np.sin(np.pi-unitcell.beta))*L_max)/5.66#intrinsic width (due to sig) with resolution width
re = 2.818e-5#electron radius
kvect=2*np.pi/wal#k vector
Egam = 6.626*(10**-34)*3*(10**8)/wal*10**10/1.602*10**19#energy in ev
LAM=1.5233e-22*Egam**6 - 1.2061e-17*Egam**5 + 2.5484e-13*Egam**4 + 1.6593e-10*Egam**3 + 1.9332e-06*Egam**2 + 1.1043e-02*Egam
exp_const = 4*kvect/LAM
auc=unitcell.a*unitcell.b*np.sin(unitcell.gamma)
#/expconstant/end#

#/coordinates_transformation/begin#
x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
basis=np.array([unitcell.a, unitcell.b, unitcell.c])
basis_Set=[[1,0,0],[0,1,0],[np.tan(unitcell.beta-np.pi/2.),0,1./np.cos(unitcell.beta-np.pi/2.)]]
T=inv(np.transpose(f1(x0_v,y0_v,z0_v,*basis_Set)))
T_INV=inv(T)
#/coordinates_transformation/end#

##<Setting electron density constraint>##
if electron_density_constraint:
    predefined_e_density=domain_creator.extract_e_density_info_muscovite(file=os.path.join(output_file_path,'temp_plot_eden'))
#globalsetting/end#

#--set instrument--#
#/instrument/begin#
inst = model.Instrument(wavel = wal, alpha = 2.0)
#/instrument/end#

#--set bulk slab--#
#/bulk/begin#
bulk = model.Slab()
bulk_file = 'muscovite_001_bulk_u_corrected_new.str'
tool_box.add_atom_in_slab(bulk,os.path.join(batch_path_head,bulk_file), height_offset = height_offset)
#/bulk/end#

#--set surface slabs--#
#/surfaceslab/begin#
surface_slab_head = 'muscovite_001_surface_AlSi_u_corrected_new_'
use_same_tag = 1
#only consider one domain for now
if num_surface_slabs != 1:
    num_surface_slabs = 1
for i in range(num_surface_slabs):
    globals()['surface_{}'.format(i+1)] = model.Slab(c = 1.0)
    if use_same_tag == None:
        tag = i+1
    else:
        tag = use_same_tag
    tool_box.add_atom_in_slab(slab = globals()['surface_{}'.format(i+1)],\
                              filename = os.path.join(batch_path_head,'{}{}.str'.format(surface_slab_head, tag)), \
                              attach = '_D{}'.format(tag),height_offset = height_offset)
#/surfaceslab/end#

#--set sorbate properties--#
#/sorbateproperties/begin#
##<Adding Gaussian peaks>##
#/gaussian_type1/begin#
number_gaussian_peak=6#how many gaussian peaks would you like to add?
el_gaussian_peak=['O']*number_gaussian_peak#the element of each gaussian peak
gaussian_shape = 'Flat'
surface_1, Gaussian_groups,Gaussian_group_names=domain_creator.add_gaussian(domain=surface_1,el=el_gaussian_peak,\
                                                                          number=number_gaussian_peak,\
                                                                          first_peak_height= 1 ,\
                                                                          spacing=2,\
                                                                          u_init=0.1,\
                                                                          occ_init=1,\
                                                                          height_offset=height_offset,\
                                                                          c=unitcell.c, \
                                                                          domain_tag='_D1',\
                                                                          shape=gaussian_shape,\
                                                                          gaussian_rms=2)
for i in range(len(Gaussian_groups)):
    vars()[Gaussian_group_names[i]]=Gaussian_groups[i]
rgh_gaussian=domain_creator.define_gaussian_vars(rgh=UserVars(),domain=surface_1,shape=gaussian_shape)
#/gaussian_type1/end#

##<Freeze Elements using adding_gaussian function>##
#/gaussian_type2/begin#
number_gaussian_peak_freeze=6#number of resonant element peaks
first_peak_height_freeze = 5
gaussian_shape_freeze = 'Flat'
surface_1, Gaussian_groups_freeze,Gaussian_group_names_freeze=domain_creator.add_gaussian(domain=surface_1,el=raxr_el,\
                                                                                        number=number_gaussian_peak_freeze,\
                                                                                        first_peak_height = 5,\
                                                                                        spacing = 2,\
                                                                                        u_init = 0.1,\
                                                                                        occ_init = 1,\
                                                                                        height_offset=height_offset,\
                                                                                        c=unitcell.c,domain_tag='_D1',\
                                                                                        shape=gaussian_shape_freeze,\
                                                                                        gaussian_rms = 2,\
                                                                                        freeze_tag=True)
for i in range(len(Gaussian_groups_freeze)):
    vars()[Gaussian_group_names_freeze[i]]=Gaussian_groups_freeze[i]
rgh_gaussian_freeze=domain_creator.define_gaussian_vars(rgh=UserVars(),domain=surface_1,shape=gaussian_shape_freeze,freeze_tag=True)
#/gaussian_type2/end#
#/sorbateproperties/end#

#--set rgh--#
#/rgh/begin#

#/rgh/global/begin#
rgh = UserVars()
rgh.new_var('beta', 0.0)#roughness factor
rgh.new_var('mu',1)#liquid film thickness
scales=['scale_nonspecular_rods','scale_specular_rod']
for scale in scales:
    rgh.new_var(scale,1.)
#/rgh/global/end#

#/rgh/domain_weight/begin#
rgh_wt = UserVars()
for i in range(num_surface_slabs):
    rgh_wt.new_var('wt_domain{}'.format(i+1),1)
#/rgh/domain_weight/end#
#/rgh/others/begin#
rgh_raxs=domain_creator.define_raxs_vars(rgh=UserVars(),number_spectra=number_raxs_spectra,number_domain=1)#RAXR spectra pars
rgh_dlw=domain_creator.define_diffused_layer_water_vars(rgh=UserVars())#Diffused Layered water pars
rgh_dls=domain_creator.define_diffused_layer_sorbate_vars(rgh=UserVars())#Diffused Layered sorbate pars
#rgh/others/end#
#/rgh/end#

#/atmgroup/begin#
#/substrate/start#
#group atom layers using methosd from Sang Soo Lee Matlab script
names,layer_groups=domain_creator.setup_atom_group_muscovite_3(domain=surface_1)
for i in range(len(layer_groups)):
    vars()[names[i]]=layer_groups[i]
#/substrate/end#
#/atmgroup/end#

##<make fit table file>##
if True:
    table_container=[]
    #global vars
    rgh_instance_list=[rgh]
    rgh_instance_name_list=['rgh']
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=os.path.join(batch_path_head,'pars_ranges.txt'))
    #surface relaxation
    rgh_instance_list=layer_groups
    rgh_instance_name_list=names
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=os.path.join(batch_path_head,'pars_ranges_surface_relaxation.txt'),sep=False)
    #interfacal structure
    rgh_instance_list=Gaussian_groups+[rgh_gaussian]+Gaussian_groups_freeze + [rgh_gaussian_freeze]
    rgh_instance_name_list=Gaussian_group_names + ['rgh_gaussian'] + Gaussian_group_names_freeze + ['rgh_gaussian_freeze']
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=os.path.join(batch_path_head,'pars_ranges.txt'))
    #diffuse layer structure(water and sorbate)
    rgh_instance_list=[rgh_dlw,rgh_dls]
    rgh_instance_name_list=['rgh_dlw','rgh_dls']
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=os.path.join(batch_path_head,'pars_ranges.txt'))

    #raxs pars
    a_range_raxr=[0,20]
    b_range_raxr=[-5,5]
    c_range_raxr=[0,10]
    A_range_raxr=[0,10]    
    table_container=make_grid.set_table_input_raxs(container=table_container,rgh_group_instance=rgh_raxs,rgh_group_instance_name='rgh_raxs',par_range={'a':a_range_raxr,'b':b_range_raxr,'c':c_range_raxr,'A':A_range_raxr,'P':[0,1]},number_spectra=number_raxs_spectra,number_domain=1)
    #build up the tab file
    make_grid.make_table(container=table_container,file_path=os.path.join(output_file_path,'par_table.tab'))

#/sample/begin#
##<format domains>##
domain={'domains':[surface_1],
        'layered_water_pars':rgh_dlw,
        'layered_sorbate_pars':rgh_dls,\
        'global_vars':rgh,
        'raxs_vars':rgh_raxs,
        'F1F2':f1f2,
        'E0':E0,
        'el':raxr_el,
        'freeze':freeze,
        'exp_factors':[exp_const,rgh.mu,re,auc,rgh.ra_conc],
        'sig_eff':sig_eff}
sample = model.Sample(inst, bulk, domain, unitcell,coherence=coherence,surface_parms={'delta1':0.,'delta2':0.})
#/sample/end#

def Sim(data,VARS=vars()):
    F =[]
    fom_scaler=[]
    beta=rgh.beta

    ##<update gaussian peaks>##
    if number_gaussian_peak>0:
        domain_creator.update_gaussian(domain=surface_1,rgh=rgh_gaussian,groups=Gaussian_groups,el=el_gaussian_peak,number=number_gaussian_peak,height_offset=height_offset,c=unitcell.c,domain_tag='_D1',shape=gaussian_shape,print_items=False,use_cumsum=True)
    if number_gaussian_peak_freeze>0:
        domain_creator.update_gaussian(domain=surface_1,rgh=rgh_gaussian_freeze,groups=Gaussian_groups_freeze,el = raxr_el,number=number_gaussian_peak_freeze,height_offset=height_offset,c=unitcell.c,domain_tag='_D1',shape=gaussian_shape_freeze,print_items=False,use_cumsum=True,freeze_tag=True)

    ##<link groups>##
    #[eval(each_command) for each_command in domain_creator.link_atom_group(gp_info=atom_group_info,gp_scheme=GROUP_SCHEME)]
    domain_creator.setup_atom_group_2(VARS)

    for data_set in data:
        f=np.array([])
        h = data_set.extra_data['h'][data_set.mask]
        k = data_set.extra_data['k'][data_set.mask]
        x = data_set.x[data_set.mask]
        y = data_set.extra_data['Y'][data_set.mask]
        LB = data_set.extra_data['LB'][data_set.mask]
        dL = data_set.extra_data['dL'][data_set.mask]

        if data_set.use:
            if x[0]>100:
                i+=1
                q=np.pi*2*unitcell.abs_hkl(h,k,y)
                rough = (1-rgh.beta)**2/(1+rgh.beta**2 - 2*rgh.beta*np.cos(q*unitcell.c*np.sin(np.pi-unitcell.beta)/2))
                pre_factor=np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            else:
                q=np.pi*2*unitcell.abs_hkl(h,k,x)
                rough = (1-rgh.beta)**2/(1+rgh.beta**2 - 2*rgh.beta*np.cos(q*unitcell.c*np.sin(np.pi-unitcell.beta)/2))
                pre_factor=np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            f=abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=raxr_fit_mode,height_offset=height_offset*unitcell.c,version=VERSION))
            F.append(3e6*pre_factor*rough*f*f)
            fom_scaler.append(1)
        else:
            if x[0]>100:
                i+=1
            f=np.zeros(len(y))
            F.append(f)
            fom_scaler.append(1)

    return F,1,fom_scaler