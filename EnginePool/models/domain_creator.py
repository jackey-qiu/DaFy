# -*- coding: utf-8 -*-
#import models.sxrd_test5_sym_new_test_new66_2_3 as model
from models.utils import UserVars
import models.sxrd_new1 as model
import numpy as np
import scipy.spatial as spatial
from operator import mul
import operator
import os
from numpy.linalg import inv
from copy import deepcopy
from random import uniform
from geometry_modules import *
from .domain_creator_water import domain_creator_water
from .domain_creator_sorbate import domain_creator_sorbate
from .domain_creator_surface import domain_creator_surface
#from best_fit_par_from_genx_to_rod import lib_creator,from_tab_to_par

"""functions in this class
create_grid_number, compare_grid, create_match_lib: supporting functions for the bv calculation
find_neighbors: find neighbors of a specific atoms within a range, retrun atm_ids and offset
cal_bond_valence1: cal bv of an atom and neiboring atoms within some range, will return a lib with keys of the id of neibor atoms and "total_valence"
cal_bond_valence2: cal bv based on a match_list of form like [['Fe1','Fe2'],['-x','+y']], which specifys the neibor atoms, will return a
                   bond_valence_container with keys from the first item of the list and "total" representing the total bv
cal_bond_valence3: cal bv based on match_lib of form like {'O1':[['Fe1','Fe2'],['-x','+y']]}, will return a lib with the same keys and the values of
                   each key the associated bv calculated

create_coor_transformation: create a spherical coor frame with three atms (z vec normal to the plane), return transformation matrix T (last col define the origin coordinates)
extract_spherical_pars: cal the r theta and phi in the spherical frame for a specific atom
set_sorbate_xyz: with known r theta and phi values, cal the associated xyz in the original coor frame and set the coor as new pst of an atom in the domain

scale_opt_batch:scale fitting parameters towards deeper layer, it takes a filename as argument
set_new_vars: set new variables in a sequence, like u_1,u_2,u_3
set_discrete_new_vars_batch:set discrete new variables, take a filename as argument
init_sim_batch: update the containor of free varibles, eg  u_containor=[u1,u2,u3,u4,u5], during fitting u will be updated, excute this function to update u_containor
"""

r0_Pb=2.04
x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])

#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])

#f2 calculate the distance b/ p1 and p2
f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

##make a pick index list specifying the type of full layer (0 for short and 1 for long slab)
##this function will return a list of indexes list to be passed to pick_full_layer
def make_pick_index(full_layer_pick,pick,half_layer_cases=8,full_layer_cases=8):
    pick_index_all=[]
    for i in range(len(full_layer_pick)):
        pick_index=[1]*full_layer_cases
        if full_layer_pick[i]!=None:
            #pick_index[pick[i][0]-half_layer_cases]=full_layer_pick[i]
            for j in pick[i]:
                pick_index[j-half_layer_cases]=full_layer_pick[i]
            pick_index_all.append(pick_index)
        else:
            pass
    return pick_index_all

def make_pick_index_half_layer(half_layer_pick,pick,half_layer_cases=8):
    pick_index_all=[]
    for i in range(len(half_layer_pick)):
        pick_index=[2]*half_layer_cases
        if half_layer_pick[i]!=None:
            #pick_index[pick[i][0]]=half_layer_pick[i]
            for j in pick[i]:
                pick_index[j]=half_layer_pick[i]
            pick_index_all.append(pick_index)
        else:
            pass
    return pick_index_all
##pick the full layer cases according to the type of full layers(pick_index is a list of list created from make_pick_index)
def pick_full_layer(LFL=[],SFL=[],pick_index=[]):
    FL_all=[]
    for pick in pick_index:
        FL=[]
        for i in range(len(pick)):
            if pick[i]==0:
                FL.append(SFL[i])
            elif pick[i]==1:
                FL.append(LFL[i])
        FL_all.append(FL)
    return FL_all

def pick_half_layer(LHL=[],SHL=[],pick_index=[]):
    HL_all=[]
    for pick in pick_index:
        HL=[]
        for i in range(len(pick)):
            if pick[i]==2:
                HL.append(SHL[i])
            elif pick[i]==3:
                HL.append(LHL[i])
        HL_all.append(HL)
    return HL_all
##pick functions
def pick(pick_list,pick_index):
    picked_box=[]
    for each in pick_index:
        picked_box.append(pick_list[each[0]])
    return picked_box

def pick_act(pick_list,pick_index):
    picked_box=[]
    for each in pick_index:
        temp_box=[]
        for i in range(len(each)):
            temp_box=temp_box+pick_list[each[i]]
        picked_box.append(temp_box)
    return picked_box

def deep_pick(pick_list,sym_site_index='sym_site_index',pick_index='pickup_index'):
    picked_box=[]
    for i in range(len(pick_index)):
        temp_box_j=[]
        for j in range(len(pick_index[i])):
            for k in sym_site_index[i][j]:
                temp_box_j.append(pick_list[pick_index[i][j]][k])
        picked_box.append(temp_box_j)
    return picked_box

#function to group the Fourier components (FC) from different domains in each RAXR spectra
#domain_index=[0,1] means setting the FC for domain2 (1+1) same as domain1 (0+1)
#domain_index=3 means setting the FC for domain2 and domain3 same as domain1, in this case the number indicate the number of total domains
def set_RAXR(domain_index=[],number_spectra=0):
    domains=None
    if type(domain_index)!=type([]):
        domains=range(domain_index)
    else:
        domains=domain_index
    for i in range(number_spectra):
        for j in domains[1:]:
            eval('rgh_raxr'+'.setA_D'+str(j+1)+'_'+str(i+1)+'(rgh_raxr'+'.getA_D'+str(domains[0]+1)+'_'+str(i+1)+'())')
            eval('rgh_raxr'+'.setP_D'+str(j+1)+'_'+str(i+1)+'(rgh_raxr'+'.getP_D'+str(domains[0]+1)+'_'+str(i+1)+'())')

#freeze A and B in the process of model fitting
def set_RAXR_AB(number_spectra=0):
    spectra=None
    if type(number_spectra)!=type([]):
        spectra=range(number_spectra)
    else:
        spectra=number_spectra
    for i in spectra:
        eval('rgh_raxr'+'.setA'+str(i+1)+'(1.)')
        eval('rgh_raxr'+'.setB'+str(i+1)+'(0.)')

#function to group outer-sphere pars from different domains (to be placed inside sim function)
def set_OS(domain_names=['domain5','domain4']):
    eval('rgh_'+domain_names[0]+'.setCt_offset_dx_OS(rgh_'+domain_names[1]+'.getCt_offset_dx_OS())')
    eval('rgh_'+domain_names[0]+'.setCt_offset_dy_OS(rgh_'+domain_names[1]+'.getCt_offset_dy_OS())')
    eval('rgh_'+domain_names[0]+'.setCt_offset_dz_OS(rgh_'+domain_names[1]+'.getCt_offset_dz_OS())')
    eval('rgh_'+domain_names[0]+'.setTop_angle_OS(rgh_'+domain_names[1]+'.getTop_angle_OS())')
    eval('rgh_'+domain_names[0]+'.setR0_OS(rgh_'+domain_names[1]+'.getR0_OS())')
    eval('rgh_'+domain_names[0]+'.setPhi_OS(rgh_'+domain_names[1]+'.getPhi_OS())')
#function to group dxdydz of sorbate atoms for two symmetry domains    
def make_commands_sorbates(sorbate='HO',sets=['set1','set1'],domain=['D3','D2']):
    commands=[]
    f1_dx='gp_{0}_{1}_{2}.set{3}(-gp_{0}_{5}_{4}.get{3}())'.format
    f1_dydz='gp_{0}_{1}_{2}.set{3}(gp_{0}_{5}_{4}.get{3}())'.format
    commands.append(f1_dx(sorbate,sets[0],domain[0],'dx',domain[1],sets[1]))
    commands.append(f1_dydz(sorbate,sets[0],domain[0],'dy',domain[1],sets[1]))
    commands.append(f1_dydz(sorbate,sets[0],domain[0],'dz',domain[1],sets[1]))
    return commands
#function to group bidentate pars from different domains (to be placed inside sim function)
def set_BD(domain_names=[2,1],sorbate_sets=1,distal_oxygen_number=1,sorbate='Pb'):
    for i in range(sorbate_sets):
        eval('rgh_domain'+str(domain_names[0]+1)+'.setOffset_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getOffset_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setOffset2_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getOffset2_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setAngle_offset_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getAngle_offset_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setR_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getR_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setPhi_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getPhi_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setTop_angle_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getTop_angle_BD_'+str(i*2)+'())')
        eval('gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc'+'(gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
        eval('gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu'+'(gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')
        for j in range(distal_oxygen_number):
            eval('gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc'+'(gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
            eval('gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu'+'(gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')

#function to group water pairs togeter from different domains
#domain_names is list of index of domains counting from 0 and number sets is the number of water pair counting from 1
def set_water_pair(domain_names=[3,2],number_sets=2):
    for i in range(number_sets):
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setdy(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getdy())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setV_shift_W_'+str(i+1)+'(rgh_domain'+str(domain_names[1]+1)+'.getV_shift_W_'+str(i+1)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setAlpha_W_'+str(i+1)+'(180-rgh_domain'+str(domain_names[1]+1)+'.getAlpha_W_'+str(i+1)+'())')

#function to group tridentate pars specifically for distal oxygens from different domains (to be placed inside sim function)
def set_TD(domain_names=['domain2','domain1']):
    eval('rgh_'+domain_names[0]+'.setTheta1_1_TD(rgh_'+domain_names[1]+'.getTheta1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setTheta1_2_TD(rgh_'+domain_names[1]+'.getTheta1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setTheta1_3_TD(rgh_'+domain_names[1]+'.getTheta1_3_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_1_TD(rgh_'+domain_names[1]+'.getPhi1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_2_TD(rgh_'+domain_names[1]+'.getPhi1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_3_TD(rgh_'+domain_names[1]+'.getPhi1_3_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_1_TD(rgh_'+domain_names[1]+'.getR1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_2_TD(rgh_'+domain_names[1]+'.getR1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_3_TD(rgh_'+domain_names[1]+'.getR1_3_TD())')

#function to group Hydrogen pars from the same domain (to be placed inside sim function)
def set_H(domain_name='domain1',tag=['W_1_2_1','W_1_1_1']):
    eval('rgh_'+domain_name+'.setPhi_H_'+tag[0]+'(180-rgh_'+domain_name+'.getPhi_H_'+tag[1]+'())')
    eval('rgh_'+domain_name+'.setR_H_'+tag[0]+'(rgh_'+domain_name+'.getR_H_'+tag[1]+'())')
    eval('rgh_'+domain_name+'.setTheta_H_'+tag[0]+'(rgh_'+domain_name+'.getTheta_H_'+tag[1]+'())')

#function to group distal oxygens based on adding in wild, N is the number of distal oxygens (to be placed inside sim function)
def set_distal_wild(domain_name=['domain2','domain1'],tag='BD',N=2):
    for i in range(N):
        eval('rgh_'+domain_name[0]+'.setPhi1_'+str(i)+'_'+tag+'(180-rgh_'+domain_name[1]+'.getPhi1_'+str(i)+'_'+tag+'())')
        eval('rgh_'+domain_name[0]+'.setR1_'+str(i)+'_'+tag+'(rgh_'+domain_name[1]+'.getR1_'+str(i)+'_'+tag+'())')
        eval('rgh_'+domain_name[0]+'.setTheta1_'+str(i)+'_'+tag+'(rgh_'+domain_name[1]+'.getTheta1_'+str(i)+'_'+tag+'())')


#calcualte the error for pb complex structure
def output_errors(edge_length=2.7,top_angle=70,error_top_angle=1,error_theta=1,error_delta1=0.02,error_delta2=0.03):
    sin_alpha_left=np.sin(np.deg2rad(top_anagle-error_top_angle)/2.)
    sin_alpha_right=np.sin(np.deg2rad(top_anagle+error_top_angle)/2.)
    tan_alpha_left=np.tan(np.deg2rad(top_anagle-error_top_angle)/2.)
    tan_alpha_right=np.tan(np.deg2rad(top_anagle+error_top_angle)/2.)
    print('error of PbO1 bond length:',edge_length/4.*(1./sin_alpha_left-1./sin_alpha_right)+error_delta1)
    print('error of pbO2 bond length:',edge_length/4.*(1./sin_alpha_left-1./sin_alpha_right))
    print('error of PbOdistal bond length:',edge_length/4.*(1./sin_alpha_left-1./sin_alpha_right)++error_delta2)
    print('error of O1PbO2 bond angle:',error_top_angle)
    print('error of O1PbOdistal and O2PbOdistal bond angle:',error_top_angle+error_theta)
    print('error of PbFe seperation:',edge_length/4.*(1./tan_alpha_left-1./tan_alpha_right))
    return None

#make dummy data set for test purpose, this function should be inside sim function
#data is data I is the F (list of calculated structure factors)
def make_dummy_data(file='D://temp_dummy_data.dat',data=None,I=None):
    data_full=np.zeros((0,8))
    for i in range(len(data)):
        data_set=data[i]
        x = data_set.x[:,np.newaxis]
        h = data_set.extra_data['h'][:,np.newaxis]
        k = data_set.extra_data['k'][:,np.newaxis]
        intensity = np.array(I[i])[:,np.newaxis]
        eI= data_set.error[:,np.newaxis]
        y = data_set.extra_data['Y'][:,np.newaxis]
        LB = data_set.extra_data['LB'][:,np.newaxis]
        dL = data_set.extra_data['dL'][:,np.newaxis]
        temp_set=np.concatenate((x,h,k,y,intensity,eI,LB,dL),axis=1)
        data_full=np.concatenate((data_full,temp_set),axis=0)
    np.savetxt(file,data_full,fmt='%.5e')
    return None

def combine_all_datasets(file='D://temp_full_dataset.dat',data=None):
    data_full=np.zeros((0,8))
    for i in range(len(data)):
        data_set=data[i]
        x = data_set.x[:,np.newaxis]
        h = data_set.extra_data['h'][:,np.newaxis]
        k = data_set.extra_data['k'][:,np.newaxis]
        intensity = data_set.y[:,np.newaxis]
        eI= data_set.error[:,np.newaxis]
        y = data_set.extra_data['Y'][:,np.newaxis]
        LB = data_set.extra_data['LB'][:,np.newaxis]
        dL = data_set.extra_data['dL'][:,np.newaxis]
        temp_set=np.concatenate((x,h,k,y,intensity,eI,LB,dL),axis=1)
        data_full=np.concatenate((data_full,temp_set),axis=0)
    np.savetxt(file,data_full,fmt='%.5e')
    return None

def define_global_vars(rgh,domain_number=2):
    rgh.new_var('beta',0)
    rgh.new_var('mu',10)#water thickness
    rgh.new_var('ra_conc',0.0)#resonant el concentration in M
    for i in range(domain_number):
        rgh.new_var('wt'+str(i+1),1)
    return rgh

def define_raxs_vars(rgh,number_spectra=0,number_domain=2):
    for i in range(number_spectra):
        rgh.new_var('a'+str(i+1),1.0)
        rgh.new_var('b'+str(i+1),0.0)
        rgh.new_var('c'+str(i+1),0.01)
        for j in range(number_domain):
            rgh.new_var('A'+str(i+1)+'_D'+str(j+1),0.0)
            rgh.new_var('P'+str(i+1)+'_D'+str(j+1),0.0)
    return rgh

def define_diffused_layer_water_vars(rgh):
    rgh.new_var('u0_w',0.4)
    rgh.new_var('ubar_w',0.4)
    rgh.new_var('first_layer_height_w',2.0)#relative height in A
    rgh.new_var('d_w',1.9)#inter-layer water seperation in A
    rgh.new_var('density_w',0.033)#number density in unit of # of waters per cubic A(0.033 is the typical value)
    return rgh

def define_diffused_layer_sorbate_vars_original(rgh):
    rgh.new_var('u0_s',0.4)
    rgh.new_var('ubar_s',0.4)
    rgh.new_var('first_layer_height_s',2.0)#relative height in A
    rgh.new_var('d_s',1.9)#inter-layer water seperation in A
    rgh.new_var('density_s',0.033)#number density in unit of # of waters per cubic A(0.033 is the typical value)
    return rgh

def define_diffused_layer_sorbate_vars(rgh):
    rgh.new_var('u0_s',0.4)
    rgh.new_var('ubar_s',0.4)
    rgh.new_var('first_layer_height_s',2.0)#relative height in A
    rgh.new_var('d_s',1.9)#inter-layer water seperation in A
    rgh.new_var('density_s',0.033)#number density in unit of # of waters per cubic A(0.033 is the typical value)
    rgh.new_var('oc_damping_factor',1.0)#high value the occupancy damping quickly
    return rgh

def extract_e_density_info_muscovite(file=''):
    pass
def setup_atom_group_muscovite(domain=[],group_number=5):
    ref_id_list_Al=[['O4_3_0','O4_4_0'],['O3_3_0','O3_4_0'],['O5_3_0','O5_4_0'],['Al1_3_0','Al1_4_0'],['Al2_3_0','Al2_4_0'],['O1_3_0','O1_4_0'],['O2_3_0','O2_4_0'],['O6_3_0','O6_4_0'],\
                 ['Al3_3_0','Al3_4_0'],['Al3_5_0','Al3_6_0'],['O6_5_0','O6_6_0'],['O2_5_0','O2_6_0'],['O1_5_0','O1_6_0'],['Al2_5_0','Al2_6_0'],['Al1_5_0','Al1_6_0'],['O5_5_0','O5_6_0'],['O4_5_0','O4_6_0'],['O3_5_0','O3_6_0']]
    ref_id_list_Si=[['O4_3_0','O4_4_0'],['O3_3_0','O3_4_0'],['O5_3_0','O5_4_0'],['Si1_3_0','Si1_4_0'],['Si2_3_0','Si2_4_0'],['O1_3_0','O1_4_0'],['O2_3_0','O2_4_0'],['O6_3_0','O6_4_0'],\
                 ['Al3_3_0','Al3_4_0'],['Al3_5_0','Al3_6_0'],['O6_5_0','O6_6_0'],['O2_5_0','O2_6_0'],['O1_5_0','O1_6_0'],['Si2_5_0','Si2_6_0'],['Si1_5_0','Si1_6_0'],['O5_5_0','O5_6_0'],['O4_5_0','O4_6_0'],['O3_5_0','O3_6_0']]
    ref_sym_list=[[[1,0,0,0,1,0,0,0,1]]*2]*18
    ref_group_names_Al=['gp_O4_O4O3','gp_O3_O4O3','gp_O5_O3O4','gp_Al1_Al4Al3','gp_Al2_Al3Al4','gp_O1_O4O3','gp_O2_O3O4','gp_O6_O3O4','gp_Al3_Al4Al3','gp_Al3_Al6Al5','gp_O6_O5O6','gp_O2_O5O6','gp_O1_O6O5','gp_Al2_Al5Al6','gp_Al1_Al6Al5','gp_O5_O5O6','gp_O4_O6O5','gp_O3_O6O5']
    ref_group_names_Si=['gp_O4_O4O3','gp_O3_O4O3','gp_O5_O3O4','gp_Si1_Al4Al3','gp_Si2_Al3Al4','gp_O1_O4O3','gp_O2_O3O4','gp_O6_O3O4','gp_Al3_Al4Al3','gp_Al3_Al6Al5','gp_O6_O5O6','gp_O2_O5O6','gp_O1_O6O5','gp_Si2_Al5Al6','gp_Si1_Al6Al5','gp_O5_O5O6','gp_O4_O6O5','gp_O3_O6O5']
    Domain1,Domain2=domain
    gp_info=[{'domain':Domain1,'ref_id_list':ref_id_list_Al[0:group_number],'ref_group_names':ref_group_names_Al[0:group_number],'ref_sym_list':ref_sym_list[0:group_number],'domain_tag':'_D1'},
                     {'domain':Domain2,'ref_id_list':ref_id_list_Si[0:group_number],'ref_group_names':ref_group_names_Si[0:group_number],'ref_sym_list':ref_sym_list[0:group_number],'domain_tag':'_D2'}]
    groups,group_names=[],[]
    for i in range(len(gp_info)):
        domain=gp_info[i]['domain']
        tag=gp_info[i]['domain_tag']
        for j in range(len(gp_info[i]['ref_id_list'])):
            temp_atom_group=model.AtomGroup()
            for k in range(len(gp_info[i]['ref_id_list'][j])):
                if gp_info[i]['ref_sym_list']!=[]:
                    temp_atom_group.add_atom(slab=gp_info[i]['domain'],id=gp_info[i]['ref_id_list'][j][k]+tag, matrix=gp_info[i]['ref_sym_list'][j][k])
                else:
                    temp_atom_group.add_atom(slab=gp_info[i]['domain'],id=gp_info[i]['ref_id_list'][j][k]+tag, matrix=np.array([1,0,0,0,1,0,0,0,1]))
            groups.append(temp_atom_group)
            group_names.append(gp_info[i]['ref_group_names'][j]+tag)
    return groups,group_names,gp_info

def setup_atom_group_muscovite_2(domain=[]):
    Domain1,Domain2=domain
    ref_id_list_Al_top=[['O4_3_0','O4_4_0','O3_3_0','O3_4_0','O5_3_0','O5_4_0'],\
                        ['Al1_3_0','Al1_4_0','Al2_3_0','Al2_4_0'],\
                        ['O1_3_0','O1_4_0','O2_3_0','O2_4_0','O6_3_0','O6_4_0'],\
                        ['Al3_3_0','Al3_4_0','Al3_5_0','Al3_6_0'],\
                        ['O6_5_0','O6_6_0','O2_5_0','O2_6_0','O1_5_0','O1_6_0'],\
                        ['Al2_5_0','Al2_6_0','Al1_5_0','Al1_6_0'],\
                        ['O5_5_0','O5_6_0','O4_5_0','O4_6_0','O3_5_0','O3_6_0'],\
                        ['K_3_0','K_4_0']]

    ref_id_list_Al_mid=[['O4_7_0','O4_8_0','O3_7_0','O3_8_0','O5_8_0','O5_7_0'],\
                        ['Al1_7_0','Al1_8_0','Al2_8_0','Al2_7_0'],\
                        ['O1_7_0','O1_8_0','O2_8_0','O2_7_0','O6_8_0','O6_7_0'],\
                        ['Al3_7_0','Al3_8_0','Al3_1_1','Al3_2_1'],\
                        ['O6_2_1','O6_1_1','O2_2_1','O2_1_1','O1_1_1','O1_2_1'],\
                        ['Al2_2_1','Al2_1_1','Al1_1_1','Al1_2_1'],\
                        ['O5_2_1','O5_1_1','O4_1_1','O4_2_1','O3_1_1','O3_2_1'],\
                        ['K_2_1','K_1_1']]

    ref_id_list_Al_bot=[['O4_3_1','O4_4_1','O3_3_1','O3_4_1','O5_3_1','O5_4_1'],\
                        ['Al1_3_1','Al1_4_1','Al2_3_1','Al2_4_1'],\
                        ['O1_3_1','O1_4_1','O2_3_1','O2_4_1','O6_3_1','O6_4_1'],\
                        ['Al3_3_1','Al3_4_1','Al3_5_1','Al3_6_1'],\
                        ['O6_5_1','O6_6_1','O2_5_1','O2_6_1','O1_5_1','O1_6_1'],\
                        ['Al2_5_1','Al2_6_1','Al1_5_1','Al1_6_1'],\
                        ['O5_5_1','O5_6_1','O4_5_1','O4_6_1','O3_5_1','O3_6_1'],\
                        ['K_3_1','K_4_1']]

    ref_id_list_Si_top=[['O4_3_0','O4_4_0','O3_3_0','O3_4_0','O5_3_0','O5_4_0'],\
                        ['Si1_3_0','Si1_4_0','Si2_3_0','Si2_4_0'],\
                        ['O1_3_0','O1_4_0','O2_3_0','O2_4_0','O6_3_0','O6_4_0'],\
                        ['Al3_3_0','Al3_4_0','Al3_5_0','Al3_6_0'],\
                        ['O6_5_0','O6_6_0','O2_5_0','O2_6_0','O1_5_0','O1_6_0'],\
                        ['Si2_5_0','Si2_6_0','Si1_5_0','Si1_6_0'],\
                        ['O5_5_0','O5_6_0','O4_5_0','O4_6_0','O3_5_0','O3_6_0'],\
                        ['K_3_0','K_4_0']]

    ref_id_list_Si_mid=[['O4_7_0','O4_8_0','O3_7_0','O3_8_0','O5_8_0','O5_7_0'],\
                        ['Si1_7_0','Si1_8_0','Si2_8_0','Si2_7_0'],\
                        ['O1_7_0','O1_8_0','O2_8_0','O2_7_0','O6_8_0','O6_7_0'],\
                        ['Al3_7_0','Al3_8_0','Al3_1_1','Al3_2_1'],\
                        ['O6_2_1','O6_1_1','O2_2_1','O2_1_1','O1_1_1','O1_2_1'],\
                        ['Si2_2_1','Si2_1_1','Si1_1_1','Si1_2_1'],\
                        ['O5_2_1','O5_1_1','O4_1_1','O4_2_1','O3_1_1','O3_2_1'],\
                        ['K_2_1','K_1_1']]

    ref_id_list_Si_bot=[['O4_3_1','O4_4_1','O3_3_1','O3_4_1','O5_3_1','O5_4_1'],\
                        ['Si1_3_1','Si1_4_1','Si2_3_1','Si2_4_1'],\
                        ['O1_3_1','O1_4_1','O2_3_1','O2_4_1','O6_3_1','O6_4_1'],\
                        ['Al3_3_1','Al3_4_1','Al3_5_1','Al3_6_1'],\
                        ['O6_5_1','O6_6_1','O2_5_1','O2_6_1','O1_5_1','O1_6_1'],\
                        ['Si2_5_1','Si2_6_1','Si1_5_1','Si1_6_1'],\
                        ['O5_5_1','O5_6_1','O4_5_1','O4_6_1','O3_5_1','O3_6_1'],\
                        ['K_3_1','K_4_1']]
    ref_id_list_Si=[ref_id_list_Si_top,ref_id_list_Si_mid,ref_id_list_Si_bot]
    ref_id_list_Al=[ref_id_list_Al_top,ref_id_list_Al_mid,ref_id_list_Al_bot]

    ref_group_names=['gp_Otop','gp_AlSi_1','gp_Octt','gp_AlAl_2','gp_Octa','gp_AlSi_3','gp_Obas','gp_K']
    atm_groups=[]
    atm_group_names=[]
    for k in range(3):
        layer_tag='_layer_'+str(k+1)
        temp_groups=[]
        ref_ids_Si=ref_id_list_Si[k]
        ref_ids_Al=ref_id_list_Al[k]
        for i in range(len(ref_group_names)):
            group_name=ref_group_names[i]+layer_tag
            atm_group_names.append(group_name)
            for j in range(len(ref_ids_Si[i])):
                temp_atom_group=model.AtomGroup()
                ids_Si,ids_Al=ref_ids_Si[i],ref_ids_Al[i]
                for m in range(len(ids_Si)):
                    temp_atom_group.add_atom(slab=Domain1,id=ids_Al[m]+'_D1', matrix=[1,0,0,0,1,0,0,0,1])
                for m in range(len(ids_Si)):
                    temp_atom_group.add_atom(slab=Domain2,id=ids_Si[m]+'_D2', matrix=[1,0,0,0,1,0,0,0,1])
            atm_groups.append(temp_atom_group)
    return atm_group_names,atm_groups

def setup_atom_group_2(vars):#group Al and Si layer relaxation to the associated coordianted oxygens based on Sang Soo Lee's Matlab script
    vars['gp_AlSi_1_layer_1'].setdz(0.75*vars['gp_Otop_layer_1'].getdz()+0.25*vars['gp_Octt_layer_1'].getdz())
    vars['gp_AlAl_2_layer_1'].setdz(0.5*vars['gp_Octt_layer_1'].getdz()+0.5*vars['gp_Octa_layer_1'].getdz())
    vars['gp_AlSi_3_layer_1'].setdz(0.25*vars['gp_Octa_layer_1'].getdz()+0.75*vars['gp_Obas_layer_1'].getdz())

    vars['gp_AlSi_1_layer_2'].setdz(0.75*vars['gp_Otop_layer_2'].getdz()+0.25*vars['gp_Octt_layer_2'].getdz())
    vars['gp_AlAl_2_layer_2'].setdz(0.5*vars['gp_Octt_layer_2'].getdz()+0.5*vars['gp_Octa_layer_2'].getdz())
    vars['gp_AlSi_3_layer_2'].setdz(0.25*vars['gp_Octa_layer_2'].getdz()+0.75*vars['gp_Obas_layer_2'].getdz())

    vars['gp_AlSi_1_layer_3'].setdz(0.75*vars['gp_Otop_layer_3'].getdz()+0.25*vars['gp_Octt_layer_3'].getdz())
    vars['gp_AlAl_2_layer_3'].setdz(0.5*vars['gp_Octt_layer_3'].getdz()+0.5*vars['gp_Octa_layer_3'].getdz())
    vars['gp_AlSi_3_layer_3'].setdz(0.25*vars['gp_Octa_layer_3'].getdz()+0.75*vars['gp_Obas_layer_3'].getdz())

def setup_atom_group(gp_info=[]):
    groups,group_names=[],[]
    for i in range(len(gp_info)):
        domain=gp_info[i]['domain']
        tag=gp_info[i]['domain_tag']
        for j in range(len(gp_info[i]['ref_id_list'])):
            temp_atom_group=model.AtomGroup()
            for k in range(len(gp_info[i]['ref_id_list'][j])):
                if gp_info[i]['ref_sym_list']!=[]:
                    temp_atom_group.add_atom(slab=gp_info[i]['domain'],id=gp_info[i]['ref_id_list'][j][k]+tag, matrix=gp_info[i]['ref_sym_list'][j][k])
                else:
                    temp_atom_group.add_atom(slab=gp_info[i]['domain'],id=gp_info[i]['ref_id_list'][j][k]+tag, matrix=np.array([1,0,0,0,1,0,0,0,1]))
            groups.append(temp_atom_group)
            group_names.append(gp_info[i]['ref_group_names'][j]+tag)
    return groups,group_names

def link_atom_group(gp_info=[],gp_scheme=[]):
    command_list=[]
    for each_link in gp_scheme:
        group1=gp_info[each_link[0]]
        group2=gp_info[each_link[1]]
        ref_group_name1=list(map(lambda x:x+group1['domain_tag'],group1['ref_group_names']))
        ref_group_name2=list(map(lambda x:x+group2['domain_tag'],group2['ref_group_names']))
        for each_name in ref_group_name1:
            command_list.append(each_name+('.setdx(%s'%ref_group_name2[ref_group_name1.index(each_name)])+'.getdx())')
            command_list.append(each_name+('.setdy(%s'%ref_group_name2[ref_group_name1.index(each_name)])+'.getdy())')
            command_list.append(each_name+('.setdz(%s'%ref_group_name2[ref_group_name1.index(each_name)])+'.getdz())')
            command_list.append(each_name+('.setu(%s'%ref_group_name2[ref_group_name1.index(each_name)])+'.getu())')
            command_list.append(each_name+('.setoc(%s'%ref_group_name2[ref_group_name1.index(each_name)])+'.getoc())')
    return command_list

def generate_sorbate_ids_original(domain,sorbate_layers,sorbate_el,number_sorbate_atom=1):#number_sorbate_atom=1 if monomer, 2 if dimmer and so on
    id_container=[]
    id_names=[]
    for i in range(sorbate_layers):
        tag=[sorbate_el+str(i*2*number_sorbate_atom+1+j) for j in range(number_sorbate_atom*2)]
        id_container.append([id for id in domain.id if sum(list(map(lambda x:x in id,tag)))])
        id_container.append([id for id in domain.id if sum(list(map(lambda x:x in id,tag))) and ('O' not in id)])
        id_container.append([id for id in domain.id if sum(list(map(lambda x:x in id,tag))) and ('O' in id)])
        id_names=id_names+['sorbate_set'+str(i+1)+'_D1',sorbate_el+'_set'+str(i+1)+'_D1','HO_set'+str(i+1)+'_D1']
    return id_container,id_names

def generate_sorbate_ids(domain,sorbate_layers,sorbate_el,number_sorbate_atom=1,symmetry=True,level=[]):#number_sorbate_atom=1 if monomer, 2 if dimmer and so on
    id_container=[]
    id_names=[]
    sym_scale=2
    if not symmetry:
        sym_scale=1
    for i in range(sorbate_layers):
        tag=[sorbate_el+str(i*sym_scale*number_sorbate_atom+1+j) for j in range((number_sorbate_atom-len(level)*2)*sym_scale)]+list(map(lambda x:sorbate_el+str(x)+'rA',level))+list(map(lambda x:sorbate_el+str(x)+'rB',level))

        id_container.append([id for id in domain.id if sum(list(map(lambda x:x in id,tag)))])
        id_container.append([id for id in domain.id if sum(list(map(lambda x:x in id,tag))) and ('O' not in id)])
        id_container.append([id for id in domain.id if sum(list(map(lambda x:x in id,tag))) and ('O' in id)])
        id_names=id_names+['sorbate_set'+str(i+1)+'_D1',sorbate_el+'_set'+str(i+1)+'_D1','HO_set'+str(i+1)+'_D1']
    return id_container,id_names

def add_sorbate(domain,anchored_atoms,func,geo_lib,info_lib,domain_tag,rgh,index_offset=[0,1],height_offset=0,level=None,symmetry_couple=True,cap=[],attach_sorbate_number=[],first_or_second=[],mirror=[]):
    domain=func([0,0,2.0+height_offset],domain,anchored_atoms,geo_lib,info_lib,domain_tag,index_offset=index_offset[0],level=level,cap=cap,attach_sorbate_number=attach_sorbate_number,first_or_second=first_or_second,mirror=mirror)
    if symmetry_couple:
        domain=func([0.5,0.5,2.0+height_offset],domain,anchored_atoms,geo_lib,info_lib,domain_tag,index_offset=index_offset[1],level=level,cap=cap,attach_sorbate_number=attach_sorbate_number,first_or_second=first_or_second,mirror=mirror)
    for key in geo_lib.keys():
        rgh.new_var(key,geo_lib[key])
    return domain,rgh

def add_sorbate_new(domain,anchored_atoms,func,geo_lib,info_lib,domain_tag,rgh,index_offset=[0,1],xy_offset=[0,0],height_offset=0,symmetry_couple=True,**args):
    domain=func([0+xy_offset[0],0+xy_offset[1],2.0+height_offset],domain,anchored_atoms,geo_lib,info_lib,domain_tag,index_offset=index_offset[0],**args)
    if symmetry_couple:
        domain=func([0.5+xy_offset[0],0.5+xy_offset[1],2.0+height_offset],domain,anchored_atoms,geo_lib,info_lib,domain_tag,index_offset=index_offset[1],**args)
    for key in geo_lib.keys():
        rgh.new_var(key,geo_lib[key])
    return domain,rgh

def update_sorbate(domain,anchored_atoms,func,info_lib,domain_tag,rgh,index_offset=[0,1],height_offset=0,level=None,symmetry_couple=True,cap=[],attach_sorbate_number=[],first_or_second=[],mirror=[]):
    domain=func([0,0,2.0+height_offset],domain,anchored_atoms,vars(rgh),info_lib,domain_tag,index_offset=index_offset[0],level=level,cap=cap,attach_sorbate_number=attach_sorbate_number,first_or_second=first_or_second,mirror=mirror)
    if symmetry_couple:
        domain=func([0.5,0.5,2.0+height_offset],domain,anchored_atoms,vars(rgh),info_lib,domain_tag,index_offset=index_offset[1],level=level,cap=cap,attach_sorbate_number=attach_sorbate_number,first_or_second=first_or_second,mirror=mirror)
    return domain

def update_sorbate_new(domain,anchored_atoms,func,info_lib,domain_tag,rgh,index_offset=[0,1],xy_offset=[0,0],height_offset=0,level=None,symmetry_couple=True,**args):
    domain=func([0+xy_offset[0],0+xy_offset[1],2.0+height_offset],domain,anchored_atoms,vars(rgh),info_lib,domain_tag,index_offset=index_offset[0],**args)
    if symmetry_couple:
        domain=func([0.5+xy_offset[0],0.5+xy_offset[1],2.0+height_offset],domain,anchored_atoms,vars(rgh),info_lib,domain_tag,index_offset=index_offset[1],**args)
    return domain

def add_gaussian_old(domain,el='O',number=3,first_peak_height=2,spacing=2,u_init=0.008,occ_init=1,height_offset=0,c=20.1058,domain_tag='_D1'):
    height_list=1.6685+height_offset+np.array([spacing/c*i+first_peak_height/c for i in range(number)])
    group_names=['Gaussian_'+el+'_'+str(i+1)+domain_tag for i in range(number)]
    groups=[]
    for i in range(number):
        groups.append(domain.add_atom(id='Gaussian_'+el+'_'+str(i+1)+domain_tag, element=el, x=0.5, y=0.5, z=height_list[i], u = u_init, oc = occ_init, m = 1.0))
    return domain,groups,group_names

def add_gaussian(domain,el='O',number=3,first_peak_height=2,spacing=10,u_init=0.008,occ_init=1,height_offset=0,c=20.1058,domain_tag='_D1',shape='Flat',gaussian_rms=2,freeze_tag=False):
    '''
    If shape is Flat then those gaussian peaks are evenly spaced with equivalent occ,
    If shape is Single_Gaussian then those gaussian peaks are evenly spaced with occs in a Gaussian distribution, determined by the spacing and gaussian_rms
    Note all those items about length are in unit of A
    The freeze_tag=True will change the group and id names whith a header of 'Freezed_el' from 'Gaussian_'
    '''
    #height_list=[]
    #oc_list=[]
    if type(el)!=type([]):
        if type(number)==type([]):#for type of Double_Gaussian
            el=[el]*number[0]+[el]*number[1]
        else:
            el=[el]*number

    if shape=='Flat':
        height_list=1.6685+height_offset+np.array([spacing/c*i+first_peak_height/c for i in range(number)])
        oc_list=[occ_init]*number
    elif shape=='Single_Gaussian':
        center=1.6685+height_offset+first_peak_height/c+spacing/c/2
        delta_z=spacing/c/float(number-1)
        peaks_left=[center]+[center-(i+1)*delta_z for i in range((number-1)/2)]
        peaks_right=[center+(i+1)*delta_z for i in range((number-1)/2)]
        height_list=peaks_left+peaks_right
        height_list.sort()
        oc_list=occ_init*np.exp(-0.5*gaussian_rms**-2*(np.array(height_list)*c-center*c)**2)
        #print spacing,number,delta_z
        #print height_list
    elif shape=='Double_Gaussian':#make sure the number of each Gaussian peak cluster is an odd number
        #peak one
        center=1.6685+height_offset+first_peak_height[0]/c+spacing[0]/c/2
        delta_z=spacing[0]/c/float(number[0]-1)
        peaks_left=[center]+[center-(i+1)*delta_z for i in range((number[0]-1)/2)]
        peaks_right=[center+(i+1)*delta_z for i in range((number[0]-1)/2)]
        height_list=peaks_left+peaks_right
        height_list.sort()
        oc_list=occ_init[0]*np.exp(-0.5*gaussian_rms[0]**-2*(np.array(height_list)*c-center*c)**2)
        #peak two
        center2=1.6685+height_offset+first_peak_height[0]/c+spacing[0]/c+first_peak_height[1]/c+spacing[1]/c/2
        delta_z2=spacing[1]/c/float(number[1]-1)
        peaks_left2=[center2]+[center2-(i+1)*delta_z2 for i in range((number[1]-1)/2)]
        peaks_right2=[center2+(i+1)*delta_z2 for i in range((number[1]-1)/2)]
        height_list2=peaks_left2+peaks_right2
        height_list2.sort()
        oc_list2=occ_init[1]*np.exp(-0.5*gaussian_rms[1]**-2*(np.array(height_list2)*c-center2*c)**2)
    group_names1=[]
    groups1=[]
    group_names2=[]
    groups2=[]
    if shape=='Single_Gaussian':
        name_header=None
        if freeze_tag:
            name_header='Freezed_el_set1_'
        else:
            name_header='Gaussian_set1_'
        group_names1=[name_header+el[i]+'_'+str(i+1)+domain_tag for i in range(number)]
        for i in range(number):
            try:
                groups1.append(domain.add_atom(id=name_header+el[i]+'_'+str(i+1)+domain_tag, element=el[i], x=0.5, y=0.5, z=height_list[i], u = u_init, oc = oc_list[i], m = 1.0))
            except:
                id=name_header+el[i]+'_'+str(i+1)+domain_tag
                index=list(domain.id).index(id)
                domain.z[index]=height_list[i]
                domain.oc[index]=oc_list[i]
    elif shape=='Double_Gaussian':
        name_header=None
        if freeze_tag:
            name_header='Freezed_el_'
        else:
            name_header='Gaussian_'
        group_names1=[name_header+'set1_'+el[i]+'_'+str(i+1)+domain_tag for i in range(number[0])]
        group_names2=[name_header+'set2_'+el[i]+'_'+str(i+1)+domain_tag for i in range(number[1])]
        for i in range(number[0]):
            try:
                groups1.append(domain.add_atom(id=name_header+'set1_'+el[i]+'_'+str(i+1)+domain_tag, element=el[i], x=0.5, y=0.5, z=height_list[i], u = u_init[0], oc = oc_list[i], m = 1.0))
            except:
                id=name_header+'set1_'+el[i]+'_'+str(i+1)+domain_tag
                index=list(domain.id).index(id)
                domain.z[index]=height_list[i]
                domain.oc[index]=oc_list[i]
        for i in range(number[1]):
            try:
                groups2.append(domain.add_atom(id=name_header+'set2_'+el[i]+'_'+str(i+1)+domain_tag, element=el[i], x=0.5, y=0.5, z=height_list2[i], u = u_init[1], oc = oc_list2[i], m = 1.0))
            except:
                id=name_header+'set2_'+el[i]+'_'+str(i+1)+domain_tag
                index=list(domain.id).index(id)
                domain.z[index]=height_list2[i]
                domain.oc[index]=oc_list2[i]
    elif shape=='Flat':
        name_header=None
        if freeze_tag:
            name_header='Freezed_el_'
        else:
            name_header='Gaussian_'
        group_names1=[name_header+el[i]+'_'+str(i+1)+domain_tag for i in range(number)]
        for i in range(number):
            try:
                groups1.append(domain.add_atom(id=name_header+el[i]+'_'+str(i+1)+domain_tag, element=el[i], x=0.5, y=0.5, z=height_list[i], u = u_init, oc = occ_init, m = 1.0))
            except:
                id=name_header+el[i]+'_'+str(i+1)+domain_tag
                index=list(domain.id).index(id)
                domain.z[index]=height_list[i]
                #domain.oc[index]=oc_list2[i]

    return domain,groups1+groups2,group_names1+group_names2

def print_gaussian_vars(domain,el=None):
    el_list=[]
    x_list=[]
    y_list=[]
    z_list=[]
    oc_list=[]
    u_list=[]
    for i in range(len(domain.id)):
        if 'Gaussian_' in domain.id[i] or 'Freezed_el_' in domain.id[i]:
            if el==None:
                el_list.append(domain.el[i])
                x_list.append(domain.x[i]+domain.dx1[i]+domain.dx2[i]+domain.dx3[i]+domain.dx4[i])
                y_list.append(domain.y[i]+domain.dy1[i]+domain.dy2[i]+domain.dy3[i]+domain.dy4[i])
                z_list.append(domain.z[i]+domain.dz1[i]+domain.dz2[i]+domain.dz3[i]+domain.dz4[i])
                u_list.append(domain.u[i])
                oc_list.append(domain.oc[i])
            else:
                if el==domain.el[i]:
                    el_list.append(domain.el[i])
                    x_list.append(domain.x[i]+domain.dx1[i]+domain.dx2[i]+domain.dx3[i]+domain.dx4[i])
                    y_list.append(domain.y[i]+domain.dy1[i]+domain.dy2[i]+domain.dy3[i]+domain.dy4[i])
                    z_list.append(domain.z[i]+domain.dz1[i]+domain.dz2[i]+domain.dz3[i]+domain.dz4[i])
                    u_list.append(domain.u[i])
                    oc_list.append(domain.oc[i])
    print('U_RAXS_LIST=',list(np.around(u_list,decimals=3)))
    print('OC_RAXS_LIST=',list(np.around(oc_list,decimals=3)))
    print('X_RAXS_LIST=',list(np.around(x_list,decimals=3)))
    print('Y_RAXS_LIST=',list(np.around(y_list,decimals=3)))
    print('Z_RAXS_LIST=',list(np.around(z_list,decimals=3)))
    print('el_freezed=',el_list)
    return None

def define_gaussian_vars(rgh,domain,shape='Flat',freeze_tag=False):
    if shape=='Flat':
        if freeze_tag:
            ids=[id for id in domain.id if 'Freezed' in id]
        else:
            ids=[id for id in domain.id if 'Gaussian' in id]
        for i in range(len(ids)):
            rgh.new_var('Gaussian_z_offset'+str(i+1),0)
            rgh.new_var('Gaussian_Spacing',2)
            rgh.new_var('Gaussian_Height',1)
    elif shape=='Single_Gaussian':
        rgh.new_var('Gaussian_RMS',2)
        rgh.new_var('Gaussian_OCC',1)
        rgh.new_var('Gaussian_U',0.004)
        rgh.new_var('Gaussian_Height',0)
        rgh.new_var('Gaussian_Spacing',10)
    elif shape=='Double_Gaussian':
        rgh.new_var('Gaussian_RMS',2)
        rgh.new_var('Gaussian_OCC',1)
        rgh.new_var('Gaussian_U',0.004)
        rgh.new_var('Gaussian_Height',0)
        rgh.new_var('Gaussian_Spacing',10)
        rgh.new_var('Gaussian_RMS_2',2)
        rgh.new_var('Gaussian_OCC_2',1)
        rgh.new_var('Gaussian_U_2',0.004)
        rgh.new_var('Gaussian_Height_2',0)
        rgh.new_var('Gaussian_Spacing_2',10)
    return rgh

def update_gaussian(domain,rgh,groups,el='O',number=3,height_offset=0,c=20.1058,domain_tag='_D1',shape='Flat',print_items=False,use_cumsum=True,freeze_tag=False):
    if shape=='Flat':
        items=list(map(lambda y:getattr(rgh,y)(),list(map(lambda x:'getGaussian_z_offset'+str(x+1), range(len(groups))))))
        gaussian_spacing=getattr(rgh,'getGaussian_Spacing')()
        gaussian_height=getattr(rgh,'getGaussian_Height')()
        add_gaussian(domain=domain,el=el,number=number,first_peak_height=gaussian_height,spacing=gaussian_spacing,height_offset=height_offset,c=c,domain_tag=domain_tag,shape=shape,freeze_tag=freeze_tag)
        if use_cumsum:
            items=np.cumsum(items)
        for i in range(len(groups)):
            getattr(groups[i],'setdz')(items[i])
        if print_items:
            print(items)
    elif shape=='Single_Gaussian':
        gaussian_rms=getattr(rgh,'getGaussian_RMS')()
        gaussian_occ=getattr(rgh,'getGaussian_OCC')()
        gaussian_u=getattr(rgh,'getGaussian_U')()
        gaussian_spacing=getattr(rgh,'getGaussian_Spacing')()
        gaussian_height=getattr(rgh,'getGaussian_Height')()
        add_gaussian(domain=domain,el=el,number=number,first_peak_height=gaussian_height,spacing=gaussian_spacing,u_init=gaussian_u,occ_init=gaussian_occ,height_offset=height_offset,c=c,domain_tag=domain_tag,shape=shape,gaussian_rms=gaussian_rms,freeze_tag=freeze_tag)
    elif shape=='Double_Gaussian':

        gaussian_rms=[getattr(rgh,'getGaussian_RMS')(),getattr(rgh,'getGaussian_RMS_2')()]
        gaussian_occ=[getattr(rgh,'getGaussian_OCC')(),getattr(rgh,'getGaussian_OCC_2')()]
        gaussian_u=[getattr(rgh,'getGaussian_U')(),getattr(rgh,'getGaussian_U_2')()]
        gaussian_spacing=[getattr(rgh,'getGaussian_Spacing')(),getattr(rgh,'getGaussian_Spacing_2')()]
        gaussian_height=[getattr(rgh,'getGaussian_Height')(),getattr(rgh,'getGaussian_Height_2')()]
        add_gaussian(domain=domain,el=el,number=number,first_peak_height=gaussian_height,spacing=gaussian_spacing,u_init=gaussian_u,occ_init=gaussian_occ,height_offset=height_offset,c=c,domain_tag=domain_tag,shape=shape,gaussian_rms=gaussian_rms,freeze_tag=freeze_tag)
    return None

def add_freezed_els(domain,el,u,oc,x,y,z,domain_tag='_D1'):
    if type(el)!=type([]):
        el=[el]*len(z)
    if x==[]:
        x=[0.5]*len(z)
    if y==[]:
        y=[0.5]*len(z)
    for i in range(len(z)):
        domain.add_atom(id='Freezed_el_'+str(i+1)+'_'+el[i]+domain_tag, element=el[i], x=x[i], y=y[i], z=z[i], u = u[i], oc = oc[i], m = 1.0)
    return domain

def add_oxygen_pair_muscovite(domain,ids,coors):
    domain.add_atom(id=ids[0],element='O', x=coors[0][0], y=coors[0][1], z=coors[0][2], oc=0.2,u = 1.)
    domain.add_atom(id=ids[1],element='O', x=coors[1][0], y=coors[1][1], z=coors[1][2], oc=0.2,u = 1.)
    atom_group=model.AtomGroup(domain,ids[0])
    atom_group.add_atom(domain,ids[1])
    return domain,atom_group
#function to group the Fourier components (FC) from different domains in each RAXR spectra
#domain_index=[0,1] means setting the FC for domain2 (1+1) same as domain1 (0+1)
#domain_index=3 means setting the FC for domain2 and domain3 same as domain1, in this case the number indicate the number of total domains
def set_RAXR(domain_index=[],number_spectra=0):
    domains=None
    if type(domain_index)!=type([]):
        domains=range(domain_index)
    else:
        domains=domain_index
    for i in range(number_spectra):
        for j in domains[1:]:
            eval('rgh_raxr'+'.setA_D'+str(j+1)+'_'+str(i+1)+'(rgh_raxr'+'.getA_D'+str(domains[0]+1)+'_'+str(i+1)+'())')
            eval('rgh_raxr'+'.setP_D'+str(j+1)+'_'+str(i+1)+'(rgh_raxr'+'.getP_D'+str(domains[0]+1)+'_'+str(i+1)+'())')

#freeze A and B in the process of model fitting
def set_RAXR_AB(number_spectra=0):
    spectra=None
    if type(number_spectra)!=type([]):
        spectra=range(number_spectra)
    else:
        spectra=number_spectra
    for i in spectra:
        eval('rgh_raxr'+'.setA'+str(i+1)+'(1.)')
        eval('rgh_raxr'+'.setB'+str(i+1)+'(0.)')

#function to group outer-sphere pars from different domains (to be placed inside sim function)
def set_OS(domain_names=['domain5','domain4']):
    eval('rgh_'+domain_names[0]+'.setCt_offset_dx_OS(rgh_'+domain_names[1]+'.getCt_offset_dx_OS())')
    eval('rgh_'+domain_names[0]+'.setCt_offset_dy_OS(rgh_'+domain_names[1]+'.getCt_offset_dy_OS())')
    eval('rgh_'+domain_names[0]+'.setCt_offset_dz_OS(rgh_'+domain_names[1]+'.getCt_offset_dz_OS())')
    eval('rgh_'+domain_names[0]+'.setTop_angle_OS(rgh_'+domain_names[1]+'.getTop_angle_OS())')
    eval('rgh_'+domain_names[0]+'.setR0_OS(rgh_'+domain_names[1]+'.getR0_OS())')
    eval('rgh_'+domain_names[0]+'.setPhi_OS(rgh_'+domain_names[1]+'.getPhi_OS())')

#function to group bidentate pars from different domains (to be placed inside sim function)
def set_BD(domain_names=[2,1],sorbate_sets=1,distal_oxygen_number=1,sorbate='Pb'):
    for i in range(sorbate_sets):
        eval('rgh_domain'+str(domain_names[0]+1)+'.setOffset_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getOffset_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setOffset2_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getOffset2_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setAngle_offset_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getAngle_offset_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setR_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getR_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setPhi_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getPhi_BD_'+str(i*2)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setTop_angle_BD_'+str(i*2)+'(rgh_domain'+str(domain_names[1]+1)+'.getTop_angle_BD_'+str(i*2)+'())')
        eval('gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc'+'(gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
        eval('gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu'+'(gp_'+sorbate+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')
        for j in range(distal_oxygen_number):
            eval('gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc'+'(gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
            eval('gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu'+'(gp_HO'+str(j+1)+'_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')

#function to group water pairs togeter from different domains
#domain_names is list of index of domains counting from 0 and number sets is the number of water pair counting from 1
def set_water_pair(domain_names=[3,2],number_sets=2):
    for i in range(number_sets):
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setoc(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getoc())')
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setu(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getu())')
        eval('gp_waters_set'+str(i+1)+'_D'+str(domain_names[0]+1)+'.setdy(gp_waters_set'+str(i+1)+'_D'+str(domain_names[1]+1)+'.getdy())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setV_shift_W_'+str(i+1)+'(rgh_domain'+str(domain_names[1]+1)+'.getV_shift_W_'+str(i+1)+'())')
        eval('rgh_domain'+str(domain_names[0]+1)+'.setAlpha_W_'+str(i+1)+'(180-rgh_domain'+str(domain_names[1]+1)+'.getAlpha_W_'+str(i+1)+'())')

#function to group tridentate pars specifically for distal oxygens from different domains (to be placed inside sim function)
def set_TD(domain_names=['domain2','domain1']):
    eval('rgh_'+domain_names[0]+'.setTheta1_1_TD(rgh_'+domain_names[1]+'.getTheta1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setTheta1_2_TD(rgh_'+domain_names[1]+'.getTheta1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setTheta1_3_TD(rgh_'+domain_names[1]+'.getTheta1_3_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_1_TD(rgh_'+domain_names[1]+'.getPhi1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_2_TD(rgh_'+domain_names[1]+'.getPhi1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setPhi1_3_TD(rgh_'+domain_names[1]+'.getPhi1_3_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_1_TD(rgh_'+domain_names[1]+'.getR1_1_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_2_TD(rgh_'+domain_names[1]+'.getR1_2_TD())')
    eval('rgh_'+domain_names[0]+'.setR1_3_TD(rgh_'+domain_names[1]+'.getR1_3_TD())')

#function to group Hydrogen pars from the same domain (to be placed inside sim function)
def set_H(domain_name='domain1',tag=['W_1_2_1','W_1_1_1']):
    eval('rgh_'+domain_name+'.setPhi_H_'+tag[0]+'(180-rgh_'+domain_name+'.getPhi_H_'+tag[1]+'())')
    eval('rgh_'+domain_name+'.setR_H_'+tag[0]+'(rgh_'+domain_name+'.getR_H_'+tag[1]+'())')
    eval('rgh_'+domain_name+'.setTheta_H_'+tag[0]+'(rgh_'+domain_name+'.getTheta_H_'+tag[1]+'())')

#function to group distal oxygens based on adding in wild, N is the number of distal oxygens (to be placed inside sim function)
def set_distal_wild(domain_name=['domain2','domain1'],tag='BD',N=2):
    for i in range(N):
        eval('rgh_'+domain_name[0]+'.setPhi1_'+str(i)+'_'+tag+'(180-rgh_'+domain_name[1]+'.getPhi1_'+str(i)+'_'+tag+'())')
        eval('rgh_'+domain_name[0]+'.setR1_'+str(i)+'_'+tag+'(rgh_'+domain_name[1]+'.getR1_'+str(i)+'_'+tag+'())')
        eval('rgh_'+domain_name[0]+'.setTheta1_'+str(i)+'_'+tag+'(rgh_'+domain_name[1]+'.getTheta1_'+str(i)+'_'+tag+'())')

def extract_column(file_name,which_column=0,deepest_row=None,split=','):
    items_container=[]
    f=open(file_name,'r')
    lines=f.readlines()
    for line in lines:
        items=line.rstrip().rsplit(split)
        if items!=[]:
            if items[0]!='#':
                items_container.append(items[which_column].strip())
    if deepest_row==None:
        return items_container
    else:
        return items_container[0:deepest_row]

def init_OS_auto(layer_index=[[0,6,6],[7],[10,14]],step_index=[2,3,1],OS_index=[6,14]):
    OS_X=[]
    OS_Y=[]
    OS_Z=[]
    for each in layer_index:
        tmp_x,tmp_y,tmp_z=[],[],[]
        for i in range(len(each)):
            if each[i] not in OS_index:
                tmp_x.append(None)
                tmp_x.append(None)
                tmp_y.append(None)
                tmp_y.append(None)
                tmp_z.append(None)
                tmp_z.append(None)
            else:
                tmp_x.append(0.)
                tmp_x.append(0.5)
                tmp_y.append(0.)
                tmp_y.append(0.5)
                if step_index[layer_index.index(each)]==0:
                    tmp_z.append(1.8)
                    tmp_z.append(1.8)
                elif step_index[layer_index.index(each)]==1:
                    tmp_z.append(2.3)
                    tmp_z.append(2.3)
                elif step_index[layer_index.index(each)]==2:
                    tmp_z.append(1.6)
                    tmp_z.append(1.6)
                elif step_index[layer_index.index(each)]==3:
                    tmp_z.append(2.1)
                    tmp_z.append(2.1)
        OS_X.append(tmp_x)
        OS_Y.append(tmp_y)
        OS_Z.append(tmp_z)
    return OS_X,OS_Y,OS_Z

def init_OS_auto2(layer_index=[[0,6,6],[7],[10,14]],OS_index=[6,14]):
    OS_X=[]
    OS_Y=[]
    OS_Z=[]
    for each in layer_index:
        tmp_x,tmp_y,tmp_z=[],[],[]
        for i in range(len(each)):
            if each[i] not in OS_index:
                tmp_x.append(None)
                tmp_x.append(None)
                tmp_y.append(None)
                tmp_y.append(None)
                tmp_z.append(None)
                tmp_z.append(None)
            else:
                tmp_x.append(0.)
                tmp_x.append(0.5)
                tmp_y.append(0.9)
                tmp_y.append(0.4)
                tmp_z.append(1.75)
                tmp_z.append(1.75)
        OS_X.append(tmp_x)
        OS_Y.append(tmp_y)
        OS_Z.append(tmp_z)
    return OS_X,OS_Y,OS_Z

def translate_domain_type(GROUPING_SCHEMES=[[0,1]],full_layer_pick=[None,None,0]):
    domain_type=[]
    def _translate_type(label):
        if label==None:return "HL"
        elif label==2:return "HL_S"
        elif label==3:return "HL_L"
        elif label==0:return "FL_S"
        elif label==1:return "FL_L"
    for each in GROUPING_SCHEMES:
        domain_type.append([_translate_type(full_layer_pick[each[0]]),_translate_type(full_layer_pick[each[1]])])
    return domain_type

#Generate commands to group surface atom layers from different domains
#each item inside the domain_index_pair is a two item integer numbers specifying the domain index (counting from 1)
#domain_type_pair has one to one corresponding to the domain_index_pair, only possible items include
#grouping_depth specify how deep you will want to group the atoms (<=10)
#'HL': half layer termination
#'FL_S': short full layer terminations
#'FL_L': long full layer terminations
#and note that the symmetry constrain has been accounted for when considering two full layer termination but with different height
def generate_commands_for_surface_atom_grouping(domain_index_pair=[[1,2],[3,4]],domain_type_pair=[['HL','HL'],['FL_S','FL_L']],grouping_depth=[10,10]):
    command_list=[]
    HL=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    HL_L=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    HL_S=['O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12']
    FL_S=['O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12']
    FL_L=['O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6']
    for i in range(len(domain_index_pair)):
        if domain_type_pair[i]==['HL','HL']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['FL_S','FL_S']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['FL_L','FL_L']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['FL_S','FL_L']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['FL_L','FL_S']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')

        elif domain_type_pair[i]==['HL_S','HL_S']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['HL_L','HL_L']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['HL_S','HL_L']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
        elif domain_type_pair[i]==['HL_L','HL_S']:
            for j in range(10-grouping_depth[i],10):
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')

    return command_list

def generate_commands_for_surface_atom_grouping_new(domain_index_pair=[[1,2],[3,4]],domain_type_pair=[['HL','HL'],['FL_S','FL_L']],grouping_depth=[[0,10],[0,10]],group_oc=False):
    command_list=[]
    HL=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    HL_L=['O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6']
    HL_S=['O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12']
    FL_S=['O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6','O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12']
    FL_L=['O11O12_O5O6','O1O2_O7O8','Fe2Fe3_Fe8Fe9','O3O4_O9O10','Fe4Fe6_Fe10Fe12','O5O6_O11O12','O7O8_O1O2','Fe8Fe9_Fe2Fe3','O9O10_O3O4','Fe10Fe12_Fe4Fe6']
    for i in range(len(domain_index_pair)):
        domain_index_pair[i][0]=domain_index_pair[i][0]+1
        domain_index_pair[i][1]=domain_index_pair[i][1]+1
        if domain_type_pair[i]==['HL','HL']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+HL[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+HL[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['FL_S','FL_S']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['FL_L','FL_L']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['FL_S','FL_L']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['FL_L','FL_S']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+FL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+FL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')

        elif domain_type_pair[i]==['HL_S','HL_S']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['HL_L','HL_L']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['HL_S','HL_L']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')
        elif domain_type_pair[i]==['HL_L','HL_S']:
            for j in range(grouping_depth[i][0],grouping_depth[i][1]):
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdx('+'-gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdx())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdy('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdy())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setdz('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getdz())')
                if group_oc:
                    command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setoc('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getoc())')
                command_list.append('gp_'+HL_L[j]+'_D'+str(domain_index_pair[i][0])+'.setu('+'gp_'+HL_S[j]+'_D'+str(domain_index_pair[i][1])+'.getu())')

    return command_list

def generate_commands_for_surface_atom_grouping_muscovit(domain_index_pair=[[1,2],[3,4]],pick_up_index=[[1],[2],[3],[4]],FL_Al=range(19),FL_Si=range(19,19+9),grouping_depth=[18,18]):
    command_list=[]
    FL_Al=['O4_O4O3_O7O8','O3_O4O3_O7O8','O5_O3O4_O8O7','Al1_Al4Al3_Al7Al8','Al2_Al3Al4_Al8Al7','O1_O4O3_O7O8','O2_O3O4_O8O7','O6_O3O4_O8O7','Al3_Al4Al3_Al7Al8','Al3_Al6Al5_Al1Al2','O6_O5O6_O2O1','O2_O5O6_O2O1','O1_O6O5_O1O2','Al2_Al5Al6_Al2Al1','Al1_Al6Al5_Al1Al2','O5_O5O6_O2O1','O4_O6O5_O1O2','O3_O6O5_O1O2']
    FL_Si=['O4_O4O3_O7O8','O3_O4O3_O7O8','O5_O3O4_O8O7','Si1_Si4Si3_Si7Si8','Si2_Si3Si4_Si8Si7','O1_O4O3_O7O8','O2_O3O4_O8O7','O6_O3O4_O8O7','Al3_Al4Al3_Al7Al8','Al3_Al6Al5_Al1Al2','O6_O5O6_O2O1','O2_O5O6_O2O1','O1_O6O5_O1O2','Si2_Si5Si6_Si2Si1','Si1_Si6Si5_Si1Si2','O5_O5O6_O2O1','O4_O6O5_O1O2','O3_O6O5_O1O2']
    for i in range(len(domain_index_pair)):
        if pick_up_index[domain_index_pair[i][0]][0] in FL_Al and pick_up_index[domain_index_pair[i][1]][0] in FL_Al:
            for j in range(18-grouping_depth[i],18):
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdx('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdx())')
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdy('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdy())')
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdz('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdz())')
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setoc('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getoc())')
        elif pick_up_index[domain_index_pair[i][0]][0] in FL_Si and pick_up_index[domain_index_pair[i][1]][0] in FL_Si:
            for j in range(18-grouping_depth[i],18):
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdx('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdx())')
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdy('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdy())')
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdz('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdz())')
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setoc('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getoc())')
        elif pick_up_index[domain_index_pair[i][0]][0] in FL_Al and pick_up_index[domain_index_pair[i][1]][0] in FL_Si:
            for j in range(18-grouping_depth[i],18):
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdx('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdx())')
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdy('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdy())')
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdz('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdz())')
                command_list.append('gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setoc('+'gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getoc())')
        elif pick_up_index[domain_index_pair[i][0]][0] in FL_Si and pick_up_index[domain_index_pair[i][1]][0] in FL_Al:
            for j in range(18-grouping_depth[i],18):
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdx('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdx())')
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdy('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdy())')
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setdz('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getdz())')
                command_list.append('gp_'+FL_Si[j]+'_D'+str(domain_index_pair[i][0]+1)+'.setoc('+'gp_'+FL_Al[j]+'_D'+str(domain_index_pair[i][1]+1)+'.getoc())')
    return command_list

#extract xyz for atom with id in domain
def extract_coor(domain,id):
    index=np.where(domain.id==id)[0][0]
    x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
    y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
    z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
    return np.array([x,y,z])

def translate_offset_symbols(symbol):
    if symbol=='-x':return np.array([-1.,0.,0.])
    elif symbol=='+x':return np.array([1.,0.,0.])
    elif symbol=='-y':return np.array([0.,-1.,0.])
    elif symbol=='+y':return np.array([0.,1.,0.])
    elif symbol==None:return np.array([0.,0.,0.])

def extract_coor_offset(domain,id=['id1','id2'],offset=[],basis=[5.038,5.434,7.3707]):
    coors=[extract_coor(domain,each_id) for each_id in id]
    offsets=[translate_offset_symbols(each_offset) for each_offset in offset]

    coors_offset=[coors[i]+offsets[i] for i in range(len(coors))]
    return f2(coors_offset[0]*basis,coors_offset[1]*basis)

def layer_spacing_calculator(domain,layer_N,half_layer):
    print("bulk structure (A), fit structure (A), percentage of change in fit")
    layer_index=range(layer_N)
    z_org=[]
    z_fit=[]
    if half_layer==True:
        layer_index.pop(1)
    for i in layer_index:
        z_org.append(domain.z[i*2]*7.3707)
        z_fit.append((domain.z[i*2]+domain.dz1[i*2]+domain.dz2[i*2]+domain.dz3[i*2])*7.3707)
    for j in range(len(z_org)-1):
        print(z_org[j]-z_org[j+1],z_fit[j]-z_fit[j+1],((z_fit[j]-z_fit[j+1])-(z_org[j]-z_org[j+1]))/(z_org[j]-z_org[j+1]))
    return True

def extract_coor2(domain,id):
    index=np.where(domain.id==id)[0][0]
    x=domain.x[index]
    y=domain.y[index]
    z=domain.z[index]
    return np.array([x,y,z])

def extract_component(domain,id,name_list):
    index=np.where(domain.id==id)[0][0]
    temp=[vars(domain)[name][index] for name in name_list]
    for i in range(len(name_list)):
        print(name_list[i]+'=',temp[i])

#set coor to atom with id in domain
def set_coor(domain,id,coor):
    index=np.where(domain.id==id)[0][0]
    domain.x[index]=coor[0]
    domain.y[index]=coor[1]
    domain.z[index]=coor[2]

#set sorbate coors to two symmetrical realated domains (rcut hematite in this case)
#domains:two items with symmetry related to each other
#ids:two items, first item is a list of ids corresponding to first domain, and second item to the second domain
#els:a list of elements with order corresponding to each item of the ids
#grids: fractional coordinates of sorbates for first domain
def set_coor_grid(domains=['domain1A','domain1B'],ids=[[],[]],els=[],grids=[[0.3,0.5,1.2]]):

    for i in range(len(els)):
        index=None
        try:
            index=np.where(domains[0].id==ids[0][i])[0][0]
        except:
            domains[0].add_atom( ids[0][i], els[i],  grids[i][0] ,grids[i][1], grids[i][2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            domains[1].add_atom( ids[1][i], els[i],  1-grids[i][0] ,grids[i][1]-0.06955, grids[i][2]-0.5 ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if index!=None:
            domain[0].x[index],domain[0].y[index],domain[0].z[index]=grids[i][0],grids[i][1],grids[i][2]
            domain[1].x[index],domain[1].y[index],domain[1].z[index]=1-grids[i][0],grids[i][1]-0.06955,grids[i][2]-0.5
#grid matching library for considering offset, x y both from -0.3 to 1.2 with each step of 0.5
#match like 1  2  3
#           6  5  4
#           7  8  9
#the match is based on closest distance
#if you consider match 3 and 6, then 6 will shift towards right by 1 unit to make it to be adjacent to 3, so in this case offset is "+y"
#5 is neighbor to all the other tiles so no offsets (depicted as None)
grid_match_lib={}
grid_match_lib[1]={2:None,3:'-x',4:'-x',5:None,6:None,7:'+y',8:'+y',9:'-x+y'}
grid_match_lib[2]={1:None,3:None,4:None,5:None,6:None,7:'+y',8:'+y',9:'+y'}
grid_match_lib[3]={2:None,1:'+x',4:None,5:None,6:'+x',7:'+x+y',8:'+y',9:'+y'}
grid_match_lib[4]={2:None,3:None,1:'+x',5:None,6:'+x',7:'+x',8:None,9:None}
grid_match_lib[5]={2:None,3:None,4:None,1:None,6:None,7:None,8:None,9:None}
grid_match_lib[6]={2:None,3:'-x',4:'-x',5:None,1:None,7:None,8:None,9:'-x'}
grid_match_lib[7]={2:'-y',3:'-x-y',4:'-x',5:None,6:None,1:'-y',8:None,9:'-x'}
grid_match_lib[8]={2:'-y',3:'-y',4:None,5:None,6:None,7:None,1:'-y',9:None}
grid_match_lib[9]={2:'-y',3:'-y',4:None,5:None,6:'+x',7:'+x',8:None,1:'+x-y'}

################################some functions to be called in GenX script#######################################
#atoms (sorbates) will be added to position specified by the coor(usually set the coor to the center, then you can easily set dxdy range to [-0.5,0.5] [
def add_atom(domain,ref_coor=[],ids=[],els=[]):
    for i in range(len(ids)):
        try:
            domain.add_atom(ids[i],els[i],ref_coor[i][0],ref_coor[i][1],ref_coor[i][2],0.5,1.0,1.0)
        except:
            index=np.where(domain.id==ids[i])[0][0]
            domain.x[index]=ref_coor[i][0]
            domain.y[index]=ref_coor[i][1]
            domain.z[index]=ref_coor[i][2]

#function to export refined atoms positions after fitting
def print_data(N_sorbate=4,domain='',z_shift=1,half_layer=False,full_layer_long=0,save_file='D://model.xyz'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
    else:
        if full_layer_long:
            index=index_all[0:22]+index_all[42:42+N_sorbate]
        else:
            index=index_all[0:12]+index_all[32:32+N_sorbate]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
            f.write(s)
    f.close()

#function to export refined atoms positions after fitting
def print_data2(N_sorbate=4,domain='',z_shift=1,half_layer=False,half_layer_long=None,full_layer_long=0,save_file='D://model.xyz'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer and half_layer_long==3:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
    elif half_layer and half_layer_long==2:
        index=index_all[0:20]+index_all[30:30+N_sorbate]
    elif not half_layer and full_layer_long==1:
        index=index_all[0:22]+index_all[42:42+N_sorbate]
    elif not half_layer and full_layer_long==0:
        index=index_all[0:22]+index_all[32:32+N_sorbate]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    if half_layer_long==2 or full_layer_long==0:
        z_shift=0.5
    else:
        z_shift=1
    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
            f.write(s)
    f.close()


def print_data2_muscovite(N_sorbate=4,domain='',z_shift=0.8,save_file='D://model.xyz'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:74]+index_all[132:132+N_sorbate]
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i]*5.1988,data[1][i]*9.0266,(data[2][i]-z_shift)*20.1058)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.1988,data[1][i]*9.0266,(data[2][i]-z_shift)*20.1058)
            f.write(s)
    f.close()

def print_data2C(N_sorbate=4,domain='',z_shift=1,half_layer=False,half_layer_long=None,full_layer_long=0,save_file='D://model.xyz',sorbate_index_list=[],each_segment_length=[]):
    #extract only one set of sorbate
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer and half_layer_long==3:
        index=index_all[0:20]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[40+sorbate_index_list[i]:40+sorbate_index_list[i]+each_segment_length[i]]
    elif half_layer and half_layer_long==2:
        index=index_all[0:20]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[30+sorbate_index_list[i]:30+sorbate_index_list[i]+each_segment_length[i]]
    elif not half_layer and full_layer_long==1:
        index=index_all[0:22]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[42+sorbate_index_list[i]:42+sorbate_index_list[i]+each_segment_length[i]]
    elif not half_layer and full_layer_long==0:
        index=index_all[0:22]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[32+sorbate_index_list[i]:32+sorbate_index_list[i]+each_segment_length[i]]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    if half_layer_long==2 or full_layer_long==0:
        z_shift=0.5
    else:
        z_shift=1
    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]-0.1391)*5.434,(data[2][i]-z_shift)*7.3707)
            f.write(s)
    f.close()


def print_data2C_muscovite(N_sorbate=4,domain='',z_shift=0.8,save_file='D://model.xyz',sorbate_index_list=[],each_segment_length=[]):
    #extract only one set of sorbate
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:74]
    for i in range(len(sorbate_index_list)):
        index=index+index_all[132+sorbate_index_list[i]:132+sorbate_index_list[i]+each_segment_length[i]]
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i]*5.1988,data[1][i]*9.0266,(data[2][i]-z_shift)*20.1058)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.1988,data[1][i]*9.0266,(data[2][i]-z_shift)*20.1058)
            f.write(s)
    f.close()

def make_cif_file(N_sorbate=4,domain='',z_shift=1,half_layer=False,half_layer_long=None,full_layer_long=0,save_file='D://model.xyz',sorbate_index_list=[],each_segment_length=[]):
    #extract only one set of sorbate
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer and half_layer_long==3:
        index=index_all[0:20]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[40+sorbate_index_list[i]:40+sorbate_index_list[i]+each_segment_length[i]]
    elif half_layer and half_layer_long==2:
        index=index_all[0:20]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[30+sorbate_index_list[i]:30+sorbate_index_list[i]+each_segment_length[i]]
    elif not half_layer and full_layer_long==1:
        index=index_all[0:22]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[42+sorbate_index_list[i]:42+sorbate_index_list[i]+each_segment_length[i]]
    elif not half_layer and full_layer_long==0:
        index=index_all[0:22]
        for i in range(len(sorbate_index_list)):
            index=index+index_all[32+sorbate_index_list[i]:32+sorbate_index_list[i]+each_segment_length[i]]
    if half_layer:
        index.pop(2)
        index.pop(2)
    if half_layer_long==2 or full_layer_long==0:
        z_shift=0.5
    else:
        z_shift=1
    c=(np.max(data[2])+0.3-z_shift)*7.3707
    f=open(save_file,'w')
    f.write('data_global\n')
    f.write("_chemical_name_mineral 'Hematite'\n")
    f.write("_chemical_formula_sum 'Fe2 O3'\n")
    f.write("_cell_length_a 5.038\n")
    f.write("_cell_length_b 5.434\n")
    f.write("_cell_length_c "+str(c)+"\n")
    f.write("_cell_angle_alpha 90\n")
    f.write("_cell_angle_beta 90\n")
    f.write("_cell_angle_gamma 90\n")
    f.write("_cell_volume 201.784\n")
    f.write("_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n  'x,y,z'\nloop_\n")
    f.write("_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")


    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i],(data[1][i]-0.1391),(data[2][i]-z_shift)*7.3707/c)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i],(data[1][i]-0.1391),(data[2][i]-z_shift)*7.3707/c)
            f.write(s)
    f.close()

def print_structure_files_muscovite(domain_list='',z_shift=0.8,matrix_info=None,save_file='D://model'):

    for domain_index in range(len(domain_list)):
        domain=domain_list[domain_index]
        data=domain._extract_values()
        index_all=range(len(data[0]))
        index=index_all[0:74]+index_all[132:]
        c=(np.max(data[2])+0.3-z_shift)*20.1058
        f=open(os.path.join(save_file,'Domain'+str(domain_index+1)+'.cif'),'w')
        f2=open(os.path.join(save_file,'Domain'+str(domain_index+1)+'.xyz'),'w')
        f3=open(os.path.join(save_file,'Domain'+str(domain_index+1)+'_id.xyz'),'w')

        f.write('data_global\n')
        f.write("_chemical_name_mineral 'Muscovite'\n")
        f.write("_chemical_formula_sum 'K Si3 Al3 O12 H2'\n")
        f.write("_cell_length_a 5.1988\n")
        f.write("_cell_length_b 9.0266\n")
        f.write("_cell_length_c "+str(c)+"\n")
        f.write("_cell_angle_alpha 90\n")
        f.write("_cell_angle_beta 95.782\n")
        f.write("_cell_angle_gamma 90\n")
        f.write("_cell_volume 938.7\n")
        f.write("_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n  'x,y,z'\nloop_\n")
        f.write("_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
        f2.write(str(len(index))+'\n#\n')
        f3.write(str(len(index))+'\n#\n')
        for i in index:
            coors=np.dot(matrix_info['T'],np.array([data[0][i],data[1][i],(data[2][i]-z_shift)])*matrix_info['basis'])
            if i==index[-1]:
                s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058/c)
                s2 = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],coors[0],coors[1],coors[2])
                s3 = '%-5s   %7.5e   %7.5e   %7.5e' % (domain.id[i],coors[0],coors[1],coors[2])
                f.write(s)
                f2.write(s2)
                f3.write(s3)
            else:
                s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058/c)
                s2 = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],coors[0],coors[1],coors[2])
                s3 = '%-5s   %7.5e   %7.5e   %7.5e\n' % (domain.id[i],coors[0],coors[1],coors[2])

                f.write(s)
                f2.write(s2)
                f3.write(s3)
        f.close()
        f2.close()
        f3.close()

def print_structure_files_muscovite_new(domain_list='',z_shift=0.8,number_gaussian=0,el='Zr',matrix_info=None,save_file='D://model'):

    for domain_index in range(len(domain_list)):
        domain=domain_list[domain_index]
        data=domain._extract_values()
        index_all=list(range(len(data[0])))
        index=index_all[0:74]+index_all[132:]
        index_f2=index_all[0:74]+index_all[132:(len(data[0])-number_gaussian)]
        c=(np.max(data[2])+0.3-z_shift)*20.1058
        f=open(os.path.join(save_file,'Domain'+str(domain_index+1)+'.cif'),'w')
        f2=open(os.path.join(save_file,'Domain'+str(domain_index+1)+'.xyz'),'w')
        f3=open(os.path.join(save_file,'Domain'+str(domain_index+1)+'_id.xyz'),'w')

        f.write('data_global\n')
        f.write("_chemical_name_mineral 'Muscovite'\n")
        f.write("_chemical_formula_sum 'K Si3 Al3 O12 H2'\n")
        f.write("_cell_length_a 5.1988\n")
        f.write("_cell_length_b 9.0266\n")
        f.write("_cell_length_c "+str(c)+"\n")
        f.write("_cell_angle_alpha 90\n")
        f.write("_cell_angle_beta 95.782\n")
        f.write("_cell_angle_gamma 90\n")
        f.write("_cell_volume 938.7\n")
        f.write("_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n  'x,y,z'\nloop_\n")
        f.write("_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
        f2.write(str(74*9+len(index_all[132:(len(data[0])-number_gaussian)]))+'\n#\n')
        f3.write(str(len(index))+'\n#\n')
        for i in index:
            coors=np.dot(matrix_info['T'],np.array([data[0][i],data[1][i],(data[2][i]-z_shift)])*matrix_info['basis'])
            if i==index[-1]:
                s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058/c)
                s3 = '%-5s   %7.5e   %7.5e   %7.5e' % (domain.id[i],coors[0],coors[1],coors[2])
                f.write(s)
                f3.write(s3)
            else:
                s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058/c)
                s3 = '%-5s   %7.5e   %7.5e   %7.5e\n' % (domain.id[i],coors[0],coors[1],coors[2])
                f.write(s)
                f3.write(s3)
        for i in index_f2:
            coors=np.dot(matrix_info['T'],np.array([data[0][i],data[1][i],(data[2][i]-z_shift)])*matrix_info['basis'])
            if i==index[-1]:
                if el in domain.id[i]:
                    s2 = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],coors[0],coors[1],coors[2])
                    f2.write(s2)
                else:
                    for grid in np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0],[0,2,0],[1,2,0],[2,2,0]]):
                        a,b,c=grid*matrix_info['basis']
                        s2 = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],coors[0]+a,coors[1]+b,coors[2]+c)
                        f2.write(s2)

            else:
                if el in domain.id[i]:
                    s2 = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],coors[0],coors[1],coors[2])
                    f2.write(s2)
                else:
                    for grid in np.array([[0,0,0],[1,0,0],[2,0,0],[0,1,0],[1,1,0],[2,1,0],[0,2,0],[1,2,0],[2,2,0]]):
                        a,b,c=grid*matrix_info['basis']
                        s2 = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],coors[0]+a,coors[1]+b,coors[2]+c)
                        f2.write(s2)
        f.close()
        f2.close()
        f3.close()


def print_data2B(N_sorbate=4,domain='',z_shift=1,half_layer=False,half_layer_long=None,full_layer_long=0,save_file='D://model.xyz'):
    #moving slab down if z_shift is a negative number
    #moving slab up if z_shift is a positive number
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    z_offset=0
    if half_layer and half_layer_long==3:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
    elif half_layer and half_layer_long==2:
        index=index_all[0:20]+index_all[30:30+N_sorbate]
    elif not half_layer and full_layer_long==1:
        index=index_all[0:22]+index_all[42:42+N_sorbate]
    elif not half_layer and full_layer_long==0:
        index=index_all[0:22]+index_all[32:32+N_sorbate]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    if half_layer_long==2 or full_layer_long==0:
        z_offset=z_shift-0.5
    else:
        z_offset=z_shift-1
    for i in index:
        if i==index[-1]:
            s = '%-5s   %7.5e   %7.5e   %7.5e' % (data[3][i],data[0][i]*5.038,(data[1][i]+z_offset*0.1391)*5.434,(data[2][i]+z_offset)*7.3707)
            f.write(s)
        else:
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (data[3][i],data[0][i]*5.038,(data[1][i]+z_offset*0.1391)*5.434,(data[2][i]+z_offset)*7.3707)
            f.write(s)
    f.close()

def print_data_for_publication(N_sorbate=4,domain='',z_shift=1,half_layer=False,full_layer_long=0,save_file='D://model.xyz'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
    else:
        if full_layer_long:
            index=index_all[0:22]+index_all[42:42+N_sorbate]
        else:
            index=index_all[0:12]+index_all[32:32+N_sorbate]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%-5s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f' % (data[3][i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
        else:
            s = '%-5s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f\n' % (data[3][i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
    f.close()

def print_data_for_publication_B(N_sorbate=4,domain='',z_shift=1,half_layer=False,full_layer_long=0,save_file='D://model.xyz'):
    #very similar to print_data_for_publication but also output the ids in the first column
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
    else:
        if full_layer_long:
            index=index_all[0:22]+index_all[42:42+N_sorbate]
        else:
            index=index_all[0:12]+index_all[32:32+N_sorbate]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    #f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f' % (domain.id[i],data[3][i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
        else:
            s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f\n' % (domain.id[i],data[3][i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
    f.close()

def print_data_for_publication_B2(N_sorbate=4,domain='',z_shift=1,layer_types=0,save_file='D://model.xyz'):
    #very similar to print_data_for_publication_B but use numbers to identify the layer type (long or short HLT/FLT)
    #0(short FLT),1(long FLT), 2(short HLT), 3(long HLT)
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if layer_types==0:
        index=index_all[0:12]+index_all[32:32+N_sorbate]
    elif layer_types==1:
        index=index_all[0:22]+index_all[42:42+N_sorbate]
    elif layer_types==2:
        index=index_all[0:10]+index_all[30:30+N_sorbate]
        index.pop(2)
        index.pop(2)
    elif layer_types==3:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    #f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f' % (domain.id[i],data[3][i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
        else:
            s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f\n' % (domain.id[i],data[3][i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
    f.close()

def print_data_for_publication_B2_muscovite(N_sorbate=4,domain='',z_shift=1,save_file='D://model.xyz'):

    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:20]
    z_temp=data[2][132:132+N_sorbate]
    for each_z in np.sort(z_temp):
        index.append(list(np.where(data[2]==each_z))[0][0])
    f=open(save_file,'w')
    #f.write(str(len(index))+'\n#\n')
    for i in index:
        #if i==index[-1]:
        #    s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f' % (domain.id[i],data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058,(data[0][i]-domain.x[i])*5.198,(data[1][i]-domain.y[i])*9.0266,(data[2][i]-domain.z[i])*20.1058,data[4][i],data[5][i]/4)
        #    f.write(s)
        #else:
        #    s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f\n' % (domain.id[i],data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058,(data[0][i]-domain.x[i])*5.198,(data[1][i]-domain.y[i])*9.0266,(data[2][i]-domain.z[i])*20.1058,data[4][i],data[5][i]/4)
        #    f.write(s)
        if i==index[-1]:
            s = '%s\t%s\t%5.3f\t%5.3f\t%4.2f' % (domain.id[i],data[3][i],(data[2][i]-z_shift)*20.1058,data[4][i],data[5][i]/4)
            f.write(s)
        else:
            s = '%s\t%s\t%5.3f\t%5.3f\t%4.2f\n' % (domain.id[i],data[3][i],(data[2][i]-z_shift)*20.1058,data[4][i],data[5][i]/4)
            f.write(s)
    f.close()

def print_data_for_publication_B3_muscovite(N_sorbate=4,domain='',z_shift=1,save_file='D://model.xyz',tab_file='D://tab_best.tab'):
#here the tab_file is the best fit par values exported from GenX, the function will combine the values of oc u and z with the associated error bars from tab_file
        data=domain._extract_values()
        index_all=range(len(data[0]))
        index=index_all[0:20]
        z_temp=data[2][132:132+N_sorbate]
        for each_z in np.sort(z_temp):
            index.append(list(np.where(data[2]==each_z))[0][0])
        f=open(save_file,'w')
        f_best_var=open(tab_file,'r')
        lines_best_var=f_best_var.readlines()
        def _find_index(lines,phrase_segment):
            return_index,return_value,return_error=None,None,None
            for i in range(len(lines)):
                if phrase_segment in lines[i]:
                    items=lines[i].rstrip().rsplit('\t')
                    if items[-1]=='-':
                        break
                    else:
                        return_index=i
                        return_error=[abs(float(each)) for each in items[-1][1:-1].rsplit(',')]
                        return_value=items[1]
                        break
            if return_index!=None:
                return return_index,float(return_value),max(return_error)
            else:
                return 0,0,0
        #f.write(str(len(index))+'\n#\n')
        for i in index:
            #if i==index[-1]:
            #    s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f' % (domain.id[i],data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058,(data[0][i]-domain.x[i])*5.198,(data[1][i]-domain.y[i])*9.0266,(data[2][i]-domain.z[i])*20.1058,data[4][i],data[5][i]/4)
            #    f.write(s)
            #else:
            #    s = '%s\t%s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f\n' % (domain.id[i],data[3][i],data[0][i],data[1][i],(data[2][i]-z_shift)*20.1058,(data[0][i]-domain.x[i])*5.198,(data[1][i]-domain.y[i])*9.0266,(data[2][i]-domain.z[i])*20.1058,data[4][i],data[5][i]/4)
            #    f.write(s)
            oc_error,u_error,z_error=0.00,0.00,0.00
            if 'Gaussian' in domain.id[i]:
                oc_tag=domain.id[i]+'.setoc'
                u_tag=domain.id[i]+'.setu'
                z_num=domain.id[i].rsplit('_')[-2]
                z_tag=['rgh_gaussian.Gaussian_z_offset'+z_num,'rgh_gaussian.Gaussian_Spacing','rgh_gaussian.Gaussian_Height']
                #extract oc error
                temp_index,temp_value,oc_error=_find_index(lines_best_var,oc_tag)
                #extract u error
                temp_index,temp_value,u_error=_find_index(lines_best_var,u_tag)
                #extract z error
                z_error_temp=0.00
                temp_z_error_return=_find_index(lines_best_var,z_tag[0])
                if temp_z_error_return[1]!=0:
                    z_error+=temp_z_error_return[2]
                else:
                    pass
                z_error+=_find_index(lines_best_var,z_tag[1])[2]+_find_index(lines_best_var,z_tag[2])[2]
            elif 'Freezed_el' in domain.id[i]:
                oc_tag='Domain1.set'+domain.id[i]+'oc'
                oc_error=_find_index(lines_best_var,oc_tag)[2]
                u_tag='Domain1.set'+domain.id[i]+'u'
                u_error=_find_index(lines_best_var,u_tag)[2]

                oc_tag='Domain1.set'+domain.id[i]+'oc'
                u_tag='Domain1.set'+domain.id[i]+'u'
                z_num=domain.id[i].rsplit('_')[-2]
                z_tag=['rgh_gaussian_freeze.Gaussian_z_offset'+z_num,'rgh_gaussian_freeze.Gaussian_Spacing','rgh_gaussian_freeze.Gaussian_Height']
                #extract oc error
                temp_index,temp_value,oc_error=_find_index(lines_best_var,oc_tag)
                #extract u error
                temp_index,temp_value,u_error=_find_index(lines_best_var,u_tag)
                #extract z error
                z_error_temp=0.00
                temp_z_error_return=_find_index(lines_best_var,z_tag[0])
                if temp_z_error_return[1]!=0:
                    z_error+=temp_z_error_return[2]
                else:
                    pass
                z_error+=_find_index(lines_best_var,z_tag[1])[2]+_find_index(lines_best_var,z_tag[2])[2]
            else:
                pass

            s = '%s\t%s\t%5.3f(%5.3f)\t%5.3f(%5.3f)\t%4.2f(%4.2f)\n' % (domain.id[i],data[3][i],(data[2][i]-z_shift)*20.1058,z_error,data[4][i],u_error**0.5,data[5][i]/4.,oc_error/4.)
            f.write(s)
        f.close()
        f_best_var.close()

def make_publication_table(model_file="D:\\Model_domain3A_publication.dat",par_file="D:\\test.tab",el_substrate=['Fe','O'],el_sorbate=['Pb'],abc=[5.038,5.434,7.3707]):
    """This script is used to combine the output model file and the parameter table to a new file of table with errors for publication
        The main idea is to extract the error info from the parameter table and append each to the right positions (atom displacements),
        so that it can save you time to do the tedious work for editing the table for publication.
        par_file:
            Parameter table exported from GenX
        model_file:
            A model files exported using the domain_creator.print_data_for_publication_B2()
        so far this function cannot handle the model containing over 10 slabs (ambiguity could be caused)
            "O1_11_0_D1A" (10th layer) could not be distinguished with "O1_11_t_D1A"(1st layer)
        """
    f_model=open(model_file,'r')
    f_publication=open(model_file.replace('.dat','_combined.dat'),'w')
    #the id of each atom looks like O1_1_0_D1A
    match_lib={'O1':'O1O2','O2':'O1O2',\
               'O3':'O3O4','O4':'O3O4',\
               'O5':'O5O6','O6':'O5O6',\
               'O7':'O7O8','O8':'O7O8',\
               'O9':'O9O10','O10':'O9O10',\
               'O11':'O11O12','O12':'O11O12',\
               'Fe2':'Fe2Fe3','Fe3':'Fe2Fe3',\
               'Fe4':'Fe4Fe6','Fe6':'Fe4Fe6',\
               'Fe8':'Fe8Fe9','Fe9':'Fe8Fe9',\
               'Fe10':'Fe10Fe12','Fe12':'Fe10Fe12'}

    parameter_values=np.zeros((0,6))
    f_par=open(par_file,'r')
    lines_par=f_par.readlines()
    for line_par in lines_par:
        if line_par[0:4] in ["gp_"+x[0] for x in el_substrate]:
            line_par_items=line_par.split('\t')[:-1]
            parameter_values=np.append(parameter_values,[line_par_items],axis=0)
    f_par.close()

    lines_model=f_model.readlines()

    f_publication.write("Element\tX(fra)\tY(fra)\tZ(fra)\tdx(errors)(Angstrom)\tdy(errors)(Angstrom)\tdz(errors)(Angstrom)\tu\tocc\n")
    for line_model in lines_model:
        line_model_items=line_model.rstrip().split('\t')
        line_publication_items=line_model_items[1:]
        id=[None,line_model_items[0].split("_")[-1]]
        if line_model_items[1] in el_substrate and el_sorbate[0] not in line_model_items[0]:
            id[0]=line_model_items[1]+line_model_items[0].split("_")[1]
            for par in parameter_values:
                if match_lib[id[0]]==par[0].split("_")[1] and id[-1][0:-1] in par[0] and par[-1]!='None' and par[-1]!='-':
                    opt=par[0].split(".")[-1]
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    if opt=="setdx":
                        if par[0].split("_")[1].split(id[0])[0]=='':#sense of left(the glide plane symmetry makes the error boundaries swab as well)
                            line_publication_items[4]=line_publication_items[4]+" ("+"%1.E" % (errors[0]*abc[0]) +", " + "%1.E" % (errors[1]*abc[0]) + ")"
                        else:#sense of right
                            line_publication_items[4]=line_publication_items[4]+" ("+"%1.E" % (-errors[1]*abc[0]) +", " + "%1.E" % (-errors[0]*abc[0]) + ")"
                    elif opt=="setdy":
                        line_publication_items[5]=line_publication_items[5]+" ("+"%1.E" % (errors[0]*abc[1]) +", " + "%1.E" % (errors[1]*abc[1]) + ")"
                    elif opt=="setdz":
                        line_publication_items[6]=line_publication_items[6]+" ("+"%1.E" % (errors[0]*abc[2]) +", " + "%1.E" % (errors[1]*abc[2]) + ")"
                    elif opt=="setu":
                        line_publication_items[7]=line_publication_items[7]+" ("+"%1.E" % errors[0] +", " + "%1.E" % errors[1] + ")"
                    elif opt=="setoc":
                        line_publication_items[8]=line_publication_items[8]+" ("+"%1.E" % errors[0] +", " + "%1.E" % errors[1] + ")"
        elif line_model_items[1] in el_sorbate:#to be completed
            pass
        f_publication.write('\t'.join(line_publication_items)+"\n")
    f_model.close()
    f_publication.close()

def make_publication_table2(model_file="D:\\Model_domain3A_publication.dat",par_file="D:\\test.tab",el_substrate=['Fe','O'],el_sorbate=['Pb'],abc=[5.038,5.434,7.3707]):
    """This script is used to combine the output model file and the parameter table to a new file of table with errors for publication
        The main idea is to extract the error info from the parameter table and append each to the right positions (atom displacements),
        so that it can save you time to do the tedious work for editing the table for publication.
        par_file:
            Parameter table exported from GenX
        model_file:
            A model files exported using the domain_creator.print_data_for_publication_B2()
        so far this function cannot handle the model containing over 10 slabs (ambiguity could be caused)
            "O1_11_0_D1A" (10th layer) could not be distinguished with "O1_11_t_D1A"(1st layer)
        """
    f_model=open(model_file,'r')
    f_publication=open(model_file.replace('.dat','_combined.dat'),'w')
    #the id of each atom looks like O1_1_0_D1A
    match_lib={'O1':'O1O2','O2':'O1O2',\
               'O3':'O3O4','O4':'O3O4',\
               'O5':'O5O6','O6':'O5O6',\
               'O7':'O7O8','O8':'O7O8',\
               'O9':'O9O10','O10':'O9O10',\
               'O11':'O11O12','O12':'O11O12',\
               'Fe2':'Fe2Fe3','Fe3':'Fe2Fe3',\
               'Fe4':'Fe4Fe6','Fe6':'Fe4Fe6',\
               'Fe8':'Fe8Fe9','Fe9':'Fe8Fe9',\
               'Fe10':'Fe10Fe12','Fe12':'Fe10Fe12'}

    parameter_values=np.zeros((0,6))
    parameter_values_other=np.zeros((0,6))
    f_par=open(par_file,'r')
    lines_par=f_par.readlines()
    for line_par in lines_par:
        if line_par !='\t\n':
            if line_par[0:4] in ["gp_"+x[0] for x in el_substrate]:
                line_par_items=line_par.split('\t')
                if len(line_par_items)==7:
                    line_par_items=line_par_items[:-1]
                else:
                    pass
                parameter_values=np.append(parameter_values,[line_par_items],axis=0)
            else:
                line_par_items=line_par.split('\t')
                if len(line_par_items)==7:
                    line_par_items=line_par_items[:-1]
                else:
                    pass
                parameter_values_other=np.append(parameter_values_other,[line_par_items],axis=0)
    f_par.close()
    lines_model=f_model.readlines()
    f_publication.write("Element\tX(fra)\tY(fra)\tZ(fra)\tdx(errors)(Angstrom)\tdy(errors)(Angstrom)\tdz(errors)(Angstrom)\tu\tocc\n")
    def _formate_values(value,errors):
        import math,decimal
        #eg value='1.245',errors=[-0.1,0.2], will return 1.2(2)
        value=float(value)
        if errors[0]==0 and errors[1]==0:
            return '{:0.3f}()'.format(value)
        error=None
        if abs(errors[0])>=abs(errors[1]):
            error=abs(errors[0])
        else:
            error=abs(errors[1])
        decimal_place=None
        error_tag=''
        try:
            temp='%e'%error
            if '+' in temp:
                error_tag=int(error)
                decimal_place=0
            else:
                decimal_place=int(temp.split('-')[1])
            if decimal_place>0:
                error_tag=int(round(10**decimal_place*error))
        except:
            pass
        if decimal_place>0 and error_tag==10:
            error_tag=9
        
        if decimal_place==0:
            if value<1:
                return '%i(%i)'%(1,error_tag)
            else:
                return '%2.1f(%i)'%(value,error_tag)

        return '{0:2.{1}f}({2})'.format(value,decimal_place,error_tag)

    for line_model in lines_model:
        line_model_items=line_model.rstrip().split('\t')
        line_publication_items=line_model_items[1:]
        id=[None,line_model_items[0].split("_")[-1]]
        if 'Os' in line_model_items[0] and line_model_items[1]=='O':#water molecules
            id=line_model_items[0]
            water_index,domain_tag=int(id.split('_')[0].replace('Os','')),id.split('_')[1].replace('A','')
            par_u_name='gp_waters_set{0}_{1}.setu'.format(water_index/3+1,domain_tag)
            par_oc_name='gp_waters_set{0}_{1}.setoc'.format(water_index/3+1,domain_tag)
            par_dy_name='gp_waters_set{0}_{1}.setdy'.format(water_index/3+1,domain_tag)
            par_dz_name='rgh_domain{0}.setV_shift_W_{1}'.format(domain_tag.replace('D',''),water_index/3+1)
            for par in parameter_values_other:
                if par[0]==par_u_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[7]=_formate_values(line_publication_items[7],errors)
                elif par[0]==par_oc_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[8]=_formate_values(line_publication_items[8],errors)
                elif par[0]==par_dy_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[2]=_formate_values(line_publication_items[2],errors)
                    line_publication_items[1]=_formate_values(line_publication_items[1],errors)
                elif par[0]==par_dz_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[3]=_formate_values(line_publication_items[3],np.array(errors)/abc[2])
            line_publication_items[4]='-'
            line_publication_items[5]='-'
            line_publication_items[6]='-'
        elif el_sorbate[0]==line_model_items[1]:#sorbate
            id=line_model_items[0]
            sorbate_index,domain_tag=int(id.split('_')[0].replace(el_sorbate[0],'')),id.split('_')[1].replace('A','')
            par_u_name='gp_{0}_set{1}_{2}.setu'.format(el_sorbate[0],sorbate_index/3+1,domain_tag)
            par_oc_name='gp_sorbates_set{0}_{1}.setoc'.format(sorbate_index,domain_tag)
            par_dx_name='gp_{0}_set{1}_{2}.setdx'.format(el_sorbate[0],sorbate_index/3+1,domain_tag)
            par_dy_name='gp_{0}_set{1}_{2}.setdy'.format(el_sorbate[0],sorbate_index/3+1,domain_tag)
            par_dz_name='gp_{0}_set{1}_{2}.setdz'.format(el_sorbate[0],sorbate_index/3+1,domain_tag)
            for par in parameter_values_other:
                if par[0]==par_u_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[7]=_formate_values(line_publication_items[7],errors)
                elif par[0]==par_oc_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[8]=_formate_values(line_publication_items[8],errors)
                elif par[0]==par_dx_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[1]=_formate_values(line_publication_items[1],errors)
                elif par[0]==par_dy_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[2]=_formate_values(line_publication_items[2],errors)
                elif par[0]==par_dz_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[3]=_formate_values(line_publication_items[3],errors)
            if '(' not in line_publication_items[1]:
                line_publication_items[1]=_formate_values(line_publication_items[1],[0,0])
            if '(' not in line_publication_items[2]:
                line_publication_items[2]=_formate_values(line_publication_items[2],[0,0])
            if '(' not in line_publication_items[3]:
                line_publication_items[3]=_formate_values(line_publication_items[3],[0,0])
            line_publication_items[4]='-'
            line_publication_items[5]='-'
            line_publication_items[6]='-'
        elif el_sorbate[0] in line_model_items[0] and 'HO' in line_model_items[0]:#distal oxygen
            id=line_model_items[0]
            distal_O_index,sorbate_index,domain_tag=int(id.split('_')[0].replace('HO','')),int(id.split('_')[1].replace(el_sorbate[0],'')),id.split('_')[2].replace('A','')
            par_u_name='gp_{0}_set{1}_{2}.setu'.format('HO',distal_O_index,domain_tag)
            par_oc_name='gp_sorbates_set{0}_{1}.setoc'.format(sorbate_index,domain_tag)
            par_dx_name='gp_{0}_set{1}_{2}.setdx'.format('HO',distal_O_index,domain_tag)
            par_dy_name='gp_{0}_set{1}_{2}.setdy'.format('HO',distal_O_index,domain_tag)
            par_dz_name='gp_{0}_set{1}_{2}.setdz'.format('HO',distal_O_index,domain_tag)
            
            for par in parameter_values_other:
                if par[0]==par_u_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[7]=_formate_values(line_publication_items[7],errors)
                elif par[0]==par_oc_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[8]=_formate_values(line_publication_items[8],errors)
                elif par[0]==par_dx_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[1]=_formate_values(line_publication_items[1],errors)
                elif par[0]==par_dy_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[2]=_formate_values(line_publication_items[2],errors)
                elif par[0]==par_dz_name:
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    line_publication_items[3]=_formate_values(line_publication_items[3],errors)
            if '(' not in line_publication_items[1]:
                line_publication_items[1]=_formate_values(line_publication_items[1],[0,0])
            if '(' not in line_publication_items[2]:
                line_publication_items[2]=_formate_values(line_publication_items[2],[0,0])
            if '(' not in line_publication_items[3]:
                line_publication_items[3]=_formate_values(line_publication_items[3],[0,0])
            line_publication_items[4]='-'
            line_publication_items[5]='-'
            line_publication_items[6]='-'
        elif line_model_items[1] in el_substrate and [True]*3==[each not in line_model_items[0] for each in [el_sorbate[0],'HO','Os']]:
            id[0]=line_model_items[1]+line_model_items[0].split("_")[1]
            for par in parameter_values:
                if match_lib[id[0]]==par[0].split("_")[1] and id[-1][0:-1] in par[0] and par[-1]!='None' and par[-1]!='-':
                    opt=par[0].split(".")[-1]
                    errors=[float(par[-1].split(",")[0][1:]),float(par[-1].split(",")[1][:-1])]
                    if opt=="setdx":
                        line_publication_items[4]=_formate_values(line_publication_items[4],np.array(errors)*abc[0])
                    elif opt=="setdy":
                        line_publication_items[5]=_formate_values(line_publication_items[5],np.array(errors)*abc[1])
                    elif opt=="setdz":
                        line_publication_items[6]=_formate_values(line_publication_items[6],np.array(errors)*abc[2])
                    elif opt=="setu":
                        line_publication_items[7]=_formate_values(line_publication_items[7],errors)
                    elif opt=="setoc":
                        line_publication_items[8]=_formate_values(line_publication_items[8],errors)
        else:
            line_publication_items[1]=_formate_values(line_publication_items[1],[0,0])
            line_publication_items[2]=_formate_values(line_publication_items[2],[0,0])
            line_publication_items[3]=_formate_values(line_publication_items[3],[0,0])
            line_publication_items[4]='-'
            line_publication_items[5]='-'
            line_publication_items[6]='-'
            line_publication_items[7]=_formate_values(line_publication_items[7],[0,0])
            line_publication_items[8]=_formate_values(line_publication_items[8],[0,0])
        f_publication.write('\t'.join(line_publication_items)+"\n")
    f_model.close()
    f_publication.close()

def print_data_for_publication2(N_sorbate=4,domain='',z_shift=1,half_layer=False,full_layer_long=0,save_file='D://model.xyz'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=None
    if half_layer:
        index=index_all[0:20]+index_all[40:40+N_sorbate]
    else:
        if full_layer_long:
            index=index_all[0:22]+index_all[42:42+N_sorbate]
        else:
            index=index_all[0:12]+index_all[32:32+N_sorbate]
    if half_layer:
        index.pop(2)
        index.pop(2)
    f=open(save_file,'w')
    #f.write(str(len(index))+'\n#\n')
    for i in index:
        if i==index[-1]:
            s = '%-5s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f' % (domain.id[i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
        else:
            s = '%-5s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%4.2f\t%4.2f\n' % (domain.id[i],data[0][i],data[1][i]-0.1391,data[2][i]-z_shift,(data[0][i]-domain.x[i])*5.038,(data[1][i]-domain.y[i])*5.434,(data[2][i]-domain.z[i])*7.3707,domain.u[i],domain.oc[i])
            f.write(s)
    f.close()

#function to export ref fit file (a connection between GenX output and ROD input)
#NOTE:only print out surface atoms, and no sorbates
def print_data_for_ROD(N_atm=40,domain='',save_file='D:\\Google Drive\\useful codes\\half_layer_GenX_s1.txt'):
    data=domain._extract_values()
    index_all=range(len(data[0]))
    index=index_all[0:40]
    f=open(save_file,'w')
    f2=open(save_file[:-6]+'s2.txt','w')
    for i in index:
        s = '%s\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\n' % (data[3][i],data[0][i],1,0,0,0,data[1][i],1,0,0,0,data[2][i],0,0,0)
        f.write(s)
    for i in index_all[0:30]:
        s = '%s\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\t%i\t%6.5f\t%i\t%i\t%i\n' % (data[3][i],1.-data[0][i],-1,0,0,0,data[1][i]-0.1391/2.,1,0,0,0,data[2][i]-0.5,0,0,0)
        f2.write(s)
    f.close()
    f2.close()

def print_data_for_postrun_test(domain='',save_file='D:\\structure_file_postrun_test1.txt'):
    data=domain._extract_values2()
    index=range(len(data[0]))
    f=open(save_file,'w')
    for i in index:
        s = '%s,%s,%6.5f,%6.5f,%6.5f,%6.5f,%6.5f,%6.5f\n' % (domain.id[i][0:-4],data[3][i],data[0][i],data[1][i],data[2][i],data[4][i],data[5][i],1)
        f.write(s)
    f.close()

def create_list(ids,off_set_begin,start_N):
    ids_processed=[[],[]]
    off_set=[None,'+x','-x','+y','-y','+x+y','+x-y','-x+y','-x-y']
    for i in range(start_N):
        ids_processed[0].append(ids[i])
        ids_processed[1].append(off_set_begin[i])
    for i in range(start_N,len(ids)):
        for j in range(9):
            ids_processed[0].append(ids[i])
            ids_processed[1].append(off_set[j])
    return ids_processed
#function to build reference bulk and surface slab
def add_atom_in_slab(slab,filename,attach='',height_offset=0):
    f=open(filename)
    lines=f.readlines()
    for line in lines:
        if line[0]!='#':
            items=line.strip().rsplit(',')
            slab.add_atom(str(items[0].strip())+attach,str(items[1].strip()),float(items[2]),float(items[3]),float(items[4])+height_offset,float(items[5]),float(items[6]),float(items[7]))

#here only consider the match in the bulk,the offset should be maintained during fitting
def create_match_lib_before_fitting(domain_class,domain,atm_list,search_range,basis=np.array([5.038,5.434,7.3707]),T=None):
    match_lib={}
    for i in atm_list:
        atms,offset=domain_class.find_neighbors(domain,i,search_range,basis,T)
        match_lib[i]=[atms,offset]
    return match_lib
#Here we consider match with sorbate atoms, sorbates move around within one unitcell, so the offset will be change accordingly
#So this function should be placed inside sim function
#Note there is no return in this function, which will only update match_lib
def create_match_lib_during_fitting(domain_class,domain,atm_list,pb_list,HO_list,match_lib):
    match_list=[[atm_list,pb_list+HO_list],[pb_list,atm_list+HO_list],[HO_list,atm_list+pb_list]]
    #[atm_list,pb_list+HO_list]:atoms in atm_list will be matched to atoms in pb_list+HO_list
    for i in range(len(match_list)):
        atm_list_1,atm_list_2=match_list[i][0],match_list[i][1]
        for atm1 in atm_list_1:
            grid=domain_class.create_grid_number(atm1,domain)
            for atm2 in atm_list_2:
                grid_compared=domain_class.create_grid_number(atm2,domain)
                offset=domain_class.compare_grid(grid,grid_compared)
                if atm1 in match_lib.keys():
                    match_lib[atm1][0].append(atm2)
                    match_lib[atm1][1].append(offset)
                else:
                    match_lib[atm1]=[[atm2],[offset]]

def create_sorbate_ids(el='Pb',N=2,tag='_D1A'):
    id_list=[]
    [id_list.append(el+str(i+1)+tag) for i in range(N)]
    return id_list

def create_HO_ids(anchor_els=['Pb','Sb'],O_N=[1,1],tag='_D1A'):
    id_list=[]
    for i in range(len(O_N)):
        for j in range(O_N[i]):
            id_list.append('HO'+str(j+1)+'_'+anchor_els[i]+tag)
    return id_list

def create_HO_ids2(anchor_els=['Pb','Sb'],O_N=[[1,1],[3,3]],tag='_D1A'):
    id_list=[]
    N=0
    for i in range(len(O_N)):
        if i>0 and sum(O_N[i-1])!=0:N=N+len(filter(lambda x:x!=0,O_N[i-1]))
        for j in range(len(O_N[i])):
            temp_ids=[]
            for k in range(O_N[i][j]):
                temp_ids.append('HO'+str(k+1)+'_'+anchor_els[i]+str(N+j+1)+tag)
            [id_list.append(temp_id) for temp_id in temp_ids]
    return id_list

def create_HO_ids3(anchor_els=['Pb','Sb'],O_N=[3,3],tag='_D1A'):
    id_list=[]
    for i in range(len(O_N)):
        for j in range(O_N[i]):
            id_list.append('HO'+str(j+1)+'_'+anchor_els[i]+str(i+1)+tag)
    return id_list

def create_sorbate_ids2(el=['Pb','Sb'],N=[1,1],tag='_D1A'):
    id_list=[]
    for i in range(len(N)):
        for j in range(N[i]):
            if i!=0:
                sum_front=sum(N[0:i])
                id_list.append(el[i]+str(j+1+sum_front)+tag)
            else:
                id_list.append(el[i]+str(j+1)+tag)
    return id_list

def create_sorbate_ids3(el=['Pb','Sb'],N=[1,1],tag='_D1A'):
    id_list=[]
    for i in range(len(N)):
        if N[i]<=2:
            for j in range(N[i]):
                if i!=0:
                    sum_front=sum(N[0:i])
                    id_list.append(el[i]+str(j+1+sum_front)+tag)
                else:
                    id_list.append(el[i]+str(j+1)+tag)
        elif N[i]>2:
            for j in range(2):
                if i!=0:
                    sum_front=sum(N[0:i])
                    for k in range(N[i]/2):
                        id_list.append(el[i]+str(j+1+sum_front)+chr(ord('a') + k)+tag)
                else:
                    for k in range(N[i]/2):
                        id_list.append(el[i]+str(j+1)+chr(ord('a') + k)+tag)
    return id_list

def create_sorbate_el_list(el=['Pb','Sb'],N=[[1,2],[1,0]]):
    el_container=[]
    for i in N:
        el_temp=[]
        for j in range(len(i)):
            for k in range(i[j]):
                el_temp.append(el[j])
        el_container.append(el_temp)
    return el_container

def create_sorbate_el_list2(el=[['Pb','Sb'],['Pb']],N=[[1,2],[1]]):
    el_container=[]
    for i in range(len(N)):
        el_temp=[]
        for j in range(len(N[i])):
            for k in range(N[i][j]):
                el_temp.append(el[i][j])
        el_container.append(el_temp)
    return el_container

#based on the binding configuration, set up a library showing the coordinated atoms for each atoms
def create_sorbate_match_lib2(metal='Pb',O_Number=[1],O_list=[['HO1_D1A']],anchors=[['O1_2_0','O1_1_0']],anchor_offsets=[['+y',None]],domain_tag=1):
    match_lib_metal={}
    match_lib_HO={}
    match_lib_O={}
    match_lib={}
    N_sorbate=len(O_Number)
    O_index_list=[0]+[sum(O_Number[0:ii+1]) for ii in range(len(O_Number))]
    shaffle=lambda a,b:[each+b for each in a]
    def _change_direction(tag):
        if tag==None:return None
        elif tag[0]=='-':return '+'+tag[1:]
        elif tag[0]=='+':return '-'+tag[1:]

    for i in range(N_sorbate):
        match_lib_metal[metal+str(i+1)+'_D'+str(domain_tag)+'A']=[O_list[0][O_index_list[i]:O_index_list[i+1]]+shaffle(anchors[i],'_D'+str(domain_tag)+'A'),[None]*O_Number[i]+anchor_offsets[i]]
        for j in range(O_Number[i]):
            match_lib_HO[O_list[0][O_index_list[i]+j]]=[[metal+str(i+1)+'_D'+str(domain_tag)+'A'],[None]]
        for k in range(len(anchors[i])):
            if anchors[i][k]+'_D'+str(domain_tag)+'A' in match_lib_O.keys():
                first_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][0]
                second_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][1]
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[first_item+[metal+str(i+1)+'_D'+str(domain_tag)+'A'],second_item+[_change_direction(anchor_offsets[i][k])]]
            else:
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[[metal+str(i+1)+'_D'+str(domain_tag)+'A'],[_change_direction(anchor_offsets[i][k])]]
    for key in match_lib_metal.keys():match_lib[key]=match_lib_metal[key]
    for key in match_lib_HO.keys():match_lib[key]=match_lib_HO[key]
    for key in match_lib_O.keys():match_lib[key]=match_lib_O[key]
    return match_lib

#here the metal is a list of sorbate element symbols
def create_sorbate_match_lib3(metal=['Pb'],O_Number=[1],O_list=[['HO1_D1A']],anchors=[['O1_2_0','O1_1_0']],anchor_offsets=[['+y',None]],domain_tag=1):
    match_lib_metal={}
    match_lib_HO={}
    match_lib_O={}
    match_lib={}
    N_sorbate=len(metal)
    O_index_list=[0]+[sum(O_Number[0:ii+1]) for ii in range(len(O_Number))]
    if O_index_list[0:2]==[0,0]:
        O_index_list=O_index_list[1:]
    shaffle=lambda a,b:[each+b for each in a]
    def _change_direction(tag):
        if tag==None:return None
        elif tag[0]=='-':return '+'+tag[1:]
        elif tag[0]=='+':return '-'+tag[1:]

    for i in range(N_sorbate):
        match_lib_metal[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A']=[O_list[0][O_index_list[i]:O_index_list[i+1]]+shaffle(anchors[i],'_D'+str(domain_tag)+'A'),[None]*O_Number[i]+anchor_offsets[i]]
        for j in range(O_Number[i]):
            match_lib_HO[O_list[0][O_index_list[i]+j]]=[[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],[None]]
        for k in range(len(anchors[i])):
            if anchors[i][k]+'_D'+str(domain_tag)+'A' in match_lib_O.keys():
                first_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][0]
                second_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][1]
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[first_item+[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],second_item+[_change_direction(anchor_offsets[i][k])]]
            else:
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],[_change_direction(anchor_offsets[i][k])]]
    for key in match_lib_metal.keys():match_lib[key]=match_lib_metal[key]
    for key in match_lib_HO.keys():match_lib[key]=match_lib_HO[key]
    for key in match_lib_O.keys():match_lib[key]=match_lib_O[key]
    return match_lib

def create_sorbate_match_lib4(metal=['Pb'],HO_list=['HO1_Pb_D1A'],anchors=[['O1_2_0','O1_1_0']],anchor_offsets=[['+y',None]],domain_tag=1):
    match_lib_metal={}
    match_lib_HO={}
    match_lib_O={}
    match_lib={}
    N_sorbate=len(metal)
    shaffle=lambda a,b:[each+b for each in a]
    def _change_direction(tag):
        if tag==None:return None
        elif tag[0]=='-':return '+'+tag[1:]
        elif tag[0]=='+':return '-'+tag[1:]

    for i in range(N_sorbate):
        try:
            match_lib_metal[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A']=[[HO_id for HO_id in HO_list if metal[i] in HO_id]+shaffle(anchors[i],'_D'+str(domain_tag)+'A'),[None]*len([HO_id for HO_id in HO_list if metal[i] in HO_id])+anchor_offsets[i]]
        except:
            match_lib_metal[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A']=[shaffle(anchors[i],'_D'+str(domain_tag)+'A'),anchor_offsets[i]]
        for j in range(len(HO_list)):
            if metal[i] in HO_list[j]:
                match_lib_HO[HO_list[j]]=[[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],[None]]
        for k in range(len(anchors[i])):
            if anchors[i][k]+'_D'+str(domain_tag)+'A' in match_lib_O.keys():
                first_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][0]
                second_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][1]
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[first_item+[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],second_item+[_change_direction(anchor_offsets[i][k])]]
            else:
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],[_change_direction(anchor_offsets[i][k])]]
    for key in match_lib_metal.keys():
        try:
            match_lib[key]=[match_lib[key][0]+match_lib_metal[key][0],match_lib[key][1]+match_lib_metal[key][1]]
        except:
            match_lib[key]=match_lib_metal[key]
    for key in match_lib_HO.keys():
        try:
            match_lib[key]=[match_lib[key][0]+match_lib_HO[key][0],match_lib[key][1]+match_lib_HO[key][1]]
        except:
            match_lib[key]=match_lib_HO[key]
    for key in match_lib_O.keys():
        try:
            match_lib[key]=[match_lib[key][0]+match_lib_O[key][0],match_lib[key][1]+match_lib_O[key][1]]
        except:
            match_lib[key]=match_lib_O[key]

    return match_lib

def create_sorbate_match_lib4_test(metal=['Pb'],HO_list=['HO1_Pb_D1A'],anchors=[['O1_2_0','O1_1_0']],anchor_offsets=[['+y',None]],domain_tag=1):
    #a bug fixed here compared to the previous function
    match_lib_metal={}
    match_lib_HO={}
    match_lib_O={}
    match_lib={}
    N_sorbate=len(metal)
    shaffle=lambda a,b:[each+b for each in a]
    def _change_direction(tag):
        if tag==None:return None
        elif tag[0]=='-':return '+'+tag[1:]
        elif tag[0]=='+':return '-'+tag[1:]

    for i in range(N_sorbate):
        try:
            match_lib_metal[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A']=[[HO_id for HO_id in HO_list if metal[i]+str(i+1) in HO_id]+shaffle(anchors[i],'_D'+str(domain_tag)+'A'),[None]*len([HO_id for HO_id in HO_list if metal[i]+str(i+1) in HO_id])+anchor_offsets[i]]
        except:
            match_lib_metal[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A']=[shaffle(anchors[i],'_D'+str(domain_tag)+'A'),anchor_offsets[i]]
        for j in range(len(HO_list)):
            if metal[i]+str(i+1) in HO_list[j]:
                match_lib_HO[HO_list[j]]=[[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],[None]]
        for k in range(len(anchors[i])):
            if anchors[i][k]+'_D'+str(domain_tag)+'A' in match_lib_O.keys():
                first_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][0]
                second_item=match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A'][1]
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[first_item+[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],second_item+[_change_direction(anchor_offsets[i][k])]]
            else:
                match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[[metal[i]+str(i+1)+'_D'+str(domain_tag)+'A'],[_change_direction(anchor_offsets[i][k])]]
    for key in match_lib_metal.keys():
        try:
            match_lib[key]=[match_lib[key][0]+match_lib_metal[key][0],match_lib[key][1]+match_lib_metal[key][1]]
        except:
            match_lib[key]=match_lib_metal[key]
    for key in match_lib_HO.keys():
        try:
            match_lib[key]=[match_lib[key][0]+match_lib_HO[key][0],match_lib[key][1]+match_lib_HO[key][1]]
        except:
            match_lib[key]=match_lib_HO[key]
    for key in match_lib_O.keys():
        try:
            match_lib[key]=[match_lib[key][0]+match_lib_O[key][0],match_lib[key][1]+match_lib_O[key][1]]
        except:
            match_lib[key]=match_lib_O[key]

    return match_lib

def create_sorbate_match_lib(metal='Pb',O_list=[['HO1_D1A']],anchors=[['O1_2_0','O1_1_0']],anchor_offsets=[['+y',None]],domain_tag=1):
    match_lib_metal={}
    match_lib_HO={}
    match_lib_O={}
    match_lib={}
    N_sorbate=len(O_list)
    shaffle=lambda a,b:[each+b for each in a]
    def _change_direction(tag):
        if tag==None:return None
        elif tag[0]=='-':return '+'+tag[1:]
        elif tag[0]=='+':return '-'+tag[1:]

    for i in range(N_sorbate):
        match_lib_metal[metal+str(i+1)+'_D'+str(domain_tag)+'A']=[O_list[i]+shaffle(anchors[i],'_D'+str(domain_tag)+'A'),[None]*len(O_list[i])+anchor_offsets[i]]
        for j in range(len(O_list[i])):
            match_lib_HO[O_list[i][j]]=[[metal+str(i+1)+'_D'+str(domain_tag)+'A'],[None]]
        for k in range(len(anchors[i])):
            match_lib_O[anchors[i][k]+'_D'+str(domain_tag)+'A']=[[metal+str(i+1)+'_D'+str(domain_tag)+'A'],[_change_direction(anchor_offsets[i][k])]]
    for key in match_lib_metal.keys():match_lib[key]=match_lib_metal[key]
    for key in match_lib_HO.keys():match_lib[key]=match_lib_HO[key]
    for key in match_lib_O.keys():match_lib[key]=match_lib_O[key]
    return match_lib

#merge two libs in the form returned by create_sorbate_match_lib
#if same key, append it, if a new key, set a new item with this key
#also merge the id and the associated offset (eg O1_1_0_D1A_+x)
def merge_two_libs(lib_main,lib2):
    lib=deepcopy(lib_main)
    lib_new_format={}
    keys_main=lib_main.keys()
    def _append_two(id,offset):
        if offset==None:return id
        else:
            return id+'_'+offset

    for key in lib2.keys():
        if key in keys_main:
            lib[key]=[lib_main[key][0]+lib2[key][0],lib_main[key][1]+lib2[key][1]]
        else:lib[key]=lib2[key]
    for key in lib.keys():
        lib_new_format[key]=[_append_two(id,offset) for (id,offset) in zip(lib[key][0],lib[key][1])]
    return lib_new_format

def rotate_along_one_axis(domain=None,pass_point_id='',rotation_ids=[],rotation_vector=[],rotation_angle=None,basis=np.array([5.038,5.434,7.3707])):
    #note it works only when the dxdydz==0, if not rewrite the assignment part at line106
    pt_ct=lambda domain,p_O1_index:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0],domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0],domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]])*basis
    u,v,w=rotation_vector[0],rotation_vector[1],rotation_vector[2]
    pass_point_index=np.where(domain.id==pass_point_id)
    rotation_index=[np.where(domain.id==rotation_id) for rotation_id in rotation_ids]
    pass_point_coor=pt_ct(domain,pass_point_index)
    a,b,c=pass_point_coor[0],pass_point_coor[1],pass_point_coor[2]
    rotation_coors=[pt_ct(domain,index) for index in rotation_index]
    def _rotation(x,y,z,a,b,c,u,v,w,theta):
        L=u**2+v**2+w**2
        x_value=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
        y_value=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
        z_value=((c*(u**2+v**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
        return [x_value,y_value,z_value]
    container=[]
    for i in range(len(rotation_coors)):
        index=rotation_index[i][0][0]
        x,y,z=rotation_coors[i][0],rotation_coors[i][1],rotation_coors[i][2]
        x_new,y_new,z_new=_rotation(x,y,z,a,b,c,u,v,w,rotation_angle)/basis
        domain.x[index],domain.y[index],domain.z[index]=x_new,y_new,z_new
        container.append([x_new,y_new,z_new])
    return container


class domain_creator(domain_creator_water,domain_creator_sorbate,domain_creator_surface):
    def __init__(self,ref_domain,id_list,terminated_layer=0,domain_tag='_D1',new_var_module=None,N_layers=5):
        #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
        #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
        #function of set_new_vars
        #N_layers is the layer offset between two symmetry related terminations, default value 5 is for rcut hematite specifically
        self.ref_domain=ref_domain
        self.id_list=id_list
        self.terminated_layer=terminated_layer
        self.domain_tag=domain_tag
        self.share_face,self.share_edge,self.share_corner=(False,False,False)
        #self.anchor_list=[]
        self.polyhedra_list=[]
        self.new_var_module=new_var_module
        self.N_layers=N_layers
        self.domain_A,self.domain_B=self.create_equivalent_domains_2()

    def build_super_cell(self,ref_domain,rem_atom_ids=None):
    #build a super cell based on the ref_domain, the super cell is actually two domains stacking together in x direction
    #rem_atom_ids is a list of atom ids you want to remove before building a super cell
        super_cell=ref_domain.copy()
        if rem_atom_ids!=None:
            for i in rem_atom_ids:
                super_cell.del_atom(i)

        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
        for id in super_cell.id:
            index=np.where(ref_domain.id==id)[0][0]
            super_cell.add_atom(id=str(id)+'_+x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])

        return super_cell

    def build_super_cell_simple(self,ref_domain,rem_atom_ids=None):
    #in this simple mode: a library instead of domain instance will be created, and the key for library is (id,el)
    #so the library looks like: {(id1,el):coords,(id2,el):coords2}
        super_cell=ref_domain.copy()
        container={}
        if rem_atom_ids!=None:
            for i in rem_atom_ids:
                super_cell.del_atom(i)
        def _round_up(x):
            new_x=x
            while 1:
                if new_x>=0 and new_x<=1:
                    new_x=x
                    break
                elif new_x<0:
                    new_x=new_x+1
                elif new_x>1:
                    new_x=new_x-1
            return new_x

        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([_round_up(x),_round_up(y),z])
        for id in super_cell.id:
            index=np.where(ref_domain.id==id)[0][0]
            container[str(id),str(super_cell.el[index])]=_extract_coor(super_cell,id)
            container[str(id)+'_+x',str(super_cell.el[index])]=container[str(id)]+[1,0,0]
            container[str(id)+'_-x',str(super_cell.el[index])]=container[str(id)]+[-1,0,0]
            container[str(id)+'_+y',str(super_cell.el[index])]=container[str(id)]+[0,1,0]
            container[str(id)+'_-y',str(super_cell.el[index])]=container[str(id)]+[0,-1,0]
            container[str(id)+'_+x-y',str(super_cell.el[index])]=container[str(id)]+[1,-1,0]
            container[str(id)+'_+x+y',str(super_cell.el[index])]=container[str(id)]+[1,1,0]
            container[str(id)+'_-x+y',str(super_cell.el[index])]=container[str(id)]+[-1,1,0]
            container[str(id)+'_-x-y',str(super_cell.el[index])]=container[str(id)]+[-1,-1,0]

        return container

    def build_super_cell2(self,ref_domain_original,index_for_atoms=[0,1,4,5]):
    #build a super cell based on the ref_domain, the super cell is actually two domains stacking together in x direction
    #rem_atom_ids is a list of atom ids you want to remove before building a super cell
    #different from version one we extract some atoms from the ref_domain to begin
    #and note the sorbate and waters are at the bottom part, so the index should be -1 -2 and so on

        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])

        ref_domain=model.Slab(c = 1.0,T_factor='B')
        for i in index_for_atoms:
            id=str(ref_domain_original.id[i])
            ref_domain.add_atom(id=str(ref_domain_original.id[i]),element=ref_domain_original.el[i], x=_extract_coor(ref_domain_original,id)[0], y=_extract_coor(ref_domain_original,id)[1], z=_extract_coor(ref_domain_original,id)[2], u = ref_domain_original.u[i], oc = ref_domain_original.oc[i], m = ref_domain_original.m[i])

        super_cell=ref_domain.copy()
        for id in super_cell.id:
            index=np.where(ref_domain.id==id)[0][0]
            super_cell.add_atom(id=str(id)+'_+x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
        #for id in super_cell.id:
        #    index=np.where(super_cell.id==id)[0][0]
        #    print id,super_cell.x[index],super_cell.dx1[index],super_cell.dx2[index],super_cell.dx3[index]
        return super_cell

    def build_super_cell2_simple(self,ref_domain_original,index_for_atoms=[0,1,4,5]):
    #in this simple mode: a library instead of domain instance will be created, and the key for library is (id,el)
    #so the library looks like: {(id1,el):coords,(id2,el):coords2}
        container={}
        def _round_up(x):
            new_x=x
            while 1:
                if new_x>=0 and new_x<=1:
                    new_x=x
                    break
                elif new_x<0:
                    new_x=new_x+1
                elif new_x>1:
                    new_x=new_x-1
            return new_x

        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([_round_up(x),_round_up(y),z])

        for i in index_for_atoms:
            id=str(ref_domain_original.id[i])
            container[id,str(ref_domain_original.el[i])]=_extract_coor(ref_domain_original,id)

        for id in container.keys():
            container[id[0]+'_+x',id[1]]=container[id]+[1,0,0]
            container[id[0]+'_-x',id[1]]=container[id]+[-1,0,0]
            container[id[0]+'_+y',id[1]]=container[id]+[0,1,0]
            container[id[0]+'_-y',id[1]]=container[id]+[0,-1,0]
            container[id[0]+'_+x-y',id[1]]=container[id]+[1,-1,0]
            container[id[0]+'_+x+y',id[1]]=container[id]+[1,1,0]
            container[id[0]+'_-x+y',id[1]]=container[id]+[-1,1,0]
            container[id[0]+'_-x-y',id[1]]=container[id]+[-1,-1,0]

        return container

    def create_equivalent_domains(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        return new_domain_A,new_domain_B

    def create_equivalent_domains_2(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #N_layers here is crystal specific, 5 for hematite(1-102) and 19 for muscovite(001)
        for id in self.id_list[:(self.terminated_layer+self.N_layers)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        new_domain_A.id=list(list(map(lambda x:x+self.domain_tag+'A',new_domain_A.id)))
        new_domain_B.id=list(list(map(lambda x:x+self.domain_tag+'B',new_domain_B.id)))
        return new_domain_A.copy(),new_domain_B.copy()

    def create_equivalent_domains_3(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        new_domain_B.x,new_domain_B.y,new_domain_B.z=1-new_domain_B.x,new_domain_B.y-0.06955,new_domain_B.z-0.5
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        temp_ids=[]
        for i in range(len(self.id_list)):
            #print id in new_domain_B.id
            if new_domain_B.z[i]<0:
                temp_ids.append(new_domain_B.id[i])
        for id in temp_ids:new_domain_B.del_atom(id)
        new_domain_B.id=new_domain_A.id[10:]
        new_domain_A.id=list(map(lambda x:x+self.domain_tag+'A',new_domain_A.id))
        new_domain_B.id=list(map(lambda x:x+self.domain_tag+'B',new_domain_B.id))
        return new_domain_A.copy(),new_domain_B.copy()

    def _extract_list(self,ref_list,extract_index):
        output_list=[]
        for i in extract_index:
            output_list.append(ref_list[i])
        return output_list

    def split_number(self,N_str):
        N_list=[]
        for i in range(len(N_str)):
            N_list.append(int(N_str[i]))
        return N_list

    def scale_opt(self,atm_gp_list,scale_factor,sign_values=None,flag='u',ref_v=1.):
        #scale the parameter from first layer atom to deeper layer atom
        #dx,dy,dz,u will decrease inward, oc decrease outward usually
        #and note the ref_v for oc and u is the value for inner most atom, while ref_v for the other parameters are values for outer most atoms
        #atm_gp_list is a list of atom group to consider the scaling operation
        #scale_factor is list of values of scale factor, note accummulated product will be used for scaling
        #flag is the parameter symbol
        #ref_v is the reference value to start off
        if sign_values==None:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1]))
        else:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*sign_values[i]*reduce(mul,scale_factor[:i+1]))

    def extract_scale_opts(self,filename):
        self.scale_opt_arr=np.array([[0,0,0,0,0,0]])[0:0]
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                self.scale_opt_arr=np.append(self.scale_opt_arr,[[line_split[i] for i in range(6)]],axis=0)
        f.close()

    #this function do the same thing as the previous one, but it call open/close file only once by assigning the file content to self.scale_opt_arr
    def scale_opt_batch(self,filename):
        #note:the original name is scale_opt_batch2b
        try:
            test=self.scale_opt_arr
        except:
            self.extract_scale_opts(filename)
        for line in self.scale_opt_arr:
            atm_gp_list=vars(self)[line[0]]
            index_list=self.split_number(line[1])
            scale_factor=vars(self)[line[2]]
            sign_values=0.
            if line[3]=='None':
                sign_values=None
            else:
                sign_values=vars(self)[line[3]]
            flag=line[4]
            ref_v=0.
            try:
                ref_v=float(line[5])
            except:
                ref_v=vars(self)[line[5]]

            self.scale_opt(self._extract_list(atm_gp_list,index_list),scale_factor,sign_values,flag,ref_v)

    def set_new_vars(self,head_list=['u_Fe_'],N_list=[2]):
    #set new vars
    #head_list is a list of heading test for a new variable,N_list is the associated number of each set of new variable to be created
        for head,N in zip(head_list,N_list):
            for i in range(N):
                getattr(self.new_var_module,'new_var')(head+str(i+1),1.)

    def set_discrete_new_vars_batch(self,filename):
    #set discrete new vars
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                #print line_split
                getattr(self.new_var_module,'new_var')(line_split[0],float(line_split[1]))
        f.close()

    def norm_sign(self,value,scale=1.):
        if value<=0.5:
            return -scale
        elif value>0.5:
            return scale

    def extract_sim(self,filename):
        self.sim_val=[]
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=list(line.rsplit(','))
                self.sim_val.append(line_split[:-1])
        f.close()

    def init_sim_batch(self,filename):
    #note: original name is init_sim_batch2
        try:
            test=self.sim_val
        except:
            self.extract_sim(filename)
        #print self.sim_val
        for line in self.sim_val:

            if (line[0]=='ocu')|(line[0]=='scale'):
                tmp_list=[]
                for i in range(len(line)-2):
                    #print line
                    tmp_list.append(getattr(self.new_var_module,line[i+2]))
                setattr(self,line[1],tmp_list)
            elif line[0]=='ref':
                tmp=getattr(vars(self)[line[2]],line[3])()
                setattr(self,line[1],tmp)
            elif line[0]=='ref_new':
                tmp=getattr(self.new_var_module,line[2])
                setattr(self,line[1],tmp)
            elif line_split[0]=='sign':
                tmp_list=[]
                for i in range(len(line)-2):
                    tmp_list.append(self.norm_sign(getattr(self.new_var_module,line[i+2])))
                setattr(self,line[1],tmp_list)

    def create_grid_number(self,atm,domain):
        atm_coor=extract_coor(domain,atm)
        x,y=atm_coor[0],atm_coor[1]
        #print atm,atm_coor
        a,b,c,d= -0.3,0.3,0.9,1.5
        if ((x>=a) & (x<b))&((y>=a) & (y<b)):
            return 7
        elif ((x>= b) & (x< c))&((y>= a) & (y< b )):
            return 8
        elif ((x>=c) & (x<d))&((y>= a) & (y<b)):
            return 9
        elif ((x>= a) & (x<b))&((y>=b) & (y<c)):
            return 6
        elif ((x>=b) & (x<c))&((y>=b) & (y<c)):
            return 5
        elif ((x>=c) & (x<d))&((y>=b) & (y<c)):
            return 4
        elif ((x>= a) & (x<b))&((y>=c) & (y<d)):
            return 1
        elif ((x>=b) & (x<c))&((y>=c) & (y<d)):
            return 2
        elif ((x>=c) & (x<d))&((y>=c) & (y<d)):
            return 3

    def compare_grid(self,grid1,grid2):
        if grid1==grid2:
            return None
        else:
            return grid_match_lib[grid1][grid2]

    def find_neighbors(self,domain,id,searching_range=2.3,basis=np.array([5.038,5.434,7.3707]),T=None):
        neighbor_container=[]
        atm_ids=[]
        offset=[]
        full_offset=['+x','-x','+y','-y','+x-y','+x+y','-x+y','-x-y']
        if T==None:
            f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        else:
            f1=lambda domain,index:np.dot(T,np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis)
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #print domain.id,id
        index=np.where(domain.id==id)[0][0]
        [neighbor_container.append(domain.id[i]) for i in range(len(domain.id)) if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.)]
        for i in neighbor_container:
            if i.rsplit('_')[-1] in full_offset:
                atm_ids.append('_'.join(i.rsplit('_')[:-1]))
                offset.append(i.rsplit('_')[-1])
            else:
                atm_ids.append(i)
                offset.append(None)
        return atm_ids,offset

    def find_neighbors2(self,domain,id,searching_range=2.3):
        neighbor_container={}
        full_offset=['+x','-x','+y','-y','+x-y','+x+y','-x+y','-x-y']
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #print domain.id,id
        index=np.where(domain.id==id)[0][0]

        for i in range(len(domain.id)):
            if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.):
                neighbor_container[domain.id[i]]=f2(f1(domain,index),f1(domain,i))
        print("neighbors of ",id," is as following:")
        for key in neighbor_container.keys():
            print(key,neighbor_container[key])
        return None

    def create_match_lib(self,domain,id_list):
        basis=np.array([5.038,5.434,7.3707])
        match_lib={}
        for i in id_list:
            match_lib[i]=[]
        f1=lambda domain,index:np.array([domain.x[index],domain.y[index],domain.z[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #index=np.where(domain.id==center_atom_id)[0][0]
        for i in range(len(id_list)):
            index_1=np.where(domain.id==id_list[i])[0][0]
            for j in range(len(domain.id)):
                index_2=np.where(domain.id==domain.id[j])[0][0]
                if (f2(f1(domain,index_1),f1(domain,index_2))<2.5):
                    print(f2(f1(domain,index_1),f1(domain,index_2)))
                    match_lib[id_list[i]].append(domain.id[j])
        return match_lib

    def cal_bond_valence1(self,domain,center_atom_id,searching_range=3.,print_file=False):
    #calculate the bond valence for an atom with specified surroundring atoms by using the equation s=exp[(r0-r)/B]
    #the atoms within the searching range (in unit of A) will be counted for the calculation of bond valence
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        index=np.where(domain.id==center_atom_id)[0][0]
        #print center_atom_id,f1(domain,index)
        for i in range(len(domain.id)):
            if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.):
                r0=0
                if ((domain.el[index]=='Pb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Pb')):r0=r0_Pb
                elif ((domain.el[index]=='Fe')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Fe')):r0=1.759
                elif ((domain.el[index]=='Sb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Sb')):r0=1.973
                else:r0=-10
                bond_valence_container[domain.id[i]]=np.exp((r0-f2(f1(domain,index),f1(domain,i)))/0.37)
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2(self,domain,center_atom_id,searching_range=2.5,coordinated_atms=[],wt=100,print_file=False):
    #calculate the bond valence for an atom with specified surroundring atoms by using the equation s=exp[(r0-r)/B]
    #the atoms within the searching range (in unit of A) will be counted for the calculation of bond valence
    #different from version one:only consider complexing ligands defined in coordinated_atms, for any other atoms the calculated
    #bv will be weighted by multiplying by wt, which is usually a high number for penalty purpose
    #that way the role for different sorbate will be more distinguishable (eg water wont be close to the sorbate)
    #ids in coordinated_atms look like 'O1_2_0', you dont have to give the full name as 'O1_2_0_D1A'
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=np.where(domain.id==center_atom_id)[0][0]
        #print center_atom_id,f1(domain,index)
        for i in range(len(domain.id)):

            dist=f2(f1(domain,index),f1(domain,i))
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):
                r0=0
                if ((domain.el[index]=='Pb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Pb')):r0=r0_Pb
                elif ((domain.el[index]=='Fe')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Fe')):r0=1.759
                elif ((domain.el[index]=='Sb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Sb')):r0=1.973
                elif ((domain.el[index]=='O')&(domain.el[i]=='O')):
                    if dist<2.:
                        r0=2.#arbitrary r0 here, ensure oxygens not too close to each other
                    else:r0=-1
                else:
                    if dist<2.3:r0=10.#exclude the situation where cations are closer than the searching range (2.5A in this case)
                    else:r0=-10
                sum_check=0
                for atm in coordinated_atms:
                    if atm in str(domain.id[i]):
                        sum_check+=1
                if sum_check==1:
                    bond_valence_container[domain.id[i]]=np.exp((r0-f2(f1(domain,index),f1(domain,i)))/0.37)
                else:
                    bond_valence_container[domain.id[i]]=np.exp((r0-f2(f1(domain,index),f1(domain,i)))/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B(self,domain,center_atom_id,center_atom_el,searching_range=2.5,coordinated_atms=[],wt=100,print_file=False,O_cutoff_limit=2.5):
        #different from new2:domain is a library in the format {(key,el):[x,y,z]}
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):

                r0=0
                if ((index[1]=='H')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='H')):r0=0.677
                if ((index[1]=='Pb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Pb')):r0=r0_Pb
                elif ((index[1]=='Fe')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Fe')):r0=1.759
                elif ((index[1]=='Sb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Sb')):r0=1.973
                elif ((index[1]=='O')&(key[1]=='O')):
                    if dist<O_cutoff_limit:
                        r0=20.#arbitrary r0 here, ensure oxygens are more than 2.65A apart
                    else:r0=-10
                elif ((index[1]=='H')&(key[1]=='H')):
                    if dist<1.5:
                        r0=0.677
                        dist=0.679 #arbitrary distance to ensure the bv result is a huge number
                    else:
                        r0=0.677
                        dist=20 #arbitrary distance to ensure the bv result is a tiny tiny number
                else:
                    if ((index[1]!='H')&(key[1]!='H')):
                        r0=-10#allow short sorbate-sorbate distance for consideration of multiple sorbate within one average structure
                    else:
                        if dist<2:#cations are not allowed to be closer than 2A to hydrogen atom
                            r0=0.677
                            dist=0.679
                        else:#ignore it if the distance bw cation and Hydrogen is less than 2.5 but higher than 2. A
                            r0=0.677
                            dist=20
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                    elif 'HB' in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B_4(self,domain,center_atom_id,center_atom_el,searching_range=2.5,coordinated_atms=[],wt=100,print_file=False,r0_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},O_cutoff_limit=2.5):
        #different from new2B: add a argument containing info of r0 for possible couples
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):

                r0=0
                if (index[1],key[1]) in r0_container.keys():
                    r0=r0_container[(index[1],key[1])]
                elif (key[1],index[1]) in r0_container.keys():
                    r0=r0_container[(key[1],index[1])]
                elif ((index[1]=='O')&(key[1]=='O')):
                    if dist<O_cutoff_limit:
                        r0=20.#arbitrary r0 here, ensure oxygens are more than 2.65A apart
                    else:r0=-10
                elif ((index[1]=='H')&(key[1]=='H')):
                    if dist<1.5:
                        r0=0.677
                        dist=0.679 #arbitrary distance to ensure the bv result is a huge number
                    else:
                        r0=0.677
                        dist=20 #arbitrary distance to ensure the bv result is a tiny tiny number
                else:
                    if ((index[1]!='H')&(key[1]!='H')):
                        r0=-10#allow short sorbate-sorbate distance for consideration of multiple sorbate within one average structure
                    else:
                        if dist<2:#cations are not allowed to be closer than 2A to hydrogen atom
                            r0=0.677
                            dist=0.679
                        else:#ignore it if the distance bw cation and Hydrogen is less than 2.5 but higher than 2. A
                            r0=0.677
                            dist=20
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                    elif 'HB' in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B_5(self,domain,center_atom_id,center_atom_el,searching_range_offset=0.2,ideal_bond_len_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},coordinated_atms=[],wt=100,print_file=False,r0_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},O_cutoff_limit=2.5):
        #different from new2B: add a argument containing info of r0 for possible couples
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            searching_range=None
            if (index[1],key[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(index[1],key[1])]+searching_range_offset
            elif (key[1],index[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(key[1],index[1])]+searching_range_offset
            else:
                searching_range=2.5
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):

                r0=0
                if (index[1],key[1]) in r0_container.keys():
                    r0=r0_container[(index[1],key[1])]
                elif (key[1],index[1]) in r0_container.keys():
                    r0=r0_container[(key[1],index[1])]
                elif ((index[1]=='O')&(key[1]=='O')):
                    if dist<O_cutoff_limit:
                        r0=20.#arbitrary r0 here, ensure oxygens are more than 2.65A apart
                    else:r0=-10
                elif ((index[1]=='H')&(key[1]=='H')):
                    if dist<1.5:
                        r0=0.677
                        dist=0.679 #arbitrary distance to ensure the bv result is a huge number
                    else:
                        r0=0.677
                        dist=20 #arbitrary distance to ensure the bv result is a tiny tiny number
                else:
                    if ((index[1]!='H')&(key[1]!='H')):
                        r0=-10#allow short sorbate-sorbate distance for consideration of multiple sorbate within one average structure
                    else:
                        if dist<2:#cations are not allowed to be closer than 2A to hydrogen atom
                            r0=0.677
                            dist=0.679
                        else:#ignore it if the distance bw cation and Hydrogen is less than 2.5 but higher than 2. A
                            r0=0.677
                            dist=20
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                    elif 'HB' in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B_6(self,domain,center_atom_id,center_atom_el,searching_range_offset=0.2,ideal_bond_len_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},coordinated_atms=[],wt=100,print_file=False,r0_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},O_cutoff_limit=2.5,waiver_atoms=[]):
        #different from version 5: there is a waiver_atom list, when each two inside the list are being considered for bond valence, such bond valence constrain will be ignored
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            searching_range=None
            if (index[1],key[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(index[1],key[1])]+searching_range_offset
            elif (key[1],index[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(key[1],index[1])]+searching_range_offset
            else:
                searching_range=2.5
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):
                r0=0
                if (index[1],key[1]) in r0_container.keys():
                    r0=r0_container[(index[1],key[1])]
                elif (key[1],index[1]) in r0_container.keys():
                    r0=r0_container[(key[1],index[1])]
                elif ((index[1]=='O')&(key[1]=='O')):
                    if sum([each in key[0] for each in waiver_atoms])==1 and sum([eachcase in center_atom_id for eachcase in waiver_atoms])==1:
                        r0=-10
                    else:
                        if dist<O_cutoff_limit:
                            r0=20.#arbitrary r0 here, ensure oxygens are more than 2.65A apart
                        else:r0=-10
                elif ((index[1]=='H')&(key[1]=='H')):
                    if dist<1.5:
                        r0=0.677
                        dist=0.679 #arbitrary distance to ensure the bv result is a huge number
                    else:
                        r0=0.677
                        dist=20 #arbitrary distance to ensure the bv result is a tiny tiny number
                else:
                    if ((index[1]!='H')&(key[1]!='H')):
                        r0=-10#allow short sorbate-sorbate distance for consideration of multiple sorbate within one average structure
                    else:
                        if dist<2:#cations are not allowed to be closer than 2A to hydrogen atom
                            r0=0.677
                            dist=0.679
                        else:#ignore it if the distance bw cation and Hydrogen is less than 2.5 but higher than 2. A
                            r0=0.677
                            dist=20
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                    elif 'HB' in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    #different from v6: not only consider for the bond valence sum but also consider for the coordination situation compared to the pre-defined coordinated members
    def cal_bond_valence1_new2B_7(self,domain,center_atom_id,center_atom_el,searching_range_offset=0.2,ideal_bond_len_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},coordinated_atms=[],wt=100,print_file=False,r0_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},O_cutoff_limit=2.5,waiver_atoms=[],basis=np.array([5.038,5.434,7.3707]),T=None,T_INV=None):
        #different from version 5: there is a waiver_atom list, when each two inside the list are being considered for bond valence, such bond valence constrain will be ignored
        bond_valence_container={}
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        for key in domain.keys():
            if T==None:
                dist=f2(domain[key]*basis,domain[index]*basis)
            else:
                dist=f2(np.dot(T,domain[key]*basis),np.dot(T,domain[index]*basis))
            searching_range=None
            if (index[1],key[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(index[1],key[1])]+searching_range_offset
            elif (key[1],index[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(key[1],index[1])]+searching_range_offset
            else:
                searching_range=2.5
            #print domain.id[i],center_atom_id,dist
            #if center_atom_id=='O6_3_0_D2A' and dist<2.5:
            #    print key[0],domain[key],dist
            if (dist<=searching_range)&(dist!=0.):
                r0=0
                if (index[1],key[1]) in r0_container.keys():
                    r0=r0_container[(index[1],key[1])]
                elif (key[1],index[1]) in r0_container.keys():
                    r0=r0_container[(key[1],index[1])]
                elif ((index[1]=='O')&(key[1]=='O')):
                    if sum([each in key[0] for each in waiver_atoms])==1 and sum([eachcase in center_atom_id for eachcase in waiver_atoms])==1:
                        r0=-10
                    else:
                        if dist<O_cutoff_limit:
                            r0=20.#arbitrary r0 here, ensure oxygens are more than 2.65A apart
                        else:r0=-10
                elif ((index[1]=='H')&(key[1]=='H')):
                    if dist<1.5:
                        r0=0.677
                        dist=0.679 #arbitrary distance to ensure the bv result is a huge number
                    else:
                        r0=0.677
                        dist=20 #arbitrary distance to ensure the bv result is a tiny tiny number
                else:
                    if ((index[1]!='H')&(key[1]!='H')):
                        r0=-10#allow short sorbate-sorbate distance for consideration of multiple sorbate within one average structure
                    else:
                        if dist<2:#cations are not allowed to be closer than 2A to hydrogen atom
                            r0=0.677
                            dist=0.679
                        else:#ignore it if the distance bw cation and Hydrogen is less than 2.5 but higher than 2. A
                            r0=0.677
                            dist=20
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                    elif 'HB' in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        #Trigger penalty for under-coordination situation
        if len(bond_valence_container.keys())<len(coordinated_atms) and center_atom_el!='O':
            sum_valence=sum_valence*wt
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B_7_2(self,domain,center_atom_id,center_atom_el,searching_range_offset=0.2,ideal_bond_len_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},coordinated_atms=[],wt=100,print_file=False,r0_container={('Fe','O'):1.759,('H','O'):0.677,('Pb','O'):2.04,('Sb','O'):1.973},O_cutoff_limit=2.5,waiver_atoms=[],basis=np.array([5.038,5.434,7.3707]),T=None,T_INV=None,check=False):
        #different from version 5: there is a waiver_atom list, when each two inside the list are being considered for bond valence, such bond valence constrain will be ignored
        bond_valence_container={}
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        output_bv_container={}
        sensor=False
        for key in domain.keys():
            if T==None:
                dist=f2(domain[key]*basis,domain[index]*basis)
            else:
                dist=f2(np.dot(T,domain[key]*basis),np.dot(T,domain[index]*basis))
            searching_range=None
            if (index[1],key[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(index[1],key[1])]+searching_range_offset
            elif (key[1],index[1]) in ideal_bond_len_container.keys():
                searching_range=ideal_bond_len_container[(key[1],index[1])]+searching_range_offset
            else:
                searching_range=2.5
            #print domain.id[i],center_atom_id,dist
            #if center_atom_id=='O6_3_0_D2A' and dist<2.5:
            #    print key[0],domain[key],dist
            if (dist<=searching_range)&(dist!=0.):
                r0=0
                if (index[1],key[1]) in r0_container.keys():
                    r0=r0_container[(index[1],key[1])]
                elif (key[1],index[1]) in r0_container.keys():
                    r0=r0_container[(key[1],index[1])]
                elif ((index[1]=='O')&(key[1]=='O')):
                    if sum([each in key[0] for each in waiver_atoms])==1 or sum([eachcase in center_atom_id for eachcase in waiver_atoms])==1:
                        r0=-10
                    else:
                        if dist<O_cutoff_limit:
                            r0=20.#arbitrary r0 here, ensure oxygens are more than 2.65A apart
                        else:r0=-10
                elif ((index[1]=='H')&(key[1]=='H')):
                    if dist<1.5:
                        r0=0.677
                        dist=0.679 #arbitrary distance to ensure the bv result is a huge number
                    else:
                        r0=0.677
                        dist=20 #arbitrary distance to ensure the bv result is a tiny tiny number
                else:
                    if ((index[1]!='H')&(key[1]!='H')):
                        r0=-10#allow short sorbate-sorbate distance for consideration of multiple sorbate within one average structure
                    else:
                        if dist<2:#cations are not allowed to be closer than 2A to hydrogen atom
                            r0=0.677
                            dist=0.679
                        else:#ignore it if the distance bw cation and Hydrogen is less than 2.5 but higher than 2. A
                            r0=0.677
                            dist=20
                if sum([eachcase in center_atom_id for eachcase in waiver_atoms])==1 or sum([eachcase in key[0] for eachcase in waiver_atoms])==1:#atoms being waived wont be considered for BV constraint
                    r0=0
                    dist=20
                    #print center_atom_id,key[0],'sensor here'
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                    elif 'HB' in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                        output_bv_container[key[0]]=(dist,np.exp((r0-dist)/0.37))
                else:
                    if r0==0.677:
                        bond_valence_container[key[0]]=0.24/(dist-r0)
                    else:
                        bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                        output_bv_container[key[0]]=(dist,np.exp((r0-dist)/0.37))
                        sensor=True

                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        #Trigger penalty for under-coordination situation
        if len(bond_valence_container.keys())<len(coordinated_atms) and center_atom_el!='O':
            sum_valence=sum_valence*wt
            sensor=True
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        if not check:
            return bond_valence_container
        else:
            if sensor:
                print('atom under consideration:',center_atom_id)
                print('coordinated_atoms:',coordinated_atms)
                print('following list key,(distance,bond valence before scaling)')
                for key in output_bv_container.keys():
                    print(key,output_bv_container[key])
            return bond_valence_container

    def cal_bond_valence1_new2B3(self,domain,center_atom_id,center_atom_el,searching_range=2.5,coordinated_atms=[],wt=100,print_file=False):
        #different from new2B:consider a very soft limit for cation cation distance cutoff (1. instead of 2.3), everything else is the same
        #purposely be used to include sorbates into one domain
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):

                r0=0
                if ((index[1]=='Pb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Pb')):r0=r0_Pb
                elif ((index[1]=='Fe')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Fe')):r0=1.759
                elif ((index[1]=='Sb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Sb')):r0=1.973
                elif ((index[1]=='O')&(key[1]=='O')):
                    if dist<2.:
                        r0=2.#arbitrary r0 here, ensure oxygens not too close to each other
                    else:r0=-1
                else:
                    if dist<1:
                        r0=10.#exclude the situation where cations are closer than the searching range (2.5A in this case)
                    else:r0=-10
                sum_check=0
                for atm in coordinated_atms:
                    if atm in key[0]:
                        sum_check+=1
                #print "sum_check",sum_check
                if sum_check==1:
                    bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B_2(self,domain,center_atom_id,center_atom_el,searching_range=2.5,coordinated_atms=[],wt=100,print_file=False):
        #different from new2B:now consider panalty for distortion associated with bond length
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        ligand_container={}
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            #print key[0],index[0],dist
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):
                r0=0
                if ((index[1]=='Pb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Pb')):r0=r0_Pb
                elif ((index[1]=='Fe')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Fe')):r0=1.759
                elif ((index[1]=='Sb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Sb')):r0=1.973
                elif ((index[1]=='O')&(key[1]=='O')):
                    if dist<2.6:#2.6 A is the typical distance for hydrogen bond with bond valence equivalent to 0.25 v.u.
                        r0=2.#arbitrary r0 here, ensure oxygens not too close to each other
                    else:r0=-1
                else:
                    if dist<2.3:
                        r0=10.#exclude the situation where cations are closer than the searching range (2.5A in this case)
                    else:r0=-10
                if key[0] in coordinated_atms:
                    ligand_container[key]=dist
                    bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
                else:
                    bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)*wt

        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        dists=[]
        for ligand in coordinated_atms:
            el_ligand='O'
            if 'Pb' in ligand:el_ligand='Pb'
            elif 'Sb' in ligand:el_ligand='Sb'
            #print ligand_container.keys()
            if (ligand,el_ligand) in ligand_container.keys():
                #print ligand
                dists.append(ligand_container[ligand,el_ligand])
            else:
                dists.append(f2(domain[ligand,el_ligand]*basis,domain[index]*basis))
        bond_length_distortion=max(dists)-min(dists)
        wt_distortion=0
        if bond_length_distortion<0.2:pass
        elif bond_length_distortion>=0.2 and bond_length_distortion<0.5: wt_distortion=0.5
        else:wt_distortion=1
        bond_valence_container['wt_distortion']=10**wt_distortion
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_bond_valence1_new2B_3(self,domain,center_atom_id,center_atom_el,searching_range=2.5):
        #different from new2B_2:there wont be any panalty and distortion function
        bond_valence_container={}
        match_lib={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        index=(center_atom_id,center_atom_el)
        #print center_atom_id,f1(domain,index)
        ligand_container={}
        for key in domain.keys():

            dist=f2(domain[key]*basis,domain[index]*basis)
            #print key[0],index[0],dist
            #print domain.id[i],center_atom_id,dist
            if (dist<=searching_range)&(dist!=0.):
                r0=0
                if ((index[1]=='Pb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Pb')):r0=r0_Pb
                elif ((index[1]=='Fe')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Fe')):r0=1.759
                elif ((index[1]=='Sb')&(key[1]=='O'))|((index[1]=='O')&(key[1]=='Sb')):r0=1.973
                elif ((index[1]=='O')&(key[1]=='O')):
                    if dist<2.65:#2.6 A is the typical distance for hydrogen bond with bond valence equivalent to 0.25 v.u.
                        r0=20.#arbitrary r0 here, ensure oxygens not too close to each other
                    else:r0=-1
                else:
                    if dist<2.3:
                        r0=10.#exclude the situation where cations are closer than the searching range (2.5A in this case)
                    else:r0=-10
                bond_valence_container[key[0]]=np.exp((r0-dist)/0.37)
        sum_valence=0.
        sum_valence=sum(bond_valence_container.values())
        sorted_list=sorted(bond_valence_container.iteritems(),key=operator.itemgetter(1))[::-1]
        N_ligand=1
        if center_atom_el=='Pb':N_ligand=3
        elif center_atom_el=='Sb':N_ligand=6
        offsets=['+x','-x','+y','-y','+x+y','-x-y','+x-y','-x+y']
        offsets_opposit=['-x','+x','-y','+y','-x-y','+x+y','-x+y','+x-y']
        for key in sorted_list[0:N_ligand]:
            id=key[0]
            id_list=id.rsplit('_')
            if id_list[-1] in offsets:
                id_list[-1]=offsets_opposit[offsets.index(id_list[-1])]
                match_lib['_'.join(id_list[:-1])]=center_atom_id+'_'+id_list[-1]
            else:
                match_lib['_'.join(id_list)]=center_atom_id

        bond_valence_container['total_valence']=sum_valence

        return bond_valence_container,match_lib

    def cal_hydrogen_bond_valence(self,domain,center_atom_id,searching_range=2.5,coordinated_atms=[],wt=100,print_file=False):
    #calculate the bond valence for an atom with specified surroundring atoms by using the equation s=exp[(r0-r)/B]
    #the atoms within the searching range (in unit of A) will be counted for the calculation of bond valence
    #different from version one:only consider complexing ligands defined in coordinated_atms, for any other atoms the calculated
    #bv will be weighted by multiplying by wt, which is usually a high number for penalty purpose
    #that way the role for different sorbate will be more distinguishable (eg water wont be close to the sorbate)
    #ids in coordinated_atms look like 'O1_2_0', you dont have to give the full name as 'O1_2_0_D1A'
    #similar as previous function but use f3 to cal the bond valence of hydrogen bond
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #f3 is a empirical polynormial equation to cal hydrogen bond valence based on Fig2 in ID Brown_Acta Cryst_1985.B41,244-247
        f3=lambda x:8.04706*x**4-93.229416*x**3+403.91415*x**2-775.95379*x+558.10065
        index=np.where(domain.id==center_atom_id)[0][0]
        #print center_atom_id,f1(domain,index)

        for i in range(len(domain.id)):
            if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.):
                sum_check=0
                for atm in coordinated_atms:
                    if atm in str(domain.id[i]):
                        sum_check+=1
                #print domain.id[index],f1(domain,index)
                #print domain.id[i],f1(domain,i)
                if sum_check==1:
                    bond_valence_container[domain.id[i]]=f3(f2(f1(domain,index),f1(domain,i)))
                else:
                    bond_valence_container[domain.id[i]]=f3(f2(f1(domain,index),f1(domain,i)))*wt
                #print domain.id[i],f1(domain,i)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container

    def cal_hydrogen_bond_valence2(self,domain,center_atom_id,searching_range=3.0,acceptable_min=2.5):
        #different from version one: set the acceptable shortest distance between oxygens, if actual distance is
        #shorter than that, return a panalty number 10 instead of calculating the real hydrogen bond valence
        #so any distance number hihger than 2.5 will be equivalent
        bond_valence=0
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        index=np.where(domain.id==center_atom_id)[0][0]

        for i in range(len(domain.id)):
            if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.):
                if f2(f1(domain,index),f1(domain,i))>=acceptable_min:
                    pass
                else:
                    bond_valence=10
                    break
        return bond_valence

    def cal_hydrogen_bond_valence2B(self,domain,center_atom_id,searching_range=3.0,acceptable_min=2.5,waiver_atoms=[]):
        #different from valence2:domain is a library in form of {(id,el):coords1}
        bond_valence=0
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        center_atm_key=(center_atom_id,'O')
        for i in domain.keys():
            if sum([each in i[0] for each in waiver_atoms])==1 and sum([eachcase in center_atom_id for eachcase in waiver_atoms])==1:
                pass
            else:
                dist=f2(domain[center_atm_key]*basis,domain[i]*basis)
                if (dist<=searching_range)&(dist!=0.):
                    if dist>=acceptable_min:
                        pass
                    else:
                        bond_valence=10
                        break
        return bond_valence

    def cal_bond_valence2(self,domain,center_atm,match_list):
        #center_atm='O1',match_list=[['Fe1','Fe2'],['-x','+y']]
        #return a library showing the bond valence contribution to O1
        #calculate the bond valence of (in this case) O1_Fe1, O1_Fe2, where Fe1 and Fe2 have offset defined by '-x' and '+y' respectively.
        #return a lib with keys in match_list[0], the value for each key is the bond valence calculated
        bond_valence_container={}
        match_list.append(0)
        for i in match_list[0]:
            bond_valence_container[i]=0
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis

        index=np.where(domain.id==center_atm)[0][0]
        for k in range(len(match_list[0])):
            j=match_list[0][k]
            index2=np.where(domain.id==j)[0][0]
            dist=f2(f1(domain,index),f1(domain,index2)+_offset_translate(match_list[1][k]))
            r0=0
            if ((domain.el[index]=='Pb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Pb')&(domain.el[index]=='O')):r0=r0_Pb
            elif ((domain.el[index]=='Fe')&(domain.el[index2]=='O'))|((domain.el[index2]=='Fe')&(domain.el[index]=='O')):r0=1.759
            elif ((domain.el[index]=='Sb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Sb')&(domain.el[index]=='O')):r0=1.973
            else:#when two atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                if dist<2.:r0=10
                else:r0=0.
                #if (i=='pb1'):
                #print j,str(match_lib[i][1][k]),dist,'pb_coor',f1(domain,index)/basis,'O_coor',(f1(domain,index2)+_offset_translate(match_lib[i][1][k]))/basis,np.exp((r0-dist)/0.37)
            if dist<3.:#take it counted only when they are not two far away

                bond_valence_container[j]=np.exp((r0-dist)/0.37)
                #print j,extract_coor(domain,j),dist,bond_valence_container[j]
                match_list[2]=match_list[2]+1
        """
        for i in bond_valence_container.keys():
            #try to add hydrogen or hydrogen bond to the oxygen with 1.6=2*OH, 1.=OH+H, 0.8=OH and 0.2=H
            index=np.where(domain.id==i)[0][0]
            if (domain.el[index]=='O')|(domain.el[index]=='o'):
                case_tag=match_lib[i][2]
                bond_valence_corrected_value=[0.]
                if case_tag==1.:
                    bond_valence_corrected_value=[1.8,1.6,1.2,1.,0.8,0.6,0.4,0.2,0.]
                elif case_tag==2.:
                    bond_valence_corrected_value=[1.6,1.,0.8,0.4,0.2,0.]
                elif case_tag==3.:
                    bond_valence_corrected_value=[0.8,0.2,0.]
                else:pass
                #bond_valence_corrected_value=[1.6,1.,0.8,0.2,0.]
                ref=np.sign(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)*(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)
                bond_valence_container[i]=bond_valence_container[i]+bond_valence_corrected_value[np.where(ref==np.min(ref))[0][0]]
        """
        cum=sum([bond_valence_container[key] for key in bond_valence_container.keys()])
        bond_valence_container['total']=cum
        return bond_valence_container

    def cal_bond_valence3(self,domain,match_lib):
        #match_lib={'O1':[['Fe1','Fe2'],['-x','+y']]}
        #calculate the bond valence of (in this case) O1_Fe1, O1_Fe2, where Fe1 and Fe2 have offset defined by '-x' and '+y' respectively.
        #return a lib with the same key as match_lib, the value for each key is the bond valence calculated
        bond_valence_container={}
        for i in match_lib.keys():
            try:
                match_lib[i][2]=0
            except:
                match_lib[i].append(0)
            bond_valence_container[i]=0

        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis

        for i in match_lib.keys():
            index=np.where(domain.id==i)[0][0]
            for k in range(len(match_lib[i][0])):
                j=match_lib[i][0][k]
                index2=np.where(domain.id==j)[0][0]
                dist=f2(f1(domain,index),f1(domain,index2)+_offset_translate(match_lib[i][1][k]))
                r0=0
                if ((domain.el[index]=='Pb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Pb')&(domain.el[index]=='O')):r0=r0_Pb
                elif ((domain.el[index]=='Fe')&(domain.el[index2]=='O'))|((domain.el[index2]=='Fe')&(domain.el[index]=='O')):r0=1.759
                elif ((domain.el[index]=='Sb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Sb')&(domain.el[index]=='O')):r0=1.973
                else:#when two atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                    if dist<2.:r0=10
                    else:r0=0.
                #if (i=='pb1'):
                    #print j,str(match_lib[i][1][k]),dist,'pb_coor',f1(domain,index)/basis,'O_coor',(f1(domain,index2)+_offset_translate(match_lib[i][1][k]))/basis,np.exp((r0-dist)/0.37)
                if dist<3.:#take it counted only when they are not two far away
                    bond_valence_container[i]=bond_valence_container[i]+np.exp((r0-dist)/0.37)
                    match_lib[i][2]=match_lib[i][2]+1

        for i in bond_valence_container.keys():
            #try to add hydrogen or hydrogen bond to the oxygen with 1.6=2*OH, 1.=OH+H, 0.8=OH and 0.2=H
            index=np.where(domain.id==i)[0][0]
            if (domain.el[index]=='O')|(domain.el[index]=='o'):
                case_tag=match_lib[i][2]
                bond_valence_corrected_value=[0.]
                if ((case_tag==1.)&(bond_valence_container[i]<2)):
                    bond_valence_corrected_value=[1.8,1.6,1.2,1.,0.8,0.6,0.4,0.2,0.]
                elif ((case_tag==2.)&(bond_valence_container[i]<2)):
                    bond_valence_corrected_value=[1.6,1.,0.8,0.4,0.2,0.]
                elif ((case_tag==3.)&(bond_valence_container[i]<2)):
                    bond_valence_corrected_value=[0.8,0.2,0.]
                else:pass
                #bond_valence_corrected_value=[1.6,1.,0.8,0.2,0.]
                ref=np.sign(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)*(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)
                bond_valence_container[i]=bond_valence_container[i]+bond_valence_corrected_value[np.where(ref==np.min(ref))[0][0]]

        return bond_valence_container

    def cal_bond_valence4(self,domain,center_atm,match_id_list):
        #center_atm='O1',match_id_list=[ID1,ID2,ID3]
        #return a library showing the bond valence contribution to O1
        #calculate the bond valence of (in this case) O1_Fe1, O1_Fe2, where Fe1 and Fe2 have offset defined by '-x' and '+y' respectively.
        #return a lib with keys in match_list[0], the value for each key is the bond valence calculated
        BV=0
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis
        index=np.where(domain.id==center_atm)[0][0]
        for k in range(len(match_id_list)):
            j=match_id_list[k]
            index2=np.where(domain.id==j)[0][0]
            dist=f2(f1(domain,index),f1(domain,index2))
            #dist=scipy.spatial.distance.cdist([f1(domain,index)],[f1(domain,index2)])
            r0=0
            if ((domain.el[index]=='Pb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Pb')&(domain.el[index]=='O')):r0=r0_Pb
            elif ((domain.el[index]=='Fe')&(domain.el[index2]=='O'))|((domain.el[index2]=='Fe')&(domain.el[index]=='O')):r0=1.759
            elif ((domain.el[index]=='Sb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Sb')&(domain.el[index]=='O')):r0=1.973
            else:#when two atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                if dist<2.:r0=10
                else:r0=0.
            if dist<3.:#take it counted only when they are not two far away
                BV=BV+np.exp((r0-dist)/0.37)

        return BV

    def cal_bond_valence4B(self,domain,center_atm,match_id_list):
        #different from valence4:domain is not a super cell but a single domain, offset was used to cal the coords
        BV=0
        basis=np.array([5.038,5.434,7.3707])
        #f2=lambda p1,p2:spatial.distance.cdist([p1],[p2])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index],domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index],domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        index=np.where(domain.id==center_atm)[0][0]

        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis

        for k in range(len(match_id_list)):
            j=match_id_list[k]
            tag=j.rsplit('_')[-1]
            name=j
            offset=np.array([0,0,0])
            if tag in ['+x','-x','+y','-y','+x+y','+x-y','-x-y','-x+y']:
                offset=_offset_translate(tag)
                name='_'.join(j.rsplit('_')[0:-1])
            #print name
            index2=np.where(domain.id==name)[0][0]
            dist=f2(f1(domain,index),f1(domain,index2)+offset)
            #dist=scipy.spatial.distance.cdist([f1(domain,index)],[f1(domain,index2)])
            r0=0
            if ((domain.el[index]=='Pb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Pb')&(domain.el[index]=='O')):r0=r0_Pb
            elif ((domain.el[index]=='H')&(domain.el[index2]=='O'))|((domain.el[index2]=='O')&(domain.el[index]=='H')):r0=0.677
            elif ((domain.el[index]=='Fe')&(domain.el[index2]=='O'))|((domain.el[index2]=='Fe')&(domain.el[index]=='O')):r0=1.759
            elif ((domain.el[index]=='Sb')&(domain.el[index2]=='O'))|((domain.el[index2]=='Sb')&(domain.el[index]=='O')):r0=1.973
            else:#when two atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                if dist<2.:r0=10
                else:r0=0.
            if dist<3.:#take it counted only when they are not two far away
                if r0==0.677:
                    BV=BV+0.241/(dist-r0)
                else:
                    BV=BV+np.exp((r0-dist)/0.37)
        return BV

    def cal_bv_deficience(self,bv_container):
        bv_df=0
        for key in bv_container.keys():
            if 'Fe' in key:bv_df=bv_df+abs(3-bv_container[key])
            elif 'O' in key: bv_df=bv_df+abs(2-bv_container[key])
            elif 'Pb' in key: bv_df=bv_df+abs(2-bv_container[key])
            elif 'Sb' in key: bv_df=bv_df+abs(5-bv_container[key])
        return bv_df

    #set reference coordinate system defined by atoms with ids in domain, create the coordinate transformation matrix between the old and the new ones
    #T is 3by4 matrix with the last column defining the origin of the new coordinate system
    def create_coor_transformation(self,domain,ids):
        origin,p1,p2=extract_coor(domain,ids[0]),extract_coor(domain,ids[1]),extract_coor(domain,ids[2])
        x_v=(p1-origin)/f2(p1,origin)
        p2_o=p2-origin
        z_v=np.cross(x_v,p2_o)
        z_v=z_v/f2(np.array([0.,0.,0.]),z_v)
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        T=np.append(T,origin[:,np.newaxis],axis=1)
        return T

    #extract the r theta and phi for atom with id in the reference coordinate system
    def extract_spherical_pars(self,domain,ref_ids,id):
        T=self.create_coor_transformation(domain,ref_ids)
        coors_old=extract_coor(domain,id)-T[:,-1]
        coors_new=np.dot(T[:,0:-1],coors_old)
        x,y,z=coors_new[0],coors_new[1],coors_new[2]
        r=f2(np.array([0.,0.,0.]),coors_new)
        theta=np.arccos(z/r)
        phi=0
        if (x>0) & (y>0):
            phi=np.arctan(y/x)
        elif (x>0) & (y<0):
            phi=2*np.pi+np.arctan(y/x)
        elif (x<0) & (y>0)|(x<0) & (y<0):
            phi=np.pi+np.arctan(y/x)
        return r,theta,phi

    #calculate xyz in old coordinate system from spherical system and set it to atom with id
    def set_sorbate_xyz(self,domain,ref_ids,r_theta_phi,id):
        T=self.create_coor_transformation(domain,ref_ids)
        r,theta,phi=r_theta_phi[0],r_theta_phi[1],r_theta_phi[2]
        x=r*np.sin(theta)*np.cos(phi)
        y=r*np.sin(theta)*np.sin(phi)
        z=r*np.cos(theta)
        coors_new=np.array([x,y,z])
        coors_old=np.dot(inv(T[:,0:-1]),coors_new)+T[:,-1]
        set_coor(domain,id,coors_old)
