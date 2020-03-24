# -*- coding: utf-8 -*-
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

class HandleVars(object):
    def __init__(self):
        pass

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