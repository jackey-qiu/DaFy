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

"""

class DomainBuilder(object):
    def __init__(self,ref_domain):
        #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
        #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
        #function of set_new_vars
        #N_layers is the layer offset between two symmetry related terminations, default value 5 is for rcut hematite specifically
        self.ref_domain=ref_domain
        self.terminated_layer = 0
        self.domain_tag = ''
        #self.create_domain()

    def create_domain(self,terminated_layer=0, domain_tag = '_D1'):
        self.terminated_layer = terminated_layer
        self.domain_tag = domain_tag
        new_domain=self.ref_domain.copy()
        for id in new_domain.id[:self.terminated_layer*2]:
            new_domain.del_atom(id)
        new_domain.id=list(map(lambda x:x+self.domain_tag,new_domain.id))
        self.domain = new_domain
        return new_domain.copy()

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




