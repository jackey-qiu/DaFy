# -*- coding: utf-8 -*-
import models.structure_tools.sxrd_dafy as model_2
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
# from coordinate_system import CoordinateSystem

ALL_MOTIF_COLLECTION = ['OCCO','CCO', 'CO']

#sorbate structure motifs
##OCCO##
structure_OCCO="""
#       O1    O2
#        \   /
#        C1-C2
#===================
"""
OCCO = {
        "els_sorbate":['O','C','C', 'O'],
        "anchor_index_list":[1, None, 1, 2 ],
        "flat_down_index": [2],
        "structure":structure_OCCO
        }

##CCO#
structure_CCO="""
#       C2
#      /  \\
#     C1   O1
#====================
"""
CCO = {
        "els_sorbate":['C','C', 'O'],
        "anchor_index_list":[None, 0, 1 ],
        "flat_down_index": [2],
        "structure":structure_CCO
        }

##CO##
structure_CO="""
#        O1   
#        |   
#        C1
#====================
"""
CO = {
        "els_sorbate":['C','O'],
        "anchor_index_list":[None, 0],
        "flat_down_index": [],
        "structure":structure_CO
        }

class CarbonOxygenMotif(object):
    def __init__(self, domain, ids, anchor_id, r = 2, delta =0, gamma = 0, flat_down_index = None, lat_pars = [3.615, 3.615, 3.615, 90, 90, 90]):
        self.domain = domain
        self.ids = ids
        self.ids_flat_down = []
        self.anchor_id = anchor_id
        self.flat_down_index = flat_down_index
        self.r_list = [r for each in ids]
        self.delta_list = [delta for i in ids]
        self.gamma_list = [gamma for each in ids]
        self.lat_abc = np.array(lat_pars[0:3])
        self.lat_angles = np.array(lat_pars[3:])
        if type(flat_down_index)!=type([]):
            pass
        else:
            for each_index in flat_down_index:
                self.delta_list[each_index] = 0
                self.ids_flat_down.append(self.ids[each_index])
        self.bond_index=[]
        #self.set_coordinate_all()

    @classmethod
    def build_instance(cls,xyzu_oc_m = [0.5, 0.5, 1.5, 0.1, 1, 1], els = ['O','C','C','O'], flat_down_index = [2],anchor_index_list = [1, None, 1, 2 ], lat_pars = [3.615, 3.615, 3.615, 90, 90, 90]):
        domain = model_2.Slab(T_factor = 'u')
        ids_all = deepcopy(els)
        for each in set(els):
            index_temp_all = list(np.where(np.array(els) == each)[0])
            for index_temp in index_temp_all:
                ids_all[index_temp] = "{}{}".format(ids_all[index_temp], index_temp_all.index(index_temp)+1)
        for i in range(len(ids_all)):
            domain.add_atom(ids_all[i],els[i],*xyzu_oc_m)
        ids = deepcopy(ids_all)
        del(ids[anchor_index_list.index(None)])
        anchor_id = ids_all[anchor_index_list.index(None)]
        flat_down_index_new = [[i,i-1][int(i>anchor_index_list.index(None))] for i in flat_down_index]

        r_list_names = []
        gamma_list_names = []
        delta_list_names = []
        for i in range(len(anchor_index_list)):
            each = anchor_index_list[i]
            if each != None:
                #if i not in flat_down_index:
                delta_list_names.append("delta_{}_{}".format(ids_all[i],ids_all[each]))
            if each != None:
                r_list_names.append("r_{}_{}".format(ids_all[i],ids_all[each]))
                gamma_list_names.append("gamma_{}_{}".format(ids_all[i],ids_all[each]))
        rgh = UserVars()
        rgh.new_var('gamma',0)
        for r in r_list_names:
            rgh.new_var(r, 1.5)
        for i in range(len(delta_list_names)):
            delta = delta_list_names[i]
            if i in flat_down_index_new:
                rgh.new_var(delta, 0)
            else:
                rgh.new_var(delta, 20)

        #for gamma in gamma_list_names:
        #    rgh.new_var(gamma, 10)
        
        instance = cls(domain, ids, anchor_id, flat_down_index = flat_down_index_new, lat_pars = lat_pars)
        instance.rgh = rgh
        instance.r_list_names = r_list_names
        instance.delta_list_names = delta_list_names
        instance.gamma_list_names = gamma_list_names
        instance.gamma_handedness = [1]*anchor_index_list.index(None)+[0]*(len(anchor_index_list)-anchor_index_list.index(None)-1)
        instance.new_anchor_list = [ids_all[i] for i in anchor_index_list if i!=None]
        return instance

    def set_coordinate_all_rgh(self):
        r_list = [getattr(self.rgh, each) for each in self.r_list_names]
        delta_list = [getattr(self.rgh, each) for each in self.delta_list_names]
        gamma_list = [self.rgh.gamma+180*each for each in self.gamma_handedness]
        self.set_coordinate_all(r_list, delta_list, gamma_list, self.new_anchor_list)


    def set_coordinate_all(self, r_list = None, delta_list = None, gamma_list = None, new_anchor_list = None):
        if r_list!=None:
            assert len(r_list) == len(self.r_list), 'Dimensions of r_list must match!'
            self.r_list = r_list
        if delta_list!=None:
            assert len(delta_list) == len(self.delta_list), 'Dimensions of delta_list must match!'
            self.delta_list = delta_list
        if gamma_list!=None:
            assert len(gamma_list) == len(self.gamma_list), 'Dimensions of gamma_list must match!'
            self.gamma_list = gamma_list
        if new_anchor_list == None:
            new_anchor_list = [None]*len(self.gamma_list)
        else:
            assert len(new_anchor_list) == len(self.gamma_list), 'Dimensions of new_anchor_list must match!'

        self.bond_index = []
        for i in range(len(new_anchor_list)):
            each = new_anchor_list[i]
            current_id = self.ids[i]
            domain_id = list(self.domain.id)
            if each==None:
                self.bond_index.append((domain_id.index(current_id),domain_id.index(self.anchor_id)))
            else:
                self.bond_index.append((domain_id.index(current_id),domain_id.index(each)))
        self.domain.bond_index = self.bond_index

        #make sure the anchor atom to be calculated first
        #new_anchor_id_unique = []
        #for each in new_anchor_list:
        #    if each!=None and each not in new_anchor_id_unique:
        #        new_anchor_id_unique.append(each)
        
        for id in self.ids:
            index = list(self.ids).index(id)
            #print(id)
            #print(self.cal_coordinate(id, self.r_list[index], self.delta_list[index], self.gamma_list[index]))
            self.set_coordinate(id, self.cal_coordinate(id, self.r_list[index], self.delta_list[index], self.gamma_list[index], new_anchor_list[index]))

    def cal_coordinate(self, id, r, delta, gamma, new_anchor_id = None):
        if id in self.ids_flat_down:
            delta = 0
        z_temp_cart = r*np.sin(np.deg2rad(delta))
        y_temp_cart = r*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(gamma))
        x_temp_cart = r*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(gamma))
        if new_anchor_id == None:
            return self.extract_coord(self.anchor_id) + [x_temp_cart, y_temp_cart, z_temp_cart]/self.lat_abc
        else:
            return self.extract_coord(new_anchor_id) + [x_temp_cart, y_temp_cart, z_temp_cart]/self.lat_abc


    def set_coordinate(self, id, coords):
        index = np.where(self.domain.id == id)[0][0]
        self.domain.x[index], self.domain.y[index], self.domain.z[index] = coords
        items = ['dx1','dx2','dx3','dx4','dy1','dy2','dy3','dy4','dz1','dz2','dz3','dz4']
        for each in items:
           getattr(self.domain,each)[index] = 0

    def extract_coord(self,id):
        #print(np.where(self.domain.id == id))
        #print(self.domain.id)
        #print(id)
        index = np.where(self.domain.id == id)[0][0]
        x = self.domain.x[index] + self.domain.dx1[index] + self.domain.dx2[index] + self.domain.dx3[index] + self.domain.dx4[index]
        y = self.domain.y[index] + self.domain.dy1[index] + self.domain.dy2[index] + self.domain.dy3[index] + self.domain.dy4[index]
        z = self.domain.z[index] + self.domain.dz1[index] + self.domain.dz2[index] + self.domain.dz3[index] + self.domain.dz4[index]
        # x = self.domain.x[index]
        # y = self.domain.y[index]
        # z = self.domain.z[index]
        return np.array([x, y, z])

    def make_atom_group(self):
        atm_gp = model_2.AtomGroup()
        for id in self.domain.id:
            atm_gp.add_atom(self.domain, id)
        return atm_gp

    def make_cif_file(self, save_file):
        with open(save_file,'w') as f:
            f=open(save_file,'w')
            f.write('data_global\n')
            f.write("_chemical_name_mineral 'Copper'\n")
            f.write("_chemical_formula_sum 'Cu'\n")
            f.write("_cell_length_a {}\n".format(self.lat_abc[0]))
            f.write("_cell_length_b {}\n".format(self.lat_abc[1]))
            f.write("_cell_length_c "+str(self.lat_abc[2])+"\n")
            f.write("_cell_angle_alpha {}\n".format(self.lat_angles[0]))
            f.write("_cell_angle_beta {}\n".format(self.lat_angles[1]))
            f.write("_cell_angle_gamma {}\n".format(self.lat_angles[2]))
            f.write("_cell_volume {}\n".format(self.lat_abc[0]*self.lat_abc[1]*self.lat_abc[2]))
            f.write("_symmetry_space_group_name_H-M 'P 1'\nloop_\n_space_group_symop_operation_xyz\n  'x,y,z'\nloop_\n")
            f.write("_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")

            for i in range(len(self.domain.id)):
                el = self.domain.el[i]
                x, y, z = self.extract_coord(self.domain.id[i])
                s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (el, x, y, z)
                f.write(s)

    def make_xyz_file(self, save_file='xyz.xyz'):
        with open(save_file,'w') as f:
            f=open(save_file,'w')
            f.write('{}\n#\n'.format(len(self.domain.id)))
            for i in range(len(self.domain.id)):
                el = self.domain.el[i]
                x, y, z = self.extract_coord(self.domain.id[i])*self.lat_abc
                s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (el, x, y, z)
                f.write(s)

class SorbateTool(object):
    def __init__(self):
        pass

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