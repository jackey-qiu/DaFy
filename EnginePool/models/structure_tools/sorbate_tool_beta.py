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
import math
from functools import partial
from geometry_modules import *

x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])

#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])

#f2 calculate the distance b/ p1 and p2
f2=lambda p1_,p2_:np.sqrt(np.sum((p1_-p2_)**2))

#anonymous function f3 is to calculate the coordinates of basis with magnitude of 1.,p1 and p2 are coordinates for two known points, the 
#direction of the basis is pointing from p1 to p2
f3=lambda p1,p2:(1./f2(p1,p2))*(p2-p1)+p1

#rotation matrix for rotation successfully about x axis for alpha, y axis for beta and z axis for gamma
f4=lambda alpha,beta,gamma:np.array([[np.cos(beta)*np.cos(gamma),np.cos(gamma)*np.sin(alpha)*np.sin(beta)-np.cos(alpha)*np.sin(gamma),np.cos(alpha)*np.cos(gamma)*np.sin(beta)+np.sin(alpha)*np.sin(gamma)],\
                                     [np.cos(beta)*np.sin(gamma),np.cos(alpha)*np.cos(gamma)+np.sin(beta)*np.sin(alpha)*np.sin(gamma),-np.cos(gamma)*np.sin(alpha)+np.sin(beta)*np.cos(alpha)*np.sin(gamma)],\
                                     [-np.sin(beta),np.cos(beta)*np.sin(alpha),np.cos(alpha)*np.cos(beta)]])

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    example 1
    rotate coordianate v [1,2,3] about z axis [0, 0, 1] by 10 degrees with m [1,2,2] as the rotation center
    Note the rotation center is not the origin
    R = rotation_matrix([0,0,1], 10)
    v [1, 2, 3] will become np.dot(R, v-m) + m after rotation

    """
    theta = np.deg2rad(theta)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# from coordinate_system import CoordinateSystem

ALL_MOTIF_COLLECTION = ['OCCO','CCO', 'CO', 'CO3','CO2']
STRUCTURE_MOTIFS = {'CarbonOxygenMotif':['OCCO','CCO', 'CO', 'CO3','CO2'],
                    'TrigonalPyramid': ['BD_CASE_TP','TD_CASE_TP','OS_CASE_TP'],
                    'Tetrahedra':['BD_CASE_TH','TD_CASE_TH','OS_CASE_TH'],
                    'Octahedra':['BD_CASE_OH','TD_CASE_OH','OS_CASE_OH'],
                    'Gaussian':['GS_FLAT']}
from .structure_motif_collection import *

##OCCO##
structure_OCCO="""
#       O1    O2
#        \   /
#        C1-C2
#===================
"""
OCCO = {
        'substrate_domain':'surface_1', 
        "els":[str(['O','C','C', 'O'])],
        "anchor_index_list":[str([1, None, 1, 2 ])],
        "flat_down_index": [str([2])],
        "xyzu_oc_m": [str([0.5, 0.5, 1.5, 0.1, 1, 1])],
        "lat_pars": [str([3.615, 3.615, 3.615, 90, 90, 90])],
        'structure_pars_dict': [str({'r':1.5, 'delta':0})],
        "binding_mode": ['OS']
        }

##CO3##
structure_CO3="""
#       O1    O2
#        \   /
#          C1
#           |   
#           O3
#===================
"""
CO3 = {
        'substrate_domain':'surface_1', 
        "els":str(['O','C','O', 'O']),
        "anchor_index_list":str([1, None, 1, 1 ]),
        "flat_down_index": str([]),
        "xyzu_oc_m": str([0.5, 0.5, 1.5, 0.1, 1, 1]),
        "lat_pars": str([3.615, 3.615, 3.615, 90, 90, 90]),
        'structure_pars_dict': str({'r':1.5, 'delta':0}),
        "binding_mode": 'OS'
        }

##CCO#
structure_CCO="""
#       C2
#      /  \\
#     C1   O1
#====================
"""
CCO = {
        'substrate_domain':'surface_1', 
        "els":str(['C','C', 'O']),
        "anchor_index_list":str([None, 0, 1 ]),
        "flat_down_index": str([2]),
        "xyzu_oc_m": str([0.5, 0.5, 1.5, 0.1, 1, 1]),
        "lat_pars": str([3.615, 3.615, 3.615, 90, 90, 90]),
        'structure_pars_dict': str({'r':1.5, 'delta':0}),
        "binding_mode": 'OS'
        }

##CO2#
structure_CO2="""
#       O1=C1=O2
#====================
"""
CO2 = {
        'substrate_domain':'surface_1', 
        "els":str(['O','C', 'O']),
        "anchor_index_list":str([1, None, 1 ]),
        "flat_down_index": str([0,2]),
        "xyzu_oc_m": str([0.5, 0.5, 1.5, 0.1, 1, 1]),
        "lat_pars": str([3.615, 3.615, 3.615, 90, 90, 90]),
        'structure_pars_dict': str({'r':1.5, 'delta':0}),
        "binding_mode": 'OS'
        }

##CO##
structure_CO="""
#        O1   
#        |   
#        C1
#====================
"""
CO = {
        'substrate_domain':'surface_1', 
        "els":str(['C','O']),
        "anchor_index_list":str([None, 0]),
        "flat_down_index": str([]),
        "xyzu_oc_m": str([0.5, 0.5, 1.5, 0.1, 1, 1]),
        "lat_pars": str([3.615, 3.615, 3.615, 90, 90, 90]),
        'structure_pars_dict': str({'r':1.5, 'delta':0}),
        "binding_mode": 'OS'
        }

SURFACE_SYMS = {
                '1001_0':[[1,0],[0,1],[0,0]],
                '1001_0.5':[[1,0],[0,1],[0.5,0.5]],
                '100-1_0':[[1,0],[0,-1],[0,0]],
                '100-1_0.5':[[1,0],[0,-1],[0.5,0.5]],
                '-1001_0':[[-1,0],[0,1],[0,0]],
                '-1001_0.5':[[-1,0],[0,1],[0.5,0.5]],
                '-100-1_0':[[-1,0],[0,-1],[0,0]],
                '-100-1_0.5':[[-1,0],[0,-1],[0,0]],
                '0110_0':[[0,1],[1,0],[0,0]],
                '0110_0.5':[[0,1],[1,0],[0.5,0.5]],
                '01-10_0':[[0,1],[-1,0],[0,0]],
                '01-10_0.5':[[0,1],[-1,0],[0.5,0.5]],
                '0-1-10_0':[[0,-1],[-1,0],[0,0]],
                '0-1-10_0.5':[[0,-1],[-1,0],[0.5,0.5]],
                '0-110_0':[[0,-1],[1,0],[0,0]],
                '0-110_0.5':[[0,-1],[1,0],[0.5,0.5]]
                }

class StructureMotif(object):
    #structure_index = 0
    def __init__(self, domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars , **kwargs):
        """[baseobject to be inherited by specific structural motif class]
        Args:
            domain (instance of model_2.Slab): [description]
            ids (type = list): list of ids of each constituent atom
            anchor_id (type = str): id of reference atom in the sorbate motif
            substrate_domain (type = instance of model_2.Slab): slab instance of substrate
            anchored_ids (type = list): list of atom ids in substrate_domain to be referenced to by motif
            binding_mode (type = str): binding mode: either 'MD' for monodentate or 'BD' for bidentate or 'TD' for tridentate or 'OS' for outersphere
            structure_pars_dict (type = dictionary): structural parameters for different structural motif
            lat_pars (type = list): lattice parameters in form of [a, b, c, alpha, beta, gamma]
        """
        self.domain = domain
        self.ids = ids
        self.els = els
        self.anchor_id = anchor_id
        self.substrate_domain = substrate_domain
        self.substrate_domain.bound_domains.append(self.domain)
        self.anchored_ids = anchored_ids
        self.binding_mode = binding_mode
        self.structure_pars_dict = structure_pars_dict
        self.lat_pars = np.array(lat_pars)
        self.kwargs = kwargs
        # self.structure_index+=1

    @classmethod
    def get_par_dict(cls, motif_type):
        return globals()[motif_type]

    def generate_script_snippet(self):
        pass

    def create_rgh(self):
        self.rgh = UserVars()
        for key, value in self.structure_pars_dict.items():
            self.rgh.new_var(key, value)

    def get_rgh(self):
        if not hasattr(self,'rgh'):
            print('No rgh attribute was created in the instance!')
        else:
            return self.rgh

    def _translate_offset_symbols(self, symbol):
        if symbol=='-x':return np.array([-1.,0.,0.])
        elif symbol=='+x':return np.array([1.,0.,0.])
        elif symbol=='-y':return np.array([0.,-1.,0.])
        elif symbol=='+y':return np.array([0.,1.,0.])
        elif symbol==None:return np.array([0.,0.,0.])

    def _add_sorbate(self, domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==id_sorbate)[0][0]
        except:
            domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]

    def cal_coor_o3(self,p0,p1,p3):
        #function to calculate the new point for p3, see document file #2 for detail procedures
        f2=lambda p1_,p2_:np.sqrt(np.sum((p1_-p2_)**2))
        r=f2(p0,p1)/2.*np.tan(np.pi/3)
        norm_vt=p0-p1
        cent_pt=(p0+p1)/2
        a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
        d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
        u,v,w=p3[0],p3[1],p3[2]
        k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
        #projection of O3 to the normal plane see http://www.9math.com/book/projection-point-plane for detail algorithm
        O3_proj=np.array([u-a*k,v-b*k,w-c*k])
        cent_proj_vt=O3_proj-cent_pt
        l=f2(O3_proj,cent_pt)
        ptOnCircle_cent_vt=cent_proj_vt/l*r
        ptOnCircle=ptOnCircle_cent_vt+cent_pt
        # print(p0,p1,ptOnCircle)
        # print(f2(p0,p1),f2(p0,ptOnCircle),f2(p1,ptOnCircle))
        return ptOnCircle

    def build_structure(self):
        #customized function to first calculate atom positions and add atoms to self.domain
        if self.binding_mode == 'MD':
            self._build_structure_MD()
        elif self.binding_mode == 'BD':
            self._build_structure_BD()
        elif self.binding_mode == 'TD':
            self._build_structure_TD()
        elif self.binding_mode == 'OS':
            self._build_structure_OS()
        else:
            print('The mode {} is not implemented yet!'.formtat(self.binding_mode))
            return

    def _build_structure_MD(self):
        #to be implimented for each specific motif
        pass

    def _build_structure_BD(self):
        #to be implimented for each specific motif
        pass
    
    def _build_structure_TD(self):
        #to be implimented for each specific motif
        pass

    def _build_structure_OS(self):
        #to be implimented for each specific motif
        pass

    def update_structure(self):
        #customized function to first calculate atom positions and add atoms to self.domain
        if self.binding_mode == 'MD':
            self._update_structure_MD()
        elif self.binding_mode == 'BD':
            self._update_structure_BD()
        elif self.binding_mode == 'TD':
            self._update_structure_TD()
        elif self.binding_mode == 'OS':
            self._update_structure_OS()
        else:
            print('The mode {} is not implemented yet!'.formtat(self.binding_mode))
            return

    def _update_structure_MD(self):
        #to be implimented for each specific motif
        pass

    def _update_structure_BD(self):
        #to be implimented for each specific motif
        pass
    
    def _update_structure_TD(self):
        #to be implimented for each specific motif
        pass

    def _update_structure_OS(self):
        #to be implimented for each specific motif
        pass

    def make_atom_group(self):
        #to be implimented for each specific motif
        #should return a list of atomgroups
        self.atom_groups = [model_2.AtomGroup()]

class TrigonalPyramid(StructureMotif):
    def __init__(self,domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs):
        super(TrigonalPyramid, self).__init__(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs)

    @classmethod
    def generate_script_from_setting_table(cls, use_predefined_motif = False, predefined_motif = '', structure_index = 1, kwargs = {}):
        if use_predefined_motif:
            temp = globals()[predefined_motif]
            temp['substrate_domain'] = 'surface_{}'.format(structure_index)
        else:
            temp = kwargs
        ids = str(temp.get('ids'))
        els = str(temp.get('els'))
        anchor_id = str(temp.get('anchor_id'))
        substrate_domain = str(temp.get('substrate_domain'))
        anchored_ids = str({'attach_atm_ids':eval(temp.get('attach_atm_ids')),'offset':eval(temp.get('offset')),'anchor_ref':temp.get('anchor_ref'),'anchor_offset':temp.get('anchor_offset')})
        lat_pars = str(temp.get('lat_pars'))
        if temp['mode'] == 'BD':
            binding_mode = str({'mode':temp['mode'],'mirror':bool(temp['mirror']),'switch':bool(temp['switch'])})
            structure_pars_dict = str({'top_angle':eval(temp['top_angle']),'phi':eval(temp['phi']),'edge_offset':eval(temp['edge_offset']), 'angle_offset':eval(temp['angle_offset'])})
        elif temp['mode'] == 'TD':
            binding_mode = str({'mode':temp['mode'],'mirror':bool(temp['mirror'])})
            structure_pars_dict = str({'top_angle':eval(temp['top_angle'])})
        elif temp['mode'] == 'OS':
            binding_mode = str({'mode':temp['mode']})
            structure_pars_dict = str({'r_sorbate_O':eval(temp['r_sorbate_O']),'ang_O_M_O':eval(temp['ang_O_M_O']),'rotation_x':eval(temp['rotation_x']),'rotation_y':eval(temp['rotation_y']),'rotation_z':eval(temp['rotation_z'])})
        #T = str(temp['T'])
        #T_INV = str(temp['T_INV'])
        return cls.generate_script_snippet(substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index)

    @classmethod
    def generate_script_snippet(cls, substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index):
        instance_name = 'sorbate_instance_{}'.format(structure_index)
        rgh_name = 'rgh_sorbate_{}'.format(structure_index)
        domain_name = 'domain_sorbate_{}'.format(structure_index)
        atm_gp_name = 'atm_gp_sorbate_{}'.format(structure_index)
        line1 = "{} = sorbate_tool.TrigonalPyramid.build_instance(substrate_domain = {},anchored_ids={},binding_mode = {},structure_pars_dict={},anchor_id = '{}',ids = {}, els = {}, lat_pars = {}, T = unitcell.lattice.RealTM, T_INV = unitcell.lattice.RealTMInv)".format(instance_name,substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars)
        line2 = f"{domain_name} = {instance_name}.domain"
        line3 = f"{rgh_name} = {instance_name}.rgh"
        line4 = f"{atm_gp_name} = {instance_name}.make_atm_group(instance_name = \'{atm_gp_name}\')"
        return('\n'.join([line1,line2,line3,line4]),f"    {instance_name}.update_structure()")

    @classmethod
    def build_instance(cls,substrate_domain = None,anchored_ids = {'attach_atm_ids':['id1','id2'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None}, binding_mode = {'mode':'BD','mirror':False, 'switch':False},
                       structure_pars_dict = {'top_angle':70.,'phi':0.,'edge_offset':[0,0], 'angle_offset':0},anchor_id='pb_id',ids = ['Pb_id','id1'], els = ['Pb','O'], lat_pars = np.array([5.038,5.434,7.3707,90,90,90]), T=None,T_INV=None):
        #initialize slab
        domain = model_2.Slab(T_factor = 'u')
        instance = cls(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars,T = T,T_INV = T_INV)
        instance.set_class_attributes()
        instance.create_rgh()
        instance.build_structure()
        return instance

    def set_class_attributes(self):
        # self.geometry_object = geometry_object
        for each in self.kwargs:
            setattr(self, each, self.kwargs[each])
        self.basis = self.lat_pars[0:3]

    def create_rgh(self):
        rgh = UserVars()
        for each in self.structure_pars_dict:
            if each!='edge_offset':
                rgh.new_var(each,self.structure_pars_dict[each])
            else:
                rgh.new_var('edge_offset_1',self.structure_pars_dict[each][0])
                rgh.new_var('edge_offset_2',self.structure_pars_dict[each][1])
        self.rgh = rgh
        return self.rgh

    def get_dict_from_rgh(self):
        return {'top_angle':self.rgh.top_angle,'phi':self.rgh.phi,'edge_offset':[self.rgh.edge_offset_1,self.rgh.edge_offset_2], 'angle_offset':self.rgh.angle_offset}

    def build_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode['mode'] == 'BD':
            self._build_structure_BD()
        elif self.binding_mode['mode'] == 'TD':
            self._build_structure_TD()
        elif self.binding_mode['mode'] == 'OS':
            self._build_structure_OS()
        else:
            print('Current binding mode is Not implemented yet!')

    def update_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode['mode'] == 'BD':
            self._update_structure_BD()
        elif self.binding_mode['mode'] == 'TD':
            self._update_structure_TD()
        elif self.binding_mode['mode'] == 'OS':
            self._update_structure_OS()
        else:
            print('Current binding mode is Not implemented yet!')

    def _add_sorbate(self, domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==id_sorbate)[0][0]
        except:
            domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]

    def _update_structure_OS(self):
        self._build_structure_OS()

    def _build_structure_OS(self):
        def _cal_geometry(cent_point,phi,r_Pb_O, O_Pb_O_ang, rotation_x,rotation_y,rotation_z):
            #a,b,c=self.lat_pars[0:3]
            r0=r_Pb_O*np.sin(O_Pb_O_ang/2*np.pi/180)/np.cos(np.pi/6)
            r1=r_Pb_O*(np.square(np.cos(O_Pb_O_ang/2*np.pi/180))-np.square(np.sin(O_Pb_O_ang/2*np.pi/180))*np.square(np.tan(np.pi/6)))**0.5
            phi=phi/180*np.pi
            p1_x,p1_y,p1_z=r0*np.cos(phi)*np.sin(np.pi/2.)+cent_point[0],r0*np.sin(phi)*np.sin(np.pi/2.)+cent_point[1],r0*np.cos(np.pi/2.)+cent_point[2]
            p2_x,p2_y,p2_z=r0*np.cos(phi+2*np.pi/3)*np.sin(np.pi/2.)+cent_point[0],r0*np.sin(phi+2*np.pi/3)*np.sin(np.pi/2.)+cent_point[1],r0*np.cos(np.pi/2.)+cent_point[2]
            p3_x,p3_y,p3_z=r0*np.cos(phi+4*np.pi/3)*np.sin(np.pi/2.)+cent_point[0],r0*np.sin(phi+4*np.pi/3)*np.sin(np.pi/2.)+cent_point[1],r0*np.cos(np.pi/2.)+cent_point[2]
            apex_x,apex_y,apex_z=cent_point[0],cent_point[1],cent_point[2]+r1
            T_rot=f4(rotation_x,rotation_y,rotation_z)
            origin=np.array([apex_x,apex_y,apex_z])
            p1=np.array([p1_x,p1_y,p1_z])
            p2=np.array([p2_x,p2_y,p2_z])
            p3=np.array([p3_x,p3_y,p3_z])
            #Transform to fractional coordinates from Cartisian ones
            p1_new=np.dot(self.T_INV,(np.dot(T_rot,p1-origin)+origin))
            p2_new=np.dot(self.T_INV,(np.dot(T_rot,p2-origin)+origin))
            p3_new=np.dot(self.T_INV,(np.dot(T_rot,p3-origin)+origin))
            return np.dot(self.T_INV, np.array([apex_x,apex_y,apex_z])),p1_new, p2_new, p3_new
        center_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        center=np.dot(self.T,pt_ct(self.substrate_domain,center_index,self.anchored_ids['offset'][0]))
        apex, p1, p2, p3 = _cal_geometry(cent_point = center, phi = 0, r_Pb_O = self.rgh.r_sorbate_O, O_Pb_O_ang = self.rgh.ang_O_M_O, rotation_x = self.rgh.rotation_x,rotation_y = self.rgh.rotation_y,rotation_z = self.rgh.rotation_z)
        O_coords = [p1,p2,p3]
        O_id = [each for each in self.ids if each!=self.anchor_id]
        assert len(O_coords)==len(O_id), 'len of O_coords and len of O_id does not match!'   
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=apex)
        for i in range(len(O_id)):
            self._add_sorbate(domain=self.domain,id_sorbate=O_id[i],el=self.els[self.ids.index(O_id[i])],sorbate_v=O_coords[i])

    def _update_structure_TD(self):
        self._build_structure_TD(update = True)

    def _build_structure_TD(self, update = False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
        p_O3_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][2])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)

        p_O1 = np.dot(self.T, pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
        p_O2 = np.dot(self.T, pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
        p_O3 = np.dot(self.T, pt_ct(self.substrate_domain,p_O3_index,self.anchored_ids['offset'][2]))
        p_O3 = trigonal_pyramid_distortion_shareface.trigonal_pyramid_distortion_shareface.cal_coor_o3(p_O1, p_O2, p_O3)
        if not update:
            pyramid_distortion=trigonal_pyramid_distortion_shareface.trigonal_pyramid_distortion_shareface(p0=p_O1,p1=p_O2,p2=p_O3, top_angle=self.rgh.top_angle/180.*np.pi)
            pyramid_distortion.cal_apex_coor(mirror=self.binding_mode['mirror'])
            self.geometry_object = pyramid_distortion
        else:
            self.geometry_object.reset_attributes(p_O1,p_O2,p_O3,self.rgh.top_angle/180.*np.pi)
            self.geometry_object.cal_apex_coor(mirror=self.binding_mode['mirror'])
           
        #self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=self.geometry_object.apex/self.basis)
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=np.dot(self.T_INV,self.geometry_object.apex))

    def _update_structure_BD(self):
        self._build_structure_BD(update = True)

    def _build_structure_BD(self, update = False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
        anchor_index=None
        if self.anchored_ids['anchor_ref']!=None:
            anchor_index=list(self.substrate_domain.id).index(self.anchored_ids['anchor_ref'])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)

        p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
        p_O2=np.dot(self.T,pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
        anchor=None
        if anchor_index!=None:
            anchor=np.dot(self.T,pt_ct(self.substrate_domain,anchor_index,self.anchored_ids['anchor_offset']))
        #print "O1",p_O1
        #print "O2",p_O2
        # print(pt_ct(self.substrate_domain,anchor_index,self.anchored_ids['anchor_offset']))
        # print(anchor_index,self.substrate_domain.x[anchor_index],self.substrate_domain.y[anchor_index],self.substrate_domain.z[anchor_index])
        if not update:
            pyramid_distortion=trigonal_pyramid_distortion_B2.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,ref=np.array(anchor),top_angle=self.rgh.top_angle/180.*np.pi,len_offset=[self.rgh.edge_offset_1, self.rgh.edge_offset_2])
            pyramid_distortion.all_in_all(switch=self.binding_mode['switch'],phi=self.rgh.phi/180*np.pi,mirror=self.binding_mode['mirror'],angle_offset=self.rgh.angle_offset/180.*np.pi)
            self.geometry_object = pyramid_distortion
        else:
            attrs_reset = {'p0':p_O1, 'p1': p_O2, 'ref':anchor, 'top_angle':self.rgh.top_angle/180.*np.pi, 'len_offset':[self.rgh.edge_offset_1, self.rgh.edge_offset_2]}
            self.geometry_object.all_in_all_reset(switch=self.binding_mode['switch'],phi=self.rgh.phi/180*np.pi,mirror=self.binding_mode['mirror'],angle_offset=self.rgh.angle_offset/180.*np.pi, attrs = attrs_reset)
        
        O_id = [each for each in self.ids if each!=self.anchor_id]
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=np.dot(self.T_INV,self.geometry_object.apex))
        if O_id!=[]:
            self._add_sorbate(domain=self.domain,id_sorbate=O_id[0],el='O',sorbate_v=np.dot(self.T_INV,self.geometry_object.p2))
        #return [np.dot(T_INV,pyramid_distortion.apex)/basis,np.dot(T_INV,pyramid_distortion.p2)/basis

    def make_atm_group(self, instance_name = 'instance_name'):
        atm_gp = model_2.AtomGroup(instance_name = instance_name)
        for id in self.domain.id:
            atm_gp.add_atom(self.domain, id)
        return atm_gp

class Tetrahedra(StructureMotif):
    def __init__(self,domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs):
        super(Tetrahedra, self).__init__(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs)

    @classmethod
    def generate_script_from_setting_table(cls, use_predefined_motif = False, predefined_motif = '', structure_index = 1, kwargs = {}):
        if use_predefined_motif:
            temp = globals()[predefined_motif]
            temp['substrate_domain'] = 'surface_{}'.format(structure_index)
        else:
            temp = kwargs
        ids = str(temp.get('ids'))
        els = str(temp.get('els'))
        anchor_id = str(temp.get('anchor_id'))
        substrate_domain = str(temp.get('substrate_domain'))
        anchored_ids = str({'attach_atm_ids':eval(temp.get('attach_atm_ids')),'offset':eval(temp.get('offset')),'anchor_ref':temp.get('anchor_ref'),'anchor_offset':temp.get('anchor_offset')})
        lat_pars = str(temp.get('lat_pars'))
        if temp['mode'] == 'BD':
            binding_mode = str({'mode':temp['mode']})
            structure_pars_dict = str({'top_angle_offset':eval(temp['top_angle_offset']),'angle_offset':eval(temp['angle_offset']),'phi':eval(temp['phi']),'edge_offset':eval(temp['edge_offset'])})
        elif temp['mode'] == 'TD':
            binding_mode = str({'mode':temp['mode'],'mirror':bool(temp['mirror'])})
            structure_pars_dict = str({'center_offset':eval(temp['center_offset']),'edge_offset':eval(temp['edge_offset'])})
        elif temp['mode'] == 'OS':
            binding_mode = str({'mode':temp['mode']})
            structure_pars_dict = str({'r_sorbate_O':eval(temp['r_sorbate_O']),'rotation_x':eval(temp['rotation_x']),'rotation_y':eval(temp['rotation_y']),'rotation_z':eval(temp['rotation_z'])})
        #T = str(temp['T'])
        #T_INV = str(temp['T_INV'])
        return cls.generate_script_snippet(substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index)

    @classmethod
    def generate_script_snippet(cls, substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index):
        instance_name = 'sorbate_instance_{}'.format(structure_index)
        rgh_name = 'rgh_sorbate_{}'.format(structure_index)
        domain_name = 'domain_sorbate_{}'.format(structure_index)
        atm_gp_name = 'atm_gp_sorbate_{}'.format(structure_index)
        line1 = "{} = sorbate_tool.Tetrahedra.build_instance(substrate_domain = {},anchored_ids={},binding_mode = {},structure_pars_dict={},anchor_id = '{}',ids = {}, els = {}, lat_pars = {}, T = unitcell.lattice.RealTM, T_INV = unitcell.lattice.RealTMInv)".format(instance_name,substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars)
        line2 = f"{domain_name} = {instance_name}.domain"
        line3 = f"{rgh_name} = {instance_name}.rgh"
        line4 = f"{atm_gp_name} = {instance_name}.make_atm_group(instance_name = \'{atm_gp_name}\')"
        return('\n'.join([line1,line2,line3,line4]),f"    {instance_name}.update_structure()")

    @classmethod
    def build_instance(cls,substrate_domain = None,anchored_ids = {'attach_atm_ids':['id1','id2'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None}, binding_mode = {'mode':'BD'},
                       structure_pars_dict = {'top_angle_offset':0, 'angle_offset':[0.,0],'phi':0.,'edge_offset':[0,0]},anchor_id='pb_id',ids = ['Pb_id','id1','id2'], els = ['Pb','O','O'], lat_pars = np.array([5.038,5.434,7.3707,90,90,90]), T=None,T_INV=None):
        #initialize slab
        domain = model_2.Slab(T_factor = 'u')
        instance = cls(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars,T = T,T_INV = T_INV)
        instance.set_class_attributes()
        instance.create_rgh()
        instance.build_structure()
        return instance

    def set_class_attributes(self):
        # self.geometry_object = geometry_object
        for each in self.kwargs:
            setattr(self, each, self.kwargs[each])
        self.basis = self.lat_pars[0:3]

    def create_rgh(self):
        rgh = UserVars()
        for each,item in self.structure_pars_dict.items():
            if type(item)!=type([]):
                rgh.new_var(each,self.structure_pars_dict[each])
            else:
                for i in range(len(item)):
                    rgh.new_var(each+'_'+str(i+1),item[i])
        self.rgh = rgh
        return self.rgh

    def get_dict_from_rgh(self):
        return {'top_angle':self.rgh.top_angle,'phi':self.rgh.phi,'edge_offset':[self.rgh.edge_offset_1,self.rgh.edge_offset_2], 'angle_offset':self.rgh.angle_offset}

    def build_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode['mode'] == 'BD':
            self._build_structure_BD()
        elif self.binding_mode['mode'] == 'TD':
            self._build_structure_TD()
        elif self.binding_mode['mode'] == 'OS':
            self._build_structure_OS()
        else:
            print('Current binding mode is Not implemented yet!')

    def update_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode['mode'] == 'BD':
            self._update_structure_BD()
        elif self.binding_mode['mode'] == 'TD':
            self._update_structure_TD()
        elif self.binding_mode['mode'] == 'OS':
            self._update_structure_OS()
        else:
            print('Current binding mode is Not implemented yet!')

    def _update_structure_OS(self):
        self._build_structure_OS()

    def _build_structure_OS(self):
        def _cal_geometry(cent_point,phi,r_sorbate_O,rotation_x,rotation_y,rotation_z):
            O_Pb_O_ang=109.5
            r0=r_sorbate_O*np.sin(O_Pb_O_ang/2*np.pi/180)/np.cos(np.pi/6)
            r1=r_sorbate_O*(np.square(np.cos(O_Pb_O_ang/2*np.pi/180))-np.square(np.sin(O_Pb_O_ang/2*np.pi/180))*np.square(np.tan(np.pi/6)))**0.5
            phi=phi/180*np.pi
            cent_point=cent_point-np.array([0,0,r1])
            p1_x,p1_y,p1_z=r0*np.cos(phi)*np.sin(np.pi/2.)+cent_point[0],r0*np.sin(phi)*np.sin(np.pi/2.)+cent_point[1],r0*np.cos(np.pi/2.)+cent_point[2]
            p2_x,p2_y,p2_z=r0*np.cos(phi+2*np.pi/3)*np.sin(np.pi/2.)+cent_point[0],r0*np.sin(phi+2*np.pi/3)*np.sin(np.pi/2.)+cent_point[1],r0*np.cos(np.pi/2.)+cent_point[2]
            p3_x,p3_y,p3_z=r0*np.cos(phi+4*np.pi/3)*np.sin(np.pi/2.)+cent_point[0],r0*np.sin(phi+4*np.pi/3)*np.sin(np.pi/2.)+cent_point[1],r0*np.cos(np.pi/2.)+cent_point[2]
            apex_x,apex_y,apex_z=cent_point[0],cent_point[1],cent_point[2]+r1
            p4_x,p4_y,p4_z=apex_x,apex_y,apex_z+r_sorbate_O
            rot_x,rot_y,rot_z=rotation_x/180*np.pi,rotation_y/180*np.pi,rotation_z/180*np.pi
            T_rot=f4(rot_x,rot_y,rot_z)
            origin=np.array([apex_x,apex_y,apex_z])
            p1=np.array([p1_x,p1_y,p1_z])
            p2=np.array([p2_x,p2_y,p2_z])
            p3=np.array([p3_x,p3_y,p3_z])
            p4=np.array([p4_x,p4_y,p4_z])
            p1_new = np.dot(self.T_INV, (np.dot(T_rot,p1-origin)+origin))
            p2_new = np.dot(self.T_INV, (np.dot(T_rot,p2-origin)+origin))
            p3_new = np.dot(self.T_INV, (np.dot(T_rot,p3-origin)+origin))
            p4_new = np.dot(self.T_INV, (np.dot(T_rot,p4-origin)+origin))
            return np.dot(self.T_INV,np.array([apex_x,apex_y,apex_z])),p1_new, p2_new, p3_new, p4_new

        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        center_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        center=np.dot(self.T,pt_ct(self.substrate_domain,center_index,self.anchored_ids['offset'][0]))
        apex, p1, p2, p3, p4 = _cal_geometry(cent_point = center, phi = 0, r_sorbate_O = self.rgh.r_sorbate_O,rotation_x = self.rgh.rotation_x,rotation_y = self.rgh.rotation_y,rotation_z = self.rgh.rotation_z)
        O_coords = [p1,p2,p3,p4]
        O_id = [each for each in self.ids if each!=self.anchor_id]
        assert len(O_coords)==len(O_id), 'len of O_coords and len of O_id does not match!'
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=apex)
        for i in range(len(O_id)):
            self._add_sorbate(domain=self.domain,id_sorbate=O_id[i],el=self.els[self.ids.index(O_id[i])],sorbate_v=O_coords[i])

    def _update_structure_TD(self):
        self._build_structure_TD(update = True)

    def _build_structure_TD(self, update = False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
        p_O3_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][2])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
        p_O2=np.dot(self.T,pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
        p_O3=np.dot(self.T,pt_ct(self.substrate_domain,p_O3_index,self.anchored_ids['offset'][2]))
        p_O3 = tetrahedra.share_face.cal_coor_o3(p_O1, p_O2, p_O3)
        if not update:
            if self.binding_mode['mirror']:
                tetrahedra_distortion=tetrahedra.share_face(np.array([p_O1,p_O2,p_O3]))
            else:
                tetrahedra_distortion=tetrahedra.share_face(np.array([p_O1,p_O3,p_O2]))
            tetrahedra_distortion.share_face_init()
            tetrahedra_distortion.apply_top_angle_offset(self.rgh.center_offset)
            tetrahedra_distortion.apply_edge_offset(self.rgh.edge_offset)
            self.geometry_object = tetrahedra_distortion
        else:
            if self.binding_mode['mirror']:
                self.geometry_object.face = np.array([p_O1,p_O2,p_O3])
            else:
                self.geometry_object.face = np.array([p_O1,p_O3,p_O2])
            self.geometry_object.share_face_init()
            self.geometry_object.apply_top_angle_offset(self.rgh.center_offset)
            self.geometry_object.apply_edge_offset(self.rgh.edge_offset)
        O_id = [each for each in self.ids if each!=self.anchor_id]
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=np.dot(self.T_INV,self.geometry_object.center_point))
        self._add_sorbate(domain=self.domain,id_sorbate=O_id[0],el=self.els[self.ids.index(O_id[0])],sorbate_v=np.dot(self.T_INV,self.geometry_object.p3))

    def _update_structure_BD(self):
        self._build_structure_BD(update = True)

    def _build_structure_BD(self, update = False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
        anchor_index=None
        if self.anchored_ids['anchor_ref']!=None:
            anchor_index=list(self.substrate_domain.id).index(self.anchored_ids['anchor_ref'])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
        p_O2=np.dot(self.T,pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
        anchor=None
        if anchor_index!=None:
            anchor=np.dot(self.T,pt_ct(self.substrate_domain,anchor_index,self.anchored_ids['anchor_offset']))
        #print "O1",p_O1
        #print "O2",p_O2
        # print(pt_ct(self.substrate_domain,anchor_index,self.anchored_ids['anchor_offset']))
        # print(anchor_index,self.substrate_domain.x[anchor_index],self.substrate_domain.y[anchor_index],self.substrate_domain.z[anchor_index])
        if not update:
            tetrahedra_distortion=tetrahedra.share_edge(edge=np.array([p_O1,p_O2]))
            tetrahedra_distortion.cal_p2(ref_p=anchor,phi=self.rgh.phi/180*np.pi)
            tetrahedra_distortion.share_face_init()
            tetrahedra_distortion.apply_top_angle_offset_BD(top_angle_offset = self.rgh.top_angle_offset)
            tetrahedra_distortion.apply_angle_offset_BD(distal_angle_offset = [self.rgh.angle_offset_1,self.rgh.angle_offset_2],distal_length_offset = [self.rgh.edge_offset_1,self.rgh.edge_offset_2])
            self.geometry_object = tetrahedra_distortion
        else:
            self.geometry_object.update_anchor_points(np.array([p_O1,p_O2]))
            self.geometry_object.cal_p2(ref_p=anchor,phi=self.rgh.phi/180*np.pi)
            self.geometry_object.share_face_init()
            self.geometry_object.apply_top_angle_offset_BD(top_angle_offset = self.rgh.top_angle_offset)
            self.geometry_object.apply_angle_offset_BD(distal_angle_offset = [self.rgh.angle_offset_1,self.rgh.angle_offset_2],distal_length_offset = [self.rgh.edge_offset_1,self.rgh.edge_offset_2])
        
        O_id = [each for each in self.ids if each!=self.anchor_id]
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=np.dot(self.T_INV,self.geometry_object.center_point))
        if O_id!=[]:
            self._add_sorbate(domain=self.domain,id_sorbate=O_id[0],el='O',sorbate_v=np.dot(self.T_INV,self.geometry_object.p2))
            self._add_sorbate(domain=self.domain,id_sorbate=O_id[1],el='O',sorbate_v=np.dot(self.T_INV,self.geometry_object.p3))
            #return [np.dot(T_INV,pyramid_distortion.apex)/basis,np.dot(T_INV,pyramid_distortion.p2)/basis

    def make_atm_group(self, instance_name = 'instance_name'):
        atm_gp = model_2.AtomGroup(instance_name = instance_name)
        for id in self.domain.id:
            atm_gp.add_atom(self.domain, id)
        return atm_gp

class Octahedra(StructureMotif):
    def __init__(self,domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs):
        super(Octahedra, self).__init__(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs)

    @classmethod
    def generate_script_from_setting_table(cls, use_predefined_motif = False, predefined_motif = '', structure_index = 1, kwargs = {}):
        if use_predefined_motif:
            temp = globals()[predefined_motif]
            temp['substrate_domain'] = 'surface_{}'.format(structure_index)
        else:
            temp = kwargs
        ids = str(temp.get('ids'))
        els = str(temp.get('els'))
        anchor_id = str(temp.get('anchor_id'))
        substrate_domain = str(temp.get('substrate_domain'))
        anchored_ids = str({'attach_atm_ids':eval(temp.get('attach_atm_ids')),'offset':eval(temp.get('offset')),'anchor_ref':temp.get('anchor_ref'),'anchor_offset':temp.get('anchor_offset')})
        lat_pars = str(temp.get('lat_pars'))
        if temp['mode'] == 'BD':
            binding_mode = str({'mode':temp['mode']})
            structure_pars_dict = str({'phi':eval(temp['phi']),'dr1':eval(temp['dr1']),'dr2':eval(temp['dr2']),'dr3':eval(temp['dr3'])})
        elif temp['mode'] == 'TD':
            binding_mode = str({'mode':temp['mode'],'mirror':bool(temp['mirror'])})
            structure_pars_dict = str({'dr1':eval(temp['dr1']),'dr2':eval(temp['dr2']),'dr3':eval(temp['dr3'])})
        elif temp['mode'] == 'OS':
            binding_mode = str({'mode':temp['mode']})
            structure_pars_dict = str({'r_sorbate_O':eval(temp['r_sorbate_O']),'rotation_x':eval(temp['rotation_x']),'rotation_y':eval(temp['rotation_y']),'rotation_z':eval(temp['rotation_z'])})
        #T = str(temp['T'])
        #T_INV = str(temp['T_INV'])
        return cls.generate_script_snippet(substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index)

    @classmethod
    def generate_script_snippet(cls, substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index):
        instance_name = 'sorbate_instance_{}'.format(structure_index)
        rgh_name = 'rgh_sorbate_{}'.format(structure_index)
        domain_name = 'domain_sorbate_{}'.format(structure_index)
        atm_gp_name = 'atm_gp_sorbate_{}'.format(structure_index)
        line1 = "{} = sorbate_tool.Octahedra.build_instance(substrate_domain = {},anchored_ids={},binding_mode = {},structure_pars_dict={},anchor_id = '{}',ids = {}, els = {}, lat_pars = {}, T = unitcell.lattice.RealTM, T_INV = unitcell.lattice.RealTMInv)".format(instance_name,substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars)
        line2 = f"{domain_name} = {instance_name}.domain"
        line3 = f"{rgh_name} = {instance_name}.rgh"
        line4 = f"{atm_gp_name} = {instance_name}.make_atm_group(instance_name = \'{atm_gp_name}\')"
        return('\n'.join([line1,line2,line3,line4]),f"    {instance_name}.update_structure()")

    @classmethod
    def build_instance(cls,substrate_domain = None,anchored_ids = {'attach_atm_ids':['id1','id2'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None}, binding_mode = {'mode':'BD'},
                       structure_pars_dict = {'phi':0.,'edge_offset':[0,0]},anchor_id='pb_id',ids = ['Pb_id','id1','id2'], els = ['Pb','O','O'], lat_pars = np.array([5.038,5.434,7.3707,90,90,90]), T=None,T_INV=None):
        #initialize slab
        domain = model_2.Slab(T_factor = 'u')
        instance = cls(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars,T = T,T_INV = T_INV)
        instance.set_class_attributes()
        instance.create_rgh()
        instance.build_structure()
        return instance

    def set_class_attributes(self):
        # self.geometry_object = geometry_object
        for each in self.kwargs:
            setattr(self, each, self.kwargs[each])
        self.basis = self.lat_pars[0:3]

    def create_rgh(self):
        rgh = UserVars()
        for each,item in self.structure_pars_dict.items():
            if type(item)!=type([]):
                rgh.new_var(each,self.structure_pars_dict[each])
            else:
                for i in range(len(item)):
                    rgh.new_var(each+'_'+str(i+1),item[i])
        self.rgh = rgh
        return self.rgh

    def get_dict_from_rgh(self):
        return {'top_angle':self.rgh.top_angle,'phi':self.rgh.phi,'edge_offset':[self.rgh.edge_offset_1,self.rgh.edge_offset_2], 'angle_offset':self.rgh.angle_offset}

    def build_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode['mode'] == 'BD':
            self._build_structure_BD()
        elif self.binding_mode['mode'] == 'TD':
            self._build_structure_TD()
        elif self.binding_mode['mode'] == 'OS':
            self._build_structure_OS()
        else:
            print('Current binding mode is Not implemented yet!')

    def update_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode['mode'] == 'BD':
            self._update_structure_BD()
        elif self.binding_mode['mode'] == 'TD':
            self._update_structure_TD()
        elif self.binding_mode['mode'] == 'OS':
            self._update_structure_OS()
        else:
            print('Current binding mode is Not implemented yet!')

    def _update_structure_OS(self):
        self._build_structure_OS()

    def _build_structure_OS(self):
        def _cal_geometry(cent_point,phi,r0,rotation_x,rotation_y,rotation_z):
            cent_point=cent_point+np.array([0,0,r0])
            angle=np.pi
            p1_x,p1_y,p1_z=r0*np.cos(phi)*np.sin(0.95531)+cent_point[0],r0*np.sin(phi)*np.sin(0.95531)+cent_point[1],r0*np.cos(0.95531)+cent_point[2]
            p2_x,p2_y,p2_z=r0*np.cos(phi+np.pi*1/3)*np.sin(angle-0.95531)+cent_point[0],r0*np.sin(phi+np.pi*1/3)*np.sin(angle-0.95531)+cent_point[1],r0*np.cos(angle-0.95531)+cent_point[2]
            p3_x,p3_y,p3_z=r0*np.cos(phi+np.pi*2/3)*np.sin(0.95531)+cent_point[0],r0*np.sin(phi+np.pi*2/3)*np.sin(0.95531)+cent_point[1],r0*np.cos(0.95531)+cent_point[2]
            p4_x,p4_y,p4_z=r0*np.cos(phi+np.pi*3/3)*np.sin(angle-0.95531)+cent_point[0],r0*np.sin(phi+np.pi*3/3)*np.sin(angle-0.95531)+cent_point[1],r0*np.cos(angle-0.95531)+cent_point[2]
            p5_x,p5_y,p5_z=r0*np.cos(phi+np.pi*4/3)*np.sin(0.95531)+cent_point[0],r0*np.sin(phi+np.pi*4/3)*np.sin(0.95531)+cent_point[1],r0*np.cos(0.95531)+cent_point[2]
            p6_x,p6_y,p6_z=r0*np.cos(phi+np.pi*5/3)*np.sin(angle-0.95531)+cent_point[0],r0*np.sin(phi+np.pi*5/3)*np.sin(angle-0.95531)+cent_point[1],r0*np.cos(angle-0.95531)+cent_point[2]
            apex_x,apex_y,apex_z=cent_point[0],cent_point[1],cent_point[2]
            T_rot=f4(rotation_x,rotation_y,rotation_z)
            origin=np.array([apex_x,apex_y,apex_z])
            p1=np.array([p1_x,p1_y,p1_z])
            p2=np.array([p2_x,p2_y,p2_z])
            p3=np.array([p3_x,p3_y,p3_z])
            p4=np.array([p4_x,p4_y,p4_z])
            p5=np.array([p5_x,p5_y,p5_z])
            p6=np.array([p6_x,p6_y,p6_z])
            p1_new = np.dot(self.T_INV, (np.dot(T_rot,p1-origin)+origin))
            p2_new = np.dot(self.T_INV, (np.dot(T_rot,p2-origin)+origin))
            p3_new = np.dot(self.T_INV, (np.dot(T_rot,p3-origin)+origin))
            p4_new = np.dot(self.T_INV, (np.dot(T_rot,p4-origin)+origin))
            p5_new = np.dot(self.T_INV, (np.dot(T_rot,p5-origin)+origin))
            p6_new = np.dot(self.T_INV, (np.dot(T_rot,p6-origin)+origin))
            return np.dot(self.T_INV,np.array([apex_x,apex_y,apex_z])),p1_new, p2_new, p3_new, p4_new, p5_new, p6_new

        center_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        center=np.dot(self.T,pt_ct(self.substrate_domain,center_index,self.anchored_ids['offset'][0]))
        apex, p1, p2, p3, p4, p5, p6 = _cal_geometry(cent_point = center, phi = 0, r0 = self.rgh.r_sorbate_O,rotation_x = self.rgh.rotation_x,rotation_y = self.rgh.rotation_y,rotation_z = self.rgh.rotation_z)
        O_coords = [p1,p2,p3,p4, p5, p6]
        O_id = [each for each in self.ids if each!=self.anchor_id]
        assert len(O_coords)==len(O_id), 'len of O_coords and len of O_id does not match!'
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=apex)
        for i in range(len(O_id)):
            self._add_sorbate(domain=self.domain,id_sorbate=O_id[i],el=self.els[self.ids.index(O_id[i])],sorbate_v=O_coords[i])

    def _update_structure_TD(self):
        self._build_structure_TD(update = True)

    def _build_structure_TD(self, update = False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
        p_O3_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][2])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
        p_O2=np.dot(self.T,pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
        p_O3=np.dot(self.T,pt_ct(self.substrate_domain,p_O3_index,self.anchored_ids['offset'][2]))
        p_O3 = self.cal_coor_o3(p_O1, p_O2, p_O3)
        if not update:
            octahedra_distortion=octahedra.share_face(np.array([p_O1,p_O2,p_O3]),self.binding_mode['mirror'])
            octahedra_distortion.share_face_init(flag='regular_triangle',dr = [self.rgh.dr1,self.rgh.dr2,self.rgh.dr3])
            self.geometry_object = octahedra_distortion 
        else:
            self.geometry_object.face = np.array([p_O1,p_O2,p_O3])
            self.geometry_object.share_face_init(flag='regular_triangle',dr = [self.rgh.dr1,self.rgh.dr2,self.rgh.dr3])
        O_id = [each for each in self.ids if each!=self.anchor_id]
        O_coords = [getattr(self.geometry_object,each) for each in ['p3', 'p4', 'p5']]
        assert len(O_id)==len(O_coords), 'The len of O_id does not match that for O_coords!'
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=np.dot(self.T_INV,self.geometry_object.center_point))
        for i, each in enumerate(O_id):
            self._add_sorbate(domain=self.domain,id_sorbate=each,el='O',sorbate_v=np.dot(self.T_INV,O_coords[i]))
                
    def _update_structure_BD(self):
        self._build_structure_BD(update = True)

    def _build_structure_BD(self, update = False):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
        p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
        anchor_index=None
        if self.anchored_ids['anchor_ref']!=None:
            anchor_index=list(self.substrate_domain.id).index(self.anchored_ids['anchor_ref'])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
        p_O2=np.dot(self.T,pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
        anchor=None
        if anchor_index!=None:
            anchor=np.dot(self.T,pt_ct(self.substrate_domain,anchor_index,self.anchored_ids['anchor_offset']))
        #print "O1",p_O1
        #print "O2",p_O2
        # print(pt_ct(self.substrate_domain,anchor_index,self.anchored_ids['anchor_offset']))
        # print(anchor_index,self.substrate_domain.x[anchor_index],self.substrate_domain.y[anchor_index],self.substrate_domain.z[anchor_index])
        if not update:
            octahedra_distortion=octahedra.share_edge(edge=np.array([p_O1,p_O2]))
            octahedra_distortion.cal_p2(ref_p=anchor,phi=self.rgh.phi/180*np.pi)
            octahedra_distortion.share_face_init(octahedra_distortion.flag, dr = [self.rgh.dr1,self.rgh.dr2,self.rgh.dr3])
            self.geometry_object = octahedra_distortion
        else:
            self.geometry_object.edge=np.array([p_O1,p_O2])
            self.geometry_object.cal_p2(ref_p=anchor,phi=self.rgh.phi/180*np.pi)
            self.geometry_object.share_face_init(self.geometry_object.flag,dr = [self.rgh.dr1,self.rgh.dr2,self.rgh.dr3])
        
        O_id = [each for each in self.ids if each!=self.anchor_id]
        O_coords = [getattr(self.geometry_object,each) for each in ['p2','p3', 'p4', 'p5']]
        assert len(O_id)==len(O_coords), 'The len of O_id does not match that for O_coords!'
        self._add_sorbate(domain=self.domain,id_sorbate=self.anchor_id,el=self.els[self.ids.index(self.anchor_id)],sorbate_v=np.dot(self.T_INV,self.geometry_object.center_point))
        for i, each in enumerate(O_id):
            self._add_sorbate(domain=self.domain,id_sorbate=each,el='O',sorbate_v=np.dot(self.T_INV,O_coords[i]))
            #return [np.dot(T_INV,pyramid_distortion.apex)/basis,np.dot(T_INV,pyramid_distortion.p2)/basis

    def make_atm_group(self, instance_name = 'instance_name'):
        atm_gp = model_2.AtomGroup(instance_name = instance_name)
        for id in self.domain.id:
            atm_gp.add_atom(self.domain, id)
        return atm_gp

class CarbonOxygenMotif(StructureMotif):
    def __init__(self,domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs):
        super(CarbonOxygenMotif, self).__init__(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs)
        #self.kwargs = kwargs
        #self.create_rgh()
        #self.build_structure()
        #self.create_sorbate_atom_groups()

    @classmethod
    def generate_script_from_setting_table(cls, use_predefined_motif = False, predefined_motif = '', structure_index = 1, kwargs = {}):
        settings = {}
        if use_predefined_motif:
            temp = globals()[predefined_motif]
            settings['substrate_domain'] = str(temp.get('substrate_domain', 'surface_1'))
            settings['xyzu_oc_m'] = str(temp.get('xyzu_oc_m', [0.5, 0.5, 1.5, 0.1, 1, 1]))
            settings['els'] = str(temp.get('els', ['O','C','C','O']))
            settings['flat_down_index'] = str(temp.get('flat_down_index',[2]))
            settings['anchor_index_list'] = str(temp.get('anchor_index_list',[1, None, 1, 2]))
            settings['lat_pars'] = str(temp.get('lat_pars',[3.615, 3.615, 3.615, 90, 90, 90]))
            settings['structure_pars_dict'] = str(temp.get('structure_pars_dict', {'r':1.5, 'delta':0}))
            settings['binding_mode'] = str(temp.get('binding_mode', 'OS'))
            settings['structure_index'] = structure_index
        else:
            settings['substrate_domain'] = str(kwargs.get('substrate_domain', 'surface_1'))
            settings['xyzu_oc_m'] = str(kwargs.get('xyzu_oc_m', [0.5, 0.5, 1.5, 0.1, 1, 1]))
            settings['els'] = str(kwargs.get('els', ['O','C','C','O']))
            settings['flat_down_index'] = str(kwargs.get('flat_down_index',[2]))
            settings['anchor_index_list'] = str(kwargs.get('anchor_index_list',[1, None, 1, 2]))
            settings['lat_pars'] = str(kwargs.get('lat_pars',[3.615, 3.615, 3.615, 90, 90, 90]))
            settings['structure_pars_dict'] = str(kwargs.get('structure_pars_dict', {'r':1.5, 'delta':0}))
            settings['binding_mode'] = str(kwargs.get('binding_mode', 'OS'))
            settings['structure_index'] = structure_index
        return cls.generate_script_snippet(substrate_domain = settings['substrate_domain'],
                                           xyzu_oc_m = settings['xyzu_oc_m'], 
                                           els = settings['els'], 
                                           flat_down_index = settings['flat_down_index'], 
                                           anchor_index_list = settings['anchor_index_list'], 
                                           lat_pars = settings['lat_pars'],
                                           structure_pars_dict = settings['structure_pars_dict'], 
                                           binding_mode = settings['binding_mode'], 
                                           structure_index = settings['structure_index'])

    @classmethod
    def generate_script_snippet(cls, substrate_domain, xyzu_oc_m = str([0.5, 0.5, 1.5, 0.1, 1, 1]), els = str(['O','C','C','O']), flat_down_index = str([2]), anchor_index_list = str([1, None, 1, 2 ]), lat_pars = str([3.615, 3.615, 3.615, 90, 90, 90]),structure_pars_dict = str({'r':1.5, 'delta':0}), binding_mode = 'OS', structure_index = 1):
        #structure_index = structure_index
        instance_name = 'sorbate_instance_{}'.format(structure_index)
        rgh_name = 'rgh_sorbate_{}'.format(structure_index)
        domain_name = 'domain_sorbate_{}'.format(structure_index)
        atm_gp_name = 'atm_gp_sorbate_{}'.format(structure_index)
        line1 = f"{instance_name} = sorbate_tool.CarbonOxygenMotif.build_instance(substrate_domain = {substrate_domain}, xyzu_oc_m = {xyzu_oc_m},els={els},flat_down_index = {flat_down_index},anchor_index_list={anchor_index_list},lat_pars = {lat_pars},structure_pars_dict = {structure_pars_dict}, binding_mode = \'{binding_mode}\',T = unitcell.lattice.RealTM, T_INV = unitcell.lattice.RealTMInv)"
        line2 = f"{instance_name}.set_coordinate_all_rgh()"
        line3 = f"{domain_name} = {instance_name}.domain"
        line4 = f"{rgh_name} = {instance_name}.rgh"
        line5 = f"{atm_gp_name} = {instance_name}.make_atm_group(instance_name = \'{atm_gp_name}\')"
        return('\n'.join([line1,line2,line3,line4,line5]),f"    {instance_name}.set_coordinate_all_rgh()")

    @classmethod
    def build_instance(cls,substrate_domain = None, xyzu_oc_m = [0.5, 0.5, 1.5, 0.1, 1, 1], els = ['O','C','C','O'], flat_down_index = [2],anchor_index_list = [1, None, 1, 2 ], lat_pars = [3.615, 3.615, 3.615, 90, 90, 90], structure_pars_dict = {'r':1.5, 'delta':0}, binding_mode = 'OS', T= None, T_INV = None):
        #initialize slab
        domain = model_2.Slab(T_factor = 'u')

        #make ids from els
        ids_all = deepcopy(els)
        for each in set(els):
            index_temp_all = list(np.where(np.array(els) == each)[0])
            for index_temp in index_temp_all:
                ids_all[index_temp] = "{}{}".format(ids_all[index_temp], index_temp_all.index(index_temp)+1)
        ids = deepcopy(ids_all)
        anchor_id = ids_all[anchor_index_list.index(None)]

        #build r, gamma, delta names
        r_list_names = []
        gamma_list_names = []
        delta_list_names = []
        for i in range(len(anchor_index_list)):
            each = anchor_index_list[i]
            if each==None:
                each = i
            delta_list_names.append("delta_{}_{}".format(ids_all[i],ids_all[each]))
            r_list_names.append("r_{}_{}".format(ids_all[i],ids_all[each]))
            gamma_list_names.append("gamma_{}_{}".format(ids_all[i],ids_all[each]))

        #build instance
        instance = cls(domain = domain, ids=ids, els=els, anchor_id=anchor_id, substrate_domain=substrate_domain, anchored_ids = [], binding_mode = binding_mode, structure_pars_dict = structure_pars_dict, lat_pars = lat_pars)
        #set attributes to the instance
        instance.set_class_attributes(xyzu_oc_m, r_list_names, delta_list_names, gamma_list_names, anchor_index_list, flat_down_index, T, T_INV)
        instance.create_rgh()
        instance.build_structure()
        #setattr(instance,'T',T)
        #setattr(instance,'T',T_INV)

        return instance

    def set_class_attributes(self,xyzu_oc_m, r_list_names, delta_list_names, gamma_list_names, anchor_index_list, flat_down_index, T, T_INV):
        self.xyzu_oc_m_begin = xyzu_oc_m
        self.lat_abc = np.array(self.lat_pars[0:3])
        self.lat_angles = np.array(self.lat_pars[3:])
        self.r_list_names = r_list_names
        self.delta_list_names = delta_list_names
        self.gamma_list_names = gamma_list_names
        self.r_list = [self.structure_pars_dict['r']]*len(self.r_list_names)
        self.delta_list = [0]*len(r_list_names)
        self.gamma_list = [0]*len(r_list_names)
        #all items left of anchor_id should be 1, while items right of anchor_id should be 0. We set anchor_id to have handedness of 0 as a placeholder.
        self.gamma_handedness = [1]*anchor_index_list.index(None)+[0]+[0]*(len(anchor_index_list)-anchor_index_list.index(None)-1)
        self.new_anchor_list = []
        self.ids_flat_down = []
        self.T = T
        self.T_INV = T_INV
        for i in anchor_index_list:
            if i!=None:
                self.new_anchor_list.append(self.ids[i])
            else:
                self.new_anchor_list.append(self.anchor_id)
        self.rot_ang_x_list = [0] * len(self.ids)
        self.rot_ang_y_list = [0] * len(self.ids)
        self.flat_down_index = flat_down_index
        if type(flat_down_index)!=type([]):
            pass
        else:
            for each_index in flat_down_index:
                self.delta_list[each_index] = 0
                self.ids_flat_down.append(self.ids[each_index])

    def create_rgh(self):
        rgh = UserVars()
        rgh.new_var('gamma',0)
        rgh.new_var('rot_ang_x',0)
        rgh.new_var('rot_ang_y',0)
        for r in self.r_list_names:
            rgh.new_var(r, self.structure_pars_dict['r'])
        for i in range(len(self.delta_list_names)):
            delta = self.delta_list_names[i]
            if i in self.flat_down_index:
                rgh.new_var(delta, 0)
            else:
                rgh.new_var(delta, 90)
        self.rgh = rgh
        return self.rgh

    def build_structure(self):
        #to be implimented for each specific motif
        if self.binding_mode == 'OS':
            for i in range(len(self.ids)):
                self.domain.add_atom(self.ids[i],self.els[i],*self.xyzu_oc_m_begin)
            self.set_coordinate_all_rgh()
        else:
            print('Current binding mode is Not implemented yet!')

    def make_atm_group(self, instance_name = 'instance_name'):
        atm_gp = model_2.AtomGroup(instance_name = instance_name)
        for id in self.domain.id:
            atm_gp.add_atom(self.domain, id)
        return atm_gp
    
    def set_coordinate_all_rgh(self):
        r_list = [getattr(self.rgh, each) for each in self.r_list_names]
        delta_list = [getattr(self.rgh, each) for each in self.delta_list_names]
        gamma_list = [self.rgh.gamma+180*each for each in self.gamma_handedness]
        if hasattr(self.rgh, 'rot_ang_x'):
            rot_ang_x_list = [self.rgh.rot_ang_x]*len(self.r_list_names)
        else:
            rot_ang_x_list = [0]*len(self.r_list_names)
        if hasattr(self.rgh, 'rot_ang_y'):
            rot_ang_y_list = [self.rgh.rot_ang_y]*len(self.r_list_names)
        else:
            rot_ang_y_list = [0]*len(self.r_list_names)
        self.set_coordinate_all(r_list, delta_list, gamma_list, rot_ang_x_list, rot_ang_y_list, self.new_anchor_list)

    def set_coordinate_all(self, r_list = None, delta_list = None, gamma_list = None, rot_ang_x_list = None, rot_ang_y_list = None, new_anchor_list = None):
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
        if rot_ang_x_list!=None:
            assert len(rot_ang_x_list) == len(self.rot_ang_x_list), 'Dimensions of rot_ang_x_list must match!'
            self.rot_ang_x_list = rot_ang_x_list
        if rot_ang_y_list!=None:
            assert len(rot_ang_y_list) == len(self.rot_ang_y_list), 'Dimensions of rot_ang_y_list must match!'
            self.rot_ang_y_list = rot_ang_y_list

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
        
        for id in self.ids:
            index = list(self.ids).index(id)
            self.set_coordinate(id, self.cal_coordinate(id, self.r_list[index], self.delta_list[index], self.gamma_list[index], self.rot_ang_x_list[index],self.rot_ang_y_list[index], new_anchor_list[index]))
    
    def set_coordinate(self, id, coords):
        index = np.where(self.domain.id == id)[0][0]
        self.domain.x[index], self.domain.y[index], self.domain.z[index] = coords
        items = ['dx1','dx2','dx3','dx4','dy1','dy2','dy3','dy4','dz1','dz2','dz3','dz4']
        for each in items:
           getattr(self.domain,each)[index] = 0

    def cal_coordinate(self, id, r, delta, gamma, rot_ang_x = 0, rot_ang_y =0, new_anchor_id = None):
        if id in self.ids_flat_down:
            delta = 0
        if id==self.anchor_id:
            return self.extract_coord(self.anchor_id)

        anchor_atom_coords = np.dot(self.T, self.extract_coord(self.anchor_id))
        z_temp_cart = r*np.sin(np.deg2rad(delta))
        y_temp_cart = r*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(gamma))
        x_temp_cart = r*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(gamma))

        #now rotate the atom about x and y axis
        if rot_ang_x != 0:
            x_temp_cart, y_temp_cart, z_temp_cart = np.dot(rotation_matrix([1,0,0], rot_ang_x), [x_temp_cart, y_temp_cart, z_temp_cart])
        if rot_ang_y != 0:
            x_temp_cart, y_temp_cart, z_temp_cart = np.dot(rotation_matrix([0,1,0], rot_ang_y), [x_temp_cart, y_temp_cart, z_temp_cart])
        if new_anchor_id == None:
            return np.dot(self.T_INV,anchor_atom_coords + np.array([x_temp_cart, y_temp_cart, z_temp_cart]))
        else:
            return np.dot(self.T_INV,np.dot(self.T,self.extract_coord(new_anchor_id))+np.array([x_temp_cart, y_temp_cart, z_temp_cart]))

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

class Gaussian(StructureMotif):
    def __init__(self,domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs):
        super(Gaussian, self).__init__(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars, **kwargs)

    @classmethod
    def generate_script_from_setting_table(cls, use_predefined_motif = False, predefined_motif = '', structure_index = 1, kwargs = {}):
        if use_predefined_motif:
            temp = globals()[predefined_motif]
            temp['substrate_domain'] = 'surface_{}'.format(structure_index)
        else:
            temp = kwargs
        ids = eval(str(temp.get('ids')))
        els = eval(str(temp.get('els')))
        anchor_id = str(temp.get('anchor_id'))
        substrate_domain = str(temp.get('substrate_domain'))
        anchored_ids = str({'attach_atm_ids':eval(temp.get('attach_atm_ids')),'offset':eval(temp.get('offset')),'anchor_ref':temp.get('anchor_ref'),'anchor_offset':temp.get('anchor_offset')})
        lat_pars = str(temp.get('lat_pars'))
        binding_mode = str({'mode':'OS','peak_number':len(ids)})
        structure_pars_dict = str({'first_peak_height':eval(temp['first_peak_height']),'inter_peak_spacing':eval(temp['inter_peak_spacing'])})
        #T = str(temp['T'])
        #T_INV = str(temp['T_INV'])
        return cls.generate_script_snippet(substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index)

    @classmethod
    def generate_script_snippet(cls, substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars, structure_index):
        instance_name = 'sorbate_instance_{}'.format(structure_index)
        rgh_name = 'rgh_sorbate_{}'.format(structure_index)
        domain_name = 'domain_sorbate_{}'.format(structure_index)
        atm_gp_name = 'atm_gp_sorbate_{}'.format(structure_index)
        line1 = "{} = sorbate_tool.Gaussian.build_instance(substrate_domain = {},anchored_ids={},binding_mode = {},structure_pars_dict={},anchor_id = '{}',ids = {}, els = {}, lat_pars = {}, T = unitcell.lattice.RealTM, T_INV = unitcell.lattice.RealTMInv)".format(instance_name,substrate_domain,anchored_ids,binding_mode,structure_pars_dict,anchor_id,ids,els,lat_pars)
        line2 = f"{domain_name} = {instance_name}.domain"
        line3 = f"{rgh_name} = {instance_name}.rgh"
        line4 = f"{atm_gp_name} = {instance_name}.make_atm_group(instance_name = \'{atm_gp_name}\')"
        #print(ids)
        lines = [f"{atm_gp_name}_{i+1} = {atm_gp_name}[{i}]" for i in range(len(ids))]
        return('\n'.join([line1,line2,line3,line4]+lines),f"    {instance_name}.update_structure()")

    @classmethod
    def build_instance(cls,substrate_domain = None,anchored_ids = {'attach_atm_ids':['id1','id2'],'offset':[None,None],'anchor_ref':None,'anchor_offset':None}, binding_mode = {'mode':'BD'},
                       structure_pars_dict = {'phi':0.,'edge_offset':[0,0]},anchor_id='pb_id',ids = ['Pb_id','id1','id2'], els = ['Pb','O','O'], lat_pars = np.array([5.038,5.434,7.3707,90,90,90]), T=None,T_INV=None):
        #initialize slab
        domain = model_2.Slab(T_factor = 'u')
        instance = cls(domain, ids, els, anchor_id, substrate_domain, anchored_ids, binding_mode, structure_pars_dict, lat_pars,T = T,T_INV = T_INV)
        instance.set_class_attributes()
        instance.create_rgh()
        instance.build_structure()
        return instance

    def set_class_attributes(self):
        # self.geometry_object = geometry_object
        for each in self.kwargs:
            setattr(self, each, self.kwargs[each])
        self.basis = self.lat_pars[0:3]

    def create_rgh(self):
        rgh = UserVars()
        for each,item in self.structure_pars_dict.items():
            if type(item)!=type([]):
                rgh.new_var(each,self.structure_pars_dict[each])
            else:
                for i in range(len(item)):
                    rgh.new_var(each+'_'+str(i+1),item[i])
        self.rgh = rgh
        return self.rgh

    def get_dict_from_rgh(self):
        return {'top_angle':self.rgh.top_angle,'phi':self.rgh.phi,'edge_offset':[self.rgh.edge_offset_1,self.rgh.edge_offset_2], 'angle_offset':self.rgh.angle_offset}

    def build_structure(self):
        #to be implimented for each specific motif
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain_,p_O1_index,symbol:np.array([domain_.x[p_O1_index]+domain_.dx1[p_O1_index]+domain_.dx2[p_O1_index]+domain_.dx3[p_O1_index],domain_.y[p_O1_index]+domain_.dy1[p_O1_index]+domain_.dy2[p_O1_index]+domain_.dy3[p_O1_index],domain_.z[p_O1_index]+domain_.dz1[p_O1_index]+domain_.dz2[p_O1_index]+domain_.dz3[p_O1_index]])+self._translate_offset_symbols(symbol)
        if len(self.anchored_ids['attach_atm_ids'])>=2:
            p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
            p_O2_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][1])
            p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
            p_O2=np.dot(self.T,pt_ct(self.substrate_domain,p_O2_index,self.anchored_ids['offset'][1]))
            ref_pt = ((p_O1 + p_O2)/2)
        else:
            p_O1_index=list(self.substrate_domain.id).index(self.anchored_ids['attach_atm_ids'][0])
            p_O1=np.dot(self.T,pt_ct(self.substrate_domain,p_O1_index,self.anchored_ids['offset'][0]))
            ref_pt = p_O1
        height_list = ref_pt[2]+np.array([self.rgh.inter_peak_spacing*i+self.rgh.first_peak_height for i in range(self.binding_mode['peak_number'])])
        coords = [np.array([ref_pt[0], ref_pt[1], each]) for each in height_list]
        assert len(coords)==len(self.ids), 'The len of coords and the len of ids do not match!'
        for i, each in enumerate(self.ids):
            self._add_sorbate(domain=self.domain,id_sorbate=each,el=self.els[i],sorbate_v=np.dot(self.T_INV,coords[i]))

    def update_structure(self):
        #to be implimented for each specific motif
        self.build_structure()

    def make_atm_group(self, instance_name = 'instance_name'):
        atm_gps = []
        for i in range(len(self.domain.id)):
            id = self.domain.id[i]
            atm_gp = model_2.AtomGroup(instance_name = instance_name+'_{}'.format(i+1))
            atm_gp.add_atom(self.domain, id)
            atm_gps.append(atm_gp)
        return atm_gps

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
class CarbonOxygenMotif2(object):
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
        rgh.new_var('rot_ang_x',0)
        rgh.new_var('rot_ang_y',0)
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
        instance.rot_ang_x_list = [0] * len(instance.ids)
        instance.rot_ang_y_list = [0] * len(instance.ids)
        return instance

    def set_coordinate_all_rgh(self):
        r_list = [getattr(self.rgh, each) for each in self.r_list_names]
        delta_list = [getattr(self.rgh, each) for each in self.delta_list_names]
        gamma_list = [self.rgh.gamma+180*each for each in self.gamma_handedness]
        if hasattr(self.rgh, 'rot_ang_x'):
            rot_ang_x_list = [self.rgh.rot_ang_x]*len(self.r_list_names)
        else:
            rot_ang_x_list = [0]*len(self.r_list_names)
        if hasattr(self.rgh, 'rot_ang_y'):
            rot_ang_y_list = [self.rgh.rot_ang_y]*len(self.r_list_names)
        else:
            rot_ang_y_list = [0]*len(self.r_list_names)
        self.set_coordinate_all(r_list, delta_list, gamma_list, rot_ang_x_list, rot_ang_y_list, self.new_anchor_list)


    def set_coordinate_all(self, r_list = None, delta_list = None, gamma_list = None, rot_ang_x_list = None, rot_ang_y_list = None, new_anchor_list = None):
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
        if rot_ang_x_list!=None:
            assert len(rot_ang_x_list) == len(self.rot_ang_x_list), 'Dimensions of rot_ang_x_list must match!'
            self.rot_ang_x_list = rot_ang_x_list
        if rot_ang_y_list!=None:
            assert len(rot_ang_y_list) == len(self.rot_ang_y_list), 'Dimensions of rot_ang_y_list must match!'
            self.rot_ang_y_list = rot_ang_y_list

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
            self.set_coordinate(id, self.cal_coordinate(id, self.r_list[index], self.delta_list[index], self.gamma_list[index], self.rot_ang_x_list[index],self.rot_ang_y_list[index], new_anchor_list[index]))

    def cal_coordinate(self, id, r, delta, gamma, rot_ang_x = 0, rot_ang_y =0, new_anchor_id = None):
        if id in self.ids_flat_down:
            delta = 0
        anchor_atom_coords = self.extract_coord(self.anchor_id)*self.lat_abc
        z_temp_cart = r*np.sin(np.deg2rad(delta))
        y_temp_cart = r*np.cos(np.deg2rad(delta))*np.sin(np.deg2rad(gamma))
        x_temp_cart = r*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(gamma))

        #now rotate the atom about x and y axis
        if rot_ang_x != 0:
            x_temp_cart, y_temp_cart, z_temp_cart = np.dot(rotation_matrix([1,0,0], rot_ang_x), [x_temp_cart, y_temp_cart, z_temp_cart])
        if rot_ang_y != 0:
            x_temp_cart, y_temp_cart, z_temp_cart = np.dot(rotation_matrix([0,1,0], rot_ang_y), [x_temp_cart, y_temp_cart, z_temp_cart])
        if new_anchor_id == None:
            return (anchor_atom_coords + [x_temp_cart, y_temp_cart, z_temp_cart])/self.lat_abc
        else:
            return (self.extract_coord(new_anchor_id)*self.lat_abc + [x_temp_cart, y_temp_cart, z_temp_cart])/self.lat_abc


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