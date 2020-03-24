# -*- coding: utf-8 -*-
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
from geometry_modules import *
from . import domain_creator

"""functions in this class
adding_oxygen: add one molecule in the frame of spherical sphere (r, theta, phi should be specified)
add_oxygen_pair: add two waters, the associated postions are determined by a ref point (not an atom position)r (in angstrom), and alpha
add_oxygen_pair2: add two waters, the ref point is determined by an atom postion, so also specify the v_shift magnitude
add_oxygen_pair_sphere: the added water will be on one of the sphere point (the sphere defined by the ref atom position + r), and alpha_list and theta_list need to be specified
add_oxygen_triple_linear: same as add_oxygen_pair, but the ref_ps will be occupied by an oxygen
add_oxygen_triple_circle: the three waters will be placed on the circle defined by a ref_point and r
"""

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

#extract xyz for atom with id in domain
def extract_coor(domain,id):
    index=np.where(domain.id==id)[0][0]
    x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
    y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
    z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
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

class domain_creator_water():
    def __init__(self,ref_domain,id_list,terminated_layer=0,domain_tag='_D1',new_var_module=None):
        #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
        #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
        #function of set_new_vars
        self.ref_domain=ref_domain
        self.id_list=id_list
        self.terminated_layer=terminated_layer
        self.domain_tag=domain_tag
        self.share_face,self.share_edge,self.share_corner=(False,False,False)
        #self.anchor_list=[]
        self.polyhedra_list=[]
        self.new_var_module=new_var_module
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
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            #print id in new_domain_B.id
            new_domain_B.del_atom(id)
        new_domain_A.id=map(lambda x:x+self.domain_tag+'A',new_domain_A.id)
        new_domain_B.id=map(lambda x:x+self.domain_tag+'B',new_domain_B.id)
        return new_domain_A.copy(),new_domain_B.copy()

    def adding_oxygen(self,domain,o_id,sorbate_coor,r,theta,phi):
        #sorbate_coor and r are in angstrom
        #the sorbate_coor is the origin of a sphere, oxygen added a point determined by r theta and phi
        basis=np.array([5.038,5.434,7.3707])
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        o_coor=(np.array([x,y,z])+sorbate_coor)/basis
        o_index=None
        try:
            o_index=np.where(domain.id==o_id)[0][0]
        except:
            domain.add_atom( o_id, "O",  o_coor[0] ,o_coor[1], o_coor[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
        if o_index!=None:
            domain.x[o_index]=o_coor[0]
            domain.y[o_index]=o_coor[1]
            domain.z[o_index]=o_coor[2]


    def add_oxygen_pair(self,domain,O_ids,ref_point,r,alpha):
        #add single oxygen pair to a ref_point,which does not stand for an atom, the xyz for this point will be set as
        #three fitting parameters.
        ref_pt=[5.038,5.434,7.3707]*np.array(ref_point)
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        index_1,index_2=0,0
        point1,point2=[],[]
        try:
            index_1=np.where(domain.id==O_ids[0])[0][0]
            index_2=np.where(domain.id==O_ids[1])[0][0]
        except:
            point1=np.array([ref_pt[0]-x_shift,ref_pt[1]-y_shift,ref_pt[2]])
            point2=np.array([ref_pt[0]+x_shift,ref_pt[1]+y_shift,ref_pt[2]])
            domain.add_atom(id=O_ids[0],element='O',x=point1[0],y=point1[1],z=point1[2])
            domain.add_atom(id=O_ids[1],element='O',x=point2[0],y=point2[1],z=point2[2])
        if not((index_1==0)&(index_2==0)):
            point1=np.array([ref_pt[0]-x_shift,ref_pt[1]-y_shift,ref_pt[2]])
            point2=np.array([ref_pt[0]+x_shift,ref_pt[1]+y_shift,ref_pt[2]])
            domain.x[index_1]=point2[0]
            domain.y[index_1]=point2[1]
            domain.z[index_1]=point2[2]
            domain.x[index_2]=point1[0]
            domain.y[index_2]=point1[1]
            domain.z[index_2]=point1[2]
        return np.append([point1],[point2],axis=0)

    def add_oxygen_pair2(self,domain,ref_id,O_ids,v_shift,r,alpha):
    #v_shift and r are in unit of angstrom
        basis=np.array([5.038,5.434,7.3707])
        ref_point=None
        if len(ref_id)==1:
            ref_index=np.where(domain.id==ref_id)[0][0]
            ref_point=[domain.x[ref_index],domain.y[ref_index],domain.z[ref_index]]*basis+[0,0,v_shift]
        if len(ref_id)==2:
            ref_index=[np.where(domain.id==ref_id[0])[0][0],np.where(domain.id==ref_id[1])[0][0]]
            ref_point1=[domain.x[ref_index[0]],domain.y[ref_index[0]],domain.z[ref_index[0]]]*basis
            ref_point2=[domain.x[ref_index[1]],domain.y[ref_index[1]],domain.z[ref_index[1]]]*basis
            ref_point=(ref_point1+ref_point2)/2+[0,0,v_shift]
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        point1=np.array([ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]])/basis
        point2=np.array([ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]])/basis
        O_index1=None
        O_index2=None
        try:
            O_index1=np.where(domain.id==O_ids[0])[0][0]
            O_index2=np.where(domain.id==O_ids[1])[0][0]
        except:
            domain.add_atom( O_ids[0], "O",  point1[0] ,point1[1], point1[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
            domain.add_atom( O_ids[1], "O",  point2[0] ,point2[1], point2[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
        if O_index1!=None:
            domain.x[O_index1],domain.y[O_index1],domain.z[O_index1]=point1[0],point1[1],point1[2]
            domain.x[O_index2],domain.y[O_index2],domain.z[O_index2]=point2[0],point2[1],point2[2]
        return np.append([point1],[point2],axis=0)

    def add_oxygen_pair2B(self,domain,ref_id,O_ids,v_shift,r,alpha):
    #v_shift and r are in unit of angstrom, and phi in degree
    #use coordinates during fitting rather than freezing the ref to the bulk position
        basis=np.array([5.038,5.434,7.3707])
        ref_point=None
        if len(ref_id)==1:
            ref_point=domain_creator.extract_coor(domain,ref_id)*basis+[0,0,v_shift]
        if len(ref_id)==2:
            ref_point1=domain_creator.extract_coor(domain,ref_id[0])*basis
            ref_point2=domain_creator.extract_coor(domain,ref_id[1])*basis
            ref_point=(ref_point1+ref_point2)/2+[0,0,v_shift]
        x_shift=r*np.cos(alpha/180*np.pi)
        y_shift=r*np.sin(alpha/180*np.pi)
        point1=np.array([ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]])/basis
        point2=np.array([ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]])/basis
        O_index1=None
        O_index2=None
        try:
            O_index1=np.where(domain.id==O_ids[0])[0][0]
            O_index2=np.where(domain.id==O_ids[1])[0][0]
        except:
            domain.add_atom( O_ids[0], "O",  point1[0] ,point1[1], point1[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
            domain.add_atom( O_ids[1], "O",  point2[0] ,point2[1], point2[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
        if O_index1!=None:
            domain.x[O_index1],domain.y[O_index1],domain.z[O_index1]=point1[0],point1[1],point1[2]
            domain.x[O_index2],domain.y[O_index2],domain.z[O_index2]=point2[0],point2[1],point2[2]
        return np.append([point1],[point2],axis=0)

    def add_oxygen_pair_muscovite(self,domain,ref_id,O_ids,v_shift,basis=np.array([5.038,5.434,7.3707])):
    #v_shift and r are in unit of angstrom, and phi in degree
    #use coordinates during fitting rather than freezing the ref to the bulk position
        alpha=60.0605#specifically for muscovite
        r=5.208335#specifically for muscovite
        ref_point=None
        if len(ref_id)==1:
            ref_point=domain_creator.extract_coor(domain,ref_id)*basis+[0,0,v_shift]
        if len(ref_id)==2:
            ref_point1=domain_creator.extract_coor(domain,ref_id[0])*basis
            ref_point2=domain_creator.extract_coor(domain,ref_id[1])*basis
            ref_point=(ref_point1+ref_point2)/2+[0,0,v_shift]
        x_shift=r*np.cos(alpha/180*np.pi)
        y_shift=r*np.sin(alpha/180*np.pi)
        point1=np.array([ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]])/basis
        point2=np.array([ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]])/basis
        O_index1=None
        O_index2=None
        try:
            O_index1=np.where(domain.id==O_ids[0])[0][0]
            O_index2=np.where(domain.id==O_ids[1])[0][0]
        except:
            domain.add_atom( O_ids[0], "O",  point1[0] ,point1[1], point1[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
            domain.add_atom( O_ids[1], "O",  point2[0] ,point2[1], point2[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
        if O_index1!=None:
            domain.x[O_index1],domain.y[O_index1],domain.z[O_index1]=point1[0],point1[1],point1[2]
            domain.x[O_index2],domain.y[O_index2],domain.z[O_index2]=point2[0],point2[1],point2[2]
        return np.append([point1],[point2],axis=0)

    def cal_geometry_pars_from_coors(self,domain,ref_ids,sorbate_ids):
        def _shift_in_unit_cell(coors):
            [x,y,z]=coors
            while x>5.038 or x<0:
                if x<0:
                    x=x+5.038
                elif x>5.038:
                    x=x-5.038
            while y>5.434 or y<0:
                if y<0:
                    y=y+5.434
                elif y>5.434:
                    y=y-5.434
            return np.array([x,y,z])

        basis=np.array([5.038,5.434,7.3707])
        ref_coors=[_shift_in_unit_cell(domain_creator.extract_coor(domain,ref_id)*basis) for ref_id in ref_ids]
        sorbate_coors=[_shift_in_unit_cell(domain_creator.extract_coor(domain,sorbate_id)*basis) for sorbate_id in sorbate_ids]
        y_shift=(sorbate_coors[0]+sorbate_coors[1]-ref_coors[0]-ref_coors[1])[1]/2./5.434
        v_shift=sorbate_coors[0][2]-ref_coors[0][2]
        r=f2(sorbate_coors[0],sorbate_coors[1])/2.
        vec_ref=np.array([1,0,0])
        vec_sorbate=sorbate_coors[1]-sorbate_coors[0]
        alpha=np.arccos(np.dot(vec_ref,vec_sorbate)/f2(vec_ref,np.array([0,0,0]))/f2(vec_sorbate,np.array([0,0,0])))/np.pi*180.
        return v_shift,alpha,y_shift

    def add_single_oxygen(self,domain,ref_id,O_id,v_shift):
    #v_shift and r are in unit of angstrom
    #use coordinates during fitting rather than freezing the ref to the bulk position
    #here only add one water molecule each time (not a couple), so dont need r and alpha
        basis=np.array([5.038,5.434,7.3707])
        ref_point=None
        if len(ref_id)==1:
            ref_point=(domain_creator.extract_coor(domain,ref_id)*basis+[0,0,v_shift])/basis
        if len(ref_id)==2:
            ref_point1=domain_creator.extract_coor(domain,ref_id[0])*basis
            ref_point2=domain_creator.extract_coor(domain,ref_id[1])*basis
            ref_point=((ref_point1+ref_point2)/2+[0,0,v_shift])/basis
        O_index=None
        try:
            O_index=np.where(domain.id==O_id)[0][0]
        except:
            domain.add_atom( O_id, "O",  ref_point[0] ,ref_point[1], ref_point[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
        if O_index!=None:
            domain.x[O_index],domain.y[O_index],domain.z[O_index]=ref_point[0],ref_point[1],ref_point[2]
        return np.array([ref_point])


    def add_single_oxygen_2(self,domain,ref_id,O_id,v_shift):
        #v_shift and r are in unit of angstrom
        #use coordinates during fitting rather than freezing the ref to the bulk position
        #here only add one water molecule each time (not a couple), so dont need r and alpha
        #water coordinates will have 0 for x and y
            basis=np.array([5.038,5.434,7.3707])*[0,0,1]
            ref_point=None
            if len(ref_id)==1:
                ref_point=(domain_creator.extract_coor(domain,ref_id)+[0,0,v_shift/basis[-1]])
            if len(ref_id)==2:
                ref_point1=domain_creator.extract_coor(domain,ref_id[0])*basis
                ref_point2=domain_creator.extract_coor(domain,ref_id[1])*basis
                ref_point=((ref_point1+ref_point2)/2+[0,0,v_shift])/[1,1,basis[-1]]
            O_index=None
            try:
                O_index=np.where(domain.id==O_id)[0][0]
            except:
                domain.add_atom( O_id, "O",  ref_point[0] ,ref_point[1], ref_point[2] ,4.,     1.00000e+00 ,     1.00000e+00 )
            if O_index!=None:
                domain.x[O_index],domain.y[O_index],domain.z[O_index]=ref_point[0],ref_point[1],ref_point[2]
            return np.array([ref_point])

    def add_oxygen_pair_sphere(self,domain,o_id_list=[],sorbate_id='O_1',r=1.,theta_list=[],phi_list=[]):
        #sorbate_coor and r are in angstrom
        #the sorbate_coor is the origin of a sphere, oxygen added at point determined by r theta and phi
        #two oxygens have the same r value
        basis=np.array([5.038,5.434,7.3707])
        index_1=np.where(domain.id==sorbate_id)[0][0]
        ref_x=domain.x[index_1]
        ref_y=domain.y[index_1]
        ref_z=domain.z[index_1]
        sorbate_coor=np.array([ref_x,ref_y,ref_z])*basis
        x1,x2=r*np.cos(phi_list[0])*np.sin(theta_list[0]),r*np.cos(phi_list[1])*np.sin(theta_list[1])
        y1,y2=r*np.sin(phi_list[0])*np.sin(theta_list[0]),r*np.sin(phi_list[1])*np.sin(theta_list[1])
        z1,z2=r*np.cos(theta_list[0]),r*np.cos(theta_list[1])
        o1_coor=(np.array([x1,y1,z1])+sorbate_coor)/basis
        o2_coor=(np.array([x2,y2,z2])+sorbate_coor)/basis
        o1_index=None
        o2_index=None
        try:
            o1_index=np.where(domain.id==o_id_list[0])[0][0]
            o2_index=np.where(domain.id==o_id_list[1])[0][0]
        except:
            domain.add_atom( o_id_list[0], "O",  o1_coor[0] ,o1_coor[1], o1_coor[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
            domain.add_atom( o_id_list[1], "O",  o2_coor[0] ,o2_coor[1], o2_coor[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
        if o1_index!=None:
            domain.x[o1_index]=o1_coor[0]
            domain.y[o1_index]=o1_coor[1]
            domain.z[o1_index]=o1_coor[2]

            domain.x[o2_index]=o2_coor[0]
            domain.y[o2_index]=o2_coor[1]
            domain.z[o2_index]=o2_coor[2]

    def add_oxygen_triple_linear(self,domain,O_ids,ref_point,r,alpha):
        #add single oxygen pair to a ref_point,which itself stands for an atom, the xyz for this point will be set as
        #three fitting parameters.O_id will be attached at the end of each id for the oxygen
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        index_1,index_2,index_3=0,0,0
        try:
            index_1=np.where(domain.id==O_ids[0])[0][0]
            index_2=np.where(domain.id==O_ids[1])[0][0]
            index_3=np.where(domain.id==O_ids[1])[0][0]
        except:
            point1=np.array([ref_pt[0]-x_shift,ref_pt[1]-y_shift,ref_pt[2]])
            point2=np.array([ref_pt[0]+x_shift,ref_pt[1]+y_shift,ref_pt[2]])
            domain.add_atom(id=O_ids[0],element='O',x=point1[0],y=point1[1],z=point1[2],u=1.)
            domain.add_atom(id=O_ids[1],element='O',x=point2[0],y=point2[1],z=point2[2],u=1.)
            domain.add_atom(id=O_ids[2],element='O',x=ref_point[0],y=ref_point[1],z=ref_point[2],u=1.)
        if not((index_1==0)&(index_2==0)&(index_3==0)):
            point1=np.array([ref_pt[0]-x_shift,ref_pt[1]-y_shift,ref_pt[2]])
            point2=np.array([ref_pt[0]+x_shift,ref_pt[1]+y_shift,ref_pt[2]])
            domain.x[index_1],domain.y[index_1],domain.z[index_1]=point1[0],point1[1],point1[2]
            domain.x[index_2],domain.y[index_2],domain.z[index_2]=point2[0],point2[1],point2[2]
            domain.x[index_3],domain.y[index_3],domain.z[index_3]=ref_point[0],ref_point[1],ref_point[2]

    def add_oxygen_triple_circle(self,domain,O_ids,ref_point,r,alpha1,alpha2,alpha3):
        #add triple oxygen to a ref_point,which itself stands for an atom, the xyz for this point will be set as
        #three fitting parameters.O_id will be attached at the end of each id for the oxygen
        x_shift1=r*np.cos(alpha1)
        y_shift1=r*np.sin(alpha1)
        x_shift2=r*np.cos(alpha2)
        y_shift2=r*np.sin(alpha2)
        x_shift3=r*np.cos(alpha3)
        y_shift3=r*np.sin(alpha3)
        index_1,index_2,index_3=0,0,0
        try:
            index_1=np.where(domain.id==O_ids[0])[0][0]
            index_2=np.where(domain.id==O_ids[1])[0][0]
            index_3=np.where(domain.id==O_ids[1])[0][0]
        except:
            point1=np.array(ref_point[0]+x_shift1,ref_point[1]+y_shift1,ref_point[2])
            point2=np.array(ref_point[0]+x_shift2,ref_point[1]+y_shift2,ref_point[2])
            point3=np.array(ref_point[0]+x_shift3,ref_point[1]+y_shift3,ref_point[2])
            domain.add_atom(id=O_ids[0],element='O',x=point1[0],y=point1[1],z=point1[2],u=1.)
            domain.add_atom(id=O_ids[1],element='O',x=point2[0],y=point2[1],z=point2[2],u=1.)
            domain.add_atom(id=O_ids[2],element='O',x=point3[0],y=point3[1],z=point3[2],u=1.)
        if not((index_1==0)&(index_2==0)&(index_3==0)):
            point1=np.array(ref_point[0]+x_shift1,ref_point[1]+y_shift1,ref_point[2])
            point2=np.array(ref_point[0]+x_shift2,ref_point[1]+y_shift2,ref_point[2])
            point3=np.array(ref_point[0]+x_shift3,ref_point[1]+y_shift3,ref_point[2])
            domain.x[index_1],domain.y[index_1],domain.z[index_1]=point1[0],point1[1],point1[2]
            domain.x[index_2],domain.y[index_2],domain.z[index_2]=point2[0],point2[1],point2[2]
            domain.x[index_3],domain.y[index_3],domain.z[index_3]=point3[0],point3[1],point3[2]
