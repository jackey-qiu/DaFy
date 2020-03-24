# -*- coding: utf-8 -*-
import models.sxrd_new1 as model
#to make compatible to historical runs
#import models.sxrd_test5_sym_new_test_new66_2_3 as model2
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
from geometry_modules import *

"""
functions in classes
#build_super_cell:build a 3 by 3 super unit cell with all the adjacent unit cells
#create_equivalent_domains: create chemically equivalent domain
#grouping_sequence_layer: group each atom layer from chemically equivalent domains (four atoms to be grouped)
#grouping_discrete_layer: group single atom from chemically equivalent domains (two atoms to be grouped)
#update_oxygen_single_coordinated: first layer oxygen will relax or contract alone the Fe-O vector
#update_oxygen_p4_symmetry: set a z vecotr and apply the 4-fold rotation on the normal plane
#update_oxygen_p4_symmetry2: project the atoms on xy plane, and then apply the 4-fold rotation
#scale_in_symmetry2:scale the movement along the bond vector
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

class domain_creator_surface():
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
    #build a super cell based on the ref_domain, the super cell is actually nine domains stacking together in xy direction
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
            new_domain_B.del_atom(id)
        return new_domain_A,new_domain_B

    def create_equivalent_domains_2(self):
    #make chemically equivalent domains (chop off top five layers from first domain to get the other domain in hematite case)
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

   
    def grouping_sequence_layer(self, domain=[], first_atom_id=[],sym_file={},id_match_in_sym={},layers_N=1,use_sym=False):
        #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
        #so 4 atoms will group together if consider two chemical equivalent domain
        #domain is list of two chemical equivalent domains
        #first_atom_id is list of first id in id array of two domains
        #sym_file is a library of symmetry file names, the keys are element symbols
        #id_match_in_sym is a library of ids, the order of which match the symmetry operation in the associated sym file
        #layers_N is the number of layer you consider for grouping operation
        #use_sym is a flag to choose the shifting rule (symmetry basis or not)
        atm_gp_list=[]
        for i in range(layers_N):
            index_1=np.where(domain[0].id==first_atom_id[0])[0][0]+i*2
            if sym_file!=None:
                temp_atm_gp=model2.AtomGroup(slab=domain[0],id=str(domain[0].id[index_1]),id_in_sym_file=id_match_in_sym[str(domain[0].el[index_1])],use_sym=use_sym,filename=sym_file[str(domain[0].el[index_1])])
            else:
                temp_atm_gp=model2.AtomGroup(slab=domain[0],id=str(domain[0].id[index_1]),id_in_sym_file=id_match_in_sym[str(domain[0].el[index_1])],use_sym=use_sym,filename=None)
            temp_atm_gp.add_atom(domain[0],str(domain[0].id[index_1+1]))
            index_2=np.where(domain[1].id==first_atom_id[1])[0][0]+i*2
            temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2]))
            temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2+1]))
            atm_gp_list.append(temp_atm_gp)

        return atm_gp_list
            
    def grouping_sequence_layer_new(self, domain=[], first_atom_id=[],layers_N=1):
        #looping the domain eg. domain=[[domain1A,domain1B],[domain2A,domain2B]]
        #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
        #domain is list of domains
        #first_atom_id is list of first id in id array of two domains eg [['id1','id2]],['id11','id21']]
        #dont consdier symmetry relationship(so for oc and u)
        #layers_N is the number of layer you consider for grouping operation

        atm_gp_list=[]
        for i in range(layers_N):
            temp_atm_gp=model.AtomGroup()
            for j in range(len(domain)):
                for k in range(len(domain[j])):
                    index=np.where(domain[j][k].id==first_atom_id[j][k])[0][0]+i*2
                    temp_atm_gp.add_atom(domain[j][k],str(domain[j][k].id[index]))
                    temp_atm_gp.add_atom(domain[j][k],str(domain[j][k].id[index+1]))
            atm_gp_list.append(temp_atm_gp)

        return atm_gp_list
            
    def grouping_sequence_layer_new2(self, domain=[], first_atom_id=[],layers_N=1,matrix_list=[[[1,0,0,0,1,0,0,0,1],[-1,0,0,0,1,0,0,0,1]],[[-1,0,0,0,1,0,0,0,1],[1,0,0,0,1,0,0,0,1]]]):
        #looping the domain eg. domain=[[domain1A,domain1B],[domain2A,domain2B]]
        #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
        #domain is list of domains
        #first_atom_id is list of first id in id array of two domains eg [['id1','id2]],['id11','id21']]
        #dont consdier symmetry relationship(so for oc and u)
        #layers_N is the number of layer you consider for grouping operation

        atm_gp_list=[]
        for i in range(layers_N):
            temp_atm_gp=model.AtomGroup()
            for j in range(len(domain)):
                for k in range(len(domain[j])):
                    matrix=matrix_list[k]
                    #if k==1:#note the k can be either 0 for domainA or 1 for domainB
                    #matrix=matrix_list[::-1]
                    index=np.where(domain[j][k].id==first_atom_id[j][k])[0][0]+i*2
                    temp_atm_gp.add_atom(domain[j][k],str(domain[j][k].id[index]),matrix[0])
                    temp_atm_gp.add_atom(domain[j][k],str(domain[j][k].id[index+1]),matrix[1])
            atm_gp_list.append(temp_atm_gp)

        return atm_gp_list
        
    def grouping_sequence_layer2(self, domain=[], first_atom_id=[],sym_file={},id_match_in_sym={},layers_N=1,use_sym=False):
        #different from first edition, we consider only one domain
        #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
        #so 4 atoms will group together if consider two chemical equivalent domain
        #domain is list of two chemical equivalent domains
        #first_atom_id is list of first id in id array of two domains
        #sym_file is a library of symmetry file names, the keys are element symbols
        #id_match_in_sym is a library of ids, the order of which match the symmetry operation in the associated sym file
        #layers_N is the number of layer you consider for grouping operation
        #use_sym is a flag to choose the shifting rule (symmetry basis or not)
        atm_gp_list=[]
        for i in range(layers_N):
            index_1=np.where(domain[0].id==first_atom_id[0])[0][0]+i*2
            temp_atm_gp=model2.AtomGroup(slab=domain[0],id=str(domain[0].id[index_1]),id_in_sym_file=id_match_in_sym[str(domain[0].el[index_1])],use_sym=use_sym,filename=sym_file[str(domain[0].el[index_1])])
            temp_atm_gp.add_atom(domain[0],str(domain[0].id[index_1+1]))
            atm_gp_list.append(temp_atm_gp)

        return atm_gp_list
        
    def grouping_discrete_layer(self,domain=[],atom_ids=[],sym_file=None,id_match_in_sym={},use_sym=False):
        #we usually do discrete grouping for sorbates, so there is no symmetry used in this case
        index=np.where(domain[0].id==atom_ids[0])[0][0]
        el=domain[0].el[index]
        atm_gp=None
        if use_sym:
            atm_gp=model2.AtomGroup(id_in_sym_file=id_match_in_sym[el],filename=sym_file[el],use_sym=use_sym)
        else:atm_gp=model2.AtomGroup()
        for i in range(len(domain)):
            atm_gp.add_atom(domain[i],atom_ids[i])
        return atm_gp
            
    def grouping_discrete_layer2(self,domain=[],atom_ids=[],sym_array=None,use_sym=False):
        #we usually do discrete grouping for sorbates, so there is no symmetry used in this case
        atm_gp=None
        if use_sym:
            atm_gp=model2.AtomGroup(id_in_sym_file=np.array(atom_ids),filename=sym_array,use_sym=use_sym)
        else:atm_gp=model2.AtomGroup()
        for i in range(len(domain)):
            atm_gp.add_atom(domain[i],atom_ids[i])
        return atm_gp
    
    def grouping_discrete_layer3(self,domain=[],atom_ids=[],sym_array=None):
        #we usually do discrete grouping for sorbates, so there is no symmetry used in this case
        atm_gp=model.AtomGroup()
        for i in range(len(domain)):
            if sym_array==None:
                atm_gp.add_atom(domain[i],atom_ids[i])
            else:
                atm_gp.add_atom(domain[i],atom_ids[i],sym_array[i])
                #print sym_array[i]
        return atm_gp
        
    def grouping_discrete_layer_batch(self,filename):
        gp_list=[]
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                N_half=len(line_split)/2
                domains=[]
                ids=[]
                for i in range(N_half):
                    domains.append(vars(self)[line_split[i]])
                    ids.append(line_split[i+N_half])
                gp_list.append(self.grouping_discrete_layer(domain=domains,atom_ids=ids))
        f.close()
        return tuple(gp_list)
      
    def update_oxygen_single_coordinated(self,domain,O_id,Fe_id,offset=[],scale_factor=1.):
        #the oxygen will relax or contract along the Fe-O bond
        #fix the recursive issue during fitting
        #to update the first layer oxygen which is singly coordinated
        #the ref length is the original Fe-O bond length, which will be scaled by the scale factor
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])
        index_O=np.where(domain.id==O_id)[0][0]
        index_Fe=np.where(domain.id==Fe_id)[0][0]
        basis=np.array([5.038,5.434,7.3707])
        coors_O_original=np.array([domain.x[index_O],domain.y[index_O],domain.z[index_O]])+_translate_offset_symbols(offset[0])
        coors_Fe_original=np.array([domain.x[index_Fe],domain.y[index_Fe],domain.z[index_Fe]])+_translate_offset_symbols(offset[1])
        ref_l=f2(coors_O_original*basis,coors_Fe_original*basis)
        Fe_O_v_new=(coors_O_original-coors_Fe_original)*scale_factor
        coors_O_new=Fe_O_v_new+coors_Fe_original
        dxdydz=coors_O_new-coors_O_original
        domain.dx1[index_O],domain.dy1[index_O],domain.dz1[index_O]=dxdydz[0],dxdydz[1],dxdydz[2]
        return coors_O_new
        

    def update_oxygen_p4_symmetry(self,domain,O_id,Fe_id,offset=[],O_id_in_order=[],dxdy=[0,0]):
        #try to fix the recursive issue
        #the second and third layer of oxygen are arranged in a p4 symmetry
        #Fe_id is the body center, O_Fe will define a z vector, the normal plane to z vector define the xy plane which is the rotation plane
        #only consider the inplane movement fraction in the original coordinate system, ie ignore the calculated dz
        #dxdy is the dxdy shiftment in the rotation plane in unit of Angstrom for the ref atom (first in the O_id_in_order)
        #the other associated shiftment will be calculated based on p4 symmetry configuration
        #finally the dxdy will be converted to dxdy in the original coordinate system
        #make sure the order of oxygen in the O_id_in_order is based on the rotation order
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])
            
        index_O=np.where(domain.id==O_id)[0][0]
        index_Fe=np.where(domain.id==Fe_id)[0][0]
        basis=np.array([5.038,5.434,7.3707])
        coors_O=(np.array([domain.x[index_O],domain.y[index_O],domain.z[index_O]])+_translate_offset_symbols(offset[0]))*basis
        coors_Fe=(np.array([domain.x[index_Fe],domain.y[index_Fe],domain.z[index_Fe]])+_translate_offset_symbols(offset[1]))*basis
        p0,p1=coors_O,coors_Fe
        n_v=p0-p1
        origin=p1
        a,b,c=n_v[0],n_v[1],n_v[2]
        x0,y0,z0=p1[0],p1[1],p1[2]
        ref_p=0
        if c!=0.:
            ref_p=np.array([1.,1.,(a*(x0-1.)+b*(y0-1.))/c+z0])
        elif b!=0.:
            ref_p=np.array([1.,(a*(x0-1.)+c*(z0-1.))/b+y0,1.])
        else:
            ref_p=np.array([(b*(y0-1.)+c*(z0-1.))/a+x0,1.,1.])
        y_v=f3(np.zeros(3),(ref_p-origin))
        z_v=f3(np.zeros(3),(p0-origin))
        x_v=np.cross(y_v,z_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        dxdydz_ref=dxdy+[0]
        M_p4=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        dxdydz_1=np.dot(M_p4,dxdydz_ref)
        dxdydz_2=np.dot(M_p4,dxdydz_1)
        dxdydz_3=np.dot(M_p4,dxdydz_2)
        dxdydz_ref=np.dot(inv(T),dxdydz_ref)/basis
        dxdydz_1=np.dot(inv(T),dxdydz_1)/basis
        dxdydz_2=np.dot(inv(T),dxdydz_2)/basis
        dxdydz_3=np.dot(inv(T),dxdydz_3)/basis
        dxdydz_list=[dxdydz_ref,dxdydz_1,dxdydz_2,dxdydz_3]
        for i in range(len(O_id_in_order)):
            index=np.where(domain.id==O_id_in_order[i])[0][0]
            domain.dx1[index],domain.dy1[index]=dxdydz_list[i][0],dxdydz_list[i][1]
        return dxdydz_list
      
    def update_oxygen_p4_symmetry2(self,domain,Fe_id,O_id_in_order=[],offset=[],theta=0,scale_factor=1):
        #a different algorithem is used in this version
        #the O atoms will be projected on xy plane, and Fe will be set as the center (origin) of the retangular (maybe not retangular)
        #Then apply rotation of theta for each oxygen and scale the final rotated vector by scale_factor
        #The difference bw the rotated and initial vector is the assciated dxdy
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])
            
        index_Fe=np.where(domain.id==Fe_id)[0][0]
        index_Os=[np.where(domain.id==O_id_in_order[i])[0][0] for i in range(len(O_id_in_order))]
        basis=np.array([5.038,5.434,7.3707])
        coors_Fe=(np.array([domain.x[index_Fe],domain.y[index_Fe],domain.z[index_Fe]]))*basis
        coors_Os=[(np.array([domain.x[index_Os[i]],domain.y[index_Os[i]],domain.z[index_Os[i]]])+_translate_offset_symbols(offset[i]))*basis for i in range(len(offset))]
        M=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for i in range(len(O_id_in_order)):
            dxdy=(np.dot(M,(coors_Os[i]-coors_Fe)[:2])*scale_factor+coors_Fe[:2]-coors_Os[i][:2])/basis[:2]
            domain.dx1[index_Os[i]],domain.dy1[index_Os[i]]=dxdy[0],dxdy[1]
        
    def scale_in_symmetry2(self,domain,center_id,scaler,ref_lib,phi_lib,theta_lib,off_set,center_offset=None):
        #different from the previous one is the oxygen atoms will not only relax along the bond valence vector,but also will
        #rotate over the vector in some angle defind by theta(a very small range adjacent to 0) and phi (0-2pi)
        #THE atom will only be allowed to move along the bond vector,to do that we need a center point defined by
        #center_id, and a reference point defining the other end of the vector which is specified by ref_lib
        #you can group several atoms together in the reference library, they will have the same scaler
        #off_set is defined to account the arbitrary movement along x or y directioin for the ref_lib
        #this function must be in sim and after the sorbate updating function, otherwise error will be seen.
        
        #anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
        #x2y2z2 are basis of new coor defined in the original frame,new=T.orig
        f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
        a0_v,b0_v,c0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])                              
        def _offset_translator(offset):
            if offset=='+x':
                return np.array([1.,0.,0.])
            elif offset=='-x':
                return np.array([-1.,0.,0.])
            elif offset=='+y':
                return np.array([0.,1.,0.])
            elif offset=='-y':
                return np.array([0.,-1.,0.])
            else:
                return np.array([0.,0.,0.]) 
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
        center_coor=_extract_coor(domain,center_id)+_offset_translator(center_offset)
        
        for i in ref_lib.keys():
            bond_vt=ref_lib[i]-center_coor
            bond_vt_scaled=bond_vt*scaler
            c_v=bond_vt/(np.dot(bond_vt,bond_vt)**0.5)
            a_v_i=np.array([1.,1.,((center_coor[0]-1.)*c_v[0]+(center_coor[1]-1.)*c_v[1])/c_v[2]+center_coor[2]])
            a_v=a_v_i/(np.dot(a_v_i,a_v_i)**0.5)
            b_v=np.cross(c_v,a_v)
            T=f1(a0_v,b0_v,c0_v,a_v,b_v,c_v)
            r=np.dot(bond_vt_scaled,bond_vt_scaled)**0.5
            theta=theta_lib[i]
            phi=phi_lib[i]
            ox_ps_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
            ox_ps_org=np.dot(inv(T),ox_ps_new)+center_coor
            ref_coor_scaled=ox_ps_org
            
            offset=_offset_translator(off_set[i])
            domain.x[np.where(domain.id==i)[0][0]]=ref_coor_scaled[0]
            domain.y[np.where(domain.id==i)[0][0]]=ref_coor_scaled[1]
            domain.z[np.where(domain.id==i)[0][0]]=ref_coor_scaled[2]

            domain.dx2[np.where(domain.id==i)[0][0]]=-offset[0]
            domain.dy2[np.where(domain.id==i)[0][0]]=-offset[1]
            domain.dz2[np.where(domain.id==i)[0][0]]=-offset[2]
        
