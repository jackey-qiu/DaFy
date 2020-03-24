# -*- coding: utf-8 -*-
import models.sxrd_new1 as model
import numpy as np
import scipy.spatial as spatial
from operator import mul
import operator
import os
from numpy.linalg import inv
from copy import deepcopy
from random import uniform

"""
small functions for different purposes 

"""

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

#function to build reference bulk and surface slab
def add_atom_in_slab(slab,filename,attach='',height_offset=0):
    f=open(filename)
    lines=f.readlines()
    for line in lines:
        if line[0]!='#':
            items=line.strip().rsplit(',')
            slab.add_atom(str(items[0].strip())+attach,str(items[1].strip()),float(items[2]),float(items[3]),float(items[4])+height_offset,float(items[5]),float(items[6]),float(items[7]))

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