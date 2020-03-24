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

"""functions in this class

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

class BondValenceTool(object):
    def __init__(self):
        pass

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