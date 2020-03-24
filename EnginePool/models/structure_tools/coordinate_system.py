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

class CoordinateSystem():
    def __init__():
        pass
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