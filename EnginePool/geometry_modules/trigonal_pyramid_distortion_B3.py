import numpy as np
from numpy.linalg import inv
import os
#here only consider the distortion caused by length difference of three edges, it is a tectrahedral configuration basically, but not a regular one
#since the top angle can be any value in [0,2*pi/3]
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
#refer the the associated ppt file when read the comments

basis=np.array([5.038,5.434,7.3707])
#atoms to be checked for distance
#for half layer
atms_cell=[[0.653,1.112,1.903],[0.847,0.612,1.903],[0.306,0.744,1.75],[0.194,0.243,1.75],\
      [0.5,1.019,1.645],[0,0.518,1.645],[0.847,0.876,1.597],[0.653,0.375,1.597]]
#for full layer
#atms_cell=[[0.153,1.062,2.113],[0.347,0.563,2.113],[0.653,1.112,1.903],[0.847,0.612,1.903],[0,0.9691,1.855],[0.5,0.469,1.855],[0.306,0.744,1.75],[0.194,0.243,1.75],\
           #[0.5,1.019,1.645],[0,0.518,1.645],[0.847,0.876,1.597],[0.653,0.375,1.597]]
atms=np.append(np.array(atms_cell),np.array(atms_cell)+[-1,0,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[1,0,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[0,-1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[0,1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[1,1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[-1,-1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[1,-1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[-1,1,0],axis=0)

atms=atms*basis
O1,O2,O3,O4=[0.653,1.1121,1.903]*basis,[0.847,0.6121,1.903]*basis,[0.306,0.744,1.75]*basis,[0.194,0.243,1.75]*basis


class trigonal_pyramid_distortion():
    
    def __init__(self,p0=[0.,0.,0.],p1=[2.,2.,2.],top_angle=1.0,len_offset=[0.,0.]):
        #top angle is p0_A_p1 in ppt file, shoulder_angle is A_P0_CP
        #len_offset[0] is CP_P1 in ppt, the other one not specified in the file
        self.top_angle=top_angle
        self.shoulder_angle=(np.pi-top_angle)/2.
        self.p0,self.p1=np.array(p0),np.array(p1)
        self.len_offset=len_offset
    
    def cal_theta(self):
    #here theta angle is angle A_P0_P1 in ppt file
        dst_p0_p1=f2(self.p0,self.p1)
        right_l=self.len_offset[0]*np.sin(self.shoulder_angle)
        self.theta=self.shoulder_angle+np.arcsin(right_l/dst_p0_p1)
        return self.theta
    
    def cal_edge_len(self):
    #cal the edge length of regular hexahedra
    #sharp angle is angle A_P1_P0 in ppt file(2nd slide)
    #rigth_side is the length of p2p5 in ppt file(1st slide)
        self.sharp_angle=np.pi-self.top_angle-self.theta
        right_side=f2(self.p0,self.p1)*np.sin(self.sharp_angle)
        self.edge_len=right_side/np.sin(np.pi-self.top_angle)
        
    def cal_apex_coor(self,switch=False,phi=0.,mirror=False):
    #basis idea: set a new coordinate frame with p0p1 as the z vector (start from p1)
    #set a arbitrary y vector on the normal plane, and cross product to solve the x vector
    #then use phi and theta (sharp angle) to solve the cross_point(CP on file) and apex (A on file)
    #note phi is in range of [0,2pi]
    
        p0,p1=self.p0,self.p1
        if switch==True:
            p0,p1=self.p1,self.p0
        n_v=p0-p1
        origin=p1
        a,b,c=n_v[0],n_v[1],n_v[2]
        x0,y0,z0=p1[0],p1[1],p1[2]
        ref_p=0
        if c==0:
            ref_p=p1+[0,0,1]
        else:
            ref_p=np.array([1.,1.,(a*(x0-1.)+b*(y0-1.))/c+z0])
        #elif b!=0.:
        #   ref_p=np.array([1.,(a*(x0-1.)+c*(z0-1.))/b+y0,1.])
        #else:
        #    ref_p=np.array([(b*(y0-1.)+c*(z0-1.))/a+x0,1.,1.])
        x_v=f3(np.zeros(3),(ref_p-origin))
        z_v=f3(np.zeros(3),(p0-origin))
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        r1=self.len_offset[0]
        r2=self.len_offset[0]+self.edge_len
        theta=self.sharp_angle
        cross_pt_new = np.array([r1*np.cos(phi)*np.sin(theta),r1*np.sin(phi)*np.sin(theta),r1*np.cos(theta)])
        apex_new = np.array([r2*np.cos(phi)*np.sin(theta),r2*np.sin(phi)*np.sin(theta),r2*np.cos(theta)])
        self.cross_pt = np.dot(inv(T),cross_pt_new)+origin
        self.apex = np.dot(inv(T),apex_new)+origin
        self.cal_p2(p0,p1,mirror)
        
    def cal_p2(self,p0,p1,mirror=False):
        #basic idea:set z vector rooting from EC to cp, x vector from EC to A (normalized to length of 1)
        #use angle of theta (pi/2 here) and phi (the angle A_EC_P2, can be calculated) to sove P2 finally 
        #if consider mirror then p2 will be on the other side
        side_center=(p0+self.cross_pt)/2.
        origin=side_center
        z_v=f3(np.zeros(3),(self.cross_pt-side_center))
        x_v=f3(np.zeros(3),(self.apex-side_center))
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        theta=np.pi/2
        dst_face_ct_edge_ct=f2(p0,self.cross_pt)/2*np.tan(np.pi/6.)
        dst_p2_edge_ct=f2(p0,self.cross_pt)/2*np.tan(np.pi/3.)
        phi=np.arccos(dst_face_ct_edge_ct/f2(self.apex,(p0+self.cross_pt)/2.))
        if mirror:phi=-phi
        r=dst_p2_edge_ct
        p2_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
        _p2=np.dot(inv(T),p2_new)+origin
        _p2_v=_p2-self.apex
        scale=(f2(_p2,self.apex)+self.len_offset[1])/f2(_p2,self.apex)
        p2_v=_p2_v*scale
        self.p2=p2_v+self.apex
        
    def all_in_all(self,switch=False,phi=0.,mirror=False):
        self.cal_theta()
        self.cal_edge_len()
        self.cal_apex_coor(switch=switch, phi=phi,mirror=mirror)
        
    def print_file(self,file):
        f=open(file,'w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.apex[0],self.apex[1],self.apex[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p2[0],self.p2[1],self.p2[2])
        f.write(s)
        f.close()

#steric_check will check the steric feasibility by changing the rotation angle (0-2pi) and top angle (0-2pi/3)
#the dist bw sorbate(both metal and oxygen) and atms (defined on top) will be cal and compared to the cutting_limit
#higher cutting limit will result in more items in return file (so be wise to choose cutting limit)
#the container has 9 items, ie phi (rotation angle), top_angle, low_dis, apex coors (x,y,z), os coors(x,y,z)
#in which the low_dis is the lowest dist between sorbate and atm (averaged value)
class steric_check(trigonal_pyramid_distortion):
    def __init__(self,p0=O1,p1=O3,len_offset=[0.,0.],cutting_limit=3.):
        self.p0,self.p1=np.array(p0),np.array(p1)
        self.len_offset=len_offset
        self.cutting_limit=cutting_limit
        self.container=np.zeros((1,9))[0:0]
    def steric_check(self,top_ang_res=0.1,phi_res=0.5,switch=False,mirror=False,print_path=None):
        for top in np.arange(1.,2.0,top_ang_res):
            for phi in np.arange(0,np.pi*2,phi_res):
                self.top_angle=top
                self.shoulder_angle=(np.pi-top)/2.
                self.all_in_all(switch=switch,phi=phi,mirror=mirror)
                low_limit=self.cutting_limit*2
                for atm in atms:
                    if ((abs(sum(atm-self.p0))>0.01)&(abs(sum(atm-self.p1))>0.01)):
                        dt_apex_atm,dt_p2_atm=f2(atm,self.apex),f2(atm,self.p2)
                        if (dt_apex_atm<self.cutting_limit)|(dt_p2_atm<self.cutting_limit):
                            low_limit=None
                            break
                        else:
                            if (dt_apex_atm+dt_p2_atm)/2.<low_limit:
                                low_limit=(dt_apex_atm+dt_p2_atm)/2.
                            else:pass
                if low_limit!=None:
                    A,p2=self.apex,self.p2
                    self.container=np.append(self.container,[[phi,top,low_limit,A[0],A[1],A[2],p2[0],p2[1],p2[2]]],axis=0)
                else:pass
        #note here consider the first slab, y and z shiftment has been made properly
        data=np.rec.fromarrays([self.container[:,0],self.container[:,1],\
                                self.container[:,2],self.container[:,3],\
                                self.container[:,4]-0.7558,self.container[:,5]-7.3707,\
                                self.container[:,6],self.container[:,7]-0.7558,\
                                self.container[:,8]-7.3707],names='phi,top,low,A1,A2,A3,P1,P2,P3')
        data.sort(order=('phi','top','low'))
        print "phi,top_angle,low_dt_limit,A_x,A_y,A_z,P2_x,P2_y,P3_z"
        if print_path!=None:
            np.savetxt(print_path,data,"%5.3f")
            print np.loadtxt(print_path)
        else:
            np.savetxt('test',data,"%5.3f")
            print np.loadtxt('test')
            os.remove('test')
    
    def steric_check_only_sorbate(self,top_ang_res=0.1,phi_res=0.5,switch=False,mirror=False,print_path=None):
        for top in np.arange(1.,2.0,top_ang_res):
            for phi in np.arange(0,np.pi*2,phi_res):
                self.top_angle=top
                self.shoulder_angle=(np.pi-top)/2.
                self.all_in_all(switch=switch,phi=phi,mirror=mirror)
                low_limit=self.cutting_limit*2
                for atm in atms:
                    if ((abs(sum(atm-self.p0))>0.01)&(abs(sum(atm-self.p1))>0.01)):
                        dt_apex_atm=f2(atm,self.apex)
                        if (dt_apex_atm<self.cutting_limit):
                            low_limit=None
                            break
                        else:
                            if dt_apex_atm<low_limit:
                                low_limit=dt_apex_atm
                            else:pass
                if low_limit!=None:
                    A=self.apex
                    self.container=np.zeros((1,6))[0:0]
                    self.container=np.append(self.container,[[phi,top,low_limit,A[0],A[1],A[2]]],axis=0)
                else:pass
        #note here consider the first slab, y and z shiftment has been made properly
        data=np.rec.fromarrays([self.container[:,0],self.container[:,1],\
                                self.container[:,2],self.container[:,3],\
                                self.container[:,4]-0.7558,self.container[:,5]-7.3707],\
                                names='phi,top,low,A1,A2,A3')
        data.sort(order=('phi','top','low'))
        print "phi,top_angle,low_dt_limit,A_x,A_y,A_z"
        if print_path!=None:
            np.savetxt(print_path,data,"%5.3f")
            print np.loadtxt(print_path)
        else:
            np.savetxt('test',data,"%5.3f")
            print np.loadtxt('test')
            os.remove('test')
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    