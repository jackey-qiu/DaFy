import numpy as np
from numpy.linalg import inv
#different from the version 3:p0,p1,p2 and apex will be on the same plane,p2 will be calculated in the same way as apex, but different r
#different from first version is the algorithem to add the p2, now the p2 is one point of the circle, which is on the perpendicular plane of line
#determined by p0 and p2, the plane cut through the apex point
#so now the apex is same distant to the p0,p1 and p2, but the angle bw apex-px and the z vector is not the same any more, so is angle distortion
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
        
    def cal_apex_coor(self,switch=False,phi=0.):
    #basis idea: set a new coordinate frame with p0p1 as the z vector (start from p1)
    #set a arbitrary y vector on the normal plane, and cross product to solve the x vector
    #then use phi and theta (sharp angle) to sove the cross_point(CP on file) and apex (A on file)
    #the calculation of p2 will based on the same coordinate system but a different r being used then
    #and note the same phi values will be used for the calculation of apex and p2
    
        p0,p1=self.p0,self.p1
        if switch==True:
            p0,p1=self.p1,self.p0
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
        r1=self.len_offset[0]
        r2=self.len_offset[0]+self.edge_len
        r3=r2+self.edge_len+self.len_offset[1]
        theta=self.sharp_angle
        cross_pt_new = np.array([r1*np.cos(phi)*np.sin(theta),r1*np.sin(phi)*np.sin(theta),r1*np.cos(theta)])
        apex_new = np.array([r2*np.cos(phi)*np.sin(theta),r2*np.sin(phi)*np.sin(theta),r2*np.cos(theta)])
        p2_new=np.array([r3*np.cos(phi)*np.sin(theta),r3*np.sin(phi)*np.sin(theta),r3*np.cos(theta)])
        self.cross_pt = np.dot(inv(T),cross_pt_new)+origin
        self.apex = np.dot(inv(T),apex_new)+origin
        self.p2=np.dot(inv(T),p2_new)+origin

        
    def all_in_all(self,switch=False,phi=0.):
        self.cal_theta()
        self.cal_edge_len()
        self.cal_apex_coor(switch=switch, phi=phi)
        
    def print_file(self):
        f=open('/mnt/sphalerite/tetrahedra_test.xyz','w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.apex[0],self.apex[1],self.apex[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p2[0],self.p2[1],self.p2[2])
        f.write(s)
        f.close()