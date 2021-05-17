import numpy as np
from numpy.linalg import inv
#here only consider the angle distortion, the regular pyramid will have a top_angle of 109.5 dg, here you can change the top_angle in the range
# from 0 to 120 dg, but the base is a equilateral triangle
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
class trigonal_pyramid_distortion_shareface():
    def __init__(self,p0=[0.,2.,0.],p1=[2.,0.,0.],p2=[0,0,2.],top_angle=1.0):
        #top angle is p0_A_p1 in ppt file, shoulder_angle is A_P0_CP
        #len_offset[0] is CP_P1 in ppt, the other one not specified in the file
        self.top_angle=top_angle
        self.p0,self.p1,self.p2=np.array(p0),np.array(p1),np.array(p2)

    def reset_attributes(self, p0, p1, p2, top_angle):
        self.p0, self.p1, self.p2, self.top_angle = p0, p1, p2, top_angle 
    
    def cal_apex_coor(self,mirror=True):
    #basis idea: set a new coordinate frame with p0p1 as the z vector (start from p1)
    #set a arbitrary y vector on the normal plane, and cross product to solve the x vector
    #then use phi and theta (sharp angle) to sove the cross_point(CP on file) and apex (A on file)
        
        origin=(self.p0+self.p1)/2.
        z_v=f3(np.zeros(3),np.cross(self.p1-origin,self.p2-origin))
        x_v=f3(np.zeros(3),self.p2-origin)
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        r=f2(self.p0,self.p1)/2./np.tan(self.top_angle/2.)
        phi=0.
        if mirror==False:       
            theta=np.arcsin(f2(self.p0,self.p1)/2.*np.tan(np.pi/6)/r)
        elif mirror==True:
            theta=np.pi-np.arcsin(f2(self.p0,self.p1)/2.*np.tan(np.pi/6)/r)
        apex_new = np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
        self.apex = np.dot(inv(T),apex_new)+origin
    
    @staticmethod
    def cal_coor_o3(p0,p1,p3):
        r=f2(p0,p1)/2.*np.tan(np.pi/3.)
        norm_vt=(p0-p1)/2.
        cent_pt=(p0+p1)/2.
        a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
        d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
        u,v,w=p3[0],p3[1],p3[2]
        k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
        O3_proj=np.array([u-a*k,v-b*k,w-c*k])
        cent_proj_vt=O3_proj-cent_pt
        l=f2(O3_proj,cent_pt)
        ptOnCircle_cent_vt=cent_proj_vt/l*r
        ptOnCircle=ptOnCircle_cent_vt+cent_pt
        return ptOnCircle
        
    def print_file(self):
        f=open('/home/jackey/apps/genx/pyramid.xyz','w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.apex[0],self.apex[1],self.apex[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p2[0],self.p2[1],self.p2[2])
        f.write(s)
        f.close()