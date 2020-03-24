import numpy as np
from numpy.linalg import inv

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
#when read the comments, open the PPT file associated with it (algorithm of distortion oin polyhedra.ppt)
class tetrahedra_edge_distortion ():

    def __init__(self,p0=[0.,0.,0.],p1=[1.,1.,1.],offset=[0.,0.,0.]):
    #p0 and p1 are two known points to start with in frational coor, offset are the edge offset for three corners 
    #in tetrahedra configuration with unit of angstrom, note one of two known points is the reference point with zero offset, which
    #is not specified in the list of offset
    #the top_angle is a fixed angle with value of 109.7o, shoulder angle is (pi-top)/2
        self.p0=np.array(p0)
        self.p1=np.array(p1)
        self.offset=offset
        self.shoulder_angle=0.61548
        self.top_angle=1.91063
        
    def reset_pars(self,p0,p1,offset):
        self.p0=np.array(p0)
        self.p1=np.array(p1)
        self.offset=offset
        self.cal_theta()
        
    def cal_theta(self):
    #here theta angle is angle p1p2p3 in ppt file
        dst_p0_p1=f2(self.p0,self.p1)
        right_l=self.offset[0]*np.sin(self.shoulder_angle)
        self.theta=np.arcsin(right_l/dst_p0_p1)+self.shoulder_angle
        return self.theta
        
    def cal_edge_len(self):
    #cal the edge length of regular hexahedra
    #sharp angle is angle p2p3p1 in ppt file
    #rigth_side is the length of p2p5 in ppt file
        sharp_angle=np.pi-self.top_angle-self.theta
        right_side=f2(self.p0,self.p1)*np.sin(sharp_angle)
        self.edge_len=right_side/np.sin(np.pi-self.top_angle)
        
    def cal_body_center (self,switch=False,phi=0.):
    #try to calculate p1 in ppt file
    #a spherical coordinate system is set up with p3p2(start from p2 in ppt) as z axis
    #then set a arbitrary vector as y axis, then use cross product to calculate the other x axis
    #in which we use the eqution of a plane with known normal vector through one known point
    #a(x-x0)+b(y-y0)+c(z-z0)=0, normal vector [a,b,c], known point [x0,y0,z0]
    #we specify phi and the thetha angle is dertermined using function above

        p0,p1=self.p0,self.p1
        if switch==True:
            p0=self.p1
            p1=self.p0
        n_v=p1-p0
        a,b,c=n_v[0],n_v[1],n_v[2]
        x0,y0,z0=p0[0],p0[1],p0[2]
        ref_p=0
        if c!=0.:
            ref_p=np.array([1.,1.,(a*(x0-1.)+b*(y0-1.))/c+z0])
        elif b!=0.:
            ref_p=np.array([1.,(a*(x0-1.)+c*(z0-1.))/b+y0,1.])
        else:
            ref_p=np.array([(b*(y0-1.)+c*(z0-1.))/a+x0,1.,1.])
        origin=p0
        #print np.dot((p1-p0),(ref_p-p0))
        y_v=f3(np.zeros(3),(ref_p-origin))
        z_v=f3(np.zeros(3),(p1-origin))
        x_v=np.cross(z_v,y_v)
        #print f2(np.zeros(3),x_v),f2(np.zeros(3),y_v),f2(np.zeros(3),z_v)
        #print y_v
        #print z_v
        #print x_v
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        print T
        #print T
        r=self.edge_len
        theta=self.theta
        body_center_new = np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
        self.body_center = np.dot(inv(T),body_center_new)+origin
        
    def cal_corner (self):
    #we know the body center, and two the other points with one showing offset
    #we set a spherical coordinate system with p2p1 as the z axis (see ppt file)
    #all the other corners should have the equal theta angle in the coordinate system, and the phi angle
    #increase by 109.7 dc, so the basis idea here is to figure out the alpha value of the know point
    #and add the angle offset to get the other two phi angles
    #note we use the offset of each side length here
        #ft calculate the angle of C 
        ft=lambda a,b,c:np.arccos((a**2+b**2-c**2)/(2*a*b))
        def cal_point_on_plane(normal,point_n,point_p):
        #a plane is defined by normal to the normal vector and through ponit_n
        #a line is parallel to the normal and throught point_p
        #return the cross point of the line through the plance
            x0,y0,z0=point_n[0],point_n[1],point_n[2]
            x1,y1,z1=point_p[0],point_p[1],point_p[2]
            a,b,c=normal[0],normal[1],normal[2]
            t=(a*(x0-x1)+b*(y0-y1)+c*(z0-z1))/(a**2+b**2+c**2)
            return np.array([x1+t*a,y1+t*b,z1+t*c])
        p0,p1=self.body_center,self.p0
        n_v=p1-p0
        a,b,c=n_v[0],n_v[1],n_v[2]
        x0,y0,z0=p0[0],p0[1],p0[2]
        ref_p=0
        try:
            ref_p=np.array([1.,1.,(a*(x0-1.)+b*(y0-1.))/c+z0])
        except:
            try:
                ref_p=np.array([1.,(a*(x0-1.)+c*(z0-1.))/b+y0,1.])
            except:
                ref_p=np.array([(b*(y0-1.)+c*(z0-1.))/a+x0,1.,1.])
        origin=p0
        y_v=f3(np.zeros(3),(ref_p-origin))
        z_v=f3(np.zeros(3),(p1-origin))
        x_v=np.cross(z_v,y_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v) 
        #print inv(T)
        r=self.edge_len
        theta = self.top_angle
        phi1=0
        #the idea to find phi1 is project the point onto xy plane, calculate the angle of the projected line 
        #with x axis and y axis, y axis is a reference to determine the real angle
        project_p1=cal_point_on_plane(normal=z_v,point_n=origin,point_p=self.p1)
        x_pt=origin+x_v
        y_pt=origin+y_v
        ox_len=f2(origin,x_pt)
        oy_len=f2(origin,y_pt)
        opp_len=f2(origin,project_p1)
        xpp_len=f2(project_p1,x_pt)
        ypp_len=f2(project_p1,y_pt)
        angle_x_o_pp=ft(ox_len,opp_len,xpp_len)
        angle_y_o_pp=ft(oy_len,opp_len,ypp_len)
        if angle_y_o_pp<=np.pi/2:
            phi1=angle_x_o_pp
        else:
            phi1=np.pi*2-angle_x_o_pp
        phi2 = phi1+np.pi*2./3.
        phi3 = phi2+np.pi*2./3.
        p2_new = np.array([(r+self.offset[1])*np.cos(phi2)*np.sin(theta),(r+self.offset[1])*np.sin(phi2)*np.sin(theta),(r+self.offset[1])*np.cos(theta)])
        p3_new = np.array([(r+self.offset[2])*np.cos(phi3)*np.sin(theta),(r+self.offset[2])*np.sin(phi3)*np.sin(theta),(r+self.offset[2])*np.cos(theta)])
        #print p2_new
        self.p2 = np.dot(inv(T),p2_new)+origin
        self.p3 = np.dot(inv(T),p3_new)+origin
    def all_in_all(self,switch=False,phi=0.):
        self.cal_theta()
        self.cal_edge_len()
        self.cal_body_center(switch=switch, phi=phi)
        self.cal_corner()
    def print_file(self):
        f=open('Y:\\codes\\my code\\modeling files\\surface modeling 1\\scripts\\tetrahedra_test.xyz','w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.body_center[0],self.body_center[1],self.body_center[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p0[0],self.p0[1],self.p0[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p1[0],self.p1[1],self.p1[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p2[0],self.p2[1],self.p2[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p3[0],self.p3[1],self.p3[2])
        f.write(s)
        f.close()