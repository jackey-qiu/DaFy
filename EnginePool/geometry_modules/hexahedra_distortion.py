import numpy as np
from numpy.linalg import inv

#note: this module is only used to consider the situation with flag of '0_2+0_1' in share_edge and '0_3' '1_2' in share_face
#in 1_2 situation,make sure theta_top_down in range of [0,pi/2], and open angle must be higher than the top angle in the triangular defined by p0,p1 and p2
#and p0 is of the up_down type 
#use switch to find the center_point reasonably (if true center_point above,if false center_point below)
#ignore the other situations unless they are modified and testified to be workable.
#the open angel in the regular hexahedra is 120 dg, while here you can specify it. What can also be specified is the r_top_down, is the distance
#between center point and one of the apex (up and down type point)
#the theta_top_down is 90 dg in the regular hexahedra, here it can be any value from 0 to pi/2. and theta_top_down is the angle between the regular
#z axis and the current cent_top_vect (vector point from center point to top point)
#and also note the projection of cent_top_vect on the midder plane rightly bisect the open_angle
#p0 is the top point in the middle triangular plane with p1 and p2 on the other two apex (so p0p1=p0p2), so when you start from share_face not share_edge
#make sure that condition satisfied (be careful about the order of elements in the face, first one must correspond to p0 point)
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

class share_face():
    def __init__(self,face=np.array([[0.,0.,0.],[0.5,0.5,0.5],[1.0,1.0,1.0]]),open_angle=np.pi*2/3,r_top_down=None,theta_top_down=0.,switch=True):
        #pass in the vector of three known vertices
        #the center point will be below the face plane if switch=False, and it will above the surface plane if switch=True 
        self.face=face
        self.open_angle=open_angle
        self.r_top_down=r_top_down
        self.theta_top_down=theta_top_down
        self.switch=switch
        
    def share_face_init(self,flag='0_3'):
        #hexahedra has five vertices, there are two more besides the known three. there are two types of vertices, one type consisting of two featuring up and down
        #the other type consisting of three vertices spreading at the middle layer, there are multipy choices for the 
        #combination of three known,flag '0_3'means 0 vertices of the first type, 3 vertices of second type, and so on for other flags
        p0,p1,p2=self.face[0,:],self.face[1,:],self.face[2,:]
        center_point=np.array([])
        #consider the possible unregular shape for the known triangle
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list)) 
        if flag=='2_1':
            #'2_1'tag means 2 atoms at upside and downside, the other one at middle layer
            if index==0:center_point=(p0+p1)/2
            elif index==1:center_point=(p1+p2)/2
            elif index==2:center_point=(p0+p2)/2
            else:center_point=(p0+p2)/2
        elif flag=='0_3':
            #here we try to solve the coordinate of the center point, we can easily find two equations based on the fact that
            #the distance from the center point to the three known vertices should be the same, the other one based on the fact
            #that the center point is on the plane defined by the three known points
            normal=np.cross(p1-p0,p2-p0)
            c3=np.sum(normal*p0)
            A=np.array([2*(p1-p0),2*(p2-p0),normal])
            C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
            center_point=np.dot(inv(A),C)
            if self.open_angle==None:
                self.open_angle=2*np.arcsin(f2(p1,p2)/2/f2(p0,p1))
                
        elif flag=='1_2':
            def _cal_center(p0,p1,p2):
                #p0 is of type up-down 
                origin=(p1+p2)/2
                y_v=f3(np.zeros(3),p1-origin)
                x_v=f3(np.zeros(3),p0-origin)
                z_v=np.cross(x_v,y_v)
                T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
                #print T
                r=f2(p1,p2)/2*np.tan(np.pi/2-self.open_angle/2.)
                phi=0.
                L1=f2(p1,p2)/2./np.tan(self.open_angle/2.)
                L2=f2(p0,origin)
                #look at document#1 in binder for detail
                a,b,c=1+np.tan(self.theta_top_down)**2,2*L1/L2*np.tan(self.theta_top_down),(L1/L2)**2-1
                sin_list=[(b+(b**2-4*a*c)**0.5)/2./a,(b-(b**2-4*a*c)**0.5)/2./a]
                theta=0
                if (sin_list[0]<1)&(sin_list[0]>0):
                    if self.switch==False:
                        theta=np.pi/2+np.arcsin(sin_list[0])
                    elif self.switch==True:
                        theta=np.pi/2-np.arcsin(sin_list[0])
                else:
                    if self.switch==False:
                        theta=np.pi/2+np.arcsin(sin_list[1])
                    elif self.switch==True:
                        theta=np.pi/2-np.arcsin(sin_list[1])
                    
                center_point_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
                center_point_org=np.dot(inv(T),center_point_new)+origin
                return center_point_org
            center_point=_cal_center(p0,p1,p2)
         
        self.center_point=center_point
        self._find_the_other_two(center_point,p0,p1,p2,flag)
        
    def _find_the_other_two(self,center_point,p0,p1,p2,flag):
        #the basic idea to calculate the other two points is setting up a sperical coordinate frame centering at the center point
        #xy plane will be the same as the middle plane of the hexahedra, calculate the transformt matrix to relate the current sperical
        #coordinate frame to the original coordinate frame
        #find the coordinates of the other two points in the spherical coordinate frame, then convert back to the original frame using the 
        #transformt matrix
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list))
        if flag=='2_1':
            def _calculate_points(center_point,p0,p1,p2):
                #here the first two arguments (p0,p1) corresponding to vertices of up-down type, p2 to vertices of middle layer
                x1_v=f3(np.zeros(3),p2-center_point)
                z1_v=f3(np.zeros(3),p1-center_point)
                y1_v=np.cross(z1_v,x1_v)
                T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
                r=f2(center_point,p0)
                p3_new=np.array([r*np.cos(2*np.pi/3)*np.sin(np.pi/2),r*np.sin(2*np.pi/3)*np.sin(np.pi/2),0])
                p4_new=np.array([r*np.cos(4*np.pi/3)*np.sin(np.pi/2),r*np.sin(4*np.pi/3)*np.sin(np.pi/2),0])
                #centering at center_point,so don't forget plus vector of center_point
                p3_org=np.dot(inv(T),p3_new)+center_point
                p4_org=np.dot(inv(T),p4_new)+center_point
                return T,r,p3_org,p4_org
            if index==0:#means p0,p1 are of up-down type
                self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p0,p1,p2)
            elif index==1:#means p1,p2 are of up-down type
                self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p1,p2,p0)
            elif index==2:#means p0,p2 are of up-down type
                self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p0,p2,p1)
        elif flag=='1_2':
            def _calculate_points(center_point,p0,p1,p2):
                #p1 and p2 correspond to middlle layer type, p0 to up_down type
                z1_v=f3(np.zeros(3),np.cross(p2-center_point,p1-center_point))
                x1_v=f3(np.zeros(3),(p2+p1)/2-center_point)
                y1_v=np.cross(z1_v,x1_v)
                T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
                #print p0,center_point,p1,center_point
                r=f2(center_point,p2)
                #p3 is in the middle layer, p4 is of up-down type
                p3_new=np.array([r*np.cos(np.pi)*np.sin(np.pi/2),r*np.sin(np.pi)*np.sin(np.pi/2),0])
                r=f2(center_point,p0)
                p4_new=np.array([r*np.cos(0.)*np.sin(np.pi-self.theta_top_down),r*np.sin(0.)*np.sin(np.pi-self.theta_top_down),r*np.cos(np.pi-self.theta_top_down)])
                
                p3_org=np.dot(inv(T),p3_new)+center_point
                p4_org=np.dot(inv(T),p4_new)+center_point
                if f2(p0,p4_org)<0.1:
                    p4_new=np.array([r*np.cos(0.)*np.sin(self.theta_top_down),r*np.sin(0.)*np.sin(self.theta_top_down),r*np.cos(self.theta_top_down)])
                    p4_org=np.dot(inv(T),p4_new)+center_point
                return T,r,p3_org,p4_org

            self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p0,p1,p2)

        elif flag=='0_3':
            z1_v0=np.cross(p0-center_point,p1-center_point)
            #here the p1 is a tricky part, since we set up a coordinate system centering at center_point
            #if use center_point as the p1, then p1 and p2 are based on two different coordinate system
            z1_v=f3(np.zeros(3),z1_v0)
            x1_v0=p0-center_point
            x1_v=f3(np.zeros(3),x1_v0)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=0
            if self.r_top_down==None:
                r=f2(center_point,p0)
            else:r=self.r_top_down
            
            phi=np.pi
            if self.theta_top_down==None:
                self.theta_top_down=0.
            theta=self.theta_top_down
            
            p3_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
            p4_new=np.array([r*np.cos(phi)*np.sin(np.pi-theta),r*np.sin(phi)*np.sin(np.pi-theta),r*np.cos(np.pi-theta)])
            p3_org=np.dot(inv(T),p3_new)+center_point
            p4_org=np.dot(inv(T),p4_new)+center_point
            
            #if (f2(p3_org,p1)-f2(p3_org,p2))>0.001:
                #p3_new=np.array([r*np.cos(-phi)*np.sin(theta),r*np.sin(-phi)*np.sin(theta),r*np.cos(theta)])
                #p4_new=np.array([r*np.cos(-phi)*np.sin(np.pi-theta),r*np.sin(-phi)*np.sin(np.pi-theta),r*np.cos(np.pi-theta)])
                #p3_org=np.dot(inv(T),p3_new)+center_point
                #p4_org=np.dot(inv(T),p4_new)+center_point
            self.T,self.r,self.p3,self.p4=T,r,p3_org,p4_org
            
    def print_xyz(self):
        f=open('/home/jackey/apps/genx/hexahedra.xyz','w')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb',self.center_point[0],self.center_point[1],self.center_point[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O',self.face[0][0],self.face[0][1],self.face[0][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O',self.face[1][0],self.face[1][1],self.face[1][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O',self.face[2][0],self.face[2][1],self.face[2][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O',self.p3[0],self.p3[1],self.p3[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O',self.p4[0],self.p4[1],self.p4[2])
        f.write(s)
        f.close()
        
    def cal_point_in_fit(self,r,theta,phi):
        #during fitting,use the same coordinate system, but a different origin
        #note the origin_coor is the new position for the sorbate0, ie new center point
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        point_in_original_coor=np.dot(inv(self.T),np.array([x,y,z]))+self.center_point
        return point_in_original_coor

class share_edge(share_face):
    def __init__(self,edge=np.array([[0.,0.,0.],[0.5,0.5,0.5]]),open_angle=np.pi*2/3,r_top_down=None,theta_top_down=None):
        self.edge=edge
        self.open_angle=open_angle
        self.r_top_down=r_top_down
        self.theta_top_down=theta_top_down
        
    def cal_p2(self,theta=None,phi=None,ref_p=None,flag='0_2+0_1',extend_flag='type1'):
        p0=self.edge[0,:]
        p1=self.edge[1,:]
        origin=(p0+p1)/2
        dist=f2(p0,p1)
        diff=p1-p0
        c=np.sum(p1**2-p0**2)
        ref_point=0
        if ref_p!=None:
            ref_point=ref_p
        else:
            x,y,z=0.,0.,0.
            #set the reference point as simply as possible,using the same distance assumption, we end up with a plane equation
            #then we try to find one cross point between one of the three basis and the plane we just got
            #here combine two line equations (ref-->p0,and ref-->p1,the distance should be the same)
            if diff[0]!=0:
                x=c/(2*diff[0])
            elif diff[1]!=0.:
                y=c/(2*diff[1])
            elif diff[2]!=0.:
                z=c/(2*diff[2])
            ref_point=np.array([x,y,z])
            if sum(ref_point)==0:
                #if the vector (p0-->p1) pass through origin [0,0,0],we need to specify another point satisfying the same-distance condition
                #here, we a known point (x0,y0,z0)([0,0,0] in this case) and the normal vector to calculate the plane equation, 
                #which is a(x-x0)+b(y-y0)+c(z-z0)=0, we specify x y to 1 and 0, calculate z value.
                #a b c coresponds to vector origin-->p0
                ref_point=np.array([1.,0.,-p0[0]/p0[2]])
        if flag=='1_1+0_1':
            #we can have two choices symmetrically relating to each other,ie p0 can either be of up_down type or middl-layer type
            #extend_flag was used to distinguish these two case,'type1' means p0 be of up_down type
            #the idea here is based on setting up two spherical coordinate frame, one for calculating center point of polyhedra
            #then set the center point as origin to the other spherical coord frame, where the p2 was calculated
            #the resulting polyhedra will be on regular basis.
            #in this way, the orientation is available through changing the coordinate of center point through 2 fold rotation
            #set up first spherical coordinate system
            z1_v=f3(np.zeros(3),ref_point-origin)
            x1_v=f3(np.zeros(3),p1-origin)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=dist/2.
            #note in this case, phi can be either pi/2 or 3pi/2, theta can be any value in the range of [0,pi]
            x_center=r*np.cos(phi)*np.sin(theta)
            y_center=r*np.sin(phi)*np.sin(theta)
            z_center=r*np.cos(theta)
            center_org=np.dot(inv(T),np.array([x_center,y_center,z_center]))+origin
            #set up second spherical coordinate frame
            x1_v_2=np.array([])
            z1_v_2=np.array([])
            if extend_flag=='type1':
                x1_v_2=f3(np.zeros(3),p1-center_org)
                z1_v_2=f3(np.zeros(3),p0-center_org)
            elif extend_flag=='type2':
                x1_v_2=f3(np.zeros(3),p0-center_org)
                z1_v_2=f3(np.zeros(3),p1-center_org)
            y1_v_2=np.cross(z1_v_2,x1_v_2)
            T1=f1(x0_v,y0_v,z0_v,x1_v_2,y1_v_2,z1_v_2)
            r1=f2(center_org,p0)
            #calculate point p2
            x_p2=r1*np.cos(2*np.pi/3)*np.sin(np.pi/2)
            y_p2=r1*np.sin(2*np.pi/3)*np.sin(np.pi/2)
            z_p2=0.
            p2_new=np.array([x_p2,y_p2,z_p2])
            p2_old=np.dot(inv(T1),p2_new)+center_org
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='1_2'
        elif flag=='2_0+0_1':
            #in this case, p2 can be calculated directely
            x1_v=f3(np.zeros(3),ref_point-origin)
            z1_v=f3(np.zeros(3),p1-origin)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=dist/2
            #here phi in the range of [0,2pi]
            x_p2=r*np.cos(phi)*np.sin(np.pi/2)
            y_p2=r*np.sin(phi)*np.sin(np.pi/2)
            z_p2=0
            p2_new=np.array([x_p2,y_p2,z_p2])
            p2_old=np.dot(inv(T),p2_new)+origin
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='2_1'
        elif flag=='0_2+0_1':
            #in this case, p2 can also be calculated directely
            x1_v=f3(np.zeros(3),ref_point-origin)
            y1_v=f3(np.zeros(3),p1-origin)
            z1_v=np.cross(x1_v,y1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            #note the r is different from that in the case above
            #note in this case, phi can be either 0 or pi(checked), theta can be any value in the range of [0,pi]
            r=dist/2*(1/np.tan(self.open_angle/4))
            x_p2=r*np.cos(phi)*np.sin(theta)
            y_p2=r*np.sin(phi)*np.sin(theta)
            z_p2=r*np.cos(theta)
            p2_new=np.array([x_p2,y_p2,z_p2])
            p2_old=np.dot(inv(T),p2_new)+origin
            self.p2=p2_old
            self.face=np.append([p2_old],self.edge,axis=0)
            self.flag='0_3'
        
class share_corner(share_edge):
    #if want to share none, then just set the corner coordinate to the first point set arbitratly.
    def __init__(self,corner=np.array([0.,0.,0.])):
        self.corner=corner
        
    def cal_p1(self,r,theta,phi):
        #here we simply use the original coordinate system converted to spherical coordinate system, but at different origin
        x_p1=r*np.cos(phi)*np.sin(theta)+self.corner[0]
        y_p1=r*np.sin(phi)*np.sin(theta)+self.corner[1]
        z_p1=r*np.cos(theta)+self.corner[2]
        p1=np.array([x_p1,y_p1,z_p1])
        self.p1=p1
        self.edge=np.append(self.corner[np.newaxis,:],p1[np.newaxis,:],axis=0)
        
if __name__=='__main__':
    test1=hexahedra_4.share_edge(np.array([[0.,0.,0],[4.,2.,-1]]))
    test1.cal_p2(theta=0,phi=np.pi/2,flag='1_1+0_1',extend_flag='type1')
    test1.share_face_init(flag=test1.flag)
    print(test1.face,test1.p3,test1.p4,test1.center_point)