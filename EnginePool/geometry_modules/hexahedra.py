import numpy as np
from numpy.linalg import inv
import os

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

basis=np.array([5.038,5.434,7.3707])
#atoms to be checked for distance
atms_cell_half=[[0.653,1.1121,1.903],[0.847,0.6121,1.903],[0.306,0.744,1.75],[0.194,0.243,1.75],\
      [0.5,1.019,1.645],[0,0.518,1.645],[0.847,0.876,1.597],[0.653,0.375,1.597]]
atms_cell_full=[[0.153,0.9452,2.097],[0.347,0.4452,2.097],[0.653,1.1121,1.903],[0.847,0.6121,1.903],[0.,0.9691,1.855],[0.5,0.4691,1.855],[0.306,0.744,1.75],[0.194,0.243,1.75],\
      [0.5,1.019,1.645],[0,0.518,1.645],[0.847,0.876,1.597],[0.653,0.375,1.597]]
atms_cell=atms_cell_half
atms=np.append(np.array(atms_cell),np.array(atms_cell)+[-1,0,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[1,0,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[0,-1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[0,1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[1,1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[-1,-1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[1,-1,0],axis=0)
atms=np.append(atms,np.array(atms_cell)+[-1,1,0],axis=0)

atms=atms*basis
O1,O2=[0.653,1.1121,1.903]*basis,[0.847,0.6121,1.903]*basis
O3,O4=[0.306,0.744,1.75]*basis,[0.194,0.243,1.75]*basis
O11_top,O12_top=[0.153,0.9452,2.097]*basis,[0.347,0.4452,2.097]*basis
anchor1,anchor2=O1,O2

class share_face():
    def __init__(self,face=np.array([[0.,0.,0.],[0.5,0.5,0.5],[1.0,1.0,1.0]])):
        #pass in the vector of three known vertices
        self.face=face
        
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
        elif flag=='1_2':
            def _cal_center(p1,p2,p0):
                #p0 is of type up-down 
                origin=(p1+p2)/2
                y_v=f3(np.zeros(3),p1-origin)
                x_v=f3(np.zeros(3),p0-origin)
                z_v=np.cross(x_v,y_v)
                T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
                r=f2(p1,p2)/2*np.tan(np.pi/6)
                phi=0.
                theta=np.pi/2+np.arctan(2)
                center_point_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
                center_point_org=np.dot(inv(T),center_point_new)+origin
                if abs(f2(center_point_org,p0)-f2(center_point_org,p1))>0.00001:
                    center_point_org=2*origin-center_point_org
                return center_point_org
            center_point=0
            if index==0:center_point=_cal_center(p0,p1,p2)
            elif index==1:center_point=_cal_center(p1,p2,p0)
            elif index==2:center_point=_cal_center(p0,p2,p1)
            else:         center_point=_cal_center(p0,p1,p2)          
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
                #p0 and p1 correspond to middlle layer type, p2 to up_down type
                #and p3 is middle layer type while p4 is up and down type
                z1_v=f3(np.zeros(3),p2-center_point)
                x1_v=f3(np.zeros(3),p1-center_point)
                y1_v=np.cross(z1_v,x1_v)
                T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
                r=f2(center_point,p0)
                p3_new=np.array([r*np.cos(4*np.pi/3)*np.sin(np.pi/2),r*np.sin(4*np.pi/3)*np.sin(np.pi/2),0])
                #check p3_new, since it is also possible p3_new=np.array([r*np.cos(2*np.pi/3)*np.sin(np.pi/2),r*np.sin(2*np.pi/3)*np.sin(np.pi/2),0])
                if (p3_new[0]-p0[0])<0.001:
                    p3_new=np.array([r*np.cos(2*np.pi/3)*np.sin(np.pi/2),r*np.sin(2*np.pi/3)*np.sin(np.pi/2),0])
                p4_new=np.array([0,0,-r])
                p3_org=np.dot(inv(T),p3_new)+center_point
                p4_org=np.dot(inv(T),p4_new)+center_point
                return T,r,p3_org,p4_org
            if index==0:
                self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p0,p1,p2)
            elif index==1:
                self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p1,p2,p0)
            elif index==2:
                self.T,self.r,self.p3,self.p4=_calculate_points(center_point,p0,p2,p1)
        elif flag=='0_3':
            z1_v0=np.cross(p0-center_point,p1-center_point)
            #here the p1 is a tricky part, since we set up a coordinate system centering at center_point
            #if use center_point as the p1, then p1 and p2 are based on two different coordinate system
            z1_v=f3(np.zeros(3),z1_v0)
            x1_v0=p1-center_point
            x1_v=f3(np.zeros(3),x1_v0)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=f2(center_point,p0)
            p3_new=np.array([0,0,r])
            p4_new=np.array([0,0,-r])
            p3_org=np.dot(inv(T),p3_new)+center_point
            p4_org=np.dot(inv(T),p4_new)+center_point
            self.T,self.r,self.p3,self.p4=T,r,p3_org,p4_org
    
    def print_file(self):
        f=open('D://hexahedra_test.xyz','w')
        f.write('6\n#\n')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Pb', self.center_point[0],self.center_point[1],self.center_point[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[0,0],self.face[0,1],self.face[0,2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[1,0],self.face[1,1],self.face[1,2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[2,0],self.face[2,1],self.face[2,2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p3[0],self.p3[1],self.p3[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e' % ('O', self.p4[0],self.p4[1],self.p4[2])
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
    def __init__(self,edge=np.array([[0.,0.,0.],[2.5,2.5,2.5]])):
        self.edge=edge
        
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
            #print x0_v,y0_v,z0_v,x1_v_2,y1_v_2,z1_v_2,T1
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
            #note in this case, phi can be either pi/2 or 3pi/2, theta can be any value in the range of [0,pi]
            r=dist/2*np.sqrt(3.)
            x_p2=r*np.cos(phi)*np.sin(theta)
            y_p2=r*np.sin(phi)*np.sin(theta)
            z_p2=r*np.cos(theta)
            p2_new=np.array([x_p2,y_p2,z_p2])
            p2_old=np.dot(inv(T),p2_new)+origin
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='0_3'
    ##thetha and phi range list
    #flag='0_2+0_1'(with two atom at middle layer known, cal the one with up/down type), phi=pi/2 or 3pi/3, theta=[0,pi]
    #flag='2_0+0_1', phi=[0,2pi], theta (not needed)
    #flag='1_1+0_1', phi=pi/2 or 3pi/2, theta=[0,pi]
    def all_in_all(self,theta=np.pi/2,phi=np.pi/2,ref_p=None,flag='0_2+0_1',extend_flag='type1'):
        self.cal_p2(theta,phi,ref_p,flag,extend_flag)
        self.share_face_init(self.flag)

#steric_check will check the steric feasibility by changing the theta angle (0-pi) and or phi [0,2pi]
#the dist bw sorbate(both metal and oxygen) and atms (defined on top) will be cal and compared to the cutting_limit
#higher cutting limit will result in fewer items in return file (so be wise to choose cutting limit)
#the container has 12 items, ie phi (rotation angle), theta, low_dis, apex coors (x,y,z), os1 coors(x,y,z),os2 coors(x,y,z)
#in which the low_dis is the lowest dist between sorbate and atm 
class steric_check(share_edge):
    def __init__(self,p0=anchor1,p1=anchor2,cutting_limit=2.):
        self.edge=np.array([p0,p1])
        self.cutting_limit=cutting_limit
        self.container=np.zeros((1,12))[0:0]
        print("distance between anchor points is ",f2(p0,p1),'anstrom')
    def steric_check(self,theta_res=0.1,phi=np.pi/2,flag='1_1+0_1',extend_flag='type1',mirror=False,print_path=None):
        #consider the steric constrain, flag '1_1+0_1' (one up_down type and one middel layer atm for the anchor point)
        #is more favorable, 0_2 or 2_0 has distance too much high that it is not easy to be fit into the rcut hematite case
        #set extend_flag to 'type1' if you want the anchor1 to be up_down atom type, and 'type2' if otherwise
        #mirror is a switch to choose the ligand postion between the possible two middle layer atom sites
        for theta in np.arange(0,np.pi,theta_res):
            self.all_in_all(theta=theta,phi=phi,ref_p=None,flag=flag,extend_flag=extend_flag)
            low_limit=self.cutting_limit*2
            target_atms=[]
            for atm in atms:
                if ((abs(sum(atm-self.edge[0,:]))>0.01)&(abs(sum(atm-self.edge[1,:]))>0.01)):
                    dt_center_atm,dt_p2_atm,dt_p3_atm,dt_p4_atm=f2(atm,self.center_point),f2(atm,self.face[2,:]),f2(atm,self.p3),f2(atm,self.p4)
                    com_atms=[]
                    if flag=='0_2+0_1':
                        com_atms=np.array([dt_center_atm,dt_p3_atm,dt_p4_atm])
                        target_atms=[self.center_point,self.p3,self.p4]
                    elif flag=='2_0+0_1':
                        com_atms=np.array([dt_center_atm,dt_p3_atm,dt_p4_atm])
                        target_atms=[self.center_point,self.p3,self.p4]
                    elif flag=='1_1+0_1':
                        #know here the p2 and p3 are middle layer type and p4 is up_down type
                        if mirror:
                            com_atms=np.array([dt_center_atm,dt_p2_atm,dt_p4_atm])
                            target_atms=[self.center_point,self.p2,self.p4]
                        else:
                            com_atms=np.array([dt_center_atm,dt_p3_atm,dt_p4_atm])
                            target_atms=[self.center_point,self.p3,self.p4]
                    if np.sum(com_atms<self.cutting_limit)!=0:
                        low_limit=None
                        break
                    else:
                        if np.min(com_atms)<low_limit:
                            low_limit=np.min(com_atms)
                        else:pass
            if low_limit!=None:
                C,P1,P2=target_atms[0],target_atms[1],target_atms[2]
                self.container=np.append(self.container,[[phi,theta,low_limit,C[0],C[1],C[2],P1[0],P1[1],P1[2],P2[0],P2[1],P2[2]]],axis=0)
            else:pass
        #note here consider the first slab, y and z shiftment has been made properly
        data=np.rec.fromarrays([self.container[:,0],self.container[:,1],\
                                self.container[:,2],self.container[:,3],\
                                self.container[:,4]-0.7558,self.container[:,5]-7.3707,\
                                self.container[:,6],self.container[:,7]-0.7558,\
                                self.container[:,8]-7.3707,
                                self.container[:,9],self.container[:,10]-0.7558,\
                                self.container[:,11]-7.3707,],names='phi,theta,low,A1,A2,A3,P10,P11,P12,P20,P21,P22')
        data.sort(order=('low','phi','theta'))
        print("phi,theta,low_dt_limit,A_x,A_y,A_z,P1_x,P1_y,P1_z,P2_x,P2_y,P2_z,P3_x,P3_y,P3_z")
        if print_path!=None:
            np.savetxt(print_path,data,"%5.3f")
            for i in np.loadtxt(print_path):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]\n")
        else:
            np.savetxt('test',data,"%5.3f")
            for i in np.loadtxt('test'):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]\n")
            os.remove('test')
            
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

#this share_cornor will place the hexahedra right over the surface(the other anchor will have same x y as the corner, but the z value will be added by twice r)
#and the center point will be sitted at the middle point of the two anchors 
class share_corner2(share_edge):
    #if want to share none, then just set the corner coordinate to the first point set arbitratly.
    def __init__(self,corner=anchor1,r=3,phi=1.57):
        self.corner=corner
        self.p1=self.corner+[0,0,r*2]
        self.edge=np.append(self.corner[np.newaxis,:],self.p1[np.newaxis,:],axis=0)
        self.all_in_all(phi=phi,ref_p=None,flag='2_0+0_1')
#this steric check is specifically for the share_corner2        
class steric_check2(share_corner2):
    def __init__(self,p0=anchor1,r=2.0,cutting_limit=2.5):
        self.corner=p0
        self.p1=self.corner+[0,0,r*2]
        self.edge=np.append(self.corner[np.newaxis,:],self.p1[np.newaxis,:],axis=0)
        self.cutting_limit=cutting_limit
        self.container=np.zeros((1,15))[0:0]

    def steric_check(self,phi_res=0.3,flag='2_0+0_1',print_path=None):
        #consider the steric constrain, flag 'off_center' (the center point is off the connection line of anchors)
        #is more favorable

        for phi in np.arange(0,np.pi*2,phi_res):
            self.all_in_all(phi=phi,ref_p=None,flag=flag)
            low_limit=self.cutting_limit*2
            for atm in atms:
                if (abs(sum(atm-self.corner))>0.01):
                    #p4 was not count, it was arbiturily taken away for the lone electron pair of lead
                    dt_center_atm,dt_p1_atm,dt_p2_atm,dt_p3_atm=f2(atm,self.center_point),f2(atm,self.p1),f2(atm,self.p2),f2(atm,self.p3)
                    com_atms=np.array([dt_center_atm,dt_p1_atm,dt_p2_atm,dt_p3_atm])
                    if np.sum(com_atms<self.cutting_limit)!=0:
                        low_limit=None
                        break
                    else:
                        if np.min(com_atms)<low_limit:
                            low_limit=np.min(com_atms)
                        else:pass
            if low_limit!=None:
                C,P1,P2,P3=self.center_point,self.p1,self.p2,self.p3
                self.container=np.append(self.container,[[phi,0,low_limit,C[0],C[1],C[2],P1[0],P1[1],P1[2],P2[0],P2[1],P2[2],P3[0],P3[1],P3[2]]],axis=0)
            else:pass
        #note here consider the first slab, y and z shiftment has been made properly
        data=np.rec.fromarrays([self.container[:,0],self.container[:,1],\
                                self.container[:,2],self.container[:,3],\
                                self.container[:,4]-0.7558,self.container[:,5]-7.3707,\
                                self.container[:,6],self.container[:,7]-0.7558,\
                                self.container[:,8]-7.3707,
                                self.container[:,9],self.container[:,10]-0.7558,\
                                self.container[:,11]-7.3707,
                                self.container[:,12],self.container[:,13]-0.7558,\
                                self.container[:,14]-7.3707],names='phi,theta,low,A1,A2,A3,P10,P11,P12,P20,P21,P22,P30,P31,P32')
        data.sort(order=('low','phi'))
        print("phi,0,low_dt_limit,A_x,A_y,A_z,P1_x,P1_y,P1_z,P2_x,P2_y,P2_z,P3_x,P3_y,P3_z")
        if print_path!=None:
            np.savetxt(print_path,data,"%5.3f")
            for i in np.loadtxt(print_path):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]\n")
        else:
            np.savetxt('test',data,"%5.3f")
            for i in np.loadtxt('test'):
                print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]\n")
            os.remove('test')
            
if __name__=='__main__':
    test1=hexahedra_4.share_edge(np.array([[0.,0.,0],[4.,2.,-1]]))
    test1.cal_p2(theta=0,phi=np.pi/2,flag='1_1+0_1',extend_flag='type1')
    test1.share_face_init(flag=test1.flag)
    print(test1.face,test1.p3,test1.p4,test1.center_point)