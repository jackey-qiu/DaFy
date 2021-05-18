import numpy as np
from numpy.linalg import inv
import os

#see detail comments in hexahedra_4

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
    def __init__(self,face=np.array([[0.,0.,2.5],[2.5,0,0.],[0,2.5,0]]),mirror=False):
        #pass in the vector of three known vertices
        #mirror setting will make the sorbate projecting in an opposite direction referenced to the p0p1p2 plane
        self.face=face
        self.mirror=mirror

    def share_face_init(self,flag='right_triangle',dr=[0,0,0]):
        #octahedra has a high symmetrical configuration,there are only two types of share face.
        #flag 'right_triangle' means the shared face is defined by a right triangle with two equal lateral and the other one
        #passing through body center;'regular_triangle' means the shared face is defined by a regular triangle
        #dr is used for fitting purpose, set this to be 0 to get a regular octahedral
        p0,p1,p2=self.face[0,:],self.face[1,:],self.face[2,:]
        #consider the possible unregular shape for the known triangle
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list))

        if flag=='right_triangle':
        #'2_1'tag means 2 atoms at upside and downside, the other one at middle layer
            if index==0:self.center_point=(p0+p1)/2
            elif index==1:self.center_point=(p1+p2)/2
            elif index==2:self.center_point=(p0+p2)/2
            else:self.center_point=(p0+p2)/2
        elif flag=='regular_triangle':
            #the basic idea is building a sperical coordinate system centering at the middle point of each two of the three corner
            #and then calculate the center point through theta angle, which can be easily calculated under that geometrical seting
            def _cal_center(p1,p2,p0):
                origin=(p1+p2)/2
                y_v=f3(np.zeros(3),p1-origin)
                x_v=f3(np.zeros(3),p0-origin)
                z_v=np.cross(x_v,y_v)
                T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
                r=f2(p1,p2)/2.
                phi=0.
                theta=np.pi/2+np.arctan(np.sqrt(2))
                if self.mirror:
                    theta=np.pi/2-np.arctan(np.sqrt(2))
                center_point_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
                center_point_org=np.dot(inv(T),center_point_new)+origin
                #the two possible points are related to each other via invertion over the origin
                if abs(f2(center_point_org,p0)-f2(center_point_org,p1))>0.00001:
                    center_point_org=2*origin-center_point_org
                return center_point_org
            self.center_point=_cal_center(p0,p1,p2)
        self._find_the_other_three(self.center_point,p0,p1,p2,flag,dr)

    def _find_the_other_three(self,center_point,p0,p1,p2,flag,dr):
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list))

        if flag=='right_triangle':
            def _cal_points(center_point,p0,p1,p2,dr=[0,0,0]):
                #here p0-->p1 is the long lateral
                z_v=f3(np.zeros(3),p2-center_point)
                x_v=f3(np.zeros(3),p0-center_point)
                y_v=np.cross(z_v,x_v)
                T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
                r=f2(center_point,p0)
                #print [r*np.cos(np.pi/2)*np.sin(np.pi/2),r*np.sin(np.pi/2)*np.sin(np.pi/2),0]
                p3_new=np.array([(r+dr[0])*np.cos(np.pi/2)*np.sin(np.pi/2),(r+dr[0])*np.sin(np.pi/2)*np.sin(np.pi/2),0])
                p4_new=np.array([(r+dr[1])*np.cos(3*np.pi/2)*np.sin(np.pi/2),(r+dr[1])*np.sin(3*np.pi/2)*np.sin(np.pi/2),0])
                p3_old=np.dot(inv(T),p3_new)+center_point
                p4_old=np.dot(inv(T),p4_new)+center_point
                p5_old=2*center_point-p2
                p5_old = (p5_old-center_point)*((r+dr[2])/r)+center_point
                return T,r,p3_old,p4_old,p5_old
            if index==0:#p0-->p1 long lateral
                self.T,self.r,self.p3,self.p4,self.p5=_cal_points(center_point,p0,p1,p2,dr)
            elif index==1:#p1-->p2 long lateral
                self.T,self.r,self.p3,self.p4,self.p5=_cal_points(center_point,p1,p2,p0,dr)
            elif index==2:#p0-->p2 long lateral
                self.T,self.r,self.p3,self.p4,self.p5=_cal_points(center_point,p0,p2,p1,dr)
        elif flag=='regular_triangle':
            x_v=f3(np.zeros(3),p2-center_point)
            y_v=f3(np.zeros(3),p0-center_point)
            z_v=np.cross(x_v,x_v)
            self.T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
            self.r=f2(center_point,p0)
            self.p3=(center_point-p0)*((self.r+dr[0])/self.r)+center_point
            self.p4=(center_point-p1)*((self.r+dr[1])/self.r)+center_point
            self.p5=(center_point-p2)*((self.r+dr[2])/self.r)+center_point
            #print f2(self.center_point,self.p3),f2(self.center_point,self.p4)

    def cal_point_in_fit(self,r,theta,phi):
        #during fitting,use the same coordinate system, but a different origin
        #note the origin_coor is the new position for the sorbate0, ie new center point
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        point_in_original_coor=np.dot(inv(self.T),np.array([x,y,z]))+self.center_point
        return point_in_original_coor

    def print_xyz(self,file="D:\\test.xyz"):
        f=open(file,"w")
        f.write('7\n#\n')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('Sb', self.center_point[0],self.center_point[1],self.center_point[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[0,:][0],self.face[0,:][1],self.face[0,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[1,:][0],self.face[1,:][1],self.face[1,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.face[2,:][0],self.face[2,:][1],self.face[2,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p3[0],self.p3[1],self.p3[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.p4[0],self.p4[1],self.p4[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e' % ('O', self.p5[0],self.p5[1],self.p5[2])
        f.write(s)
        f.close()

class share_edge(share_face):
    def __init__(self,edge=np.array([[0.,0.,0.],[5,5,5]])):
        self.edge=edge

    def cal_p2(self,ref_p=np.array([None]),phi=np.pi/2,flag='off_center',**args):
        p0=self.edge[0,:]
        p1=self.edge[1,:]
        origin=(p0+p1)/2
        dist=f2(p0,p1)
        diff=p1-p0
        c=np.sum(p1**2-p0**2)
        ref_point=0
        #if ref_p==[]:
        #ref_p=[None]
        ref_p=np.array(ref_p)
        if ref_p.any()!=None:
            ref_point=np.cross(p0-origin,np.cross(p0-origin,ref_p-origin))+origin
            #print ref_point
        elif diff[2]==0:
            ref_point=origin+[0,0,1]
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
        if flag=='cross_center':
            x1_v=f3(np.zeros(3),ref_point-origin)
            z1_v=f3(np.zeros(3),p1-origin)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=dist/2
            #here phi=[0,2pi]
            x_p2=r*np.cos(phi)*np.sin(np.pi/2)
            y_p2=r*np.sin(phi)*np.sin(np.pi/2)
            z_p2=0
            p2_new=np.array([x_p2,y_p2,z_p2])
            p2_old=np.dot(inv(T),p2_new)+origin
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='right_triangle'
        elif flag=='off_center':
            x1_v=f3(np.zeros(3),ref_point-origin)
            z1_v=f3(np.zeros(3),p1-origin)
            y1_v=np.cross(z1_v,x1_v)
            T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
            r=dist/2.
            #note in this case, phi can be in the range of [0,2pi]
            x_center=r*np.cos(phi)*np.sin(np.pi/2)
            y_center=r*np.sin(phi)*np.sin(np.pi/2)
            z_center=r*np.cos(np.pi/2)
            center_org=np.dot(inv(T),np.array([x_center,y_center,z_center]))+origin
            p2_old=2*center_org-p0
            self.p2=p2_old
            self.face=np.append(self.edge,[p2_old],axis=0)
            self.flag='right_triangle'

    def all_in_all(self,phi=np.pi/2,ref_p=[None],flag='off_center'):
        self.cal_p2(ref_p=ref_p,phi=phi,flag=flag)
        self.share_face_init(self.flag)

#steric_check will check the steric feasibility by changing the theta angle (0-pi) and or phi [0,2pi]
#the dist bw sorbate(both metal and oxygen) and atms (defined on top) will be cal and compared to the cutting_limit
#higher cutting limit will result in fewer items in return file (so be wise to choose cutting limit)
#the container has 12 items, ie phi (rotation angle), theta, low_dis, apex coors (x,y,z), os1 coors(x,y,z),os2 coors(x,y,z)
#in which the low_dis is the lowest dist between sorbate and atm

class steric_check(share_edge):
    def __init__(self,p0=anchor1,p1=anchor2,cutting_limit=2.5):
        self.edge=np.array([p0,p1])
        self.cutting_limit=cutting_limit
        self.container=np.zeros((1,18))[0:0]
        print("distance between anchor points is ",f2(p0,p1),'anstrom')
    def steric_check(self,theta_res=0.1,phi=np.pi/2,flag='off_center',print_path=None):
        #consider the steric constrain, flag 'off_center' (the center point is off the connection line of anchors)
        #is more favorable

        for theta in np.arange(0,np.pi,theta_res):
            self.all_in_all(theta=theta,phi=phi,ref_p=None,flag=flag)
            low_limit=self.cutting_limit*2
            for atm in atms:
                if ((abs(sum(atm-self.edge[0,:]))>0.01)&(abs(sum(atm-self.edge[1,:]))>0.01)):
                    dt_center_atm,dt_p2_atm,dt_p3_atm,dt_p4_atm,dt_p5_atm=f2(atm,self.center_point),f2(atm,self.p2),f2(atm,self.p3),f2(atm,self.p4),f2(atm,self.p5)
                    com_atms=np.array([dt_center_atm,dt_p2_atm,dt_p3_atm,dt_p4_atm,dt_p5_atm])
                    if np.sum(com_atms<self.cutting_limit)!=0:
                        low_limit=None
                        break
                    else:
                        if np.min(com_atms)<low_limit:
                            low_limit=np.min(com_atms)
                        else:pass
            if low_limit!=None:
                #print low_limit
                C,P2,P3,P4,P5=self.center_point,self.p2,self.p3,self.p4,self.p5
                self.container=np.append(self.container,[[phi,theta,low_limit,C[0],C[1],C[2],P2[0],P2[1],P2[2],P3[0],P3[1],P3[2],P4[0],P4[1],P4[2],P5[0],P5[1],P5[2]]],axis=0)
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
                                self.container[:,14]-7.3707,
                                self.container[:,15],self.container[:,16]-0.7558,\
                                self.container[:,17]-7.3707],names='phi,theta,low,A1,A2,A3,P20,P21,P22,P30,P31,P32,P40,P41,P42,P50,P51,P52')
        data.sort(order=('phi','theta','low'))
        print("phi,theta,low_dt_limit,A_x,A_y,A_z,P2_x,P2_y,P2_z,P3_x,P3_y,P3_z,P4_x,P4_y,P4_z,P5_x,P5_y,P5_z")
        if print_path!=None:
            np.savetxt(print_path,data,"%5.3f")
            for i in np.loadtxt(print_path):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]")
        else:
            np.savetxt('test',data,"%5.3f")
            for i in np.loadtxt('test'):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]")
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

#this share_cornor will place the octehedra right over the surface(the other anchor will have same x y as the corner, but the z value will be added by twice r)
#and the center point will be sitted at the middle point of the two anchors
class share_corner2(share_edge):
    #if want to share none, then just set the corner coordinate to the first point set arbitratly.
    def __init__(self,corner=anchor1-[0,0.7558,7.3707],r=2.2,phi=6):
        self.corner=corner
        self.p1=self.corner+[0,0,r*2]
        self.edge=np.append(self.corner[np.newaxis,:],self.p1[np.newaxis,:],axis=0)
        self.all_in_all(theta=0,phi=phi,ref_p=None,flag='cross_center')

class steric_check2(share_corner2):
    def __init__(self,p0=anchor1,r=2.0,cutting_limit=2.5):
        self.corner=p0
        self.p1=self.corner+[0,0,r*2]
        self.edge=np.append(self.corner[np.newaxis,:],self.p1[np.newaxis,:],axis=0)
        self.cutting_limit=cutting_limit
        self.container=np.zeros((1,21))[0:0]

    def steric_check(self,phi_res=0.3,flag='cross_center',print_path=None):
        #consider the steric constrain, flag 'off_center' (the center point is off the connection line of anchors)
        #is more favorable

        for phi in np.arange(0,np.pi*2,phi_res):
            self.all_in_all(phi=phi,ref_p=None,flag=flag)
            low_limit=self.cutting_limit*2
            for atm in atms:
                if (abs(sum(atm-self.corner))>0.01):
                    dt_center_atm,dt_p1_atm,dt_p2_atm,dt_p3_atm,dt_p4_atm,dt_p5_atm=f2(atm,self.center_point),f2(atm,self.p1),f2(atm,self.p2),f2(atm,self.p3),f2(atm,self.p4),f2(atm,self.p5)
                    com_atms=np.array([dt_center_atm,dt_p1_atm,dt_p2_atm,dt_p3_atm,dt_p4_atm,dt_p5_atm])
                    if np.sum(com_atms<self.cutting_limit)!=0:
                        low_limit=None
                        break
                    else:
                        if np.min(com_atms)<low_limit:
                            low_limit=np.min(com_atms)
                        else:pass
            if low_limit!=None:
                C,P1,P2,P3,P4,P5=self.center_point,self.p1,self.p2,self.p3,self.p4,self.p5
                self.container=np.append(self.container,[[phi,0,low_limit,C[0],C[1],C[2],P1[0],P1[1],P1[2],P2[0],P2[1],P2[2],P3[0],P3[1],P3[2],P4[0],P4[1],P4[2],P5[0],P5[1],P5[2]]],axis=0)
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
                                self.container[:,14]-7.3707,
                                self.container[:,15],self.container[:,16]-0.7558,\
                                self.container[:,17]-7.3707,
                                self.container[:,18],self.container[:,19]-0.7558,
                                self.container[:,20]-7.3707],names='phi,theta,low,A1,A2,A3,P10,P11,P12,P20,P21,P22,P30,P31,P32,P40,P41,P42,P50,P51,P52')
        data.sort(order=('low','phi'))
        print("phi,0,low_dt_limit,A_x,A_y,A_z,P2_x,P2_y,P2_z,P3_x,P3_y,P3_z,P4_x,P4_y,P4_z,P5_x,P5_y,P5_z")
        if print_path!=None:
            np.savetxt(print_path,data,"%5.3f")
            for i in np.loadtxt(print_path):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]")
        else:
            np.savetxt('test',data,"%5.3f")
            for i in np.loadtxt('test'):print("["+",".join(["%2.3f" for j in i])%tuple(i)+"]")
            os.remove('test')

if __name__=='__main__':
    test1=octahedra_2.share_edge(edge=np.array([[0.,0.,0.],[5.,5.,5.]]))
    test1.cal_p2(theta=0,phi=np.pi/2,flag='cross_center')
    test1.share_face_init(flag=test1.flag)
    print(test1.face,test1.p3,test1.p4,test1.p5,test1.center_point)
