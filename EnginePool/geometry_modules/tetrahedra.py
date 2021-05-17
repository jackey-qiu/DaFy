import numpy as np
from numpy.linalg import inv

#see detail comment in hexahedra_4
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
    def __init__(self,face=np.array([[0.,0.,0.],[0.5,0.5,0.5],[1.0,1.0,1.0]])):
        self.face=face

    @staticmethod
    def cal_coor_o3(p0,p1,p3):
        #function to calculate the new point for p3, see document file #2 for detail procedures
        r=f2(p0,p1)/2.*np.tan(np.pi/3)
        norm_vt=p0-p1
        cent_pt=(p0+p1)/2
        a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
        d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
        u,v,w=p3[0],p3[1],p3[2]
        k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
        #projection of O3 to the normal plane see http://www.9math.com/book/projection-point-plane for detail algorithm
        O3_proj=np.array([u-a*k,v-b*k,w-c*k])
        cent_proj_vt=O3_proj-cent_pt
        l=f2(O3_proj,cent_pt)
        ptOnCircle_cent_vt=cent_proj_vt/l*r
        ptOnCircle=ptOnCircle_cent_vt+cent_pt
        return ptOnCircle

    def share_face_init(self,**args):
        p0,p1,p2=self.face[0,:],self.face[1,:],self.face[2,:]
        #consider the possible unregular shape for the known triangle
        dist_list=[np.sqrt(np.sum((p0-p1)**2)),np.sqrt(np.sum((p1-p2)**2)),np.sqrt(np.sum((p0-p2)**2))]
        index=dist_list.index(max(dist_list))
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        z_v=f3(np.zeros(3),np.cross(p1-center_point,p0-center_point))
        x_v=f3(np.zeros(3),p1-center_point)
        y_v=np.cross(z_v,x_v)
        T=f1(x0_v,y0_v,z0_v,x_v,y_v,z_v)
        self.T=T
        r=f2(p0,center_point)
        r_bc=r*(np.sqrt(2.)/4.)
        r_ed=r*(np.sqrt(2.))
        body_center_new=np.array([0.,0.,r_bc*np.cos(0.)])
        body_center_old=np.dot(inv(T),body_center_new)+center_point
        p3_new=np.array([0.,0.,r_ed*np.cos(0.)])
        p3_old=np.dot(inv(T),p3_new)+center_point
        self.p3,self.center_point,self.r=p3_old,body_center_old,f2(body_center_old,p0)

    def apply_edge_offset(self,offset=0):
        p3_old=self.p3
        ct_point=self.center_point
        unit_vector=f3(ct_point,p3_old)-ct_point
        new_length=self.r+offset
        p3_new=unit_vector*new_length+ct_point
        self.p3=p3_new

    def apply_top_angle_offset(self,offset=0):
        #move original body center along vector defined by old_bodycenter and the distal oxygen for some distance dedined by offset in A
        #the distal oxygen will be moved accordingly
        p3_old=self.p3
        ct_point=self.center_point
        unit_vector=f3(ct_point,p3_old)-ct_point
        new_length=self.r+offset
        p3_new=unit_vector*new_length+ct_point
        self.p3=p3_new
        self.center_point=self.center_point+unit_vector*offset

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
        f.write('5\n#\n')
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

        f.close()
class share_edge(share_face):
    def __init__(self,edge=np.array([[0.,0.,0.],[2.5,2.5,2.5]])):
        self.edge=edge
        self.flag=None
        self.p0,self.p1=edge[0],edge[1]

    def update_anchor_points(self, edge):
        self.edge = edge
        self.p0, self.p1 = edge[0],edge[1]

    def cal_p2(self,ref_p=None,phi=0,**args):
        p0=self.edge[0,:]
        p1=self.edge[1,:]
        origin=(p0+p1)/2
        dist=f2(p0,p1)
        diff=p1-p0
        c=np.sum(p1**2-p0**2)
        ref_point=0
        if diff[2]==0 and ref_p==None:
            ref_point=origin+[0,0,1]
        elif ref_p.any()!=None:
            ref_point=np.cross(p0-p1,np.cross(p0-p1,ref_p-p1))+origin
            #ref_point=ref_p
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
        x1_v=f3(np.zeros(3),ref_point-origin)
        z1_v=f3(np.zeros(3),p1-origin)
        y1_v=np.cross(z1_v,x1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        #note the r is different from that in the case above
        #note in this case, phi can be in the range of [0,2pi], theta is pi/2
        r=dist/2*np.sqrt(3.)
        #ang_offset is used to ensure the ref point, body center and the two anchors are on the same plane when phi is set to 0
        ang1=109.5/180*np.pi/2
        ang2=np.pi/3
        ang_offset=np.arccos((np.cos(ang1)**2+(2*np.sin(ang1)*np.sin(ang2))**2-1)/(2*np.cos(ang1)*2*np.sin(ang1)*np.sin(ang2)))

        theta=np.pi/2
        x_p2=r*np.cos(phi+ang_offset)*np.sin(theta)
        y_p2=r*np.sin(phi+ang_offset)*np.sin(theta)
        z_p2=r*np.cos(theta)
        p2_new=np.array([x_p2,y_p2,z_p2])
        p2_old=np.dot(inv(T),p2_new)+origin
        self.p2=p2_old
        self.face=np.append(self.edge,[p2_old],axis=0)

    def apply_angle_offset_BD(self,distal_angle_offset=[0,0],distal_length_offset=[0,0]):

        p2,p3,ct,r=self.p2,self.p3,self.center_point,self.r
        r1,r2=r+distal_length_offset[0],r+distal_length_offset[1]
        ang1,ang2=distal_angle_offset[0]/180*np.pi,(distal_angle_offset[1]+109.5)/180*np.pi
        z1_v=f3(np.zeros(3),p2-ct)
        y1_v=f3(np.zeros(3),np.cross(p3-ct,p2-ct))
        x1_v=np.cross(z1_v,y1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        p2_new=np.dot(inv(T),np.array([r1*np.cos(0)*np.sin(ang1),r1*np.sin(0)*np.sin(ang1),r1*np.cos(ang1)]))+ct
        p3_new=np.dot(inv(T),np.array([r2*np.cos(0)*np.sin(ang2),r2*np.sin(0)*np.sin(ang2),r2*np.cos(ang2)]))+ct
        self.p2,self.p3=p2_new,p3_new

    def apply_top_angle_offset_BD(self,top_angle_offset=0):
        #the top angle by default is 109.5 dg, by using this function, you can customize the top_angle by setting the angle offset
        p0,p1,p2,p3,ct,r=self.p0,self.p1,self.p2,self.p3,self.center_point,self.r
        origin=(p0+p1)/2
        base=f2(p0,p1)
        original_top_angle=109.47/180*np.pi
        new_top_angle=(109.47+top_angle_offset)/180*np.pi
        height_tri_old=base/2/np.tan(original_top_angle/2)
        height_tri_new=base/2/np.tan(new_top_angle/2)
        length_diff=height_tri_new-height_tri_old
        transfer_vector=(f3(origin,ct)-origin)*length_diff
        self.center_point,self.p2,self.p3=self.center_point+transfer_vector,self.p2+transfer_vector,self.p3+transfer_vector

    def apply_edge_offset_BD(self,edge_offset=0):
        #make a asymetric bond length of two anchored bonds by shifting the sorbate alone the direction of p1_ct, same vector transfer will be applied to distal oxygens
        #note after this operation, the top angle will change as well
        #edge_offset in unit of Angstrom can be positive or negative
        p0,p1,p2,p3,ct=self.p0,self.p1,self.p2,self.p3,self.center_point
        vec_transfer=(p1-ct)/f2(p1,ct)*edge_offset
        self.p2,self.p3,self.center_point=p2+vec_transfer,p3+vec_transfer,ct+vec_transfer



class share_corner(share_edge):
#if want to share none, then just set the corner coordinate to the first point arbitratly.
    def __init__(self,corner=np.array([0.,0.,0.])):
        self.corner=corner
        self.flag=None
    def cal_p1(self,r,theta,phi):
    #here we simply use the original coordinate system converted to spherical coordinate system, but at different origin
        x_p1=r*np.cos(phi)*np.sin(theta)+self.corner[0]
        y_p1=r*np.sin(phi)*np.sin(theta)+self.corner[1]
        z_p1=r*np.cos(theta)+self.corner[2]
        p1=np.array([x_p1,y_p1,z_p1])
        self.p1=p1
        self.edge=np.append(self.corner[np.newaxis,:],p1[np.newaxis,:],axis=0)

#build a dimmer of tetrahedra As knowing the two center points (two sorbate postions)
#bond_length is the As-O length (the default value is the ideal value calculated using bond valence bond length relationship)
#rotation_angle_ver is the angle in degree for the rotation of dimmer about the axis formed by the two sorbates, and the angle will only affect the distal oxygens

#AB_len is the center distance in the real case
#AB_max is the maximum distance allowed for two centers (ensure the AB_len<=AB_max, otherwise error will occur)
#rotation_axis_len is the distance between each two distal oxygens, which can be calculated by knwoing the bond_length and the dihedral angle (109.5 degree)

class make_dimmer():
    def __init__(self,rotation_angle_ver=0,center_A=np.array([2.,2.,2]),center_B=np.array([3,3,3]),bond_length=1.68):
        self.rotation_angle_ver=rotation_angle_ver/180.*np.pi
        self.center_A=center_A
        self.center_B=center_B
        self.AB_len=f2(self.center_A,self.center_B)
        self.AB_max=bond_length*np.cos(109.5/2/180*np.pi)*2
        self.OM=((self.AB_max/2)**2-(self.AB_len/2)**2)**0.5
        self.rotation_axis_len=bond_length*np.sin(109.5/2/180*np.pi)*2
        self.all_in_all()
#the idea is based on a geometry setup of cone with apex of T, A and B on its base representative of two center points
#O is the base center point, M is the projection of O on the line segment of A and B
#N is on the line, which is normal to AB and crossing M and having a rotation_angle_ver degree relative to the base normal
#O is the center point of TT_r line segment
    def cal_rotation_axis(self):
        M=(self.center_A+self.center_B)/2.
        z1_v=f3(np.zeros(3),np.array([0,0,1]))
        x1_v=f3(np.zeros(3),self.center_B-M)
        y1_v=np.cross(z1_v,x1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        N=np.dot(inv(T),np.array([1.*np.cos(np.pi/2)*np.sin(self.rotation_angle_ver),1.*np.sin(np.pi/2)*np.sin(self.rotation_angle_ver),1*np.cos(self.rotation_angle_ver)]))+M
        z1_v=f3(np.zeros(3),N-M)
        x1_v=f3(np.zeros(3),self.center_B-M)
        y1_v=np.cross(z1_v,x1_v)
        T=f1(x0_v,y0_v,z0_v,x1_v,y1_v,z1_v)
        O=np.dot(inv(T),np.array([self.OM*np.cos(np.pi/2)*np.sin(np.pi/2),self.OM*np.sin(np.pi/2)*np.sin(np.pi/2),self.OM*np.cos(np.pi/2)]))+M
        self.T=np.dot(inv(T),np.array([0,self.OM,self.rotation_axis_len/2]))+M
        self.T_r=np.dot(inv(T),np.array([0,self.OM,-self.rotation_axis_len/2]))+M
        self.ref_p_A=O*2.-self.center_A
        self.ref_p_B=O*2.-self.center_B

    def cal_the_other_distals(self):
        self.tet_caseA=share_edge(edge=np.array([self.T,self.T_r]))
        self.tet_caseA.cal_p2(ref_p=self.ref_p_A,phi=0)
        self.tet_caseA.share_face_init()

        self.tet_caseB=share_edge(edge=np.array([self.T,self.T_r]))
        self.tet_caseB.cal_p2(ref_p=self.ref_p_B,phi=0)
        self.tet_caseB.share_face_init()

    def print_xyz(self,file="D://test.xyz"):
        f=open(file,"w")
        f.write('8\n#\n')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('As', self.tet_caseA.center_point[0],self.tet_caseA.center_point[1],self.tet_caseA.center_point[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.face[0,:][0],self.tet_caseA.face[0,:][1],self.tet_caseA.face[0,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.face[1,:][0],self.tet_caseA.face[1,:][1],self.tet_caseA.face[1,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.face[2,:][0],self.tet_caseA.face[2,:][1],self.tet_caseA.face[2,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.p3[0],self.tet_caseA.p3[1],self.tet_caseA.p3[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('As', self.tet_caseB.center_point[0],self.tet_caseB.center_point[1],self.tet_caseB.center_point[2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseB.face[2,:][0],self.tet_caseB.face[2,:][1],self.tet_caseB.face[2,:][2])
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseB.p3[0],self.tet_caseB.p3[1],self.tet_caseB.p3[2])
        f.write(s)
        f.close()

    def print_xyz_for_genx(self,file="D://test_fractions.xyz"):

        f=open(file,"w")
        f.write('8\n#\n')
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('As', self.tet_caseA.center_point[0]/5.038,self.tet_caseA.center_point[1]/5.434+0.1391,self.tet_caseA.center_point[2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.face[0,:][0]/5.038,self.tet_caseA.face[0,:][1]/5.434+0.1391,self.tet_caseA.face[0,:][2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.face[1,:][0]/5.038,self.tet_caseA.face[1,:][1]/5.434+0.1391,self.tet_caseA.face[1,:][2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.face[2,:][0]/5.038,self.tet_caseA.face[2,:][1]/5.434+0.1391,self.tet_caseA.face[2,:][2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseA.p3[0]/5.038,self.tet_caseA.p3[1]/5.434+0.1391,self.tet_caseA.p3[2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('As', self.tet_caseB.center_point[0]/5.038,self.tet_caseB.center_point[1]/5.434+0.1391,self.tet_caseB.center_point[2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseB.face[2,:][0]/5.038,self.tet_caseB.face[2,:][1]/5.434+0.1391,self.tet_caseB.face[2,:][2]/7.3707+1)
        f.write(s)
        s = '%-5s   %7.5e   %7.5e   %7.5e\n' % ('O', self.tet_caseB.p3[0]/5.038,self.tet_caseB.p3[1]/5.434+0.1391,self.tet_caseB.p3[2]/7.3707+1)
        f.write(s)
        f.close()

    def all_in_all(self):
        self.cal_rotation_axis()
        self.cal_the_other_distals()
        self.print_xyz()

def steric_check_for_dimmer(file_xyz='D://temp_steric.xyz',sorbate=np.array([[0.909,1.615,10.104],[1.184,3.629,9.831]]),cutoff_length=2.6,angle_range=(0,360,10)):
    def _build_super_cell(atoms=np.array([[]]),abc=np.array([5.038,5.434,7.3707])):
        super_cell=atoms
        np.append(super_cell,atoms+[abc[0],0,0],axis=0)
        np.append(super_cell,atoms-[abc[0],0,0],axis=0)
        np.append(super_cell,atoms+[0,abc[1],0],axis=0)
        np.append(super_cell,atoms-[0,abc[1],0],axis=0)
        np.append(super_cell,atoms+[abc[0],abc[1],0],axis=0)
        np.append(super_cell,atoms+[-abc[0],-abc[1],0],axis=0)
        np.append(super_cell,atoms+[abc[0],-abc[1],0],axis=0)
        np.append(super_cell,atoms+[-abc[0],+abc[1],0],axis=0)
        return super_cell

    atoms=np.loadtxt(file_xyz)
    for i in range(len(atoms)):
        if atoms[i][0]<0:
            atoms[i][0]=atoms[i][0]+5.038
        elif atoms[i][0]>5.038:
            atoms[i][0]=atoms[i][0]-5.038
        if atoms[i][1]<0:
            atoms[i][1]=atoms[i][1]+5.434
        elif atoms[i][1]>5.434:
            atoms[i][1]=atoms[i][1]-5.434
    atoms_super_cell=_build_super_cell(atoms)

    steric_feasibility_container={}
    for angle in np.arange(angle_range[0],angle_range[1],angle_range[2]):
        hydrogen_bond_distance=[]
        tmp_case=make_dimmer(rotation_angle_ver=angle,center_A=sorbate[0],center_B=sorbate[1],bond_length=1.8)
        compared_atoms=[tmp_case.tet_caseA.p0,tmp_case.tet_caseA.p1,tmp_case.tet_caseA.p2,tmp_case.tet_caseA.p3,\
                        tmp_case.tet_caseB.p2,tmp_case.tet_caseB.p3]
        for each_atom in compared_atoms:
            for each_surface_atom in atoms_super_cell:
                hydrogen_bond_distance.append(f2(each_atom,each_surface_atom))
        if min(hydrogen_bond_distance)>=cutoff_length:
            new_distance_list=[]
            for each_distance in hydrogen_bond_distance:
                if each_distance<3.0:
                    new_distance_list.append(each_distance)
            steric_feasibility_container[angle]=new_distance_list
            print(angle,new_distance_list)

    return steric_feasibility_container







if __name__=='__main__':
    test1=tetrahedra_3.share_edge(edge=np.array([[0.,0.,0.],[5.,5.,5.]]))
    test1.cal_p2(theta=0,phi=np.pi/2)
    test1.share_face_init()
    print(test1.face,test1.p3,test1.center_point)
