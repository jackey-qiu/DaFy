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

class polymer():
    def __init__(self,origin=np.array([0.,0.,0.]),build_grid=3,r=2.2,center_el='Zr',coor_el='O',domain_tag='_D1',index_offset=1,build_type='fill_up'):
        #build_grid=[10,9,4], build_type='decrease', this setting will make a regular ellipsoid-like structure
        self.r=r
        self.a=(4*r**2/3)**0.5
        self.center_point={}
        self.coordinative_members={}
        self.center_el=center_el
        self.coor_el=coor_el
        self.origin=origin
        self.domain_tag=domain_tag
        self.offset=index_offset
        self.build_index=self.translate_build_grid(build_grid,build_type)
        self.build()

    def translate_build_grid_original(self,build_grid):
        build_grid_return=[]
        if type(build_grid)==type(int(1)):
            for i in range(build_grid):
                for j in range(build_grid):
                    for k in range(build_grid):
                        switch=0
                        if (-1)**(i+j)==1:
                            switch=1
                        build_grid_return.append([k*2+switch,j,i])
        elif type(build_grid)==type([]) and type(build_grid[0])==type([]):
            build_grid_return=build_grid
        elif type(build_grid)==type([]) and type(build_grid[0])==type(int(1)):
            for i in range(build_grid[2]):
                for j in range(build_grid[1]):
                    for k in range(build_grid[0]):
                        switch=0
                        if (-1)**(i+j)==1:
                            switch=1
                        build_grid_return.append([k*2+switch,i,j])
        return build_grid_return

    def translate_build_grid(self,build_grid,build_type='fill_up'):
        build_grid_return=[]
        if type(build_grid)==type(int(1)):
            for i in range(build_grid):
                for j in range(build_grid):
                    for k in range(build_grid):
                        switch=0
                        if (-1)**(i+j)==1:
                            switch=1
                        build_grid_return.append([k*2+switch,j,i])
        elif type(build_grid)==type([]) and type(build_grid[0])==type([]):
            build_grid_return=build_grid
        elif type(build_grid)==type([]) and type(build_grid[0])==type(int(1)):
            if build_type=='fill_up':
                for i in range(build_grid[2]):
                    for j in range(build_grid[1]):
                        for k in range(build_grid[0]):
                            switch=0
                            if (-1)**(i+j)==1:
                                switch=1
                            build_grid_return.append([k*2+switch,j,i])
            elif build_type=='decrease':
                for i in range(build_grid[2]):
                    for j in range(i,build_grid[1]-i):
                        for k in range(build_grid[0]):
                            switch=0
                            if (-1)**(i+j)==1:
                                switch=1
                            if ((k*2+switch)>i) and ((k*2+switch)<(build_grid[0]-i)):
                                build_grid_return.append([k*2+switch,j,i])
                                build_grid_return.append([k*2+switch,j,-i])

        return build_grid_return

    def build(self,**arg):
        h=self.a/2
        for each_center in self.build_index:
            i,j,k=each_center
            center_point_name=self.center_el+'%s_%s_%s_%s_offset_%s%s'%(str(self.build_index.index(each_center)+1),str(i),str(j),str(k),str(self.offset),self.domain_tag)
            center_point_coord=self.a*np.array(each_center)+self.origin
            self.center_point[center_point_name]=center_point_coord
            n=1
            for x in [-h,h]:
                for y in [-h,h]:
                    for z in [-h,h]:
                        coordinative_member_xyz=np.around(center_point_coord+[x,y,z],5)
                        #print list(coordinative_member_xyz)  in self.coordinative_members.values()
                        if list(coordinative_member_xyz) not in self.coordinative_members.values():
                            coordinative_member_name=center_point_name.replace(self.domain_tag,self.coor_el+str(n)+self.domain_tag)
                            self.coordinative_members[coordinative_member_name]=list(coordinative_member_xyz)
                            n+=1

        return True

    def rotate_translate(self,translate_mag=np.array([0.,0.,0.]),rot_axis=np.array([0,0,1]),rot_point=np.array([0,0,0]),rot_angle=0):
        def _rotate(original_point,rot_axis,rot_point,rot_angle):
            #rotating original_point about the line through rot_point with direction vector defined by rot_axis by angle rot_angle
            x,y,z=original_point
            u,v,w=rot_axis
            a,b,c=rot_point
            theta=np.deg2rad(rot_angle)
            L=u**2+v**2+w**2
            x_after_rot=((a*(v**2+w**2)-u*(b*v+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*x*np.cos(theta)+L**0.5*(-c*v+b*w-w*y+v*z)*np.sin(theta))/L
            y_after_rot=((b*(u**2+w**2)-v*(a*u+c*w-u*x-v*y-w*z))*(1-np.cos(theta))+L*y*np.cos(theta)+L**0.5*(c*u-a*w+w*x-u*z)*np.sin(theta))/L
            z_after_rot=((c*(v**2+u**2)-w*(a*u+b*v-u*x-v*y-w*z))*(1-np.cos(theta))+L*z*np.cos(theta)+L**0.5*(-b*u+a*v-v*x+u*y)*np.sin(theta))/L
            return np.array([x_after_rot,y_after_rot,z_after_rot])
        self.center_point[self.center_el+str(1+self.offset)+self.domain_tag]=_rotate(self.center_point[self.center_el+str(1+self.offset)+self.domain_tag],rot_axis,rot_point,rot_angle)+translate_mag
        for key in self.coordinative_members.keys():
            self.coordinative_members[key]=_rotate(self.coordinative_members[key],rot_axis,rot_point,rot_angle)+translate_mag
        return True

    def print_xyz_file(self,file_name='D:\\test.xyz'):
        f=open(file_name,"w")
        number_atoms=len(self.center_point.keys())+len(self.coordinative_members.keys())
        f.write('%s\n#\n'%(str(number_atoms)))
        for key in self.center_point.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.center_el, self.center_point[key][0],self.center_point[key][1],self.center_point[key][2])
            f.write(s)
        for key in self.coordinative_members.keys():
            s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (self.coor_el, self.coordinative_members[key][0],self.coordinative_members[key][1],self.coordinative_members[key][2])
            f.write(s)
        f.close()
