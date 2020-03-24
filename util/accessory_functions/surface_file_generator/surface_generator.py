import numpy as num
from numpy.linalg import inv
###########variables inside class#############
"""
bulk_cell:a,b,c value for traditional bulk unit cell in anstrom
surface_cell:a,b,c value for surface unit cell in anstrom
bulk_to_surf:coordinate transformation vectors from bulk to surface unit cell (Trainor_2002, eq 24)
            [[1.,1.,0.],[-0.3333,0.333,0.333],[0.713,-0.713,0.287]] means:
            [1,1,0]:x(s)=x(b)+y(b)
            [-0.3333,0.3333,0.3333]:y(s)=-0.3333x(b)+0.3333y(b)+0.3333z(b)
            [0.713,-0.713,0.287]:z(s)=0.713x(b)-0.713y(b)+0.287z(b)
sym_file:file path to cif file (only achieve _space_group_symop_operation_xyz, delete the other parts)
atm_p1_bulk:library (two keys, including 'Fe' and 'O') to store fractional atom coordinates in bulk cell
atm_p1_surf:library (two keys, including 'Fe' and 'O') to store fractional atom coordinates in surface cell
sym_bulk:symmetry operation in bulk unit cell (3 by 4 matrix, with last column storing the translation magnitude)
sym_surf:library with keys of 'Fe' and 'o', symmetry operation in surface unit cell (3 by 4 matrix, with last column storing the translation magnitude)
asym_atm: asymmetry atom in bulk
asym_atm_new:new asymmetry atom in bulk after after translation (physically inside surface unit cell)
asym_atm_surface:asym_atm_new expressed in surface unit cell
F:coordinate system transformation matrix (num.dot(F,coor_bulk)=coor_surf,num.dot(inv(F),coor_surf)=coor_bulk))

In these variables we have the following relationship:
Ti.asym_atm_surface+t=atm_p1_surf,[Ti t]=sym_surf
"""

class surface_generator():
    
    def __init__(self,bulk_cell=[5.0346,5.0346,13.7473],surf_cell=[5.0346,5.4266,7.3637],bulk_to_surf=[[1.,1.,0.],[-0.3333,0.333,0.333],[0.713,-0.713,0.287]],asym_atm={'Fe':[0.,0.,0.35534],'O':[0.3056,0.,0.25]},sym_file='P:\\apps\\genx_pc_qiu\\batchfile\\symmetry of hematite.txt'):
        self.bulk_cell=bulk_cell
        self.surf_cell=surf_cell
        self.bulk_to_surf=bulk_to_surf
        self.sym_file=sym_file
        self.atm_p1_bulk={}
        self.atm_p1_surf={}
        self.sym_bulk=[]
        self.sym_surf={}
        self.asym_atm=asym_atm
        self.asym_atm_new={}
        self.asym_atm_surface={}
        self.F=num.array(self.bulk_to_surf,dtype=float)
        self.F=inv(self.F).transpose()
        self.action()
    
    def create_bulk_sym(self):
        f=open(self.sym_file,'r')
        #you may manually add a '.' after a number (eg. 1/3+x-->1./3+x, since eval(1/3)=0 and eval(1./3)=0.33333)
        fl=f.readlines()
        for i in fl:
            if i[0]!='#':
                sym_tmp=num.array([[0,0,0,0]],dtype=float)[0:0]
                i=eval(i).rsplit(',')
                for j in [0,1,2]:
                    x,y,z=(0.,0.,0.)
                    c=eval(i[j])
                    x,y,z=(1.,0.,0.)
                    x_=eval(i[j])-c
                    x,y,z=(0.,1.,0.)
                    y_=eval(i[j])-c
                    x,y,z=(0.,0.,1.)
                    z_=eval(i[j])-c
                    sym_tmp=num.append(sym_tmp,[[x_,y_,z_,c]],axis=0)
                self.sym_bulk.append(sym_tmp)
            
    def find_atm_bulk(self):
        for i in self.asym_atm.keys():
            self.atm_p1_bulk[i]=self._find_atm_bulk(self.asym_atm[i])
            
    def _find_atm_bulk(self,asym_atm):
        asym_atm=asym_atm
        atm_container=[]
        for i in [-1.,0.,1.]:
            for j in [-1.,0.,1.]:
                for k in [-1.,0.,1.]:
                    for s in self.sym_bulk:
                        temp= list(num.dot(s[0:3,0:3],asym_atm)+s[0:3,3]+[i,j,k])
                        #print list(temp)
                        temp[0]=num.round(temp[0],4)
                        temp[1]=num.round(temp[1],4)
                        temp[2]=num.round(temp[2],4)
                        if temp not in atm_container:
                            temp=num.array(temp)
                            if sum((temp>=0.)&(temp<1.))==3:
                                atm_container.append(list(temp))
        return atm_container
        
    def find_asym_in_surface(self):
        for key in self.asym_atm.keys():
            asym_atm=self.asym_atm[key]
            for r in range(1,10):
                for i in range(-r,r):
                    for j in range(-r,r):
                        for k in range(-r,r):
                            tmp_asym=num.dot(self.F,num.array(asym_atm)+[i,j,k])
                            if sum((tmp_asym>=0)&(tmp_asym<1))==3:
                                self.asym_atm_surface[key]=tmp_asym
                                self.asym_atm_new[key]=num.array(asym_atm)+[i,j,k]
                                break

    def create_surface_sym(self):
        for key in self.asym_atm_new.keys():
            asym_atm=self.asym_atm_surface[key]
            atm_container=[]
            sym_container=[]
            for r in range(1,10):
                ct=len(atm_container)
                for i in range(-r,r):
                    for j in range(-r,r):
                        for k in range(-r,r):
                            for s in self.sym_bulk:
                                temp_sym=num.dot(num.dot(self.F,s[0:3,0:3]),inv(self.F))
                                temp_sym=num.append(temp_sym,num.dot(self.F,s[0:3,3]+[i,j,k])[:,num.newaxis],axis=1)
                                temp=list(num.dot(temp_sym[0:3,0:3],asym_atm)+temp_sym[0:3,3])
                                temp[0]=num.round(temp[0],4)
                                temp[1]=num.round(temp[1],4)
                                temp[2]=num.round(temp[2],4)
                                if (temp not in atm_container):
                                    temp=num.array(temp)
                                    if (sum((temp>=0)&(temp<1.)))==3:
                                        atm_container.append(list(temp))
                                        sym_container.append(temp_sym)
                if ct==len(atm_container):
                    break
            data=[]
            for i in range(len(atm_container)):
                data.append((sym_container[i],atm_container[i][0],atm_container[i][1],atm_container[i][2]))
            dtype=[('sym',num.ndarray),('x',float),('y',float),('z',float)]
            data=num.array(data,dtype=dtype)
            data=num.sort(data,order=['z','y','x'])
            data=data[::-1]
            for i in range(len(data)):
                sym_container[i]=data[i][0]
                atm_container[i]=[data[i][1],data[i][2],data[i][3]]
            self.atm_p1_surf[key]=atm_container
            self.sym_surf[key]=sym_container
    
    #this function is for test purpose
    def test_right(self,el):
        atm=self.asym_atm_surface[el]
        new_atm_container=[]
        for i in range(len(self.sym_surf[el])):
            atm_new=num.dot(self.sym_surf[el][i][0:3,0:3],atm)+self.sym_surf[el][i][0:3,3]
            new_atm_container.append(atm_new)
        print num.array(new_atm_container)
        
    def action(self):
        self.create_bulk_sym()
        self.find_atm_bulk()
        self.find_asym_in_surface()
        self.create_surface_sym()
    #create stacked slabs with arbitrary height
    def create_stacked_slab(self,unit_cell=None,repeat_offset=[0,0.1391,1],column_height=2,abc=[1,1,1],file=None):
        abc=num.array(abc)
        stacked_slab=unit_cell[0:0]
        for i in range(column_height):
            stacked_slab=num.append(stacked_slab,unit_cell+num.array(repeat_offset)*i,axis=0)
        if file!=None:
            num.savetxt(stacked_slab*abc,file)
        return stacked_slab*abc
        
if __name__=='__main__':
    #a,b,c for bulk and surface unit cell
    bulk_cell=[5.038,5.038,13.772]
    surf_cell=[5.038,5.434,7.3707]
    #basis vector of surface unit cell expressed in bulk unit cell
    bulk_to_surf=[[1.,1.,0.],[-0.3333,0.333,0.333],[0.713,-0.713,0.287]]
    #asymmetry atoms
    asym_atm={'Fe':(0.,0.,0.3553),'O':(0.3059,0.,0.25)}
    #symmetry operations copy from cif file
    sym_file='D:\\Programming codes\\geometry codes\\symmetry-creator\\symmetry of hematite.txt'
    test=surface_generator.surface_generator(bulk_cell=bulk_cell,surf_cell=surf_cell,bulk_to_surf=bulk_to_surf,asym_atm=asym_atm,sym_file=sym_file)
    #you can print variables in test now
    print test.atm_p1_bulk #print atom coordinates in bulk cell
    print test.atm_p1_surf #print atom coordinates in surface cell
    print test.sym_bulk #print symmetry operations in bulk cell (3 by 4 matrix)
    print test.sym_surf #print symmetry operations in surface cell
    print test.asym_atm_surface #print asymmetry atom in surface unit cell
    print test.F #print coordinate transformation matrix (num.dot(F,coor_bulk)=coor_surface;num.dot(inv(F),coor_surface)=coor_bulk)