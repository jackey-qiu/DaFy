import numpy as num
import numpy as np
from numpy.linalg import inv
'''
updated from version 4: when calculate the atoms within unit cell, set the limit to [0,1) instead of [0,1] since 0 and 1 are symmetrically the same
'''
"""
Sym_creator is developped to generate symmetry operations for surface unit cell, it can also generate p1 atoms for bulk cell and surface cell
#########globals########
bulk_cell:a 3 item list, corresponding to a,b,c in bulk cell
surf_cell:a 3 item list, corresponding to a,b,c in bulk cell
bulk_to_surf:basis vector of surface unit cell in bulk unit cell
asym_atoms:a library containing asymmetry atom information
sym_file:a text file containing symmetry operations copied from cif file (no heads, no comments)
atm_p1_bulk:p1 atoms in original bulk unit cell,sorted by decreasing z value
atm_p1_surf:p1 atoms in surface unit cell, sorted by decreasing z value
sym_bulk:symmetry operations in bulk unit cell, expressed in matrix (3 by 4, rotation+shift)
sym_surf: symmetry operations in surface unit cell for each asymmetry atoms, in 3 by 4 matrix, the order is associated with (z,y,x) of atoms in surface unit cell
self.sym_surf_new_ref:symmetry operation of surface atoms for user defined reference atoms
#############################
print_file will create four text files, surface atom positions in fractional and angstrom, bulk atom positions in fractional and angstrom
"""
#just test the github
class sym_creator():
    def __init__(self,bulk_cell=[5.038,5.038,13.772],surf_cell=[5.038,5.434,7.3707],bulk_to_surf=[[1.,1.,0.],[-0.3333,0.333,0.333],[0.713,-0.713,0.287]],asym_atm={'Fe':(0.,0.,0.3553),'O':(0.3059,0.,0.25)},sym_file='D:\\Programming codes\\geometry codes\\symmetry-creator\\symmetry of hematite.txt'):
        self.bulk_cell=bulk_cell
        self.surf_cell=surf_cell
        self.bulk_to_surf=bulk_to_surf
        self.sym_file=sym_file
        self.atm_p1_bulk={}
        self.atm_p1_surf={}
        self.sym_bulk=[]
        self.sym_surf={}
        self.asym_atm=asym_atm
        self.sym_surf_new_ref={}
        self.F=num.array(self.bulk_to_surf,dtype=float)
        self.F=inv(self.F).transpose()
        for k in self.asym_atm.keys():
            self.sym_surf_new_ref[k]={}

    def create_bulk_sym(self):
        f=open(self.sym_file,'r')
        fl=f.readlines()
        for i in fl:
            sym_tmp=num.array([[0,0,0,0]],dtype=float)
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
            sym_tmp=sym_tmp[1::]
            self.sym_bulk.append(sym_tmp)
        
    def find_atm_bulk(self):
        for i in self.asym_atm.keys():
            self.atm_p1_bulk[i]=self._find_atm_bulk(self.asym_atm[i])
            
    def find_atm_surf(self):
        for i in self.asym_atm.keys():
            self.atm_p1_surf[i]=self._find_atm_surf(self.asym_atm[i])[0]
            self.sym_surf[i]=self._find_atm_surf(self.asym_atm[i])[1]
            
    def _find_atm_bulk(self,asym_atm):
        asym_atm=asym_atm
        atm_container=[]
        for i in [-1.,0.,1.]:
            for j in [-1.,0.,1.]:
                for k in [-1.,0.,1.]:
                    for s in self.sym_bulk:
                        temp= list(num.dot(s[0:3,0:3],asym_atm)+s[0:3,3]+[i,j,k])
                        #print list(temp)
                        temp[0]=num.round(temp[0],5)
                        temp[1]=num.round(temp[1],5)
                        temp[2]=num.round(temp[2],5)
                        if temp not in atm_container:
                            temp=num.array(temp)
                            tf=(temp>=0.)&(temp<1.)
                            if (int(tf[0])+int(tf[1])+int(tf[2]))==3:
                                atm_container.append(list(temp))
        return atm_container
                
    def _find_atm_surf(self,asym_atm):
        asym_atm=asym_atm
        atm_container=[]
        sym_container=[]
        for r in range(1,10):
            ct=len(atm_container)
            for i in range(-r,r):
                for j in range(-r,r):
                    for k in range(-r,r):
                        for s in self.sym_bulk:
                            temp_sym=num.dot(self.F,s[0:3,0:3])
                            temp_sym=num.append(temp_sym,num.dot(self.F,s[0:3,3]+[i,j,k])[:,num.newaxis],axis=1)
                            temp=list(num.dot(temp_sym[0:3,0:3],asym_atm)+temp_sym[0:3,3])
                            temp[0]=num.round(temp[0],5)
                            temp[1]=num.round(temp[1],5)
                            temp[2]=num.round(temp[2],5)
                            if temp not in atm_container:
                                temp=num.array(temp)
                                tf=(temp>=0.)&(temp<1.)
                                if (int(tf[0])+int(tf[1])+int(tf[2]))==3:
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
        return atm_container,sym_container
        
    def set_new_ref_atm_surf(self,el=['Fe','O'],rn=[0,1,2],print_file=False):
    #know the algorithem,T1A0+T1t=A1-->T1A0=A1-T1t-->iT1(T1A0)=A0=iT1(A1-T1t), for TnA0+Tnt=An,
    #just plug in the A0, here T1 is operation matrix,T1t is tranlation item,A0 is asymmetry atm and A1 is the ref atm
    #Basic idea is calculate the asymmetry atm from the reference atm and cal the coors for the other atms
        for element in el:
            for ref_N in rn:
                sym=num.copy(self.sym_surf[element])
                ref_atm=sym[ref_N]
                T=inv(ref_atm[0:3,0:3])
                t=-num.dot(T,ref_atm[0:3,3])
                for i in range(len(sym)):
                    M=num.dot(sym[i][0:3,0:3],T)
                    c=num.dot(sym[i][0:3,0:3],t)+sym[i][0:3,3]
                    sym[i]=num.append(M,c[:,num.newaxis],axis=1)
                self.sym_surf_new_ref[element][ref_N]=sym
                if print_file==True:
                    sym_output=num.array([[0,0,0,0,0,0,0,0,0]],dtype=float)
                    for i in sym:
                    #here we only need the rotation part to set the dxdydz in Genx, the transpose is fit with function in Genx
                        sym_output=num.append(sym_output,i[0:3,0:3].transpose().reshape(1,9),axis=0)
                    sym_output=sym_output[1::]
                    num.savetxt(element+str(ref_N)+' output file for Genx reading.txt', sym_output, delimiter=',')
    

    def set_ref_all(self,print_file=False):    
        for i in self.atm_p1_surf.keys():
            for j in range(len(self.atm_p1_surf[i])):
                self.set_new_ref_atm_surf(el=[i],rn=[j],print_file=print_file)
                #print self.sym_surf_new_ref[i][j]
            
    def ouput_sym_file_new(self,el='O',ref_N=0,print_file=False):
    #new function added after to extract sym data based on that dxdy shift at each layer won't affect dz,
    #and dz shift won't affect dxdy shift, the reference layer for atoms at one layer is now one of the atom at the same layer
    #Each row in the output file: dx1_f,dy1_f,dz1_f(0),dx2_f,dy2_f,dz2_f(0),dx3_f(0),dy3_f(0),dz3_f(1) 
    #the fact is: dx will change dx1(=dx*dx1_f) dy1(=dx*dy1_f) and dz1(=dx*dz1_f), dy will change dx2 dy2 and dz2, dz will change dx3 dy3 and dz3
    #eg (dx1_f=0.1,dy1_f=0.2, dz1_f=0.3,dx=1)-->(dx1=0.1,dy1=0.2,dz1=0.3)
        sym_output=num.array([[0,0,0,0,0,0,0,0,0]],dtype=float)[0:0]
        for i in range(len(self.sym_surf_new_ref[el][ref_N])):
            sym_output=num.append(sym_output,self.sym_surf_new_ref[el][ref_N][i][0:3,0:3].transpose().reshape(1,9),axis=0)
        sym_output[:,-1]=1.#set dz3_f to 1
        for i in range(len(sym_output)):
            for j in [2,5,6,7]:#set dz1_f dz2_f dx3_f and dy3_f to 0.
                sym_output[i][j]=0.
        if print_file==True:num.savetxt(el+' output file for Genx reading.txt', sym_output, delimiter=',')
        return sym_output
    
    def output_sym_file_layer_basis(self,el='O',print_file=False):
    #sym opts on layer basis:one atom at each layer will be selected as the reference atom to cal the other atom at the same layer
    #in other words we group atoms at the same layer, atoms of different layer are moved independently.
    
        sym_output=num.array([[0,0,0,0,0,0,0,0,0]],dtype=float)[0:0]
        N=len(self.sym_surf_new_ref[el][0])
        for i in range(0,N/2-1,2):
            sym_output=num.append(sym_output,self.sym_surf_new_ref[el][i][i][0:3,0:3].transpose().reshape(1,9),axis=0)
            sym_output=num.append(sym_output,self.sym_surf_new_ref[el][i][i+1][0:3,0:3].transpose().reshape(1,9),axis=0)
        sym_output[:,-1]=1.#set dz3_f to 1
        for i in range(len(sym_output)):
            for j in [2,5,6,7]:#set dz1_f dz2_f dx3_f and dy3_f to 0.
                sym_output[i][j]=0. 
        #repeat for the chemically equivalent half unit cell, be careful about the labels 
        #for O from top to bottom: O1 O2 O3 O4 O5 O6 (first half)<--|-->(second half) O8 O7 O10 O9 O12 O11 since O1 correspond to O8 and so on
        #for Fe from top to bottom: Fe2 Fe3 Fe4 Fe6<--|--> Fe9 Fe8 Fe12 Fe10 the number is not continual for historic reasons (in old version 12 Fe atoms are calculated with 4 duplicates)
        sym_output=num.append(sym_output,sym_output,axis=0)
        #now repeat the whole unit cell to consider a super unit cell by stacking the other unit cell on top of one cell
        sym_output=num.append(sym_output,sym_output,axis=0)
        if print_file==True:num.savetxt(el+' output file for Genx reading.txt', sym_output, delimiter=',')
        return sym_output
        
    def cal_coor(self,ref_N,element):
    #this function is for test purpose
        asym=self.atm_p1_surf[element][ref_N]
        atm_surf=num.array([[0.,0.,0.]])
        for i in self.sym_surf_new_ref[element][ref_N]:
            atm_surf=num.append(atm_surf,[num.dot(i[0:3,0:3],asym)+i[0:3,3]],axis=0)
        atm_surf=atm_surf[1:len(atm_surf)]
        return atm_surf
            
    def _test_sym_mat(self,atm_N,element):
        print 'surf_atm',self.atm_p1_surf[element][atm_N]
        for i in range(len(self.atm_p1_surf[element])):
            print ('calc_atm'+str(i),num.dot(self.sym_surf_new_ref[element][i][atm_N][0:3,0:3],self.atm_p1_surf[element][i])+self.sym_surf_new_ref[element][i][atm_N][0:3,3])
            
    def print_files(self,filename='hematite',b_f=True,b_a=True,s_f=True,s_a=True):
        if b_f==True:
            file=open(filename+'bulk_xyz_fract.txt','w')
            s = '%-5i\n' % sum([len(self.atm_p1_bulk[i]) for i in self.atm_p1_bulk.keys()])
            file.write(s)
            s= '%-5s\n' % '#bulk_xyz fractionals'
            file.write(s)
            for i in self.atm_p1_bulk.keys():
                for j in self.atm_p1_bulk[i]:
                    s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (i,j[0],j[1],j[2])
                    file.write(s)
            file.close()
        
        if b_a==True:
            file=open(filename+'bulk_xyz_angstrom.txt','w')
            s = '%-5i\n' % sum([len(self.atm_p1_bulk[i]) for i in self.atm_p1_bulk.keys()])
            file.write(s)
            s= '%-5s\n' % '#bulk_xyz in angstrom'
            file.write(s)
            for i in self.atm_p1_bulk.keys():
                for j in self.atm_p1_bulk[i]:
                    s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (i,j[0]*self.bulk_cell[0],j[1]*self.bulk_cell[1],j[2]*self.bulk_cell[2])
                    file.write(s)
            file.close()
        
        if s_f==True:
            file=open(filename+'surf_xyz_fract.txt','w')
            s = '%-5i\n' % sum([len(self.atm_p1_surf[i]) for i in self.atm_p1_surf.keys()])
            file.write(s)
            s= '%-5s\n' % '#surf_xyz fractionals'
            file.write(s)
            for i in self.atm_p1_surf.keys():
                for j in self.atm_p1_surf[i]:
                    s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (i,j[0],j[1],j[2])
                    file.write(s)
            file.close()
        
        if s_a==True:
            file=open(filename+'surf_xyz_angstrom.txt','w')
            s = '%-5i\n' % sum([len(self.atm_p1_surf[i]) for i in self.atm_p1_surf.keys()])
            file.write(s)
            s= '%-5s\n' % '#surf_xyz in angstrom'
            file.write(s)
            for i in self.atm_p1_surf.keys():
                for j in self.atm_p1_surf[i]:
                    s = '%-5s   %7.5e   %7.5e   %7.5e\n' % (i,j[0]*self.surf_cell[0],j[1]*self.surf_cell[1],j[2]*self.surf_cell[2])
                    file.write(s)
            file.close()

def test():
    bulk_cell=[6.1812,6.1812,5.698]
    surf_cell=[6.1812,6.1812,5.698]
    #basis vector of surface unit cell expressed in bulk unit cell
    bulk_to_surf=[[1,0,0.],[0,1.,0.],[0.,0.,1.]]
    #asymmetry atoms
    asym_atm={'O':(0.75, 0.25, 0.75),'O':(0.75, 0.5295, 0.1339),'H':(0.75, 0.4628, -.0137),'H':(0.75, 0.3703, -0.137),'H':(0.75, 0.6872, 0.1248),'H':(0.1325,0.5307,0.7844)}
    #symmetry operations copy from cif file
    sym_file='C:\\Users\\Canrong Qiu\\Desktop\\IceVI.txt'
    test=sym_creator(bulk_cell=bulk_cell,surf_cell=surf_cell,bulk_to_surf=bulk_to_surf,asym_atm=asym_atm,sym_file=sym_file)
    #express the symmetry operations in form of matrix (3 by 4, rotation+shift)
    test.create_bulk_sym()
    #generate p1 atoms in the bulk unit cell
    test.find_atm_bulk()
    #generate p1 atoms in the surface unit cell, and at the same time generate the symmetry operations for each atom, expressed in array (3 by 4)
    test.find_atm_surf()
    test.set_ref_all()
    return test
    
def test2():
    bulk_cell=[3.3,3.3,3.3]
    surf_cell=[3.3,3.3,3.3]
    #basis vector of surface unit cell expressed in bulk unit cell
    bulk_to_surf=[[1,0,0.],[0,1.,0.],[0.,0.,1.]]
    #asymmetry atoms
    asym_atm={'O':(0.,0,0),'H':(0.17,0.17,0.17)}
    #symmetry operations copy from cif file
    sym_file='C:\\Users\\Canrong Qiu\\Desktop\\IceVII.txt'
    test=sym_creator(bulk_cell=bulk_cell,surf_cell=surf_cell,bulk_to_surf=bulk_to_surf,asym_atm=asym_atm,sym_file=sym_file)
    #express the symmetry operations in form of matrix (3 by 4, rotation+shift)
    test.create_bulk_sym()
    #generate p1 atoms in the bulk unit cell
    test.find_atm_bulk()
    #generate p1 atoms in the surface unit cell, and at the same time generate the symmetry operations for each atom, expressed in array (3 by 4)
    test.find_atm_surf()
    test.set_ref_all()
    return test
    
def test_muscovite():
    bulk_cell=[5.1988,9.0266,20.1058]
    surf_cell=[5.1988,9.0266,20.1058]
    #basis vector of surface unit cell expressed in bulk unit cell
    bulk_to_surf=[[1,0,0.],[0,1.,0.],[0.,0.,1.]]
    #asymmetry atoms
    asym_atm={'K':(0.00000,0.09920,0.25000),
              'Si1':(0.45100,0.2587,0.13550),
              'Al1':(0.45100,0.2587,0.13550),
              'Si2':(0.03540,0.4298,0.36460),
              'Al2':(0.03540,0.4298,0.36460),
              'Al3':(0.25060,0.0838,0.00020),
              'O1':(0.38720,0.25250,0.05430),
              'O2':(0.03660,0.44310,0.44590),
              'O3':(0.41780,0.09310,0.16850),
              'O4':(0.24750,0.37120,0.16850),
              'O5':(0.25090,0.31320,0.34240),
              'O6':(0.04220,0.06220,0.44920)}
    #symmetry operations copy from cif file
    sym_file="D:\\Google Drive\\useful codes\\symmetry-creator\\muscovite_sym.txt"
    test=sym_creator(bulk_cell=bulk_cell,surf_cell=surf_cell,bulk_to_surf=bulk_to_surf,asym_atm=asym_atm,sym_file=sym_file)
    #express the symmetry operations in form of matrix (3 by 4, rotation+shift)
    test.create_bulk_sym()
    #generate p1 atoms in the bulk unit cell
    test.find_atm_bulk()
    #generate p1 atoms in the surface unit cell, and at the same time generate the symmetry operations for each atom, expressed in array (3 by 4)
    test.find_atm_surf()
    test.set_ref_all()
    return test
    
#surf_lib, bulk_lib are achieved from sym_creator.atm_p1_surf and sym_creator.atm_p1_bulk, respectively.
#the defaults are specifically for muscovite 001 surface
def make_bulk_surface_files_for_GenX(func_handle=test_muscovite,file_name_surf="D:\\Google Drive\\useful codes\\symmetry-creator\\muscovite_001_surface.str",\
                                    file_name_bulk="D:\\Google Drive\\useful codes\\symmetry-creator\\muscovite_001_bulk.str",\
                                    surface_layers =2,\
                                    u={'K':0.03467,'Si1':0.01940,'Si2':0.02105,'Al1':0.01940,'Al2':0.02105,'Al3':0.01858,\
                                       'O1':0.02807,'O2':0.02724,'O3':0.03467,'O4':0.03632,'O5':0.03921,'O6':0.03344},\
                                    oc={'K':1,'Si1':0.75,'Si2':0.75,'Al1':0.25,'Al2':0.25,'Al3':1.0,\
                                       'O1':1,'O2':1,'O3':1,'O4':1,'O5':1,'O6':1},\
                                    element={'K':'K','Si1':'Si','Si2':'Si','Al1':'Al','Al2':'Al','Al3':'Al',\
                                       'O1':'O','O2':'O','O3':'O','O4':'O','O5':'O','O6':'O'},delta={'delta1':0.,'delta2':0}):
    test=func_handle()
    surf_lib,bulk_lib=test.atm_p1_surf,test.atm_p1_bulk
    f_surf_Al=open(file_name_surf.replace('.str','_Al.str'),'w')
    f_surf_Si=open(file_name_surf.replace('.str','_Si.str'),'w')
    f_bulk=open(file_name_bulk,'w')
    surf_lib_reformated={}
    bulk_lib_reformated={}
    for key in surf_lib.keys():
        temp_array=np.rec.fromarrays(np.transpose(surf_lib[key]),names='x,y,z')
        temp_array.sort(order = ('z','y','x'))
        temp_array=temp_array[::-1]
        for i in range(len(temp_array)):
            surf_lib_reformated[key+'_'+str(i+1)]=[key]+list(temp_array[i])
    for key in bulk_lib.keys():
        temp_array=np.rec.fromarrays(np.transpose(bulk_lib[key]),names='x,y,z')
        temp_array.sort(order = ('z','y','x'))
        temp_array=temp_array[::-1]
        for i in range(len(temp_array)):
            bulk_lib_reformated[key+'_'+str(i+1)]=[key]+list(temp_array[i])
    index_bulk=[]
    index_surf=[]
    temp_surf=np.rec.fromarrays(np.transpose(surf_lib_reformated.values()),names='el,x,y,z')
    temp_surf.sort(order=('z','el','x','y'))
    temp_surf=temp_surf[::-1]
    for each in np.transpose(temp_surf):
        index_surf.append(list(surf_lib_reformated.values()).index([each[0],float(each[1]),float(each[2]),float(each[3])]))
        
    temp_bulk=np.rec.fromarrays(np.transpose(bulk_lib_reformated.values()),names='el,x,y,z')
    temp_bulk.sort(order=('z','el','x','y'))
    temp_bulk=temp_bulk[::-1]
    for each in np.transpose(temp_bulk):
        index_bulk.append(list(bulk_lib_reformated.values()).index([each[0],float(each[1]),float(each[2]),float(each[3])]))
    
    for i in index_bulk:
        temp_value=bulk_lib_reformated.values()[i]
        temp_key=bulk_lib_reformated.keys()[i]
        s='%-5s,%-5s,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e\n'%(temp_key,element[temp_value[0]],temp_value[1],temp_value[2],temp_value[3],u[temp_key.split('_')[0]],oc[temp_key.split('_')[0]],1.)
        f_bulk.write(s)
    for i in range(surface_layers):
        j=surface_layers-i-1
        for k in index_surf:
            temp_value=surf_lib_reformated.values()[k]
            temp_key=surf_lib_reformated.keys()[k]
            if oc[temp_key.split('_')[0]]==0.75:
                s='%-5s,%-5s,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e\n'%(temp_key+'_'+str(i),element[temp_value[0]],temp_value[1]+j*delta['delta1'],temp_value[2]+j*delta['delta2'],temp_value[3]+j,u[temp_key.split('_')[0]],1.0,1.)
                f_surf_Si.write(s)
            elif oc[temp_key.split('_')[0]]==0.25:
                s='%-5s,%-5s,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e\n'%(temp_key+'_'+str(i),element[temp_value[0]],temp_value[1]+j*delta['delta1'],temp_value[2]+j*delta['delta2'],temp_value[3]+j,u[temp_key.split('_')[0]],1.0,1.)
                f_surf_Al.write(s)
            else:
                s='%-5s,%-5s,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e,%7.5e\n'%(temp_key+'_'+str(i),element[temp_value[0]],temp_value[1]+j*delta['delta1'],temp_value[2]+j*delta['delta2'],temp_value[3]+j,u[temp_key.split('_')[0]],oc[temp_key.split('_')[0]],1.)
                f_surf_Si.write(s)
                f_surf_Al.write(s)
    f_surf_Al.close()
    f_surf_Si.close()
    f_bulk.close()
    return surf_lib_reformated,bulk_lib_reformated

def make_script(filename='Y:\\codes\\my code\\modeling files\\hematitesurf_xyz_fract.txt',domains=2,u={'Fe':0.32,'O':0.33},element={'Fe':0,'O':0},delta={'delta1':0.,'delta2':0.1391}):
    f=open(filename)
    fl=f.readlines()
    ff=open(filename+'_new.txt','w')
    
    for i in fl:
        line=i.rsplit()
        element[line[0]]=element[line[0]]+1
        s = '%-5s %-5s %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s\n' % \
        ('bulk.add_atom(','"'+line[0]+str(element[line[0]])+'",','"'+line[0]+'",',float(line[1]),\
        ',',float(line[2]),',',float(line[3]),',',u[line[0]],',',1.,',',1.,')')
        ff.write(s)
    for ii in element.keys():
        element[ii]=0
    for i in range(domains):
        for j in fl:
            line=j.rsplit()
            element[line[0]]=element[line[0]]+1
            s = '%-5s %-5s %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s\n' % \
            ('domain'+str(i)+'.add_atom(','"'+line[0]+str(element[line[0]])+'_'+str(i)+'",','"'+line[0]+'",',float(line[1]),\
            ',',float(line[2]),',',float(line[3]),',',u[line[0]],',',1.,',',1.,')')
            ff.write(s)
        for ii in element.keys():
            element[ii]=0
        for j in fl:
            line=j.rsplit()
            element[line[0]]=element[line[0]]+1
            s = '%-5s %-5s %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s %7.5e %-5s\n' % \
            ('domain'+str(i)+'.add_atom(','"'+line[0]+'1_'+str(element[line[0]])+'_'+str(i)+'",','"'+line[0]+'",',float(line[1])+delta['delta1'],\
            ',',float(line[2])+delta['delta2'],',',float(line[3])+1.,',',u[line[0]],',',1.,',',1.,')')
            ff.write(s)
        for ii in element.keys():
            element[ii]=0
    f.close()
    ff.close()
    
if __name__=='__main__':
    #a,b,c for bulk and surface unit cell
    bulk_cell=[5.0346,5.0346,13.7473]
    surf_cell=[5.0346,5.4266,7.3637]
    #basis vector of surface unit cell expressed in bulk unit cell
    bulk_to_surf=[[1.,1.,0.],[-0.3333333,0.333333,0.333333],[0.71308292,-0.7130829,0.286917]]
    #asymmetry atoms
    asym_atm={'Fe':(0.,0.,0.35534),'O':(0.3056,0.,0.25)}
    #symmetry operations copy from cif file
    sym_file='P:\\apps\\genx_pc_qiu\\batchfile\\symmetry of hematite.txt'
    test=sym_creator(bulk_cell=bulk_cell,surf_cell=surf_cell,bulk_to_surf=bulk_to_surf,asym_atm=asym_atm,sym_file=sym_file)
    #express the symmetry operations in form of matrix (3 by 4, rotation+shift)
    test.create_bulk_sym()
    #generate p1 atoms in the bulk unit cell
    test.find_atm_bulk()
    #generate p1 atoms in the surface unit cell, and at the same time generate the symmetry operations for each atom, expressed in array (3 by 4)
    test.find_atm_surf()
    test.print_files()
    #set the asymmetry atom in the surface unit cell, the surface atoms has been sorted by deceasing z value, so 0 here means first Fe atom
    #test.set_new_ref_atm_surf(0,'Fe')
    #print the p1 atoms in surface unit cell
    #num.array(test.atm_p1_surf['Fe'])
    #print the p1 atoms generated using new surface symmetry operations, the result should be the same as the original printing result
    #test.cal_coor(0,'Fe')
