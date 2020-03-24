import numpy as np

#formate the xyz file to vtk file format, which could be imported by paraview
#global variables

cal_dist=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
POLY_LIB={'tectrahedra':['As','P','Cr'],'pyramid':['Pb'],'octahedron':['Sb','Fe','Cd','Cu','Zn']}
HT_CUTOFF={'As':6.0,'Pb':6.0,'Fe':0.0,'Sb':0.0,'P':0.,'Cr':0.,'Cd':0.,'Cu':0.,'Zn':0.}#height cutoff values for different elements,the associated elements with the heights above the cutoff will be shown in polyhedron otherwise shown in sphere
BOND_CUTOFF={'Fe':2.22,'Pb':2.6,'Sb':2.35,'As':2.0,'P':1.8,'Cr':1.84,'Cd':2.51,'Cu':2.3,'Zn':2.31}#cutoff bond length used to locate all the coordinated members
PT_INFO={'Pb':[0.0,0.5],'Fe':[0.3,0.3],'O':[0.6,0.4],'Sb':[0.9,0.6],'As':[0.8,0.6],'P':[0.2,0.6],'Cr':[0.25,0.6],'Cd':[0.4,0.6],'Cu':[0.45,0.6],'Zn':[0.55,0.6]}#first value is scalar for the color and the second one the size
LINE_INFO={'Pb':[0.,0.5],'Fe':[0.3,0.3],'Sb':[0.9,0.6],'As':[0.8,0.6],'P':[0.2,0.6],'Cr':[0.25,0.6],'Cd':[0.4,0.6],'Cu':[0.45,0.6],'Zn':[0.55,0.6]}#the same as above
FACE_INFO={'Pb':0,'Fe':0.3,'Sb':0.9,'As':0.8,'P':0.2,'Cr':0.25,'Cd':0.4,'Cu':0.45,'Zn':0.55}#color for each face of polyhedron
#rgba values for each element, these values needed to be manually set inside the paraview GUI by setting the color transfer function values, make sure the scalar values matches to the settings above
COLOR_LIB={'Pb':[0.33984375,0.3515625,0.375,1.0],'Fe':[0.50390625,  0.48046875,  0.7734375,1],'O':[0.9375,0,0,1],'Sb':[0.62109375,  0.39453125,  0.70703125],'As':[0.742,  0.504,  0.887],\
           'P':[0.33984375,0.3515625,0.375,1.0],'Cr':[0.50390625,  0.48046875,  0.7734375,1],'Cd':[0.9375,0,0,1],'Cu':[0.62109375,  0.39453125,  0.70703125],'Zn':[0.742,  0.504,  0.887]}


class formate_vtk():
    def __init__(self,xyz_file_path="D:\\Model_domain1.xyz",lattice_pars=[5.038,5.434,7.3707],size_of_super_cell=[3,3],point_info=PT_INFO,line_info=LINE_INFO,face_info=FACE_INFO):
        self.xyz_file_path=xyz_file_path
        self.point_info,self.line_info,self.face_info=point_info,line_info,face_info
        self.points=[]
        self.els=[]
        self.lattice=lattice_pars
        self.size_of_super_cell=size_of_super_cell
        self.bond_container=None
        self.polyhedron_container=None
        self.triangle_unit_container={}
        return None
        
    def make_super_cell(self):
        f=open(self.xyz_file_path,'r')
        lines=f.readlines()[2:]
        for line in lines:
            items=line.rstrip().rsplit()
            xyz=np.array([float(items[1]),float(items[2]),float(items[3])])
            for i_h in range(self.size_of_super_cell[0]):
                for i_v in range(self.size_of_super_cell[1]):
                    self.els.append(items[0])
                    x_shift=i_h*self.lattice[0]
                    y_shift=i_v*self.lattice[1]
                    z_shift=0
                    self.points.append(xyz+[x_shift,y_shift,z_shift])
        return None
        
    def locate_bonds(self,cutoff=BOND_CUTOFF):
        keys=cutoff.keys()
        bond_container={}
        for key in keys:
            bond_container[key]=[] 
        for key in keys:
            for i in range(len(self.els)):
                if key==self.els[i]:
                    for j in range(len(self.els)):
                        if self.els[j] not in keys:
                            if abs(self.points[j][0]-self.points[i][0])<cutoff[key] and abs(self.points[j][1]-self.points[i][1])<cutoff[key] and abs(self.points[j][2]-self.points[i][2])<cutoff[key]:
                                dist=cal_dist(np.array(self.points[i]),np.array(self.points[j]))
                                if dist<cutoff[key]:
                                    bond_container[key].append([i,j])
        self.bond_container=bond_container
        
    def locate_polyhedron_unit(self):
        keys=self.bond_container.keys()
        polyhedron_container={}
        for key in keys:
            polyhedron_container[key]={} 
        for key in keys:
            pair_temp=self.bond_container[key]
            unique_index_box=[]
            [unique_index_box.append(each_pair[0]) for each_pair in pair_temp if each_pair[0] not in unique_index_box]
            for each in unique_index_box:
                polyhedron_container[key][each]=[]
                for each_pair in pair_temp:
                    if each_pair[0]==each:
                        polyhedron_container[key][each].append(each_pair[1])
        self.polyhedron_container=polyhedron_container
        return None
        
    def locate_triangle_unit(self,polyhedron_lib=POLY_LIB,height_cutoff=HT_CUTOFF):
        keys=self.polyhedron_container.keys()
        for key in keys:
            self.triangle_unit_container[key]=[]
            temp_shell_list=[]
            temp_cation_list=[]
            for each in self.polyhedron_container[key].keys():
                if self.points[each][2]>height_cutoff[key]:
                    temp_shell_list.append(self.polyhedron_container[key][each])
                    temp_cation_list.append(each)
            poly_type=None
            for each_key in polyhedron_lib.keys():
                if key in polyhedron_lib[each_key]:
                    poly_type=each_key
                    break
            if poly_type=='octahedron':
                for i in range(len(temp_shell_list)):
                    if len(temp_shell_list[i])==6:
                        initial_index=temp_shell_list[i][0]
                        initial_pt=self.points[temp_shell_list[i][0]]
                        
                        def _find_max_dist(data_list,index_list,initial_pt):
                            max_dist=0
                            max_index=None
                            for index in index_list:
                                temp_pt=data_list[index]
                                if cal_dist(np.array(initial_pt),np.array(temp_pt))>max_dist:
                                    max_dist=cal_dist(np.array(initial_pt),np.array(temp_pt))
                                    max_index=index
                            return max_index
                        max_index=_find_max_dist(self.points,temp_shell_list[i][1:],initial_pt)
                        equitorial_plane_list=[]
                        [equitorial_plane_list.append(j) for j in temp_shell_list[i][1:] if j!=max_index]
                        initial_index_equitorial_plane=equitorial_plane_list[0]
                        initial_pt_equitorial_plane=self.points[initial_index_equitorial_plane]
                        max_index_equitorial_plane=_find_max_dist(self.points,equitorial_plane_list[1:],initial_pt_equitorial_plane)
                        rest_two_index=[]
                        [rest_two_index.append(j) for j in equitorial_plane_list[1:] if j!=max_index_equitorial_plane]
                        self.triangle_unit_container[key].append([initial_index,initial_index_equitorial_plane,rest_two_index[0]])
                        self.triangle_unit_container[key].append([initial_index,initial_index_equitorial_plane,rest_two_index[1]])
                        self.triangle_unit_container[key].append([initial_index,max_index_equitorial_plane,rest_two_index[0]])
                        self.triangle_unit_container[key].append([initial_index,max_index_equitorial_plane,rest_two_index[1]])
                        self.triangle_unit_container[key].append([max_index,initial_index_equitorial_plane,rest_two_index[0]])
                        self.triangle_unit_container[key].append([max_index,initial_index_equitorial_plane,rest_two_index[1]])
                        self.triangle_unit_container[key].append([max_index,max_index_equitorial_plane,rest_two_index[0]])
                        self.triangle_unit_container[key].append([max_index,max_index_equitorial_plane,rest_two_index[1]])
            elif poly_type=='pyramid':
                for i in range(len(temp_shell_list)):
                    if len(temp_shell_list[i])==3:
                        self.triangle_unit_container[key].append(temp_shell_list[i])
            elif poly_type=='tectrahedra':
                for i in range(len(temp_shell_list)):
                    if len(temp_shell_list[i])==4:
                        i0,i1,i2,i3=temp_shell_list[i]
                        self.triangle_unit_container[key].append([i0,i1,i2])
                        self.triangle_unit_container[key].append([i0,i1,i3])
                        self.triangle_unit_container[key].append([i1,i2,i3])
                        self.triangle_unit_container[key].append([i0,i2,i3])
        return None
        
    def write_vtk_file(self):
        f=open(self.xyz_file_path.replace('.xyz','.vtk'),'w')
        f.write('# vtk DataFile Version 5.10.1\n')
        f.write('vtk file formated from the xyz file of structure model\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        #write points
        f.write('POINTS '+str(len(self.points))+' FLOAT\n')
        for i in range(len(self.points)):
            f.write("%3.4f %3.4f %3.4f\n"%(self.points[i][0],self.points[i][1],self.points[i][2]))
        #write triangles
        keys_tri_bond=[key for key in self.triangle_unit_container.keys() if self.triangle_unit_container[key]!=[]]
        N_triangles=np.sum([len(self.triangle_unit_container[each]) for each in keys_tri_bond])
        f.write('POLYGONS '+str(N_triangles)+' '+str(4*N_triangles)+'\n')
        for key in keys_tri_bond:
            for each in self.triangle_unit_container[key]:
                f.write("%i %i %i %i\n"%(3,each[0],each[1],each[2]))
        #write lines
        N_bonds=np.sum([len(self.bond_container[each]) for each in keys_tri_bond])
        f.write('LINES '+str(N_bonds)+' '+str(3*N_bonds)+'\n')
        for key in keys_tri_bond:
            for each in self.bond_container[key]:
                f.write("%i %i %i\n"%(2,each[0],each[1]))
                
        #write data attributes
        f.write('\n')
        f.write('POINT_DATA '+str(len(self.points))+'\n')
        f.write('SCALARS Color_of_Sphere float\n')
        f.write('LOOKUP_TABLE table_point\n')
        for i in range(len(self.els)):
            f.write(str(self.point_info[self.els[i]][0])+'\n')
        f.write('LOOKUP_TABLE table_point '+str(len(self.els))+'\n')
        for each_el in self.els:
            f.write("%1.5f %1.5f %1.5f %1.1f\n"%(COLOR_LIB[each_el][0],COLOR_LIB[each_el][1],COLOR_LIB[each_el][2],1.0))
        f.write('\n')
        
        f.write('SCALARS Size_of_Sphere float\n')
        f.write('LOOKUP_TABLE default\n')
        for i in range(len(self.els)):
            f.write(str(self.point_info[self.els[i]][1])+'\n')
        
        #note that the line attributes must be in front of the triangle attributes    
        f.write('\n')    
        f.write('CELL_DATA '+str(N_triangles+N_bonds)+'\n')
        f.write('SCALARS Color_of_Cells float\n')
        f.write('LOOKUP_TABLE default\n')
        for key in keys_tri_bond:
            for each in self.bond_container[key]:
                f.write(str(self.line_info[key][0])+'\n')
        for key in keys_tri_bond:
            for each in self.triangle_unit_container[key]:
                f.write(str(self.face_info[key])+'\n')

        f.write('\n')
        f.write('SCALARS Size_of_Cells float\n')
        f.write('LOOKUP_TABLE default\n')
        for key in keys_tri_bond:
            for each in self.bond_container[key]:
                f.write(str(self.line_info[key][1])+'\n')
        for key in keys_tri_bond:
            for each in self.triangle_unit_container[key]:
                if each==self.triangle_unit_container[keys_tri_bond[-1]][-1]:
                    f.write(str(self.face_info[keys_tri_bond[-1]]))
                else:
                    f.write(str(self.face_info[key])+'\n')
        f.close()
    def all_in_all(self):
        self.make_super_cell()
        self.locate_bonds()
        self.locate_polyhedron_unit()
        self.locate_triangle_unit()
        self.write_vtk_file()
        return None
if __name__=='__main__':
    test=formate_vtk("D:\\Model_domain1.xyz")
    test.all_in_all()
    test2=formate_vtk("D:\\Model_domain2.xyz")
    test2.all_in_all()
    test3=formate_vtk("D:\\Model_domain3.xyz")
    test3.all_in_all()
    test4=formate_vtk("D:\\Model_domain4.xyz")
    test4.all_in_all()
        
        
        
        
        
        
        