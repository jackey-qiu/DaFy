import numpy as np
import vtk
import matplotlib as mpl
##unfinished code"

class visualize_crystal_structure():
    def __init__(self,structure_file_path,size_of_sc,abc,el_info,bond_length_offset=0.5):
    
        self.structure_file_path=structure_file_path
        self.size_of_super_cell=size_of_sc
        self.abc=abc
        self.atoms=self.build_super_cell()
        self.el_info=el_info
        self.bond_length_offset=bond_length_offset
        
    def build_super_cell(self,):
        
        atoms={}
        atoms_sc={}
        f=open(self.structure_file_path)
        lines=f.readlines()
        for line in lines:
            items=line.rstrip().rsplit('\t')
            if items[0] not in atoms.keys():
                atoms[items[0]]=[[float(items[1]),float(items[2]),float(items[3])]]
            else:
                atoms[items[0]].append([float(items[1]),float(items[2]),float(items[3])])
        for key in atoms.keys():
            temp_atoms=np.array(atoms[key])
            order=np.array(temp_atoms)[:,2].argsort()
            atoms[key]=np.take(np.array(temp_atoms),order,0)
            
        for x in range(self.size_of_super_cell[0]):
            for y in range(self.size_of_super_cell[1]):
                for z in range(self.size_of_super_cell[2]):
                    tag=str(x)+str(y)+str(z)
                    for key in atoms.keys():
                        temp_atoms=atoms[key]
                        for i in range(len(temp_atoms)):
                            el_label=key+'_'+str(i)+'_'+tag
                            atoms_sc[el_label]=temp_atoms[i]+self.abc*[x,y,z]
        return atoms_sc
                            
    def find_chemical_bonds(self,el):
        ids=[]
        for key in self.atoms.keys():
            if el in key:
                ids.append(key)
        for id in ids:
            


    def customize_visuralization(self,ball,line,polyhedral):



    def pick_atom_group(self,el,height,x=None,y=None):
        atoms_picked={}
        for key self.atoms.keys():
            if el in key:
                coors=atoms[key]
                if coors[2]>=height[0] and coors[2]<=height[1]:
                    if x==None and y==None:
                        atoms_picked[key]=coors
                    else:
                        if coors[0]>=x[0] and coors[0]<=x[1] and coors[0]>=y[0] and coors[0]<=y[1]:
                            atoms_picked[key]=coors
        return atoms_picked
    

    def pick_polyhedral_group(self,el):
    
    
    def create_sphere(self,):
    

    def create_cylinder(self,):
    
    
    
    
    
    def visulize_structure(self,)
    
    
    
    
    