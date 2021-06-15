import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import copy
from scipy.spatial.distance import pdist
import itertools

class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, X, Y, Z, text, color = QtCore.Qt.white, font_size = 10):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z
        self.color = color
        self.font_size = font_size

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        font = QtGui.QFont("Arial")
        font.setPointSize(self.font_size)
        self.GLViewWidget.qglColor(self.color)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text, font)

#color_lib = {'C':(1,0,0,1),'O':(0,1,0,1),'Cu':(1,0,1,1)}
# color_lib = {'C':0xFFFFFF,'O':(0,1,0,1),'Cu':(1,0,1,1)}
def color_to_rgb(hex_str):
    rgb=[]
    for i in [0,2,4]:
        rgb.append(int(hex_str[i:(i+2)],16)/255.)
    rgb.append(1)
    return tuple(rgb)

color_lib = {
    "H": "FFFFFF",
    "HE": "D9FFFF",
    "LI": "CC80FF",
    "BE": "C2FF00",
    "B": "FFB5B5",
    "C": "909090",
    "N": "3050F8",
    "O": "FF0D0D",
    "F": "90E050",
    "NE": "B3E3F5",
    "NA": "AB5CF2",
    "MG": "8AFF00",
    "AL": "BFA6A6",
    "SI": "F0C8A0",
    "P": "FF8000",
    "S": "FFFF30",
    "CL": "1FF01F",
    "AR": "80D1E3",
    "K": "8F40D4",
    "CA": "3DFF00",
    "SC": "E6E6E6",
    "TI": "BFC2C7",
    "V": "A6A6AB",
    "CR": "8A99C7",
    "MN": "9C7AC7",
    "FE": "E06633",
    "CO": "F090A0",
    "NI": "50D050",
    "CU": "C88033",
    "ZN": "7D80B0",
    "GA": "C28F8F",
    "GE": "668F8F",
    "AS": "BD80E3",
    "SE": "FFA100",
    "BR": "A62929",
    "KR": "5CB8D1",
    "RB": "702EB0",
    "SR": "00FF00",
    "Y": "94FFFF",
    "ZR": "94E0E0",
    "NB": "73C2C9",
    "MO": "54B5B5",
    "TC": "3B9E9E",
    "RU": "248F8F",
    "RH": "0A7D8C",
    "PD": "006985",
    "AG": "C0C0C0",
    "CD": "FFD98F",
    "IN": "A67573",
    "SN": "668080",
    "SB": "9E63B5",
    "TE": "D47A00",
    "I": "940094",
    "XE": "429EB0",
    "CS": "57178F",
    "BA": "00C900",
    "LA": "70D4FF",
    "CE": "FFFFC7",
    "PR": "D9FFC7",
    "ND": "C7FFC7",
    "PM": "A3FFC7",
    "SM": "8FFFC7",
    "EU": "61FFC7",
    "GD": "45FFC7",
    "TB": "30FFC7",
    "DY": "1FFFC7",
    "HO": "00FF9C",
    "ER": "00E675",
    "TM": "00D452",
    "YB": "00BF38",
    "LU": "00AB24",
    "HF": "4DC2FF",
    "TA": "4DA6FF",
    "W": "2194D6",
    "RE": "267DAB",
    "OS": "266696",
    "IR": "175487",
    "PT": "D0D0E0",
    "AU": "FFD123",
    "HG": "B8B8D0",
    "TL": "A6544D",
    "PB": "575961",
    "BI": "9E4FB5",
    "PO": "AB5C00",
    "AT": "754F45",
    "RN": "428296",
    "FR": "420066",
    "RA": "007D00",
    "AC": "70ABFA",
    "TH": "00BAFF",
    "PA": "00A1FF",
    "U": "008FFF",
    "NP": "0080FF",
    "PU": "006BFF",
    "AM": "545CF2",
    "CM": "785CE3",
    "BK": "8A4FE3",
    "CF": "A136D4",
    "ES": "B31FD4",
    "FM": "B31FBA",
}
As_O=1.68
Cr_O=1.64
Cd_O=2.31
Cu_O=2.09
Zn_O=2.11
Fe_O=2.02
Pb_O=2.19
Sb_O=2.04
P_O=1.534
covalent_bond_length = {
    ('Si','O'):2.0,
    ('O','Si'):2.0,
    ('Al','O'):2.0,
    ('O','Al'):2.0,
    ('Cu','O'):2.2+4,
    ('O','Cu'):2.2+4,
    ('Cu','C'):3.5+2,
    ('C','Cu'):3.5+2,
    ('As','O'):1.9+1.1,
    ('O','As'):1.9+0.3,
    ('Fe','O'):2.5,
    ('O','Fe'):2.5,
    ('Pb','O'):2.3,
    ('O','Pb'):2.3,
    ('Sb','O'):2.2,
    ('O','Sb'):2.2,
    ('C','O'):1.4,
    ('O','C'):1.4,
    ('Cr','O'):1.8,
    ('O','Cr'):1.8,
    ('Zn','O'):2.3,
    ('O','Zn'):2.3,
    ('Cd','O'):2.5,
    ('O','Cd'):2.5
}
class GLViewWidget_cum(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #set a near ortho projection (i.e. non-projective view)
        #if need a parallel view, set dis=2000, fov=1
        self.opts['distance'] = 25
        self.opts['fov'] = 60
        self.grid_num = 15
        self.abc = np.array([5.038,5.434,7.3707])
        self.T = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.T_INV = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.show_bond_length = False
        self.super_cell_size = (3,3,2)

    def clear(self):
        """
        Remove all items from the scene.
        """
        for item in self.items:
            item._setView(None)
        self.items = []
        self.update()
        
    def draw_chemical_bond(self,v1, v2, color = (1,0,0,0.8),mesh_item = None):
        dist = np.linalg.norm(np.array(v1)-np.array(v2))
        c = np.dot([0,0,1],v2-v1)/np.linalg.norm(v2-v1)
        ang = np.arccos(np.clip(c,-1,1))/np.pi*180
        vec_norm = np.cross([0,0,1],v2-v1)
        md = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1,0.1],length = dist)
        if mesh_item == None:
            mesh_item = gl.GLMeshItem(meshdata=md, smooth=True,color=color, shader='shaded', glOptions='opaque')
        else:
            mesh_item.setMeshData(meshdata = md)
        mesh_item.rotate(ang,*vec_norm)
        mesh_item.translate(*v1)
        return mesh_item

    def draw_two_chemical_bonds(self,v1, v2, colors = [(1,0,0,0.8),(1,0,0,0.8)], mesh_items = [None,None]):
        v12 = (np.array(v1)+np.array(v2))/2
        item1 = self.draw_chemical_bond(v1,v12,colors[0],mesh_items[0])
        item2 = self.draw_chemical_bond(v12,v2,colors[1],mesh_items[1]) 
        text = CustomTextItem(*v12, '{} Å'.format(round(np.linalg.norm(np.array(v1)-np.array(v2)),2)),font_size = 10)
        return item1, item2, text

    def make_super_cell_(self,super_cell_size = [3,3,1]):
        a,b,c = self.abc
        x,y,z = np.array(super_cell_size) + 1
        items = []
        for i in range(z):
            new_grid = gl.GLGridItem()
            new_grid.scale(a,b,0)
            new_grid.setSize(x-1,y-1,0)
            if (x-1)%2!=0:
                new_grid.translate(0.5*a,0,0)
            if (y-1)%2!=0:
                new_grid.translate(0,0.5*b,0)
            new_grid.translate(0,0,i*c)
            items.append(new_grid)
        for i in range(x):
            new_grid = gl.GLGridItem()
            new_grid.scale(c,b,0)
            new_grid.rotate(90, 0, 1, 0)
            new_grid.setSize(z-1,y-1,0)
            new_grid.translate(0,0,(z-1)/2*c)
            if (x-1)%2!=0:
                new_grid.translate(0.5*a+i*a-(x-1)/2*a,0,0)
            else:
                new_grid.translate(i*a-(x-1)/2*a,0,0)
            if (y-1)%2!=0:
                new_grid.translate(0,0.5*b,0)
            items.append(new_grid)

        for i in range(y):
            new_grid = gl.GLGridItem()
            new_grid.scale(a,c,0)
            new_grid.rotate(90, 1, 0, 0)
            new_grid.setSize(x-1,z-1,0)
            new_grid.translate(0,0,(z-1)/2*c)
            if (y-1)%2!=0:
                new_grid.translate(0,0.5*b+i*b-(y-1)/2*b,0)
            else:
                new_grid.translate(0,i*b-(y-1)/2*b,0)
            if (x-1)%2!=0:
                new_grid.translate(0.5*a,0,0)
            items.append(new_grid)
        self.grid_num = len(items)

        return items

    def make_super_cell(self,super_cell_size = [3,3,2]):
        line_segments = self.prepare_apex_for_supercell(super_cell_size)
        self.grid_num = len(line_segments)
        return [gl.GLLinePlotItem(pos = np.array(each),color = (0.1,0.1,0.1,1)) for each in line_segments]

    def prepare_apex_for_supercell(self, size = (3,3,2)):
        def _get_matrix(bounds = (3,3,2)):
            matrix_container = []
            x, y, _ = bounds
            for i in range(x):
                for j in range(y):
                    matrix_container.append([i-int(x/2),j-int(y/2),0])
            return np.array(matrix_container)
        translation_matrix_one_slab = _get_matrix(size)
        #np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0]])
        translation_matrix = np.array([[0,0,0]])[0:0]
        for i in range(size[2]):
            translation_matrix = np.append(translation_matrix,translation_matrix_one_slab+[0,0,i],axis = 0)
        apexs_in_one_unitcell = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
        line_segment_index_tuple = [(0,1),(0,2),(0,4),(3,7),(3,1),(3,2),(6,4),(6,7),(6,2),(5,4),(5,7),(5,1)]
        line_segment_coords = []
        for sym in translation_matrix:
            apexs_temp = [np.dot(self.T, apex) for apex in (apexs_in_one_unitcell + sym)]
            for each in line_segment_index_tuple:
                line_segment_coords.append((apexs_temp[each[0]],apexs_temp[each[1]]))
        return line_segment_coords

    def show_structure(self, xyz, show_id = False):
        # self.setCameraPosition(distance=55, azimuth=-90)
        # self.setCameraPosition(azimuth=0)
        # self.setProjection()
        a,b,c = self.abc
        if len(xyz[0])==5:
            xyz_ = [(each[0], each[2], each[3], each[4]) for each in xyz]
            ids = [each[1] for each in xyz]
            xyz = xyz_
        ii=0
        xyz_values = []
        el_list = []
        self.text_item = []
        if len(self.items)==0:
            for each in self.make_super_cell(self.super_cell_size):
                self.addItem(each)
            for i in range(len(xyz)):
                each = xyz[i]
                e, x, y, z = each
                md = gl.MeshData.sphere(rows=10, cols=20)
                m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color_to_rgb(color_lib[e.upper().strip()]), shader='shaded', glOptions='opaque')
                # print(dir(m1.metaObject()))
                m1.translate(x, y, z)
                m1.scale(0.3, 0.3, 0.3)
                self.addItem(m1)
                if show_id:
                    self.text_item.append(CustomTextItem(*[x,y,z], ids[i],font_size = 10))
                    self.text_item[-1].setGLViewWidget(self)
                xyz_values.append([x,y,z])
                el_list.append(e)
            dist_container = pdist(np.array(xyz_values),'euclidean')
            index_all = list(itertools.combinations(range(len(xyz_values)),2))
            index_dist_all = np.where(dist_container<3)[0]
            bond_index_all = [index_all[each] for each in np.where(dist_container<3)[0]]
            bond_index = []
            for i in range(len(bond_index_all)):
                each = bond_index_all[i]
                if el_list[each[0]]!=el_list[each[1]]:
                    try:
                        bond_length = covalent_bond_length[(el_list[each[0]],el_list[each[1]])]
                        if dist_container[index_dist_all[i]]<bond_length:
                            bond_index.append(each)
                    except:
                        pass
            self.bond_index = bond_index
            
            if self.bond_index!=None:
                for each_bond_index in self.bond_index:
                    v1 = np.array(xyz[each_bond_index[0]][1:])
                    v2 = np.array(xyz[each_bond_index[1]][1:])
                    color1= color_to_rgb(color_lib[xyz[each_bond_index[0]][0].upper()])
                    color2= color_to_rgb(color_lib[xyz[each_bond_index[1]][0].upper()])
                    items = self.draw_two_chemical_bonds(v1, v2, [color1,color2])
                    items[-1].setGLViewWidget(self)
                    self.text_item.append(items[-1])
                    [self.addItem(item) for item in items[0:2]]
            #append text item at the end
            if self.show_bond_length:
                [self.addItem(item) for item in self.text_item]

        else:
            for each in xyz:
                _,x,y,z = each
                #first item is grid net
                self.items[ii+self.grid_num].resetTransform()
                self.items[ii+self.grid_num].translate(x,y,z)
                self.items[ii+self.grid_num].scale(0.3, 0.3, 0.3)
                ii += 1
            if self.bond_index!=None:
                #remove text item first
                [self.removeItem(each) for each in self.text_item]
                self.text_item = []
                for each_bond_index in self.bond_index:
                    self.items[ii+self.grid_num].resetTransform()
                    self.items[ii+self.grid_num+1].resetTransform()
                    v1 = np.array(xyz[each_bond_index[0]][1:])
                    v2 = np.array(xyz[each_bond_index[1]][1:])
                    color1= color_to_rgb(color_lib[xyz[each_bond_index[0]][0].upper()])
                    color2= color_to_rgb(color_lib[xyz[each_bond_index[1]][0].upper()])
                    items = self.draw_two_chemical_bonds(v1, v2, [color1,color2],[self.items[ii+self.grid_num],self.items[ii+self.grid_num+1]])
                    items[-1].setGLViewWidget(self)
                    self.text_item.append(items[-1])
                    self.items[ii+self.grid_num], self.items[ii+self.grid_num+1]= items[0:2]
                    ii +=2
            #append text item at the end
            if self.show_bond_length:
                [self.addItem(item) for item in self.text_item]
        # self.setProjection()

    def update_structure(self, xyz):
        for i in range(len(xyz)):
            _,x,y,z = xyz[i]
            #first item is grid net
            self.items[i+self.grid_num].resetTransform()
            self.items[i+self.grid_num].translate(x,y,z)
            self.items[i+self.grid_num].scale(0.3, 0.3, 0.3)
        if self.bond_index!=None:
            ii=1
            [self.removeItem(each) for each in self.text_item]
            self.text_item = []
            for each_bond_index in self.bond_index:
                self.items[ii+i+self.grid_num].resetTransform()
                self.items[ii+i+self.grid_num+1].resetTransform()
                v1 = np.array(xyz[each_bond_index[0]][1:])
                v2 = np.array(xyz[each_bond_index[1]][1:])
                color1= color_to_rgb(color_lib[xyz[each_bond_index[0]][0].upper()])
                color2= color_to_rgb(color_lib[xyz[each_bond_index[1]][0].upper()])
                items = self.draw_two_chemical_bonds(v1, v2, [color1,color2],[self.items[ii+i+self.grid_num],self.items[ii+i+self.grid_num+1]])
                self.items[ii+i+self.grid_num],self.items[ii+i+self.grid_num+1] = items[0:2]
                items[-1].setGLViewWidget(self)
                self.text_item.append(items[-1])
                i +=2
            #append text item at the end
            if self.show_bond_length:
                [self.addItem(item) for item in self.text_item]

