import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph.Qt import QtGui,QtCore
import copy
from scipy.spatial.distance import pdist
import itertools
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
    ('Cu','O'):2.2,
    ('O','Cu'):2.2,
    ('As','O'):1.9,
    ('O','As'):1.9,
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

class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, X, Y, Z, text):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

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
        font.setPointSize(10)
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text, font)

class GLViewWidget_cum(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #set a near ortho projection (i.e. non-projective view)
        #if need a parallel view, set dis=2000, fov=1
        # self.opts['distance'] = 25
        # self.opts['fov'] = 60
        self.opts['distance'] = 2000
        self.opts['fov'] = 1
        #self.setConfigOption('background', 'w')
        #self.setConfigOption('foreground', 'k')
        self.setBackgroundColor((100,100,100))

        self.lines = []
        self.spheres = [
                        [[0,0,0],(1,0,0,0.8),0.2],
                        [[5,0,0],(1,1,0,0.8),0.2]]
        self.arrows = [
                       [[0,0,0],[0,0,1],0.1,0.2,(1,0,0,0.8)],
                       [[0,0,0],[0,1,0],0.1,0.2,(1,0,0,0.8)],
                       [[0,0,0],[1,0,0],0.1,0.2,(1,0,0,0.8)]]
        self.grids = []
        self.texts = [[0,0,0,'o']]

    def clear(self):
        """
        Remove all items from the scene.
        """
        for item in self.items:
            item._setView(None)
        self.items = []
        self.update()
        
    def draw_line_between_two_points(self, v1, v2, color = (1,0,0,0.8), width = 3):
        line = gl.GLLinePlotItem(pos=np.array([v1,v2]), width=width, color = color, antialias=False)
        return line

    #tip_length_scale is the percentage [0,1] of the length of arrow wrt the vector length
    def draw_arrow(self, v1, v2, tip_width, tip_length_scale, color):
        v2, v1 = np.array(v1), np.array(v2)
        line = gl.GLLinePlotItem(pos=np.array([v1,v2]), width=2, color = color, antialias=False)
        dist = np.linalg.norm(np.array(v1)-np.array(v2))
        c = np.dot([0,0,1],v2-v1)/np.linalg.norm(v2-v1)
        ang = np.arccos(np.clip(c,-1,1))/np.pi*180
        vec_norm = np.cross([0,0,1],v2-v1)
        md = gl.MeshData.cylinder(rows=10, cols=20, radius=[0,tip_width],length = dist*tip_length_scale)
        # if mesh_item == None:
        mesh_item = gl.GLMeshItem(meshdata=md, smooth=True,color=color, shader='shaded', glOptions='opaque')
        # else:
            # mesh_item.setMeshData(meshdata = md)
        mesh_item.rotate(ang,*vec_norm)
        mesh_item.translate(*v1)
        return line, mesh_item

    def draw_sphere(self, v1, color = (1,0,0,0.8), scale_factor = 1):
        md = gl.MeshData.sphere(rows=10, cols=20)
        m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions='opaque')
        # m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions='additive')
        # print(dir(m1.metaObject()))
        x,y,z = v1
        m1.translate(x, y, z)
        m1.scale(*([scale_factor]*3))
        return m1

    def show_structure(self):
        self.clear()
        for each_line in self.lines:
            v1, v2, color = each_line
            self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 3))
        for each_line in self.grids:
            v1, v2, color = each_line
            self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 1))
        for each_sphere in self.spheres:
            v1_, color_, scale_factor = each_sphere
            self.addItem(self.draw_sphere(v1_, color_, scale_factor)) 
        for each_arrow in self.arrows:
            v1, v2, tip_width, tip_length_scale, color = each_arrow
            items = self.draw_arrow(v1, v2, tip_width, tip_length_scale, color)
            for each in items:
                self.addItem(each)
        for each_text in self.texts:
            text = CustomTextItem(*each_text)
            text.setGLViewWidget(self)
            self.addItem(text)
        self.setProjection()

    def update_structure(self, xyz):
        for i in range(len(xyz)):
            _,x,y,z = xyz[i]
            #first item is grid net
            self.items[i+self.grid_num].resetTransform()
            self.items[i+self.grid_num].translate(x,y,z)
            self.items[i+self.grid_num].scale(0.5, 0.5, 0.5)
        if self.bond_index!=None:
            ii=1
            for each_bond_index in self.bond_index:
                self.items[ii+i+self.grid_num].resetTransform()
                self.items[ii+i+self.grid_num+1].resetTransform()
                v1 = np.array(xyz[each_bond_index[0]][1:])
                v2 = np.array(xyz[each_bond_index[1]][1:])
                color1= color_to_rgb(color_lib[xyz[each_bond_index[0]][0].upper()])
                color2= color_to_rgb(color_lib[xyz[each_bond_index[1]][0].upper()])
                items = self.draw_two_chemical_bonds(v1, v2, [color1,color2],[self.items[ii+i+self.grid_num],self.items[ii+i+self.grid_num+1]])
                self.items[ii+i+self.grid_num],self.items[ii+i+self.grid_num+1] = items
                # self.items[ii+i+grid_number].scale(0.3,0.3,0.3)
                ii +=2
