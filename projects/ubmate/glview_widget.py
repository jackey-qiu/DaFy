import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph.Qt import QtGui,QtCore
import copy
from scipy.spatial.distance import pdist
import itertools
from OpenGL.GL import *
#color_lib = {'C':(1,0,0,1),'O':(0,1,0,1),'Cu':(1,0,1,1)}
# color_lib = {'C':0xFFFFFF,'O':(0,1,0,1),'Cu':(1,0,1,1)}
def color_to_rgb(hex_str):
    rgb=[]
    for i in [0,2,4]:
        rgb.append(int(hex_str[i:(i+2)],16)/255.)
    rgb.append(1)
    return tuple(rgb)

def RotationMatrix(theta_x, theta_y, theta_z):
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)],
                  [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    return Rx.dot(Ry).dot(Rz)

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

class GLViewWidget_cum(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        self.RM = np.eye(3)
        #set a near ortho projection (i.e. non-projective view)
        #if need a parallel view, set dis=2000, fov=1
        # self.opts['distance'] = 25
        # self.opts['fov'] = 60
        self.opts['distance'] = 2000
        self.opts['fov'] = 1
        # self.setConfigOption('background', 'w')
        #self.setConfigOption('foreground', 'k')
        self.setBackgroundColor((0,0,0))

        self.lines = []
        self.lines_dict = {}
        self.cross_points_info = {}
        self.spheres = [
                        [[0,0,0],(1,0,0,0.8),0.2],
                        [[5,0,0],(1,1,0,0.8),0.2]]
        self.ewarld_sphere = []
        self.arrows = [
                       [[0,0,0],[0,0,1],0.1,0.2,(1,0,0,0.8)],
                       [[0,0,0],[0,1,0],0.1,0.2,(1,0,0,0.8)],
                       [[0,0,0],[1,0,0],0.1,0.2,(1,0,0,0.8)]]
        self.grids = []
        self.texts = [[0,0,0,'o']]
        self.text_selected_rod = []
        self.text_item_selected_rod = None
        self.items_subject_to_transformation = []
        self.items_subject_to_recreation = []

    def apply_xyz_rotation(self):
        self.RM = RotationMatrix(np.deg2rad(self.theta_x), np.deg2rad(self.theta_y), np.deg2rad(self.theta_z)).dot(self.RM)

    #calculate the cross point(s) (if any) between a line segment and a sphere
    #The line segment is defined by two end points: line_p1 and line_p2
    #The sphere is defined by its center coordinate and its radius
    @staticmethod
    def compute_line_intersection_with_sphere(line_p1, line_p2, sphere_center, radius):
        x1, y1, z1 = line_p1
        x2, y2, z2 = line_p2
        xc, yc, zc = sphere_center
        a = (x2-x1)**2 + (y2 - y1)**2 + (z2-z1)**2
        b = 2*((x2-x1)*(xc-x1)+(y2-y1)*(yc-y1)+(zc-z1)*(z2-z1))
        c = (xc - x1)**2 + (yc-y1)**2 + (zc-z1)**2 - radius**2
        value_in_rms = b**2 - 4*a*c
        if value_in_rms>0:
            ts = np.array([(-b + value_in_rms**0.5)/(2*a),(-b - value_in_rms**0.5)/(2*a)])
        elif value_in_rms == 0:
            ts = np.array([-b/(2*a)])
        else:
            ts = np.array([])
        pts = []
        for each  in ts:
            _pt = np.array(line_p1) + (np.array(line_p2) - np.array(line_p1))*each
            _x, _y, _z = _pt
            # if (_x-x1)*(_x-x2)<=0 and (_y-y1)*(_y-y2)<=0 and (_z-z1)*(_z-z2)<=0:
            if (_z-z1)*(_z-z2)<=0:
                pts.append(_pt)
        return pts

    #v1 and v2 are orthogonal unit vector
    #The circle to be computed will be parallel to the plane defined by v1 and v2, with the center defined by center, with a radius defined by r
    @staticmethod
    def compute_points_on_3d_circle(center, v1, v2, r,resolution = 1000):
        ts = np.linspace(0,np.pi*2,resolution)
        v1, v2, p = v1/np.linalg.norm(v1),v2/np.linalg.norm(v2), np.array(center)
        points = np.array([p + r*np.cos(t)*v1 + r*np.sin(t)*v2 for t in ts])
        return points

    def mouseReleaseEvent(self, ev):
        pass
        '''
        # Example item selection code:
        region = (ev.pos().x()-0.1, ev.pos().y()-0.01, 0.01, 0.01)
        print(self.itemsAt(region))
        
        ## debugging code: draw the picking region
        
        glViewport(*self.getViewport())
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        region = (region[0], self.height()-(region[1]+region[3]), region[2], region[3])
        self.paintGL(region=region)
        self.swapBuffers()
        '''

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
    def draw_arrow(self, v1, v2, tip_width, tip_length_scale, color,line_width = 2):
        v2, v1 = np.array(v1), np.array(v2)
        line = gl.GLLinePlotItem(pos=np.array([v1,v2]), width=line_width, color = color, antialias=False)
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

    def draw_sphere(self, v1, color = (1,0,0,0.8), scale_factor = 1, rows=10, cols= 20,glOption = 'opaque'):
        md = gl.MeshData.sphere(rows=rows, cols=cols)
        # m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions='opaque')
        m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions=glOption)
        # m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions='additive')
        # print(dir(m1.metaObject()))
        x,y,z = v1
        m1.translate(x, y, z)
        m1.scale(*([scale_factor]*2+[scale_factor]))
        return m1

    def recal_cross_points(self):
        # self.apply_xyz_rotation()
        for each_item in self.items_subject_to_recreation:
            self.removeItem(each_item)
        self.items_subject_to_recreation = []
        self.cross_points_info = {}
        for each_line in self.lines:
            v1, v2, color = each_line
            v1, v2 = list(self.RM.dot(np.array(v1))), list(self.RM.dot(np.array(v2)))
            name = None
            for each_name in self.lines_dict:
                if each_line in self.lines_dict[each_name]:
                    name = each_name
                    break
            #self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 3))
            #self.items_subject_to_transformation.append(self.items[-1])
            if len(self.ewarld_sphere)!=0:
                #print('yes')
                v1_, _, scale_factor = self.ewarld_sphere
                cross_points = list(self.compute_line_intersection_with_sphere(v1, v2, v1_, scale_factor))
                if name in self.cross_points_info:
                    self.cross_points_info[name] += cross_points
                else:
                    self.cross_points_info[name] = cross_points
                for each in cross_points:
                    points_on_circle_full_circle = self.compute_points_on_3d_circle(center=v1_, v1=each, v2=(each-np.array(v1_)*2), r=scale_factor,resolution = 100)
                    points_on_circle = self.compute_points_on_3d_circle(center=np.array(each)*[0,1,0], v1=np.array([0,0,1]), v2=np.array([1,0,0]), r=(scale_factor**2-(scale_factor-abs(each[1]))**2)**0.5,resolution = 100)
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=False))
                    self.items_subject_to_recreation.append(self.items[-1])
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle_full_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=False))
                    self.items_subject_to_recreation.append(self.items[-1])
                    self.addItem(self.draw_sphere(each, (0,0,1,1), 0.1))
                    self.items_subject_to_recreation.append(self.items[-1])
                    #ki = self.draw_arrow(v1_,[0,0,0], tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    kfs=self.draw_arrow(v1_, each, tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    q_vs= self.draw_arrow([0,0,0], each, tip_width=0.1, tip_length_scale=0.02, color=(1.,1.,1,1),line_width =0.5)
                    for kf in kfs:
                        self.addItem(kf)
                        self.items_subject_to_recreation.append(self.items[-1])
                    for q_v in q_vs:
                        self.addItem(q_v)
                        self.items_subject_to_recreation.append(self.items[-1])

    def show_structure(self):
        self.clear()
        self.cross_points_info = {}
        self.items_subject_to_transformation = []
        self.items_subject_to_recreation = []
        self.RM = np.eye(3)
        for each_line in self.lines:
            v1, v2, color = each_line
            name = None
            for each_name in self.lines_dict:
                if each_line in self.lines_dict[each_name]:
                    name = each_name
                    break
            self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 3))
            self.items_subject_to_transformation.append(self.items[-1])
            if len(self.ewarld_sphere)!=0:
                v1_, _, scale_factor = self.ewarld_sphere
                cross_points = list(self.compute_line_intersection_with_sphere(v1, v2, v1_, scale_factor))
                if name in self.cross_points_info:
                    self.cross_points_info[name] += cross_points
                else:
                    self.cross_points_info[name] = cross_points
                for each in cross_points:
                    points_on_circle_full_circle = self.compute_points_on_3d_circle(center=v1_, v1=each, v2=(each-np.array(v1_)*2), r=scale_factor,resolution = 100)
                    points_on_circle = self.compute_points_on_3d_circle(center=np.array(each)*[0,1,0], v1=np.array([0,0,1]), v2=np.array([1,0,0]), r=(scale_factor**2-(scale_factor-abs(each[1]))**2)**0.5,resolution = 100)
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=False))
                    self.items_subject_to_recreation.append(self.items[-1])
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle_full_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=False))
                    self.items_subject_to_recreation.append(self.items[-1])
                    self.addItem(self.draw_sphere(each, (0,0,1,1), 0.1))
                    self.items_subject_to_recreation.append(self.items[-1])
                    ki = self.draw_arrow(v1_,[0,0,0], tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    kfs=self.draw_arrow(v1_, each, tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    q_vs= self.draw_arrow([0,0,0], each, tip_width=0.1, tip_length_scale=0.02, color=(1.,1.,1,1),line_width =0.5)
                    for kf in kfs:
                        self.addItem(kf)
                        self.items_subject_to_recreation.append(self.items[-1])
                    for q_v in q_vs:
                        self.addItem(q_v)
                        self.items_subject_to_recreation.append(self.items[-1])
                    for each_ki in ki:
                        self.addItem(each_ki)
                    # text = CustomTextItem(*each, 'x', color = QtCore.Qt.red)
                    # text.setGLViewWidget(self)
                    # self.addItem(text)
        for each_line in self.grids:
            v1, v2, color = each_line
            self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 1))
            self.items_subject_to_transformation.append(self.items[-1])
        for each_sphere in self.spheres:
            v1_, color_, scale_factor = each_sphere
            self.addItem(self.draw_sphere(v1_, color_, scale_factor)) 
            self.items_subject_to_transformation.append(self.items[-1])
        if len(self.ewarld_sphere)!=0:
            v1_, color_, scale_factor = self.ewarld_sphere
            self.addItem(self.draw_sphere(v1_, color_, scale_factor,rows=100, cols=100, glOption = 'additive'))
        for each_arrow in self.arrows:
            v1, v2, tip_width, tip_length_scale, color = each_arrow
            items = self.draw_arrow(v1, v2, tip_width, tip_length_scale, color)
            for each in items:
                self.addItem(each)
        for each_text in self.texts:
            text = CustomTextItem(*each_text)
            text.setGLViewWidget(self)
            self.addItem(text)
        if len(self.text_selected_rod)!=0:
            text = CustomTextItem(*self.text_selected_rod, font_size = 20)
            text.setGLViewWidget(self)
            self.addItem(text)
            self.text_item_selected_rod = self.items[-1]
        self.setProjection()

    def update_text_item_selected_rod(self):
        text = CustomTextItem(*self.text_selected_rod, font_size = 20)
        text.setGLViewWidget(self)
        self.addItem(text)
        if self.text_item_selected_rod!=None:
            self.removeItem(self.text_item_selected_rod)
        self.text_item_selected_rod = self.items[-1]
        self.update()


    def update_structure(self):
        self.apply_xyz_rotation()
        for each in self.items_subject_to_transformation:
            # each.resetTransform()
            each.rotate(self.theta_x, 1,0,0, local = False)
            each.rotate(self.theta_y, 0,1,0, local = False)
            each.rotate(self.theta_z, 0,0,1, local = False)
        self.recal_cross_points()
        self.update()
