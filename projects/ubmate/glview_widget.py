import pyqtgraph as pg
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')
import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph.Qt import QtGui,QtCore
import copy
from scipy.spatial.distance import pdist
from scipy.stats import multivariate_normal
import itertools
from OpenGL.GL import *
import matplotlib.pyplot as plt
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
    # return Rx*Ry*Rz
    return Rx.dot(Ry).dot(Rz)

def RotationMatrix_along_arbitrary_vector_(vec, ang):
    vec = np.array(vec)
    vec = vec/np.linalg.norm(vec)
    ux, uy, uz = vec
    W = np.array([[0,-uz,uy],[uz,0,-ux],[-uy,ux,0]])
    RRMT = np.eye(3) + np.sin(np.deg2rad(ang))*W + 2*(np.sin(np.deg2rad(ang/2))**2)*W**2
    return RRMT

def RotationMatrix_along_arbitrary_vector(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    The calculation is based on  Rodrigues rotation formula
    """
    theta = np.deg2rad(theta)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

#cal the angle between two vectors
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

PILATUS_SIZE = [423.6,434.6]
PIXEL_SIZE = [0.172,0.172]
DISTANCE_SAMPLE_DETECTOR = 500

PILATUS_DIM = list((np.array(PILATUS_SIZE[::-1])/PIXEL_SIZE[::-1]).astype(int))
PRIMARY_BEAM_POS= [int(PILATUS_DIM[0]-1), int(PILATUS_DIM[1]/2)]

def index_on_pilatus(pilatus_size = PILATUS_SIZE,pixel_size = PIXEL_SIZE, distance_sample_detector = DISTANCE_SAMPLE_DETECTOR, delta=0, gamma =0, primary_beam_pos = None, debug = False):
    #default is PILATUS 6M
    #map the index of a reciprocal vector (vertical detector angle delta (in degree), horizontal detector angle gamma (in degree)) 
    # on a pilatus with size of pilatus_size of [width,height], pixel_size of [width, height] (all in unit mm)
    #in a geometry defined by distance_sample_detector (mm)
    #assume the primary beam Ki will hit the detector image (dimention of [r,c]) at [r-1,c/2] (the position of o as shown below)

    #++++++++++++
    #++++++++++++
    #++++++++++++
    #++++++++++++
    #+++++o++++++
    pilatus_dim = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))#[rows, columns]
    def _out_of_detector(pos):
        if pos[0]<0 or pos[1]<0:
            return True
        if pos[0]>pilatus_dim[0] or pos[1]>pilatus_dim[1]:
            return True
        return False
    if primary_beam_pos==None:
        primary_beam_pos = [int(pilatus_dim[0]-1), int(pilatus_dim[1]/2)]
    horizontal_offset_in_pixels = int(np.tan(np.deg2rad(gamma))*distance_sample_detector/pixel_size[0])
    vertical_offset_in_pixels = int(distance_sample_detector/np.cos(np.deg2rad(gamma))*np.tan(np.deg2rad(delta))/pixel_size[1])
    pos = [primary_beam_pos[0]-vertical_offset_in_pixels, primary_beam_pos[1]-horizontal_offset_in_pixels]
    #outside_detector = False
    #if abs(horizontal_offset_in_pixels)>pilatus_dim[1]/2 or vertical_offset_in_pixels>pilatus_dim[0]:
    #    outside_detector = True
    if debug:
        print(f'pilatus dim: {pilatus_dim}\nprimary beam position:{primary_beam_pos}\nconditions: delta={delta}, gamma = {gamma}')
        print(f'Is the calcualted spot outside the detector:{outside_detector}\nThe calculated pixel index is {pos}')
    else:
        if _out_of_detector(pos):
            return []
        else:
            return pos

def simulate_pixel_image_(pilatus_size = PILATUS_SIZE,pixel_size = PIXEL_SIZE,pos = [0,0], peak_dim = 101, sigma = 0.55):
    def gaussian_3d_peak(peak_dim, sigma):
        x, y = np.mgrid[-1.0:1.0:peak_dim*1j, -1.0:1.0:peak_dim*1j]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array([0.0, 0.0])
        sigma = np.array([.25, .25])
        covariance = np.diag(sigma**2)
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        # Reshape back to a (30, 30) grid.
        z = z.reshape(x.shape)
        return z
    pilatus_dim = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))#[rows, columns]
    img = np.zeros(pilatus_dim)
    if len(pos)==0:
        return img
    relative_center_of_single_peak = np.array([int((peak_dim-1)/2)+1]*2)
    offset = np.array(pos) - relative_center_of_single_peak
    x, y = np.mgrid[0:(peak_dim-1):peak_dim*1j, 0:(peak_dim-1):peak_dim*1j]
    indexs = np.column_stack([x.flat, y.flat]).astype(int)
    for each_index in indexs:
        temp_index = each_index + offset.astype(int)
        if (temp_index[0]>=0) and (temp_index[0]<pilatus_dim[0]) and (temp_index[1]>=0) and (temp_index[1]<pilatus_dim[1]):
            img[temp_index[0],temp_index[1]]+= gaussian_3d_peak(peak_dim,sigma)[each_index[0],each_index[1]]
    return img

def simulate_pixel_image(pilatus_size = PILATUS_SIZE, pixel_size = PIXEL_SIZE, pos = [0,0], peak_dim = 50, gaussian_sim = False):
    pilatus_dim = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))#[rows, columns]
    if not gaussian_sim:
        return np.zeros(pilatus_dim)+1
    else:
        return makeGaussian(pilatus_dim, fwhm = peak_dim, center = pos)

def makeGaussian(size=[10,30], fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0,size[0],1,float)[:,np.newaxis]
    #y = x[:,np.newaxis]

    if center is None:
        x0 = size[1] // 2
        y0 = size[0] // 2
    else:
        x0 = center[1]
        y0 = center[0]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


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
        #absolute values
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        #relative to previous values
        self.theta_x_r = 0
        self.theta_y_r = 0
        self.theta_z_r = 0
        self.theta_SN = 0
        self.theta_SN_r = 0
        self.RM = np.eye(3)
        self.RRM = np.eye(3)
        self.SN_vec = np.array([0,0,1])
        #set a near ortho projection (i.e. non-projective view)
        #if need a parallel view, set dis=2000, fov=1
        # self.opts['distance'] = 25
        # self.opts['fov'] = 60
        self.opts['distance'] = 2000
        self.opts['fov'] = 1
        self.primary_beam_position = PRIMARY_BEAM_POS[::-1]
        # self.setConfigOption('background', 'w')
        # self.setConfigOption('foreground', 'k')
        # self.setBackgroundColor((100,100,100))
        self.setBackgroundColor('k')

        self.lines = []
        self.lines_dict = {}
        self.unit_cell_edges = {}
        self.cross_points_info = {}
        self.cross_points_info_HKL = {}
        self.pixel_index_of_cross_points = {}
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
        self.text_item_sym_rods = None
        self.text_sym_rods = []

        self.items_subject_to_transformation = []
        self.items_subject_to_recreation = []

    def calculate_index_on_pilatus_image_from_cross_points_info(self, pilatus_size, pixel_size, distance_sample_detector, primary_beam_pos = None):
        PILATUS_DIM = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))
        if primary_beam_pos == None:
            PRIMARY_BEAM_POS= [int(PILATUS_DIM[0]-1), int(PILATUS_DIM[1]/2)]
            self.primary_beam_position = PRIMARY_BEAM_POS[::-1]
        else:
            self.primary_beam_position = primary_beam_pos[::-1]
        if len(self.ewarld_sphere)==0:
            return
        else:
            self.pixel_index_of_cross_points = {}
            center = np.array(self.ewarld_sphere[0])
            ki = -center
            for each in self.cross_points_info:
                index_container = []
                for each_item in self.cross_points_info[each]:
                    item_vector = each_item - center
                    item_vector_projected_on_xy_plane = item_vector*[1,1,0]
                    delta = np.rad2deg(angle_between(item_vector,item_vector_projected_on_xy_plane))
                    gamma = np.rad2deg(angle_between(ki, item_vector_projected_on_xy_plane))
                    # print('delta={},gamma={}'.format(delta,gamma))
                    #depend on the projected vector position, either on the left side of ki or right size of ki, both case the calcualted gamma is the same value
                    #so here you need to manually differentiate the symmetry positions
                    if item_vector[0]<0:
                        gamma = -gamma
                    index_container.append(index_on_pilatus(pilatus_size = pilatus_size,pixel_size = pixel_size, distance_sample_detector = distance_sample_detector, delta=delta, gamma =gamma, primary_beam_pos = primary_beam_pos))
                self.pixel_index_of_cross_points[each] = index_container

    def calculate_index_on_pilatus_image_from_angles(self, pilatus_size, pixel_size, distance_sample_detector,angle_info, primary_beam_pos = None):
        PILATUS_DIM = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))
        if primary_beam_pos==None:
            PRIMARY_BEAM_POS= [int(PILATUS_DIM[0]-1), int(PILATUS_DIM[1]/2)]
            self.primary_beam_position = PRIMARY_BEAM_POS[::-1]
        else:
            self.primary_beam_position = primary_beam_pos[::-1]
        if len(self.ewarld_sphere)==0:
            print("Ewarld sphere is not defined! Check it first!")
            return
        else:
            self.pixel_index_of_Bragg_reflections = {}
            center = np.array(self.ewarld_sphere[0])
            ki = -center
            for each in angle_info:
                index_container = []
                for each_item in angle_info[each]:
                    gamma, delta = each_item
                    if (not np.isnan(gamma)) and (not np.isnan(delta)):
                        index_container.append(index_on_pilatus(pilatus_size = pilatus_size,pixel_size = pixel_size, distance_sample_detector = distance_sample_detector, delta=delta, gamma =gamma, primary_beam_pos= primary_beam_pos))
                self.pixel_index_of_Bragg_reflections[each] = index_container

    def cal_simuated_2d_pixel_image(self, pilatus_size, pixel_size, gaussian_sim = False):
        image_sum = 0
        if not gaussian_sim:
            pilatus_dim = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))#[rows, columns]
            return np.zeros(pilatus_dim)+1
        for each in self.pixel_index_of_cross_points:
            for each_item in self.pixel_index_of_cross_points[each]:
                if len(each_item)!=0:
                    image_sum += simulate_pixel_image(pilatus_size = pilatus_size,pixel_size = pixel_size,pos = each_item, gaussian_sim = gaussian_sim)
        return image_sum
        #plt.imshow(image_sum)
        #plt.show()
    def cal_simuated_2d_pixel_image_Bragg_peaks(self, pilatus_size, pixel_size, gaussian_sim = False):
        image_sum = 0
        if not gaussian_sim:
            pilatus_dim = list((np.array(pilatus_size[::-1])/pixel_size[::-1]).astype(int))#[rows, columns]
            return np.zeros(pilatus_dim)+1
        for each in self.pixel_index_of_Bragg_reflections:
            for each_item in self.pixel_index_of_Bragg_reflections[each]:
                if len(each_item)!=0:
                    image_sum += simulate_pixel_image(pilatus_size = pilatus_size,pixel_size = pixel_size,pos = each_item, gaussian_sim = gaussian_sim)
        return image_sum

    def apply_xyz_rotation(self):
        # self.RM = RotationMatrix(np.deg2rad(self.theta_x), np.deg2rad(self.theta_y), np.deg2rad(self.theta_z)).dot(self.RM)
        self.RM = RotationMatrix(np.deg2rad(self.theta_x), np.deg2rad(self.theta_y), np.deg2rad(self.theta_z))

    def apply_SN_rotation(self):
        vec = RotationMatrix(np.deg2rad(self.theta_x), np.deg2rad(self.theta_y), np.deg2rad(self.theta_z)).dot(np.array([0,0,1]))
        self.SN_vec = vec
        self.RRM = RotationMatrix_along_arbitrary_vector(vec, self.theta_SN)
        #print(RotationMatrix_along_arbitrary_vector(vec, self.tehta_SN))

    #calculate the cross point(s) (if any) between a line segment and a sphere
    #The line segment is defined by two end points: line_p1 and line_p2
    #The sphere is defined by its center coordinate and its radius
    @staticmethod
    def compute_line_intersection_with_sphere_old(line_p1, line_p2, sphere_center, radius):
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

    @staticmethod
    def compute_line_intersection_with_sphere(line_p1, line_p2, sphere_center, radius):
        x1, y1, z1 = line_p1
        x2, y2, z2 = line_p2
        x3, y3, z3 = sphere_center
        r = radius
        a = (x2-x1)**2 + (y2 - y1)**2 + (z2-z1)**2
        b = 2*((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
        c = x3**2 + y3**2 + z3**2 + x1**2 + y1**2 + z1**2 - 2*(x3 * x1 + y3 * y1 + z3 * z1) - r**2
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
                #remove cross points at origins
                if np.abs(_pt).sum()>0.001:
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
        line = gl.GLLinePlotItem(pos=np.array([v1,v2]), width=width, color = color, antialias=True)
        return line

    #tip_length_scale is the percentage [0,1] of the length of arrow wrt the vector length
    def draw_arrow(self, v1, v2, tip_width, tip_length_scale, color,line_width = 2):
        v2, v1 = np.array(v1), np.array(v2)
        line = gl.GLLinePlotItem(pos=np.array([v1,v2]), width=line_width, color = color, antialias=True)
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
        m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions=glOption)
        # m1 = gl.GLMeshItem(meshdata=md, smooth=True, color=color)
        x,y,z = v1
        m1.translate(x, y, z)
        m1.scale(*([scale_factor]*2+[scale_factor]))
        return m1

    def recal_cross_points(self, xyz_dir = True):
        # self.apply_xyz_rotation()
        self.recreated_items_according_to_substrate_and_hkl = {}
        for each_item in self.items_subject_to_recreation:
            self.removeItem(each_item)
        self.items_subject_to_recreation = []
        self.cross_points_info = {}
        for each_line in self.lines:
            recreated_items = []
            v1, v2, color = each_line
            v1, v2 = list(self.RRM.dot(self.RM.dot(np.array(v1)))), list(self.RRM.dot(self.RM.dot(np.array(v2))))
            '''
            if xyz_dir:
                v1, v2 = list(self.RM.dot(np.array(v1))), list(self.RM.dot(np.array(v2)))
            else:
                v1, v2 = list(self.RRM.dot(self.RM.dot(np.array(v1)))), list(self.RRM.dot(self.RM.dot(np.array(v2))))
                # v1, v2 = list(self.RRM.dot(np.array(v1))), list(self.RRM.dot((np.array(v2))))
            '''
            name = None
            for each_name in self.lines_dict:
                if each_line in self.lines_dict[each_name]:
                    name = each_name
                    break
            #self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 3))
            #self.items_subject_to_transformation.append(self.items[-1])
            hkl_key_ = self.HKLs_dict[name][self.lines_dict[name].index(each_line)]
            if len(self.ewarld_sphere)!=0:
                #print('yes')
                v1_, _, scale_factor = self.ewarld_sphere
                cross_points = list(self.compute_line_intersection_with_sphere(v1, v2, v1_, scale_factor))
                # print(cross_points)
                if name in self.cross_points_info:
                    self.cross_points_info[name] += cross_points
                else:
                    self.cross_points_info[name] = cross_points
                for each in cross_points:
                    points_on_circle_full_circle = self.compute_points_on_3d_circle(center=v1_, v1=each, v2=(each-np.array(v1_)*2), r=scale_factor,resolution = 100)
                    points_on_circle = self.compute_points_on_3d_circle(center=np.array(each)*[0,1,0], v1=np.array([0,0,1]), v2=np.array([1,0,0]), r=(scale_factor**2-(scale_factor-abs(each[1]))**2)**0.5,resolution = 100)
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=True))
                    self.items_subject_to_recreation.append(self.items[-1])
                    recreated_items.append(self.items[-1])
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle_full_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=True))
                    self.items_subject_to_recreation.append(self.items[-1])
                    recreated_items.append(self.items[-1])
                    self.addItem(self.draw_sphere(each, (0,0,1,1), 0.1))
                    self.items_subject_to_recreation.append(self.items[-1])
                    recreated_items.append(self.items[-1])
                    #ki = self.draw_arrow(v1_,[0,0,0], tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    kfs=self.draw_arrow(v1_, each, tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    q_vs= self.draw_arrow([0,0,0], each, tip_width=0.1, tip_length_scale=0.02, color=(1.,1.,1,1),line_width =0.5)
                    for kf in kfs:
                        self.addItem(kf)
                        self.items_subject_to_recreation.append(self.items[-1])
                        recreated_items.append(self.items[-1])
                    for q_v in q_vs:
                        self.addItem(q_v)
                        self.items_subject_to_recreation.append(self.items[-1])
                        recreated_items.append(self.items[-1])
                if name not in self.recreated_items_according_to_substrate_and_hkl:
                    self.recreated_items_according_to_substrate_and_hkl[name] = {tuple(hkl_key_):recreated_items}
                else:
                    self.recreated_items_according_to_substrate_and_hkl[name][tuple(hkl_key_)] = recreated_items
                    
    def generate_detector_object_cube(self, origin = [1/2,1/6,1/2], unit_basis = 1, face_color = [0,1,0,1.], or_100 = 0, or_001 = 0):
        vertexes = unit_basis*(np.array([[1, 0, 0], #0
                     [0, 0, 0], #1
                     [0, 1/3, 0], #2
                     [0, 0, 1], #3
                     [1, 1/3, 0], #4
                     [1, 1/3, 1], #5
                     [0, 1/3, 1], #6
                     [1, 0, 1]])-[1/2,1/6,1/2])#7

        faces = np.array([[1,0,7], [1,3,7],
                        [1,2,4], [1,0,4],
                        [1,2,6], [1,3,6],
                        [0,4,5], [0,7,5],
                        [2,4,5], [2,6,5],
                        [3,6,5], [3,7,5]])

        colors = np.array([face_color for i in range(12)])

        cube = gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors,
                            drawEdges=False, edgeColor=(1, 0, 0, 1))
        cube.rotate(or_100,1,0,0,local = False)
        cube.rotate(or_001,0,0,1,local = False)
        cube.translate(*origin)
        return cube

    def _get_detector_pos(self, delta, gam):
        r = self.ewarld_sphere[2]*2
        #since ki point towards y axis which is 90 deg away from x axis
        phi = np.deg2rad(gam + 90 )
        theta = np.deg2rad(90 - delta)
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)
        return np.array([x, y, z])+self.ewarld_sphere[0]

    def send_detector(self, delta, gam):
        pos = self._get_detector_pos(delta,gam)
        self.detector, self.detector_line = self.generate_detector_object(pos, or_100 = delta, or_001 = gam)

    def generate_detector_object(self, origin, color = [0,1,0,0.8], or_100 = 0, or_001 =0):
        if self.detector!=None:
            self.removeItem(self.detector)
            self.removeItem(self.detector_line)
            # self.addItem(self.draw_sphere(origin,color,scale_factor = self.ewarld_sphere[2]*0.1))
            self.addItem(self.generate_detector_object_cube(origin,unit_basis = self.ewarld_sphere[2]*0.1, or_100= or_100, or_001 = or_001))
            self.addItem(self.draw_line_between_two_points(v1 = self.ewarld_sphere[0], v2 = origin,color = [1,0,0,0.3],width = 1))
        else:
            # self.addItem(self.draw_sphere(origin,color,scale_factor = self.ewarld_sphere[2]*0.1))
            self.addItem(self.generate_detector_object_cube(origin,unit_basis = self.ewarld_sphere[2]*0.1, or_100= or_100, or_001 = or_001))
            self.addItem(self.draw_line_between_two_points(v1 = self.ewarld_sphere[0], v2 = origin,color = [1,0,0,0.3],width = 1))
        return self.items[-1], self.items[-2]

    def show_structure(self):
        self.clear()
        self.detector = None
        self.detector_line = None
        self.cross_points_info = {}
        self.items_subject_to_transformation = []
        self.items_subject_to_recreation = []
        self.RM = np.eye(3)
        self.RRM = np.eye(3)
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        self.theta_x_r = 0
        self.theta_y_r = 0
        self.theta_z_r = 0
        self.theta_SN = 0
        self.theta_SN_r = 0
        self.SN_vec = np.array([0,0,1])
        self.sphere_items_according_to_substrate_and_hkl = {}
        self.line_items_according_to_substrate_and_hkl = {}
        self.recreated_items_according_to_substrate_and_hkl = {}
        if len(self.ewarld_sphere)!=0:
            points_on_circle = self.compute_points_on_3d_circle(center=self.ewarld_sphere[0], v1=np.array([1,0,0]), v2=np.array([0,1,0]), r=self.ewarld_sphere[2],resolution = 100)
            self.addItem(gl.GLLinePlotItem(pos=points_on_circle, width=0.8, color = (0.8,0,0,0.8),antialias=True))
            points_on_circle = self.compute_points_on_3d_circle(center=self.ewarld_sphere[0], v1=np.array([1,0,0]), v2=np.array([0,0,1]), r=self.ewarld_sphere[2],resolution = 100)
            self.addItem(gl.GLLinePlotItem(pos=points_on_circle, width=0.8, color = (0.8,0,0,0.8),antialias=True))
        #self.addItem(self.detector)

        for each_line in self.lines:
            recreated_items = []
            v1, v2, color = each_line
            name = None
            for each_name in self.lines_dict:
                if each_line in self.lines_dict[each_name]:
                    name = each_name
                    break
            self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 3))
            self.items_subject_to_transformation.append(self.items[-1])

            hkl_key_ = self.HKLs_dict[name][self.lines_dict[name].index(each_line)]
            if name not in self.line_items_according_to_substrate_and_hkl:
                self.line_items_according_to_substrate_and_hkl[name] = {tuple(hkl_key_):self.items[-1]}
            else:
                self.line_items_according_to_substrate_and_hkl[name][tuple(hkl_key_)] = self.items[-1]
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
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=True))
                    self.items_subject_to_recreation.append(self.items[-1])
                    recreated_items.append(self.items[-1])
                    self.addItem(gl.GLLinePlotItem(pos=points_on_circle_full_circle, width=0.5, color = (0.8,0.8,0.8,0.8),antialias=True))
                    self.items_subject_to_recreation.append(self.items[-1])
                    recreated_items.append(self.items[-1])
                    self.addItem(self.draw_sphere(each, (0,0,1,1), 0.1))
                    self.items_subject_to_recreation.append(self.items[-1])
                    recreated_items.append(self.items[-1])
                    ki = self.draw_arrow(v1_,[0,0,0], tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    kfs=self.draw_arrow(v1_, each, tip_width=0.1, tip_length_scale=0.02, color=(1,1.,1,1),line_width =0.5)
                    q_vs= self.draw_arrow([0,0,0], each, tip_width=0.1, tip_length_scale=0.02, color=(1.,1.,1,1),line_width =0.5)
                    for kf in kfs:
                        self.addItem(kf)
                        self.items_subject_to_recreation.append(self.items[-1])
                        recreated_items.append(self.items[-1])
                    for q_v in q_vs:
                        self.addItem(q_v)
                        self.items_subject_to_recreation.append(self.items[-1])
                        recreated_items.append(self.items[-1])
                    for each_ki in ki:
                        self.addItem(each_ki)
                    # text = CustomTextItem(*each, 'x', color = QtCore.Qt.red)
                    # text.setGLViewWidget(self)
                    # self.addItem(text)
                if name not in self.recreated_items_according_to_substrate_and_hkl:
                    self.recreated_items_according_to_substrate_and_hkl[name] = {tuple(hkl_key_):recreated_items}
                else:
                    self.recreated_items_according_to_substrate_and_hkl[name][tuple(hkl_key_)] = recreated_items
        for each_line in self.grids:
            v1, v2, color = each_line
            self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 1))
            self.items_subject_to_transformation.append(self.items[-1])
        for each_structure in self.unit_cell_edges:
            for each_unitcell in self.unit_cell_edges[each_structure]:
                v1, v2, color = each_unitcell
                self.addItem(self.draw_line_between_two_points(v1,v2,color, width = 2))
                self.items_subject_to_transformation.append(self.items[-1])
        for each_sphere in self.spheres:
            v1_, color_, scale_factor, _hkl, _name = each_sphere
            self.addItem(self.draw_sphere(v1_, color_, scale_factor)) 
            self.items_subject_to_transformation.append(self.items[-1])
            if _name not in self.sphere_items_according_to_substrate_and_hkl:
                self.sphere_items_according_to_substrate_and_hkl[_name] = {tuple(_hkl): self.items[-1]}
            else:
                self.sphere_items_according_to_substrate_and_hkl[_name][tuple(_hkl)] = self.items[-1]
        if len(self.ewarld_sphere)!=0:
            v1_, color_, scale_factor = self.ewarld_sphere
            self.addItem(self.draw_sphere(v1_, color_, scale_factor,rows=100, cols=100, glOption = 'additive'))
            self.detector_line, self.detector = self.generate_detector_object(origin=-np.array(self.ewarld_sphere[0]))
        labels_axis = ['H','K','L','x','y','z']
        for i,each_arrow in enumerate(self.arrows):
            v1, v2, tip_width, tip_length_scale, color = each_arrow
            items = self.draw_arrow(v1, v2, tip_width, tip_length_scale, color)
            for each in items:
                self.addItem(each)
                if labels_axis[i] in ['H','K','L']:
                    self.items_subject_to_transformation.append(self.items[-1])
            #add axis label
            text = CustomTextItem(*list(np.array(v2)+[0.05,0.05,0.05]),labels_axis[i],color = QtCore.Qt.black, font_size = 13)
            text.setGLViewWidget(self)
            self.addItem(text)
            if labels_axis[i] in ['H','K','L']:
                self.items_subject_to_transformation.append(self.items[-1])
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

    def update_text_item_sym_rods(self):
        texts = [CustomTextItem(*each, font_size = 20) for each in self.text_sym_rods]
        for text in texts:
            text.setGLViewWidget(self)
            self.addItem(text)
        if self.text_item_sym_rods!=None:
            for each in self.text_item_sym_rods:
                try:
                    self.removeItem(each) 
                except:
                    pass
        self.text_item_sym_rods = self.items[-len(texts):]
        self.update()

    def update_structure(self, xyz_dir = True):
        self.apply_xyz_rotation()
        # print('I am here!')
        for each in self.items_subject_to_transformation:
            # each.resetTransform()
            each.rotate(self.theta_x_r, 1,0,0, local = False)
            each.rotate(self.theta_y_r, 0,1,0, local = False)
            each.rotate(self.theta_z_r, 0,0,1, local = False)
        self.apply_SN_rotation()
        for each in self.items_subject_to_transformation:
            # each.resetTransform()
            each.rotate(self.theta_SN_r, *self.SN_vec, local = False)
        self.recal_cross_points(xyz_dir = xyz_dir)
        self.update()
