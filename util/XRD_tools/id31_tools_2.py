import numpy as np
from PyMca5.PyMcaIO import EdfFile
import scipy.ndimage
import scipy.interpolate
import scipy.optimize
from matplotlib.colors import LinearSegmentedColormap

sqrt3 = np.sqrt(3)


class P3_Image:
    def __init__(self, filename):
        edf = EdfFile.EdfFile(filename, 'r')
        self.header = edf.GetHeader(0)
        self.motors = dict()
        motor_mne = self.header['motor_mne'].split()
        motor_pos = self.header['motor_pos'].split()
        for i in xrange(len(motor_mne)):
            self.motors[motor_mne[i]] = float(motor_pos[i])
        self.counters = dict()
        counter_mne = self.header['counter_mne'].split()
        counter_pos = self.header['counter_pos'].split()
        for i in xrange(len(counter_mne)):
            self.counters[counter_mne[i]] = float(counter_pos[i])
        self.img = np.array(edf.GetData(0), dtype=float)


class MPX_Image:
    def __init__(self, filename):
        edf = EdfFile.EdfFile(filename, 'r')
        self.header = edf.GetHeader(0)
        self.motors = dict()
        motor_mne = self.header['motor_mne'].split()
        motor_pos = self.header['motor_pos'].split()
        for i in xrange(len(motor_mne)):
            self.motors[motor_mne[i]] = float(motor_pos[i])
        self.counters = dict()
        counter_mne = self.header['counter_mne'].split()
        counter_pos = self.header['counter_pos'].split()
        for i in xrange(len(counter_mne)):
            self.counters[counter_mne[i]] = float(counter_pos[i])
        self.img = np.array(edf.GetData(0))


class PE_Image:
    def __init__(self, filename):
        edf = EdfFile.EdfFile(filename, 'r')
        self.header = edf.GetHeader(0)
        self.motors = dict()
        motor_mne = self.header['motor_mne'].split()
        motor_pos = self.header['motor_pos'].split()
        for i in xrange(len(motor_mne)):
            self.motors[motor_mne[i]] = float(motor_pos[i])
        self.counters = dict()
        counter_mne = self.header['counter_mne'].split()
        counter_pos = self.header['counter_pos'].split()
        for i in xrange(len(counter_mne)):
            self.counters[counter_mne[i]] = float(counter_pos[i])

        self.img = np.array(edf.GetData(0))
        # detector was mounted rotated by 90deg to the right
        self.img = np.rot90(self.img, 3)


class Imshow_Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        y_pos = int(y+0.5)
        x_pos = int(x+0.5)
        img = self.im.get_array()
        if(x_pos > img.shape[0]-1):
            x_pos = img.shape[0]-1
        if(y_pos > img.shape[1]-1):
            y_pos = img.shape[1]-1
        z = img[y_pos, x_pos]
        return 'x={:.01f}, y={:.01f}, z={:d}'.format(x, y, z)


class coords_formatter:
    def __init__(self, img):
        self.img = img

    def format_coord(self, x, y):
        numrows, numcols = self.img.shape
        col = int(x+0.5)
        row = int(y+0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = self.img[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)


def rad(x):
    return x*np.pi/180.


def deg(x):
    return x*180./np.pi


def get_K_0(E_keV):
    # AA-1
    return 2.*np.pi/12.39854*E_keV

# xy, del-gam conversion for


def get_del(x, cen_x, d):
    return deg(np.arctan((x-cen_x)*0.1720/d))


def get_gam(y, cen_y, d):
    return deg(np.arctan(-(y-cen_y)*0.1720/d))


def get_x(delta, cen_x, d):
    return cen_x + d/0.1720*np.tan(rad(delta))


def get_y(gamma, cen_y, d):
    return cen_y - d/0.1720*np.tan(rad(gamma))


def get_q(k_0, theta, delta, gamma):
    q_x = k_0*(+np.cos(rad(gamma))*np.sin(rad(delta-theta))+np.sin(rad(theta)))
    q_y = k_0*(np.sin(rad(gamma)))
    q_z = k_0*(-np.cos(rad(gamma))*np.cos(rad(delta-theta))+np.cos(rad(theta)))
    return np.array((q_x, q_y, q_z))


def get_q_from_HKL(k_0, H, K, L, a_star, c_star, alpha):
    q = np.zeros(3)
    q[0] = (H+K/2.)*a_star
    q[1] = np.sqrt(3)/2.*K*a_star
    q[2] = L*c_star
    q_Rx = np.cos(rad(-alpha))*q[0]+np.sin(rad(-alpha))*q[1]
    q_Ry = -np.sin(rad(-alpha))*q[0]+np.cos(rad(-alpha))*q[1]

    q[0] = q_Rx
    q[1] = q_Ry

    return q


def get_q_from_HKL_cubic(k_0, H, K, L, a_star, b_star, c_star, alpha):
    q = np.zeros(3)
    q[0] = H*a_star
    q[1] = K*b_star
    q[2] = L*c_star
    q_Rx = np.cos(rad(-alpha))*q[0]+np.sin(rad(-alpha))*q[1]
    q_Ry = -np.sin(rad(-alpha))*q[0]+np.cos(rad(-alpha))*q[1]

    q[0] = q_Rx
    q[1] = q_Ry

    return q


def get_HKL(q, a_star, c_star, alpha):
    q_Rx = np.cos(rad(alpha))*q[0]+np.sin(rad(alpha))*q[1]
    q_Ry = -np.sin(rad(alpha))*q[0]+np.cos(rad(alpha))*q[1]
    H = (q_Rx - q_Ry/sqrt3)/a_star
    K = 2./sqrt3*q_Ry/a_star
    L = q[2]/c_star
    return np.array((H, K, L))


def get_HKL_cubic(q, a_star, b_star, c_star, alpha):
    q_Rx = np.cos(rad(alpha))*q[0]+np.sin(rad(alpha))*q[1]
    q_Ry = -np.sin(rad(alpha))*q[0]+np.cos(rad(alpha))*q[1]
    H = q_Rx/a_star
    K = q_Ry/b_star
    L = q[2]/c_star
    return np.array((H, K, L))


def get_HKL_from_xyth(x, cen_x, y, cen_y, theta, k_0, a_star, c_star, alpha,
                      d):
    delta = get_del(x, cen_x, d)
    gamma = get_gam(y, cen_y, d)
    return get_HKL(get_q(k_0, theta, delta, gamma), a_star, c_star, alpha)


def get_HKL_from_xyth_cubic(x, cen_x, y, cen_y, theta, k_0, a_star, b_star,
                            c_star, alpha, d):
    delta = get_del(x, cen_x, d)
    gamma = get_gam(y, cen_y, d)
    return get_HKL_cubic(get_q(k_0, theta, delta, gamma), a_star, b_star,
                         c_star, alpha)


def _help_func(theta, gamma, q_x, q_z, k_0):
    # all angles in rad
    return (q_z - k_0 * (np.cos(gamma) * np.cos(np.arcsin((q_x / k_0 -
            np.sin(theta) / np.cos(gamma))))-np.cos(theta)))


def get_angles(k_0, H, K, L, a_star, c_star, alpha=0, theta=None):
    q_Rx = a_star*(H+K/2.)
    q_Ry = a_star*(sqrt3/2.*K)
    q_x = np.cos(rad(-alpha))*q_Rx+np.sin(rad(-alpha))*q_Ry
    q_y = -np.sin(rad(-alpha))*q_Rx+np.cos(rad(-alpha))*q_Ry

    th_given = (theta is not None)
    gamma = np.arcsin(q_y/k_0)
    if(not th_given):
        q_z = L*c_star
        A = np.cos(gamma)
        # TODO: this part needs testing
        theta1 = -2*np.arctan((2*q_x/k_0 -np.sqrt(-A**4 +2*A**2*(q_x/k_0)**2 + 2*A**2*(q_z/k_0)**2 + 2*A**2 - (q_x/k_0)**4 - 2*(q_x/k_0)**2*(q_z/k_0)**2 + 2*(q_x/k_0)**2 - (q_z/k_0)**4 + 2*(q_z/k_0)**2 - 1))/(A**2 - (q_x/k_0)**2 - (q_z/k_0)**2 - 2*q_z/k_0 - 1))
        theta2 = -2*np.arctan((2*q_x/k_0 + np.sqrt(-A**4 + 2*A**2*(q_x/k_0)**2 + 2*A**2*(q_z/k_0)**2 + 2*A**2 - (q_x/k_0)**4 - 2*(q_x/k_0)**2*(q_z/k_0)**2 + 2*(q_x/k_0)**2 - (q_z/k_0)**4 + 2*(q_z/k_0)**2 - 1))/(A**2 - (q_x/k_0)**2 - (q_z/k_0)**2 - 2*q_z/k_0 - 1))        
        if(np.abs(theta1) > np.abs(theta2)):
            theta = theta2
        else:
            theta = theta1
    else:
        theta = rad(theta)

    arg = (q_x/k_0-np.sin(theta))/np.cos(gamma)
    delta = np.arcsin(arg) + theta

    if (arg > 1 or arg < -1):
        print('Invalid value ' + str(arg) + ' in arcsin!')
        print('qx = ' + str(q_x) + ', k_0 = ' + str(k_0) + ', theta = ' +
              str(theta) + ', gamma = ' + str(gamma))
        print('H = ' + str(H) + ', K = ' + str(K) + ', L = ' + str(L) +
              ', alpha = ' + str(alpha) + ', theta = ' + str(theta))

    if(th_given):
        q_z = k_0*(-np.cos(gamma)*np.cos(delta-theta)+np.cos(theta))
        LL = q_z/c_star
        return np.array((LL, deg(delta), deg(gamma)))

    return np.array((deg(theta), deg(delta), deg(gamma)))


def get_angles_cubic(k_0, H, K, L, a_star, b_star, c_star, alpha=0,
                     theta=None):
    q_Rx = a_star*H
    q_Ry = a_star*K
    q_x = np.cos(rad(-alpha))*q_Rx+np.sin(rad(-alpha))*q_Ry
    q_y = -np.sin(rad(-alpha))*q_Rx+np.cos(rad(-alpha))*q_Ry

    th_given = (theta is not None)
    gamma = np.arcsin(q_y/k_0)
    if(not th_given):
        q_z = L*c_star
        A = np.cos(gamma)
        # TODO: this part needs testing
        theta1 = -2*np.arctan((2*q_x/k_0 -np.sqrt(-A**4 +2*A**2*(q_x/k_0)**2 + 2*A**2*(q_z/k_0)**2 + 2*A**2 - (q_x/k_0)**4 - 2*(q_x/k_0)**2*(q_z/k_0)**2 + 2*(q_x/k_0)**2 - (q_z/k_0)**4 + 2*(q_z/k_0)**2 - 1))/(A**2 - (q_x/k_0)**2 - (q_z/k_0)**2 - 2*q_z/k_0 - 1))
        theta2 = -2*np.arctan((2*q_x/k_0 + np.sqrt(-A**4 + 2*A**2*(q_x/k_0)**2 + 2*A**2*(q_z/k_0)**2 + 2*A**2 - (q_x/k_0)**4 - 2*(q_x/k_0)**2*(q_z/k_0)**2 + 2*(q_x/k_0)**2 - (q_z/k_0)**4 + 2*(q_z/k_0)**2 - 1))/(A**2 - (q_x/k_0)**2 - (q_z/k_0)**2 - 2*q_z/k_0 - 1))        
        if(np.abs(theta1) > np.abs(theta2)):
            theta = theta2
        else:
            theta = theta1
    else:
        theta = rad(theta)

    arg = (q_x/k_0-np.sin(theta))/np.cos(gamma)
    delta = np.arcsin(arg) + theta

    if (arg > 1 or arg < -1):
        print('Invalid value ' + str(arg) + ' in arcsin!')
        print('qx = ' + str(q_x) + ', k_0 = ' + str(k_0) + ', theta = ' +
              str(theta) + ', gamma = ' + str(gamma))
        print('H = ' + str(H) + ', K = ' + str(K) + ', L = ' + str(L) +
              ', alpha = ' + str(alpha) + ', theta = ' + str(theta))

    if(th_given):
        q_z = k_0*(-np.cos(gamma)*np.cos(delta-theta)+np.cos(theta))
        LL = q_z/c_star
        return np.array((LL, deg(delta), deg(gamma)))

    return np.array((deg(theta), deg(delta), deg(gamma)))


def get_angles_old(k_0, H, K, L, a_star, c_star, alpha, theta=None):
    q_Rx = a_star*(H+K/2.)
    q_Ry = a_star*(sqrt3/2.*K)
    q_x = np.cos(rad(-alpha))*q_Rx+np.sin(rad(-alpha))*q_Ry
    q_y = -np.sin(rad(-alpha))*q_Rx+np.cos(rad(-alpha))*q_Ry

    th_given = (theta is not None)
    gamma = np.arcsin(q_y/k_0)
    if(not th_given):
        q_z = L*c_star
        theta = scipy.optimize.brentq(_help_func, -np.pi*0.48, np.pi*0.48,
                                      (gamma, q_x, q_z, k_0))
    else:
        theta = rad(theta)

    delta = np.arcsin((q_x/k_0-np.sin(theta))/np.cos(gamma))+theta

    if(th_given):
        q_z = k_0*(-np.cos(gamma)*np.cos(delta-theta)+np.cos(theta))
        LL = q_z/c_star
        return np.array((LL, deg(delta), deg(gamma)))

    return np.array((deg(theta), deg(delta), deg(gamma)))


def get_xy_theta(k_0, H, K, L, a_star, c_star, alpha, cen_x, cen_y, d):
    angles = get_angles(k_0, H, K, L, a_star, c_star, alpha)
    x = get_x(angles[1], cen_x, d)
    y = get_y(angles[2], cen_y, d)
    return np.array((x, y, angles[0]))


def get_xy_theta_cubic(k_0, H, K, L, a_star, b_star, c_star, alpha, cen_x,
                       cen_y, d):
    angles = get_angles_cubic(k_0, H, K, L, a_star, b_star, c_star, alpha)
    x = get_x(angles[1], cen_x, d)
    y = get_y(angles[2], cen_y, d)
    return np.array((x, y, angles[0]))


def get_xy_L(k_0, H, K, theta, a_star, c_star, alpha, cen_x, cen_y, d):
    angles = get_angles(k_0, H, K, 0, a_star, c_star, alpha, theta)
    x = get_x(angles[1], cen_x, d)
    y = get_y(angles[2], cen_y, d)
    return np.array((x, y, angles[0]))


def blue_white_red_cmap(white_region=0.1):
    region_2 = (0.5 - white_region)/2.
    region_3 = 0.5 - white_region/2.
    region_4 = 0.5 + white_region/2.
    region_5 = 1. - (0.5 - white_region)/2.

    cdict = {'red':  ((0.0, 0.0, 0.0),
                      (region_2, 0.0, 0.0),
                      (region_3, 1.0, 1.0),
                      (region_4, 1.0, 1.0),
                      (region_5, 1.0, 1.0),
                      (1.0, 0.5, 0.0)),

             'green': ((0.0, 0.0, 0.0),
                       (region_2, 0.0, 0.0),
                       (region_3, 1.0, 1.0),
                       (region_4, 1.0, 1.0),
                       (region_5, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.0, 0.5),
                       (region_2, 1.0, 1.0),
                       (region_3, 1.0, 1.0),
                       (region_4, 1.0, 1.0),
                       (region_5, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
             }

    return LinearSegmentedColormap('BlueRed', cdict)

def draw_heaxagonal_grid(ax, k_0, theta, alpha, d, a_star, c_star, color='k', linewidth=1, draw_axes=True, axis_color='k', cen_x=1024, cen_y=1024):    
    for HH in xrange(-4, 4):
        for KK in xrange(-4, 4):
            xyL1 = get_xy_L(k_0, HH, KK, theta, a_star, c_star, alpha, cen_x, cen_y, d)
            xyL2 = get_xy_L(k_0, HH+1, KK, theta, a_star, c_star, alpha, cen_x, cen_y, d)
            xyL3 = get_xy_L(k_0, HH, KK+1, theta, a_star, c_star, alpha, cen_x, cen_y, d)
            
            ax.plot([xyL1[0], xyL2[0]], [xyL1[1], xyL2[1]], '-', color=color)
            ax.plot([xyL1[0], xyL3[0]], [xyL1[1], xyL3[1]], '-', color=color)
            ax.plot([xyL3[0], xyL2[0]], [xyL3[1], xyL2[1]], '-', color=color)
    if(draw_axes):
        xy_H = get_xy_L(k_0, 1, 0, theta, a_star, c_star, alpha, cen_x, cen_y, d)
        xy_K = get_xy_L(k_0, 0, 1, theta, a_star, c_star, alpha, cen_x, cen_y, d)
        ax.arrow(cen_x, cen_y, (xy_H[0]-cen_x)*0.9, (xy_H[1]-cen_y)*0.9, width=2, head_width=(xy_H[0]-cen_x)*0.05, head_length=(xy_H[0]-cen_x)*0.1, fc=axis_color, ec=axis_color)
        ax.arrow(cen_x, cen_y, (xy_K[0]-cen_x)*0.9, (xy_K[1]-cen_y)*0.9, width=2, head_width=(xy_H[0]-cen_x)*0.05, head_length=(xy_H[0]-cen_x)*0.1, fc=axis_color, ec=axis_color)
