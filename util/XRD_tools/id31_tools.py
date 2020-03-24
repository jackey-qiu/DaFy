import matplotlib.pyplot as plt
import numpy as np
from PyMca5.PyMcaIO import EdfFile
from pyspec import spec
import scipy.ndimage
import scipy.interpolate
import pyspec
from pyspec import spec
from pyspec import fit
import scipy.optimize


sqrt3 = np.sqrt(3)

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
        self.img = np.rot90(self.img, 3) # detector was mounted rotated by 90deg to the right
    def subtract_radial_bg(self):
        line_1 = np.zeros(1024)
        line_2 = np.zeros(1024)
        line_3 = np.zeros(1024)
        line_4 = np.zeros(1024)
        for i in xrange(1024):
            line_1[i] = self.img[1023-i,1024+i]
            line_2[i] = self.img[1023-i, 1023-i]
            line_3[i] = self.img[1024+i,1023-i]
            line_4[i] = self.img[1024+i,1024+i]

        d = np.arange(1024)*np.sqrt(2)
        f_1 = scipy.interpolate.interp1d(d, line_1)
        f_2 = scipy.interpolate.interp1d(d, line_2)
        f_3 = scipy.interpolate.interp1d(d, line_3)
        f_4 = scipy.interpolate.interp1d(d, line_4)
        
        for x in xrange(1024):
            print x
            for y in xrange(1024):
                self.img[1023-y,1024+x] -= f_1(np.sqrt(x**2+y**2))
                self.img[1023-y, 1023-x] -= f_2(np.sqrt(x**2+y**2))
                self.img[1024+y,1023-x] -= f_3(np.sqrt(x**2+y**2))
                self.img[1024+y,1024+x] -= f_4(np.sqrt(x**2+y**2))





class Imshow_Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        y_pos = int(y+0.5)
        x_pos = int(x+0.5)
        img = self.im.get_array()
        if(x_pos > img.shape[0]-1): x_pos = img.shape[0]-1
        if(y_pos > img.shape[1]-1): y_pos = img.shape[1]-1
        z = img[y_pos, x_pos]
        return 'x={:.01f}, y={:.01f}, z={:d}'.format(x, y, z)

def rad(x):
    return x*np.pi/180.
def deg(x):
    return x*180./np.pi

def get_K_0(E_keV): #AA-1
    return 2.*np.pi/12.39854*E_keV

def get_del(x, d):
    return deg(np.arctan((x-1024)*0.2/d))
def get_gam(y, d):
    return deg(np.arctan(-(y-1024)*0.2/d))
def get_x(delta, d):
    return 1024+d/0.2*np.tan(rad(delta))
def get_y(gamma, d):
    return 1024-d/0.2*np.tan(rad(gamma))

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
    

def get_HKL(q, a_star, c_star, alpha):
    q_Rx = np.cos(rad(alpha))*q[0]+np.sin(rad(alpha))*q[1]
    q_Ry = -np.sin(rad(alpha))*q[0]+np.cos(rad(alpha))*q[1]
    H = (q_Rx - q_Ry/sqrt3)/a_star
    K = 2./sqrt3*q_Ry/a_star
    L = q[2]/c_star
    return np.array((H, K, L))

def get_HKL_from_xyth(x, y, theta, k_0, a_star, c_star, alpha, d):
    delta = get_del(x, d)
    gamma = get_gam(y, d)
    return get_HKL(get_q(k_0, theta, delta, gamma), a_star, c_star, alpha)


def _help_func(theta, gamma, q_x, q_z, k_0):
    # all angles in rad
    return q_z - k_0*(np.cos(gamma)*np.cos(np.arcsin((q_x/k_0-np.sin(theta)/np.cos(gamma))))-np.cos(theta))

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
        theta1 = -2*np.arctan((2*q_x/k_0 - np.sqrt(-A**4 + 2*A**2*(q_x/k_0)**2 + 2*A**2*(q_z/k_0)**2 + 2*A**2 - (q_x/k_0)**4 - 2*(q_x/k_0)**2*(q_z/k_0)**2 + 2*(q_x/k_0)**2 - (q_z/k_0)**4 + 2*(q_z/k_0)**2 - 1))/(A**2 - (q_x/k_0)**2 - (q_z/k_0)**2 - 2*q_z/k_0 - 1))
        theta2 = -2*np.arctan((2*q_x/k_0 + np.sqrt(-A**4 + 2*A**2*(q_x/k_0)**2 + 2*A**2*(q_z/k_0)**2 + 2*A**2 - (q_x/k_0)**4 - 2*(q_x/k_0)**2*(q_z/k_0)**2 + 2*(q_x/k_0)**2 - (q_z/k_0)**4 + 2*(q_z/k_0)**2 - 1))/(A**2 - (q_x/k_0)**2 - (q_z/k_0)**2 - 2*q_z/k_0 - 1))        
        if(np.abs(theta1) > np.abs(theta2)):
            theta = theta2
        else:
            theta = theta1
    else:
        theta = rad(theta)

    delta = np.arcsin((q_x/k_0-np.sin(theta))/np.cos(gamma))+theta
    
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
        theta = scipy.optimize.brentq(_help_func, -np.pi*0.48, np.pi*0.48, (gamma, q_x, q_z, k_0))
    else:
        theta = rad(theta)

    delta = np.arcsin((q_x/k_0-np.sin(theta))/np.cos(gamma))+theta
    
    if(th_given):
        q_z = k_0*(-np.cos(gamma)*np.cos(delta-theta)+np.cos(theta))
        LL = q_z/c_star
        return np.array((LL, deg(delta), deg(gamma)))
    
    return np.array((deg(theta), deg(delta), deg(gamma)))

def get_xy_theta(k_0, H, K, L, a_star, c_star, alpha, d):
    angles = get_angles(k_0, H, K, L, a_star, c_star, alpha)
    x = get_x(angles[1], d)
    y = get_y(angles[2], d)
    return np.array((x, y, angles[0]))

def get_xy_L(k_0, H, K, theta, a_star, c_star, alpha, d):
    angles = get_angles(k_0, H, K, 0, a_star, c_star, alpha, theta)
    x = get_x(angles[1], d)
    y = get_y(angles[2], d)
    return np.array((x, y, angles[0]))



def draw_heaxagonal_grid_2(ax, k_0, theta, alpha, d, a_star, c_star, color='k', linewidth=1, draw_axes=True, axis_color='k', origin=(1024, 1024)):    
    for HH in xrange(-10, 10):
        for KK in xrange(-10, 10):
            xyL1 = get_xy_L(k_0, HH, KK, theta, a_star, c_star, alpha, d)
            xyL2 = get_xy_L(k_0, HH+1, KK, theta, a_star, c_star, alpha, d)
            xyL3 = get_xy_L(k_0, HH, KK+1, theta, a_star, c_star, alpha, d)
            
            ax.plot([xyL1[0], xyL2[0]], [xyL1[1], xyL2[1]], '-', color=color)
            ax.plot([xyL1[0], xyL3[0]], [xyL1[1], xyL3[1]], '-', color=color)
            ax.plot([xyL3[0], xyL2[0]], [xyL3[1], xyL2[1]], '-', color=color)
    if(draw_axes):
        #origin = (1024, 1024)
        xy_H = get_xy_L(k_0, 1, 0, theta, a_star, c_star, alpha, d)
        xy_K = get_xy_L(k_0, 0, 1, theta, a_star, c_star, alpha, d)
        ax.arrow(origin[0], origin[1], (xy_H[0]-origin[0])*0.9, (xy_H[1]-origin[1])*0.9, width=2, head_width=(xy_H[0]-origin[0])*0.05, head_length=(xy_H[0]-origin[0])*0.1, fc=axis_color, ec=axis_color)
        ax.arrow(origin[0], origin[1], (xy_K[0]-origin[0])*0.9, (xy_K[1]-origin[1])*0.9, width=2, head_width=(xy_H[0]-origin[0])*0.05, head_length=(xy_H[0]-origin[0])*0.1, fc=axis_color, ec=axis_color)


def draw_hexagonal_grid(ax, k_0, a_star, c_star, alpha, d, draw_axes=True, color='k', draw_label=False, label_color='w', draw_reconstruction=False, delta_rec=0.038, show_only_reconstruction=False, linewidth=1):
    H_pars = get_xy_theta(k_0, 1, 0, 0, a_star, c_star, alpha, d)
    K_pars = get_xy_theta(k_0, 0, 1, 0, a_star, c_star, alpha, d)
    cen = np.array((1024, 1024))
    arrow_length = np.sqrt((cen[0]-K_pars[0])**2 + (cen[1]-K_pars[1])**2)
    head_length = arrow_length*0.1
    arrow_length *= 0.9
        
    if(draw_axes):
        ax.arrow(cen[0], cen[1], (H_pars[0]-cen[0])*0.9, (H_pars[1]-cen[1])*0.9, width=2, head_width=head_length, head_length=head_length, fc=color, ec=color)
        ax.arrow(cen[0], cen[1], (K_pars[0]-cen[0])*0.9, (K_pars[1]-cen[1])*0.9, width=2, head_width=head_length, head_length=head_length, fc=color, ec=color)
    if(draw_label):        
        H_pars_label = get_xy_theta(k_0, 1, 0, 0, a_star, c_star, alpha+20, d)
        K_pars_label = get_xy_theta(k_0, 0, 1, 0, a_star, c_star, alpha+20, d)
        dir_H = (H_pars_label[:2]-cen)/2.
        dir_K = (K_pars_label[:2]-cen)/2.
        ax.text(cen[0]+dir_H[0], cen[1]+dir_H[1], 'H', color=label_color, fontsize=20)
        ax.text(cen[0]+dir_K[0], cen[1]+dir_K[1], 'K', color=label_color, fontsize=20)

    r_Hx = (H_pars[0]-cen[0])
    r_Hy = (H_pars[1]-cen[1])
    r_Kx = (K_pars[0]-cen[0])
    r_Ky = (K_pars[1]-cen[1])

    dHK = delta_rec/np.sqrt(3)



    for HH in xrange(-10, 10):
        for KK in xrange(-10, 10):
            p_x = cen[0]+HH*r_Hx+KK*r_Kx
            p_y = cen[0]+HH*r_Hy+KK*r_Ky
            if(not show_only_reconstruction):
                ax.plot([p_x, p_x+r_Hx], [p_y, p_y+r_Hy], '-', color=color, linewidth=linewidth)
                ax.plot([p_x, p_x+r_Kx], [p_y, p_y+r_Ky], '-', color=color, linewidth=linewidth)
                ax.plot([p_x+r_Kx, p_x+r_Hx], [p_y+r_Ky, p_y+r_Hy], '-', color=color, linewidth=linewidth)
            
            if(draw_reconstruction):
                dH = np.array([1,  2,  1, -1, -2, -1, 1])*dHK
                dK = np.array([1, -1, -2, -1,  1,  2, 1])*dHK
                p_hex_x = []
                p_hex_y = []
                for i in xrange(len(dH)):
                    p_hex_x.append(p_x+dH[i]*r_Hx+dK[i]*r_Kx)
                    p_hex_y.append(p_y+dH[i]*r_Hy+dK[i]*r_Ky)
                ax.plot(p_hex_x, p_hex_y, '-', color='K', linewidth=linewidth)

                    

def get_HK_in_BZ(BZ=1):
    H=BZ
    K=0
    Hs = []
    Ks = []
    for i in xrange(6):
        for j in xrange(BZ):
            if(i == 0):
                H -= 1
                K += 1
            elif(i == 1):
                H -= 1
            elif(i == 2):
                K -= 1
            elif(i == 3):
                H += 1
                K -= 1
            elif(i == 4):
                H += 1
            elif(i == 5):
                K += 1
            Hs.append(H)
            Ks.append(K)
    return (np.array(Hs), np.array(Ks))
    


class peak_params:
    def __init__(self):
        self.COM = np.zeros(2)
        self.max_pos = np.zeros(2)
        self.max = 0
        self.FWHM_para = 0
        self.FWHM_perp = 0
        self.integrated_intensity = 0
    def Print(self):
        print 'Parameters of peak'
        print '------------------'
        print 'COM(x,y) = (%.1f, %.1f)' % (self.COM[0], self.COM[1])
        print 'Max_pos(x,y) = (%.1f, %.1f)' % (self.max_pos[0], self.max_pos[1])
        print 'Max = %.3e' % (self.max)
        print 'Integrated Intensity = %.3e' % (self.integrated_intensity)
        print 'FWHM_para = %.3f' % (self.FWHM_para)
        print 'FWHM_perp = %.3f' % (self.FWHM_perp)

def get_max_pos(img, x, y, dxy=40):
    x = int(x+0.5)
    y = int(y+0.5)
    sub_img = img[y-dxy:y+dxy, x-dxy:x+dxy].copy()    
    max_pos = scipy.ndimage.measurements.maximum_position(sub_img)
    x_max = x-dxy+max_pos[1]
    y_max = y-dxy+max_pos[0]
    return np.array((x_max, y_max))

def peak_analysis(img, x, y, dxy=40, interpolation_points=80, show_plots=False, center_on_max=True, calculate_width=False):
    x = int(x+0.5)
    y = int(y+0.5)
    
    sub_img = img[y-dxy:y+dxy, x-dxy:x+dxy].copy()    
    params = peak_params()
    
    # calculate COM
    COM = scipy.ndimage.measurements.center_of_mass(sub_img)
    max_pos = scipy.ndimage.measurements.maximum_position(sub_img)
    
    #TODO: check if we want to use the max pixel or the COM for the width measurements
    if(center_on_max):
        used_peak_pos_small = np.array((max_pos[1], max_pos[0])) 
        used_peak_pos_large = np.array((x-dxy+max_pos[1], y-dxy+max_pos[0]))
    else:
        used_peak_pos_small = np.array((COM[1], COM[0])) 
        used_peak_pos_large = np.array((x-dxy+COM[1], y-dxy+COM[0]))        

    params.COM = np.array((x-dxy+COM[1], y-dxy+COM[0]))
    params.max_pos = np.array((x-dxy+max_pos[1], y-dxy+max_pos[0]))
    
    r_x_para = 1024-used_peak_pos_large[0]
    r_y_para = 1024-used_peak_pos_large[1]
    r_mag = np.sqrt(r_x_para**2+r_y_para**2)
    r_x_para /= r_mag
    r_y_para /= r_mag
    
    if(r_x_para == 0 or r_y_para == 0):
        r_x_perp = r_y_para
        r_y_perp = r_x_para
    else:
        r_x_perp = 1./r_x_para
        r_y_perp = -1./r_y_para
        r_mag = np.sqrt(r_x_perp**2+r_y_perp**2)
        r_x_perp /= r_mag
        r_y_perp /= r_mag
    
    # subtract background
    sub_img_bg = img[y-2*dxy:y+2*dxy, x-2*dxy:x+2*dxy].copy()
    xi = np.arange(4*dxy)
    yi = np.arange(4*dxy)
    func_bg = scipy.interpolate.RectBivariateSpline(xi, yi, sub_img_bg)

    d_vals = np.arange(4*dxy)-4*dxy/2.
    f_bg1 = np.zeros(len(d_vals))    
    f_bg2 = np.zeros(len(d_vals))    
    f_bg = np.zeros(len(d_vals))    

    p_1_x = np.sqrt(2)*dxy*r_x_perp + 2*dxy
    p_1_y = np.sqrt(2)*dxy*r_y_perp + 2*dxy

    p_2_x = -np.sqrt(2)*dxy*r_x_perp + 2*dxy
    p_2_y = -np.sqrt(2)*dxy*r_y_perp + 2*dxy

    line_p1_x = []
    line_p1_y = []
    line_p2_x = []
    line_p2_y = []
    
    for i in xrange(len(d_vals)):        
        f_bg1[i] = func_bg(p_1_x+d_vals[i]*r_x_para, p_1_y+d_vals[i]*r_y_para)
        f_bg2[i] = func_bg(p_2_x+d_vals[i]*r_x_para, p_2_y+d_vals[i]*r_y_para)
        f_bg[i] =  (f_bg1[i]+f_bg2[i])/2. 
   

        #f_bg[i] /= 2.
        
        line_p1_x.append(p_1_x+d_vals[i]*r_x_para)
        line_p1_y.append(p_1_y+d_vals[i]*r_y_para)
        line_p2_x.append(p_2_x+d_vals[i]*r_x_para)
        line_p2_y.append(p_2_y+d_vals[i]*r_y_para)
    
    if(show_plots):
        plt.figure()
        plt.plot(f_bg1)
        plt.plot(f_bg2)
        
        fig = plt.figure()
        plt.title('Image with Background')
        ax = fig.add_subplot(111)
        im = ax.imshow(sub_img, interpolation='None')
        ax.plot([used_peak_pos_small[0]], [used_peak_pos_small[1]], 'wd')
        ax.format_coord = Imshow_Formatter(im)
        plt.show()
    
    dummy, bg_arr = np.meshgrid(f_bg, f_bg)
    bg_arr_rot = scipy.ndimage.interpolation.rotate(bg_arr, deg(np.arctan2(r_x_para, r_y_para)), reshape=False)  
    bg_arr_rot_cut = bg_arr_rot[dxy:3*dxy, dxy:3*dxy]
    sub_img -= bg_arr_rot_cut
        
    params.max = scipy.ndimage.measurements.maximum(sub_img)
    params.integrated_intensity = np.sum(sub_img)
    
    

    if(show_plots):
        fig = plt.figure()
        plt.title('Background subtracted')
        ax = fig.add_subplot(111)
        im = ax.imshow(sub_img, interpolation='None')
        ax.plot([used_peak_pos_small[0]], [used_peak_pos_small[1]], 'wd')
        ax.format_coord = Imshow_Formatter(im)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(sub_img_bg, interpolation='None')
        ax.plot(line_p1_x, line_p1_y, 'w')
        ax.plot(line_p2_x, line_p2_y, 'w')
        ax.format_coord = Imshow_Formatter(im)
        
    # calculate peak width
    if(calculate_width):
        xi = np.arange(2*dxy)
        yi = np.arange(2*dxy)
        func = scipy.interpolate.RectBivariateSpline(xi, yi, sub_img)
        
        d_vals = np.arange(interpolation_points)-int(interpolation_points/2.)
        d_x_para = np.zeros(len(d_vals))
        d_y_para = np.zeros(len(d_vals))
        d_ints_para = np.zeros(len(d_vals))
        d_x_perp = np.zeros(len(d_vals))
        d_y_perp = np.zeros(len(d_vals))
        d_ints_perp = np.zeros(len(d_vals))
    
        for i in xrange(len(d_vals)):
            d_x_para[i] = used_peak_pos_small[0]+d_vals[i]*r_x_para
            d_y_para[i] = used_peak_pos_small[1]+d_vals[i]*r_y_para
            d_ints_para[i] = func(used_peak_pos_small[0]+d_vals[i]*r_x_para, used_peak_pos_small[1]+d_vals[i]*r_y_para)
            d_x_perp[i] = used_peak_pos_small[0]+d_vals[i]*r_x_perp
            d_y_perp[i] = used_peak_pos_small[1]+d_vals[i]*r_y_perp
            d_ints_perp[i] = func(used_peak_pos_small[0]+d_vals[i]*r_x_perp, used_peak_pos_small[1]+d_vals[i]*r_y_perp)
            
        try:
            df_para = fit.fit(x=d_vals, y=d_ints_para, funcs=[pyspec.fitfuncs.gauss, pyspec.fitfuncs.linear])
            df_para.run()   
            df_perp = fit.fit(x=d_vals, y=d_ints_perp, funcs=[pyspec.fitfuncs.gauss, pyspec.fitfuncs.linear])
            df_perp.run()  
        
            params.FWHM_para = 2.*np.sqrt(2.*np.log(2.))*df_para.result[1]
            params.FWHM_perp = 2.*np.sqrt(2.*np.log(2.))*df_perp.result[1]
    
            
            if(show_plots):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                im = ax.imshow(sub_img, interpolation='None')
                ax.plot([used_peak_pos_small[0]], [used_peak_pos_small[1]], 'wd')
                ax.plot(d_x_para, d_y_para, 'r-')
                ax.plot(d_x_perp, d_y_perp, 'b-')
                ax.format_coord = Imshow_Formatter(im)
            
                plt.figure()
                plt.plot(d_vals, d_ints_para, 'r-')
                plt.plot(d_vals, d_ints_perp, 'b-')
                plt.plot(d_vals, pyspec.fitfuncs.gauss(d_vals, df_para.result[0:3])+pyspec.fitfuncs.linear(d_vals, df_para.result[3:]), 'r--')   
                plt.plot(d_vals, pyspec.fitfuncs.gauss(d_vals, df_perp.result[0:3])+pyspec.fitfuncs.linear(d_vals, df_perp.result[3:]), 'b--')   
        except:
            return params

    return params    

    
    