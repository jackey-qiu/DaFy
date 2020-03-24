import scipy.optimize as opt
from scipy import signal
try:
    import ConfigParser as configparser
except:
    import configparser
try:
    import cv2#install opencv-python if you want to have a boost mapping for l scan
except:
    pass
import matplotlib
# matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from util.UtilityFunctions import collect_args
from ctr_corr import ctr_data
import sys, copy
from numpy import dtype
from scipy.interpolate import griddata
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
# import p23_tools_debug as p23
import subprocess
from DataFilterPool import *
from util.Reciprocal_space_tools.HKLVlieg import Crystal, printPos, UBCalculator, VliegAngles, printPos_prim, vliegDiffracAngles
from PyMca5.PyMcaPhysics import SixCircle
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from numpy.matlib import repmat
from numpy.linalg import pinv
from matplotlib import pyplot
from scipy import misc
from ctr_corr import ctr_data
import matplotlib.patches as patches
import time
from nexusformat.nexus import *

if (sys.version_info > (3, 0)):
    raw_input = input

def gauss(x, x0, sig, amp):
    return amp*np.exp(-(x-x0)**2/2./sig**2)

def lor(x, x0, FWHM, amp):
    return amp*FWHM/((x-x0)**2+FWHM**2/4)

def pvoigt2(x, x0, FWHM, amp, lorfact):
    w = FWHM/2.
    return amp*(lorfact/(1+((x-x0)/w)**2)+(1.-lorfact)*np.exp(-np.log(2)*((x-x0)/w)**2))

def pvoigt(x, x0, FWHM, area, lfrac):
    return area / FWHM / ( lfrac*np.pi/2 + (1-lfrac)*np.sqrt(np.pi/4/np.log(2)) ) * ( lfrac / (1 + 4*((x-x0)/FWHM)**2) + (1-lfrac)*np.exp(-4*np.log(2)*((x-x0)/FWHM)**2) )

def model2(x, x0, FWHM, amp, bg_slope, bg_offset):
    return lor(x, x0, FWHM, amp) + x*bg_slope*0 + bg_offset

def model3(x, x0, FWHM, amp, bg_slope, bg_offset):
    sig = FWHM/2.35482
    return gauss(x, x0, sig, amp) + x*bg_slope*0 + bg_offset

def model(x, x0, FWHM, area, lfrac, bg_slope, bg_offset):
    return pvoigt(x, x0, FWHM, area, lfrac) + x*bg_slope + bg_offset

def calculate_UB_matrix_p23(lattice_constants, energy, or0_angles, or1_angles,or0_hkl,or1_hkl):
    return p23.cal_UB(lattice_constants, energy, or0_angles, or1_angles,or0_hkl, or1_hkl)

def normalize_img_intensity(img, q_grid,mask_img, mask_profile, cen, offset, direction = 'horizontal'):
    if direction == 'horizontal':
        cut_mask = np.sum(mask_img[cen-offset:cen+offset+1,:], axis=0)
        cut_img = np.sum(img[cen-offset:cen+offset+1,:],axis=0)
        cut_img = cut_img/cut_mask
        cut_img = cut_img[mask_profile]
        cut_q = q_grid[0,mask_profile]

        return cut_img[~cut_img_nan],cut_q[~cut_img_nan]
    elif direction == 'vertical':
        cut_mask = np.sum(mask_img[:,cen-offset:cen+offset+1], axis=1)
        cut_img = np.sum(img[:,cen-offset:cen+offset+1],axis=1)
        cut_img = cut_img/cut_mask
        cut_img = cut_img[mask_profile]
        cut_q = q_grid[mask_profile,0]
        #now remove nan values
        cut_img_nan = cut_img == np.nan
        return cut_img[~cut_img_nan],cut_q[~cut_img_nan]

def sawtooth_pot(x, cycles=2, x_offset = 0, y_offset=0, ampt = 0.4):
    cycles = int(cycles)
    x_offset = int(x_offset)
    return signal.sawtooth(2*np.pi*cycles*(x-x_offset),ampt)+y_offset

def triang_(y,phase,length,y_offset,amplitude):
    x = np.array(range(len(y)))
    y = np.array(y)
    phase, length = int(phase),int(length)
    alpha=(amplitude)/(length/2)
    y_cal = -amplitude/2+amplitude*((x-phase)%length==length/2) \
            +alpha*((x-phase)%(length/2))*((x-phase)%length<=length/2) \
            +(amplitude-alpha*((x-phase)%(length/2)))*((x-phase)%length>length/2)\
            +y_offset
    return y_cal

def triang(y,phase,length,y_offset,amplitude):
    x = np.array(range(len(y)))
    y = np.array(y)
    phase, length = int(phase),int(length)
    alpha=(amplitude)/(length/2)
    y_cal = -amplitude/2+amplitude*((x-phase)%length==length/2) \
            +alpha*((x-phase)%(length/2))*((x-phase)%length<=length/2) \
            +(amplitude-alpha*((x-phase)%(length/2)))*((x-phase)%length>length/2)\
            +y_offset
    return np.abs(y_cal - y)

def fit_pot_profile(x, y, show_fig = False):
    x = np.array(x)
    y = np.array(y)
    max_y, min_y = y.max(), y.min()
    y_offset = (max_y+min_y)/2
    ampt = (max_y-min_y)
    # print(ampt,y_offset)
    y_partial = y[0:200]
    x_partial = x[0:200]
    phase = np.argmin(y_partial)
    length = abs(np.argmin(y_partial)-np.argmax(y_partial))*2
    try:
        popt, pcov = opt.curve_fit(triang, y_partial,y_partial*0, p0=[phase,length,y_offset,ampt],bounds =([phase-10,length-10,y_offset-0.0005,ampt-0.0005],[phase+10,length+10,y_offset+0.0005,ampt+0.0005]),max_nfev = 10000)
        # print(popt)
        if show_fig:
            plt.plot(x,y,'or')
            plt.plot(x,triang_(y,*popt),'g-')
            # print(sawtooth_pot(x_partial,*popt))
            plt.show()
        return triang_(y,*popt)
    except:
        return y_partial
    
#engine function to subtraction background
def backcor(n,y,ord_cus,s,fct):
    # Rescaling
    N = len(n)
    index = np.argsort(n)
    n=np.array([n[i] for i in index])
    y=np.array([y[i] for i in index])
    maxy = max(y)
    dely = (maxy-min(y))/2.
    n = 2. * (n-n[N-1]) / float(n[N-1]-n[0]) + 1.
    n=n[:,np.newaxis]
    y = (y-maxy)/dely + 1

    # Vandermonde matrix
    p = np.array(range(ord_cus+1))[np.newaxis,:]
    T = repmat(n,1,ord_cus+1) ** repmat(p,N,1)
    Tinv = pinv(np.transpose(T).dot(T)).dot(np.transpose(T))

    # Initialisation (least-squares estimation)
    a = Tinv.dot(y)
    z = T.dot(a)

    # Other variables
    alpha = 0.99 * 1/2     # Scale parameter alpha
    it = 0                 # Iteration number
    zp = np.ones((N,1))         # Previous estimation

    # LEGEND
    while np.sum((z-zp)**2)/np.sum(zp**2) > 1e-10:
        it = it + 1        # Iteration number
        zp = z             # Previous estimation
        res = y - z        # Residual
        # Estimate d
        if fct=='sh':
            d = (res*(2*alpha-1)) * (abs(res)<s) + (-alpha*2*s-res) * (res<=-s) + (alpha*2*s-res) * (res>=s)
        elif fct=='ah':
            d = (res*(2*alpha-1)) * (res<s) + (alpha*2*s-res) * (res>=s)
        elif fct=='stq':
            d = (res*(2*alpha-1)) * (abs(res)<s) - res * (abs(res)>=s)
        elif fct=='atq':
            d = (res*(2*alpha-1)) * (res<s) - res * (res>=s)
        else:
            pass
        # Estimate z
        a = Tinv.dot(y+d)   # Polynomial coefficients a
        z = T.dot(a)            # Polynomial

    #z=np.array([(z[list(index).index(i)]-1)*dely+maxy for i in range(len(index))])
    #print(index)
    z=(z-1)*dely+maxy
    #t7=time.time()
    #print((t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))
    #
    #print((t2-t1,t3-t2,t4-t3,t5-t4,t6-t5))
    return z,a,it,ord_cus,s,fct

class XRD_Peak_Fitting(object):
    def __init__(self, img, cen, kwarg, model = model):
        self.img = img
        #self.mask = mask
        self.peak_center = cen
        self.previous_peak_center = cen
        self.prim_beam_pot = cen
        self.model = model
        self.first_frame = True
        self.q_ip = None #note this is gridded q
        self.q_oop = None# note this is gridded q
        for key in kwarg:
            setattr(self, key, kwarg[key])
        self.fit_data = {'hor':{'x':[],'y':[]},'ver':{'x':[],'y':[]}}
        self.fit_results = {'hor':[],'ver':[]}
        self.fit_results_plot = {'hor':[],'ver':[]}
        self.peak_intensity = 0
        self.fit_status = False
        #self.fit()

    @property
    def grid_q_ip(self):
        return self.q_ip

    @grid_q_ip.setter
    def grid_q_ip(self,grid_q_ip):
        self.q_ip = grid_q_ip

    @property
    def grid_q_oop(self):
        return self.q_oop

    @grid_q_oop.setter
    def grid_q_oop(self,grid_q_oop):
        self.q_oop = grid_q_oop

    def update_peak_intensity(self,intensity):
        self.peak_intensity = intensity

    def get_peak_width(self):
        # peak_width={'hor':0, 'ver':0}
        hor = int(self.fit_results['hor'][0][1]*2/abs(self.q_ip[0,1]-self.q_ip[0,2]))
        ver = int(self.fit_results['ver'][0][1]*2/abs(self.q_oop[0,1]-self.q_oop[1,1]))
        # hor = int(self.fit_p0['hor'][1]*2/abs(self.q_ip[0,1]-self.q_ip[0,2]))
        # ver = int(self.fit_p0['ver'][1]*2/abs(self.q_oop[0,1]-self.q_oop[1,1]))
        return {'hor':hor,'ver':ver}

    def reset_fit(self,img, check = False, level = 0.05, **kwarg):
        self.first_frame = False
        self.img = img
        check_result = self.fit(check, level)
        return check_result

    def initiat_p0_and_bounds(self):
        try:
            ip_q_cen= self.q_ip[0,self.peak_center[1]]
            oop_q_cen=self.q_oop[self.peak_center[0],0]
            self.fit_bounds['hor'][0][0] = ip_q_cen - 0.2
            self.fit_bounds['hor'][1][0] = ip_q_cen + 0.2
            self.fit_bounds['ver'][0][0] = oop_q_cen - 0.2
            self.fit_bounds['ver'][1][0] = oop_q_cen + 0.2
            self.fit_p0['hor'][0] = ip_q_cen
            self.fit_p0['ver'][0] = oop_q_cen
            self.fit_p0_2['hor'][0] = ip_q_cen
            self.fit_p0_2['ver'][0] = oop_q_cen
        except:
            pass

    def update_bounds(self, cen_oop, cen_ip):
        if cen_oop>self.fit_bounds['ver'][1][0]:
            self.fit_bounds['ver'][1][0]=cen_oop+0.1
        elif cen_oop<self.fit_bounds['ver'][0][0]:
            self.fit_bounds['ver'][0][0]=cen_oop-0.1

        if cen_ip>self.fit_bounds['hor'][1][0]:
            self.fit_bounds['hor'][1][0]=cen_ip+0.1
        elif cen_ip<self.fit_bounds['hor'][0][0]:
            self.fit_bounds['hor'][0][0]=cen_ip-0.1

    def cut_profile_from_2D_img_around_center(self, img, cut_offset = {'hor':10, 'ver':20}, data_range_offset = {'hor':50, 'ver':50}, center_index = None, sum_result = True):

        def _cut_profile_from_2D_img(img, cut_range, cut_direction, sum_result=True):
            if cut_direction=='horizontal':
                if sum_result:
                    return np.sum(img[cut_range[0]:cut_range[1],:],axis = 0)
                else:
                    return np.average(img[cut_range[0]:cut_range[1],:],axis = 0)

            elif cut_direction=='vertical':
                if sum_result:
                    return np.sum(img[:,cut_range[0]:cut_range[1]],axis = 1)
                else:
                    return np.average(img[:,cut_range[0]:cut_range[1]],axis = 1)
        size = img.shape
        f = lambda x, y: [max([x-y,0]), x+y]
        # print(center_index)
        try:
            if center_index ==None:
                center_index = [int(each/2) for each in size]
            else:
                pass
        except:
            pass
        data_range = {'hor':f(center_index[1],data_range_offset['hor']),'ver':f(center_index[0], data_range_offset['ver'])}
        cut_range = {'hor': f(center_index[0],cut_offset['hor']),'ver': f(center_index[1], cut_offset['ver'])}
        # print(data_range['hor'])
        cut = {'hor':_cut_profile_from_2D_img(img, cut_range['hor'], cut_direction ='horizontal', sum_result = sum_result)[data_range['hor'][0]:data_range['hor'][1]],
                'ver':_cut_profile_from_2D_img(img, cut_range['ver'], cut_direction ='vertical', sum_result = sum_result)[data_range['ver'][0]:data_range['ver'][1]]}
        return cut

    def fit_tweak(self, fig_handle, plot_fun, plot_lib):
        data = copy.deepcopy(plot_lib['data'])
        #self.img[self.mask ==0]=0
        fit_p0 = self.fit_p0_2
        center_index = copy.deepcopy(self.peak_center)
        cut_offset_temp =dict([(key,value[1]) for key, value in self.cut_offset.items()])
        data_range_offset_temp =dict([(key,value[1]) for key, value in self.data_range_offset.items()])
        fit_data_temp = copy.deepcopy(self.fit_data)
        update = False
        offset = [0,0]
        print('Starting the tweak...')
        while update==False:
            #after each move reset the offset back to 0
            offset = [0, 0]
            center_offset = raw_input("left(l)/right(r)/up(u)/down(d) + pix_number.\
                                      \ne.g. l5 move center towards left by 5 pixes.\
                                      \nType any key to accept current point.\
                                      \nYour input is:") or 'r0'
            opt_lib = {"l":"-", "r":"+", "u":"-", "d":"+"}
            if center_offset[0] in ["l", "r"]:
                offset[1] = eval(center_offset.replace(center_offset[0],opt_lib[center_offset[0]]))
            elif center_offset[0] in ["u", "d"]:
                offset[0] = eval(center_offset.replace(center_offset[0],opt_lib[center_offset[0]]))
            else:
                pass
            if offset ==[0,0]:
                accept_it = 'y'
            else:
                accept_it = 'n'
            center_index += np.array(offset)
            print(offset, center_index)
            cut = self.cut_profile_from_2D_img_around_center(self.img,cut_offset_temp,data_range_offset_temp, center_index, sum_result = True)
            cut_mask = self.cut_profile_from_2D_img_around_center(self.mask,cut_offset_temp,data_range_offset_temp, center_index, sum_result = False)
            cut_ip_q = self.cut_profile_from_2D_img_around_center(self.q_ip,cut_offset_temp, data_range_offset_temp, center_index, sum_result = False)['hor']
            cut_oop_q = self.cut_profile_from_2D_img_around_center(self.q_oop,cut_offset_temp, data_range_offset_temp, center_index, sum_result = False)['ver']
            #normalize the dead pixel contribution
            cut = {'hor':cut['hor']/cut_mask['hor'],'ver':cut['ver']/cut_mask['ver']}
            #check nan and inf values
            index_used = {'hor':~(np.isnan(cut['hor'])|np.isinf(cut['hor'])), 'ver':~(np.isnan(cut['ver'])|np.isinf(cut['ver']))}
            fit_ip,fom_ip = opt.curve_fit(f=self.model, xdata=cut_ip_q[index_used['hor']], ydata=cut['hor'][index_used['hor']], p0 = fit_p0['hor'], bounds = self.fit_bounds['hor'], max_nfev = 10000)
            fit_oop,fom_oop = opt.curve_fit(f=self.model, xdata=cut_oop_q[index_used['ver']], ydata=cut['ver'][index_used['ver']], p0 = fit_p0['ver'], bounds = self.fit_bounds['ver'], max_nfev = 10000)
            #plot the results now!
            fit_data_temp['hor']['x'][-1] = cut_ip_q[index_used['hor']]
            fit_data_temp['hor']['y'][-1] = cut['hor'][index_used['hor']]
            fit_data_temp['ver']['x'][-1] = cut_oop_q[index_used['ver']]
            fit_data_temp['ver']['y'][-1] = cut['ver'][index_used['ver']]
            fig_handle = plot_fun(fig_handle,self.img,self.grid_q_ip,self.grid_q_oop,\
                                  vmin = plot_lib['vmin'], vmax =plot_lib['vmax'], cmap = 'jet', is_zap_scan = plot_lib['is_zap_scan'],\
                                  fit_data = fit_data_temp, model=self.model, \
                                  fit_results={'hor':[fit_ip,fom_ip],'ver':[fit_oop,fom_oop]},\
                                  processed_data_container=data, cut_offset = self.cut_offset,\
                                  peak_center = center_index,\
                                  title = 'Frame_{}, E ={:04.2f}V'.format(plot_lib['frame_number'],plot_lib['potential']))
            fig_handle.canvas.draw()
            fig_handle.tight_layout()
            plt.pause(0.05)
            plt.show()
            #do you want to update the fit results.
            if accept_it == 'n':
                accept_or_not = raw_input("Accept current tweak: yes(y) or no (n).\Any other keys to reject!") or 'n'
            else:
                accept_or_not = 'y'
            if accept_or_not == 'y':
                update = True
            elif accept_or_not == 'n':
                update = False
            else:
                update = False

            if update:
                print('Update the current tweak results to fit instance attributes and abort the tweak mode!')
                self.fit_data['hor']['x'][-1] = cut_ip_q[index_used['hor']]
                self.fit_data['hor']['y'][-1] = cut['hor'][index_used['hor']]
                self.fit_data['ver']['x'][-1] = cut_oop_q[index_used['ver']]
                self.fit_data['ver']['y'][-1] = cut['ver'][index_used['ver']]

                self.fit_p0_2['ver'] = fit_oop
                self.fit_p0_2['hor'] = fit_ip
                self.update_bounds(fit_oop[0],fit_ip[0])
                self.peak_center = [np.argmin(np.abs(self.q_oop[:,0]-fit_oop[0])),np.argmin(np.abs(self.q_ip[0,:]-fit_ip[0]))]
                self.fit_results_plot['hor'] = [copy.deepcopy(fit_ip), copy.deepcopy(fom_ip)]
                self.fit_results_plot['ver'] = [copy.deepcopy(fit_oop), copy.deepcopy(fom_oop)]
                data = self.update_data(data)
            else:
                pass
        return data


    def fit(self, check=True, level = 0.05, **kwarg):
        #self.img[self.mask ==0]=0
        self.fit_data = {'hor':{'x':[],'y':[]},'ver':{'x':[],'y':[]}}
        fit_ip_0, fom_ip_0 = None, None
        fit_oop_0, fom_oop_0 = None, None
        peak_locating_step = True
        center_far_off_test = True
        #first cut with large window
        for i in range(len(self.cut_offset['hor'])):
            if i==0:
                cycles = 1
            elif i==1:
                cycles = 2
            for j in range(cycles):
                fit_p0 = {0:self.fit_p0,1:self.fit_p0_2}
                if self.first_frame:
                    center_index = {0:self.prim_beam_pot, 1:self.peak_center}
                else:
                    center_index = {0:self.peak_center, 1:self.peak_center}
                cut_offset_temp =dict([(key,value[i]) for key, value in self.cut_offset.items()])
                data_range_offset_temp =dict([(key,value[i]) for key, value in self.data_range_offset.items()])
                cut = self.cut_profile_from_2D_img_around_center(self.img,cut_offset_temp,data_range_offset_temp, center_index[i], sum_result = True)
                #cut_mask = self.cut_profile_from_2D_img_around_center(self.mask,cut_offset_temp,data_range_offset_temp, center_index[i], sum_result = False)
                cut_mask = {'hor':1,'ver':1}
                cut_ip_q = self.cut_profile_from_2D_img_around_center(self.q_ip,cut_offset_temp, data_range_offset_temp, center_index[i], sum_result = False)['hor']
                cut_oop_q = self.cut_profile_from_2D_img_around_center(self.q_oop,cut_offset_temp, data_range_offset_temp, center_index[i], sum_result = False)['ver']
                #normalize the dead pixel contribution
                cut = {'hor':cut['hor']/cut_mask['hor'],'ver':cut['ver']/cut_mask['ver']}
                #print(cut['hor'])

                #check nan and inf values
                index_used = {'hor':~(np.isnan(cut['hor'])|np.isinf(cut['hor'])), 'ver':~(np.isnan(cut['ver'])|np.isinf(cut['ver']))}
                if self.first_frame and i==0:
                    self.fit_p0['ver'][0] = self.q_oop[self.peak_center[0],0]
                    self.fit_p0['hor'][0] = self.q_ip[0,self.peak_center[1]]
                try:
                    fit_ip,fom_ip = opt.curve_fit(f=self.model, xdata=cut_ip_q[index_used['hor']], ydata=cut['hor'][index_used['hor']], p0 = fit_p0[i]['hor'], bounds = self.fit_bounds['hor'], max_nfev = 10000)
                    fit_oop,fom_oop = opt.curve_fit(f=self.model, xdata=cut_oop_q[index_used['ver']], ydata=cut['ver'][index_used['ver']], p0 = fit_p0[i]['ver'], bounds = self.fit_bounds['ver'], max_nfev = 10000)
                    self.fit_status = True
                except:
                    self.fit_status = False
                    fit_ip, fom_ip = [0,0,0,0,0,0],[0]
                    fit_oop, fom_oop = [0,0,0,0,0,0],[0]
                    print('Peak fit failed!')
                    #break
                # print('updata fit_data now!',i,min(cut_ip_q[index_used['hor']]),min(cut['hor'][index_used['hor']]),min(cut_oop_q[index_used['ver']]),min(cut['ver'][index_used['ver']]))
                if i==0 or (i==1 and j==1):
                    self.fit_data['hor']['x'].append(cut_ip_q[index_used['hor']])
                    self.fit_data['hor']['y'].append(cut['hor'][index_used['hor']])
                    self.fit_data['ver']['x'].append(cut_oop_q[index_used['ver']])
                    self.fit_data['ver']['y'].append(cut['ver'][index_used['ver']])
                if i==0:
                    fit_ip_0, fom_ip_0 = fit_ip, fom_ip
                    fit_oop_0, fom_oop_0 = fit_oop, fom_oop
                peak_center_ = [np.argmin(np.abs(self.q_oop[:,0]-fit_oop[0])),np.argmin(np.abs(self.q_ip[0,:]-fit_ip[0]))]
                if i==0 or (i==1 and j==0):
                    self.peak_center = peak_center_
                elif (i==1 and j==1):#in second where run j=1 and j=0, the peakcenter should be very close to each other, if not peak is not located correctly! Note 10 pixel away is only arbitrary value, which may be changed accordingly!
                    if (not self.pot_step_scan) and np.abs(np.array(self.previous_peak_center)-np.array(peak_center_)).sum()>4 and (not self.first_frame):
                        center_far_off_test = False
                        print('Two successive fit results is far off!! CHECK frame!!')
                        print('current peak center:{},previous peak center:{}'.format(peak_center_,self.previous_peak_center))
                    else:#CV scan without large offset of peak center or pot_step_scan
                        self.peak_center = peak_center_
                        self.previous_peak_center = peak_center_
                        #update the peak center, but not change the other par values
                        self.fit_p0['ver'] = fit_oop
                        self.fit_p0['hor'] = fit_ip

                        self.fit_p0_2['ver'] = fit_oop
                        self.fit_p0_2['hor'] = fit_ip
                        self.update_bounds(fit_oop[0],fit_ip[0])
                # quit()

        # quit()
        #finish the fit and update the fit par values
        # self.update_bounds(fit_oop[0],fit_ip[0])
        self.fit_results_plot['hor'] = [copy.deepcopy(fit_ip), copy.deepcopy(fom_ip)]
        self.fit_results_plot['ver'] = [copy.deepcopy(fit_oop), copy.deepcopy(fom_oop)]
        if self.use_first_fit_for_pos and peak_locating_step:
            fit_ip[0], fom_ip[0] = fit_ip_0[0], fom_ip_0[0]
            fit_oop[0], fom_oop[0] = fit_oop_0[0], fom_oop_0[0]
        def _check(old, new, level = level):
            check_result = bool((abs((np.array(old)[0:2] - np.array(new)[0:2])/np.array(old)[0:2])>level).sum())
            # print(check_result)
            return check_result
        if check:
            if center_far_off_test:
            # if (_check(self.fit_results['hor'][0],fit_ip) | _check(self.fit_results['ver'][0],fit_oop))==False:
                self.fit_results['hor'] = [fit_ip, fom_ip]
                self.fit_results['ver'] = [fit_oop, fom_oop]
                return True
            else:
                self.peak_center = self.previous_peak_center
                #print(self.fit_results['hor'],self.fit_results['ver'])
                return False
        else:
            self.fit_results['hor'] = [fit_ip, fom_ip]
            self.fit_results['ver'] = [fit_oop, fom_oop]
            return True
        # print fit_ip,fit_oop

    def save_data(self,data):
        # data = container
        pcov_ip = self.fit_results['hor'][1]
        pcov_oop = self.fit_results['ver'][1]

        popt_ip = self.fit_results['hor'][0]
        popt_oop = self.fit_results['ver'][0]

        data['pcov_ip'].append(pcov_ip)
        data['pcov_oop'].append(pcov_oop)

        data['cen_ip'].append(popt_ip[0])
        data['FWHM_ip'].append(popt_ip[1])
        data['amp_ip'].append(popt_ip[2])
        data['lfrac_ip'].append(popt_ip[3])
        data['bg_slope_ip'].append(popt_ip[4])
        data['bg_offset_ip'].append(popt_ip[5])

        data['cen_oop'].append(popt_oop[0])
        data['FWHM_oop'].append(popt_oop[1])
        data['amp_oop'].append(popt_oop[2])
        data['lfrac_oop'].append(popt_oop[3])
        data['bg_slope_oop'].append(popt_oop[4])
        data['bg_offset_oop'].append(popt_oop[5])
        # data['peak_intensity'].append(self.peak_intensity)
        # print popt_oop[0],data['cen_oop']
        return data

    def update_data(self,data):
        # data = container
        pcov_ip = self.fit_results['hor'][1]
        pcov_oop = self.fit_results['ver'][1]

        popt_ip = self.fit_results['hor'][0]
        popt_oop = self.fit_results['ver'][0]

        data['pcov_ip'][-1]=pcov_ip
        data['pcov_oop'][-1]=pcov_oop

        data['cen_ip'][-1]=popt_ip[0]
        data['FWHM_ip'][-1]=popt_ip[1]
        data['amp_ip'][-1]=popt_ip[2]
        data['lfrac_ip'][-1]=popt_ip[3]
        data['bg_slope_ip'][-1]=popt_ip[4]
        data['bg_offset_ip'][-1]=popt_ip[5]

        data['cen_oop'][-1]=popt_oop[0]
        data['FWHM_oop'][-1]=popt_oop[1]
        data['amp_oop'][-1]=popt_oop[2]
        data['lfrac_oop'][-1]=popt_oop[3]
        data['bg_slope_oop'][-1]=popt_oop[4]
        data['bg_offset_oop'][-1]=popt_oop[5]
        data['peak_intensity'][-1]=self.peak_intensity
        # print popt_oop[0],data['cen_oop']
        return data

class Reciprocal_Space_Mapping():
    def __init__(self, img, cen, kwarg):
        #E_keV=19.5, cen=(234,745), pixelsize=(0.055,0.055), sdd=714, UB=[],motor_angles=None, boost_mapping = False
        self.img = img
        self.intensity = None
        #self.E_keV = E_keV

        self.cen = cen
        #self.pixelsize = pixelsize
        #self.sdd = sdd
        #self.UB=UB
        #self.motor_angles = motor_angles
        self.q=None
        self.grid_indices = None
        self.grid_intensity = None
        for key in kwarg:
            setattr(self, key, kwarg[key])
        self.wavelength = 12.39854*self.e_kev
        self.k0 = 2.*np.pi/self.wavelength
        #self.boost_mapping = boost_mapping
        # self.prepare_frame()
        # self.get_grid_q()

    def update_img(self, img,UB=None, motor_angles=None,update_q = True):
        self.img = img
        if UB!=None:
            self.UB =UB
        if motor_angles!=None:
            self.motor_angles = motor_angles
        self.prepare_frame()
        if update_q:
            self.get_grid_q()
        else:
            self.grid_intensity = self.intensity.ravel()[self.grid_indices].reshape(self.intensity.shape)

    def prepare_frame(self, norm_mon=True, norm_transm=True,trans='attenpos',mon='avg_beamcurrent'):
        transm_= 1
        mon_= 1
        th_= self.motor_angles['mu']
        gam_= self.motor_angles['delta']
        del_= self.motor_angles['gamma']
        #the chi and phi values are arbitrary in the fio file, should be set to the same values as the ones that are usd to cal UB matrix(all 0 so far)
        phi_= self.motor_angles['phi']
        chi_= self.motor_angles['chi']
        mu_= self.motor_angles['omega_t']
        # print del_,gam_
        #first item is the incident angle (mu_ here)
        del_,gam_=np.rad2deg(vliegDiffracAngles(np.deg2rad([mu_,del_,gam_,mu_,0,0]))[1:3])
        # print del_,gam_
        intensity = self.img
        #detector dimension is (516,1556)
        #You may need to put a negative sign in front, check the rotation sense of delta and gamma motors at P23
        delta_range = np.arctan((np.arange(intensity.shape[1])-self.cen[1])*self.pixelsize[0]/self.sdd)*180/ np.pi + del_
        #the minus sign here because the column index increase towards bottom, then 0 index(top most) will give a negative gam offset
        #a minus sign in front correct this.
        gamma_range =-np.arctan((np.arange(intensity.shape[0])-self.cen[0])*self.pixelsize[1]/self.sdd)*180/ np.pi + gam_
        #polarisation correction
        # TODO: what is this doing?
        delta_grid , gamma_grid= np.meshgrid(delta_range,gamma_range)
        Pver = 1 - np.sin(delta_grid * np.pi / 180.)**2 * np.cos(gamma_grid * np.pi / 180.)**2
        intensity=np.divide(intensity,Pver)
        self.intensity = intensity
        self.vlieg_angles = {'gamma_range':gamma_range, 'delta_range':delta_range,'th_':th_, 'mu_':mu_, 'chi_':chi_, 'phi_':phi_}

    def get_HKL(self):
        for each in ['gamma_range','delta_range', 'th_', 'mu_', 'chi_', 'phi_']:
            locals()[each]=self.vlieg_angles[each]
        d = SixCircle.SixCircle()
        d.setEnergy(self.E_keV)
        d.setUB(self.UB)
        HKL = d.getHKL(delta=delta_range, theta=th_, chi=chi_, phi=phi_, mu=mu_, gamma=gamma_range, gamma_first=False)
        shape =  gamma_range.size,delta_range.size
        # shape =  delta_range.size,gamma_range.size
        H = HKL[0,:].reshape(shape)
        K = HKL[1,:].reshape(shape)
        L = HKL[2,:].reshape(shape)
        self.HKL = {'H':H, 'K':K, 'L':L}

    def get_grid_q(self):
        for each in ['gamma_range','delta_range', 'th_', 'mu_', 'chi_', 'phi_']:
            globals()[each]=self.vlieg_angles[each]
        d = SixCircle.SixCircle()
        d.setEnergy(self.e_kev)
        d.setUB(self.ub)
        Q = d.getQSurface(theta=th_, chi=chi_, phi=phi_, mu=mu_, delta=delta_range, gamma=gamma_range, gamma_first=False)
        shape =  gamma_range.size,delta_range.size
        # print(shape)
        # print(self.vlieg_angles)
        qx = Q[0,:].reshape(shape)
        qy = Q[1,:].reshape(shape)
        qz = Q[2,:].reshape(shape)
        q_para = np.sqrt(qx**2 + qy**2)
        size =self.intensity.shape
        #shape=(vertical,horizontal),len(vertical)=size(1),len(horizontal)=size(0)
        grid_q_perp, grid_q_para = np.mgrid[np.max(qz):np.min(qz):(1.j*size[0]), np.min(q_para):np.max(q_para):(1.j*size[1])]
        # def _intensity_mapping(grid_q_para, grid_q_perp, q_para, q_z, intensity, shift_buffer = 10):
            # shape = intensity.shape
            # grid_intensity = np.zeros(shape)
            # for i in range(shape[0]):
                # for j in range(shape[1]):
                    # single_q_para = q_para[i,j]
                    # single_q_perp = q_z[i,j]
                    # jj = np.argmin(abs(grid_q_para[i]-single_q_para))
                    # ii = np.argmin(abs(grid_q_perp[i]-single_q_perp))
                    # grid_intensity[ii,jj]=intensity[i,j]
            # return grid_intensity

        if self.boost_mapping:
            new_data = self.intensity
            # calculate the pixel coordinates of the
            # computational domain corners in the data array
            w,e,s,n = np.min(q_para),np.max(q_para),np.min(qz),np.max(qz)
            # w,e,s,n = q_para[-1][0],q_para[0][-1],qz[:,-1][-1],qz[:,0][0]

            # data corners:
            lon = np.array([[w,        w],
                            [e, e]])
            lat = np.array([[s,        n],
                            [s, n]])
            dx = float(e-w)/new_data.shape[1]
            dy = float(n-s)/new_data.shape[0]
            x = (lon.ravel()-w)/dx
            y = (n-lat.ravel())/dy

            computational_domain_corners = np.float32(list((zip(x,y))))

            data_array_corners = np.float32([[new_data.shape[0],0],
                                            [0,0],
                                            [new_data.shape[0],new_data.shape[1]],
                                            [0,new_data.shape[1]]])
            # data_array_corners = np.float32([[0,new_data.shape[0]],
                                            # [0,0],
                                            # [new_data.shape[1],new_data.shape[0]],
                                            # [new_data.shape[1],0]])

            # Compute the transformation matrix which places
            # the corners of the data array at the corners of
            # the computational domain in data array pixel coordinates
            tranformation_matrix = cv2.getPerspectiveTransform(data_array_corners,
                                                            computational_domain_corners)

            # Make the transformation making the final array the same shape
            # as the data array, cubic interpolate the data placing NaN's
            # outside the new array geometry
            grid_intensity = cv2.warpPerspective(new_data,tranformation_matrix,
                                            (new_data.shape[1],new_data.shape[0]),
                                            flags=1,
                                            borderMode=0,
                                            borderValue=0)
        else:
            # grid_intensity = _intensity_mapping(grid_q_para, grid_q_perp, q_para, qz, self.intensity, shift_buffer = 10)
            grid_intensity = griddata((q_para.ravel(), qz.ravel()), self.intensity.ravel(), (grid_q_para, grid_q_perp), method='nearest')
        self.grid_intensity = grid_intensity
        self.q={'qx':qx,'qy':qy,'qz':qz,\
                'q_par':q_para, 'q_perp':qz,\
                'grid_q_par':grid_q_para,'grid_q_perp':grid_q_perp}
        if self.grid_indices is None:
            self.grid_indices = griddata((q_para.ravel(), qz.ravel()), np.arange(self.intensity.size).ravel(), (grid_q_para, grid_q_perp), method='nearest')
        else:
            pass

    def show_image(self):
        grid_q_para, grid_q_perp, grid_intensity = self.get_grid_q_in_out_plane(self.scan_no, self.frame_no,self.frame_prefix)
        plt.figure()
        # plt.imshow(grid_intensity, vmin=0, vmax=100.05)
        plt.imshow(grid_intensity,cmap='jet',vmin=0, vmax=100.05)
        plt.title("plt.imshow(grid_intensity")
        # plt.colorbar(extend='both',orientation='Vertical')
        plt.clim(0,90)
        plt.show()

import copy
class background_subtraction_single_img():
    def __init__(self,cen, config_file = '../config/config_bkg_sub.ini',sections = ['Integration_setup','Correction_pars','Spec_info']):
        self.config_file = config_file
        self.config_file_parser(config_file, sections)
        self.img = None
        self.center_pix = cen
        self.center_pix_origin = copy.copy(cen)
        self.opt_values = {'cen':None, 'peak_width':None, 'row_width': None, 'col_width': None,\
                           'fit_threshold': None, 'int_power': None, \
                           'int_dir':None, 'cost_fun': None}
        self.fit_results = {'F':None, 'Ferr':None, 'I':None, 'Ierr':None, 'ctot':None,'bkg':None}
        self.fit_data = {'x':[],'y_total':[],'y_bkg':[]}
        self.x_min = None
        self.y_min = None
        self.x_span = None
        self.y_span = None
        self.fit_status = False
        self.col_width_origin = 5
        self.col_width = 5
        self.row_width_origin = 10
        self.row_width = 10
        self.bkg_row_width = 10
        self.bkg_col_width = 10
        self.ord_cus_s = [1]
        self.ss = [1]
        self.ss_factor = 1
        self.fct = 'atq'
        self.peak_shift = 0
        self.peak_width = 15
        self.bkg_win_cen_offset_lr = 10
        self.bkg_win_cen_offset_ud = 10

    def config_file_parser(self, config_file, sections):
        config = configparser.ConfigParser()
        config.read(config_file)
        for section in sections:
            for each in config.items(section):
                try:
                    setattr(self,each[0], eval(each[1]))
                except:
                    setattr(self,each[0], each[1])
        #self.center_pix_origin = self.center_pix
        # self.col_width_origin = self.col_width
        # self.row_width_origin = self.row_width

    def update_center_pix_up_and_down(self,offset):
        offset = int(offset)
        #offset = np.array([-offset,0])
        self.center_pix[0] = self.center_pix_origin[0] - offset
        return None

    def update_center_pix_left_and_right(self,offset):
        offset = int(offset)
        #offset = np.array([0,offset])
        self.center_pix[1] = self.center_pix_origin[1] + offset
        return None

    def update_integration_window_column_width(self,offset):
        offset = int(offset)
        self.col_width = self.col_width_origin + offset
        return None

    def update_integration_window_row_width(self,offset):
        offset = int(offset)
        self.row_width = self.row_width_origin + offset
        return None


    def update_bkg_center_pix_up_and_down(self,offset):
        offset = int(offset)
        self.bkg_win_cen_offset_ud = self.bkg_win_cen_offset_ud + offset
        return None

    def update_bkg_center_pix_left_and_right(self,offset):
        offset = int(offset)
        self.bkg_win_cen_offset_lr = self.bkg_win_cen_offset_lr + offset
        return None

    def update_bkg_column_width(self,offset):
        offset = int(offset)
        self.bkg_col_width = self.bkg_col_width + offset
        return None

    def update_bkg_row_width(self,offset):
        offset = int(offset)
        self.bkg_row_width = self.bkg_row_width + offset
        return None

    def update_integration_order(self,order):
        self.ord_cus_s = [int(order)]

    def update_integration_function(self,fct):
        self.fct = fct

    def update_ss_factor(self,sf):
        self.ss_factor = float(sf)

    def update_peak_width(self,offset):
        self.peak_width = self.peak_width + int(offset)

    def update_peak_shift(self,offset):
        self.peak_shift = self.peak_shift + int(offset)

    def update_var_for_tweak(self,str_parser):
        cmds_list = str_parser.split(',')
        lib_arg = {'UD':self.update_bkg_center_pix_up_and_down,\
                   'LR':self.update_bkg_center_pix_left_and_right,\
                   'CW':self.update_bkg_column_width,\
                   'RW':self.update_bkg_row_width,\
                   'ud':self.update_center_pix_up_and_down,\
                   'lr':self.update_center_pix_left_and_right,\
                   'cw':self.update_integration_window_column_width,\
                   'rw':self.update_integration_window_row_width,\
                   'od':self.update_integration_order,\
                   'sf':self.update_ss_factor,\
                   'ft':self.update_integration_function,\
                   'pw':self.update_peak_width,\
                   'ps':self.update_peak_shift}
        if cmds_list == ['qw']:
            return 'qw'
        elif cmds_list == ['rm']:
            return 'rm'
        elif cmds_list == ['r']:
            return 'repeat_last'
        elif cmds_list == ['#r']:
            return 'process_through'
        elif cmds_list[0][0:2] in lib_arg.keys():
            for each in cmds_list:
                if each[0:2] in lib_arg:
                    lib_arg[each[0:2]](each[2:])
                else:
                    pass
            return 'tweak'
        else:
            print('Warning: Motion not valid! Make another tweak!')
            return 'tweak'

    #find the peak width automatically (col_width if direction = 'vertical'; row_width if direction = 'horizontal')
    def find_peak_width(self, img, img_no = 0, initial_c_width = 400, initial_r_width = 50):
        SN_ratio_container = []
        SN_ratio_sub = []
        c_width_container=[]
        c_width_container_sub = []
        if initial_r_width == None:
            initial_r_width = self.row_width
        if initial_c_width == None:
            initial_c_width = self.col_width

        best_c_width = None
        original_direction = self.int_direct
        if self.int_direct == 'x':
            direction ='vertical'
            initial_r_width = self.row_width
        elif self.int_direct == 'y':
            direction = 'horizontal'
            initial_c_width = self.col_width
            
        for i in range(1,100,2):
            current_c_width = int(initial_c_width/i)
            c_width_container.append(current_c_width)
            if direction=='vertical':
                SN_ratio_container.append(self.integrate_one_image_light_v(img,initial_r_width, current_c_width))
            elif direction == 'horizontal':
                SN_ratio_container.append(self.integrate_one_image_light_v(img,current_c_width,initial_c_width))
            if i>1:
                if np.mean(SN_ratio_container)<0.01:
                    print('No peak found!')
                    return best_c_width
                if SN_ratio_container[-1]<SN_ratio_container[0]*0.9:#0.9 is an empirical value
                    SN_ratio_sub = [SN_ratio_container[-2]]
                    # print('i={}'.format(i))
                    for j in range(1,11):
                        # print('j={}'.format(j))
                        c_width_l, c_width_r = current_c_width*2, current_c_width
                        current_c_width_sub = int(c_width_l - (c_width_l - c_width_r)/10*j)
                        c_width_container_sub.append(current_c_width_sub)
                        if direction == 'vertical':
                            current_SN = self.integrate_one_image_light_v(img,initial_r_width, current_c_width_sub)
                        elif direction == 'horizontal':
                            current_SN = self.integrate_one_image_light_v(img, current_c_width_sub,initial_c_width)
                        # print('current={}, mean={}'.format(current_SN, np.mean(SN_ratio_sub)))
                        if current_SN < SN_ratio_sub[0]*0.95:#0.95 is an empirical value
                            best_c_width = current_c_width_sub + int((c_width_l - c_width_r)/10)
                            break
                        else:
                            SN_ratio_sub.append(current_SN)
                    break
                else:
                    print('i={}'.format(i))
            else:
                pass
        print('Best {} width = {}'.format(['column','row'][int(direction =='horizontal')],best_c_width))

        if best_c_width!=None:
            if self.int_direct == 'x':
                self.col_width = best_c_width + 10#10 is arbitrary offset value
            elif self.int_direct == 'y':
                self.row_width = best_c_width + 10#10 is arbitrarz offset value
        #self.int_direct = original_direction
        return best_c_width

    def integrate_one_image_light_v(self, img, r_width, c_width):
        self.img = img
        center_pix=self.center_pix
        # r_width=self.row_width
        # c_width=self.col_width
        integration_direction=self.int_direct
        ord_cus_s=self.ord_cus_s
        ss=self.ss
        fct=self.fct
        #print(r_width,c_width)
        if center_pix[0]<c_width:
            c_width = center_pix[0]-10

        if center_pix[1]<r_width:
            r_width = center_pix[1]-10
        index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        #print(center_pix)
        #print(sub_index)
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        pil_y,pil_x=img.shape#shape of the pilatus image
        #print(img.shape)
        #reset the boundary if the index number is beyond the pilatus shape
        #x_min,x_max,y_min,y_max=[int(x_min>0)*x_min,int(x_max>0)*x_max,int(y_min>0)*y_min,int(y_max>0)*y_max]
        #x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,int(x_max<pil_x)*x_max,int(y_min<pil_y)*y_min,int(y_max<pil_y)*y_max]
        x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,[pil_x,x_max][int(x_max<pil_x)],int(y_min<pil_y)*y_min,[pil_y,y_max][int(y_max<pil_y)]]

        x_span,y_span=x_max-x_min,y_max-y_min
        clip_image_center = [int(y_span/2)+self.peak_shift,int(x_span/2)+self.peak_shift]
        peak_l = max([clip_image_center[int(self.int_direct=='x')]-self.peak_width,0])#peak_l>0
        peak_r = clip_image_center[int(self.int_direct=='x')]+self.peak_width
        if self.x_min ==None:
            self.x_min, self.y_min, self.x_span, self.y_span = x_min, y_min, x_span, y_span
        else:
            pass
        #print (y_min,y_max,x_min,x_max)
        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        #Now normalized the data
        n=np.array(range(len(y)))
        #by default the contamination rate is 25%
        #the algorithm may fail if the peak cover >40% of the cut profile
        bkg_n = int(len(y)/4)
        y_sorted = list(copy.deepcopy(y))
        y_sorted.sort()
        # std_bkg =np.array(list(y[0:bkg_n])+list(y[-bkg_n:-1])).std()/(max(y)-min(y))
        std_bkg =np.array(y_sorted[0:bkg_n*3]).std()/(max(y_sorted)-min(y_sorted))
        if hasattr(self,'ss_factor'):
            ss = [self.ss_factor*std_bkg]
        else:
            ss = [5*std_bkg]

        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        s_container=[]
        ord_cus_container=[]
        index=None
        def _cal_FOM(y,z,peak_width):
            ct=int(len(y)/2)
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[0:lf]-z[0:lf]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            left_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            right_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            std_fom=np.array(left_boundary+right_boundary).std()
            return sum_temp/(len(y)-peak_width*2),std_fom#averaged offset (goodness of result), standard deviation of residual (counted as error during data integration)
            #return std_fom

        def _cal_FOM_2(y,z,s):
            y=np.array(y)
            z=np.array(z)
            y_scaled=(y-y.max())/(y.max()-y.min())+1#scaled to [0,1]
            index_container=np.where(y_scaled<s)[0]
            return np.abs(y[index_container]-z[index_container]).std()

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[peak_l:peak_r][index]-z[peak_l:peak_r][index]))
                indexs_bkg=list(range(0,peak_l))+list(range(peak_r,len(y)))
                if len(indexs_bkg)!=0:
                    std_I_bkg = np.array(y[indexs_bkg]-z[indexs_bkg]).std()
                else:
                    std_I_bkg = 0
                Ibgr_container.append(abs(np.sum(z[peak_l:peak_r][index])))
                FOM_container.append(_cal_FOM(y,z,peak_width=30))
                Ierr_container.append((I_container[-1])**.5+std_I_bkg*(peak_r-peak_l))#possoin error + error from integration + 3% of current intensity

                z_container.append(z)
                s_container.append(s)
                ord_cus_container.append(ord_cus)
        index_best=np.argmin(np.array(FOM_container)[:,0])
        #print 'std=',FOM_container[index_best]
        #print 'all std=',FOM_container
        index = np.argsort(n)
        return I_container[index_best]

    def integrate_one_image(self,fig, img, data=None, plot_live = False, freeze_sf=False):
        self.img = img
        center_pix=self.center_pix
        #print(center_pix)
        r_width=self.row_width
        c_width=self.col_width
        integration_direction=self.int_direct
        ord_cus_s=self.ord_cus_s
        ss=self.ss
        fct=self.fct
        #print(r_width,c_width)
        if center_pix[0]<c_width:
            c_width = center_pix[0]-10

        if center_pix[1]<r_width:
            r_width = center_pix[1]-10
        index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        #print(sub_index)
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        pil_y,pil_x=img.shape#shape of the pilatus image
        #reset the boundary if the index number is beyond the pilatus shape
        #x_min,x_max,y_min,y_max=[int(x_min>0)*x_min,int(x_max>0)*x_max,int(y_min>0)*y_min,int(y_max>0)*y_max]
        #x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,int(x_max<pil_x)*x_max,int(y_min<pil_y)*y_min,int(y_max<pil_y)*y_max]
        x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,[pil_x,x_max][int(x_max<pil_x)],int(y_min<pil_y)*y_min,[pil_y,y_max][int(y_max<pil_y)]]
        # x_min_new,x_max_new,y_min_new,y_max_new=[int(x_min>0)*x_min,int(x_max>0)*x_max,int(y_min>0)*y_min,int(y_max>0)*y_max]
        # x_min_new,x_max_new,y_min_new,y_max_new=[int(x_min_new<pil_x)*x_min_new,int(x_max_new<pil_x)*x_max_new,int(y_min_new<pil_y)*y_min_new,int(y_max_new<pil_y)*y_max_new]
        # compare_results =[x_min == x_min_new, x_max == x_max_new, y_min == y_min_new, y_max == y_max_new]
        # x_corner, y_corner =None, None

        # for i in range(4):
            # if i in [0,1] and compare_results[i]:
                # x_corner = i
            # elif i in [2,3] and compare_results[i]:
                # y_corner = i
        # if [x_corner,y_corner] == [0,1]:

        # elif [x_corner,y_corner] == [1,1]:
        # elif [x_corner,y_corner] == [0,0]:
        # elif [x_corner,y_corner] == [1,0]:

        x_span,y_span=x_max-x_min,y_max-y_min
        clip_image_center = [int(y_span/2)+self.peak_shift,int(x_span/2)+self.peak_shift]
        peak_l = max([clip_image_center[int(self.int_direct=='x')]-self.peak_width,0])#peak_l>0
        peak_r = clip_image_center[int(self.int_direct=='x')]+self.peak_width
        self.x_min, self.y_min, self.x_max, self.y_max, self.x_span, self.y_span = x_min, y_min, x_max, y_max, x_span, y_span
        #print (y_min,y_max,x_min,x_max)
        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        
        clip_image_center_bkg = clip_image_center+np.array([-self.bkg_win_cen_offset_ud,self.bkg_win_cen_offset_lr])
        y_min_bkg = clip_image_center_bkg[0]-self.bkg_col_width
        y_max_bkg = clip_image_center_bkg[0]+self.bkg_col_width

        x_min_bkg = clip_image_center_bkg[1]-self.bkg_row_width
        x_max_bkg = clip_image_center_bkg[1]+self.bkg_row_width

        x_span_bkg = x_max_bkg - x_min_bkg
        y_span_bkg = y_max_bkg - y_min_bkg
        self.x_min_bkg, self.y_min_bkg, self.x_max_bkg, self.y_max_bkg, self.x_span_bkg, self.y_span_bkg = x_min_bkg, y_min_bkg, x_max_bkg, y_max_bkg, x_span_bkg, y_span_bkg
        bkg_sum = np.sum(img[y_min_bkg:y_max_bkg+1,x_min_bkg:x_max_bkg+1])

        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        #Now normalized the data
        #y = y/data['mon'][-1]/data['transm'][-1]
        n=np.array(range(len(y)))
        #by default the contamination rate is 25%
        #the algorithm may fail if the peak cover >40% of the cut profile
        bkg_n = int(len(y)/4)
        y_sorted = list(copy.deepcopy(y))
        y_sorted.sort()
        # std_bkg =np.array(list(y[0:bkg_n])+list(y[-bkg_n:-1])).std()/(max(y)-min(y))
        std_bkg =np.array(y_sorted[0:bkg_n*3]).std()/(max(y_sorted)-min(y_sorted))
        # if hasattr(self,'ss_factor'):
            # ss = [self.ss_factor*std_bkg]
        # else:
            # ss = [5*std_bkg]
        if freeze_sf:
            ss = [self.ss_factor*std_bkg]
        else:
            ss = [std_bkg, 1.5*std_bkg, 2*std_bkg, 2.5*std_bkg, 3*std_bkg, 4*std_bkg]

        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        s_container=[]
        ord_cus_container=[]
        index=None
        # peak_width=10
        # if self.int_direct=='y':
            # peak_width==int(self.col_width*peak_width_percent)
        # elif self.int_direct=='x':
            # peak_width=int(self.row_width*peak_width_percent)
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=int(len(y)/2)
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[0:lf]-z[0:lf]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            left_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            right_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            std_fom=np.array(left_boundary+right_boundary).std()
            return sum_temp/(len(y)-peak_width*2),std_fom#averaged offset (goodness of result), standard deviation of residual (counted as error during data integration)
            #return std_fom

        def _cal_FOM_2(y,z,s):
            y=np.array(y)
            z=np.array(z)
            y_scaled=(y-y.max())/(y.max()-y.min())+1#scaled to [0,1]
            index_container=np.where(y_scaled<s)[0]
            return np.abs(y[index_container]-z[index_container]).std()

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[peak_l:peak_r][index]-z[peak_l:peak_r][index]))
                indexs_bkg=list(range(0,peak_l))+list(range(peak_r,len(y)))
                if len(indexs_bkg)!=0:
                    std_I_bkg = np.array(y[indexs_bkg]-z[indexs_bkg]).std()
                else:
                    std_I_bkg = 0
                # print('sensor',y[index])
                # I_container.append(np.sum(y[index]))
                Ibgr_container.append(abs(np.sum(z[peak_l:peak_r][index])))
                FOM_container.append(_cal_FOM(y,z,peak_width=int(len(y)/4)))
                #Ierr_container.append((I_container[-1]+FOM_container[-1])**0.5)
                # Ierr_container.append((I_container[-1])**0.5+FOM_container[-1][1]+I_container[-1]*0.03)#possoin error + error from integration + 3% of current intensity
                # Ierr_container.append(std_bkg/abs(I_container[-1])*std_bkg+(np.sum(y)/data['mon'][-1]/data['transm'][-1])**0.5)#possoin error + error from integration + 3% of current intensity
                #Ierr_container.append((I_container[-1])**.5+std_I_bkg*(peak_r-peak_l))#possoin error + error from integration + 3% of current intensity
                Ierr_container.append((I_container[-1])**.5)#possoin error + error from integration + 3% of current intensity

                # try:
                    # Ierr_container.append((np.sum(y)/data['mon'][-1]/data['transm'][-1])**0.5)#possoin error + error from integration + 3% of current intensity
                # except:
                    # Ierr_container.append((np.sum(y))**0.5)#possoin error + error from integration + 3% of current intensity

                z_container.append(z)
                s_container.append(s)
                ord_cus_container.append(ord_cus)
        index_best=np.argmin(np.array(FOM_container)[:,0])
        #print 'std=',FOM_container[index_best]
        #print 'all std=',FOM_container
        index = np.argsort(n)
        # data['peak_intensity'].append(I_container[index_best])
        check_result = True
        # if check:
            # if I_container[index_best]<check_level:
                # check_result = False

        self.fit_data['x'] = n[index]
        self.fit_data['y_total'] = y[index]
        self.fit_data['y_bkg'] = z[index]
        # print ("When s=",s_container[index_best],'pow=',ord_cus_container[index_best],"integration sum is ",I_container[index_best], " counts!",'S/N ratio is {:3.2f}'.format(I_container[index_best]/Ibgr_container[index_best]+1))
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5
        return I_container[index_best],FOM_container[index_best][1],Ierr_container[index_best],s_container[index_best],ord_cus_container[index_best],center_pix,30,r_width,c_width,bkg_sum,check_result

    def update_motor_angles(self, motor_lib):
        # keys_motor_new = ['gamma','delta','mu','omega_t', 'phi', 'chi']
        #doublecheck the following mapping is correct
        keys_motor_new = ['gamma','mu','omega_t', 'delta','phi', 'chi']
        keys_motor_class = ['del', 'eta', 'mu', 'nu', 'phi', 'chi']
        for key1, key2 in zip(keys_motor_new, keys_motor_class):
            self.motors[key2] = motor_lib[key1]
        #now correct nu and chi between psic and sixc
        self.motors['nu']=self.motors['nu']+self.motors['mu']
        self.motors['chi']=self.motors['chi']+90


    def fit_background(self,fig,img,data=None,plot_live=False,freeze_sf = False):
        # import time
        # t1=time.time()
        I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width,bkg_sum,check_result=self.integrate_one_image(fig,img,data,plot_live=plot_live,freeze_sf = freeze_sf)
        try:
            I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width,bkg_sum,check_result=self.integrate_one_image(fig,img,data,plot_live=plot_live,freeze_sf = freeze_sf)
            self.fit_status = True
        except:
            self.fit_status = False
        # t2=time.time()
        #I_temp.append(I)
        # print(result_dict)
        self.fit_results['F']=I**0.5
        self.fit_results['Ferr']=I_err**0.5
        self.fit_results['I']=I
        self.fit_results['Ierr']=I_err#scale intensity error
        self.fit_results['bkg']=bkg_sum
        #fit parameters values
        self.opt_values['cen'] = center_pix
        self.opt_values['peak_width'] = peak_width
        self.opt_values['row_width'] = r_width
        self.opt_values['col_width'] = c_width
        self.opt_values['fit_threshold'] = s
        self.opt_values['int_power'] = ord_cus
        self.opt_values['int_dir'] = self.int_direct
        self.opt_values['cost_fun'] = self.fct
        # t3=time.time()
        # print(t3-t2,t2-t1)
        # return data
        return check_result

class background_subtraction_single_img_old():
    def __init__(self,config_file = '../config/config_bkg_sub.ini'):
        self.config_file = config_file
        self.config_file_parser(config_file)
        self.img = None
        self.opt_values = {'cen':None, 'peak_width':None, 'row_width': None, 'col_width': None,\
                           'fit_threshold': None, 'int_power': None, \
                           'int_dir':None, 'cost_fun': None}
        self.fit_results = {'F':None, 'Ferr':None, 'I':None, 'Ierr':None, 'ctot':None}

    def config_file_parser(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        for section in config.sections():
            for each in config.items(section):
                setattr(self,each[0], eval(each[1]))

    def get_image_from_nexus_file(self, nexus_path = 'C:\\apps\\DaFy_P23\\data\\test.nx5', image_path = ['scan', 'data', 'counts_XSpectrumLambda'], image_number = 0):
        data=nxload(nexus_path)
        images = eval('.'.join(['data']+image_path))
        img=np.array(images[image_number])
        return img

    def integrate_one_image(self,img, peak_width=30, plot_live = False):
        self.img = img
        center_pix=self.center_pix
        r_width=self.row_width
        c_width=self.col_width
        integration_direction=self.int_direct
        ord_cus_s=self.ord_cus_s
        ss=self.ss
        fct=self.fct
        #print(r_width,c_width)
        index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        #print(sub_index)
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        pil_y,pil_x=img.shape#shape of the pilatus image
        #reset the boundary if the index number is beyond the pilatus shape
        x_min,x_max,y_min,y_max=[int(x_min>0)*x_min,int(x_max>0)*x_max,int(y_min>0)*y_min,int(y_max>0)*y_max]
        x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,int(x_max<pil_x)*x_max,int(y_min<pil_y)*y_min,int(y_max<pil_y)*y_max]
        x_span,y_span=x_max-x_min,y_max-y_min
        #print (y_min,y_max,x_min,x_max)
        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        n=np.array(range(len(y)))
        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        fig,ax=pyplot.subplots()
        ax.imshow(img)
        rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.show()

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        s_container=[]
        ord_cus_container=[]
        index=None
        # peak_width=10
        # if self.int_direct=='y':
            # peak_width==int(self.col_width*peak_width_percent)
        # elif self.int_direct=='x':
            # peak_width=int(self.row_width*peak_width_percent)
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=int(len(y)/2)
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[0:lf]-z[0:lf]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            left_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            right_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            std_fom=np.array(left_boundary+right_boundary).std()
            return sum_temp/(len(y)-peak_width*2),std_fom#averaged offset (goodness of result), standard deviation of residual (counted as error during data integration)
            #return std_fom

        def _cal_FOM_2(y,z,s):
            y=np.array(y)
            z=np.array(z)
            y_scaled=(y-y.max())/(y.max()-y.min())+1#scaled to [0,1]
            index_container=np.where(y_scaled<s)[0]
            return np.abs(y[index_container]-z[index_container]).std()

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[index]-z[index]))
                Ibgr_container.append(abs(np.sum(z[index])))
                FOM_container.append(_cal_FOM(y,z,peak_width))
                #Ierr_container.append((I_container[-1]+FOM_container[-1])**0.5)
                Ierr_container.append((I_container[-1])**0.5+FOM_container[-1][1]+I_container[-1]*0.03)#possoin error + error from integration + 3% of current intensity
                z_container.append(z)
                s_container.append(s)
                ord_cus_container.append(ord_cus)
        index_best=np.argmin(np.array(FOM_container)[:,0])
        #print 'std=',FOM_container[index_best]
        #print 'all std=',FOM_container
        index = np.argsort(n)
        if plot_live:
            z=z_container[index_best]
            fig,ax=pyplot.subplots()
            ax.imshow(img)
            rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.figure()
            plt.plot(n[index],y[index],color='blue',label="data")
            plt.plot(n[index],z[index],color="red",label="background")
            plt.plot(n[index],y[index]-z[index],color="m",label="data-background")
            plt.plot(n[index],[0]*len(index),color='black')
            plt.legend()
            plt.show()
            print ("When s=",s_container[index_best],'pow=',ord_cus_container[index_best],"integration sum is ",np.sum(y[index]-z[index]), " counts!")
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5
        return I_container[index_best],FOM_container[index_best][1],Ierr_container[index_best],s_container[index_best],ord_cus_container[index_best],center_pix,peak_width,r_width,c_width

    def _formate_scan_from_data_info(self, I, Ierr, Ibgr):

        psicG=(self.cell, self.or0_lib,self.or1_lib,self.n_azt)
        scan_dict = {'I':[I],
                     'norm':[self.mon['norm']],
                     'Ierr':[Ierr],
                     'Ibgr':[Ibgr],
                     'dims':(1,0),
                     'transmision':[self.mon['trams']],
                     'phi':self.motors['phi'],
                     'chi':self.motors['chi'],
                     'eta':self.motors['eta'],
                     'mu':self.motors['mu'],
                     'nu':self.motors['nu'],
                     'del':self.motors['delta'],
                     'G':psicG}
        return scan_dict

    def update_geom(self,geo={'cell':[],'or0_lib':[],'or1_lib':[],'or1_lib':[],'motors':[],'mon':[],'n_azt':[]}):
        for key, value in geo.items():
            if value!=[]:
                setattr(self,key,value)
            else:
                pass

    def update_motor_angles(self, motor_lib):
        # keys_motor_new = ['gamma','delta','mu','omega_t', 'phi', 'chi']
        #doublecheck the following mapping is correct
        keys_motor_new = ['gamma','mu','omega_t', 'delta','phi', 'chi']
        keys_motor_class = ['del', 'eta', 'mu', 'nu', 'phi', 'chi']
        for key1, key2 in zip(keys_motor_new, keys_motor_class):
            self.motors[key2] = motor_lib[key1]
        #now correct nu and chi between psic and sixc
        self.motors['nu']=self.motors['nu']+self.motors['mu']
        self.motors['chi']=self.motors['chi']+90


    def fit_background_corr(self,img,peak_width,plot_live=False):
        I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image(img,peak_width,plot_live=plot_live)
        try:
            I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image(img,peak_width,plot_live=plot_live)
        except:
            self.img=img
            I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width=1,0,0,2,0.1,[100,100],10,10,10
        #I_temp.append(I)
        scan_dict=self._formate_scan_from_data_info(I,I_err,I_bgr)
        #calculate the correction factor
        result_dict=ctr_data.image_point_F(scan_dict,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=self.corr_params, preparsed=True)
        # print(result_dict)
        self.fit_results['F']=result_dict['F']
        self.fit_results['Ferr']=result_dict['Ferr']
        self.fit_results['I']=result_dict['F']**2
        self.fit_results['Ierr']=I_err*result_dict['F']**2/I#scale intensity error
        self.fit_results['ctoto']=result_dict['ctot']
        self.fit_results['alpha']=result_dict['alpha']
        self.fit_results['beta']=result_dict['beta']
        #fit parameters values
        self.opt_values['cen'] = center_pix
        self.opt_values['peak_width'] = peak_width
        self.opt_values['row_width'] = r_width
        self.opt_values['col_width'] = c_width
        self.opt_values['fit_threshold'] = s
        self.opt_values['int_power'] = ord_cus
        self.opt_values['int_dir'] = self.int_direct
        self.opt_values['cost_fun'] = self.fct
