import h5py
import numpy as np
from subprocess import call
import binoculars

H=0
K=1
L=2


def convert_scan_to_HKL_map(BINocular_config_filename, scan_no):
    #TODO: this should work directly without a config file
    call(['python', '/home/finn/software/binoculars-master/binoculars.py', 'process', BINocular_config_filename, str(scan_no)])
    

def find_bin(min_max, axis, ax_range, res):
    _min = 0
    _max = ax_range[axis][1]-ax_range[axis][0]
    for i in xrange(0, ax_range[axis][1]-ax_range[axis][0]+1):
        if ((ax_range[axis][0]+i)*res[axis] < min_max[0]):
            _min = i
        elif ((ax_range[axis][0]+i)*res[axis] > min_max[1]):
            _max = i
            break
    return (_min, _max)

class HKL_limits():
    def __init__(self, H_lim=None, K_lim=None, L_lim=None):
        self.H_lim = H_lim
        self.K_lim = K_lim
        self.L_lim = L_lim

class detector_image():
    def __init__(self, filename, ax_range=None, res=None, counts=None, contributions=None):
        if(filename):
            fp = h5py.File(filename, 'r')
            if 'binoculars' in fp:
                fp = fp['binoculars']
                self.ax_range = np.array((fp['axes']['H'][4:], fp['axes']['K'][4:], fp['axes']['L'][4:]), dtype=int)
                self.res = (fp['axes']['H'][3], fp['axes']['K'][3], fp['axes']['L'][3])
            else:
                self.ax_range = fp['axes_range']
                self.res = fp['axes_res']
            self.counts = np.array(fp['counts'])
            self.counts.astype(float)
            self.contributions = np.array(fp['contributions'])
            self.contributions.astype(float)
            self.shape = self.counts.shape
        else:
            self.ax_range = ax_range
            self.res = res
            self.counts = counts
            self.counts.astype(float)
            self.contributions = contributions
            self.contributions.astype(float)
            self.shape = self.counts.shape

        H_lim = self.ax_range[0]*self.res[0]
        K_lim = self.ax_range[1]*self.res[1]
        L_lim = self.ax_range[2]*self.res[2]
        self.HKL_lim = [H_lim, K_lim, L_lim]
        
    def get_sub_volume(self, HKL_Limits):
        H_minmax_index = (0, self.counts.shape[H]-1)
        K_minmax_index = (0, self.counts.shape[K]-1)
        L_minmax_index = (0, self.counts.shape[L]-1)

        if(HKL_Limits.H_lim):
            H_minmax_index = find_bin(HKL_Limits.H_lim, H, self.ax_range, self.res)
        if(HKL_Limits.K_lim):
            K_minmax_index = find_bin(HKL_Limits.K_lim, K, self.ax_range, self.res)
            print K_minmax_index
        if(HKL_Limits.L_lim):
            L_minmax_index = find_bin(HKL_Limits.L_lim, L, self.ax_range, self.res)

        H_min = H_minmax_index[0]
        H_max = H_minmax_index[1]
        K_min = K_minmax_index[0]
        K_max = K_minmax_index[1]
        L_min = L_minmax_index[0]
        L_max = L_minmax_index[1]
                
        counts_new = self.counts[H_min:H_max, K_min:K_max, L_min:L_max]
        contributions_new = self.contributions[H_min:H_max, K_min:K_max, L_min:L_max]
        
        ax_range_new = np.zeros((3,2), dtype=int)
        ax_range_new[H][0] = self.ax_range[H][0] + H_min
        ax_range_new[H][1] = self.ax_range[H][0] + H_max
        ax_range_new[K][0] = self.ax_range[K][0] + K_min
        ax_range_new[K][1] = self.ax_range[K][0] + K_max
        ax_range_new[L][0] = self.ax_range[L][0] + L_min
        ax_range_new[L][1] = self.ax_range[L][0] + L_max
        
        return detector_image(None, ax_range_new, self.res, counts_new, contributions_new)    

    def project(self, projection_axis):
        counts_2d = np.sum(self.counts, axis=projection_axis)
        contributions_2d = np.sum(self.contributions, axis=projection_axis)
        mask = contributions_2d==0
        intensity = np.zeros(counts_2d.shape)
        intensity[~mask] = (counts_2d/contributions_2d)[~mask]
        return intensity
    
    def getHKLlimits(self):
        return self.HKL_lim
    
    def getAxRange(self):
        return self.ax_range
    
    def getResolution(self):
        return self.res
    
    def getHKLfromXYZ(self, x, y, z):
        # x, y, z from 0 to size_x/y/z
        return ((self.ax_range[0][0]+x)*self.res[0], (self.ax_range[1][0]+y)*self.res[1], (self.ax_range[2][0]+z)*self.res[2])
    
    def getXYZfromHKL(self, h, k, l):
        x = find_bin((h, h), H, self.ax_range, self.res)[0]
        y = find_bin((h, h), H, self.ax_range, self.res)[0]
        z = find_bin((h, h), H, self.ax_range, self.res)[0]
        return (x, y, z)