from PyMca5.PyMcaPhysics import SixCircle 
from PyMca5.PyMcaIO import specfilewrapper
import edf_image_v3 as edf_image
import numpy as np
from scipy.interpolate import griddata    
from scipy.optimize import brenth

def get_K_0(E_keV): #AA-1
    return 2.*np.pi/12.39854*E_keV

def get_q_from_HKL(H, K, L, theta, a_star, c_star):
    qx = (H+K/2.)*a_star
    qy = np.sqrt(3)/2.*K*a_star
    qz = L*c_star
    qRx = np.cos(np.deg2rad(theta))*qx-np.sin(np.deg2rad(theta))*qy
    qRy = np.sin(np.deg2rad(theta))*qx+np.cos(np.deg2rad(theta))*qy
    return (qRx, qRy, qz)

class DetectorImg(): 
    def __init__(self, E_keV, cen=(0,0), pixelsize=(0,0), sdd=0.0, spec_filename="", edf_path=""):
        self.set_params(E_keV, cen, pixelsize, sdd, spec_filename, edf_path)
        self.UB = [1,0,0,0,1,0,0,0,1]
    
    def set_params(self, E_keV=0.0, cen=(0,0), pixelsize=(0,0), sdd=0.0, spec_filename="", edf_path=""):
        self.E_keV = E_keV
        self.wavelength = 12.39854*self.E_keV
        self.k0 = 2.*np.pi/self.wavelength
        self.cen = cen
        self.pixelsize = pixelsize
        self.sdd = sdd
        self.spec = specfilewrapper.Specfile(spec_filename)
        self.img_loader = edf_image.edf_image_loader(spec_filename, edf_path)
        
    def load_frame(self, scan_no, frame_no, gz_compressed=False, normalize=False, monitor_name=None, monitor_names=None, remove_rows=None, remove_cols=None):
        return self.img_loader.load_frame(scan_no, frame_no, gz_compressed, normalize, monitor_name, monitor_names, remove_rows, remove_cols)
        
    def _prepare_frame(self, scan_no, frame_no, norm_mon=True, norm_transm=True, gz_compressed=False):
        img = self.img_loader.load_frame(scan_no, frame_no, gz_compressed)        
        scan = self.spec.select('{0}.1'.format(scan_no))
        
        UB = np.array(scan.header('G')[2].split(' ')[-9:],dtype=np.float)
        self.UB = UB
        chi_ = scan.motorpos('Chi')
        phi_ = scan.motorpos('Phi')
        if('ccoscan' in scan.header('S')[0]):
            th_ = scan.datacol('zap_thcnt')[frame_no]
            gam_ = scan.datacol('zap_gamcnt')[frame_no]
            del_ = scan.datacol('zap_delcnt')[frame_no]
            mon_ = scan.datacol('zap_mon')[frame_no]
            transm_ = scan.datacol('zap_transm')[frame_no]
            mu_ = scan.datacol('zap_mucnt')[frame_no]
        else:
            th_ = scan.datacol('thcnt')[frame_no]
            gam_ = scan.datacol('gamcnt')[frame_no]
            del_ = scan.datacol('delcnt')[frame_no]
            mon_ = scan.datacol('mon')[frame_no]
            transm_ = scan.datacol('transm')[frame_no]
            mu_ = scan.datacol('mucnt')[frame_no]
        
        intensity = img.img 
        if(norm_mon):
            intensity /= mon_ 
        if(norm_transm):
            intensity /= transm_ 

        gamma_range = -np.arctan((np.arange(intensity.shape[1])-self.cen[1])*self.pixelsize[1]/self.sdd)*180/ np.pi + gam_
        delta_range = np.arctan((np.arange(intensity.shape[0])-self.cen[0])*self.pixelsize[0]/self.sdd)*180/ np.pi + del_
        
        #polarisation correction
        delta_grid, gamma_grid = np.meshgrid(delta_range, gamma_range)
        Pver = 1 - np.sin(delta_grid * np.pi / 180.)**2 * np.cos(gamma_grid * np.pi / 180.)**2
        intensity /= Pver
        
        return intensity, (UB, gamma_range, delta_range, th_, mu_, chi_, phi_)
        
    def _get_HKL(self, params):
        UB, gamma_range, delta_range, th_, mu_, chi_, phi_ = params
        #self._prepare_frame(scan_no, frame_no, norm_mon, norm_transm)
        d = SixCircle.SixCircle()
        d.setEnergy(self.E_keV)
        d.setUB(UB)
        HKL = d.getHKL(delta=delta_range, theta=th_, chi=chi_, phi=phi_, mu=mu_, gamma=gamma_range, gamma_first=False)
        shape = gamma_range.size, delta_range.size
        H = HKL[0,:].reshape(shape)
        K = HKL[1,:].reshape(shape)
        L = HKL[2,:].reshape(shape)
        
        return (H, K, L)
    
    def _get_q(self, params):
        UB, gamma_range, delta_range, th_, mu_, chi_, phi_ = params
        d = SixCircle.SixCircle()
        d.setEnergy(self.E_keV)
        d.setUB(UB)
        #Q = d.getQLab(mu=mu_, delta=delta_range, gamma=gamma_range, gamma_first=False)
        Q = d.getQSurface(theta=th_, chi=chi_, phi=phi_, mu=mu_, delta=delta_range, gamma=gamma_range, gamma_first=False)
        shape = gamma_range.size, delta_range.size
        qx = Q[0,:].reshape(shape)
        qy = Q[1,:].reshape(shape)
        qz = Q[2,:].reshape(shape)
        return (qx, qy, qz)

    def get_HKL(self, scan_no, frame_no, norm_mon=True, norm_transm=True):
        intensity, params = self._prepare_frame(scan_no, frame_no, norm_mon=norm_mon, norm_transm=norm_transm)
        return (intensity, self._get_HKL(params))
    
    def get_q(self, scan_no, frame_no, norm_mon=True, norm_transm=True, gz_compressed=False):
        intensity, params = self._prepare_frame(scan_no, frame_no, norm_mon=norm_mon, norm_transm=norm_transm, gz_compressed=gz_compressed)
        return (intensity, self._get_q(params))
    
    def get_HKL_q(self, scan_no, frame_no, norm_mon=True, norm_transm=True):
        intensity, params = self._prepare_frame(scan_no, frame_no, norm_mon=norm_mon, norm_transm=norm_transm)
        return (intensity, self._get_HKL(params), self._get_q(params))
    
    def get_grid_q_in_out_plane(self, scan_no, frame_no, norm_mon=True, norm_transm=True):
        intensity, q = self.get_q(scan_no, frame_no, norm_mon=norm_mon, norm_transm=norm_transm)
        qx, qy, qz = q
        q_para = np.sqrt(qx**2 + qy**2)
        size = intensity.shape
        grid_q_perp, grid_q_para = np.mgrid[np.max(qz):np.min(qz):(1.j*size[1]), np.min(q_para):np.max(q_para):(1.j*size[0])]
        grid_intensity = griddata((q_para.ravel(), qz.ravel()), intensity.ravel(), (grid_q_para, grid_q_perp), method='nearest')
        return (grid_q_para, grid_q_perp, grid_intensity)   
        

if __name__ == '__main__':
    spec_filename = '/home/finn/data/2016_06_MA2858/ma2858_sixcvertical.spec'
    edf_path =  '/home/finn/data/2016_06_MA2858/ma2858_img/'
    
    scan_no = 1055
    frame_no = 10
    
    DI = DetectorImg(22.5, (128, 385), (0.055, 0.055), 914, spec_filename, edf_path)
    intensity, HKL, q = DI.get_HKL_q(scan_no, frame_no, norm_mon=True, norm_transm=True)
    
    H, K, L = HKL
    qx, qy, qz = q
    q_para = np.sqrt(qx**2 + qy**2)
    grid_q_para, grid_q_perp, grid_intensity = DI.get_grid_q_in_out_plane(scan_no, frame_no)
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.imshow(grid_intensity, vmin=0, vmax=0.05)
    
    plt.figure()
    plt.pcolormesh(grid_q_para, grid_q_perp, grid_intensity, vmin=0, vmax=0.05)
    plt.xlabel(r'$q_{\parallel}$', fontsize=25)
    plt.ylabel(r'$q_{\perp}$', fontsize=25)
    
    plt.figure()
    plt.pcolormesh(K, L, intensity, vmin=0, vmax=0.05)
    plt.xlabel(r'K', fontsize=25)
    plt.ylabel(r'L', fontsize=25)
    
    plt.figure()
    plt.pcolormesh(q_para, qz, intensity, vmin=0, vmax=0.05)
    plt.xlabel(r'$q_{\parallel}$', fontsize=25)
    plt.ylabel(r'$q_{\perp}$', fontsize=25)
    
    plt.show()
    
