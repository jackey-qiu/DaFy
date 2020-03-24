from PyMca5.PyMcaPhysics import SixCircle 
import nexus_image
import numpy as np
from scipy.interpolate import griddata    
from scipy.optimize import brenth
import Fio
from util.HKLVlieg import Crystal, printPos, UBCalculator, VliegAngles, printPos_prim, vliegDiffracAngles
import matplotlib.pyplot as plt

INCIDENT_ANGLE=0.4
PHI=0
CHI=0

lattice_constants=[2.8837,2.8837,7.0636,90,90,120]
energy=18.739
or0_angles=[INCIDENT_ANGLE,15.4,22.43,-30.9,PHI,CHI]
or1_angles=[INCIDENT_ANGLE,7.61,13.63,-38.,PHI,CHI]
or0_hkl=[1.0009,-1.0009,4.0359]
or1_hkl=[0.0,-0.5045,2.5225]

def get_K_0(E_keV): #AA-1
    return 2.*np.pi/12.39854*E_keV

def get_q_from_HKL(H, K, L, theta, a_star, c_star):
    qx = (H+K/2.)*a_star
    qy = np.sqrt(3)/2.*K*a_star
    qz = L*c_star
    qRx = np.cos(np.deg2rad(theta))*qx-np.sin(np.deg2rad(theta))*qy
    qRy = np.sin(np.deg2rad(theta))*qx+np.cos(np.deg2rad(theta))*qy
    return (qRx, qRy, qz)

def cal_UB(lattice_constants=[2.8837,2.8837,7.0636,90,90,120],energy=18.739,or0_angles=[0.4,15.4,22.43,-30.9,0.,0.],or1_angles=[0.4,7.61,13.63,-38.,0.,0.],or0_hkl=[1.0009,1.-1.0009,4.0359],or1_hkl=[0.0,-0.5045,2.5225]):
    substrate=Crystal(lattice_constants[0:3],lattice_constants[3:])
    ub_substrate=UBCalculator(substrate,energy)
    or0_angles=np.deg2rad(or0_angles)
    or1_angles=np.deg2rad(or1_angles)
    or0_angles = vliegDiffracAngles(or0_angles)
    or1_angles = vliegDiffracAngles(or1_angles)
    ub_substrate.setPrimaryReflection(or0_angles,or0_hkl)
    ub_substrate.setSecondayReflection(or1_angles,or1_hkl)
    ub_substrate.calculateU()
    return ub_substrate.getUB()

UB=cal_UB(lattice_constants,energy,or0_angles,or1_angles,or0_hkl,or1_hkl)

class DetectorImg(): 
    def __init__(self, E_keV=19.5, cen=(234,745), pixelsize=(0.055,0.055), sdd=714, UB=UB,spec_filename='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/startup/FirstTest_00671.fio', edf_path='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/FirstTest_00671/lmbd'):
        self.set_params(E_keV, cen, pixelsize, sdd, spec_filename, edf_path,UB)
    
    def set_params(self, E_keV=0.0, cen=(0,0), pixelsize=(0,0), sdd=0.0, spec_filename="", nexus_path="",UB=[]):
        self.E_keV = E_keV
        self.wavelength = 12.39854*self.E_keV
        self.k0 = 2.*np.pi/self.wavelength
        self.cen = cen
        self.pixelsize = pixelsize
        self.sdd = sdd
        self.fio = Fio.Fiofile(spec_filename)
        self.img_loader = nexus_image.nexus_image_loader(spec_filename, nexus_path)
        self.UB=UB
        
    def load_frame(self, scan_no, frame_no,frame_prefix):
        return self.img_loader.load_frame(scan_no, frame_no,frame_prefix)

    def _prepare_frame(self, scan_no, frame_no, frame_prefix,norm_mon=True, norm_transm=True,UB=UB,trans='attenpos',mon='avg_beamcurrent'):
        self.scan_no = scan_no
        self.frame_no = frame_no
        self.frame_prefix = frame_prefix
        img =self.img_loader.load_frame(scan_no, frame_no,frame_prefix)        
        self.UB = UB
        transm_=self.fio.get_col(trans)[frame_no]
        mon_=self.fio.get_col(mon)[frame_no]
        th_=self.fio.get_col('mu')[frame_no]
        gam_=self.fio.get_col('delta')[frame_no]
        del_=self.fio.get_col('gamma')[frame_no]
        #the chi and phi values are arbitrary in the fio file, should be set to the same values as the ones that are usd to cal UB matrix(all 0 so far)
        phi_=self.fio.get_col('phi')[frame_no]*0
        chi_=self.fio.get_col('chi')[frame_no]*0
        mu_=self.fio.get_col('omega_t')[frame_no]
        # print del_,gam_
        del_,gam_=np.rad2deg(vliegDiffracAngles(np.deg2rad([0.4,del_,gam_,mu_,0,0]))[1:3])
        # print del_,gam_
        intensity = img 
        #let us remove the pixs at the edges, where intensity are unreasonablly high
        intensity[:,range(5)]=intensity[:,5].mean()
        intensity[:,[-1,-2,-3,-4,-5]]=intensity[:,-6].mean()
        intensity[range(5),:]=intensity[5,:].mean()
        intensity[[-1,-2,-3,-4,-5],:]=intensity[-6,:].mean()
        # if(norm_mon):
            # intensity /= mon_ 
        # if(norm_transm):
            # intensity *= transm_ 
        #detector dimension is (516,1556)
        #You may need to put a negative sign in front, check the rotation sense of delta and gamma motors at P23
        # delta_range = np.arctan((np.arange(intensity.shape[0])-self.cen[0])*self.pixelsize[0]/self.sdd)*180/ np.pi + del_
        # gamma_range = -np.arctan((np.arange(intensity.shape[1])-self.cen[1])*self.pixelsize[1]/self.sdd)*180/ np.pi + gam_
        delta_range = np.arctan((np.arange(intensity.shape[1])-self.cen[0])*self.pixelsize[0]/self.sdd)*180/ np.pi + del_
        #the minus sign here because the column index increase towards bottom, then 0 index(top most) will give a negative gam offset
        #a minus sign in front correct this.
        gamma_range =-np.arctan((np.arange(intensity.shape[0])-self.cen[1])*self.pixelsize[1]/self.sdd)*180/ np.pi + gam_
        #polarisation correction
        # TODO: what is this doing?
        delta_grid , gamma_grid= np.meshgrid(delta_range,gamma_range)
        # gamma_grid,delta_grid = np.mgrid[gamma_range,delta_range]
        # print delta_grid.shape,gamma_grid.shape
        # gamma_grid,delta_grid = np.meshgrid(delta_range,gamma_range)
        Pver = 1 - np.sin(delta_grid * np.pi / 180.)**2 * np.cos(gamma_grid * np.pi / 180.)**2
        intensity=np.divide(intensity,Pver)
        
        return intensity, (UB, gamma_range, delta_range, th_, mu_, chi_, phi_)
        
    def _get_HKL(self, params):
        UB, gamma_range, delta_range, th_, mu_, chi_, phi_ = params
        d = SixCircle.SixCircle()
        d.setEnergy(self.E_keV)
        d.setUB(UB)
        HKL = d.getHKL(delta=delta_range, theta=th_, chi=chi_, phi=phi_, mu=mu_, gamma=gamma_range, gamma_first=False)
        shape =  gamma_range.size,delta_range.size
        # shape =  delta_range.size,gamma_range.size
        H = HKL[0,:].reshape(shape)
        K = HKL[1,:].reshape(shape)
        L = HKL[2,:].reshape(shape)
        return (H, K, L)
    
    def _get_q(self, params):
        UB, gamma_range, delta_range, th_, mu_, chi_, phi_ = params
        d = SixCircle.SixCircle()
        d.setEnergy(self.E_keV)
        d.setUB(UB)
        Q = d.getQSurface(theta=th_, chi=chi_, phi=phi_, mu=mu_, delta=delta_range, gamma=gamma_range, gamma_first=False)
        # shape =  gamma_range.size,delta_range.size
        # print 'size of gamma_range',len(gamma_range)
        # print 'size of delta_range',len(delta_range)
        # shape =  delta_range.size,gamma_range.size
        shape =  gamma_range.size,delta_range.size
        # print 'shape of Q',shape
        qx = Q[0,:].reshape(shape)
        qy = Q[1,:].reshape(shape)
        qz = Q[2,:].reshape(shape)
        return (qx, qy, qz)

    def get_HKL(self, scan_no, frame_no, frame_prefix,norm_mon=True, norm_transm=True):
        intensity, params = self._prepare_frame(scan_no, frame_no, frame_prefix,norm_mon=norm_mon, norm_transm=norm_transm)
        return (intensity, self._get_HKL(params))
    
    def get_q(self, scan_no, frame_no, frame_prefix,norm_mon=True, norm_transm=True):
        intensity, params = self._prepare_frame(scan_no, frame_no, frame_prefix,norm_mon=norm_mon, norm_transm=norm_transm)
        return (intensity, self._get_q(params))
    
    def get_HKL_q(self, scan_no, frame_no, frame_prefix,norm_mon=True, norm_transm=True):
        intensity, params = self._prepare_frame(scan_no, frame_no, frame_prefix,norm_mon=norm_mon, norm_transm=norm_transm)
        return (intensity, self._get_HKL(params), self._get_q(params))
    
    def get_grid_q_in_out_plane(self, scan_no, frame_no,frame_prefix,norm_mon=False, norm_transm=False):
        intensity, q = self.get_q(scan_no, frame_no,frame_prefix, norm_mon=norm_mon, norm_transm=norm_transm)
        qx, qy, qz = q
        q_para = np.sqrt(qx**2 + qy**2)
        size = intensity.shape
        #shape=(vertical,horizontal),len(vertical)=size(1),len(horizontal)=size(0)
        grid_q_perp, grid_q_para = np.mgrid[np.max(qz):np.min(qz):(1.j*size[0]), np.min(q_para):np.max(q_para):(1.j*size[1])]
        grid_intensity = griddata((q_para.ravel(), qz.ravel()), intensity.ravel(), (grid_q_para, grid_q_perp), method='nearest')
        # print intensity.shape
        # print grid_intensity.shape
        # print grid_q_perp.shape
        return (grid_q_para, grid_q_perp, grid_intensity)  

    def show_image(self):
        grid_q_para, grid_q_perp, grid_intensity = self.get_grid_q_in_out_plane(self.scan_no, self.frame_no,self.frame_prefix)
        plt.figure()
        # plt.imshow(grid_intensity, vmin=0, vmax=100.05)
        plt.imshow(grid_intensity,cmap='jet',vmin=0, vmax=100.05)
        plt.title("plt.imshow(grid_intensity")
        # plt.colorbar(extend='both',orientation='Vertical')
        plt.clim(0,90)
        plt.show()
        

if __name__ == '__main__':
    scan_no=666
    frame_no = 0

    spec_filename = '/home/qiu/data/beamtime/P23_11_18_I20180114/raw/startup/FirstTest_00{}.fio'.format(scan_no)
    edf_path ='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/FirstTest_00{}/lmbd'.format(scan_no)
    frame_prefix='FirstTest'

    UB=cal_UB()
    #center_x is index of horizontal direction towards right
    #center_y is index of vertical direction towards bottom
    #The corresponding numpy array index is actually [center_y,center_x]
    center_pix=(275,770)
    pix_size=(0.055,0.055)
    sdd=700

    DI = DetectorImg(energy, center_pix, pix_size, sdd,cal_UB(), spec_filename, edf_path)
    intensity, HKL, q = DI.get_HKL_q(scan_no, frame_no,frame_prefix, norm_mon=False, norm_transm=False)
    
    H, K, L = HKL
    qx, qy, qz = q
    q_para = np.sqrt(qx**2 + qy**2)
    grid_q_para, grid_q_perp, grid_intensity = DI.get_grid_q_in_out_plane(scan_no, frame_no,frame_prefix)
    
    import matplotlib.pyplot as plt
    DI.img_loader.show_frame(scan_no,frame_no)

    plt.figure()
    # plt.imshow(grid_intensity, vmin=0, vmax=100.05)
    plt.imshow(grid_intensity,cmap='jet')
    plt.title("plt.imshow(grid_intensity")
    # plt.colorbar(extend='both',orientation='Vertical')
    plt.clim(0,90)
    print 'grid_intensity.shape',grid_intensity.shape

    plt.figure()
    plt.pcolormesh(grid_q_para, grid_q_perp, grid_intensity, vmin=0, vmax=90,cmap='jet')
    plt.xlabel(r'$q_{\parallel}$', fontsize=25)
    plt.ylabel(r'$q_{\perp}$', fontsize=25)
    plt.title("plt.pcolormesh(grid_q_para, grid_q_perp,qrid_intensity")
    print 'grid_q_perp.shape',grid_q_perp.shape

    plt.figure()
    plt.pcolormesh(K, L, intensity, vmin=0, vmax=90,cmap='jet')
    plt.xlabel(r'K', fontsize=25)
    plt.ylabel(r'L', fontsize=25)
    plt.title("plt.pcolormesh(K, L, intensity")
    print 'intensity.shape',intensity.shape

    plt.figure()
    plt.pcolormesh(q_para, qz, intensity, vmin=0, vmax=90,cmap='jet')
    plt.title("plt.pcolormesh(q_para, qz, intensity")
    plt.xlabel(r'$q_{\parallel}$', fontsize=25)
    plt.ylabel(r'$q_{\perp}$', fontsize=25)
    
    plt.show()
   
