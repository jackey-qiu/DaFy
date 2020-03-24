from PyMca5.PyMca import specfilewrapper, EdfFile, SixCircle, specfile
from pyspec import spec
import numpy as np
import edf_image
import scipy.interpolate
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import ndimage
import pyspec
from pyspec import spec
from pyspec import fit
from scipy.interpolate import interp1d

import sys
sys.path.append('..')
import ids_reader.ids_reader as ir

ec_path =   '/home/finn/data/2014_12_04_MA2254/MA2254_EC/'
spec_path = '/home/finn/data/2014_12_04_MA2254/MA2254_Spec/'
edf_path =  '/home/finn/data/2014_12_04_MA2254/MA2254_IMG/'

def get_point_params(scan, frame_no):
    GAM, DEL, TH, CHI, PHI, MU, MON, TRANSM = range(8)
    params = np.zeros(8) # gamma delta theta chi phi mu mon transm
    params[CHI] = scan.motorpos('Chi')
    params[PHI] = scan.motorpos('Phi')
    params[TH] = scan.datacol('thcnt')[frame_no]
    params[GAM] = scan.datacol('gamcnt')[frame_no]
    params[DEL] = scan.datacol('delcnt')[frame_no]
    params[MON] = scan.datacol('mon')[frame_no]
    params[TRANSM] = scan.datacol('transm')[frame_no]
    params[MU] = scan.datacol('mucnt')[frame_no]
    return params

def process_image(scanparams, pointparams, image, detector_info):
    gamma, delta, theta, chi, phi, mu, mon, transm = pointparams
    wavelength, UB = scanparams
    centralpixel, sdd, pixelsize, ymask, xmask = detector_info
    
    print 'gamma: {0}, delta: {1}, theta: {2}, mu: {3}'.format(gamma, delta, theta, mu)

    # pixels to angles
    pixelsize = np.array(pixelsize) 
    app = np.arctan(pixelsize / sdd) * 180 / np.pi

    gamma_range= -app[1] * (np.arange(image.shape[1]) - centralpixel[1]) + gamma
    delta_range= app[0] * (np.arange(image.shape[0]) - centralpixel[0]) + delta

    # masking
    gamma_range = gamma_range[ymask[0]:ymask[1]+1]
    delta_range = delta_range[xmask[0]:xmask[1]+1]
    
    roi = image[ymask[0]:ymask[1]+1, :]
    roi[:, xmask[0]:xmask[1]+1]
    intensity = roi

    #polarisation correction
    delta_grid, gamma_grid = np.meshgrid(delta_range, gamma_range)
    Pver = 1 - np.sin(delta_grid * np.pi / 180.)**2 * np.cos(gamma_grid * np.pi / 180.)**2
    intensity /= Pver

    return intensity, (wavelength, UB, gamma_range, delta_range, theta, mu, chi, phi)

def project(wavelength, UB, gamma, delta, theta, mu, chi, phi):
    R = SixCircle.getHKL(wavelength, UB, gamma=gamma, delta=delta, theta=theta, mu=mu, chi=chi, phi=phi)
    shape = gamma.size, delta.size
    H = R[0,:].reshape(shape)
    K = R[1,:].reshape(shape)
    L = R[2,:].reshape(shape)
    return (H, K, L)

def convert_image(scan_no, frame_no, detector_info):
                
    _spec = specfilewrapper.Specfile(spec_path + 'ma2254_sixcvertical.spec') 
    scan = _spec.select('{0}.1'.format(scan_no))
    
    UB = np.array(scan.header('G')[2].split(' ')[-9:],dtype=np.float)
    wavelength = float(scan.header('G')[1].split(' ')[-1])
    scanparams = wavelength, UB

    pointparams = get_point_params(scan, frame_no) # 2D array of diffractometer angles + mon + transm

    img_loader = edf_image.edf_image_loader(spec_path + 'ma2254_sixcvertical.spec')
    img = img_loader.load_frame(edf_path, scan_no, frame_no, gz_compressed=True, normalize=True, monitor_names=['transm', 'mon'])
    
    proc_img, img_params = process_image(scanparams, pointparams, img, detector_info)
    wavelength, UB, gamma, delta, theta, mu, chi, phi = img_params
    HKL = project(wavelength, UB, gamma, delta, theta, mu, chi, phi)
    return HKL, proc_img
  
  
'''# Determine the potential during scan
f = spec.SpecDataFile(spec_path + 'ma2254_sixcvertical.spec')
ids_f = ir.ids_file(ec_path + 'CV_AuCo_086.ids')

V_ids = ids_f.datasets[1].col1
I_ids = ids_f.datasets[1].col2*1e3

t_ids = np.arange(0, len(ids_f.datasets[1].col1))*0.1
j_ids = I_ids/(np.pi*0.2**2)

V_t = interp1d(t_ids, V_ids, bounds_error=False, fill_value=-1.05)
t = f[416].Time-11
V_XRD = V_t(t)

plt.figure()
plt.plot(V_XRD)
plt.show()'''
    
do_plot = False

scan_no = 416

centralpixel = (388, 338)
sdd = 961 # sample to detector distance (mm)
pixelsize = (0.055, 0.055) # pixel size x/y (mm)
ymask = (0, 515) 
xmask = (0, 515)
detector_info = (centralpixel, sdd, pixelsize, ymask, xmask)


img_loader = edf_image.edf_image_loader(spec_path + 'ma2254_sixcvertical.spec')

FWHM_Ls = []
FWHM_Ks = []
Ls = []
Ks = []

plt.figure()
ax1 = plt.subplot()

for frame_no in xrange(img_loader.get_no_frames(scan_no)):

    HKL, intensity = convert_image(scan_no, frame_no, detector_info)

    img = img_loader.load_frame(edf_path, scan_no, frame_no, gz_compressed=True, normalize=True, monitor_names=['transm', 'mon'])
    
    _H = np.array(HKL)[0]
    _K = np.array(HKL)[1]
    _L = np.array(HKL)[2]
    _int = np.array(intensity)
    
    # Determine COM
    COM = ndimage.measurements.center_of_mass(img)
    print 'COM(xy) =', np.round(COM, 0) 
    print 'COM: H =', _H[COM[0], COM[1]], '   K =', _K[COM[0], COM[1]], '   L =', _L[COM[0], COM[1]]
    
    if(do_plot):
        plt.figure()
        plt.imshow(img)
    
    # Fit L cuts
    L_cut_half_width = 10
    L_cut_img = img[:, COM[1]-L_cut_half_width:COM[1]+L_cut_half_width+1] #img[: , 372:408] 
    L_cut = np.sum(L_cut_img, axis=1)
    L_cut[255:261] = (L_cut[253]+L_cut[254]+L_cut[261]+L_cut[262])/4. # remove border between ccd panels
    L_cut[338] = (L_cut[337]+L_cut[339])/2. # remove dead pixel
    
    df = fit.fit(x=np.arange(516), y=L_cut, xlimits=[66, 515], funcs=[pyspec.fitfuncs.pvoight, pyspec.fitfuncs.constant])
    df.run()   
    
    _x = np.arange(800)
    _y = pyspec.fitfuncs.pvoight(_x, df.result[0:4])+pyspec.fitfuncs.constant(_x, df.result[4:])
    
    
    if(do_plot):
        plt.figure()
        plt.plot(L_cut)
        plt.plot(_x, _y)
    
    L_params = df.result
    print df.result
    
    # Fit K cut
    K_cut_half_width = 28
    K_cut_img = img[int(np.round(L_params[0], 0)-K_cut_half_width):int(np.round(L_params[0], 0)+K_cut_half_width+1), :]
    K_cut = np.sum(K_cut_img, axis=0)
    df = fit.fit(x=np.arange(516), y=K_cut, xlimits=[260, 515], funcs=[pyspec.fitfuncs.pvoight, pyspec.fitfuncs.constant])
    df.run()   
    
    
    _x = np.arange(700)
    _y = pyspec.fitfuncs.pvoight(_x, df.result[0:4])+pyspec.fitfuncs.constant(_x, df.result[4:])
    
    if(frame_no%10 == 0):
        ax1.plot(_x, _y)
    
    if(do_plot):
        plt.figure()
        plt.plot(K_cut)
        plt.plot(_x, _y)
    
    K_params = df.result
    print df.result
    
    # Print final parameters
    LL = interp1d(np.arange(516), _L[:, int(np.round(K_params[0],0))])
    KK = interp1d(np.arange(516), _K[int(np.round(L_params[0],0)), :])

    L_pos  = LL(L_params[0]) #_L[int(np.round(L_params[0], 0)), int(np.round(K_params[0],0))]
    K_pos  = KK(K_params[0]) #_K[int(np.round(L_params[0], 0)), int(np.round(K_params[0],0))]
    
    
    L_FWHM = np.abs(LL(L_params[0]+L_params[1]/2.)- LL(L_params[0]-L_params[1]/2.))
    K_FWHM = np.abs(KK(K_params[0]+K_params[1]/2.)- KK(K_params[0]-K_params[1]/2.))
    #L_FWHM = np.abs(_L[int(np.round(L_params[0]+L_params[1]/2., 0)), int(np.round(K_params[0],0))] - _L[int(np.round(L_params[0]-L_params[1]/2., 0)), int(np.round(K_params[0],0))])
    #K_FWHM = np.abs(_K[int(np.round(L_params[0], 0)), int(np.round(K_params[0]+K_params[1]/2.,0))] - _K[int(np.round(L_params[0], 0)), int(np.round(K_params[0]-K_params[1]/2.,0))])
    
    FWHM_Ls.append(L_FWHM)
    FWHM_Ks.append(K_FWHM)
    Ls.append(L_pos)
    Ks.append(K_pos)
    
    print 'L_pos = %.5f   L_FWHM = %.5f' % (L_pos, L_FWHM)
    print 'K_pos = %.5f   K_FWHM = %.5f' % (K_pos, K_FWHM)

    if(do_plot):
        plt.show()



# Determine the potential during scan
f = spec.SpecDataFile(spec_path + 'ma2254_sixcvertical.spec')
ids_f = ir.ids_file(ec_path + 'CV_AuCo_086.ids')

V_ids = ids_f.datasets[1].col1
I_ids = ids_f.datasets[1].col2*1e3

t_ids = np.arange(0, len(ids_f.datasets[1].col1))*0.1
j_ids = I_ids/(np.pi*0.2**2)

V_t = interp1d(t_ids, V_ids, bounds_error=False, fill_value=-1.05)
t = f[scan_no].Time-11
V_XRD = V_t(t)



FWHM_Ls = np.array(FWHM_Ls)
FWHM_Ks = np.array(FWHM_Ks)
Ls = np.array(Ls)
Ks = np.array(Ks)

d_Co = 2*np.pi/FWHM_Ls/0.89
dom_size = 2*np.pi/FWHM_Ks/2.52
strain_K = (1.151/Ks-1.)*100
strain_L = (1.737/Ls-1.)*100

data_cols = [V_XRD, FWHM_Ls, FWHM_Ks, Ls, Ks]
data_cols = np.array(data_cols, dtype=float).transpose()
header = 'Potential, FWHM_L, FWHM_K, L_pos, K_pos'
np.savetxt('#416_CycloDiffractogram.csv', data_cols, delimiter=',', fmt='%.6e', header=header)


plt.figure()
plt.ylabel('d_Co')
plt.plot(V_XRD[5:36], d_Co[5:36], 'ko-')
plt.plot(V_XRD[35:70], d_Co[35:70], 'ko-', markerfacecolor='white', fillstyle='full')
plt.plot(V_XRD[69:74], d_Co[69:74], 'ko-')
#plt.plot(V_XRD, d_Co)

plt.figure()
plt.ylabel('dom_size')
plt.plot(V_XRD[5:36], dom_size[5:36], 'ko-')
plt.plot(V_XRD[35:70], dom_size[35:70], 'ko-', markerfacecolor='white', fillstyle='full')
plt.plot(V_XRD[69:74], dom_size[69:74], 'ko-')

plt.figure()
plt.ylabel('Strain K')
plt.plot(V_XRD[5:36], strain_K[5:36], 'ko-')
plt.plot(V_XRD[35:70], strain_K[35:70], 'ko-', markerfacecolor='white', fillstyle='full')
plt.plot(V_XRD[69:74], strain_K[69:74], 'ko-')

plt.figure()
plt.ylabel('Strain L')
plt.plot(V_XRD[5:36], strain_L[5:36], 'ko-')
plt.plot(V_XRD[35:70], strain_L[35:70], 'ko-', markerfacecolor='white', fillstyle='full')
plt.plot(V_XRD[69:74], strain_L[69:74], 'ko-')

plt.figure()
plt.ylabel('Pos K')
plt.plot(V_XRD[5:36], Ks[5:36], 'ko-')
plt.plot(V_XRD[35:70], Ks[35:70], 'ko-', markerfacecolor='white', fillstyle='full')
plt.plot(V_XRD[69:74], Ks[69:74], 'ko-')


plt.show()


'''
plt.figure()
plt.imshow(L_cut_img)
plt.figure()
plt.imshow(K_cut_img)
plt.show()

# Position of image corners
coords = [[0,0], [0, 515], [515, 0], [515, 515]]
for coord in coords:
    print 'H = %.5f   K = %.5f   L = %.5f' % (_H[coord[0], coord[1]], _K[coord[0], coord[1]], _L[coord[0], coord[1]])
'''