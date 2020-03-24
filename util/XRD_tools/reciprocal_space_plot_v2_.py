import reciprocal_space_v2 as rsp
from timeit import itertools
import numpy as np
from PyMca5.PyMcaPhysics import SixCircle 
import id03_tools_old as id03

'''Bi_lat = rsp.lattice(a   = 4.75, 
              b          = 4.75, 
              c          = 4.75,
              alpha      = 57.35,     
              beta       = 57.35,     
              gamma      = 57.35,     
              basis      = [[1, 0.234, 0.234, 0.234], [1, 0.766, 0.766, 0.766]], 
              HKL_normal = [0,1,2],
              HKL_para_x = [1,0,0]
              )'''


print id03.get_K_0(25)

Co3O4_lat = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Co3O4.cif', HKL_normal = [1,1,1], HKL_para_x=[1,1,-2])
Co3O4_lat2 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Co3O4.cif', HKL_normal = [1,1,1], HKL_para_x=[1,1,-2], offset_angle=90)
Co3O4_lat3 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Co3O4.cif', HKL_normal = [1,1,1], HKL_para_x=[1,1,-2], offset_angle=180)
Co3O4_lat4 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Co3O4.cif', HKL_normal = [1,1,1], HKL_para_x=[1,1,-2], offset_angle=270)
Co3O4_lat5 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Co3O4.cif', HKL_normal = [1,0,0], HKL_para_x=[0,0,1])

PbBrF_lat = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/PbBrF.cif', HKL_normal = [0,0,1], HKL_para_x=[0,1,0], E_keV=25)
Bi_lat = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Bi.cif', HKL_normal = [0,0,1], HKL_para_x=[0,1,0], E_keV=25)
Bi_lat2 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Bi.cif', HKL_normal = [0,0,1], HKL_para_x=[0,1,0], offset_angle=90, E_keV=25)
Bi_lat3 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Bi.cif', HKL_normal = [0,0,1], HKL_para_x=[0,1,0], offset_angle=180, E_keV=25)
Bi_lat4 = rsp.lattice.from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Bi.cif', HKL_normal = [0,1,1], HKL_para_x=[0,1,0], offset_angle=270, E_keV=25)


from mayavi import mlab


try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene(size=(600, 800))
scene = engine.scenes[0]
fig = mlab.gcf(engine)
mlab.figure(figure=fig, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), engine=engine)


Au_lat = rsp.lattice(
                 a          = 4.0782, 
                 b          = 4.0782, 
                 c          = 4.0782,
                 alpha      = 90,     
                  beta       = 90,     
                  gamma      = 90,     
                  basis      = [['Au', 0,0,0],['Au', 0.5,0.5,0],['Au', 0,0.5,0.5],['Au', 0.5,0,0.5]], 
                  HKL_normal = [0,0,1],
                  HKL_para_x = [0,1,0]
                  )

Au_lat_hexa = rsp.lattice(
                      a          = 4.0782/np.sqrt(2), 
                      b          = 4.0782/np.sqrt(2), 
                      c          = 4.0782*np.sqrt(3),
                      alpha      = 90,     
                      beta       = 90,     
                      gamma      = 120,     
                      basis      = [[1, 0,0,0],[1, 1./3.,2./3.,1./3.],[1, 2./3.,1./3.,2./3.]], 
                      HKL_normal = [0,0,1],
                      HKL_para_x = [0,1,0]
                      )

Co_lat = rsp.lattice(a          = 2.5071, 
              b          = 2.5071, 
              c          = 4.0695,
              alpha      = 90,     
              beta       = 90,     
              gamma      = 120,     
              basis      = [[1, 0,0,0], [1, 1./3., 2./3., 0.5]], 
              HKL_normal = [0,0,1],
              HKL_para_x = [1,0,0]
              )

'''Bi_lat = rsp.lattice(a   = 4.75, 
              b          = 4.75, 
              c          = 4.75,
              alpha      = 57.35,     
              beta       = 57.35,     
              gamma      = 57.35,     
              basis      = [[1, 0.234, 0.234, 0.234], [1, 0.766, 0.766, 0.766]], 
              HKL_normal = [1,1,0],
              HKL_para_x = [1,0,0]
              )
'''
class space_plot():
    def __init__(self, lattice):
        self.lattice = lattice
    def plot_peaks(self, qz_lims=[-1e-5,1], q_para_lim=6, qx_lims=None, qy_lims=None, HKL_lims=[-10, 10], color=(0,0,0)):
        HKLs = list(itertools.product(np.arange(HKL_lims[0],HKL_lims[1]), repeat=3))
        qs = []
        Is = []
        for HKL in HKLs:
            q = self.lattice.q(HKL)
            if(q_para_lim != None):
                if(q[2] >= qz_lims[0] and q[2] < qz_lims[1] and np.sqrt(q[0]**2+q[1]**2)<= q_para_lim):
                    qs.append(q)
                    Is.append(self.lattice.I(HKL))
            elif(qx_lims != None and qy_lims != None):
                if(q[0] >= qx_lims[0] and q[0] < qx_lims[1] and q[1] >= qy_lims[0] and q[1] < qy_lims[1] and q[2] >= qz_lims[0] and q[2] < qz_lims[1]):
                    qs.append(q)
                    Is.append(self.lattice.I(HKL))

        qs = np.array(qs)
        qs = np.swapaxes(qs, 0, 1)
        Is = np.array(Is)
        Is /= np.max(Is)
        mlab.points3d(qs[0], qs[1], qs[2], Is, scale_factor=1, color=color)

    def get_rod_positions(self, qz_lims=[-1e-5,1], q_para_lim=6, qx_lims=None, qy_lims=None, HKL_lims=[-10, 10], color=(0,0,0)): 
        HKLs = list(itertools.product(np.arange(HKL_lims[0],HKL_lims[1]), repeat=3))
        qIs = []
        for HKL in HKLs:
            qI = self.lattice.qI(HKL)
            if(q_para_lim != None):
                if(qI[2] >= qz_lims[0] and qI[2] < qz_lims[1] and np.sqrt(qI[0]**2+qI[1]**2)<= q_para_lim):
                    qIs.append(qI)
            elif(qx_lims != None and qy_lims != None):
                if(qI[0] >= qx_lims[0] and qI[0] < qx_lims[1] and qI[1] >= qy_lims[0] and qI[1] < qy_lims[1] and qI[2] >= qz_lims[0] and qI[2] < qz_lims[1]):
                    qIs.append(qI)
        
        qIs = np.array(qIs)        
        rows = np.where(qIs[:,3] > 1e-20)
        qIs = qIs[rows]
        qIs_filtered = []
        for qI in qIs:
            add_qI = True
            for qI_filtered in qIs_filtered:
                if((abs(qI[0]-qI_filtered[0]) < 0.001) and (abs(qI[1]-qI_filtered[1]) < 0.001)):
                    add_qI = False
                    break
            if(add_qI):
                qIs_filtered.append(qI)
        return qIs_filtered
    def plot_rods(self, qz_lims=[-1e-5,1], q_para_lim=6, qx_lims=None, qy_lims=None, HKL_lims=[-10, 10], color=(0,0,0)): 
        qIs = self.get_rod_positions(qz_lims=qz_lims, q_para_lim=q_para_lim, qx_lims=qx_lims, qy_lims=qy_lims, HKL_lims=HKL_lims, color=color)
        for qI in qIs:
            mlab.plot3d([qI[0], qI[0]], [qI[1], qI[1]], qz_lims, color=color)
            
    def plot_grid(self, grid_qz=0, qz_lims=[-1e-5,1], q_para_lim=6, qx_lims=None, qy_lims=None, HKL_lims=[-10, 10], color=(0,0,0)):
        qIs = self.get_rod_positions(qz_lims=qz_lims, q_para_lim=q_para_lim, qx_lims=qx_lims, qy_lims=qy_lims, HKL_lims=HKL_lims, color=color)
        pos_pairs = []
        for i in xrange(len(qIs)):
            for j in xrange(i+1, len(qIs)):
                pos_pairs.append([qIs[i][0], qIs[i][1], qIs[j][0], qIs[j][1], np.sqrt((qIs[i][0]-qIs[j][0])**2 + (qIs[i][1]-qIs[j][1])**2)])
                
        pos_pairs = np.array(pos_pairs)
        distance_min = np.min(np.swapaxes(pos_pairs, 0, 1)[4])
        rows = np.where(pos_pairs[:, 4] < 1.1*distance_min)
        pos_pairs = pos_pairs[rows]
        for pos_pair in pos_pairs:
            mlab.plot3d([pos_pair[0], pos_pair[2]], [pos_pair[1], pos_pair[3]], [grid_qz,grid_qz], color=color)
    def plot_unit_cell(self, color=(0,0,0)):
        HKLs = [[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]]
        lines = [[0,1], [0,2], [0,4], [5,1], [5,4], [5,7], [3,1], [3,2], [3,7], [6,4], [6,2], [6,7]]
        for line in lines:
            q1 = self.lattice.q(HKLs[line[0]])
            q2 = self.lattice.q(HKLs[line[1]])

            mlab.plot3d([q1[0], q2[0]], [q1[1], q2[1]], [q1[2], q2[2]], color=color)

class detector_plot():
    def __init__(self, energy_keV, central_pixel, detector_sample_distance, detector_size_pix=(516, 516), pixel_size=(55e-6, 55e-6)):
        self.energy_keV = energy_keV
        self.k0 = id03.get_K_0(energy_keV)
        self.central_pixel = central_pixel
        self.detector_sample_distance = detector_sample_distance
        self.detector_size_pix = detector_size_pix
        self.pixel_size = pixel_size
       
    def plot_ki_kf(self, theta, delta, gamma, mu):
        return 0
        
    def plot_detector(self, HKL, lattice):
        return 0


class detector_plot2():
    def __init__(self, mu, UB, energy_keV, central_pixel, detector_sample_distance, detector_size_pix=(516, 516), pixel_size=(55e-6, 55e-6)):
        self.mu = mu
        self.UB = UB
        self.energy_keV = energy_keV
        self.central_pixel = central_pixel
        self.detector_sample_distance = detector_sample_distance
        self.detector_size_pix = detector_size_pix
        self.pixel_size = pixel_size
        
    def plot_detector(self, HKL, lattice):
        return 0
    
            
#mlab.points3d([0], [0], [0], color=(1,0,0))
mlab.plot3d([0,10], [0,0], [0,0], color=(1,0,0))
mlab.plot3d([0,0], [0,10], [0,0], color=(0,1,0))
mlab.plot3d([0,0], [0,0], [0,10], color=(0,0,1))
mlab.plot3d([0,-10], [0,0], [0,0])
mlab.plot3d([0,0], [0,-10], [0,0])
mlab.plot3d([0,0], [0,0], [0,-10])

def R(theta_x, theta_y, theta_z):
    Rx = np.array([[1,0,0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0,1,0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0,0,1]])
    return Rx.dot(Ry).dot(Rz)
    
energy = 22.5 #keV
k0 = id03.get_K_0(energy)
mu = 10
theta = 45
gamma = 10
delta = 10

ki = R(0, np.deg2rad(mu), np.deg2rad(theta)).dot([k0,0,0])

kf = R(0, np.deg2rad(-gamma), np.deg2rad(theta+delta)).dot([k0,0,0])

q = kf-ki
         
#mlab.plot3d([-ki[0], 0], [-ki[1], 0], [-ki[2], 0], color=(0,0,1))
#mlab.plot3d([0, kf[0]], [0, kf[1]], [0, kf[2]], color=(0,0,1))
#mlab.plot3d([-ki[0], -ki[0]+kf[0]], [-ki[1], -ki[1]+kf[1]], [-ki[2], -ki[2]+kf[2]], color=(0,0,1))

mu = 0.34
#angles = Au_lat_hexa.angles([0,1,4], mu=mu)
#theta, gamma, delta = angles[0]
#print angles[0]
#print angles[1]

'''
ki = R(0, np.deg2rad(mu), np.deg2rad(theta)).dot([k0,0,0])
kf = R(0, np.deg2rad(-gamma), np.deg2rad(delta)).dot([k0,0,0])
mlab.plot3d([-ki[0], 0], [-ki[1], 0], [-ki[2], 0], color=(0,0,1))
mlab.plot3d([-ki[0], -ki[0]+kf[0]], [-ki[1], -ki[1]+kf[1]], [-ki[2], -ki[2]+kf[2]], color=(0,0,1))
'''

'''print PbBrF_lat.F([0,0,1])
print PbBrF_lat.F([0,0,2])
print PbBrF_lat.F([1,0,1])
print PbBrF_lat.F([1,1,0])
print PbBrF_lat.F([1,0,2])
print PbBrF_lat.F([1,1,1])
print PbBrF_lat.F([0,0,3])
print PbBrF_lat.F([1,1,2])
print PbBrF_lat.F([1,0,3])
'''

#quit()

if(0):
    PbBrF_space = space_plot(PbBrF_lat)
    PbBrF_space.plot_peaks(q_para_lim=4, qz_lims=[-1e-5,5], HKL_lims=[-10,10], color=(0,0.5,0))
    #PbBrF_space.plot_rods(q_para_lim=4, qz_lims=[-1e-5,5], color=(0,0.5,0))
    #PbBrF_space.plot_grid(q_para_lim=4 , grid_qz=0, qz_lims=[-1e-5,5], color=(0,0.5,0))
    #PbBrF_space.plot_grid(grid_qz=PbBrF_space.lattice.q([0,0,1])[2], qz_lims=[-1e-5,5], color=(0,0.5,0))
    #PbBrF_space.plot_unit_cell()

if(0):
    Co_space = space_plot(Co_lat)
    Co_space.plot_peaks(qz_lims=[-1e-5,5], color=(0,0.5,0))
    Co_space.plot_rods(qz_lims=[-1e-5,5], color=(0,0.5,0))
    Co_space.plot_grid(grid_qz=0, qz_lims=[-1e-5,5], color=(0,0.5,0))
    #Co_space.plot_grid(grid_qz=Co_space.lattice.q([0,0,1])[2], qz_lims=[-1e-5,5], color=(0,0.5,0))
    Co_space.plot_unit_cell()

if(1):      
    Bi_space = space_plot(Bi_lat)
    Bi_space.plot_peaks(HKL_lims=[-10, 10], q_para_lim=6, qz_lims=[-1e-5,2], color=(1,0,0))
    #Bi_space.plot_rods(qz_lims=[-1e-5,5], color=(1,0,0))
    #Bi_space.plot_grid(qz_lims=[-1e-5,5], color=(1,0,0))
    Bi_space.plot_unit_cell()
    
    Bi_space2 = space_plot(Bi_lat2)
    Bi_space2.plot_peaks(HKL_lims=[-10, 10], q_para_lim=6, qz_lims=[-1e-5,3], color=(1,0,0))
    Bi_space3 = space_plot(Bi_lat3)
    Bi_space3.plot_peaks(HKL_lims=[-10, 10], q_para_lim=6, qz_lims=[-1e-5,3], color=(1,0,0))    
    Bi_space4 = space_plot(Bi_lat4)
    Bi_space4.plot_peaks(HKL_lims=[-10, 10], q_para_lim=6, qz_lims=[-1e-5,3], color=(1,0,0))    

if(0):      
    Au_hexa_space = space_plot(Au_lat_hexa)
    Au_hexa_space.plot_peaks(qz_lims=[-1e-5,3], color=(0.8,0.8,0))
    Au_hexa_space.plot_rods(qz_lims=[-1e-5,3], color=(0.8,0.8,0))
    Au_hexa_space.plot_grid(qz_lims=[-1e-5,3], color=(0.8,0.8,0))
    Au_hexa_space.plot_unit_cell()
 
if(1):      
    Au_space = space_plot(Au_lat)
    Au_space.plot_peaks(qz_lims=[-1e-5,5], color=(0.8,0.8,0))
    Au_space.plot_rods(qz_lims=[-1e-5,5], color=(0.8,0.8,0))
    Au_space.plot_grid(qz_lims=[-1e-5,5], color=(0.8,0.8,0))
    Au_space.plot_unit_cell()

if(0):      
    Co3O4_space = space_plot(Co3O4_lat)
    Co3O4_space.plot_peaks(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    #Co3O4_space.plot_rods(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    #Co3O4_space.plot_grid(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    Co3O4_space.plot_unit_cell()
    
    hkls = [[1,1,3], [1,3,1], [3,1,1],
            [-1,1,3], [-1,3,1], [-3,1,1],
            [1,-1,3], [1,-3,1], [3,-1,1],
            [1,1,-3], [1,3,-1], [3,1,-1],
            [-1,-1,3], [-1,-3,1], [-3,-1,1],
            [-1,1,-3], [-1,3,-1], [-3,1,-1],
            [1,-1,-3], [1,-3,-1], [3,-1,-1],
            [-1,-1,-3], [-1,-3,-1], [-3,-1,-1]
            ]
    for hkl in hkls: 
        q = Co3O4_lat.q(hkl)
        mlab.plot3d([0,q[0]], [0, q[1]], [0, q[2]], color=(0,0,0))
    ''''q = Co3O4_lat.q([-1,1,3])
    mlab.plot3d([0,q[0]], [0, q[1]], [0, q[2]], color=(0,0,0))
    q = Co3O4_lat.q([3,-1,-1])
    mlab.plot3d([0,q[0]], [0, q[1]], [0, q[2]], color=(0,0,0))'''
  
if(0):
    #print Co3O4_lat5.I([1,1,1])
    print Au_lat.HKL(Co3O4_lat.q([0,-4,4]))

    #print Co3O4_lat5.I([4,4,0])
    #print Co3O4_lat5.I([1,1,3])
    #print Co3O4_lat5.I([4,0,0])
  
if(0):      
    Co3O4_space5 = space_plot(Co3O4_lat5)
    Co3O4_space5.plot_peaks(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    Co3O4_space5.plot_rods(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    Co3O4_space5.plot_grid(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    Co3O4_space5.plot_unit_cell()  
    

if(0):
    Co3O4_space2 = space_plot(Co3O4_lat2)
    Co3O4_space2.plot_peaks(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
    Co3O4_space3 = space_plot(Co3O4_lat3)
    Co3O4_space3.plot_peaks(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))  
    Co3O4_space4 = space_plot(Co3O4_lat4)
    Co3O4_space4.plot_peaks(HKL_lims=[-10, 10], qz_lims=[-1e-5,5], color=(1,0,0))
      
'''sphere = mlab.points3d(-ki[0], -ki[1], -ki[2], scale_mode='none', scale_factor=2*k0, color=(0.67, 0.77, 0.93), resolution=50, opacity=0.7)
sphere.actor.property.specular = 0.45
sphere.actor.property.specular_power = 5
sphere.actor.property.backface_culling = True'''
   
      
mlab.show()
