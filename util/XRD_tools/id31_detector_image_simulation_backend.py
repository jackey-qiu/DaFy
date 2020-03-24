from mayavi import mlab
import numpy as np
import id31_tools as id31
import matplotlib.pyplot as plt
import constants as const

def mayavi_init():
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

    return engine


# Plot Detector
def plot_detector(engine, k_0, theta, d):
    x = np.arange(0,2049, 16)
    y = np.arange(0,2049, 16)
    X, Y = np.meshgrid(x, y)
    q = id31.get_q(k_0, theta, id31.get_del(X, d), id31.get_gam(Y, d))
    mlab.mesh(q[0], q[1], q[2], scalars=q[2], colormap='RdBu')
    module_manager = engine.scenes[0].children[0].children[0].children[0]
    module_manager.scalar_lut_manager.reverse_lut = True

    # Plot L=0 plane
    mlab.mesh(q[0], q[1], q[2]*0.0, scalars=q[2]*0.0, color=(0,0,0), opacity=0.4)
    
    qx_range = [np.min(q[0]), np.max(q[0])]
    qy_range = [np.min(q[1]), np.max(q[1])]
    return (qx_range, qy_range)



def selection_rules_hcp(H,K,L, give_structure_factor=False):
    if((L%2 == 0) and (H + 2*K)%3 != 0):
        return (0.5 if give_structure_factor else 1.0)
    if((L%2 != 0) and (H + 2*K)%3 != 0):
        return (1.5 if give_structure_factor else 1.0)
    if((L%2 == 0) and (H + 2*K)%3 == 0):
        return (2.0 if give_structure_factor else 1.0)
    else:
        return 0
    
def selection_rules_fcc(H,K,L, give_structure_factor=False):
    h = -4*H - 2*K + L
    k = 2*H - 2*K + L
    l = 2*H + 4*K + L
    
    if((h%6 == 0) and (k%6 == 0) and (l%6 == 0)):
        return (2.0 if give_structure_factor else 1.0)
    if((h%6 == 3) and (k%6 == 3) and (l%6 == 3)):
        return (2.0 if give_structure_factor else 1.0)
    else:
        return 0.
    
def selection_rules_diamond(H, K, L, give_structure_factor=False):
    h = -4.*H - 2.*K + 1.*L
    k = 2.*H - 2.*K + 1.*L
    l = 2.*H + 4.*K + 1.*L
    
    if((h%6 == 0) and (k%6 == 0) and (l%6 == 0) and ((h+k+l)%12 == 0)):
        return (2.0 if give_structure_factor else 1.0)
    if((h%6 == 1) and (k%6 == 1) and (l%6 == 1)):
        return (1.0 if give_structure_factor else 1.0)
    else:
        return 0.

def plot_rods(qx_range, qy_range, a_star, c_star, k_0, selection_rule_function, alpha=0, rod_length=2, color=(0,0,0)):
    HKL1 = id31.get_HKL(np.array([qx_range[0], qy_range[0], 0]), a_star, c_star, alpha)
    HKL2 = id31.get_HKL(np.array([qx_range[0], qy_range[1], 0]), a_star, c_star, alpha)
    HKL3 = id31.get_HKL(np.array([qx_range[1], qy_range[0], 0]), a_star, c_star, alpha)
    HKL4 = id31.get_HKL(np.array([qx_range[1], qy_range[1], 0]), a_star, c_star, alpha)
    
    Hlim = np.array([HKL1[0], HKL1[0]])
    Klim = np.array([HKL1[1], HKL1[1]])
    
    for HKLX in [HKL2, HKL3, HKL4]:
        if Hlim[0] > HKLX[0]:
            Hlim[0] = HKLX[0]
        if Hlim[1] < HKLX[0]:
            Hlim[1] = HKLX[0]
            
        if Klim[0] > HKLX[1]:
            Klim[0] = HKLX[1]
        if Klim[1] < HKLX[1]:
            Klim[1] = HKLX[1]
            
    Hlim *= 2
    Klim *= 2    
    
    H = np.arange(int(Hlim[0]),int(Hlim[1])+1,1)
    K = np.arange(int(Klim[0]),int(Klim[1])+1,1)
    L = np.arange(-rod_length,rod_length+1,1)
    

    qx = []
    qy = []
    qz = []
    I = []
    
    print H, K, L
    
    
    for HH in H:
        for KK in K:
            q_rod = id31.get_q_from_HKL(k_0, HH, KK, 1, a_star, c_star, alpha=alpha)
            
            # do not plot if rod is not on the detector
            if(q_rod[0] < qx_range[0] or q_rod[0] > qx_range[1] or q_rod[1] < qy_range[0] or q_rod[1] > qy_range[1]):
                continue
            
            # plot rods
            mlab.plot3d([q_rod[0], q_rod[0]], [q_rod[1], q_rod[1]], [-q_rod[2]*rod_length, q_rod[2]*rod_length], color=color)
    
            # plot Bragg peaks taking into account selection rules  
            for LL in L:
                Int = selection_rule_function(HH, KK, LL)
                if(Int != 0):       
                    q = id31.get_q_from_HKL(k_0, HH, KK, LL, a_star, c_star, alpha=alpha)
                    qx.append(q[0])
                    qy.append(q[1])
                    qz.append(q[2])
                    I.append(Int)
    
    qx = np.array(qx)
    qy = np.array(qy)
    qz = np.array(qz)
    I = np.array(I)                
    mlab.points3d(qx, qy, qz, I, scale_factor=0.2, color=color)


def plot_origin():
    mlab.points3d([0], [0], [0], scale_factor=0.2, color=(1,0,0))

def show():
    mlab.show()

