# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:49:47 2017

@author: Timo Fuchs
"""

import numpy as np
import numpy.linalg as LA
from . import util
#import util.util as util
import scipy.optimize as opt

try:
    from PyMca import SixCircle
except ImportError:
    from PyMca5.PyMca import SixCircle
    

# needs to be multiplied with K = 2*pi/lambda to get Qphi
def calculate_q_phi(pos,K=1.):
    [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] = createVliegMatrices(pos)

    u1a = (DELTA * GAMMA - ALPHA.I) * np.matrix([0.,K,0.]).T
    u1p = PHI.I * CHI.I * OMEGA.I * u1a
    return u1p


"""
pos = [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] (angles)
accepts also 1d-arrays of the angles in pos
""" 
def createVliegMatrices(pos):

    ALPHA = None if pos[0] is None else calcALPHA(pos[0])
    DELTA = None if pos[1] is None else calcDELTA(pos[1])
    GAMMA = None if pos[2] is None else calcGAMMA(pos[2])
    OMEGA = None if pos[3] is None else calcOMEGA(pos[3])
    CHI = None if pos[4] is None else calcCHI(pos[4])
    PHI = None if pos[5] is None else calcPHI(pos[5])
    return ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI

def calcALPHA(alpha): # alpha = mu ???
    if isinstance(alpha,np.ndarray):
        return util.x_rotationArray(alpha)
    return util.x_rotation(alpha)


def calcDELTA(delta):
    if isinstance(delta,np.ndarray):
        return util.z_rotationArray(-delta)
    return util.z_rotation(-delta)


def calcGAMMA(gamma):
    if isinstance(gamma,np.ndarray):
        return util.x_rotationArray(gamma)
    return util.x_rotation(gamma)


def calcOMEGA(omega): 
    if isinstance(omega,np.ndarray):
        return util.z_rotationArray(-omega)
    return util.z_rotation(-omega)


def calcCHI(chi):
    if isinstance(chi,np.ndarray):
        return util.y_rotationArray(chi)
    return util.y_rotation(chi)


def calcPHI(phi):
    if isinstance(phi,np.ndarray):
        return util.z_rotationArray(-phi)
    return util.z_rotation(-phi)

def calcSIGMA(sigma):
    if isinstance(sigma,np.ndarray):
        return util.y_rotationArray(sigma)
    return util.y_rotation(sigma)

def calcTAU(tau):
    if isinstance(tau,np.ndarray):
        return util.y_rotationArray(-tau)
    return util.y_rotation(-tau)


def primBeamAngles(pos):
    [alpha,delta,gamma,omega,chi,phi] = pos
    gamma_p = np.arcsin( np.cos(alpha)*np.sin(gamma) + np.sin(alpha)*np.cos(delta)*np.cos(gamma) )
    delta_p = np.arcsin( (np.sin(delta)*np.cos(gamma))/np.cos(gamma_p) )
    return [alpha,delta_p,gamma_p,omega,chi,phi]

def vliegDiffracAngles(pos_p):
    [alpha,delta_p,gamma_p,omega,chi,phi] = pos_p
    gamma = np.arcsin( np.cos(alpha)*np.sin(gamma_p) - np.sin(alpha)*np.cos(delta_p)*np.cos(gamma_p) )
    delta = np.arcsin( (np.sin(delta_p)*np.cos(gamma_p))/np.cos(gamma) )
    return [alpha,delta,gamma,omega,chi,phi]
    

"""
transforms alpha and gamma into the crystal frame using 
snellius refraction law
"""
def crystalAngles(pos,refraction_index):
    pos = np.array(pos)
    if len(pos.shape) == 1:
        pos[0] = np.arccos(np.cos(pos[0]) / refraction_index)
        pos[2] = np.arccos(np.cos(pos[2]) / refraction_index)
        pos[np.isnan(pos)] = 0. 
    else:
        pos[:,0] = np.arccos(np.cos(pos[:,0]) / refraction_index)
        pos[:,2] = np.arccos(np.cos(pos[:,2]) / refraction_index)
        pos[np.isnan(pos)] = 0. 
    return pos


def crystalAngles_singleArray(angle,refraction_index):
    if isinstance(angle,np.ndarray):
        sign = np.sign(angle)
        angle = np.arccos(np.cos(angle) / refraction_index)
        angle[np.isnan(angle)] = 0.
        angle *= sign
    else:
        sign = np.sign(angle)
        angle = np.arccos(np.cos(angle) / refraction_index)
        if np.isnan(angle):
            angle = 0.
        angle *= sign
    return angle

def printPos(pos,phichi=False):
    pos = np.rad2deg(pos)
    if phichi:
        print("alp=%.2f, del=%.2f, gam=%.2f, om=%.2f, phi=%.2f, chi=%.2f" % tuple(pos))
    else:
        print("alp=%.2f, del=%.2f, gam=%.2f, om=%.2f" % tuple(np.array((pos))[:-2]) )
        
def strPos(pos,phichi=False):
    pos = np.rad2deg(pos)
    if phichi:
        spos = "alp=%.2f, del=%.2f, gam=%.2f, om=%.2f, phi=%.2f, chi=%.2f" % tuple(pos)
    else:
        spos = "alp=%.2f, del=%.2f, gam=%.2f, om=%.2f" % tuple(np.array((pos))[:-2])
    return spos
    
def strPos_prim(pos,phichi=False):
    spos = "Vlieg angles:\n"
    spos += strPos(pos,phichi)
    spos += "\nangles ref prim beam:\n"
    spos += strPos(primBeamAngles(pos),phichi)
    return spos
    
def printPos_prim(pos,phichi=False):
    print("Vlieg angles:")
    printPos(pos,phichi)
    print("angles ref prim beam")
    printPos(primBeamAngles(pos),phichi)

def spec_pa(ub):
    print(str(ub.getCrystal()))
    
    
    
 
"""

"""

class Crystal(object):
    def __init__(self,a,alpha):
        self.setLattice(a,alpha)
    
    def setLattice(self,a,alpha):
        self._alpha = np.deg2rad(alpha)
        self._a = a
        self.calcReciprocalLattice()
        self._BMatrix = Crystal.calcB(self._a,self._alpha,self._b,self._beta)
        self.RealspaceMatrix = Crystal.realspaceMatrix(self._a,self._alpha,self._b,self._beta)
    
    #def setBasis(self,atoms):
        
    
    def getLatticeParameters(self):
        return self._a, self._alpha, self._b, self._beta
    
    def getB(self):
        return self._BMatrix
    
    def getHKL(self,b):
        return b/self._b
    
    #in rad, shape of hkl must be either (3,) or (3,n)
    def get2ThetaFromHKL(self,hkl,energy):
        wavelength = 12.39842 / energy
        hkl = np.array(hkl)
        if len(hkl.shape) == 1:
            G = LA.norm(self.getReciprocalVectorCart(hkl).T)
        else:
            G = LA.norm(self.getReciprocalVectorCart(hkl).T,axis=1)
        return 2*np.arcsin((G*wavelength)/(4*np.pi))
    
    # input: array like [h, k, l]
    def getReciprocalVectorCryst(self,hkl):
        return hkl*self._b
    
    # calculates atomic positions from fractional coordinates xyz_frac 
    # to cartesian coordinates in Angstroms  
    def directVectorCart(self,xyz_frac):
        return self.RealspaceMatrix*np.matrix(xyz_frac).T
    
        
    # calculates reciprocal vector in cartesian coordinates 
    # from lattice units hkl
    def getReciprocalVectorCart(self,hkl):
        return self.getB()*np.matrix(hkl).T
    
    def calcReciprocalLattice(self):
        self._beta = np.empty(3)
        
        self._beta[0] = np.arccos((np.cos(self._alpha[1]) * np.cos(self._alpha[2]) - np.cos(self._alpha[0])) /
                           (np.sin(self._alpha[1]) * np.sin(self._alpha[2])))

        self._beta[1] = np.arccos((np.cos(self._alpha[0]) * np.cos(self._alpha[2]) - np.cos(self._alpha[1])) /
                                   (np.sin(self._alpha[0]) * np.sin(self._alpha[2])))

        self._beta[2] = np.arccos((np.cos(self._alpha[0]) * np.cos(self._alpha[1]) - np.cos(self._alpha[2])) /
                                   (np.sin(self._alpha[0]) * np.sin(self._alpha[1])))
        
        self._b = np.empty(3)
        
        volume = (np.product(self._a) *
          np.sqrt(1 + 2 * np.cos(self._alpha[0]) * np.cos(self._alpha[1]) * np.cos(self._alpha[2]) -
               np.cos(self._alpha[0]) ** 2 - np.cos(self._alpha[1]) ** 2 - np.cos(self._alpha[2]) ** 2))

        self._b[0] = 2 * np.pi * self._a[1] * self._a[2] * np.sin(self._alpha[0]) / volume
        self._b[1] = 2 * np.pi * self._a[0] * self._a[2] * np.sin(self._alpha[1]) / volume
        self._b[2] = 2 * np.pi * self._a[0] * self._a[1] * np.sin(self._alpha[2]) / volume
    
    
    @staticmethod
    def calcB(a,alpha,b,beta):
        return np.matrix([(b[0],     b[1]*np.cos(beta[2]),   b[2]*np.cos(beta[1])),
                         (0.,        b[1]*np.sin(beta[2]),   -b[2]*np.sin(beta[1])*np.cos(alpha[0])),
                         (0.,        0.,                     2.*np.pi/a[2])])
    @staticmethod 
    def realspaceMatrix(a,alpha,b,beta):
        return np.matrix([(a[0],     a[1]*np.cos(alpha[2]),   a[2]*np.cos(alpha[1])),
                         (0.,        a[1]*np.sin(alpha[2]),   -a[2]*np.sin(alpha[1])*np.cos(beta[0])),
                         (0.,        0.,                     2.*np.pi/b[2])])

    
    
    def __str__(self):
        name = "Lattice:\nReal space: \t" + str(self._a) + " / " + str(np.rad2deg(self._alpha)) + "\n"
        name += "Reciprocal space: \t" + str(self._b) + " / " + str(np.rad2deg(self._beta))
        return name


    def __repr__(self):
        return str(self)

class UBCalculator():
    
    def __init__(self,crystal, energy):
        self.setCrystal(crystal)
        self.setEnergy(energy)
    
    def setCrystal(self,crystal):
        self._crystal = crystal
        self._U = None
        self._UB = None
        
    def setEnergy(self,energy):
        self._energy = energy
        self._lambda = 12.39842 / energy
        self._K = (2*np.pi)/self._lambda
        
    def setLambda(self,lmbda):
        self._energy = 12.39842 / lmbda
        self._lambda = lmbda
        self._K = (2*np.pi)/self._lambda
    
    def getCrystal(self):
        return self._crystal
    
    def getEnergy(self):
        return self._energy
    
    def getLambda(self):
        return self._lambda
    
    def getK(self):
        return self._K
    
    def setPrimaryReflection(self,pos,hkl):
        self._primary = (pos,hkl)
        
    def setSecondayReflection(self,pos,hkl):
        self._secondary = (pos,hkl)
        
    # l in z-direction, (0,1,0) in x-direction at omega = 0°
    def defaultU(self):
        TwoTheta1 = self._crystal.get2ThetaFromHKL([0,0,1],self._energy)
        TwoTheta2 = self._crystal.get2ThetaFromHKL([0,1,0],self._energy)
        self.setPrimaryReflection([TwoTheta1/2.,0.,TwoTheta1/2.,0,0.,0.],[0,0,1])
        self.setSecondayReflection([0.,TwoTheta2,0.,0.,0.,0.],[0,1,0])
        self.calculateU()
        
    
    # primary and secondary reflections must have been set
    # modified version from diffcalc
    def calculateU(self):
        ppos, phkl =  self._primary
        spos, shkl =  self._secondary
        
        # Compute the two reflections' reciprical lattice vectors in the
        # cartesian crystal frame (hc = B * hkl)
        h1c = self._crystal.getReciprocalVectorCart(phkl).flatten()
        h2c = self._crystal.getReciprocalVectorCart(shkl).flatten()
        
        # Calculate vector in the plane normal
        u1p = calculate_q_phi(ppos).flatten()
        u2p = calculate_q_phi(spos).flatten()
        
        # Create modified unit vectors t1, t2 and t3 in crystal and phi systems
        t1c = h1c
        t3c = np.cross(h1c, h2c)
        t2c = np.cross(t3c, t1c)
        
        t1p = u1p
        t3p = np.cross(u1p, u2p)
        t2p = np.cross(t3p, t1p)
        
        # ...and nornmalise and check that the reflections used are appropriate
        SMALL = 1e-5

        def normalise(m):
            d = LA.norm(m)
            if d < SMALL:
                raise Exception("Error: Reflections not useful")
            return m / d

        t1c = normalise(t1c)
        t2c = normalise(t2c)
        t3c = normalise(t3c)

        t1p = normalise(t1p)
        t2p = normalise(t2p)
        t3p = normalise(t3p)

        Tc = np.hstack([t1c.T, t2c.T, t3c.T])
        Tp = np.hstack([t1p.T, t2p.T, t3p.T])
        
        self._U = Tp*Tc.I
        self._UB = self._U * self._crystal.getB()
        return self._U

    def getU(self):
        if self._U is not None: return self._U 
        else: raise Exception("No U calculated")
        
    def getUB(self):
        if self._UB is not None: return self._UB
        else: raise Exception("No UB calculated")
        
    def getUmB(self):
        B = self._crystal.getB()
        #B[0] *= -1.
        #B[1] *= -1.
        return  self._U * B
    
    def setU(self,U):
        self._U = U.reshape((3,3))
        self._UB = self._U * self._crystal.getB()
    
    def refineU(self,hkl,angles,allowPhiChi_opt=False,rod=None,factor=100.):
        qphi = []
        for pos in angles:
            qphi.append(calculate_q_phi(pos,self.getK()).T.A)
        qphi = np.array(qphi).T
        # p[0], p[1], p[2]: rotation angles
        
        weights = np.ones(angles.shape[0])
        
        if rod is not None:
            weights[np.all(hkl[:,:2] == rod,axis=1)] = factor
        
        if allowPhiChi_opt:
            def Chi2(p):
                qphi = []
                UBnew = util.x_rotation(p[0])*util.y_rotation(p[1])*util.z_rotation(p[2])*self._U*self._crystal.getB()
                #if input("Type exit") == "exit": raise Exception("bla")
                for pos in angles:
                    pos[4] = p[3]
                    pos[5] = p[4]
                    qphi.append(calculate_q_phi(pos,self.getK()).T.A)
                qphi = np.array(qphi).T
                
                hklnew = (UBnew.I * qphi).T
                #print(np.sum(LA.norm(hkl - hklnew,axis=1)))
                return np.sum(LA.norm(hkl - hklnew,axis=1) * weights )
            res = opt.minimize(Chi2,[0,0,0,0,0])
        else:
            
            def Chi2(p):
                UBnew = util.x_rotation(p[0])*util.y_rotation(p[1])*util.z_rotation(p[2])*self._U*self._crystal.getB()
                #if input("Type exit") == "exit": raise Exception("bla")
                hklnew = (UBnew.I * qphi).T
                #print(np.sum(LA.norm(hkl - hklnew,axis=1)))
                return np.sum(LA.norm(hkl - hklnew,axis=1) * weights)
            res = opt.minimize(Chi2,[0,0,0])
        print(res)
        self._U = util.x_rotation(res.x[0])*util.y_rotation(res.x[1])*util.z_rotation(res.x[2])*self._U
        self._UB = self._U * self._crystal.getB()
        

    def bruteForceU(self,hkl,angles):
        qphi = []
        for pos in angles:
            qphi.append(calculate_q_phi(pos,self.getK()).T.A)
        qphi = np.array(qphi).T
        # p[0], p[1], p[2]: rotation angles
        #I = np.matrix([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        def Chi2(p):
            UBnew = util.x_rotation(p[0])*util.y_rotation(p[1])*util.z_rotation(p[2])*self._crystal.getB()
            hklnew = (UBnew.I * qphi).T
            #print(hklnew)
            #print(hkl)
            #print(np.sum(LA.norm(hkl - hklnew,axis=1)))
            return np.sum(LA.norm(hkl - hklnew,axis=1))
        res = opt.minimize(Chi2,[0,0,0])
        self._U = util.x_rotation(res.x[0])*util.y_rotation(res.x[1])*util.z_rotation(res.x[2])
        self._UB = self._U * self._crystal.getB()
        print(res)    
    
    def __str__(self):
        Estr = 'E = ' + str(self.getEnergy()) + ' keV, lambda = ' +str(self.getLambda()) + "\n"
        xtalstr = str(self.getCrystal())
        ppos, phkl =  self._primary
        spos, shkl =  self._secondary 
        primstr = '\nprimary reflection (or0): %s\n%s\n' % (phkl , strPos_prim(ppos) )
        secstr = 'secondary reflection (or1): %s\n%s\n' % (shkl , strPos_prim(spos) )
        return Estr + xtalstr + primstr + secstr
    
    def __repr__(self):
        return str(self)
        
    
    


class VliegAngles():
    def __init__(self,ubCalculator):
        self._ubCalculator = ubCalculator
        
    """
    Returns the hkl values of a aingle detector frame. 
    phi,chi,alpha,omega are fixed, gamma and delta are 1-d arrays
    The calculation is optimized with numpy
    Only kinematical calculation!
    """
    def anglesToHklDetector(self,alpha,delta,gamma,omega,chi,phi):
        [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] = createVliegMatrices([alpha,delta,gamma,omega,chi,phi])
        hkl = np.empty((gamma.size,delta.size,3))
        K = self._ubCalculator.getK()
        UBi = self._ubCalculator.getUB().I.A
        ALPHAi = ALPHA.I.A
        OMEGAi = OMEGA.I.A
        CHIi = CHI.I.A
        PHIi = PHI.I.A
        for i in range(gamma.size):
            #calculate ( DELTA * GAMMA - ALPHA**-1 ) * K_lab = Q_alpha
            DEL_GAM_minALP = np.matmul((np.matmul(DELTA,GAMMA[i]) - ALPHAi), np.array([0.,K,0.]) ).T
            #calculate UBi * PHIi * CHIi * OMEGAi * Q_alpha 
            hkl[i] = np.matmul(UBi,np.matmul(PHIi,np.matmul(CHIi,np.matmul(OMEGAi,DEL_GAM_minALP)))).T
        return hkl[:,:,0], hkl[:,:,1], hkl[:,:,2] # h k l
    
    def anglesToHklDetector_PyMca(self,alpha,delta,gamma,omega,chi,phi):
        wavelength = self._ubCalculator.getLambda()
        ub = self._ubCalculator.getUB().A1
        alpha = np.rad2deg(alpha)
        delta = np.rad2deg(delta)
        gamma = np.rad2deg(gamma)
        omega = np.rad2deg(omega)
        chi = np.rad2deg(chi)
        phi = np.rad2deg(phi)
        R = SixCircle.getHKL(wavelength, ub, gamma=gamma, delta=delta,
                             theta=omega, mu=alpha, chi=chi, phi=phi)

        shape = gamma.size, delta.size
        H = R[0, :].reshape(shape)
        K = R[1, :].reshape(shape)
        L = R[2, :].reshape(shape)
        return H, K, L
    
    
    # only for single points
    def anglesToHkl(self,pos):
        """
        Returns hkl from pos object in radians.
        Only kinematical calculation!
        """
        pos = np.array(pos)
        if len(pos.shape) == 1:
            return self._ubCalculator.getUB().I * calculate_q_phi(pos,self._ubCalculator.getK())
        else:
            qphi = []
            for p in pos:
                qphi.append(calculate_q_phi(p,self._ubCalculator.getK()).T.A)
            qphi = np.array(qphi).T
            return (self._ubCalculator.getUB().I * qphi).T
        
    def QAlphaDetector(self,alpha,delta,gamma):
        [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] = createVliegMatrices([alpha,delta,gamma,None,None,None])
        K = self._ubCalculator.getK()
        Qxyz = np.empty((gamma.size,delta.size,3))
        ALPHAi = ALPHA.I.A
        for i in range(gamma.size):
            #calculate ( DELTA * GAMMA - ALPHA**-1 ) * K_lab = Q_alpha
            Qxyz[i] = np.matmul((np.matmul(DELTA,GAMMA[i]) - ALPHAi), np.array([0.,K,0.]) )
        return Qxyz[:,:,0], Qxyz[:,:,1], Qxyz[:,:,2] # Qx Qy Qz
    
    def anglesZmode(self,hkl,fixedangle,fixed='in',chi=0,phi=0,**keyargs):
        [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] = createVliegMatrices([None,None,None,None,chi,phi])
        K = self._ubCalculator.getK()
        Hphi = np.matmul(self._ubCalculator.getUB().A,hkl)
        if 'mirrorx' in keyargs:
            if keyargs['mirrorx'] == True:
                #Hphi = np.matmul(self._ubCalculator.getUmB().A,hkl)
                Hphi[0] *= -1
        Homega = np.matmul((CHI*PHI).A,Hphi)
        if fixed == 'in':
            alpha = fixedangle
            gamma = np.arcsin(Homega[2]/K - np.sin(alpha) )
        elif fixed == 'out':
            gamma = fixedangle
            alpha = np.arcsin(Homega[2]/K - np.sin(gamma) )
        elif fixed == 'eq':
            gamma = alpha = np.arcsin(Homega[2]/(2*K))
        else:
            raise Exception("No valid angle constraint given. Should be one of 'in', 'out' or 'eq'")
        
        delta = np.arccos((1. - np.dot(Homega,Homega.T) / (2*K**2) + np.sin(gamma)*np.sin(alpha)) *
                          (np.cos(gamma)*np.cos(alpha))**-1)
        
        omega = np.arctan2((Homega[1]*np.sin(delta)*np.cos(gamma) - Homega[0]*(np.cos(delta)*np.cos(gamma) - np.cos(alpha))),
                          (Homega[0]*np.sin(delta)*np.cos(gamma) + Homega[1]*(np.cos(delta)*np.cos(gamma) - np.cos(alpha))))
        
        if 'mirrorx' in keyargs:
            if keyargs['mirrorx'] == True:
                delta *= -1.
                omega *= -1.
        
        return alpha, delta, gamma, omega, chi, phi
        
    def anglesZmode_np18(self,hkl,fixedangle,fixed='in',chi=0,phi=0,**keyargs):
        [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] = createVliegMatrices([None,None,None,None,chi,phi])
        K = self._ubCalculator.getK()
        Hphi = np.dot(self._ubCalculator.getUB().A,hkl.T)
        if 'mirrorx' in keyargs:
            if keyargs['mirrorx'] == True:
                #Hphi = np.matmul(self._ubCalculator.getUmB().A,hkl)
                Hphi[0] *= -1
        Homega = np.dot((CHI*PHI).A,Hphi.T).T
        if fixed == 'in':
            alpha = fixedangle
            gamma = np.arcsin(Homega[2]/K - np.sin(alpha) )
        elif fixed == 'out':
            gamma = fixedangle
            alpha = np.arcsin(Homega[2]/K - np.sin(gamma) )
        elif fixed == 'equal':
            gamma = alpha = np.arcsin(Homega[2]/(2*K))
        else:
            raise Exception("No valid angle constraint given.")
        
        delta = np.arccos((1. - np.dot(Homega,Homega.T) / (2*K**2) + np.sin(gamma)*np.sin(alpha)) *
                          (np.cos(gamma)*np.cos(alpha))**-1)
        
        omega = np.arctan2((Homega[1]*np.sin(delta)*np.cos(gamma) - Homega[0]*(np.cos(delta)*np.cos(gamma) - np.cos(alpha))),
                          (Homega[0]*np.sin(delta)*np.cos(gamma) + Homega[1]*(np.cos(delta)*np.cos(gamma) - np.cos(alpha))))
        
        if 'mirrorx' in keyargs:
            if keyargs['mirrorx'] == True:
                delta *= -1.
                omega *= -1.
        
        return alpha, delta, gamma, omega, chi, phi
        
    
    def anglesToHkl_dyn(self,pos,refraction_index):
        """
        Returns hkl from pos object in radians
        corrects for L-shift due to refraction at a surface
        only real part of refraction index is considered
        !!! assumes alpha = beta_in and gamma-alpha = beta_out !!!
        """
        pos = np.array(pos)
        if len(pos.shape) == 1:
            pos = crystalAngles(pos,refraction_index)
            return self._ubCalculator.getUB().I * calculate_q_phi(pos,self._ubCalculator.getK())
        else:
            qphi = []
            for p in pos:
                p = crystalAngles(p,refraction_index)
                qphi.append(calculate_q_phi(p,self._ubCalculator.getK()).T.A)
            qphi = np.array(qphi).T
            return (self._ubCalculator.getUB().I * qphi).T
    
    def hkIntersect(self,rod,pos):
        [alpha,delta,gamma,omega,chi,phi] = pos
        H_H,H_K = rod
        
        [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] = createVliegMatrices(pos)
        K = self._ubCalculator.getK()
        ub = self._ubCalculator.getUB()
        Vmat = np.matmul(np.matmul(np.matmul(OMEGA,CHI),PHI),ub)
        ALPHAi = ALPHA.I.A
        #print( np.matmul(ALPHAi,np.array([0.,K,0.])))
        C1, C2, C3 =  (np.matmul(ALPHAi,np.array([0.,K,0.])) + np.matmul(Vmat,np.array([H_H,H_K,0.]))).A1
        
        V1, V2, V3 = Vmat[:,2].A1
        
        #Vn = V1 + V2 + V3
        Vn2 = V1**2 + V2**2 + V3**2
        
        Cn2 = C1**2 + C2**2 + C3**2
        
        CmV = V1*C1 + V2*C2 + V3*C3
        sqrtIncl =  (CmV/Vn2)**2 - ((Cn2 - K**2) / Vn2)
        if sqrtIncl < 0:
            raise Exception("Can not find any intersection of the rod with\n"
                            "the current location of the Ewald sphere")
        sqrtTerm = np.sqrt(sqrtIncl) 
        
        # somewhere is a sign error!!!
        L1 = -1*(CmV/Vn2 + sqrtTerm)
        L2 = -1*(CmV/Vn2 - sqrtTerm)
        
        delta1 = np.arctan2(C1 + V1*L1,C2 + V2*L1)
        delta2 = np.arctan2(C1 + V1*L2,C2 + V2*L2)
        
        gam1 = np.arcsin((C3 + V3*L1)/K)
        gam2 = np.arcsin((C3 + V3*L2)/K)
        
        pos1 = [alpha,delta1,gam1,omega,chi,phi]
        pos2 = [alpha,delta2,gam2,omega,chi,phi]
        
        return ([H_H,H_K,L1],pos1),([H_H,H_K,L2],pos2)
    
    def getGeometryCorrection(self):
        return GeometryCorrection(self)

# only for phi/omega scans, partially zmode
class GeometryCorrection():
    def __init__(self,vliegangles):
        self._angles = vliegangles
    
    def lorentzFactor(self,delta,beta_in,gamma):
        return 1./(np.sin(delta)*np.cos(beta_in)*np.cos(gamma))
    
    def polarization(self,delta,gamma,alpha,fraction_horiz=1.):
        P_hor = 1. - (np.sin(alpha)*np.cos(delta)*np.cos(gamma) + np.cos(alpha)*np.sin(gamma))**2
        P_vert = 1. - (np.sin(delta)**2)*(np.cos(gamma)**2) 
        return fraction_horiz*P_hor + (1.-fraction_horiz)*P_vert
    
    # without footprint correction
    def activeSurfaceArea(self,delta,alpha,beta_in):
        return 1./(np.sin(delta)*np.cos(alpha-beta_in))
    
    def correctionZmode(self,hkl,fixedangle,fixed='in',polarization_horiz=1.):
        alpha, delta, gamma, omega, chi, phi = self._angles.anglesZmode(hkl,fixedangle,fixed)
        P = self.polarization(delta,gamma,alpha,polarization_horiz)
        #L_phi = self.lorentzFactor(delta,alpha,gamma)
        Carea = self.activeSurfaceArea(delta,alpha,alpha)
        return P*Carea
    
    def correctDatasetZmode(self,hkl,I,fixedangle,fixed='in',polarization_horiz=1.):
        corr = np.empty_like(I)
        for i in range(I.size):
            corr[i] = self.correctionZmode(hkl[i],fixedangle,fixed,polarization_horiz)
        corr /= np.mean(corr)
        return I/corr
    
    def correctionFactorZmode(self,alpha,delta,gamma,polarization_horiz=1.):
        delta = np.abs(delta)
        corr = np.empty_like(delta)
        for i in range(delta.shape[0]):
            P = self.polarization(delta[i],gamma[i],alpha,polarization_horiz)
            Carea = self.activeSurfaceArea(delta[i],alpha,alpha)
            corr[i] = (P*Carea)
        return corr
    
    def correctImageZmode(self,intensity,alpha,delta,gamma,polarization_horiz=1.):
        I = np.copy(intensity)
        delta = np.abs(delta)
        
        for i in range(delta.shape[0]):
            #print(i)
            P = self.polarization(delta[i],gamma[i],alpha,polarization_horiz)
            Carea = self.activeSurfaceArea(delta[i],alpha,alpha)
            #print(I[i].shape)
            #print((P*Carea).shape)
            I[i] = I[i]/(P*Carea)
        return I
    
    def applyImageZmode(self,intensity,alpha,delta,gamma,polarization_horiz=1.):
        I = np.copy(intensity)
        delta = np.abs(delta)
        
        for i in range(delta.shape[0]):
            #print(i)
            P = self.polarization(delta[i],gamma[i],alpha,polarization_horiz)
            Carea = self.activeSurfaceArea(delta[i],alpha,alpha)
            #print(I[i].shape)
            #print((P*Carea).shape)
            I[i] = I[i]*(P*Carea)
        return I
        
        

#pos = [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] (angles)

if __name__ == "__main__":

    pt111 = Crystal([ 2.7748 , 2.7748 , 6.7969],[  90. ,  90. , 120.])
    """
    basis = [[1.,0.,0.,0.],[1.,2./3.,1./3.,1./3.],[1.,1./3.,2./3.,2./3.]]
    h = 1.
    k = -1
    for l in range(10):
        print("[%s , %s, %s]: F = %s" % (h,k,l,np.absolute(pt111.F_hkl([h,k,l],basis))))
    """
    ub = UBCalculator(pt111,69.971)
    ub.defaultU()
    
    #ub.setPrimaryReflection(np.deg2rad([0.15,4.2168,1.354,0,0.,0.]),[1.,0.,1.])
    #ub.setSecondayReflection(np.deg2rad([0.15,4.211,2.839,+32.43 + 28.12,0.,0.]),[0.,1.,2.])
    #ub.calculateU()
    
    
    angles = VliegAngles(ub)
    
    
    
    #delta = np.linspace(-np.pi/8,np.pi/8,1100)
    #gamma = np.linspace(-np.pi/8,np.pi/8,1600)
    h , k , l = angles.anglesToHklDetector(0.15,delta,gamma,0.1,0,0)
    
    #pos = [ALPHA, DELTA, GAMMA, OMEGA, CHI, PHI] (angles)
    #print (angles.anglesToHkl(np.deg2rad([0.1,4.211,2.839,32.43 + 28.12,0.,0.])))
    
    
    
    
    try:
        from PyMca import SixCircle
    except ImportError:
        from PyMca5.PyMca import SixCircle
        
    #sixc = SixCircle.SixCircle()
    #sixc.setEnergy(69.971)
    #sixc.setUB(ub.getUB())
    #print(sixc.getHKL(0.,0.,32.43 + 28.12,0.6,4.211,2.839))
    
    #print(np.rad2deg(l.get2ThetaFromHKL([1,0,1],69.971)))
    
    #angles 


