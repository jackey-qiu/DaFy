import numpy as np
from numpy.linalg import inv
import CifFile
import re
import ast
from periodictable import elements
from periodictable.xsf import xray_energy, xray_sld_from_atoms, xray_sld
import id03_tools_old as id03
from scipy.optimize import minimize, fmin

def RotationMatrix(theta_x, theta_y, theta_z):
    Rx = np.array([[1,0,0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0,1,0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0,0,1]])
    return Rx.dot(Ry).dot(Rz)

class lattice():
    def __init__(self, a, b, c, alpha=90, beta=90, gamma=90, basis=[0,0,0], HKL_normal=[0,0,1], HKL_para_x=[1,0,0], energy_keV=22.5):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = np.deg2rad(alpha)
        self.beta = np.deg2rad(beta)
        self.gamma = np.deg2rad(gamma)
        self.energy_keV = energy_keV
        self.k0 = id03.get_K_0(energy_keV)
        self.basis = basis # list of [Atomic_form_factor, x, y, z]
        for i in xrange(len(self.basis)):
            if(isinstance(self.basis[i][0], basestring)):
                f1, f2 = elements.symbol(self.basis[i][0]).xray.scattering_factors(energy=energy_keV)
                self.basis[i][0] = f1 + 1.j*f2
        self.HKL_normal = HKL_normal
        self.HKL_para_x = HKL_para_x
        
        # calculate real space unit cell vectors
        self.A1 = np.array([self.a, 0, 0])
        self.A2 = np.array([self.b*np.cos(self.gamma), b*np.sin(self.gamma), 0])
        A31 = self.c*np.cos(self.alpha)
        A32 = self.c/np.sin(self.gamma)*(np.cos(self.beta)-np.cos(self.gamma)*np.cos(self.alpha))
        A33 = np.sqrt(self.c**2-A31**2-A32**2)
        self.A3 = np.array([A31, A32, A33])
        
        # calculate reciprocal space unit cell vectors
        self._V_real = self.A1.dot(np.cross(self.A2,self.A3))
        self.B1 = 2*np.pi*np.cross(self.A2, self.A3)/self._V_real
        self.B2 = 2*np.pi*np.cross(self.A3, self.A1)/self._V_real
        self.B3 = 2*np.pi*np.cross(self.A1, self.A2)/self._V_real
        self.RecTM = np.array([[self.B1[0], self.B2[0], self.B3[0]], [self.B1[1], self.B2[1], self.B3[1]], [self.B1[2], self.B2[2], self.B3[2]]])
        
        self._V_rec = self.B1.dot(np.cross(self.B2,self.B3))
        
        # align surface normal to z axis
        q_normal = self.q(HKL_normal)
        q_normal /= np.sqrt(q_normal.dot(q_normal))
        z = np.array([0,0,1])
        v = np.cross(q_normal, z)
        I = np.zeros((3,3));I[0,0]=1;I[1,1]=1;I[2,2]=1
        R = np.zeros((3,3))
        if(v[0] == 0 and v[1] == 0 and v[2] == 0):
            R = I # unit matrix
        elif(q_normal[0] == 0 and q_normal[1] == 0 and q_normal[2] == -1):
            R = np.array([[1,0,0],[0,-1, 0], [0, 0, -1]]) # rotation by 180deg around x
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = I + vx + vx.dot(vx)/(1.+q_normal.dot(z))
            
        self.RecTM = R.dot(self.RecTM)
        
        # align projection of HKL_para_x to x axis
        q = self.q(HKL_para_x)
        rot = -np.arctan2(q[1], q[0])
        R = np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0,0,1]])
        self.RecTM = R.dot(self.RecTM)
        
        self.RecTMInv = inv(self.RecTM)
    @staticmethod 
    def lattice_from_cif(filename, HKL_normal=[0,0,1], HKL_para_x=[1,0,0], energy=22.5):
        cf = CifFile.ReadCif(filename)
        data = cf.first_block()
        a = float(data['_cell_length_a'])
        b = float(data['_cell_length_b'])
        c = float(data['_cell_length_c'])
        alpha = float(data['_cell_angle_alpha'])
        beta = float(data['_cell_angle_beta'])
        gamma = float(data['_cell_angle_gamma'])
        
        _basis = []
        atom_sites = data.GetLoop('_atom_site_label')
        for atom_site in atom_sites:
            f1, f2 = elements.symbol(atom_site[0]).xray.scattering_factors(energy=energy)
            #_atom_site_x = atom_site['_atom_site_fract_x']
            #print _atom_site_x
            _basis.append([f1+1.j*f2, np.array([float(atom_site[1]), float(atom_site[2]), float(atom_site[3]), 1])])
        
        basis = []        
        sym_pos_arr = data.GetLoop('_symmetry_equiv_pos_as_xyz')
        for sym_pos in sym_pos_arr:
            toks = sym_pos[0].split(',')
            pos = ['x', 'y', 'z']
            translation_matrix = np.zeros((4,4))
            translation_matrix[3,3] = 1 # [row, column]
            
            for i in xrange(len(toks)):
                for j in xrange(len(pos)):
                    if(('-%s'%(pos[j])) in toks[i]):
                        translation_matrix[i, j] = -1
                        toks[i] = toks[i].replace('-%s'%(pos[j]), '')
                        break
                    elif(('+%s'%(pos[j])) in toks[i]):
                        translation_matrix[i, j] = 1
                        toks[i] = toks[i].replace('+%s'%(pos[j]), '')
                        break
                    elif(('%s'%(pos[j])) in toks[i]):
                        translation_matrix[i, j] = 1
                        toks[i] = toks[i].replace('%s'%(pos[j]), '')
                        break
                translation_matrix[i,3] = float(eval(toks[i]+'.0'))
                  
            print translation_matrix
                    
            for atom in _basis:
                x, y, z, w = translation_matrix.dot(atom[1])
                while(x < 0):
                    x += 1
                while(x >= 1):
                    x -=1
                while(y < 0):
                    y += 1
                while(y >= 1):
                    y -=1
                while(z < 0):
                    z += 1
                while(z >= 1):
                    z -=1
                add_atom = True
                for i in xrange(len(basis)):
                    if(basis[i] == [atom[0], x, y, z]):  
                        add_atom = False
                if(add_atom):
                    basis.append([atom[0], x, y, z])
                    
        print 'basis:', len(basis)
        print basis
            
        return lattice(a, b, c, alpha, beta, gamma, basis, HKL_normal, HKL_para_x)
        

    def q(self, HKL):
        return self.RecTM.dot(HKL)
    def HKL(self, q):
        return self.RecTMInv.dot(q)
    def F(self, HKL):
        F = 0
        for i in xrange(len(self.basis)):
            F += self.basis[i][0]*np.exp(-2.j*np.pi*(HKL[0]*self.basis[i][1]+HKL[1]*self.basis[i][2]+HKL[2]*self.basis[i][3]))
        return F#/self._V_real
    def I(self, HKL):
        F = self.F(HKL)
        #return 1
        return (F*F.conjugate()).real
    
    def qI(self, HKL):
        q = self.q(HKL)
        return [q[0], q[1], q[2], self.I(HKL)]
    
    def V_real(self):
        return self._V_real
    def V_rec(self):
        return self._V_rec
    
    ''' Angles th, gam, del in sixcvertical session.'''
    def angles(self, HKL, mu):
        q = self.q(HKL)
        k02 = self.k0**2
        def f(theta):
            v = q + RotationMatrix(0, np.deg2rad(mu), np.deg2rad(theta)).dot([self.k0,0,0])
            return (v.dot(v)-k02)**2
                
        import matplotlib.pyplot as plt
        plt.figure()
        x = np.arange(360)
        plt.plot(x, f(x))
        plt.show()
        
        thetas = []
        for theta in xrange(0,360, 10):
            res = fmin(f, (theta))#, bounds=((-360, 360),), method='L-BFGS-B')
            '''if(res.success):
                theta = res.x[0]
                print res.message
            else:
                raise 'Determining Theta failed.'''
            while(theta < 0):
                theta += 360
            while(theta > 360):
                theta -= 360
            theta = np.round(theta, 4)
            if(not theta in thetas):
                thetas.append(theta)
            if(len(thetas) == 2):
                break
            
            print 'Theta =', theta
        
        if(len(thetas) != 2):
            raise 'Found incorrect number of theta values.'
        
        gammas = []
        deltas = []
        
        for theta in thetas:
            # calculate k_f
            k_f = q + RotationMatrix(0, np.deg2rad(mu), np.deg2rad(theta)).dot([self.k0,0,0])
            # gamma
            gammas.append(np.rad2deg(np.arctan2(k_f[2], np.sqrt(k_f[0]**2+k_f[1]**2))))
            deltas.append(np.rad2deg(np.sign(k_f[1])*np.arccos(k_f[0]/np.sqrt(k_f[0]**2+k_f[1]**2))))
            
        return (np.array([thetas[0], gammas[0], deltas[0]]), np.array([thetas[1], gammas[1], deltas[1]]))        

        
    
if __name__ == '__main__':
    lat = lattice.lattice_from_cif('/home/finn/Documents/eclipse_workspace/2017_MA3074/preparation/structure_simulation/Co3O4.cif')
    #print lat.q([1,0,0])
    
    lat.angles([7,1,0], mu=0.34)
    
    from sympy.solvers import solve
    from sympy import symbols
    from sympy import sin, cos
    x, a, b, c, d, e, A, B, C, D = symbols('x a b c d e A B C D')
    #solve((a+cos(d)*cos(x))**2+(b+sin(x))**2+(c-sin(d)*cos(x))**2-1, x)
    #A = a**2 + c**2 + d**2 -1
    #B = 2*a*b - 2*d*e
    #C = b**2 + e**2
    #D = 2*c
    #solve(A+B*cos(x)+C*cos(x)**2+sin(x)**2+D*sin(x), x)
    #solve(a + b*cos(x) + c*cos(x)**2 + d*sin(x), x)
    
    
    
    