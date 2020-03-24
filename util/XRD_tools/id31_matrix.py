import numpy as np
from numpy.linalg import inv

import reciprocal_space_v3 as rsp

def R_x(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
def R_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
def R_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

class id31_geometry:
    def __init__(self, cen_x, cen_y, sdd, lattice, energy_keV=40, pix_size_x=0.1720, pix_size_y=0.1720):
        self.k0 = self.calculated_k0(energy_keV)
        self.cen_x = cen_x
        self.cen_y = cen_y
        self.sdd = sdd
        self.pix_size_x = pix_size_x
        self.pix_size_y = pix_size_y
        self.alpha = 0
        self.theta = 0
        self.chi = 0
        self.set_angles(self.alpha, self.theta, self.chi)
        self.lattice = lattice
        
    def calculated_k0(self, energy_keV):
        return 2.*np.pi/12.39854*energy_keV
        
    def set_angles(self, alpha=0, theta=0, chi=0):
        self.alpha = np.deg2rad(alpha)
        self.theta = np.deg2rad(theta)
        self.chi = np.deg2rad(chi)
        self.RM = R_z(np.pi/2.).dot(R_y(np.pi/2.)).dot(R_x(-self.alpha)).dot(R_z(-self.theta)).dot(R_y(-self.chi))
        self.RM_inv = inv(self.RM)
        
    def set_energy_keV(self, energy_keV):
        self.k0 = self.calculated_k0(energy_keV)
    
    def delta(self, x):
        return np.arctan(-(x-self.cen_x)*self.pix_size_x/self.sdd)
    
    def gamma(self, y):
        return np.arctan((y-self.cen_y)*self.pix_size_y/self.sdd)

    def x(self, delta):
        return self.cen_x - self.sdd/self.pix_size_x*np.tan(delta)

    def y(self, gamma):    
        return self.cen_y + self.sdd/self.pix_size_y*np.tan(gamma)

    def q_lab(self, gamma, delta):
        return self.k0*(R_z(delta).dot(R_y(gamma)).dot(np.array([1,0,0]))-np.array([1,0,0]))   
    
    def q(self, x, y):
        gamma = self.gamma(y)
        delta = self.delta(x)
        return self.RM.dot(self.q_lab(gamma, delta))
    
    def xy(self, q):
        q_lab = self.RM_inv.dot(q)
        the_gam = np.arcsin(-q_lab[2]/self.k0)
        the_del = np.arcsin(q_lab[1]/self.k0/np.cos(the_gam))
        return (self.x(the_del), self.y(the_gam))

    def xy_from_HKL(self, HKL):
        return self.xy(self.lattice.q(HKL))


if __name__ == '__main__':
    # test functions
    Au100_lat = rsp.lattice(a=4.0782, b=4.0782, c=4.0782, alpha=90, beta=90,gamma=90, basis=[['Au', 0,0,0],['Au', 0.5,0.5,0],['Au', 0,0.5,0.5],['Au', 0.5,0,0.5]], HKL_normal = [0,0,1],HKL_para_x = [1,0,0])
    geometry = id31_geometry(cen_x=735, cen_y=829, sdd=650, lattice=Au100_lat, energy_keV=70, pix_size_x=0.1720, pix_size_y=0.1720)
    
    print geometry.xy_from_HKL([0,2,2])
    



'''import numpy as np

def RotationMatrix(theta_x, theta_y, theta_z):
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)],
                  [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    return Rx.dot(Ry).dot(Rz)


from sympy import *


def R_x(theta):
    return Matrix([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
def R_y(theta):
    return Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
def R_z(theta):
    return Matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

k0, theta, gamma, delta, alpha, chi, q_lab, qx, qy, qz= symbols('k0 theta gamma delta alpha chi q_lab qx qy qz')

#init_printing()

ki = Matrix([k0, 0, 0])
kf = R_z(delta)*R_y(gamma)*ki


q_lab = kf-ki
zero = q_lab-Matrix([qx, qy, 0])
zero[2] = 0
pprint(zero)
q_lab = simplify(q_lab)
pprint(simplify(solve(zero, gamma, delta)))
quit()

chi=0
alpha=0
#theta=0
q_sample = R_z(theta)*R_y(chi)*R_x(alpha)*q_lab

q_sample_rot = R_z(-pi/2)*R_y(-pi/2)*q_sample
q_sample_rot = simplify(q_sample_rot)

pprint(q_lab)
pprint(q_sample)
pprint(q_sample_rot)




ki = Matrix([0,0,-k0]) 
R_theta = Matrix([[cos(theta), 0, sin(theta)], [0,1,0], [-sin(theta), 0, cos(theta)]])
R_delta = Matrix([[cos(delta), 0, sin(delta)], [0,1,0], [-sin(delta), 0, cos(delta)]])
R_gamma = Matrix([[1,0,0], [0, cos(gamma), -sin(gamma)], [0, sin(gamma), cos(gamma)]])
R_alpha = Matrix([[cos(alpha),-sin(alpha),0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])
R_chi = Matrix([[1,0,0], [0, cos(chi), -sin(chi)], [0, sin(chi), cos(chi)]])
kf = R_delta*R_gamma*ki
pprint(kf)
pprint(ki)
pprint(kf)
q_lab = kf-ki
q_crystal = R_theta*R_alpha*R_chi*q_lab

pprint(simplify(q_lab))
pprint(simplify(R_theta*q_lab))
pprint(simplify(q_crystal))'''