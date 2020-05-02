# -*- coding: ascii -*-
"""
Created on Mon Jan 30 12:15:58 2017

@author: reikowski, wiegmann
"""

import numpy as np
from numpy.linalg import inv
import CifFile
from periodictable import elements
from scipy.optimize import fmin
from timeit import itertools


# Calculate k0 in inverse Angstrom from E in keV
def get_k0(E_keV):
    return 2. * np.pi / 12.39854 * E_keV


# Rotation matrix for a rotation of the Cartesian coordinate system (!)
# by theta_x around the x-axis,
# by theta_y around the y-axis,
# by theta_z around the z-axis
def RotationMatrix(theta_x, theta_y, theta_z):
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)],
                  [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    return Rx.dot(Ry).dot(Rz)

def Rx(theta_x):
    return np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x), np.cos(theta_x)]])
def Ry(theta_y):
    return np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
def Rz(theta_z):
    return np.array([[np.cos(theta_z), -np.sin(theta_z), 0],[np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

# Strips all brackets from a string
def strip_brackets(s):
    s1 = str.replace(str(s), '(', '')
    s2 = str.replace(s1, ')', '')
    return s2


# Models a crystal lattice
# a, b, c: length of real-space unit cell vectors in Angstrom
# alpha, beta, gamma: real-space unit cell angles in degrees
#     alpha is the angle between b and c, and so on.
# basis: list of atoms in unit cell
#     Basis elements in format [form factor, fract_x, fract_y, fract_z]
#     or in format ['element', fract_x, fract_y, fract_z]
# HKL_normal: The qz-axis will be parallel to this vector
# HKL_para_x: The qx-axis will be parallel to this vector
# offset_angle: Turn the qx-axis by an additional offset angle
#               (after the alignment with HKL_para_x has happened)
# E_keV: Used for determining the atomic form factor.
#     Energies above 30 keV are treated as 30 keV
class lattice():
    def __init__(self, a, b, c, alpha=90, beta=90, gamma=90, basis=[0, 0, 0],
                 HKL_normal=[0, 0, 1], HKL_para_x=[1, 0, 0], offset_angle=0,
                 E_keV=22.5):
        self.a = a
        self.b = b
        self.c = c

        self.alpha = np.deg2rad(alpha)
        self.beta = np.deg2rad(beta)
        self.gamma = np.deg2rad(gamma)

        self.E_keV = E_keV
        self.k0 = get_k0(E_keV)

        self.basis = basis  # list of [Atomic_form_factor, x, y, z]
        max_E = 30
        for i in range(len(self.basis)):
            if(isinstance(self.basis[i][0], str)):
                f1, f2 = (elements.symbol(self.basis[i][0]).
                          xray.scattering_factors(energy=min(E_keV, max_E)))
                #TODO: test
                #self.basis[i][0] = f1 + 1.j*f2

        self.HKL_normal = HKL_normal
        self.HKL_para_x = HKL_para_x

        # Calculate real space unit cell vectors
        # We choose a Cartesian coordinate system, in which the a-vector
        # is parallel to the x-axis ...
        self.A1 = np.array([self.a, 0, 0])
        # ... and the b-vector lies in the xy-plane.
        self.A2 = np.array([self.b * np.cos(self.gamma),
                            b * np.sin(self.gamma), 0])
        # The c-vector in these Cartesian coordinates is then:
        A31 = self.c * np.cos(self.beta)
        A32 = (self.c / np.sin(self.gamma) *
               (np.cos(self.alpha) - (np.cos(self.beta) * np.cos(self.gamma))))
        A33 = np.sqrt(self.c**2 - (A31**2 + A32**2))
        self.A3 = np.array([A31, A32, A33])
        # Contact Finn / Tim for a more detailed explanation

        # RealTM * (a, b, c) = (x, y, z)
        self.RealTM = np.array([[self.A1[0], self.A2[0], self.A3[0]],
                                [self.A1[1], self.A2[1], self.A3[1]],
                                [self.A1[2], self.A2[2], self.A3[2]]])

        # Calculate reciprocal space unit cell vectors
        self._V_real = self.A1.dot(np.cross(self.A2, self.A3))
        self.B1 = 2 * np.pi * np.cross(self.A2, self.A3) / self._V_real
        self.B2 = 2 * np.pi * np.cross(self.A3, self.A1) / self._V_real
        self.B3 = 2 * np.pi * np.cross(self.A1, self.A2) / self._V_real
        # RecTM * (a*, b*, c*) = (qx, qy, qz)
        self.RecTM = np.array([[self.B1[0], self.B2[0], self.B3[0]],
                               [self.B1[1], self.B2[1], self.B3[1]],
                               [self.B1[2], self.B2[2], self.B3[2]]])

        self._V_rec = self.B1.dot(np.cross(self.B2, self.B3))

        # Align surface normal to qz axis
        # In reciprocal space, the surface normal will point "up" (along qz)
        q_normal = self.q(HKL_normal)
        q_normal /= np.linalg.norm(q_normal)

        z = np.array([0, 0, 1])
        v = np.cross(q_normal, z)  # Our rotation axis

        # I = Identity matrix
        I = np.zeros((3, 3))
        I[0, 0] = 1
        I[1, 1] = 1
        I[2, 2] = 1

        R = np.zeros((3, 3))

        if(v[0] == 0 and v[1] == 0 and v[2] == 0):
            R = I
            # If v = (0,0,0), q_normal is already parallel to q_z

        elif(q_normal[0] == 0 and q_normal[1] == 0 and q_normal[2] == -1):
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            # If q_normal = (0,0,-1), q_normal is already antiparallel to q_z,
            # so we just need to rotate by 180 deg around the x-axis

        # If none of the previous two edge cases apply, we can use the
        # solution as per http://math.stackexchange.com/questions/180418
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = I + vx + vx.dot(vx) / (1. + q_normal.dot(z))

        self.RecTM = R.dot(self.RecTM)

        # Align projection of HKL_para_x onto x-axis
        # After this, turn by an additional offset_angle
        q = self.q(HKL_para_x)
        rot = -np.arctan2(q[1], q[0]) + np.deg2rad(offset_angle)
        R = np.array([[np.cos(rot), -np.sin(rot), 0],
                      [np.sin(rot), np.cos(rot), 0], [0, 0, 1]])
        self.RecTM = R.dot(self.RecTM)

        self.RecTMInv = inv(self.RecTM)

        self.unique_elements = np.unique(np.array(self.basis).T[0])
        self.basis_xyz = []
        self.basis_el = []
        for atom in self.basis:
            self.basis_el.append(atom[0])
            self.basis_xyz.append([atom[1], atom[2], atom[3]])
        self.basis_xyz = np.array(self.basis_xyz)
        self.basis_el = np.array(self.basis_el)


    # Import a lattice from a CIF file stored at filename
    # For other parameters, see description above
    @staticmethod
    def from_cif(filename, HKL_normal=[0, 0, 1], HKL_para_x=[1, 0, 0],
                 offset_angle=0, E_keV=22.5, override_a=None, override_b=None, override_c=None, override_alpha=None, override_beta=None, override_gamma=None ):
        cf = CifFile.ReadCif(filename)
        data = cf.first_block()

        # Load unit cell parameters
        a = override_a if override_a else float(strip_brackets(data['_cell_length_a']))
        b = override_b if override_b else float(strip_brackets(data['_cell_length_b']))
        c = override_c if override_c else float(strip_brackets(data['_cell_length_c']))
        alpha = override_alpha if override_alpha else float(strip_brackets(data['_cell_angle_alpha']))
        beta = override_beta if override_beta else float(strip_brackets(data['_cell_angle_beta']))
        gamma = override_gamma if override_gamma else float(strip_brackets(data['_cell_angle_gamma']))

        #a = float(strip_brackets(data['_cell_length_a']))
        #b = float(strip_brackets(data['_cell_length_b']))
        #c = float(strip_brackets(data['_cell_length_c']))
        #alpha = float(strip_brackets(data['_cell_angle_alpha']))
        #beta = float(strip_brackets(data['_cell_angle_beta']))
        #gamma = float(strip_brackets(data['_cell_angle_gamma']))

        # Create the basis.
        # In the CIF file, the coordinates of the basis atoms are given
        # in fractions of the unit cell vectors a, b, c.

        _basis = []
        atom_sites = data.GetLoop('_atom_site_label')
        elnames = atom_sites['_atom_site_label']
        fractxs = atom_sites['_atom_site_fract_x']
        fractys = atom_sites['_atom_site_fract_y']
        fractzs = atom_sites['_atom_site_fract_z']

        for i in range(len(elnames)):
            # In some files, the atoms have names like 'Co1', 'O1', 'O2',
            # and we need to strip the numbers to get the element name.
            elname = str.strip(str(elnames[i]), '123456789')
            el = elements.symbol(elname)
            max_E = 30
            f1, f2 = el.xray.scattering_factors(energy=min(max_E, E_keV))
            x = float(strip_brackets(fractxs[i]))
            y = float(strip_brackets(fractys[i]))
            z = float(strip_brackets(fractzs[i]))
            #TODO: test
            #_basis.append([f1 + 1.j * f2, np.array([x, y, z, 1])])
            _basis.append([elname, np.array([x, y, z, 1])])
        # The basis given by the atom_site_label entries is not the full
        # content of the unit cell.
        # We can create the full unit cell by applying the symmetry operations
        # defined by the crystal's space group to the atoms.
        # Luckily, we do not need to do this by ourselves, because all the
        # symmetry-equivalent positions are already given in the CIF file
        # under symmetry_equiv_pos_as_xyz.
        # "Copying" all atoms in atom_site_label to all symmetry-equivalent
        # positions will give the full unit cell.

        basis = []
        sym_pos_arr = data.GetLoop('_symmetry_equiv_pos_as_xyz')
        for sym_pos in sym_pos_arr:
            # Some CIF files use the format
            # x,y,z
            # Others use the format:
            # 1 'x, y, z'
            # This approach will handle both cases:
            symp = sym_pos[len(sym_pos) - 1]

            # We will create a translation matrix that we can apply to an
            # atom to move it to this symmetry-equivalent position
            toks = symp.split(',')
            pos = ['x', 'y', 'z']
            translation_matrix = np.zeros((4, 4))
            translation_matrix[3, 3] = 1  # [row, column]

            # Loop through the three coordinates
            # of the symmetry-equivalent position
            for i in range(len(toks)):
                # Moves by x, y, or z are expressed by an suitable entry
                # in the translation matrix. The summand is then deleted from
                # the string.
                for j in range(len(pos)):
                    if(('-%s' % (pos[j])) in toks[i]):
                        translation_matrix[i, j] = -1
                        toks[i] = toks[i].replace('-%s' % (pos[j]), '')
                        # break
                    elif(('+%s' % (pos[j])) in toks[i]):
                        translation_matrix[i, j] = 1
                        toks[i] = toks[i].replace('+%s' % (pos[j]), '')
                        # break
                    elif(('%s' % (pos[j])) in toks[i]):
                        translation_matrix[i, j] = 1
                        toks[i] = toks[i].replace('%s' % (pos[j]), '')
                        # break
                # Now, the string should only consist of a fractional
                # coordinate move along the respective axis,
                # which we also enter into the translation matrix
                translation_matrix[i, 3] = float(eval(toks[i]+'.0'))

            if(False):
                print(translation_matrix)

            # Copy all atoms using the translation matrix.
            # Atoms landing outside the unit cell will be moved inside,
            # and duplicates will be removed.
            for atom in _basis:
                x, y, z, w = translation_matrix.dot(atom[1])
                while(x < 0):
                    x += 1
                while(x >= 1):
                    x -= 1
                while(y < 0):
                    y += 1
                while(y >= 1):
                    y -= 1
                while(z < 0):
                    z += 1
                while(z >= 1):
                    z -= 1
                add_atom = True
                for i in range(len(basis)):
                    if((basis[i][0] == atom[0]) and
                       (np.round(basis[i][1], 10) == np.round(x, 10)) and
                       (np.round(basis[i][2], 10) == np.round(y, 10)) and
                       (np.round(basis[i][3], 10) == np.round(z, 10))):
                        add_atom = False
                        break
                if(add_atom):
                    basis.append([atom[0], x, y, z])

        if(False):
            print('basis:', len(basis))
            print(basis)

        return lattice(a, b, c, alpha, beta, gamma, basis, HKL_normal, HKL_para_x, offset_angle, E_keV=E_keV)

    # Input: (N, 3) array of HKL position
    # Returns:

    def set_E_keV(self, E_keV):
        self.E_keV = E_keV
        self.k0 = get_k0(E_keV)

    def F_many(self, HKLs, qs=None):
        if(qs.any() == None):
            qs = self.q_many(HKLs)
        Q = np.sqrt(np.sum(qs*qs, axis=1))
        f_dict = dict()
        for el in self.unique_elements:
            f_dict[el] = elements.symbol(el).xray.f0(Q)
        f = []
        for el in self.basis_el:
            f.append(f_dict[el])
        f = np.array(f)

        H, x = np.meshgrid(HKLs.T[0], self.basis_xyz.T[0])
        K, y = np.meshgrid(HKLs.T[1], self.basis_xyz.T[1])
        L, z = np.meshgrid(HKLs.T[2], self.basis_xyz.T[2])

        return np.sum(f*np.exp(-2.j*np.pi*(H*x + K*y + L*z)), axis=0)

    # Input: Reciprocal space vector in HKL coordinates (h, k, l)
    # Returns: Reciprocal space vector in Cartesian coordinates (qx, qy, qz)
    def q(self, HKL):
        return self.RecTM.dot(HKL)

    def q_many(self, HKLs):
        return np.dot(self.RecTM, HKLs.transpose()).transpose()

    # Input: Reciprocal space vector in HKL coordinates (h, k, l)
    # Returns: |q|
    def Q(self, HKL):
        q = self.RecTM.dot(HKL)
        return np.sqrt(q.dot(q))

    # Input: Reciprocal space vector in Cartesian coordinates (qx, qy, qz)
    # Returns: Reciprocal space vector in HKL coordinates (h, k, l)
    def HKL(self, q):
        return self.RecTMInv.dot(q)

    # Input: Reciprocal space vector in HKL coordinates (h, k, l)
    # Returns: Structure factor
    def F(self, HKL, rod_int=False, rod_alpha=0.1):
        F = 0
        q = self.q(HKL)
        Q = np.sqrt(q.dot(q))
        for i in range(len(self.basis)):
            f = elements.symbol(self.basis[i][0]).xray.f0(Q)
            F += (f * np.exp(-2.j*np.pi * (HKL[0] * self.basis[i][1] +
                                           HKL[1] * self.basis[i][2] +
                                           HKL[2] * self.basis[i][3])))
        return F

    # Input: Reciprocal space vector in HKL coordinates (h, k, l)
    # Returns: Scattering intensity
    def I(self, HKL, rod_int=False, rod_alpha=0.1):
        F = self.F(HKL, rod_int, rod_alpha)
        return (F*F.conjugate()).real

    # Input: Reciprocal space vector in HKL coordinates (h, k, l)
    # Returns: [qx, qy, qz, scattering intensity]
    def qI(self, HKL):
        q = self.q(HKL)
        return [q[0], q[1], q[2], self.I(HKL)]

    # Returns: Volume of real space unit cell in cubic Angstrom
    def V_real(self):
        return self._V_real

    # Returns: Volume of reciprocal space unit cell in cubic inverse Angstrom
    def V_rec(self):
        return self._V_rec

    # Angles th, gam, del in sixcvertical session.
    def angles(self, HKL, mu):
        q = self.q(HKL)
        k02 = self.k0**2

        def f(theta):
            #v = q + Ry(np.deg2rad(mu)).dot(Rz(np.deg2rad(theta)).dot([self.k0, 0, 0]))
            #v = q + np.dot(Rz(np.deg2rad(theta)), np.dot(Ry(np.deg2rad(mu)), [self.k0, 0, 0]))
            v = q + np.dot(Ry(np.deg2rad(mu)), np.dot(Rz(np.deg2rad(theta)), [self.k0, 0, 0]))
            #v = q + RotationMatrix(0, np.deg2rad(mu), np.deg2rad(theta)).dot([self.k0, 0, 0])
            return (v.dot(v)-k02)**2

        #import matplotlib.pyplot as plt
        #plt.figure()
        #x = np.arange(360)
        #plt.plot(x, f(x))
        #plt.show()

        thetas = []
        for theta in range(0, 360, 10):
            res = fmin(f, (theta))
            #print res
            #if(res.success):
            #    theta = res.x[0]
            #    print res.message
            #else:
            #    raise 'Determining Theta failed.'''
            theta = res[0]

            while(theta < 0):
                theta += 360
            while(theta > 360):
                theta -= 360
            theta = np.round(theta, 3)
            if(theta not in thetas):
                thetas.append(theta)
            if(len(thetas) == 2):
                break

            print('Theta =', theta)

        if(len(thetas) != 2):
            raise 'Found incorrect number of theta values.'

        gammas = []
        deltas = []

        for theta in thetas:
            # calculate k_f
            k_f = q + RotationMatrix(0, np.deg2rad(mu),
                                     np.deg2rad(theta)).dot([self.k0, 0, 0])
            # gamma
            gammas.append(np.rad2deg(np.arctan2(k_f[2], np.sqrt(k_f[0]**2 +
                                                                k_f[1]**2))))
            deltas.append(np.rad2deg(np.sign(k_f[1]) * np.arccos(k_f[0] /
                                     np.sqrt(k_f[0]**2 + k_f[1]**2))))

        return (np.array([thetas[0], gammas[0], deltas[0]]),
                np.array([thetas[1], gammas[1], deltas[1]]))
    def powder(self, q_max, HKL_lims=[-10, 10]):
        HKL_range = np.arange(HKL_lims[0], HKL_lims[1] + 1)
        HKLs = list(itertools.product(HKL_range, repeat=3))
        qIs = []
        Is = dict()
        # Find all lattice positions within limits
        for HKL in HKLs:
            q = self.q(HKL)
            Q = np.sqrt(q.dot(q))
            if(Q <= q_max):
                if Q in Is:
                    Is[Q] += self.I(HKL)
                else:
                    Is[Q] = self.I(HKL)
        Q = []
        I = []
        for key in Is:
            Q.append(key)
            I.append(Is[key])

        indices = np.argsort(Q)
        Q_sorted = []
        I_sorted = []
        for index in indices:
            Q_sorted.append(Q[index])
            I_sorted.append(I[index])
        Q_sorted = np.array(Q_sorted)
        I_sorted = np.array(I_sorted)/I_sorted[0]

        return (Q_sorted, I_sorted)



