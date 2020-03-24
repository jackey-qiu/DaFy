# -*- coding: ascii -*-
"""
Created on Mon Jan 30 12:15:58 2017

@author: reikowski, wiegmann
"""

import numpy as np
from timeit import itertools
from mayavi import mlab
from tvtk.tools import visual

def Arrow_From_A_to_B(x1, y1, z1, x2, y2, z2, color=(1,1,1)):
    ar1=visual.arrow(x=x1, y=y1, z=z1)
    arrow_length=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    ar1.length_cone=0.4/arrow_length
    ar1.radius_shaft = 0.03/arrow_length
    ar1.radius_cone = 0.1/arrow_length
    
    ar1.actor.scale=[arrow_length, arrow_length, arrow_length]
    ar1.pos = ar1.pos/arrow_length
    ar1.axis = [x2-x1, y2-y1, z2-z1]
    ar1.color = color
    return ar1


class space_plot():
    def __init__(self, lattice):
        self.lattice = lattice

    def is_in_limits(self, q, qx_lims, qy_lims, q_inplane_lim, qz_lims, mag_q_lims=None):
        qx = q[0]
        qy = q[1]
        qz = q[2]
        q_inpl = np.sqrt(qx**2 + qy**2)
        mag_q = np.sqrt(qx**2 + qy**2 + qz**2)
        
        return not ((qx_lims is not None and (qx < qx_lims[0] or qx > qx_lims[1])) or
                    (qy_lims is not None and (qy < qy_lims[0] or qy > qy_lims[1])) or
                    (qz_lims is not None and (qz < qz_lims[0] or qz > qz_lims[1])) or
                    (q_inplane_lim is not None and (q_inpl > q_inplane_lim)) or
                    (mag_q_lims is not None and (mag_q < mag_q_lims[0] or mag_q > mag_q_lims[1]))
                    )

    # Plot Bragg peaks, with point size proportional to scattering intensity
    def plot_peaks(self, HKL_lims=[-10, 10], qx_lims=None, qy_lims=None,
                   q_inplane_lim=None, qz_lims=None, mag_q_lims=None, color=(0, 0, 0), scale_factor=1):
        HKL_range = np.arange(HKL_lims[0], HKL_lims[1] + 1)
        HKLs = list(itertools.product(HKL_range, repeat=3))
        qs = []
        Is = []
        for HKL in HKLs:
            q = self.lattice.q(HKL)
            if(self.is_in_limits(q, qx_lims, qy_lims, q_inplane_lim, qz_lims, mag_q_lims)):
                qs.append(q)
                Is.append(self.lattice.I(HKL))

        qs = np.array(qs)
        qs = np.swapaxes(qs, 0, 1)
        Is = np.array(Is)
        if(len(Is > 0)):
            # Normalize to (0,0,0) intensity and plot the sqrt 
            Is /= self.lattice.I([0,0,0])
            Is = np.sqrt(Is)
            mlab.points3d(qs[0], qs[1], qs[2], Is, scale_factor=scale_factor, color=color)

    # For each CTR (within given limits), returns one (arbitrary)
    # Bragg peak that is on the CTR
    def get_rod_positions(self, HKL_lims=[-10, 10], qx_lims=None, qy_lims=None,
                          q_inplane_lim=None, qz_lims=None, color=(0, 0, 0)):
        HKL_range = np.arange(HKL_lims[0], HKL_lims[1] + 1)
        HKLs = list(itertools.product(HKL_range, repeat=3))
        qIs = []
        # Find all lattice positions within limits
        for HKL in HKLs:
            q = self.lattice.q(HKL)
            if(self.is_in_limits(q, qx_lims, qy_lims, q_inplane_lim, qz_lims)):
                qIs.append(self.lattice.qI(HKL))

        # Select all positions with non-zero intensity
        qIs = np.array(qIs)
        rows = np.where(qIs[:, 3] > 1e-20)
        qIs = qIs[rows]

        # We only need one point per (qx, qy) coordinate
        qIs_filtered = []
        for qI in qIs:
            add_qI = True
            for qI_filtered in qIs_filtered:
                if((abs(qI[0]-qI_filtered[0]) < 0.001) and
                   (abs(qI[1]-qI_filtered[1]) < 0.001)):
                    add_qI = False
                    break
            if(add_qI):
                qIs_filtered.append(qI)
        return qIs_filtered

    # Plot CTRs
    def plot_rods(self, HKL_lims=[-10, 10], qx_lims=None, qy_lims=None,
                  q_inplane_lim=None, qz_lims=[-10, 10], color=(0, 0, 0)):
        qIs = self.get_rod_positions(HKL_lims=HKL_lims, qx_lims=qx_lims,
                                     qy_lims=qy_lims,
                                     q_inplane_lim=q_inplane_lim,
                                     qz_lims=qz_lims, color=color)
        for qI in qIs:
            mlab.plot3d([qI[0], qI[0]], [qI[1], qI[1]], qz_lims, color=color)

    # Plot a grid at a specified qz value
    def plot_grid(self, HKL_lims=[-10, 10], grid_qz=0, qx_lims=None,
                  qy_lims=None, q_inplane_lim=None, qz_lims=None,
                  color=(0, 0, 0)):
        # Get lateral positions of all CTRs
        qIs = self.get_rod_positions(HKL_lims=HKL_lims, qx_lims=qx_lims,
                                     qy_lims=qy_lims,
                                     q_inplane_lim=q_inplane_lim,
                                     qz_lims=qz_lims, color=color)

        # Calculate the distances between every point and every other point
        pos_pairs = []
        for i in xrange(len(qIs)):
            for j in xrange(i+1, len(qIs)):
                distance = np.sqrt((qIs[i][0] - qIs[j][0])**2 +
                                   (qIs[i][1] - qIs[j][1])**2)
                pos_pairs.append([qIs[i][0], qIs[i][1], qIs[j][0], qIs[j][1],
                                  distance])

        # Find the minimum in-plane distance between to CTR positions
        pos_pairs = np.array(pos_pairs)
        distance_min = np.min(np.swapaxes(pos_pairs, 0, 1)[4])

        # Draw lines between CTR positions that have the minimum distance
        rows = np.where(pos_pairs[:, 4] < 1.1*distance_min)
        pos_pairs = pos_pairs[rows]
        for pos_pair in pos_pairs:
            mlab.plot3d([pos_pair[0], pos_pair[2]], [pos_pair[1], pos_pair[3]],
                        [grid_qz, grid_qz], color=color)

    # Plot the unit cell
    def plot_unit_cell(self, color=(0, 0, 0)):
        # Unit cell edge lines to be drawn are hard-coded
        HKLs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],
                [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        lines = [[0, 1], [0, 2], [0, 4], [5, 1], [5, 4], [5, 7], [3, 1],
                 [3, 2], [3, 7], [6, 4], [6, 2], [6, 7]]
        for line in lines:
            q1 = self.lattice.q(HKLs[line[0]])
            q2 = self.lattice.q(HKLs[line[1]])
            mlab.plot3d([q1[0], q2[0]], [q1[1], q2[1]], [q1[2], q2[2]],
                        color=color)