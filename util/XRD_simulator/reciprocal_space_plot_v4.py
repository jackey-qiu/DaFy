# -*- coding: ascii -*-
"""
Created on Mon Jan 30 12:15:58 2017

@author: reikowski, wiegmann
"""

import numpy as np
from timeit import itertools
from mayavi import mlab
from tvtk.tools import visual
import timeit

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
    def is_in_limits_many(self, qs, qx_lims, qy_lims, q_inplane_lim, qz_lims, mag_q_lims=None):
        qsT = qs.transpose()
        qx = qsT[0]
        qy = qsT[1]
        qz = qsT[2]
        q_inpl = np.sqrt(qx**2 + qy**2)
        mag_q = np.sqrt(qx**2 + qy**2 + qz**2)
        
        in_lims = np.zeros(len(qs))
        
        if(qx_lims is not None):
            in_lims = np.logical_or(in_lims, qx < qx_lims[0])
            in_lims = np.logical_or(in_lims, qx > qx_lims[1])
        if(qy_lims is not None):
            in_lims = np.logical_or(in_lims, qy < qy_lims[0])
            in_lims = np.logical_or(in_lims, qy > qy_lims[1])
        if(qz_lims is not None):
            in_lims = np.logical_or(in_lims, qz < qz_lims[0])
            in_lims = np.logical_or(in_lims, qz > qz_lims[1])
        if(q_inplane_lim is not None):
            in_lims = np.logical_or(in_lims, q_inpl > q_inplane_lim)
        if(mag_q_lims is not None):
            in_lims = np.logical_or(in_lims, mag_q < mag_q_lims[0]) 
            in_lims = np.logical_or(in_lims, mag_q > mag_q_lims[1]) 

        return np.logical_not(in_lims) 

    # Plot Bragg peaks, with point size proportional to scattering intensity
    def plot_peaks(self, HKL_lims=[-10, 10], qx_lims=None, qy_lims=None,
                   q_inplane_lim=None, qz_lims=None, mag_q_lims=None, color=(0, 0, 0), scale_factor=1, scale_q=[1., 1., 1.]):
        HKL_range = np.arange(HKL_lims[0], HKL_lims[1] + 1)
        HKLs = np.array(list(itertools.product(HKL_range, repeat=3)))
        I0 = self.lattice.I([0,0,0])
        print('I0 =', I0)
        qs = self.lattice.q_many(HKLs)
        in_limits = self.is_in_limits_many(qs, qx_lims, qy_lims, q_inplane_lim, qz_lims, mag_q_lims)
        HKLs = HKLs[in_limits == True]
        qs = qs[in_limits == True]
        
        F = self.lattice.F_many(HKLs, qs)
        I = (F*F.conjugate()).real/I0

        indx = np.where(I > 1e-10)
        Is = I[indx]
        qs = qs[indx]
        HKLs = HKLs[indx]
        
        qs = np.swapaxes(qs, 0, 1)
        if(len(Is) > 0):
            return (mlab.points3d(qs[0]*scale_q[0], qs[1]*scale_q[1], qs[2]*scale_q[2], np.sqrt(Is), scale_factor=scale_factor, color=color, opacity=1), qs, Is, self.lattice)                
        return None
    # For each CTR (within given limits), returns one (arbitrary)
    # Bragg peak that is on the CTR
    def get_rod_positions(self, HKL_lims=[-10, 10], qx_lims=None, qy_lims=None,
                          q_inplane_lim=None, qz_lims=None, color=(0, 0, 0), ignore00rod=False, mag_q_lims=None):
        HKL_range = np.arange(HKL_lims[0], HKL_lims[1] + 1)
        HKLs = np.array(list(itertools.product(HKL_range, repeat=3)))
        I0 = self.lattice.I([0,0,0])
        qs = self.lattice.q_many(HKLs)
        in_limits = self.is_in_limits_many(qs, qx_lims, qy_lims, q_inplane_lim, qz_lims, mag_q_lims)
        HKLs = HKLs[in_limits == True]
        qs = qs[in_limits == True]
        
        F = self.lattice.F_many(HKLs, qs)
        I = (F*F.conjugate()).real/I0
        indx = np.where(I > 1e-10)
        Is = I[indx]
        qs = qs[indx]

        qIs = np.array([qs.T[0], qs.T[1], qs.T[2], Is]).T

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
                  q_inplane_lim=None, qz_lims=[-10, 10], color=(0, 0, 0), plot00rod=True, scale_q=[1., 1., 1.], line_width=4):
        qIs = self.get_rod_positions(HKL_lims=HKL_lims, qx_lims=qx_lims,
                                     qy_lims=qy_lims,
                                     q_inplane_lim=q_inplane_lim,
                                     qz_lims=qz_lims, color=color, ignore00rod=(not plot00rod))
        scaled_qz_lims = [qz_lims[0]*scale_q[2], qz_lims[1]*scale_q[2]]
        qx = []
        qy = []
        qz = []
        connections = []
        for qI in qIs:
            connections.append([len(qx), len(qx)+1])
            for i in range(2):
                qx.append(qI[0]*scale_q[0])
                qy.append(qI[1]*scale_q[1])            
            qz.append(scaled_qz_lims[0])
            qz.append(scaled_qz_lims[1])

            #mlab.plot3d([qI[0]*scale_q[0], qI[0]*scale_q[0]], [qI[1]*scale_q[1], qI[1]*scale_q[1]], scaled_qz_lims, color=color)
        src = mlab.pipeline.scalar_scatter(qx, qy, qz, color=color)
        src.mlab_source.dataset.lines = connections
        lines = mlab.pipeline.stripper(src)
        mlab.pipeline.surface(lines, color=color, line_width=line_width)
        
        #mlab.plot3d(qx, qy, qz, color=color)
 

    # Plot a grid at a specified qz value
    def plot_grid(self, HKL_lims=[-10, 10], grid_qz=0, qx_lims=None,
                  qy_lims=None, q_inplane_lim=None, qz_lims=None,
                  color=(0, 0, 0), scale_q=[1., 1., 1.]):
        # Get lateral positions of all CTRs
        qIs = self.get_rod_positions(HKL_lims=HKL_lims, qx_lims=qx_lims,
                                     qy_lims=qy_lims,
                                     q_inplane_lim=q_inplane_lim,
                                     qz_lims=qz_lims, color=color)

        # Calculate the distances between every point and every other point
        pos_pairs = []
        for i in range(len(qIs)):
            for j in range(i+1, len(qIs)):
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
        qx = []
        qy = []
        qz = []
        connections = []
        for pos_pair in pos_pairs:
            connections.append([len(qx), len(qx)+1])
            qx.append(pos_pair[0]*scale_q[0])
            qx.append(pos_pair[2]*scale_q[0])
            qy.append(pos_pair[1]*scale_q[1])
            qy.append(pos_pair[3]*scale_q[1])
            qz.append(grid_qz*scale_q[2])
            qz.append(grid_qz*scale_q[2])
            #mlab.plot3d([pos_pair[0]*scale_q[0], pos_pair[2]*scale_q[0]], [pos_pair[1]*scale_q[1], pos_pair[3]*scale_q[1]],
            #            [grid_qz*scale_q[2], grid_qz*scale_q[2]], color=color)
        src = mlab.pipeline.scalar_scatter(qx, qy, qz, color=color)
        src.mlab_source.dataset.lines = connections
        lines = mlab.pipeline.stripper(src)
        mlab.pipeline.surface(lines, color=color, line_width=4)

    # Plot the unit cell
    def plot_unit_cell(self, color=(0, 0, 0), scale_q=[1., 1., 1.]):
        # Unit cell edge lines to be drawn are hard-coded
        HKLs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],
                [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        lines = [[0, 1], [0, 2], [0, 4], [5, 1], [5, 4], [5, 7], [3, 1],
                 [3, 2], [3, 7], [6, 4], [6, 2], [6, 7]]
        for line in lines:
            q1 = self.lattice.q(HKLs[line[0]])
            q2 = self.lattice.q(HKLs[line[1]])
            mlab.plot3d([q1[0]*scale_q[0], q2[0]*scale_q[0]], [q1[1]*scale_q[1], q2[1]*scale_q[1]], [q1[2]*scale_q[2], q2[2]*scale_q[2]],
                        color=color)
