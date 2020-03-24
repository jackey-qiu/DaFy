# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:42:18 2017

@author: Timo
"""

import numpy as np
import numpy.linalg as LA
import scipy.optimize as opt
from math import *
import os
import matplotlib.pyplot as plt
from matplotlib import colors as colors
#import configparser
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def makeSurePathExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def as_ndarray(obj):
    """make sure a float, int, list of floats or ints,
    or tuple of floats or ints, acts as a numpy array
    """
    if isinstance(obj, (float, int)):
        return np.array([obj])
    return np.asarray(obj)
    
def readNDarrayConfig(filename):
    
    header = StringIO()
    
    with open(filename,'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            header.write(line[2:])
    config = configparser.ConfigParser()
    config.read_string(header.getvalue())
    
    data = np.loadtxt(filename)
    return data.T, config
    
    

def sumRoi(image,roi):
	xlo,xhi = roi[0]
	ylo,yhi = roi[1]
	image_tmp = image[xlo:xhi,ylo:yhi]
	I = np.nansum(image_tmp)
	pixel = np.count_nonzero(~np.isnan(image_tmp))
	return I, pixel

def calcIntersect(g1,g2):
    g1P1 , g1P2 = g1
    g2P1 , g2P2 = g2
    
    a = g1P1 - g1P2
    b = g2P1 - g2P2
    
    res = LA.solve(np.array([a,-b]),g2P1-g1P1)
    
    return a*res[0] + g1P1
"""
gives delta in:
    
"""

def solveTrigEquation(theta1,theta2,l1,l2):
    
    def Chi2(delta):
        res = function(theta1,theta2,l1,l2,delta)
        #print("%s : %s" % (delta,res))
        return res
    
    res = opt.minimize(Chi2,np.array([0.0]),bounds=[(-np.pi,np.pi)],method='TNC')
    print(res)
    print(np.rad2deg(res.x))
    return res.x

def function(theta1,theta2,l1,l2,delta):
    t1 = cos(theta1 - delta[0])**2 * l1**2 
    t2 = cos(theta1 + delta[0])**2 * l2**2
    t3 = 2 * l1 * l2 * cos(theta1 + delta[0]) * cos(theta1 - delta[0]) * cos(2*theta2)
    t4 = ( (l2**2) * cos(theta1 + delta[0])**2 * sin(2*theta2)**2 )/( cos(delta[0] - theta2)**2 )
    return fabs(t1 + t2 - t3 - t4)

#solveTrigEquation(np.deg2rad(2.),np.deg2rad(76.),1.,1.1)

"""
only single value of th:
"""
def x_rotation(th):
    return np.matrix(((1., 0., 0.), (0., np.cos(th), -np.sin(th)), (0., np.sin(th), np.cos(th))))


def y_rotation(th):
    return np.matrix(((np.cos(th), 0., np.sin(th)), (0, 1., 0.), (-np.sin(th), 0., np.cos(th))))


def z_rotation(th):
    return np.matrix(((np.cos(th), -np.sin(th), 0.), (np.sin(th), np.cos(th), 0.), (0., 0., 1.)))

"""
th is now a 1d-array, returns an array of rotation matrices
"""
def x_rotationArray(th):
    matrices = np.zeros((th.size,3,3))
    matrices[:,0,0] = 1
    matrices[:,1,1] = np.cos(th)
    matrices[:,1,2] = -np.sin(th)
    matrices[:,2,1] = np.sin(th)
    matrices[:,2,2] = np.cos(th)
    return matrices
    

def y_rotationArray(th):
    matrices = np.zeros((th.size,3,3))
    matrices[:,1,1] = 1
    matrices[:,0,0] = np.cos(th)
    matrices[:,0,2] = np.sin(th)
    matrices[:,2,0] = -np.sin(th)
    matrices[:,2,2] = np.cos(th)
    return matrices


def z_rotationArray(th):
    matrices = np.zeros((th.size,3,3))
    matrices[:,2,2] = 1
    matrices[:,0,0] = np.cos(th)
    matrices[:,0,1] = -np.sin(th)
    matrices[:,1,0] = np.sin(th)
    matrices[:,1,1] = np.cos(th)
    return matrices

def orthogonal(matrix):
    matrix = np.matrix(matrix)
    
    SMALL = 1e-4
    
    def normalise(m):
        d = LA.norm(m)
        if d < SMALL:
            raise Exception("Error: can't make matrix orthogonal")
        return m / d
    #print(LA.norm(v1))
    v1 = normalise(matrix[:,0])
    print(matrix[:,0])
    print(LA.norm(v1))
    v2 = normalise(matrix[:,1])
    v3 = normalise(matrix[:,2])
    
    return np.hstack([v1, v2, v3]).A

# slow!!!
def calcHighPixel(image,threshold):
    highintensity = image > threshold
    highpixel = []

    for x,xrow in enumerate(highintensity):
        for y in range(xrow.size):
            if highintensity[x][y]:
                highpixel.append([y,x])

    highpixel = np.array(highpixel).T
    return highpixel

def plotP3Image(image,vmin=0,vmax=None,cmap='jet',thresholdMarker=None,**keyargs):
    if 'figure' in keyargs:
        fig = keyargs['figure']
        ax = fig.add_subplot(111)
    elif 'axis' in keyargs:
        ax = keyargs['axis']
        fig = ax.get_figure()
    else:
        fig = plt.figure(figsize=(12,14))
        ax = fig.add_subplot(111)

    if not vmax:
        ax.imshow(image,interpolation='none',cmap=plt.get_cmap(cmap),norm=colors.SymLogNorm(linthresh=1,linscale=1,vmin=vmin))
    else:
        ax.imshow(image,interpolation='none',cmap=plt.get_cmap(cmap),norm=colors.SymLogNorm(linthresh=1,linscale=1,vmin=vmin,vmax=vmax))
    
    if thresholdMarker is not None:
        highpixel = calcHighPixel(thresholdMarker)
        if highpixel.size > 0:
            ax.plot(highpixel[0],highpixel[1],'ro')
    ax.set_ylim([1700,0])
    ax.set_xlim([0,1475])
    numrows, numcols = image.shape
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = image[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    ax.format_coord = format_coord
    return fig, ax

#def deltaGamma(drr,gamma):
#    return np.arctan((1 + drr) / np.abs(np.cos(gamma))) - gamma