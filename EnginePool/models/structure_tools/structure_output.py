# -*- coding: utf-8 -*-
from models.utils import UserVars
import models.sxrd_new1 as model
import numpy as np
import scipy.spatial as spatial
from operator import mul
import operator
import os
from numpy.linalg import inv
from copy import deepcopy
from random import uniform

class StructureOutput(object):
    def __init__(self):
        pass

    #make dummy data set for test purpose, this function should be inside sim function
    #data is data I is the F (list of calculated structure factors)
    def make_dummy_data(file='D://temp_dummy_data.dat',data=None,I=None):
        data_full=np.zeros((0,8))
        for i in range(len(data)):
            data_set=data[i]
            x = data_set.x[:,np.newaxis]
            h = data_set.extra_data['h'][:,np.newaxis]
            k = data_set.extra_data['k'][:,np.newaxis]
            intensity = np.array(I[i])[:,np.newaxis]
            eI= data_set.error[:,np.newaxis]
            y = data_set.extra_data['Y'][:,np.newaxis]
            LB = data_set.extra_data['LB'][:,np.newaxis]
            dL = data_set.extra_data['dL'][:,np.newaxis]
            temp_set=np.concatenate((x,h,k,y,intensity,eI,LB,dL),axis=1)
            data_full=np.concatenate((data_full,temp_set),axis=0)
        np.savetxt(file,data_full,fmt='%.5e')
        return None

    def combine_all_datasets(file='D://temp_full_dataset.dat',data=None):
        data_full=np.zeros((0,8))
        for i in range(len(data)):
            data_set=data[i]
            x = data_set.x[:,np.newaxis]
            h = data_set.extra_data['h'][:,np.newaxis]
            k = data_set.extra_data['k'][:,np.newaxis]
            intensity = data_set.y[:,np.newaxis]
            eI= data_set.error[:,np.newaxis]
            y = data_set.extra_data['Y'][:,np.newaxis]
            LB = data_set.extra_data['LB'][:,np.newaxis]
            dL = data_set.extra_data['dL'][:,np.newaxis]
            temp_set=np.concatenate((x,h,k,y,intensity,eI,LB,dL),axis=1)
            data_full=np.concatenate((data_full,temp_set),axis=0)
        np.savetxt(file,data_full,fmt='%.5e')
        return None