import numpy as np
from nexusformat.nexus import *
import os
import matplotlib as mat
from matplotlib import pyplot
import tkinter
class nexus_image_loader(object):
    def __init__(self,fio_path='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/startup/FirstTest_00666.fio',nexus_path='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/FirstTest_00666/lmbd',frame_prefix='FirstTest'):
        self.fio_path=fio_path
        self.nexus_path=nexus_path
        self.frame_prefix=frame_prefix
        self.frame_number=1

    def get_frame_number(self, scan_number, one_frame_in_one_nxs = True):
        if one_frame_in_one_nxs:
            self.frame_number = 1
        else:
            img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
            img_path=os.path.join(self.nexus_path,img_name)
            data=nxload(img_path)
            self.frame_number = len(np.array(data.entry.instrument.detector.data))
        return self.frame_number

    def load_frame(self,scan_number,frame_number,one_frame_in_one_nxs=True,flip=True):
        if one_frame_in_one_nxs:
            img_name='{}_{:0>5}_{:0>5}.nxs'.format(self.frame_prefix,scan_number,frame_number)
            img_path=os.path.join(self.nexus_path,img_name)
        else:
            img_name='{}_{:0>5}.nxs'.format(self.frame_prefix,scan_number)
            img_path=os.path.join(self.nexus_path,img_name)
        data=nxload(img_path)
        if one_frame_in_one_nxs:
            img=np.array(data.entry.instrument.detector.data.nxdata[0])
        else:
            img=np.array(data.entry.instrument.detector.data[frame_number])
        if flip:
            return np.flip(img.T,1)
        else:
            return img

    def show_frame(self,scan_number,frame_number,one_frame_in_one_nxs=True,flip=True):
        img=self.load_frame(scan_number,frame_number,one_frame_in_one_nxs,flip)
        fig,ax=pyplot.subplots()
        pyplot.imshow(img,cmap='jet')
        if flip:
            pyplot.colorbar(extend='both',orientation='vertical')
        else:
            pyplot.colorbar(extend='both',orientation='horizontal')
        pyplot.clim(0,205)
        # pyplot.show()
        return img 

    def find_dead_pix(self,scan_number=666,img_end=100):
        dead_pix_container=self.load_frame(scan_number,0)==self.load_frame(scan_number,1)
        dead_pix_container=np.where(dead_pix_container==True)
        dead_pix_container=zip(tuple(dead_pix_container[0]),tuple(dead_pix_container[1]))
        img0= self.load_frame(scan_number,0)
        print(len(dead_pix_container))
        for i in range(2,img_end):
            print('Processing img_',i)
            img = self.load_frame(scan_number,i)
            temp= img != img0
            temp= np.where(temp==True)
            temp= zip(tuple(temp[0]),tuple(temp[1]))
            for each in temp:
                if each in dead_pix_container:
                    dead_pix_container.remove(each)
        return dead_pix_container

if __name__=='__main__':
    test=nexus_image_loader()
    test.show_frame(666,0)
