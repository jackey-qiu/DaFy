import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Button

class Roi:
    def __init__(self, xlim, ylim, color='w'):
        self.xlim = xlim
        self.ylim = ylim
        self.color = color
    def get_no_pix(self):
        return (self.xlim[1]-self.xlim[0])*(self.ylim[1]-self.ylim[0])
    
'''
    Can be used to integrate the roi on an image and to subtract the background.
    
    The ImageIntegrator class is used to define the roi and the background ranges.
    The integrate function can then be called on images to integrate the roi.
'''
class ImageIntegrator:
    def __init__(self, roi, bg_rois=[]):
        self.roi = roi
        self.bg_rois = bg_rois
        
    def integrate(self, image, subtract_bg=True):
        I = np.sum(image[self.roi.ylim[0]:self.roi.ylim[1]+1, self.roi.xlim[0]:self.roi.xlim[1]+1])
        if(self.bg_rois):
            if(subtract_bg):
                I_pix = (self.roi.xlim[1]-self.roi.xlim[0])*(self.roi.ylim[1]-self.roi.ylim[0])
                bg = 0
                bg_pix = 0
                for bg_roi in self.bg_rois:
                    bg += np.sum(image[bg_roi.ylim[0]:bg_roi.ylim[1]+1, bg_roi.xlim[0]:bg_roi.xlim[1]+1])
                    bg_pix += (bg_roi.xlim[1]-bg_roi.xlim[0])*(bg_roi.ylim[1]-bg_roi.ylim[0])  
                return I-bg*I_pix/bg_pix
        
        return I
    
    def plot_cuts(self, image, figure=None, log_scale=False, hor_lim=[0,515], ver_lim=[0,515]):
        fig = figure if(figure) else plt.figure()
        if(np.round(fig.get_size_inches()[0], 0) != 12.0 or  np.round(fig.get_size_inches()[1], 0) != 6.0):
            fig.set_size_inches(12, 6, forward=True)
        #ax1 = plt.subplot2grid((3,2), (1,0), colspan=2, rowspan=2)
        ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=2)
        plt.setp(ax1.get_xticklabels(), fontsize=12)
        plt.setp(ax1.get_yticklabels(), fontsize=12)

        if(log_scale):
            _min = np.min(image[image > 0])
            image[image <= 0] = _min
            cax = ax1.imshow(image,interpolation='nearest', cmap=plt.get_cmap('jet'), norm=LogNorm(vmin=_min, vmax=np.max(image)))
        else:
            cax = ax1.imshow(image,interpolation='nearest', cmap=plt.get_cmap('jet'))
        im_size = image.shape
        
        ax1.plot([self.roi.xlim[0], self.roi.xlim[0]], [0, im_size[1]-1], color='r')
        ax1.plot([self.roi.xlim[1], self.roi.xlim[1]], [0, im_size[1]-1], color='r')
        ax1.plot([0, im_size[0]-1], [self.roi.ylim[0], self.roi.ylim[0]], color='r')
        ax1.plot([0, im_size[0]-1], [self.roi.ylim[1], self.roi.ylim[1]], color='r')
        
        ax1.plot([self.roi.xlim[0], self.roi.xlim[1]], [self.roi.ylim[0], self.roi.ylim[0]], color=self.roi.color)
        ax1.plot([self.roi.xlim[0], self.roi.xlim[1]], [self.roi.ylim[1], self.roi.ylim[1]], color=self.roi.color)
        ax1.plot([self.roi.xlim[0], self.roi.xlim[0]], [self.roi.ylim[0], self.roi.ylim[1]], color=self.roi.color)
        ax1.plot([self.roi.xlim[1], self.roi.xlim[1]], [self.roi.ylim[0], self.roi.ylim[1]], color=self.roi.color)
        
        for bg_roi in self.bg_rois:
            ax1.plot([bg_roi.xlim[0], bg_roi.xlim[1]], [bg_roi.ylim[0], bg_roi.ylim[0]], color=bg_roi.color)
            ax1.plot([bg_roi.xlim[0], bg_roi.xlim[1]], [bg_roi.ylim[1], bg_roi.ylim[1]], color=bg_roi.color)
            ax1.plot([bg_roi.xlim[0], bg_roi.xlim[0]], [bg_roi.ylim[0], bg_roi.ylim[1]], color=bg_roi.color)
            ax1.plot([bg_roi.xlim[1], bg_roi.xlim[1]], [bg_roi.ylim[0], bg_roi.ylim[1]], color=bg_roi.color)
        ax1.set_xlim([0, im_size[1]])
        ax1.set_ylim([im_size[0], 0])

        # horizontal cut
        #ax2 = plt.subplot2grid((3,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,0))        
        hor_cut = np.sum(image[self.roi.ylim[0]:self.roi.ylim[1]+1, :], 0)/(self.roi.ylim[1]-self.roi.ylim[0])
        ax2.plot(np.arange(len(hor_cut)), hor_cut)
        ax2.plot([self.roi.xlim[0], self.roi.xlim[0]], [0, np.max(hor_cut)], 'r-')
        ax2.plot([self.roi.xlim[1], self.roi.xlim[1]], [0, np.max(hor_cut)], 'r-')
        ax2.set_xlim(hor_lim)
        ax2.set_xlabel('horizontal cut', fontsize=12)
        plt.setp(ax2.get_xticklabels(), fontsize=12)
        plt.setp(ax2.get_yticklabels(), fontsize=12)
    
        #vertical cut
        #ax3 = plt.subplot2grid((3,2), (0,1))
        ax3 = plt.subplot2grid((2,2), (1,0))
        ver_cut = np.sum(image[:, self.roi.xlim[0]:self.roi.xlim[1]+1], 1)/(self.roi.xlim[1]-self.roi.xlim[0])
        ax3.plot(np.arange(len(ver_cut)), ver_cut)

        ax3.plot([self.roi.ylim[0], self.roi.ylim[0]], [0, np.max(ver_cut)], 'r-')
        ax3.plot([self.roi.ylim[1], self.roi.ylim[1]], [0, np.max(ver_cut)], 'r-')
        ax3.set_xlim(ver_lim)
        ax3.set_xlabel('vertical cut', fontsize=12)
        plt.setp(ax3.get_xticklabels(), fontsize=12)
        plt.setp(ax3.get_yticklabels(), fontsize=12)
        
        fig.tight_layout()
        plt.show()
       
class CutPlotterFigure:
    def __init__(self, figsize=(12, 6), frames_to_save=0): 
        self.figure = plt.figure(figsize=figsize)
        #TODO add stop button
        
if(__name__ == '__main__'):
    from pyspec import spec
    from PyMca5.PyMcaIO import EdfFile
    import time

    spec_path = '/home/finn/data/2014_12_04_MA2254/MA2254_Spec/'
    edf_path =  '/home/finn/data/2014_12_04_MA2254/MA2254_IMG/'
    
    f = spec.SpecDataFile(spec_path + 'ma2254_sixcvertical.spec')
    scan_no = 416
    frame_no = 0
    header = dict()
    for line in f[scan_no].header.split('\n'):
        header[line.split(' ')[0]] = line[len(line.split(' ')[0]):]
    
    frames = len(f[scan_no].Epoch)
    first_frame = int(header['#UCCD'].split('#r')[-1].split('.')[0])
    img_folder = header['#UCCD'].split('/')[-2]+'/' 
    
    
            
    cen_pix = [388, 338]
    dx, dy = 25, 25
    bg_dx = 15
    roi = Roi([cen_pix[0]-dx, cen_pix[0]+dx], [cen_pix[1]-dy, cen_pix[1]+dy])
    
    bg_rois = []
    bg_rois.append(Roi([cen_pix[0]-dx-bg_dx, cen_pix[0]-dx], [cen_pix[1]-dy, cen_pix[1]+dy]))
    bg_rois.append(Roi([cen_pix[0]+dx, cen_pix[0]+dx+bg_dx], [cen_pix[1]-dy, cen_pix[1]+dy]))

    integrator = ImageIntegrator(roi, bg_rois)
    
    plt.ion()
    fig = plt.figure()
    for frame_no in xrange(30): 
        img_filename = edf_path + "ma2254_mpx01/ma2254_" + str(scan_no).zfill(3) + '_' + str(frame_no).zfill(3) + '_'  + str(first_frame+frame_no).zfill(3) + '.edf.gz'
        edf = EdfFile.EdfFile(img_filename, 'r')
        img = edf.GetData(0)
        
        integrator.plot_cuts(img, fig)
        plt.draw()
        time.sleep(0.5)
    plt.ioff()
    plt.show()
    