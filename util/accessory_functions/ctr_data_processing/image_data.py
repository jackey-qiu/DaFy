"""
Methods to handle image processing

Authors/Modifications:
----------------------
* Tom Trainor (tptrainor@alaska.edu)
* Frank Heberling (Frank.Heberling@ine.fzk.de)
* Matt Newville (newville@cars.uchicago.edu)

Notes:
------
Reads in data for the pilatus detector
Simple integrations and plotting

Todo:
-----
* Improve /speedup image background determination
* Try other background options - spline/interpolate and variable size to compute stats
* Also need arbitrary roi polygon...

Notes:
------
There is an issue with organization of PIL/Image modules when
try to build an exe.  Therefore, just do import of Image
inside the functions that need it (read_file)

Also, note in older versions of PIL:
 To read the pilatus, you must add this line:
    (1, 1, 1, (32,), ()): ("F", "F;32F"),
 to the OPEN_INFO dict in TiffImagePlugin.py (part of the Image module)

 starting at line 130 of TiffImagePlugin.py (from PIL 1.1.6):
   (1, 1, 2, (8,), ()): ("L", "L;R"),
   (1, 1, 1, (16,), ()): ("I;16", "I;16"),
   (1, 1, 1, (32,), ()): ("F", "F;32F"),
   (1, 2, 1, (16,), ()): ("I;16S", "I;16S"),
"""
#######################################################################

import types
import os
import copy
import numpy as num
from matplotlib import pyplot
from scipy import ndimage

from background import background

########################################################################
IMG_BGR_PARAMS = {'bgrflag':1,
                  'cnbgr':5,'cwidth':0,'cpow':2.,'ctan':False,
                  'rnbgr':5,'rwidth':0,'rpow':2.,'rtan':False,
                  'nline':1,'filter':False,'compress':1}

#######################################################################
def read(file,pixel_map=None):
    """
    read file or list of files
    """
    try:
        from PIL import Image
        imopen  = Image.open
    except:
        print "Error importing Image.open"
        return None
    # see if there is a pixel map
    if pixel_map != None:
        (bad_pixels,good_pixels) = read_pixel_map(pixel_map)
    else:
        bad_pixels = None
        good_pixels = None
    # read fcn
    def rd(file,bad_pixels=None,good_pixels=None):
        try:
            im  = imopen(file)
            arr = num.fromstring(im.tostring(), dtype='int32')
            arr.shape = (im.size[1],im.size[0])
            #return arr.transpose()
            if bad_pixels != None:
                arr = pixel_mask(arr,bad_pixels,good_pixels)
            return arr
        except:
            print "Error reading file: %s" % file
            return None
    if type(file) == types.StringType:
        image = rd(file,bad_pixels=bad_pixels,good_pixels=good_pixels)
        return image
    elif type(file) == types.ListType:
        image = []
        for f in file:
            tmp = rd(file,bad_pixels=bad_pixels,good_pixels=good_pixels)
            image.append(tmp)
    else:
        return None
    return image

#######################################################################
def correct_image(image,pixel_map=None):
    """
    Given an image and a path to a bad pixel map,
    return the corrected image.
    """

    if pixel_map is None or pixel_map.startswith('[]') or \
                                                pixel_map.startswith('None'):
        return image
    else:
        try:
            pixel_map = eval(pixel_map)
            (bad_pixels,good_pixels) = pixel_map
        except:
            (bad_pixels,good_pixels) = ([], [])
        return pixel_mask(image, bad_pixels, good_pixels)
        
#######################################################################
def read_files(file_prefix,start=0,end=100,nfmt=3,pixel_map=None):
    """
    read files that have a numerical suffix
    """
    images  = []
    format = '%' + str(nfmt) + '.' + str(nfmt) + 'd'
    for j in range(start,end+1):
        ext  = format % j
        file = file_prefix + '_' + ext + '.tif'
        arr  = read(file,pixel_map=pixel_map)
        images.append(arr)
    return images

############################################################################
def pixel_mask(image,bad_pixels=[],good_pixels=[]):
    """
    Correct bad pixels in the image

    Paramters:
    ---------
    * image is a image array
    * bad_pixels is a list of [x,y] pairs corresponding
      to bad pixel locations
    * good_pixels is a list of [x,y] pairs corresponding
      to good pixel locations (optional)

    Returns:
    -------
    * The image with the bad_pixels set to either the average value of
      the nieghbors or to the corresponding good pixel value

    Notes:
    ------
    *  should we be concerned about types and rounding errors?
    """
    if bad_pixels == None: return
    if good_pixels == None:
        good_pixels = []
    if len(good_pixels) == 0:
        do_ave = True
    else:
        do_ave = False
    ny = len(image)
    nx = num.max(image[0])
    for j in range(len(bad_pixels)):
        x = bad_pixels[j][0]
        y = bad_pixels[j][1]
        if do_ave == True:
            avg = 0.
            na = 0.
            if (x+1) < nx:
                avg = avg + image[y][x+1]
                na = na + 1.
            if (x-1) >= 0:
                avg = avg + image[y][x-1]
                na = na + 1.
            if (y+1) < ny:
                avg = avg + image[y+1][x]
                na = na + 1.
            if (y-1) >= 0:
                avg = avg + image[y-1][x]
                na = na + 1.
            if na > 0:
                #print int(avg / na), image[y,x]
                image[y,x] = int(avg / na)
        else:
            x2 = good_pixels[j][0]
            y2 = good_pixels[j][1]
            image[y,x] = image[y2,x2]
    return image

def read_pixel_map(fname):
    """
    read pixel map file

    assume the file format is
    (bad pixel)  (good pixel)
    """
    bad_pixels = []
    good_pixels = []
    try:
        bad = []
        good = []
        f  = open(fname)
        for line in f.readlines():
            tmp = line.strip().split()
            bad.append(tmp[0])
            if len(tmp)>1:
                good.append(tmp[1])
        f.close()
        for p in bad:
            pp = p.split(',')
            bad_pixels.append(map(int,pp))
        for p in good:
            pp = p.split(',')
            good_pixels.append(map(int,pp))
        return bad_pixels, good_pixels
    except:
        print "Error reading file: %s" % fname
        return []
    
############################################################################
def clip_image(image,roi=[],rotangle=0.0,cp=False):
    """
    Clip an image given the roi

    Parameters:
    -----------
    * roi is a list [c1,r1,c2,r2] = [x1,y1,x2,y2]
      Note take x as the horizontal image axis index (col index), 
      and y as vertical axis index (row index).
      Therefore, in image indexing:
         image[y1:y2,x1:x2] ==> rows y1 to y2 and cols x1 to x2
    * rotangle is the angle to rotate the image by
    * cp is flag to indicate if a new copy of the image is generated
    
    """
    if len(roi) != 4:
        roi = [0,0,image.shape[1], image.shape[0]]
    else:
        [c1,r1,c2,r2] = _sort_roi(roi)
    # rotate
    if rotangle != 0:
        image = ndimage.rotate(image,rotangle)    

    if cp == True:
        return copy.copy(image[r1:r2, c1:c2])
    else:
        return image[r1:r2, c1:c2]

def _sort_roi(roi):
    """
    roi = [c1,r1,c2,r2]
    """
    if roi[0] < roi[2]:
        c1 = roi[0]
        c2 = roi[2]
    else:
        c1 = roi[2]
        c2 = roi[0]
    if roi[1] < roi[3]:
        r1 = roi[1]
        r2 = roi[3]
    else:
        r1 = roi[3]
        r2 = roi[1]
    return [c1,r1,c2,r2]
    
def calc_roi(dx=100,dy=100,shape=(195,487),cen=None):
    """
    Calculate an roi given a width, hieght and center

    Parameters:
    -----------
    * dx is the roi width (number of columns) in pixels
    * dy is the roi height (number of rows) in pixels
    * shape is the image shape (nrows,ncols)
    * cen is the tuple (r,c) for the central pixel.  If None
      then cen is calculated from the shape

    Returns:
    --------
    * [c1,r1,c2,r2] = [x1,y1,x2,y2] where c1/2,r1/2 are the
      column (x) and row (y) values defining the roi box.  
      Therefore the image can be indexed as: image[y1:y2,x1:x2]

    Examples:
    ---------
    >> r = calc_roi(dx=20,dy=50,shape=a.shape)

    """
    if cen == None:
        cen = (int(shape[0]/2),int(shape[1]/2))
    c1 = int(cen[1] - int(dx/2))
    c2 = int(cen[1] + int(dx/2))
    r1 = int(cen[0] - int(dy/2))
    r2 = int(cen[0] + int(dy/2))
    roi = _sort_roi([c1,r1,c2,r2])
    return roi

############################################################################
def image_plot(img,fig=None,figtitle='',cmap=None,verbose=False,
               im_max=None,rotangle=0.0,roi=None):
    """
    show image
    
    Parameters:
    -----------
    * img              # the image array to be displayed
    * fig = None       # Figure to plot to
    * figtitle = ''    # Title
    * cmap = None      # Colormap.  None uses default
                       # you can pass a string name if its in pyplot.cm.colormaps
                       # or you can pass explicitly the colormap
    * verbose = False  # Print some fig statistics
    * im_max  = None   # Max intensity value
    * rotangle = 0.0   # Rotation angle in degrees ccw
    * roi = None       # Plot an roi -> [x1,y1,x2,y2]

    Examples:
    ---------
    >>image_plot(im,fig=1,figtitle='Image',cmap='hot')
    >>image_plot(im,fig=1,figtitle='Image',cmap=pyplot.cm.Spectral)
    """
    if verbose:
        print '-----'
        print 'Some statistics for plotted image'
        print 'Image total= ', img.sum()
        print 'Max value = ',  img.max()
        print 'Min value = ',  img.min()
        print '-----'
    if fig != None:
        pyplot.figure(fig)
        pyplot.clf()

    if rotangle != 0:
        img = ndimage.rotate(img,rotangle)
    if cmap != None:
        if type(cmap) == types.StringType:
            #if cmap in pyplot.cm.cmapnames:
            try:
                cmap = getattr(pyplot.cm,cmap)
            except:
                cmap = None
    if im_max != None:
        if im_max < 1: im_max = None
    #
    if roi != None:
        [c1,r1,c2,r2] = _sort_roi(roi)
        bild = num.zeros(img.shape)
        bild[r1-2:r1,c1:c2] = img.max()
        bild[r2:r2+2,c1:c2] = img.max()
        bild[r1:r2,c1-2:c1] = img.max()
        bild[r1:r2,c2:c2+2] = img.max()
        img = img+bild
        if im_max == None:
            im_max = num.max(img[r1:r2, c1:c2])
    #
    pyplot.imshow(img,cmap=cmap,vmax=im_max)
    pyplot.colorbar(orientation='horizontal')

    if figtitle:
        pyplot.title(figtitle, fontsize = 12)

############################################################################
def sum_plot(image,bgrflag=0,
             cnbgr=5,cwidth=0,cpow=2.,ctan=False,
             rnbgr=5,rwidth=0,rpow=2.,rtan=False,
             fig=None):
    """
    Plot sums with background.

    Note should we calc the bgr according
    to bgrflag???
    """
    #
    if fig != None:
        pyplot.figure(fig)
        pyplot.clf()
    else:
        pyplot.figure()

    # col sum
    pyplot.subplot(211)
    pyplot.title('Column Sum')
    (data, data_idx, bgr) = line_sum(image,sumflag='c',nbgr=cnbgr,
                                     width=cwidth,pow=cpow,tangent=ctan)
    pyplot.plot(data_idx, data, 'r')
    pyplot.plot(data_idx, bgr, 'b')

    # row sum
    pyplot.subplot(212)
    pyplot.title('Row Sum')
    (data, data_idx, bgr) = line_sum(image,sumflag='r',nbgr=rnbgr,
                                     width=rwidth,pow=rpow,tangent=rtan)
    pyplot.plot(data_idx, data, 'r')
    pyplot.plot(data_idx, bgr, 'b')

############################################################################
def line_sum(image,sumflag='c',nbgr=0,width=0,pow=2.,tangent=False,compress=1):
    """
    Sum down 'c'olumns or across 'r'ows
    
    This returns the summed data and background
    """
    if sumflag == 'c':
        data = image.sum(axis=0)
    elif sumflag == 'r':
        data = image.sum(axis=1)
    npts     = len(data)
    data_idx = num.arange(npts,dtype=float)
    #data_err  = data**(0.5)

    ### compute background
    bgr = background(data,nbgr=nbgr,width=width,pow=pow,tangent=tangent,
                     compress=compress)
    
    return (data, data_idx, bgr)

############################################################################
def line_sum_integral(image,sumflag='c',nbgr=0,width=0,pow=2.,
                      tangent=False,compress=1):
    """
    Calc the integral after image is summed down 'c'olumns
    or across 'r'ows.
    
    Returns:
    --------
    * tuple (I,Ierr,Ibgr)
    * I = the integrated background subtracted intensity
    * Ierr = the standard deviation (std_dev) of I.  
             Assume that the errors follow counting statistics:
             err = std_dev = sqrt(variance) = sqrt(signal)
    * Ibgr = the background intensity
    
    """
    # get line sum
    (data, data_idx, bgr) = line_sum(image,sumflag=sumflag,nbgr=nbgr,width=width,
                                     pow=pow,tangent=tangent,compress=compress)
    ### integrate
    Itot = data.sum()
    Ibgr = bgr.sum()
    #Itot = num.trapz(data)
    #Ibgr = num.trapz(bgr)
    I    = Itot - Ibgr
    
    ### compute errors
    #Ierr = (data.sum() + bgr.sum())**(0.5)
    Ierr = (Itot + Ibgr)
    if Ierr > 0.:
        Ierr = num.sqrt(Ierr)
    else:
        Ierr = 0.0
    
    return(I,Ierr,Ibgr)

##############################################################################
def image_bgr(image,lineflag='c',nbgr=3,width=100,pow=2.,tangent=False,
              nline=1,filter=False,compress=1,plot=False):
    """
    Calculate a 2D background for the image.

    Parameters:
    -----------
    * image is the (hopefully clipped) image data

    * lineflag ('c' or 'r') corresponds to to direction which is
      used to generate the background 

    * nbgr = number of end points to use in linear background determination
      (see background.background)
          
    * width should correspond roughly to the actual peak
      width in the lineflag direction. The background should fit
      features that are in general broader than this value
      Note that width = 0 corresponds to no polynomial bgr
      
    * pow is the power of the polynomial used in background determination
      (see background.background)
      
    * tangent is a flag to indicate if local slope of the data should be fitted 
      (see background.background)

    * nline = number of lines to average for each bgr line fit

    * filter (True/False) flag indicates if a spline filter should be applied
      before background subtraction.  (see scipy.ndimage.filters)

    * compress is a factor by which the number of points in each line is
      reduced.  This helps speed up the background fits.  
     
    * plot is a flag to indicate if a 'plot' should be made
    """
    bgr_arr = num.zeros(image.shape)

    # note this works poorly if the filter removes
    # too much intensity.  Use with caution!
    if filter == True:
        #image = ndimage.laplace(image)
        #print 'spline filter'
        image = ndimage.interpolation.spline_filter(image,order=3)

    # fit to rows
    if lineflag=='r':
        if nline > 1:
            ll = int(nline/2.)
            n = image.shape[0]
            line = num.zeros(len(image[0]))
            for j in range(n):
                idx = [k for k in range(j-ll,j+ll+1) if k>=0 and k<n]
                line = line * 0.0
                for k in idx:
                    line = line + image[k]
                line = line/float(len(idx))
                bgr_arr[j,:] = background(line,nbgr=nbgr,width=width,pow=pow,
                                          tangent=tangent,compress=compress)
        else:
            for j in range(image.shape[0]):
                bgr_arr[j,:] = background(image[j],nbgr=nbgr,width=width,pow=pow,
                                          tangent=tangent,compress=compress)

    # fit to cols
    if lineflag=='c':
        if nline > 1:
            ll = int(nline/2.)
            n = image.shape[1]
            line = num.zeros(len(image[:,0]))
            for j in range(n):
                idx = [k for k in range(j-ll,j+ll+1) if k>=0 and k<n]
                line = line * 0.0
                for k in idx:
                    line = line + image[:,k]
                line = line/float(len(idx))
                bgr_arr[:,j] = background(line,nbgr=nbgr,width=width,pow=pow,
                                          tangent=tangent,compress=compress)
        else:
            for j in range(image.shape[1]):
                bgr_arr[:,j] = background(image[:,j],nbgr=nbgr,width=width,pow=pow,
                                          tangent=tangent,compress=compress)
    #show
    if plot:
        
        pyplot.figure(3)
        pyplot.clf()
        pyplot.subplot(3,1,1)
        pyplot.imshow(image)
        pyplot.title("image")
        pyplot.colorbar()

        pyplot.subplot(3,1,2)
        pyplot.imshow(bgr)
        pyplot.title("background")
        pyplot.colorbar()

        pyplot.subplot(3,1,3)
        pyplot.imshow(image-bgr)
        pyplot.title("image - background")
        pyplot.colorbar()

    return bgr_arr

################################################################################
class ImageAna:
    """
    Analyze images
    """
    def __init__(self,image,roi=[],rotangle=0.0,
                 bgrflag=1,
                 cnbgr=5,cwidth=0,cpow=2.,ctan=False,
                 rnbgr=5,rwidth=0,rpow=2.,rtan=False,
                 nline=1,filter=False,compress=1,
                 plot=True,fig=None,figtitle='',
                 clpimg=None,bgrimg=None,integrated=False,
                 I=0.0,Ibgr=0.0,Ierr=0.0,I_c=0.0,I_r=0.0,
                 Ibgr_c=0.0,Ibgr_r=0.0,Ierr_c=0.0,Ierr_r=0.0,
                 im_max=-1):
        """
        Initialize

        Parameters:
        -----------
        * image is the image data

        * roi is the 'peak' roi = [x1,y1,x2,y2]
          Note take x as the horizontal image axis index (col index), 
          and y as vertical axis index (row index).
          Therefore, in image indexing:
            image[y1:y2,x1:x2] ==> rows y1 to y2 and cols x1 to x2

        * rotangle is the image rotation angle in degrees ccw

        * bgrflag is flag for how to do backgrounds:
           = 0 determine row and column backgrounds after summation
           = 1 determine 2D background using 'c'olumn direction 
           = 2 determine 2D background using 'r'ow direction 
           = 3 determine 2D background from the average 'r'ow and 'c'olumn directions 

        -> below params are for 'c'olumn and 'r'ow directions

        * c/rnbgr = number of end points to use in linear background determination
          (see background.background)
              
        * c/rwidth should correspond roughly to the actual peak
          widths. The background function should fit
          features that are in general broader than these values
          Note estimate cwidth using width of peak in row sum
          and rwidth using the width of the peak in the col sum.
          Note that width = 0 corresponds to no polynomial bgr
          
        * c/rpow is the power of the polynomial used in background determination
          (see background.background)
          
        * c/rtangent is a flag to indicate if local slope of the data should be fitted 
          (see background.background)

        * nline = number of lines to average for each bgr line fit
    
        * filter (True/False) flag indicates if a spline filter should be applied
          before background subtraction.  (see scipy.ndimage.filters)

        * compress is a factor by which the number of points in each line is
          reduced.  This helps speed up the background fits.  
        
        * plot = show fancy plot 

        """
        ### roi = [x1,y1,x2,y2] = [c1,r1,c2,r2]
        if len(roi) < 4:
            roi = [0,0,image.shape[1], image.shape[0]]
        if roi[0] < roi[2]:
            c1 = roi[0]
            c2 = roi[2]
        else:
            c1 = roi[2]
            c2 = roi[0]
        if roi[1] < roi[3]:
            r1 = roi[1]
            r2 = roi[3]
        else:
            r1 = roi[3]
            r2 = roi[1]
        self.roi    = (int(c1),int(r1),int(c2),int(r2))
        self.rotangle   = rotangle
        self.image      = image
        self.clpimg     = clpimg
        self.bgrimg     = bgrimg
        self.integrated = integrated
        #
        self.title  = figtitle
        #
        self.I      = I
        self.Ibgr   = Ibgr
        self.Ierr   = Ierr
        #
        self.I_c    = I_c
        self.I_r    = I_r
        self.Ibgr_c = Ibgr_c
        self.Ibgr_r = Ibgr_r
        self.Ierr_c = Ierr_c
        self.Ierr_r = Ierr_r
        #
        self.bgrflag = bgrflag
        self.cbgr = {'nbgr':cnbgr,'width':cwidth,'pow':cpow,'tan':ctan}
        self.rbgr = {'nbgr':rnbgr,'width':rwidth,'pow':rpow,'tan':rtan}
        self.nline  = nline
        self.filter = filter
        self.compress = compress
        self.plotflag = plot
        self.im_max = im_max
        
        if not self.integrated:
            self.integrate()

        # plot
        if self.plotflag: self.plot(fig=fig)
            
    ############################################################################
    def get_vars(self):
        ret = (self.clpimg, self.bgrimg, 
               self.integrated, self.I, 
               self.Ibgr, self.Ierr,
               self.I_c, self.I_r,
               self.Ibgr_c, self.Ibgr_r,
               self.Ierr_c, self.Ierr_r )
        return ret
    
    ############################################################################
    def integrate(self):
        """
        Integrate image.

        Note approx error by assuming data and bgr std
        deviation (sig) are:
           sig_i = sqrt(I_i)
           
        Therefore:
          (sig_Itot)**2 = Sum(I_i)
          (sig_Ibgr)**2 = Sum(Ibgr_i)
          Ierr = sqrt((sig_Itot)**2 + (sig_Ibgr)**2)

        """
        # clip image
        self.clpimg = clip_image(self.image,self.roi,rotangle=self.rotangle)

        # integrate the roi
        #self.I = num.sum(num.trapz(self.clpimg))
        self.I = num.sum(self.clpimg)

        # calculate and subtract image background
        self.bgrimg = None
        self.Ibgr = 0.0
        if self.bgrflag > 0:
            if self.bgrflag == 1:
                self.bgrimg = image_bgr(self.clpimg,lineflag='c',nbgr=self.cbgr['nbgr'],
                                        width=self.cbgr['width'],pow=self.cbgr['pow'],
                                        tangent=self.cbgr['tan'],nline=self.nline,
                                        filter=self.filter,compress=self.compress,
                                        plot=False)
            elif self.bgrflag == 2:
                self.bgrimg = image_bgr(self.clpimg,lineflag='r',nbgr=self.rbgr['nbgr'],
                                        width=self.rbgr['width'],pow=self.rbgr['pow'],
                                        tangent=self.rbgr['tan'],nline=self.nline,
                                        filter=self.filter,compress=self.compress,
                                        plot=False)
            else:
                bgr_r = image_bgr(self.clpimg,lineflag='c',nbgr=self.cbgr['nbgr'],
                                  width=self.cbgr['width'],pow=self.cbgr['pow'],
                                  tangent=self.cbgr['tan'],nline=self.nline,
                                  filter=self.filter,compress=self.compress,plot=False)
                bgr_c = image_bgr(self.clpimg,lineflag='r',nbgr=self.rbgr['nbgr'],
                                  width=self.rbgr['width'],pow=self.rbgr['pow'],
                                  tangent=self.rbgr['tan'],nline=self.nline,
                                  filter=self.filter,compress=self.compress,plot=False)
                # combine the two bgrs by taking avg
                self.bgrimg = (bgr_r + bgr_c)/2.
                    
            # correct for 2D bgr
            #self.Ibgr = num.sum(num.trapz(self.bgrimg))
            self.Ibgr = num.sum(self.bgrimg)
            self.I    = self.I - self.Ibgr
        
        # error
        self.Ierr = (self.I + self.Ibgr)**0.5
        
        # integrate col sum
        if self.bgrimg != None:
            (I,Ierr,Ibgr) = line_sum_integral(self.clpimg-self.bgrimg,sumflag='c',nbgr=0)
        else:
            (I,Ierr,Ibgr) = line_sum_integral(self.clpimg,sumflag='c',
                                              nbgr=self.rbgr['nbgr'],
                                              width=self.rbgr['width'],
                                              pow=self.rbgr['pow'],
                                              tangent=self.rbgr['tan'],
                                              compress=self.compress)
        self.I_c      = I
        self.Ierr_c   = Ierr
        self.Ibgr_c   = Ibgr
        
        # integrate row sum
        if self.bgrimg != None:
            (I,Ierr,Ibgr) = line_sum_integral(self.clpimg-self.bgrimg,sumflag='r',nbgr=0)
        else:
            (I,Ierr,Ibgr) = line_sum_integral(self.clpimg,sumflag='r',
                                              nbgr=self.cbgr['nbgr'],
                                              width=self.cbgr['width'],
                                              pow=self.cbgr['pow'],
                                              tangent=self.cbgr['tan'],
                                              compress=self.compress)
        self.I_r      = I
        self.Ierr_r   = Ierr
        self.Ibgr_r   = Ibgr
        #
        self.integrated = True

    ############################################################################
    def plot(self,fig=None):
        """
        make fancy 4-panel plot. fig is figure number to plot to
        """
        if self.integrated == False:
            self.integrate()
        
        colormap = pyplot.cm.hot
        if fig != None:
            pyplot.figure(fig)
            pyplot.clf()
            pyplot.figure(fig,figsize=[12,8])
        else:
            pyplot.figure(fig,figsize=[12,8])
        title_c = 'Col sum\nI_c = %g, Ierr_c = %g, Ibgr_c = %g' % (self.I_c,self.Ierr_c,self.Ibgr_c)
        title_r = 'Row sum\nI_r = %g, Ierr_r = %g, Ibgr_r = %g' % (self.I_r,self.Ierr_r,self.Ibgr_r)
        title_roi = 'I = %g, Ierr = %g, Ibgr = %g' % (self.I,self.Ierr,self.Ibgr)
        if self.bgrimg != None:
            title_roi = title_roi + '\n(background subtracted)'

        # calc full image with an roi box
        (c1,r1,c2,r2) = self.roi
        if self.rotangle != 0.0:
            bild = ndimage.rotate(self.image,self.rotangle)
        else:
            bild = copy.copy(self.image)
        # whats the max inside the roi
        im_max = num.max(bild[r1:r2, c1:c2])
        #
        bild[r1-1:r1,c1:c2] = bild.max()
        bild[r2:r2+1,c1:c2] = bild.max()
        bild[r1:r2,c1-1:c1] = bild.max()
        bild[r1:r2,c2:c2+1] = bild.max()

        ###################################
        #### plot column sum
        pyplot.subplot(221)
        pyplot.title(title_c, fontsize = 12)
        # plot raw sum
        (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='c',nbgr=0)
        rawmax = data.max()
        pyplot.plot(data_idx, data, 'k',label='raw sum')
        # get bgr and data-bgr
        if self.bgrimg != None:
            # here data is automatically bgr subracted
            (data, data_idx, xx) = line_sum(self.clpimg-self.bgrimg,sumflag='c',nbgr=0)
            bgr = self.bgrimg.sum(axis=0)
        else:
            # here data is data and bgr is correct, therefore data = data-bgr
            (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='c',
                                             nbgr=self.rbgr['nbgr'],
                                             width=self.rbgr['width'],
                                             pow=self.rbgr['pow'],
                                             tangent=self.rbgr['tan'])
            data = data-bgr
        # plot bgr and bgr subtracted data
        pyplot.plot(data_idx, bgr, 'r',label='bgr')
        pyplot.plot(data_idx, data, 'b',label='data-bgr')
        pyplot.axis([0, data_idx.max(), 0, rawmax*1.25])
        pyplot.legend(loc=0)

        ####################################
        # plot full image with ROI
        pyplot.subplot(222)
        pyplot.title(self.title, fontsize = 12)
        pyplot.imshow(bild,cmap=colormap,vmax=im_max)
        pyplot.colorbar(orientation='horizontal')

        ####################################
        # plot zoom on image 
        pyplot.subplot(223)
        pyplot.title(title_roi,fontsize = 12)
        if self.bgrimg != None:
            pyplot.imshow(self.clpimg-self.bgrimg, cmap=colormap, aspect='auto')
        else:
            pyplot.imshow(self.clpimg, cmap=colormap, aspect='auto')
            
        ####################################
        # plot row sum
        pyplot.subplot(224)
        pyplot.title(title_r, fontsize = 12)
        # plot raw sum
        (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='r',nbgr=0)
        rawmax = data.max()
        pyplot.plot(data, data_idx, 'k',label='raw sum')
        # get bgr and data-bgr
        if self.bgrimg != None:
            # here data is automatically bgr subracted
            (data, data_idx, xx) = line_sum(self.clpimg-self.bgrimg,sumflag='r',nbgr=0)
            bgr = self.bgrimg.sum(axis=1)
        else:
            # here data is data and bgr is correct, therefore data = data-bgr
            (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='r',
                                             nbgr=self.cbgr['nbgr'],
                                             width=self.cbgr['width'],
                                             pow=self.cbgr['pow'],
                                             tangent=self.cbgr['tan'])
            data = data-bgr
        # plot bgr and bgr subtracted data
        pyplot.plot(bgr, data_idx, 'r',label='bgr')
        pyplot.plot(data, data_idx, 'b',label='data-bgr')
        pyplot.axis([0,rawmax*1.25, data_idx.max(), 0])
        pyplot.xticks(rotation=-45)
        pyplot.legend(loc=0)
        
        
    ############################################################################
    def embed_plot(self,fig):
        """
        make fancy 4-panel plot to embed in wxPython.
        fig is the Figure in which to embed the plots
        
        Notes:
        -----
        Is this method redundant?  We should combine this and plot so 
        we only have one display method
        """
        if self.integrated == False:
            self.integrate()
        
        fig.clear()
        colormap = None
        #if fig != None:
        #    pyplot.figure(fig)
        #    pyplot.clf()
        #    pyplot.figure(fig,figsize=[12,8])
        #else:
        #    pyplot.figure(fig,figsize=[12,8])
        title_c = 'Col sum\nI_c = %g, Ierr_c = %g, Ibgr_c = %g' % (self.I_c,self.Ierr_c,self.Ibgr_c)
        title_r = 'Row sum\nI_r = %g, Ierr_r = %g, Ibgr_r = %g' % (self.I_r,self.Ierr_r,self.Ibgr_r)
        title_roi = 'I = %g, Ierr = %g, Ibgr = %g' % (self.I,self.Ierr,self.Ibgr)
        if self.bgrimg != None:
            title_roi = title_roi + '\n(background subtracted)'

        # calc full image with an roi box
        (c1,r1,c2,r2) = self.roi
        if self.rotangle != 0.0:
            bild = ndimage.rotate(self.image,self.rotangle)
        else:
            bild = copy.copy(self.image)
        # whats the max inside the roi
        if self.im_max == -1:
            im_max = num.max(bild[r1:r2, c1:c2])
        else:
            im_max = self.im_max
        #
        bildMax = bild.max()
        bild[r1-2:r1,c1:c2] = bildMax
        bild[r2:r2+2,c1:c2] = bildMax
        bild[r1:r2,c1-2:c1] = bildMax
        bild[r1:r2,c2:c2+2] = bildMax

        ###################################
        #### plot column sum
        self.subplot1 = fig.add_subplot(221)
        self.subplot1.set_title(title_c, fontsize = 12)
        # plot raw sum
        (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='c',nbgr=0)
        rawmax = data.max()
        (sp1Data1, sp1Data_Idx1, sp1Bgr1) = (data, data_idx, bgr)
        sp1Rawmax = rawmax
        self.subplot1.plot(data_idx, data, 'k',label='raw sum')
        # get bgr and data-bgr
        if self.bgrimg != None:
            # here data is automatically bgr subracted
            (data, data_idx, xx) = line_sum(self.clpimg-self.bgrimg,sumflag='c',nbgr=0)
            bgr = self.bgrimg.sum(axis=0)
        else:
            # here data is data and bgr is correct, therefore data = data-bgr
            (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='c',
                                             nbgr=self.rbgr['nbgr'],
                                             width=self.rbgr['width'],
                                             pow=self.rbgr['pow'],
                                             tangent=self.rbgr['tan'])
            data = data-bgr
        (sp1Data2, sp1Data_Idx2, sp1Bgr2) = (data, data_idx, bgr)
        # plot bgr and bgr subtracted data
        self.subplot1.plot(data_idx, bgr, 'r',label='bgr')
        self.subplot1.plot(data_idx, data, 'b',label='data-bgr')
        self.subplot1.axis([0, data_idx.max(), 0, rawmax*1.25])
        self.subplot1.legend(loc=0)

        ####################################
        # plot full image with ROI
        self.subplot2 = fig.add_subplot(222, title = self.title)
        self.subplot2.set_title(self.title, fontsize = 12)
        forColorbar = self.subplot2.imshow(bild,cmap=colormap,vmax=im_max)
        fig.colorbar(forColorbar, ax=self.subplot2, orientation='horizontal')

        ####################################
        # plot zoom on image 
        self.subplot3 = fig.add_subplot(223, title = title_roi)
        self.subplot3.set_title(title_roi,fontsize = 12)
        if self.bgrimg != None:
            self.subplot3.imshow(self.clpimg-self.bgrimg, cmap=colormap, aspect='auto')
        else:
            self.subplot3.imshow(self.clpimg, cmap=colormap, aspect='auto')
            
        ####################################
        # plot row sum
        self.subplot4 = fig.add_subplot(224, title = title_r)
        self.subplot4.set_title(title_r, fontsize = 12)
        # plot raw sum
        (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='r',nbgr=0)
        rawmax = data.max()
        (sp4Data1, sp4Data_Idx1, sp4Bgr1) = (data, data_idx, bgr)
        sp4Rawmax = rawmax
        self.subplot4.plot(data, data_idx, 'k',label='raw sum')
        # get bgr and data-bgr
        if self.bgrimg != None:
            # here data is automatically bgr subracted
            (data, data_idx, xx) = line_sum(self.clpimg-self.bgrimg,sumflag='r',nbgr=0)
            bgr = self.bgrimg.sum(axis=1)
        else:
            # here data is data and bgr is correct, therefore data = data-bgr
            (data, data_idx, bgr) = line_sum(self.clpimg,sumflag='r',
                                             nbgr=self.cbgr['nbgr'],
                                             width=self.cbgr['width'],
                                             pow=self.cbgr['pow'],
                                             tangent=self.cbgr['tan'])
            data = data-bgr
        (sp4Data2, sp4Data_Idx2, sp4Bgr2) = (data, data_idx, bgr)
        # plot bgr and bgr subtracted data
        self.subplot4.plot(bgr, data_idx, 'r',label='bgr')
        self.subplot4.plot(data, data_idx, 'b',label='data-bgr')
        self.subplot4.axis([0,rawmax*1.25, data_idx.max(), 0])
        #self.subplot4.xticks(rotation=-45)
        self.subplot4.legend(loc=0)
        
        im_max = num.max(bild[r1:r2, c1:c2])
        
        return (im_max,
                colormap,
                self.subplot2)

##############################################################################
class ImageScan:
    """
    Class to hold a collection of images associated with a scan
    ie one image per scan point

    Should create an _archive() and _unarchive() methods...   
    """
    def __init__(self,image=[],rois=None,rotangle=None,bgr_params=None,
                 archive=None):
        """
        Initialize

        Parameters:
        -----------
        * image is a list of images (2d arrays)
        * rois is a list of roi values
        * rotangle is a list of rotangles
        * bgr_params is a list of backgorund params
        * archive is a dictionary with archive info
          archive['file'] = file name for image archive
          archive['path'] = path for image archive
          archive['setname'] = set name for image archive
          archive['descr'] = description of data for image archive
        """
        if type(image) != types.ListType: image = [image]
        if archive != None:
            file    = archive.get('file','images.h5')
            path    = archive.get('path')
            setname = archive.get('setname','S1')
            descr   = archive.get('descr','Scan Data Archive')
            self.image = _ImageList(image,file=file,path=path,
                                    setname=setname,descr=descr)
        else:
            self.image = image
        self.rois     = None
        self.rotangle = None
        self.bgrpar   = None
        self.im_max   = []
        self.peaks    = {}
        self._is_integrated = False
        self._init_image()
        #
        npts = len(self.image)
        # update roi
        if rois!=None:
            if len(rois) == 4:
                if type(rois[0]) == types.ListType and npts == 4:
                    for j in range(npts):
                        self.rois[j] = rois[j]
                else:
                    for j in range(npts):
                        self.rois[j] = copy.copy(rois)
            elif len(rois) == len(npts):
                for j in range(npts):
                    self.rois[j] = rois[j]
        # update rot angles
        if rotangle!=None:
            if len(rotangle) == 1:
                for j in range(npts):
                    self.rotangle[j] = rotangle
            elif len(rotangle) == npts:
                for j in range(npts):
                    self.rotangle[j] = rotangle[j]
        # update bgr
        if bgr_params!=None:
            if type(bgr_params) == types.DictType:
                for j in range(npts):
                    self.bgrpar[j] = copy.copy(bgr_params)
            elif len(bgr_params) == npts:
                for j in range(npts):
                    self.bgrpar[j] = bgr_params[j]

    ################################################################
    def __repr__(self,):
        lout = 'Number of images = %i' % (len(self.image))
        return lout

    ################################################################
    def _init_image(self):
        npts = len(self.image)
        # init arrays
        if npts == 0:
            self.rois     = []
            self.rotangle = []
            self.bgrpar   = []
            self.peaks    = {}
        if self.rois == None:
            self.rois = []
        if self.rotangle == None:
            self.rotangle = []
        if self.bgrpar == None:
            self.bgrpar = []
        # init rois
        if len(self.rois) != npts:
            self.rois = []
            for j in range(npts):
                self.rois.append([0,0,self.image[j].shape[1],
                                  self.image[j].shape[0]])
        # init rotangle
        if len(self.rotangle) != npts:
            self.rotangle = []
            for j in range(npts):
                self.rotangle.append(0.0)
        # init im_max
        if len(self.im_max) != npts:
            self.im_max = []
            for j in range(npts):
                self.im_max.append(-1)
        # init bgr
        if len(self.bgrpar) != npts:
            self.bgrpar = []
            for j in range(npts):
                self.bgrpar.append(copy.copy(IMG_BGR_PARAMS))
        # should we init all these or set based on an integrate flag?
        self.peaks  = {}
        self.peaks['I']      = num.zeros(npts,dtype=float)
        self.peaks['Ierr']   = num.zeros(npts,dtype=float)
        self.peaks['Ibgr']   = num.zeros(npts,dtype=float)
        #
        self.peaks['I_c']    = num.zeros(npts,dtype=float)
        self.peaks['Ierr_c'] = num.zeros(npts,dtype=float)
        self.peaks['Ibgr_c'] = num.zeros(npts,dtype=float)
        #
        self.peaks['I_r']    = num.zeros(npts,dtype=float)
        self.peaks['Ierr_r'] = num.zeros(npts,dtype=float)
        self.peaks['Ibgr_r'] = num.zeros(npts,dtype=float)
        #
        self._is_integrated = False

    ################################################################
    def _is_init(self):
        if len(self.peaks) > 0:
            if len(self.peaks['I']) == len(self.image):
                init = True
            else: init = False
        else:
            init = False
        return init

    ################################################################
    def plot(self,idx=0,fig=None,figtitle='',cmap=None,verbose=False):
        """
        Make a plot of image
        """
        image_plot(self.image[idx],fig=fig,figtitle=figtitle,
                   cmap=cmap,verbose=verbose,
                   im_max=self.im_max[idx],
                   rotangle=self.rotangle[idx],
                   roi=self.rois[idx])
               
    ################################################################
    def integrate(self,idx=[],roi=None,rotangle=None,bgr_params=None,
                  bad_points=[],plot=False,fig=None):
        """
        integrate images

        Parameters:
        -----------
        * idx are the indicies to integrate
        * other parameters are same as on __init__ and can
          be updated here or pass as None to use existing values.
        """
        # make sure arrays exist:
        if self._is_init()==False:
            self._init_image()
        # idx of images to integrate
        if len(idx) == 0:
            idx = num.arange(len(self.image))
        # update roi
        if roi!=None:
            if len(roi) == 4:
                if type(roi[0]) == types.ListType:
                    for j in idx:
                        self.rois[j] = roi[j]
                else:
                    for j in idx:
                        self.rois[j] = roi
            elif len(roi) == len(idx):
                for j in idx:
                    self.rois[j] = roi[j]
        # update rot angles
        if rotangle!=None:
            if type(rotangle) == types.FloatType:
                for j in idx:
                    self.rotangle[j] = rotangle
            elif len(rotangle) == len(idx):
                for j in idx:
                    self.rotangle[j] = rotangle[j]
        # update bgr
        if bgr_params!=None:
            if type(bgr_params) == types.DictType:
                for j in idx:
                    self.bgrpar[j] = copy.copy(bgr_params)
            elif len(bgr_params) == len(idx):
                for j in idx:
                    self.bgrpar[j] = copy.copy(bgr_params[j])
        # do integrations
        for j in idx:
            if j not in bad_points:
                self._integrate(idx=j,plot=plot,fig=fig)
            else:
                self.peaks['I'][j]      = 0.
                self.peaks['Ierr'][j]   = 0.
                self.peaks['Ibgr'][j]   = 0.
                #
                self.peaks['I_c'][j]    = 0.
                self.peaks['Ierr_c'][j] = 0.
                self.peaks['Ibgr_c'][j] = 0.
                #
                self.peaks['I_r'][j]    = 0.
                self.peaks['Ierr_r'][j] = 0.
                self.peaks['Ibgr_r'][j] = 0.

        self._is_integrated = True
    
    ################################################################
    def _integrate(self,idx=0,plot=True,fig=None):
        """
        integrate an image
        """
        if idx < 0 or idx > len(self.image): return None
        #
        figtitle = "Scan Point = %i" % (idx)
        roi        = self.rois[idx]
        rotangle   = self.rotangle[idx]
        bgr_params = self.bgrpar[idx]

        img_ana = ImageAna(self.image[idx],roi=roi,rotangle=rotangle,
                           plot=plot,fig=fig,figtitle=figtitle,
                           **bgr_params)

        # results into image_peaks dictionary
        self.peaks['I'][idx]      = img_ana.I
        self.peaks['Ierr'][idx]   = img_ana.Ierr
        self.peaks['Ibgr'][idx]   = img_ana.Ibgr
        #
        self.peaks['I_c'][idx]    = img_ana.I_c
        self.peaks['Ierr_c'][idx] = img_ana.Ierr_c
        self.peaks['Ibgr_c'][idx] = img_ana.Ibgr_c
        #
        self.peaks['I_r'][idx]    = img_ana.I_r
        self.peaks['Ierr_r'][idx] = img_ana.Ierr_r
        self.peaks['Ibgr_r'][idx] = img_ana.Ibgr_r

################################################################
class _ImageList:
    """
    Keep images in a hdf file using pytables
    
    Note an alternative is to use:
       num.savez(fname,image)
       im = num.read(fname)
    """
    ################################################################
    def __init__(self,images,file='images.h5',path=None,
                 setname='S000',descr='Scan data images'):
        self.path = path
        self.file = file
        self.setname = setname
        #
        if images != None:
            self.nimages = len(images)
            try:
                self._write_image_tables(images,setname,descr)
            except:
                self._cleanup()
                print "Unable to write images:"
                print "   Setname %s, hdf file %s" % (file,setname) 
    
    ################################################################
    def _cleanup(self):
        try:
            import tables
            tables.file.close_open_files()
        except:
            pass

    ################################################################
    def __len__(self):
        return self.nimages

    ################################################################
    def __getitem__(self,arg):
        """
        Get item.  
        """
        images = self._read_image_tables()
        #return copy.copy(images[arg])
        return images[arg]

    ################################################################
    def __setitem__(self,arg):
        """
        Set item.  
        """
        print "Cannot set item"
        return

    ################################################################
    def _make_fname(self):
        if self.path != None:
            fname = os.path.join(self.path,self.file)
            fname = os.path.abspath(fname)
        else:
            fname = os.path.abspath(self.file)
        return fname

    ################################################################
    def _write_image_tables(self,images,setname,descr):
        """
        Write images to file
        """
        import tables
        images = num.array(images)
        fname  = self._make_fname()
        if os.path.exists(fname):
            h    = tables.openFile(fname,mode="a")
            if not hasattr(h.root,'image_data'):
                h.createGroup(h.root,'image_data',"Image Data")
        else:
            h    = tables.openFile(fname,mode="w",title="Scan Data Archive")
            root = h.createGroup(h.root, "image_data", "Image Data")
        # if setname == None:
        #   look under '/images for 'SXXX'
        # find the highest one and
        # auto generate set name as next in the sequence 
        if hasattr(h.root.image_data,setname):
            print "Warning: Image Archive File '%s'" % fname
            print "-->Setname '%s' already exists, data is not overwritten\n" % setname
        else:
            h.createGroup('/image_data',setname,"Image Data")
            grp = '/image_data/' + setname
            h.createArray(grp,'images',images,descr)
        h.close()

    ################################################################
    def _read_image_tables(self,):
        import tables
        fname = self._make_fname()
        if not os.path.exists(fname):
            print "Archive file not found:", fname
            return None
        grp = '/image_data/' + self.setname
        try:
            h  = tables.openFile(fname,mode="r")
            #n  = h.getNode('/images',self.setname)
            n  = h.getNode(grp,'images')
            im = n.read()
            h.close()
            return im
        except:
            self._cleanup()
            print "Error reading image tables: %s" % grp
            return None

################################################################################
################################################################################
if __name__ == '__main__':
    import sys
    try:
        fname = sys.argv[1]
    except:
        print 'read_pilatus  tiff file'
        sys.exit()
    a = read(fname)
    image_show(a)
    
