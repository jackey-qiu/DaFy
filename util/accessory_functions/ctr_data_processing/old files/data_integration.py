import numpy as np
from numpy.matlib import repmat
from numpy.linalg import pinv
from matplotlib import pyplot
from scipy import misc
import fnmatch
import os
import matplotlib.patches as patches
import ctr_data
import ntpath
import pickle,copy
from scipy.optimize import curve_fit
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import ndimage


##The background subtraction algoritem is developped by Vincent Mazet with the copyright notice as below###
##The code was originally written by Vincent Mazet based on MATLAB. Canrong Qiu (me) translated the scripts to Python language##
##Correction factors are calculated using TDL modules, developped and maintained by GSECARS 13IDC beamline at APS (Peter Eng and Joanne Stubbs are responsible persons)##

'''
Copyright (c) 2012, Vincent Mazet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

"""
# BACKCOR   Background estimation by minimizing a non-quadratic cost function.
#
#   [EST,COEFS,IT] = BACKCOR(N,Y,ord_cusER,THRESHOLD,FUNCTION) computes and estimation EST
#   of the background (aka. baseline) in a spectroscopic signal Y with wavelength N.
#   The background is estimated by a polynomial with ord_cuser ord_cusER using a cost-function
#   FUNCTION with parameter THRESHOLD. FUNCTION can have the four following values:
#       'sh'  - symmetric Huber function :  f(x) = { x^2  if abs(x) < THRESHOLD,
#                                                  { 2*THRESHOLD*abs(x)-THRESHOLD^2  otherwise.
#       'ah'  - asymmetric Huber function :  f(x) = { x^2  if x < THRESHOLD,
#                                                   { 2*THRESHOLD*x-THRESHOLD^2  otherwise.
#       'stq' - symmetric truncated quadratic :  f(x) = { x^2  if abs(x) < THRESHOLD,
#                                                       { THRESHOLD^2  otherwise.
#       'atq' - asymmetric truncated quadratic :  f(x) = { x^2  if x < THRESHOLD,
#                                                        { THRESHOLD^2  otherwise.
#   COEFS returns the ord_cusER+1 vector of the estimated polynomial coefficients.
#   IT returns the number of iterations.
#
#   [EST,COEFS,IT] = BACKCOR(N,Y) does the same, but run a graphical user interface
#   to help setting ord_cusER, THRESHOLD and FCT.
#
# For more informations, see:
# - V. Mazet, C. Carteret, D. Brie, J. Idier, B. Humbert. Chemom. Intell. Lab. Syst. 76 (2), 2005.
# - V. Mazet, D. Brie, J. Idier. Proceedings of EUSIPCO, pp. 305-308, 2004.
# - V. Mazet. PhD Thesis, University Henri Poincare Nancy 1, 2005.
#
# 22-June-2004, Revised 19-June-2006, Revised 30-April-2010,
# Revised 12-November-2012 (thanks E.H.M. Ferreira!)
# Comments and questions to: vincent.mazet@unistra.fr.

# Check arguments
if nargin < 2, error('backcor:NotEnoughInputArguments','Not enough input arguments'); end;
if nargin < 5, [z,a,it,ord_cus,s,fct] = backcorgui(n,y); return; end; # delete this line if you do not need GUI
if ~isequal(fct,'sh') && ~isequal(fct,'ah') && ~isequal(fct,'stq') && ~isequal(fct,'atq'),
    error('backcor:UnknownFunction','Unknown function.');
end;
"""

#global variables
PLOT_LIVE=True
BRAGG_PEAKS=range(0,19)#L values of Bragg peaks
BRAGG_PEAK_CUTOFF=0.07#excluding range on L close to a Bragg peak (L_Bragg+-this value will be excluded for plotting)
BRAGG_PEAK_CENTER_CHECK=range(2,10,2)
HK_CENTER_CHECK=[0,0]
###########integration setup here##############
INTEG_PARS={}
INTEG_PARS['cutoff_scale']=0.001
INTEG_PARS['use_scale']=False#Set this to False always
INTEG_PARS['center_pix']=[55,254]#[53,153] or [91,247] fro ESRF#Center pixel index (know Python is column basis, so you need to swab the order of what you see at pixe image)
INTEG_PARS['r_width']=15#15#integration window in row direction (total row length is twice that value)
INTEG_PARS['c_width']=50#50#integration window in column direction (total column length is twice that value)
INTEG_PARS['integration_direction']='y'#'y'#integration direction (x-->row direction, y-->column direction), you should use 'y' for horizontal mode (Bragg peak move left to right), and 'x' for vertical mode (Bragg peak move up and down)
INTEG_PARS['ord_cus_s']=[1,2,4,6,8] #A list of integration power to be tested for finding the best background subtraction. Flat if the value is 0. More wavy higher value
INTEG_PARS['ss']=[0.01,0.02,0.05,0.06]#a list of thereshold factors used in cost function (0: all signals are through, means no noise background;1:means all backround, no peak signal. You should choose a value between 0 and 1)
INTEG_PARS['fct']='ah'#Type of cost function ('sh','ah','stq' or 'atq')

################################################
#############spec file info here################
beamline='APS'
if beamline=='APS':
    GENERAL_LABELS={'H':'H','K':'K','L':'L','E':'Energy'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    CORRECTION_LABELS={'time':'Seconds','norm':'io','transmision':'transm'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    ANGLE_LABELS={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    ANGLE_LABELS_ESCAN={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    #ANGLE_LABELS=ANGLE_LABELS_ESCAN
    #G label positions (n_azt: azimuthal reference vector positions @3rd to 6th numbers counting from left to right at G0 line)
    #so are the other symbols: cell (lattice cell info), or0 (first orientation matrix), or1 (second orientation matrix), lambda (x ray wavelength)
    G_LABELS={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[31]],'lambda':['G4',range(3,4)]}
    IMG_EXTENTION='.tif'#image extention (.tif or .tiff)
    CORR_PARAMS={'scale':1,'geom':'psic','beam_slits':{'horz':0.06,'vert': 1},'det_slits':None,'sample':{'dia':10,'polygon':[],'angles':[]}}#slits are in mm
    INTEG_PARS['integration_direction']='y'
    INTEG_PARS['r_width']=18
    INTEG_PARS['c_width']=40
elif beamline=='ESRF':
    GENERAL_LABELS={'H':'H','K':'K','L':'L','E':'energy'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    CORRECTION_LABELS={'time':'sec','norm':'ic0','transmision':'transm'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    ANGLE_LABELS={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    ANGLE_LABELS_ESCAN={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
    #G label positions (n_azt: azimuthal reference vector positions @3rd to 6th numbers counting from left to right at G0 line)
    #so are the other symbols: cell (lattice cell info), or0 (first orientation matrix), or1 (second orientation matrix), lambda (x ray wavelength)
    G_LABELS={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[30]],'lambda':['G4',range(3,4)]}
    IMG_EXTENTION='.tiff'#image extention (.tif or .tiff)
    CORR_PARAMS={'scale':1000000,'geom':'psic','beam_slits':{'horz':0.06,'vert': 1},'det_slits':None,'sample':{'dia':10,'polygon':[],'angles':[]}}#slits are in mm
    INTEG_PARS['integration_direction']='x'
    INTEG_PARS['r_width']=35
    INTEG_PARS['c_width']=12
###############################################################################################################################################################
def image_process_temp(spec_path='/Users/cqiu/data/APS/Dec_2017_APS/images/s2_zr_100mM_KCl_1/S015/s2_zr_100mM_KCl_1_S015_00203.tif'):
    im=misc.imread(spec_path)
    mask=im>im.mean()*8
    label_im,nb_labels=ndimage.label(mask)
    print nb_labels
    plt.figure(figsize=(4,9))

    plt.subplot(311)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.subplot(313)
    plt.imshow(label_im, cmap=plt.cm.spectral)
    plt.axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
    plt.show()

    # Find the largest connected component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    print sizes
    mask_size = sizes < 2
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)

    # Now that we have only one connected component, extract it's bounding box
    slice_x, slice_y = ndimage.find_objects(label_im==2)[0]
    print slice_x,slice_y
    roi = im[slice_x, slice_y]

    plt.figure(figsize=(4, 2))
    plt.axes([0, 0, 1, 1])
    plt.imshow(roi)
    plt.axis('off')

    plt.show()

class data_integration:
    def __init__(self,spec_path='M:\\fwog\\members\\qiu05\\Dec_2017_APS\\fer',spec_name='rcut_cmp_zn_7mm_1.spec',scan_number=range(12,34),\
                corr_params=CORR_PARAMS,
                integ_pars=INTEG_PARS,\
                general_labels=GENERAL_LABELS,\
                correction_labels=CORRECTION_LABELS,\
                angle_labels=ANGLE_LABELS,\
                angle_labels_escan=ANGLE_LABELS_ESCAN,\
                G_labels=G_LABELS,\
                img_extention=IMG_EXTENTION,\
                find_center_pix=False):
        '''
        spec_path:full path to the directory of the spec file
        spec_name:full name of the spec file
        scan_number=
            []:do nothing but initilize the instance (do this when you want to reload a saved dump data_info file)
            None: do all rodscan and Escan throughout the spec file
            [13,14,15,16]:a list of integers specify the scan numbers you want to do
        '''
        self.spec_path=spec_path
        self.spec_name=spec_name
        self.scan_number=scan_number
        self.data_info={}
        self.corr_params=corr_params
        self.integ_pars=integ_pars
        self.general_labels=general_labels
        self.correction_labels=correction_labels
        self.angle_labels=angle_labels
        self.angle_labels_escan=angle_labels_escan
        self.G_labels=G_labels
        self.img_extention=img_extention
        self.combine_spec_image_info()
        if scan_number!=[] and find_center_pix:
            self.find_center_pix()
        self.batch_image_integration()
        #if scan_number!=[]:
        #    self.remove_spikes_from_data_info()

    def help(self):
        print "##########################useful help info########################"
        print 'All data info are saved in self.data_info'
        print 'you should navigate to the associated functions to get the formate of arguments in the function'
        print '1. reload_pickle_dump_file() to reload a saved pickle file'
        print '2. plot_results() to plot all integrated data'
        print '3. report_info() to report scan information'
        print '4. remove_spikes_from_data_info() remove permanantly sharp spikes'
        print '5. q_correction(scan_number) to do q correction for the scan'
        print "6. show_scan_images(scan_number=13) to show the integration of one scan"
        print '7. integrate_images_twick_mode() to launch a twick mode for data integration'
        print '8. dump_data_info() to pickle dump data_info'
        print '9. save_data() to save and formate data as ascill file fomate'
        print '10. append_scan_info(scan_number=[]) to append scans to current data_info'

    def reload_pickle_dump_file(self,dump_file='./dump_file.dump'):
        if os.path.basename(dump_file)==dump_file:
            dump_file=os.path.join(self.spec_path,dump_file)#if not provide the full path, take the spec path as the parent directory by default
        else:
            pass
        new_data_info=pickle.load(open(dump_file,"rb"))
        self.data_info=new_data_info
        self.update_spec_path_in_data_info()
        #self.remove_spikes_from_data_info()
        return None

    def set_spec_info(self,path=None,name=None,scan_number=None):
        if path!=None:
            self.spec_path=path
        if name!=None:
            self.spec_name=name
        if scan_number!=None:
            self.scan_number=scan_number
        if [path,name,scan_number]!=[None,None,None]:
            self.combine_spec_image_info()
            self.batch_image_integration()
        return None

    def append_scan_info(self,scan_number=None):
        original_scan_number=self.scan_number
        if scan_number!=None:
            self.scan_number=scan_number
        if scan_number!=None:
            self.combine_spec_image_info(append_scan=True)
            self.batch_image_integration(append_scan=True)
        self.scan_number=original_scan_number+scan_number
        #self.remove_spikes_from_data_info()
        return None

    def update_spec_path_in_data_info(self):
        self.data_info['spec_path']=os.path.join(self.spec_path,self.spec_name)
        for j in range(len(self.data_info["scan_number"])):
            each=self.data_info["images_path"][j]
            for i in range(len(each)):
                this_img_original=each[i]
                scan_name=ntpath.basename(this_img_original).rsplit("_")[-2]
                this_img_new=os.path.join(self.spec_path,"images")
                this_img_new=os.path.join(this_img_new,self.spec_name.replace(".spec",""))
                this_img_new=os.path.join(this_img_new,scan_name)
                this_img_new=os.path.join(this_img_new,ntpath.basename(this_img_original))
                self.data_info["images_path"][j][i]=this_img_new
        return None

    def set_corr_params(self,corr_params):
        self.corr_params=corr_params
        self.combine_spec_image_info()
        self.batch_image_integration()
        return None

    def set_integ_pars(self,integ_pars={'ord_cus':4,'s':0.1,'fct':'sh'}):
        self.integ_pars=integ_pars
        self.combine_spec_image_info()
        self.batch_image_integration()
        return None

    #engine function to subtraction background
    def backcor(self,n,y,ord_cus,s,fct):
        # Rescaling
        N = len(n)
        index = np.argsort(n)
        n=np.array([n[i] for i in index])
        y=np.array([y[i] for i in index])
        maxy = max(y)
        dely = (maxy-min(y))/2.
        n = 2. * (n-n[N-1]) / float(n[N-1]-n[0]) + 1.
        n=n[:,np.newaxis]
        y = (y-maxy)/dely + 1

        # Vandermonde matrix
        p = np.array(range(ord_cus+1))[np.newaxis,:]
        T = repmat(n,1,ord_cus+1) ** repmat(p,N,1)
        Tinv = pinv(np.transpose(T).dot(T)).dot(np.transpose(T))

        # Initialisation (least-squares estimation)
        a = Tinv.dot(y)
        z = T.dot(a)

        # Other variables
        alpha = 0.99 * 1/2     # Scale parameter alpha
        it = 0                 # Iteration number
        zp = np.ones((N,1))         # Previous estimation

        # LEGEND
        while np.sum((z-zp)**2)/np.sum(zp**2) > 1e-10:

            it = it + 1        # Iteration number
            zp = z             # Previous estimation
            res = y - z        # Residual

            # Estimate d
            if fct=='sh':
                d = (res*(2*alpha-1)) * (abs(res)<s) + (-alpha*2*s-res) * (res<=-s) + (alpha*2*s-res) * (res>=s)
            elif fct=='ah':
                d = (res*(2*alpha-1)) * (res<s) + (alpha*2*s-res) * (res>=s)
            elif fct=='stq':
                d = (res*(2*alpha-1)) * (abs(res)<s) - res * (abs(res)>=s)
            elif fct=='atq':
                d = (res*(2*alpha-1)) * (res<s) - res * (res>=s)
            else:
                pass

            # Estimate z
            a = Tinv.dot(y+d)   # Polynomial coefficients a
            z = T.dot(a)            # Polynomial

        z=np.array([(z[list(index).index(i)]-1)*dely+maxy for i in range(len(index))])

        return z,a,it,ord_cus,s,fct

    def _get_col_from_file(self,lines,start_row,end_row,col,type=float):
        numbers=[]
        for i in range(start_row,end_row):
            numbers.append(type(lines[i].rstrip().rsplit()[col]))
        return numbers

    def find_center_pix(self,bragg_peaks=BRAGG_PEAK_CENTER_CHECK,hk=HK_CENTER_CHECK):
        print 'Locating center pix now...'
        images=[]
        center_pixs=[]
        def _find_images_index(L_all_list,L_bragg_list):
            index_container=[]
            for L_bragg in L_bragg_list:
                if max(L_all_list)>L_bragg:
                    offset=np.abs(np.array(L_all_list)-L_bragg)
                    index_temp_1=np.where(offset==np.min(offset))[0][0]
                    #print offset
                    #print index_temp_1
                    if L_all_list[index_temp_1]>L_bragg:
                        index_temp_2=index_temp_1-1
                    else:
                        index_temp_2=index_temp_1+1
                    index_container.append([index_temp_1,index_temp_2])
                else:
                    pass
            return index_container
        for i in range(len(self.data_info['L'])):
            if [self.data_info['H'][i][0],self.data_info['K'][i][0]]==list(hk):
                index_container=_find_images_index(self.data_info['L'][i],bragg_peaks)
                if index_container!=[]:
                    for each in index_container:
                        images.append([self.data_info['images_path'][i][each[0]],self.data_info['images_path'][i][each[1]]])
                else:
                    pass
            else:
                pass
        for image_pair in images:
            print 'One trail is going ...'
            img1=misc.imread(image_pair[0])
            img2=misc.imread(image_pair[1])
            imag1_index=[np.where(img1==np.max(img1))[0][0],np.where(img1==np.max(img1))[1][0]]
            imag2_index=[np.where(img2==np.max(img2))[0][0],np.where(img2==np.max(img2))[1][0]]
            ave_temp=[int((imag1_index[0]+imag2_index[0])/2.),int((imag1_index[1]+imag2_index[1])/2.)]
            center_pixs.append(ave_temp)
        average_center=np.average(center_pixs,axis=0)
        #print center_pixs
        #print average_center
        print 'Center pix is located at: ','[',int(average_center[0]),int(average_center[1]),']'
        self.integ_pars['center_pix']=[int(average_center[0]),int(average_center[1])]
        return None

    def q_correction(self,scan_number,scale=1,delta=0,c_off=0):
        R_tth=1100#2theta circle radius in mm
        scan_index=self.data_info['scan_number'].index(scan_number)
        L=np.array(self.data_info['L'][scan_index])
        data = np.array(self.data_info['I'][scan_index])
        eb = np.array(self.data_info['Ierr'][scan_index])
        cell=self.data_info['cell'][scan_index]
        clat=cell[2]*np.sin(np.deg2rad(cell[4]))
        lam =self.data_info['lambda'][scan_index][0]
        tth=2*np.arcsin(L*lam/2./clat)

        L_Bragg_container=[]
        scale_container=[]

        for i in range(10000):
            input_items=raw_input("Type in scale factor (around 1)\nIf you want to save current factor, type 'SL'(eg S3)\nIf you are done, type q: ")
            if input_items.startswith('S'):
                L_Bragg_container.append(int(input_items[1:]))
                scale_container.append(scale)
                print 'L and Scale are saved!'
            elif input_items!='q' and input_items!='':
                scale=float(input_items)
                offtth=2/R_tth*np.cos(tth/2)*(delta+c_off)-c_off/R_tth
                Q=4*np.pi/lam*np.sin((tth-offtth)/2)
                clat_corrt=clat/scale
                normcalc = (np.sin(Q*clat_corrt/4))**2
                normdata = data*normcalc
                y_max,y_min=normdata.max(),normdata.min()
                normeb = eb*normcalc
                LL=Q*clat_corrt/2./np.pi
                pyplot.close()
                fig,ax=pyplot.subplots(1,figsize=(16,4))
                ax.set_yscale('log')
                ax.scatter(LL,normdata,marker='s',s=5)
                ax.errorbar(LL,normdata,yerr=normeb,fmt=None)
                for each_l_bragg in range(2,17,2):
                    ax.plot([each_l_bragg,each_l_bragg],[y_min,y_max],'r:')
            elif input_items=='q':
                break
            else:
                offtth=2/R_tth*np.cos(tth/2)*(delta+c_off)-c_off/R_tth
                Q=4*np.pi/lam*np.sin((tth-offtth)/2)
                clat_corrt=clat/scale
                normcalc = (np.sin(Q*clat_corrt/4))**2
                normdata = data*normcalc
                normdata = data*normcalc
                y_max,y_min=normdata.max(),normdata.min()
                normeb = eb*normcalc
                LL=Q*clat_corrt/2./np.pi
                pyplot.close()
                fig,ax=pyplot.subplots(1,figsize=(16,4))
                ax.set_yscale('log')
                ax.scatter(LL,normdata,marker='s',s=5)
                ax.errorbar(LL,normdata,yerr=normeb,fmt=None)
                for each_l_bragg in range(2,17,2):
                    ax.plot([each_l_bragg,each_l_bragg],[y_min,y_max],'r:')
        delta,c_off,scale=self.fit_q_correction(lam,R_tth,L_Bragg_container,scale_container)
        offtth=2/R_tth*np.cos(tth/2)*(delta+c_off)-c_off/R_tth
        Q=4*np.pi/lam*np.sin((tth-offtth)/2)
        clat_corrt=clat/scale
        normcalc = (np.sin(Q*clat_corrt/4))**2
        normdata = data*normcalc
        y_max,y_min=normdata.max(),normdata.min()
        normeb = eb*normcalc
        LL=Q*clat_corrt/2./np.pi
        pyplot.close()
        fig,ax=pyplot.subplots(1,figsize=(16,4))
        ax.set_yscale('log')
        ax.scatter(LL,normdata,marker='s',s=5)
        ax.errorbar(LL,normdata,yerr=normeb,fmt=None)
        for each_l_bragg in range(2,17,2):
            ax.plot([each_l_bragg,each_l_bragg],[y_min,y_max],'r:')
        input_items=raw_input("Are you happy with the q correction results (y or n):")
        if input_items=='y':
            print 'Update c lattice...'
            self.data_info['cell'][scan_index][2]=clat_corrt/np.sin(np.deg2rad(cell[4]))#update c lattice parameters
            print 'Update L column...'
            self.data_info['L'][scan_index]=Q*clat_corrt/(2*np.pi)
            print 'q correction is completed!'
        return delta,c_off,scale

    def fit_q_correction(self,lam,R_tth,L_container,scale_container,clat=19.9490):
        tth = 2*np.arcsin(np.array(L_container)*lam/2/clat)
        data=4*np.pi/lam*np.sin( tth/2 )/np.array(scale_container)
        def _cal_q(tth,delta,c,s):
            del_tth = 2/R_tth*np.cos(tth/2)*(delta+c) - c/R_tth
            return 4*np.pi/lam*np.sin((tth-del_tth)/2)/s
        popt, pcov = curve_fit(_cal_q, tth, data, bounds=([-0.3,-0.3,0.98], [0.3, 0.3, 1.1]))
        print 'fit is completed!'
        print 'delta,c_off,scale=',popt
        return popt

    #extract info from spec file
    def sort_spec_file(self,spec_path='.',spec_name='mica-zr_s2_longt_1.spec',scan_number=[16,17,19],\
                    general_labels={'H':'H','K':'K','L':'L','E':'Energy'},correction_labels={'time':'Seconds','norm':'io','transmision':'tra'},\
                    angle_labels={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'},\
                    angle_labels_escan={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'},\
                    G_labels={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[31]],'lambda':['G4',range(3,4)]}):
        matches = []
        data_info,col_label={},{}
        data_info['scan_type']=[]
        data_info['scan_number']=scan_number
        data_info['row_number_range']=[]
        data_info['spec_path']=os.path.join(spec_path,spec_name)

        for key in general_labels.keys():
            data_info[key]=[]

        for key in correction_labels.keys():
            data_info[key]=[]

        for key in angle_labels.keys():
            data_info[key]=[]

        for key in G_labels.keys():
            data_info[key]=[]

        f_spec=open(os.path.join(spec_path,spec_name))
        spec_lines=f_spec.readlines()
        scan_rows=[]
        data_rows=[]
        G0_rows=[]
        G1_rows=[]
        G3_rows=[]
        G4_rows=[]
        for i in range(len(spec_lines)):
            if spec_lines[i].startswith("#S"):
                scan_rows.append([i,int(spec_lines[i].rsplit()[1])])
            elif spec_lines[i].startswith("#L"):
                data_rows.append(i+1)
            elif spec_lines[i].startswith("#G0"):
                G0_rows.append(i)
            elif spec_lines[i].startswith("#G1"):
                G1_rows.append(i)
            elif spec_lines[i].startswith("#G3"):
                G3_rows.append(i)
            elif spec_lines[i].startswith("#G4"):
                G4_rows.append(i)

        if scan_number==None:#if None, then take all rodscan and Escan existing in the spec file
            for i in range(len(scan_rows)):
                scan=scan_rows[i]
                data_start=data_rows[i]
                r_index_temp,scan_number_temp=scan
                scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
                if scan_type_temp=="raxr_ascan":
                    scan_type_temp="Escan"
                j=0

                while (not spec_lines[data_start+j].startswith("#")) and (spec_lines[data_start+j]!="\n"):
                    j+=1
                row_number_range=[data_start,data_start+j]
                data_info['scan_type'].append(scan_type_temp)
                data_info['scan_number'].append(scan_number_temp)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                    except:
                        data_info[key].append([])

                for key in correction_labels.keys():
                    if key=="transmision" and beamline=="ESRF":
                        data_info[key].append([1]*j)
                    else:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))

                for key in angle_labels.keys():#we only extract rodscan and Escan data info and skip ascan
                    if scan_type_temp=='rodscan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                    if scan_type_temp=='Escan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))

                for key in G_labels.keys():
                    G_type=G_labels[key][0]
                    inxes=G_labels[key][1]
                    #ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                    ff=lambda items,inxes:[float(items[i]) for i in indxes]
                    if G_type=='G0':
                        data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G1':
                        data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G3':
                        data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G4':
                        data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

                if scan_type_temp in col_label.keys():
                    pass
                else:
                    col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()[1:]

        elif scan_number==[]:#if no spec number is specfied, then do nothing
            pass
        else:
            for ii in range(len(scan_number)):
                _scan=scan_number[ii]
                i=np.where(np.array(scan_rows)[:,1]==_scan)[0][0]
                scan=scan_rows[i]
                data_start=data_rows[i]
                r_index_temp,scan_number_temp=scan
                scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
                if scan_type_temp=="raxr_ascan":
                    scan_type_temp="Escan"
                j=0
                while (not spec_lines[data_start+j].startswith("#")) and (spec_lines[data_start+j]!="\n"):#continue until you hit a blank line or '#'
                    #print spec_lines[data_start+j].startswith(""),j,j+1
                    j+=1
                row_number_range=[data_start,data_start+j]
                data_info['scan_type'].append(scan_type_temp)
                #data_info['scan_number'].append(scan)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                    except:
                        data_info[key].append([])
                for key in correction_labels.keys():
                    if key=="transmision" and beamline=="ESRF":
                        data_info[key].append([1]*j)
                    else:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))
                for key in angle_labels.keys():
                    if scan_type_temp=='rodscan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                    if scan_type_temp=='Escan':
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))
                for key in G_labels.keys():
                    G_type=G_labels[key][0]
                    inxes=G_labels[key][1]
                    #ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                    ff=lambda items,inxes:[float(items[i]) for i in inxes]
                    if G_type=='G0':
                        data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G1':
                        data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G3':
                        data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                    if G_type=='G4':
                        data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

                #data_info['scan_type'].append(scan_type_temp)
                #data_info['scan_number'].append(_scan)
                #data_info['row_number_range'].append(row_number_range)
                if scan_type_temp in col_label.keys():
                    pass
                else:
                    col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()
            data_info['col_label']=col_label
            #print data_info['scan_number']
            f_spec.close()
        return data_info

    #build images path based on scan number and info from spec file
    def match_images(self,data_info,img_extention='.tiff'):
        data_info=data_info
        spec_name=os.path.basename(os.path.normpath(data_info['spec_path'])).replace(".spec","")
        image_head=os.path.join(os.path.dirname(data_info['spec_path']),"images")
        image_head=os.path.join(image_head,spec_name)
        data_info["images_path"]=[]
        def _number_to_string(place=4,number=1):
            i=0
            #print place-i
            if number==0:
                return '0'*place
            else:
                while int(number/(10**(place-i)))==0:
                    i+=1
                return '0'*(i-1)+str(number)

        for i in range(len(data_info["scan_number"])):
            scan_temp=data_info["scan_number"][i]
            scan_number_str='S'+_number_to_string(3,scan_temp)
            range_data_temp=data_info["row_number_range"][i]
            temp_img_container=[]
            for j in range(range_data_temp[1]-range_data_temp[0]):
                if beamline=='APS':
                    img_number=_number_to_string(5,j)+img_extention
                elif beamline=='ESRF':
                    img_number=_number_to_string(4,j)+img_extention
                temp_img_container.append(os.path.join(os.path.join(image_head,scan_number_str),"_".join([spec_name,scan_number_str,img_number])))
            data_info["images_path"].append(temp_img_container)

        return data_info

    def combine_spec_image_info(self,append_scan=False):
        data_info=self.sort_spec_file(spec_path=self.spec_path,spec_name=self.spec_name,scan_number=self.scan_number,general_labels=self.general_labels,correction_labels=self.correction_labels, angle_labels=self.angle_labels,angle_labels_escan=self.angle_labels_escan,G_labels=self.G_labels)
        data_info=self.match_images(data_info,self.img_extention)
        if append_scan:
            self.data_info_append=data_info
        else:
            self.data_info=data_info
        return None

    def find_image_from_pool(self,scan_number=3,Vs=[]):
        scan_index=self.data_info['scan_number'].index(scan_number)
        scan_type=self.data_info['scan_type'][scan_index]
        image_index=[]
        if scan_type=='rodscan':
            if Vs==[]:#take all images in that scan
                image_index=range(len(self.data_info['images_path'][scan_index]))
            elif len(Vs)==2:#means a range in between the first and the second L value
                for i in range(len(self.data_info['images_path'][scan_index])):
                    if self.data_info['L'][scan_index][i]>Vs[0] and self.data_info['L'][scan_index][i]<Vs[1]:
                        image_index.append(i)
                    else:
                        pass
            else:
                image_index=Vs
        elif scan_type=='Escan':
            if Vs==[]:#take all images in that scan
                image_index=range(len(self.data_info['images_path'][scan_index]))
            elif len(Vs)==2:#means a range in between the first and the second E value
                for i in range(len(self.data_info['images_path'][scan_index])):
                    if self.data_info['E'][scan_index][i]>Vs[0] and self.data_info['E'][scan_index][i]<Vs[1]:
                        image_index.append(i)
                    else:
                        pass
            else:
                image_index=Vs
        return [scan_number],[image_index]

    #If you want to twick through data points (If necessary run this after finishing one run of auto integration)
    def integrate_images_twick_mode(self,scan_number=[],image_indexes=None):
        for scan in scan_number:
            scan_index=self.data_info['scan_number'].index(scan)
            scan_images=self.data_info['images_path'][scan_index]
            if image_indexes!=None:
                scan_images=list(np.array(self.data_info['images_path'][scan_index])[image_indexes[scan_number.index(scan)]])

            scan_check=raw_input('Doing scan'+str(scan)+' now! Continue? (y) or n ')
            if scan_check=='y' or scan_check=='':
                temp_center_pix,temp_r_width,temp_c_width=[],9,9
                continue_tag='n'
                for image in scan_images:
                    #temp_center_pix,temp_r_width,temp_c_width=[],9,9
                    #center_pix=copy.copy(self.integ_pars['center_pix'])
                    #r_width=copy.copy(self.integ_pars['r_width'])
                    #c_width=copy.copy(self.integ_pars['c_width'])
                    if image_indexes==None:
                        image_index=scan_images.index(image)
                    else:
                        image_index=scan_images.index(image)+image_indexes[scan_number.index(scan)][0]
                    center_pix=copy.copy(self.data_info['center_pix'][scan_index][image_index])
                    r_width=copy.copy(self.data_info['r_width'][scan_index][image_index])
                    c_width=copy.copy(self.data_info['c_width'][scan_index][image_index])
                    if continue_tag=="cs":
                        image_check='y'
                        #input_items='imc'
                        print 'Doing image '+str(image_index+1)+' now!'
                    else:
                        image_check=raw_input('Doing image '+str(image_index+1)+' now! Continue? (y) or n or q ')
                    if continue_tag=='n':
                        k,kk,kkk,k1,kk1,kkk1,k2,kk2,kkk2=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                    else:
                        pass
                    if image_check=='y' or image_check=='':
                        #k,kk,kkk,k1,kk1,kkk1,k2,kk2,kkk2=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                        I,Ibrg,Ier=0,0,0
                        while 1:
                            if continue_tag=="cs":
                                input_items="imc"
                            else:
                                input_items=raw_input('Move window? using: \nw(UP), s(DOWN), a(LEFT), d(RIGHT)\nChange integration width? using:\n ci(column increasing), cd (column decreasing), \nri(row increasing),rd(row decrasing)\nimc(take the previous integration parameter values for current image)\ncs(take the previous integration parameter values for rest images)\nq(quit current loop)')
                            if input_items.startswith('w'):
                                value=int(input_items.rsplit('w')[-1])
                                center_pix[0]=center_pix[0]-value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('s'):
                                value=int(input_items.rsplit('s')[-1])
                                center_pix[0]=center_pix[0]+value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('a'):
                                value=int(input_items.rsplit('a')[-1])
                                center_pix[1]=center_pix[1]-value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('d'):
                                value=int(input_items.rsplit('d')[-1])
                                center_pix[1]=center_pix[1]+value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('ci'):
                                value=int(input_items.rsplit('i')[-1])
                                c_width=c_width+value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('cd'):
                                value=int(input_items.rsplit('d')[-1])
                                c_width=c_width-value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('ri'):
                                value=int(input_items.rsplit('i')[-1])
                                r_width=r_width+value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('rd'):
                                value=int(input_items.rsplit('d')[-1])
                                r_width=r_width-value
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,center_pix,r_width,c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('cs'):
                                continue_tag='cs'
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,temp_center_pix,temp_r_width,temp_c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                            elif input_items.startswith('imc'):
                                pyplot.close()
                                I,Ibrg,Ier,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image_for_tick(scan,image,temp_center_pix,temp_r_width,temp_c_width,True)
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                                break
                            elif input_items=='q':
                                temp_center_pix,temp_r_width,temp_c_width=center_pix,r_width,c_width
                                pyplot.close()
                                break
                            #elif input_items=='rm':
                            #    pyplot.close()
                            #    break
                        if continue_tag=="cs":
                            input_outside='y'
                        else:
                            input_outside=raw_input('Keep the current value? (y) or (n) or (rm)ove ')
                        if (input_outside=='y' and [I,Ibrg,Ier]!=[0,0,0]) or (input_outside=='' and [I,Ibrg,Ier]!=[0,0,0]):
                            #self.data_info['I'][scan_index][image_index]=I
                            #self.data_info['Ierr'][scan_index][image_index]=Ier
                            self.data_info['Ibgr'][scan_index][image_index]=Ibrg
                            self.data_info['s'][scan_index][image_index]=s
                            self.data_info['ord_cus'][scan_index][image_index]=ord_cus
                            self.data_info['center_pix'][scan_index][image_index]=center_pix
                            self.data_info['peak_width'][scan_index][image_index]=peak_width
                            self.data_info['r_width'][scan_index][image_index]=r_width
                            self.data_info['c_width'][scan_index][image_index]=c_width
                            scan_dict=self._formate_scan_from_data_info(self.data_info,scan,image_index,I,Ier,Ibrg)
                            result_dict=ctr_data.image_point_F(scan_dict,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=self.corr_params, preparsed=True)
                            self.data_info['F'][scan_index][image_index]=result_dict['F']
                            self.data_info['Ferr'][scan_index][image_index]=result_dict['Ferr']
                            self.data_info['I'][scan_index][image_index]=(result_dict['F']**2)#scaled intensity
                            self.data_info['Ierr'][scan_index][image_index]=(Ier*result_dict['F']**2/I)#scale intensity error
                        elif input_outside=="rm":
                            self.data_info['I'][scan_index][image_index]=0#set I to zero if you have removed such point
                        else:
                            pass
                    elif image_check=='n':
                        pass
                    elif image_check=='q':
                        break
                index_to_pop=np.where(np.array(self.data_info['I'][scan_index])==0)[0]
                print index_to_pop
                keys=['images_path','ord_cus','Ibgr','ctot','transmision','peak_width','chi','Ferr','nu','norm','phi','F','I','H','K','L','c_width','beta','alpha','Ierr','center_pix','mu','s','eta','del','time','r_width']
                for index in index_to_pop:
                    for key in keys:
                        self.data_info[key][scan_index].pop(index)
            else:
                pass
        return None

    def integrate_one_image(self,img_path="S3_Zr_100mM_KCl_3_S136_0000.tiff",plot_live=PLOT_LIVE):
        cutoff_scale=self.integ_pars['cutoff_scale']
        use_scale=self.integ_pars['use_scale']
        center_pix=self.integ_pars['center_pix']
        r_width=self.integ_pars['r_width']
        c_width=self.integ_pars['c_width']
        integration_direction=self.integ_pars['integration_direction']
        ord_cus_s=self.integ_pars['ord_cus_s']
        ss=self.integ_pars['ss']
        fct=self.integ_pars['fct']
        img=misc.imread(img_path)
        center_pix_temp= list(np.where(img==np.max(img[center_pix[0]-30:center_pix[0]+30,center_pix[1]-30:center_pix[1]+30])))
        center_pix=[center_pix_temp[0][0],center_pix_temp[1][0]]
        #print center_pix
        if use_scale:
            if cutoff_scale<1:
                cutoff=np.max(img)*cutoff_scale
            else:
                cutoff=cutoff_scale
            index_cutoff=np.argwhere(img>=cutoff)
        else:
            index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        pil_y,pil_x=img.shape#shape of the pilatus image
        #reset the boundary if the index number is beyond the pilatus shape
        x_min,x_max,y_min,y_max=[int(x_min>0)*x_min,int(x_max>0)*x_max,int(y_min>0)*y_min,int(y_max>0)*y_max]
        x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,int(x_max<pil_x)*x_max,int(y_min<pil_y)*y_min,int(y_max<pil_y)*y_max]
        x_span,y_span=x_max-x_min,y_max-y_min

        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        n=np.array(range(len(y)))
        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        s_container=[]
        ord_cus_container=[]
        #center_pix_container=[]
        #peak_width_container=[]
        #r_width_container=[]
        #c_width_container=[]
        index=None
        peak_width=10
        if self.integ_pars['integration_direction']=='y':
            peak_width==self.integ_pars['c_width']/5
        elif self.integ_pars['integration_direction']=='x':
            peak_width==self.integ_pars['r_width']/5
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=len(y)/2
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[rt:-1]-z[rt:-1]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            left_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            right_boundary=list(np.abs(y[rt:-1]-z[rt:-1]))
            std_fom=np.array(left_boundary+right_boundary).std()
            return sum_temp/(len(y)-peak_width*2),std_fom#averaged offset (goodness of result), standard deviation of residual (counted as error during data integration)
            #return std_fom

        def _cal_FOM_2(y,z,s):
            y=np.array(y)
            z=np.array(z)
            y_scaled=(y-y.max())/(y.max()-y.min())+1#scaled to [0,1]
            index_container=np.where(y_scaled<s)[0]
            return np.abs(y[index_container]-z[index_container]).std()

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = self.backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[index]-z[index]))
                Ibgr_container.append(abs(np.sum(z[index])))
                FOM_container.append(_cal_FOM(y,z,peak_width))
                #Ierr_container.append((I_container[-1]+FOM_container[-1])**0.5)
                Ierr_container.append((I_container[-1])**0.5+FOM_container[-1][1])#possoin error + error from integration
                z_container.append(z)
                s_container.append(s)
                ord_cus_container.append(ord_cus)
        index_best=np.argmin(np.array(FOM_container)[:,0])
        #print 'std=',FOM_container[index_best]
        #print 'all std=',FOM_container
        index = np.argsort(n)
        if plot_live:
            z=z_container[index_best]
            fig,ax=pyplot.subplots()
            ax.imshow(img)
            rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            pyplot.figure()
            pyplot.plot(n[index],y[index],color='blue',label="data")
            pyplot.plot(n[index],z[index],color="red",label="background")
            pyplot.plot(n[index],y[index]-z[index],color="m",label="data-background")
            pyplot.plot(n[index],[0]*len(index),color='black')
            pyplot.legend()
            print "When s=",s_container[index_best],'pow=',ord_cus_container[index_best],"integration sum is ",np.sum(y[index]-z[index]), " counts!"
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5
        return I_container[index_best],FOM_container[index_best],Ierr_container[index_best],s_container[index_best],ord_cus_container[index_best],center_pix,peak_width,r_width,c_width

    def show_scan_images(self,scan_number):
        scan_index=self.data_info["scan_number"].index(scan_number)
        img_num=len(self.data_info["images_path"][scan_index])
        current_index=0
        for i in range(10000):
            input_items=raw_input("Enter to move on to next point or a number (+ or -) to move forward or backward by the number of point.\nIf you want to quit then type q:")
            if input_items=="q":
                break
            elif input_items!='':
                index_temp=current_index+int(input_items)
                if index_temp<0:
                    index_temp=0
                elif index_temp>(len(self.data_info["images_path"][scan_index])-1):
                    index_temp=len(self.data_info["images_path"][scan_index])-1
                else:
                    pass
                img_temp=self.data_info["images_path"][scan_index][index_temp]
                current_index=index_temp
                self._show_one_image(img_temp)
            else:
                img_temp=self.data_info["images_path"][scan_index][current_index]
                current_index=current_index+1
                self._show_one_image(img_temp)
        return None

    def _show_one_image(self,img_path="S3_Zr_100mM_KCl_3_S136_0000.tiff"):

        scan_index=None
        image_index=None
        for i in range(len(self.data_info["scan_number"])):
            for j in range (len(self.data_info["images_path"][i])):
                if img_path==self.data_info["images_path"][i][j]:
                    scan_index,image_index=i,j
                    break
                else:
                    pass
            if scan_index!=None:
                break
            else:
                pass
        center_pix=self.data_info['center_pix'][scan_index][image_index]
        r_width=self.data_info['r_width'][scan_index][image_index]
        c_width=self.data_info['c_width'][scan_index][image_index]
        peak_width=self.data_info['peak_width'][scan_index][image_index]
        s=self.data_info['s'][scan_index][image_index]
        ord_cus=self.data_info['ord_cus'][scan_index][image_index]
        integration_direction=self.integ_pars['integration_direction']
        fct=self.integ_pars['fct']
        img=misc.imread(img_path)
        #center_pix= list(np.where(img==np.max(img[center_pix[0]-20:center_pix[0]+20,center_pix[1]-20:center_pix[1]+20])))

        index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        x_span,y_span=x_max-x_min,y_max-y_min

        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        n=np.array(range(len(y)))
        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        s_continer=[]
        ord_cus_container=[]
        #center_pix_container=[]
        #peak_width_container=[]
        #r_width_container=[]
        #c_width_container=[]
        index=None
        def _cal_FOM(y,z,peak_width):
            ct=len(y)/2
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[rt:-1]-z[rt:-1]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            return sum_temp/(len(y)-peak_width*2)*len(y)

        z,a,it,ord_cus,s,fct = self.backcor(n,y,ord_cus,s,fct)
        I=np.sum(y[index]-z[index])
        Ibgr=abs(np.sum(z[index]))
        FOM=_cal_FOM(y,z,peak_width)
        Ierr=(I+FOM)**0.5

        index = np.argsort(n)
        #fig,ax=pyplot.subplots()
        pyplot.close()
        fig,ax=pyplot.subplots(3,figsize=(10,10))
        ax[0].imshow(img,vmax=np.amax(clip_img))
        rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
        ax[0].add_patch(rect)
        ax[1].plot(n[index],y[index],color='blue',label="data")
        ax[1].plot(n[index],z[index],color="red",label="background")
        ax[1].plot(n[index],y[index]-z[index],color="m",label="data-background")
        ax[1].plot(n[index],[0]*len(index),color='black')
        self._plot_results_in_show_mode(ax[2],scan=self.data_info["scan_number"][scan_index],image=img_path,F_temp=self.data_info['F'][scan_index][image_index],Ferr_temp=self.data_info['Ferr'][scan_index][image_index])
        pyplot.legend()
        print "When s=",s,'pow=',ord_cus,"integration sum is ",np.sum(y[index]-z[index]), " counts!"
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5
        return I,FOM,Ierr

    def report_info(self):
        scan_number=self.data_info['scan_number']
        for scan in scan_number:
            i=scan_number.index(scan)
            HKL=None
            HKLE=None
            if self.data_info['scan_type'][i]=='rodscan':
                HKL='HKL='+str(int(self.data_info['H'][i][0]))+' '+str(int(self.data_info['K'][i][0]))+' L,'
                print "scan ",scan, ", ",self.data_info['scan_type'][i], ", ",HKL,len(self.data_info['images_path'][i])," images in total!"
            elif self.data_info['scan_type'][i]=='Escan':
                HKLE='HKL='+str(int(self.data_info['H'][i][0]))+' '+str(int(self.data_info['K'][i][0]))+' '+str(self.data_info['L'][i][0])+', '
                print "scan ",scan, ", ",self.data_info['scan_type'][i], ", ",HKLE,len(self.data_info['images_path'][i])," images in total!"
        return None

    def integrate_one_image_for_tick(self,scan,img_path,center_pix,r_width,c_width,plot_live=PLOT_LIVE):
        cutoff_scale=self.integ_pars['cutoff_scale']
        use_scale=self.integ_pars['use_scale']
        #center_pix=self.integ_pars['center_pix']
        #r_width=self.integ_pars['r_width']
        #c_width=self.integ_pars['c_width']
        integration_direction=self.integ_pars['integration_direction']
        ord_cus_s=self.integ_pars['ord_cus_s']
        ss=self.integ_pars['ss']
        fct=self.integ_pars['fct']
        img=misc.imread(img_path)
        #center_pix= list(np.where(img==np.max(img[center_pix[0]-20:center_pix[0]+20,center_pix[1]-20:center_pix[1]+20])))
        if use_scale:
            if cutoff_scale<1:
                cutoff=np.max(img)*cutoff_scale
            else:
                cutoff=cutoff_scale
            index_cutoff=np.argwhere(img>=cutoff)
        else:
            index_cutoff=np.array([[center_pix[0]-c_width,center_pix[1]-r_width],[center_pix[0]+c_width,center_pix[1]+r_width]])
        sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
        x_min,x_max=sub_index[0][1],sub_index[1][1]
        y_min,y_max=sub_index[0][0],sub_index[1][0]
        pil_y,pil_x=img.shape#shape of the pilatus image
        #reset the boundary if the index number is beyond the pilatus shape
        x_min,x_max,y_min,y_max=[int(x_min>0)*x_min,int(x_max>0)*x_max,int(y_min>0)*y_min,int(y_max>0)*y_max]
        x_min,x_max,y_min,y_max=[int(x_min<pil_x)*x_min,int(x_max<pil_x)*x_max,int(y_min<pil_y)*y_min,int(y_max<pil_y)*y_max]
        x_span,y_span=x_max-x_min,y_max-y_min

        clip_img=img[y_min:y_max+1,x_min:x_max+1]
        if integration_direction=="x":
            #y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
            y=clip_img.sum(axis=0)[:,np.newaxis]
        elif integration_direction=="y":
            #y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]
            y=clip_img.sum(axis=1)[:,np.newaxis]
        n=np.array(range(len(y)))
        ## Then, use BACKCOR to estimate the spectrum background
        #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

        # If you know the parameter, use this syntax:
        I_container=[]
        Ibgr_container=[]
        Ierr_container=[]
        FOM_container=[]
        z_container=[]
        s_container=[]
        ord_cus_container=[]
        index=None
        peak_width=10
        if self.integ_pars['integration_direction']=='y':
            peak_width==self.integ_pars['c_width']/5
        elif self.integ_pars['integration_direction']=='x':
            peak_width==self.integ_pars['r_width']/5
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=len(y)/2
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[0:lf]-z[0:lf]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
            return sum_temp/(len(y)-peak_width*2)*len(y)

        for s in ss:
            for ord_cus in ord_cus_s:
                z,a,it,ord_cus,s,fct = self.backcor(n,y,ord_cus,s,fct)
                I_container.append(np.sum(y[index]-z[index]))
                Ibgr_container.append(abs(np.sum(z[index])))
                FOM_container.append(_cal_FOM(y,z,peak_width))
                Ierr_container.append((I_container[-1]+FOM_container[-1])**0.5)
                z_container.append(z)
                s_container.append(s)
                ord_cus_container.append(ord_cus)
        index_best=FOM_container.index(min(FOM_container))
        index = np.argsort(n)
        if plot_live:
            z=z_container[index_best]
            fig,ax=pyplot.subplots(3,figsize=(10,10))
            ax[0].imshow(img,vmax=np.amax(clip_img))
            rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
            ax[0].add_patch(rect)

            ax[1].plot(n[index],y[index],color='blue',label="data")
            ax[1].plot(n[index],z[index],color="red",label="background")
            ax[1].plot(n[index],y[index]-z[index],color="m",label="data-background")
            ax[1].plot(n[index],[0]*len(index),color='black')
            I=I_container[index_best]
            Ier=Ierr_container[index_best]
            Ibrg=FOM_container[index_best]
            scan_dict=self._formate_scan_from_data_info(self.data_info,scan,self.data_info['images_path'][self.data_info['scan_number'].index(scan)].index(img_path),I,Ier,Ibrg)
            result_dict=ctr_data.image_point_F(scan_dict,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=self.corr_params, preparsed=True)
            F_temp=result_dict['F']
            Ferr_temp=result_dict['Ferr']
            self._plot_results_in_twick_mode(ax[2],scan=scan,image=img_path,F_temp=F_temp,Ferr_temp=Ferr_temp)
            pyplot.legend()
            print "When s=",s_container[index_best],'pow=',ord_cus_container[index_best],"integration sum is ",np.sum(y[index]-z[index]), " counts!"
        #return np.sum(y[index]-z[index]),abs(np.sum(z[index])),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5

        return I_container[index_best],FOM_container[index_best],Ierr_container[index_best],s_container[index_best],ord_cus_container[index_best],center_pix,peak_width,r_width,c_width

    def batch_image_integration(self,append_scan=False):
        if append_scan:
            data_info=self.data_info_append
        else:
            data_info=self.data_info
        if data_info['scan_number']==[]:
            return None
        else:
            scan_number=data_info['scan_number']
            scan_type=data_info['scan_type']
            images_path=data_info['images_path']
            data_info['I']=[]
            data_info['Ierr']=[]
            data_info['Ibgr']=[]
            data_info['F']=[]
            data_info['Ferr']=[]
            data_info['ctot']=[]
            data_info['alpha']=[]
            data_info['beta']=[]
            data_info['s']=[]
            data_info['ord_cus']=[]
            data_info['center_pix']=[]
            data_info['peak_width']=[]
            data_info['r_width']=[]
            data_info['c_width']=[]

            for i in range(len(scan_number)):
                images_temp=images_path[i]
                I_temp,I_bgr_temp,I_err_temp,F_temp,Ferr_temp,ctot_temp,alpha_temp,beta_temp=[],[],[],[],[],[],[],[]
                s_temp,ord_cus_temp,center_pix_temp,peak_width_temp,r_width_temp,c_width_temp=[],[],[],[],[],[]
                for image in images_temp:
                    print 'processing scan',str(scan_number[i]),'image',images_temp.index(image)
                    I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image(image,plot_live=False)
                    #I_temp.append(I)
                    I_bgr_temp.append(I_bgr)
                    #I_err_temp.append(I_err)
                    s_temp.append(s)
                    ord_cus_temp.append(ord_cus)
                    center_pix_temp.append(center_pix)
                    peak_width_temp.append(peak_width)
                    r_width_temp.append(r_width)
                    c_width_temp.append(c_width)
                    scan_dict=self._formate_scan_from_data_info(data_info,scan_number[i],images_temp.index(image),I,I_err,I_bgr)
                    #calculate the correction factor
                    result_dict=ctr_data.image_point_F(scan_dict,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=self.corr_params, preparsed=True)
                    F_temp.append(result_dict['F'])
                    Ferr_temp.append(result_dict['Ferr'])
                    I_temp.append(result_dict['F']**2)#scaled intensity
                    I_err_temp.append(I_err*result_dict['F']**2/I)#scale intensity error
                    ctot_temp.append(result_dict['ctot'])
                    alpha_temp.append(result_dict['alpha'])
                    beta_temp.append(result_dict['beta'])
                data_info['I'].append(I_temp)
                data_info['Ierr'].append(I_err_temp)
                data_info['Ibgr'].append(I_bgr_temp)
                data_info['F'].append(F_temp)
                data_info['Ferr'].append(Ferr_temp)
                data_info['ctot'].append(ctot_temp)
                data_info['alpha'].append(alpha_temp)
                data_info['beta'].append(F_temp)
                data_info['s'].append(s_temp)
                data_info['ord_cus'].append(ord_cus_temp)
                data_info['center_pix'].append(center_pix_temp)
                data_info['peak_width'].append(peak_width_temp)
                data_info['r_width'].append(r_width_temp)
                data_info['c_width'].append(c_width_temp)
            if append_scan:
                for key in self.data_info.keys():
                    if (key!='spec_path') and (key!='col_label'):
                        self.data_info[key]=self.data_info[key]+data_info[key]
                self.data_info_append=[]
            else:
                self.data_info=data_info

            return None

    def _formate_scan_from_data_info(self,data_info,scan_number,image_number,I,Ierr,Ibgr):
        scan_index=data_info['scan_number'].index(scan_number)
        image_index=image_number
        or0_list=data_info['or0'][scan_index]
        or1_list=data_info['or1'][scan_index]
        or0_lib={'h':or0_list[0:3]}
        or0_lib['delta'],or0_lib['eta'],or0_lib['chi'],or0_lib['phi'],or0_lib['nu'],or0_lib['mu'],or0_lib['lam']=or0_list[3:10]
        or1_lib={'h':or1_list[0:3]}
        or1_lib['delta'],or1_lib['eta'],or1_lib['chi'],or1_lib['phi'],or1_lib['nu'],or1_lib['mu'],or1_lib['lam']=or1_list[3:10]

        psicG=(data_info['cell'][scan_index],or0_lib,or1_lib,data_info['n_azt'][scan_index])
        scan_dict = {'I':[I],
                     'norm':[data_info['norm'][scan_index][image_index]],
                     'Ierr':[Ierr],
                     'Ibgr':[Ibgr],
                     'dims':(1,0),
                     'transmision':[data_info['transmision'][scan_index][image_index]],
                     'phi':[data_info['phi'][scan_index][image_index]],
                     'chi':[data_info['chi'][scan_index][image_index]],
                     'eta':[data_info['eta'][scan_index][image_index]],
                     'mu':[data_info['mu'][scan_index][image_index]],
                     'nu':[data_info['nu'][scan_index][image_index]],
                     'del':[data_info['del'][scan_index][image_index]],
                     'G':psicG}
        return scan_dict

    #remove spikes for plotting and saving results purpose
    def remove_spikes(self,L,col_data,bragg_peaks=BRAGG_PEAKS,offset=BRAGG_PEAK_CUTOFF):
        cutoff_ranges=[]
        L_new=[]
        col_data_new=[]
        for peak in bragg_peaks:
            cutoff_ranges.append([peak-offset,peak+offset])
        for i in range(len(L)):
            l=L[i]
            sensor=False
            for cutoff in cutoff_ranges:
                if l>cutoff[0] and l<cutoff[1]:
                    sensor=True
                    break
                else:pass
            if not sensor:
                L_new.append(l)
                col_data_new.append(col_data[i])
            else:pass
        return L_new,col_data_new

    #the row_range is not updated in this function. It doesn't matter, since we are not using row_range after batch integration!
    #This issue needs to be fixed whenever in future we need to use correct row_range for some purpose
    def remove_spikes_from_data_info(self):
        #this list may expand in future update
        keys=['images_path','ord_cus','Ibgr','ctot','transmision','peak_width','chi','Ferr','nu','norm','phi','F','I','H','K','L','c_width','beta','alpha','Ierr','center_pix','mu','s','eta','del','time','r_width']
        for i in range(len(self.data_info['scan_number'])):
            if self.data_info['scan_type'][i]=='rodscan':
                L_column=copy.copy(self.data_info['L'][i])
                for key in keys:
                    self.data_info[key][i]=self.remove_spikes(L_column,self.data_info[key][i])[1]

        return None


    #you can plot results for several scans
    def _plot_results_in_twick_mode(self,ax,scan,image,F_temp,Ferr_temp):
        data=self.data_info
        scan_index=data['scan_number'].index(scan)
        image_index=data['images_path'][scan_index].index(image)
        scan_type=data['scan_type'][scan_index]

        if scan_type=='rodscan':
            x,y,yer=data['L'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
            #x_,y_=self.remove_spikes(x,y)
            #x_,yer_=self.remove_spikes(x,yer)
            #x,y,yer=x_,y_,yer_
            x_s,y_s,yer_s=[data['L'][scan_index][image_index]],F_temp,Ferr_temp
            title='CTR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+'L)'
            ax.set_yscale('log')
            print len(x),len(y)
            ax.scatter(x,y,marker='s',s=5)
            ax.errorbar(x,y,yerr=yer,fmt=None)
            ax.plot(x,y,linestyle='-',lw=1.5)
            ax.scatter(x_s,y_s,marker='d',s=12,color='r')
            ax.errorbar(x_s,y_s,yerr_s=yer,fmt=None,color='r')
            pyplot.xlim(xmin=-0.3,xmax=max(x)+0.3)
            pyplot.title(title)
        elif scan_type=='Escan':
            x,y,yer=data['E'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
            x_s,y_s,yer_s=[data['E'][scan_index][image_index]],F_temp,Ferr_temp
            title='RAXR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+str(data['L'][scan_index][0])+')'
            #ax.set_yscale('log')
            ax.scatter(x,y,marker='s',s=7)
            ax.errorbar(x,y,yerr=yer,fmt=None)
            ax.plot(x,y,linestyle='-',lw=1.5)
            ax.scatter(x_s,y_s,marker='d',s=8,color='r')
            ax.errorbar(x_s,y_s,yerr=yer_s,fmt=None,color='r')
            pyplot.title(title)
        else:
            pass
        return None

    def _plot_results_in_show_mode(self,ax,scan,image,F_temp,Ferr_temp):
        data=self.data_info
        scan_index=data['scan_number'].index(scan)
        image_index=data['images_path'][scan_index].index(image)
        scan_type=data['scan_type'][scan_index]

        if scan_type=='rodscan':
            x,y,yer=data['L'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
            x_,y_=self.remove_spikes(x,y)
            x_,yer_=self.remove_spikes(x,yer)
            x,y,yer=x_,y_,yer_
            x_s,y_s,yer_s=[data['L'][scan_index][image_index]],F_temp,Ferr_temp
            title='CTR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+'L)'
            ax.set_yscale('log')
            ax.scatter(x,y,marker='s',s=5)
            ax.errorbar(x,y,yerr=yer,fmt=None)
            ax.plot(x,y,linestyle='-',lw=1.5)
            ax.scatter(x_s,y_s,marker='d',s=8,color='r')
            ax.errorbar(x_s,y_s,yerr_s=yer,fmt=None,color='r')
            pyplot.xlim(xmin=-0.3,xmax=max(x)+0.3)
            pyplot.title(title)
        elif scan_type=='Escan':
            x,y,yer=data['E'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
            x_s,y_s,yer_s=[data['E'][scan_index][image_index]],F_temp,Ferr_temp
            title='RAXR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+str(data['L'][scan_index][0])+')'
            #ax.set_yscale('log')
            ax.scatter(x,y,marker='s',s=7)
            ax.errorbar(x,y,yerr=yer,fmt=None)
            ax.plot(x,y,linestyle='-',lw=1.5)
            ax.scatter(x_s,y_s,marker='d',s=8,color='r')
            ax.errorbar(x_s,y_s,yerr=yer_s,fmt=None,color='r')
            pyplot.title(title)
        else:
            pass
        return None

    #you can plot results for several scans
    def plot_results(self,scan_number=None):
        data=self.data_info
        if scan_number!=None:
            scan_number=scan_number
        else:
            scan_number=data['scan_number']
        fig1=pyplot.figure(figsize=(10,5))
        ax1=fig1.add_subplot(1,1,1)
        ax1.set_yscale('log')

        for scan in scan_number:
            scan_index=data['scan_number'].index(scan)
            scan_type=data['scan_type'][scan_index]
            if scan_type=='rodscan':
                x,y,yer=data['L'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
                #x,y,yer=data['L'][scan_index],np.array(data['I'][scan_index])*np.array(data['ctot'][scan_index]),data['Ferr'][scan_index]
                x_,y_=self.remove_spikes(x,y)
                x_,yer_=self.remove_spikes(x,yer)
                x,y,yer=x_,y_,yer_
                title='CTR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+'L)'
                #fig=pyplot.figure(figsize=(10,5))
                #ax=fig.add_subplot(1,1,1)
                #ax.set_yscale('log')
                ax1.scatter(x,y,marker='s',s=5)
                ax1.errorbar(x,y,yerr=yer,fmt=None)
                ax1.plot(x,y,linestyle='-',lw=1.5)
                pyplot.xlim(xmin=-0.3,xmax=max(x)+0.3)
                pyplot.title(title)
            elif scan_type=='Escan':
                x,y,yer=data['E'][scan_index],data['F'][scan_index],data['Ferr'][scan_index]
                title='RAXR data: HKL=('+str(int(data['H'][scan_index][0]))+str(int(data['K'][scan_index][0]))+str(data['L'][scan_index][0])+')'
                fig2=pyplot.figure(figsize=(8,4))
                ax2=fig2.add_subplot(1,1,1)
                ax2.scatter(x,y,marker='s',s=7)
                ax2.errorbar(x,y,yerr=yer,fmt=None)
                ax2.plot(x,y,linestyle='-',lw=1.5)
                pyplot.title(title)
            else:
                pass
        return None

    def save_data(self,file_path=None,file_name=None,formate={'rodscan':['L','H','K',0,'I','Ierr',2,2],'Escan':['E','H','K','L','I','Ierr',2,2]},data_formate=['%10.8f','%5.2f','%5.2f','%5.2f','%10.8f','%10.8f','%i','%i'],labels={'First_point':None},combine_in_one=True):
        #if both file path and file name are None, the data file will be save in the spec file folder
        #formate is a library with two keys (rodscan and Escan), and the key values are a set of list of either str or number:
        #####If it is a str, it could be any str taken by self.data_info.keys() with image_like formate (full list is ["H","K","L","chi","mu","nu","eta","del","phi","time","norm","images_path","I","Ierr","F","Ferr","E","s","ord_cus","center_pix","r_width","c_width","peak_width","Ibgr","ctot","transmision","beta","alpha"])
        #####if it is a number, it will fill one column with this number
        #labels: you specify the corresponding scans for different point, eg labels={'First_point':[1,15],'Second_point':[16,35]} means you have collected two sets of data at two spots (one goes from Scan 1 to scan 15, and the other goese from Scan 16 to Scan 35)
        #Set combine_in_one to True if you want to combine all data subsets into a single full set (in this case the shape of rodscan and Escan items in formate must match to each other (same length))
        #Set combine_in_one to False if you want to save data subsets to a seperate data file (in this case the shape of rodscan and Escan items in formate does not need to match to each other (different length is OK))
        if file_path==None:
            file_path=os.path.dirname(self.data_info['spec_path'])
        if file_name==None:
            file_name=ntpath.basename(self.data_info['spec_path']).replace('.spec','')
        if len(labels.keys())==1 and labels.values()==[None]:
            labels={labels.keys()[0]:[min(self.data_info['scan_number']),max(self.data_info['scan_number'])]}
        scan_number=self.data_info['scan_number']
        save_data_scan_container=[]
        labels_container=[]
        def _return_index(pool,min_v,max_v):
            return_list=[]
            for each in pool:
                if each>=min_v and each<=max_v:
                    return_list.append(each)
            return return_list

        for key in labels.keys():
            scan_temp=_return_index(scan_number,labels[key][0],labels[key][1])
            save_data_scan_container.append(scan_temp)
            labels_container.append(key)
        for i in range(len(save_data_scan_container)):
            current_scan=save_data_scan_container[i]
            current_label=labels_container[i]
            if combine_in_one:
                file_temp=os.path.join(file_path,file_name)+'_'+current_label+'_combined.dat'
                data_array=np.zeros((1,len(formate['rodscan'])))[0:0]
                for each_scan in current_scan:
                    scan_index=scan_number.index(each_scan)
                    scan_type=self.data_info['scan_type'][scan_index]
                    items=formate[scan_type]
                    first_non_number=None
                    for each in items:
                        if type(each)!=type(' '):
                            pass
                        else:
                            first_non_number=each
                            break
                    data_span_col_dir=len(self.data_info[first_non_number][scan_index])
                    temp_data_array=np.zeros(data_span_col_dir)[:,np.newaxis]
                    for item in items:
                        if type(item)==type(' '):
                            temp_data_array=np.append(temp_data_array,np.array(self.data_info[item][scan_index])[:,np.newaxis],axis=1)
                        else:
                            temp_data_array=np.append(temp_data_array,np.array([item]*data_span_col_dir)[:,np.newaxis],axis=1)
                    #print temp_data_array.shape
                    data_array=np.append(data_array,temp_data_array[:,1:],axis=0)
                header=None
                if 'rodscan' in formate.keys() and 'Escan' in formate.keys():
                    header='rodscan: '+str(formate['rodscan'])+'\n'+'Escan: '+str(formate['Escan'])
                if 'rodscan' in formate.keys() and 'Escan' not in formate.keys():
                    header='rodscan: '+str(formate['rodscan'])
                if 'rodscan' not in formate.keys() and 'Escan' in formate.keys():
                    'Escan: '+str(formate['Escan'])
                np.savetxt(file_temp,data_array,fmt=data_formate,header=header)

            else:
                file_temp_head=os.path.join(file_path,file_name)+'_'+current_label+'_'
                for each_scan in current_scan:
                    scan_index=scan_number.index(each_scan)
                    scan_type=self.data_info['scan_type'][scan_index]
                    file_temp=file_temp_head+scan_type+'_Scan'+str(each_scan)+'_'
                    if scan_type=='rodscan':
                        file_temp=file_temp+str(int(self.data_info['H'][scan_index][0]))+str(int(self.data_info['K'][scan_index][0]))+'L.dat'
                    elif scan_type=='Escan':
                        file_temp=file_temp+str(int(self.data_info['H'][scan_index][0]))+str(int(self.data_info['K'][scan_index][0]))+str(self.data_info['L'][scan_index][0])+'.dat'
                    items=formate[scan_type]
                    first_non_number=None
                    for each in items:
                        if type(each)!=type(' '):
                            pass
                        else:
                            first_non_number=each
                            break
                    data_span_col_dir=len(self.data_info[first_non_number][scan_index])
                    temp_data_array=np.zeros(data_span_col_dir)[:,np.newaxis]
                    for item in items:
                        if type(item)==type(' '):
                            temp_data_array=np.append(temp_data_array,np.array(self.data_info[item][scan_index])[:,np.newaxis],axis=1)
                        else:
                            temp_data_array=np.append(temp_data_array,np.array([item]*data_span_col_dir)[:,np.newaxis],axis=1)
                    np.savetxt(file_temp,temp_data_array[:,1:],fmt=data_formate,header=str(formate[scan_type]))
        return None

    def dump_data_info(self,file_path=None,file_name=None):#dump data_info to file
        if file_path==None:
            file_path=os.path.dirname(self.data_info['spec_path'])
        if file_name==None:
            file_name=ntpath.basename(self.data_info['spec_path']).replace('.spec','')+'_dump_data_info.dump'
        pickle.dump(self.data_info,open(os.path.join(file_path,file_name),"wb"))
        return None

#Call this when you have no idea where is the center pixe
def show_pixe_image(data_info,scan_number):
    scan_index=data_info['scan_number'].index(scan_number)
    images=data_info['images_path'][scan_index][0:min([10,len(data_info['images_path'][scan_index])])]
    for image in images:
        img=misc.imread(image)
        fig,ax=pyplot.subplots()
        ax.imshow(img)
    return None

if __name__=="__main__":
    #import the data_integration first if you do the following step by step in Python terminal
    spec_path='M:\\fwog\\members\\qiu05\\1704_APS_13IDC\\mica'
    spec_name='sb1_32mM_CaCl2_Zr_1.spec'
    scan_number=[13,14,15,16,17,18,19,20,21]#13 14 is CTR, the others are RAXR in this example
    #edit the global pars in the head part (especially center_pix, and integration setup)
    dataset=data_integration.data_integration(spec_path=spec_path,spec_name=spec_name,scan_number=scan_number)
    dataset.report_info()#get some info about the scan
    dataset.plot_results()#plot all scans
    dataset.append_scan([23,24])#if you want to append new scan
    scan,image=dataset.find_image_from_pool(scan_number=11,Vs=[10.3,11.6])#if you are not satisfied with the rod scan of L ranging between 10.3 and 11.6
    dataset.integrate_images_twick_mode(scan, image)#this function will launch a twick mode to twick through the images you have selected
    dataset.show_scan_images(13)#now you can quickly check the integration by specfifying the scan number
    dataset.save_data()#save the data inf the formate you wish (read the ducumentation under the save_data function for details)
    dataset.dump_data_info()#pickle self.data_into to file for being reloaded in future
    ##to reload saved dump file##
    dataset=data_integration.data_integration(spec_path=spec_path,spec_name=spec_name,scan_number=[])
    dataset.reload_pickle_dump_file('sb1_32mM_CaCl2_Zr_1_dump_data_info.dump')
    ##from here you can check anything you want now##
