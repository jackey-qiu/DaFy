import numpy as np
from numpy.matlib import repmat
from numpy.linalg import pinv
from matplotlib import pyplot
from scipy import misc
import fnmatch
import os
import matplotlib.patches as patches
import ctr_data


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

def backcor(n,y,ord_cus,s,fct):


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

    #z = (z-1)*dely + maxy


    return z,a,it,ord_cus,s,fct
def _get_col_from_file(lines,start_row,end_row,col,type=float):
    numbers=[]
    for i in range(start_row,end_row):
        numbers.append(type(lines[i].rstrip().rsplit()[col]))
    return numbers

def sort_spec_file(spec_path='.',spec_name='mica-zr_s2_longt_1.spec',scan_number=[16,17,19],\
                general_labels={'H':'H','K':'K','L':'L','E':'Energy'},correction_labels={'time':'Seconds','norm':'io','transmision':'transm'},\
                angle_labels={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'},\
                angle_labels_escan={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'},\
                G_labels={'n_azt':['G0',[3,6]],'cell':['G1',[0,6]],'or0':['G1',[11,14]],'or1':['G1',[14,17]],'lambda':['G4',[3,4]]}):
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

    if scan_number==[]:
        for i in range(len(scan_rows)):
            scan=scan_rows[i]
            data_start=data_rows[i]
            r_index_temp,scan_number_temp=scan
            scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
            j=0
            while not spec_lines[data_start+j].startswith("#"):
                j+=1
            row_number_range=[data_start,data_start+j]
            data_info['scan_type'].append(scan_type_temp)
            data_info['scan_number'].append(scan)
            data_info['row_number_range'].append(row_number_range)
            data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

            for key in general_labels.keys():
                try:
                    data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                except:
                    data_info[key].append([])

            for key in correction_labels.keys():
                data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))

            for key in angle_labels.keys():
                if scan_type_temp=='rodscan':
                    data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                if scan_type_temp=='Escan':
                    data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))

            for key in G_labels.keys():
                G_type=G_labels[key][0]
                inxes=G_labels[key][1]
                ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
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
    else:
        for ii in range(len(scan_number)):
            _scan=scan_number[ii]
            i=np.where(np.array(scan_rows)[:,1]==_scan)[0][0]
            scan=scan_rows[i]
            data_start=data_rows[i]
            r_index_temp,scan_number_temp=scan
            scan_type_temp=spec_lines[r_index_temp].rsplit()[2]
            j=0
            while not spec_lines[data_start+j].startswith("#"):
                j+=1
            row_number_range=[data_start,data_start+j]
            data_info['scan_type'].append(scan_type_temp)
            data_info['scan_number'].append(scan)
            data_info['row_number_range'].append(row_number_range)
            data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

            for key in general_labels.keys():
                try:
                    data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                except:
                    data_info[key].append([])

            for key in correction_labels.keys():
                data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(correction_labels[key]),type=float))

            for key in angle_labels.keys():
                if scan_type_temp=='rodscan':
                    data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels[key]),type=float))
                if scan_type_temp=='Escan':
                    data_info[key].append(_get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(angle_labels_escan[key]),type=float))

            for key in G_labels.keys():
                G_type=G_labels[key][0]
                inxes=G_labels[key][1]
                ff=lambda items,inxes:[float(item) for item in items[inxes[0]:inxes[1]]]
                if G_type=='G0':
                    data_info[key].append(ff(spec_lines[G0_rows[i]].rstrip().rsplit()[1:],inxes))
                if G_type=='G1':
                    data_info[key].append(ff(spec_lines[G1_rows[i]].rstrip().rsplit()[1:],inxes))
                if G_type=='G3':
                    data_info[key].append(ff(spec_lines[G3_rows[i]].rstrip().rsplit()[1:],inxes))
                if G_type=='G4':
                    data_info[key].append(ff(spec_lines[G4_rows[i]].rstrip().rsplit()[1:],inxes))

            data_info['scan_type'].append(scan_type_temp)
            #data_info['scan_number'].append(_scan)
            data_info['row_number_range'].append(row_number_range)
            if scan_type_temp in col_label.keys():
                pass
            else:
                col_label[scan_type_temp]=spec_lines[data_start-1].rstrip().rsplit()
        data_info['col_label']=col_label
        f_spec.close()
    return data_info

def match_images(data_info,img_extention='.tiff'):
    data_info=data_info
    spec_name=os.path.basename(os.path.normpath(data_info['spec_path'])).replace(".spec","")
    image_head=os.path.join(os.path.dirname(data_info['spec_path']),"images")
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
            img_number=_number_to_string(5,j)+img_extention
            temp_img_container.append(os.path.join(image_head,"_".join([spec_name,scan_number_str,img_number])))
        data_info["images_path"].append(temp_img_container)

    return data_info

def combine_spec_image_info(spec_path='.',spec_name='mica-zr_s2_longt_1.spec',scan_number=[16,19],img_extention='.tiff'):
    data_info=sort_spec_file(spec_path,spec_name,scan_number)
    data_info=match_images(data_info,img_extention)
    return data_info

def integrate_one_image(img_path="S3_Zr_100mM_KCl_3_S136_0000.tiff",cutoff_scale=1,use_scale=False,center_pix=[100,247],r_width=20,c_width=20,integration_direction="x",ord_cus=4,ss=[0.0005],fct='ah',plot_live=False):
    img=misc.imread(img_path)
    center_pix= list(np.where(img==np.max(img[center_pix[0]-20:center_pix[0]+20,center_pix[1]-20:center_pix[1]+20])))
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
    counts=[]
    index=None
    for s in ss:
        z,a,it,ord_cus,s,fct = backcor(n,y,ord_cus,s,fct)
        index = np.argsort(n)
        if plot_live:
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
        counts.append(np.sum(y[index]-z[index]))
        print "When s=",s,"integration sum is ",np.sum(y[index]-z[index]), " counts!"
    return np.sum(y[index]-z[index]),np.sum(z[index]),np.sum(y[index])**0.5+np.sum(y[index]-z[index])**0.5

def batch_image_integration(data_info,normalization={'i0':'i0','ct':'t'}):
    data_info=data_info
    scan_number=data_info['scan_number']
    scan_type=data_info['scan_type']
    images_path=data_info['images_path']
    data_info['I']=[]
    data_info['Ierr']=[]
    data_info['Ibgr']=[]
    for i in range(len(scan_number)):
        images_temp=images_path[i]
        I_temp,I_bgr_temp,I_err_temp=[],[]
        for image in images_temp:
            I,I_bgr_temp,I_err=integrate_one_image(image,cutoff_scale=1,integration_direction="x",ord_cus=4,ss=[0.0005],fct='ah',plot_live=False)
            I_temp.append(I)
            I_bgr_temp.append(I_bgr)
            I_err.append(I_err)
        data_info['I'].append(I_temp)
        data_info['Ierr'].append(I_err_temp)
        data_info['Ibgr'].append(I_bgr_temp)
    return data_info

def formate_scan_from_data_info(data_info,scan_number,image_number):
    scan_index=data_info['scan_number'].index(scan_number)
    image_index=image_number
    psicG=(data_info['cell'][scan_index][image_index],data_info['or0'][scan_index][image_index],data_info['or1'][scan_index][image_index],data_info['n_azt'][scan_index][image_index])
    scan_dict = {'I':[data_info['I'][scan_index][image_index]],
                 'norm':[data_info['norm'][scan_index][image_index]],
                 'Ierr':[data_info['Ierr'][scan_index][image_index]],
                 'Ibgr':[data_info['Ibgr'][scan_index][image_index]],
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

def cal_cor_factors(scan_info,corr_params={'scale':1,'geom':'psic','beam_slits':None,'det_slits':None,'sample':{'dia':10,'polygon':[],'angles':[]}}):
    d=ctr_data.image_point_F(scan,point=0,I='I',Inorm='norm',Ierr='Ierr',Ibgr='Ibgr', transm='transmision', corr_params=corr_params, preparsed=False)
    return d
if __name__=="__main__":
    # First, let's start with a simulated spectrum
    #  The simulated spectrum is the sum of Gaussian peaks, a background and Gaussian noise.

    # Wavelength
    N = 50                                # Signal length
    n = rand(N)*500+400                # Non-uniform & non-sorted wavelengths
    m = (n-min(n))/(max(n)-min(n))         # Normalized wavelengths (between -1 and 1)

    # Peaks
    sigx = 1                               # Peak amplitude deviation
    sigw = 0.01                            # Peak width deviation
    K = ceil(abs(randn(1)*10))             # Peak number
    K=1
    c = rand(K,1)                          # Peak positions
    a = abs(randn(K,1) * sigx)             # Peak amplitudes
    w = abs(randn(K,1) * sigw)             # Peak widths
    nn = repmat(m[np.newaxis,:],float(K),1.)
    cc = repmat(c,1,N)
    aa = repmat(a,1,N)
    ww = repmat(w,1,N)
    peaks = aa*exp(-(nn-cc)**2/2./ww**2)  # Gaussian peaks

    # Background
    o = floor(rand(1)*2) + 4               # Polynomial order for the simulated background
    a = randn(o+1,1)                       # Polynomial coefficients
    p = np.arange(o+1)[:,np.newaxis]
    mm = repmat(np.transpose(m),float(o+1),1)
    aa = repmat(a,1,N)
    pp = repmat(p,1,N)
    z = aa * mm**pp                       # Polynomial background
    z = np.sum(z,0).transpose() + sin(m)                 # Add a sine to the polynomial
    z=z[:,np.newaxis]
    background = z / max(z) * sigx         # Rescale background

    # Noise
    sign = 0.05
    noise = randn(N,1) * sign

    # Final Spectrum
    y = np.sum(peaks,0)[:,np.newaxis] + background + noise

    #print np.shape(y)
    img=misc.imread("S3_Zr_100mM_KCl_3_S136_0000.tiff")
    cutoff=np.max(img)*0.00008
    cutoff=100
    index_cutoff=np.argwhere(img>=cutoff)
    sub_index=[np.min(index_cutoff,axis=0),np.max(index_cutoff,axis=0)]
    pyplot.imshow(img)
    integration_direction="y"#x or y
    if integration_direction=="x":
        y=img.sum(axis=0)[:,np.newaxis][sub_index[0][1]:sub_index[1][1]]
    elif integration_direction=="y":
        y=img.sum(axis=1)[:,np.newaxis][sub_index[0][0]:sub_index[1][0]]

    n=np.array(range(len(y)))
    ## Then, use BACKCOR to estimate the spectrum background
    #  Either you know which cost-function to use as well as the polynomial order and the threshold value or not.

    # If you know the parameter, use this syntax:
    ord_cus = 6
    ss = np.arange(0.00001,0.02,0.001)
    fct = 'ah'
    counts=[]
    for s in ss:
        z,a,it,ord_cus,s,fct = backcor(n,y,ord_cus,s,fct)

        if s==ss[0] or s==ss[-1]:
            index = np.argsort(n)
            pyplot.figure()
            pyplot.plot(n[index],y[index],color='blue',label="data")
            pyplot.plot(n[index],z[index],color="red",label="background")
            pyplot.plot(n[index],(y[index]-z[index])*((y[index]-z[index])>=0),color="m",label="data-background")
        counts.append(np.sum(y[index]-z[index]))
        print it
        print "When s=",s,"integration sum is ",np.sum(y[index]-z[index]), " counts!"
    pyplot.figure()
    pyplot.plot(ss,counts)
