#!/usr/bin/python
#from mpi4py import MPI
import numpy as np
from numpy import *
from datetime import datetime
import sys
import time,os
import pickle
from numpy.matlib import repmat
from numpy.linalg import pinv
from matplotlib import pyplot
from scipy import misc
import fnmatch
import os
import matplotlib.patches as patches
import ctr_data

spec_path='/net/filet/team/fwog/members/qiu05/1704_APS_13IDC/mica'
spec_name='sb1_32mM_CaCl2_Zr_1.spec'
scan_number=[13,14,15,16,17]
scan_number=[13]

#global variables
PLOT_LIVE=False

###########integration setup here##############
INTEG_PARS={}
INTEG_PARS['cutoff_scale']=0.001
INTEG_PARS['use_scale']=False#Set this to False always
INTEG_PARS['center_pix']=[53,153]#Center pixel index (know Python is column basis, so you need to swab the order of what you see at pixe image)
INTEG_PARS['r_width']=15#integration window in row direction (total row length is twice that value)
INTEG_PARS['c_width']=50#integration window in column direction (total column length is twice that value)
INTEG_PARS['integration_direction']='y'#integration direction (x-->row direction, y-->column direction), you should use 'y' for horizontal mode (Bragg peak move left to right), and 'x' for vertical mode (Bragg peak move up and down)
INTEG_PARS['ord_cus_s']=[1,2,4,6] #A list of integration power to be tested for finding the best background subtraction. Flat if the value is 0. More wavy higher value
INTEG_PARS['ss']=[0.01,0.05,0.1]#a list of thereshold factors used in cost function (0: all signals are through, means no noise background;1:means all backround, no peak signal. You should choose a value between 0 and 1)
INTEG_PARS['fct']='ah'#Type of cost function ('sh','ah','stq' or 'atq')
################################################
#############spec file info here################
GENERAL_LABELS={'H':'H','K':'K','L':'L','E':'Energy'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
CORRECTION_LABELS={'time':'Seconds','norm':'io','transmision':'transm'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
ANGLE_LABELS={'del':'TwoTheta','eta':'theta','chi':'chi','phi':'phi','nu':'Nu','mu':'Psi'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
ANGLE_LABELS_ESCAN={'del':'del','eta':'eta','chi':'chi','phi':'phi','nu':'nu','mu':'mu'}#Values are symbols for keys. (Beamline dependent, default symbols are based on APS 13IDC beamline)
#G label positions (n_azt: azimuthal reference vector positions @3rd to 6th numbers counting from left to right at G0 line)
#so are the other symbols: cell (lattice cell info), or0 (first orientation matrix), or1 (second orientation matrix), lambda (x ray wavelength)
G_LABELS={'n_azt':['G0',range(3,6)],'cell':['G1',range(0,6)],'or0':['G1',range(12,15)+range(18,24)+[30]],'or1':['G1',range(15,18)+range(24,30)+[31]],'lambda':['G4',range(3,4)]}
IMG_EXTENTION='.tif'#image extention (.tif or .tiff)
CORR_PARAMS={'scale':1000000,'geom':'psic','beam_slits':{'horz':0.06,'vert': 1},'det_slits':None,'sample':{'dia':10,'polygon':[],'angles':[]}}#slits are in mm

#comm=MPI.COMM_WORLD
#size=comm.Get_size()
#rank=comm.Get_rank()
rank=1
size=64

def find_boundary(n_process,n_jobs,rank):
    step_len=int(n_jobs/n_process)
    remainder=int(n_jobs%n_process)
    left,right=0,0
    if rank<=remainder-1:
        left=rank*(step_len+1)
        right=(rank+1)*(step_len+1)-1
    elif rank>remainder-1:
        left=remainder*(step_len+1)+(rank-remainder)*step_len
        right=remainder*(step_len+1)+(rank-remainder+1)*step_len-1
    return left,right+1

class data_integration_mpi:
    def __init__(self,spec_path='/Users/cqiu/data/APS/1704_APS_13IDC',spec_name='sb1_32mM_CaCl2_Zr_1.spec',scan_number=[13,14,15,16],\
                corr_params=CORR_PARAMS,
                integ_pars=INTEG_PARS,\
                general_labels=GENERAL_LABELS,\
                correction_labels=CORRECTION_LABELS,\
                angle_labels=ANGLE_LABELS,\
                angle_labels_escan=ANGLE_LABELS_ESCAN,\
                G_labels=G_LABELS,\
                img_extention=IMG_EXTENTION,\
                rank=rank,\
                size=size):

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
        self.rank=rank
        self.size=size
        self.combine_spec_image_info()
        self.assign_jobs()
        self.extract_data_info()
        #self.batch_image_integration()

    def assign_jobs(self):
        scan_holder=[]
        image_holder=[]
        images=[]

        for scan in self.data_info['scan_number']:
            scan_index=self.data_info['scan_number'].index(scan)
            for image in self.data_info['images_path'][scan_index]:
                images.append(image)
        start_index,end_index=None,None
        start_index,end_index=find_boundary(size,len(images),rank)

        #if rank!=(size-1):
        #    start_index=(len(images)/size)*rank
        #    end_index=start_index+len(images)/size
        #else:
        #    start_index=(len(images)/size)*rank
        #    end_index=len(images)-1
        #print start_index,end_index,len(images),range(start_index,end_index)
        for i in range(start_index,end_index):
            image=images[i]
            items=image.replace(self.img_extention,"").rsplit("_")
            scan_holder.append(int(items[-2][1:]))
            image_holder.append(int(items[-1]))
        self.scan_holder=scan_holder
        self.image_holder=image_holder
        #print len(images)
        return None

    def extract_data_info(self):
        data_info=self.data_info
        data_info_partial={}
        data_info_partial['spec_path']=data_info['spec_path']
        data_info_partial['col_label']=data_info['col_label']
        data_info_partial['scan_number']=[]
        data_info_partial['scan_type']=[]
        data_info_partial['row_number_range']=[]
        data_info_partial['images_path']=[]
        data_info_partial['or0']=[]
        data_info_partial['or1']=[]
        data_info_partial['n_azt']=[]
        data_info_partial['transmision']=[]
        data_info_partial['chi']=[]
        data_info_partial['nu']=[]
        data_info_partial['mu']=[]
        data_info_partial['eta']=[]
        data_info_partial['del']=[]
        data_info_partial['phi']=[]
        data_info_partial['time']=[]
        data_info_partial['H']=[]
        data_info_partial['K']=[]
        data_info_partial['L']=[]
        data_info_partial['E']=[]
        data_info_partial['lambda']=[]
        data_info_partial['cell']=[]
        data_info_partial['norm']=[]
        data_info_partial['images_path']=[]
        parced_items=["H","K","L","chi","mu","nu","eta","del","phi","time","norm","images_path"]
        scans_unique=[]

        for scan in self.scan_holder:
            if scan not in scans_unique:
                scans_unique.append(scan)
            else:
                pass
        for scan in scans_unique:
            parced_items_temp=parced_items
            index_temp=list(np.where(np.array(self.scan_holder)==scan)[0])
            scan_index=self.data_info['scan_number'].index(scan)
            image_index=np.array(self.image_holder)[index_temp]
            if self.data_info['scan_type'][scan_index]=="Escan":
                parced_items_temp.append("E")
            for item in data_info.keys():
                if item in parced_items_temp:
                    data_info_partial[item].append(list(np.array(data_info[item][scan_index])[image_index]))
                elif item=="row_number_range":
                    data_info_partial[item].append([data_info[item][scan_index][0]+image_index[0],data_info[item][scan_index][0]+image_index[-1]+1])
                elif item in ["spec_path","col_label"]:
                    pass
                else:
                    data_info_partial[item].append(data_info[item][scan_index])
        self.data_info_full=self.data_info
        self.data_info=data_info_partial
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

    #extract info from spec file
    def sort_spec_file(self,spec_path='.',spec_name='mica-zr_s2_longt_1.spec',scan_number=[16,17,19],\
                    general_labels={'H':'H','K':'K','L':'L','E':'Energy'},correction_labels={'time':'Seconds','norm':'io','transmision':'transm'},\
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
                data_info['scan_number'].append(scan_number_temp)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                    except:
                        data_info[key].append([])

                for key in correction_labels.keys():
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
                #data_info['scan_number'].append(scan)
                data_info['row_number_range'].append(row_number_range)
                data_item_labels=spec_lines[data_start-1].rstrip().rsplit()[1:]

                for key in general_labels.keys():
                    try:
                        data_info[key].append(self._get_col_from_file(lines=spec_lines,start_row=data_start,end_row=data_start+j,col=data_item_labels.index(general_labels[key]),type=float))
                    except:
                        data_info[key].append([])

                for key in correction_labels.keys():
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
                img_number=_number_to_string(5,j)+img_extention
                temp_img_container.append(os.path.join(os.path.join(image_head,scan_number_str),"_".join([spec_name,scan_number_str,img_number])))
            data_info["images_path"].append(temp_img_container)

        return data_info

    def combine_spec_image_info(self):
        data_info=self.sort_spec_file(spec_path=self.spec_path,spec_name=self.spec_name,scan_number=self.scan_number,general_labels=self.general_labels,angle_labels=self.angle_labels,angle_labels_escan=self.angle_labels_escan,G_labels=self.G_labels)
        data_info=self.match_images(data_info,self.img_extention)
        self.data_info=data_info
        return None
    def integrate_one_image(self,img_path="S3_Zr_100mM_KCl_3_S136_0000.tiff",plot_live=PLOT_LIVE):
        cutoff_scale=INTEG_PARS['cutoff_scale']
        use_scale=INTEG_PARS['use_scale']
        center_pix=INTEG_PARS['center_pix']
        r_width=INTEG_PARS['r_width']
        c_width=INTEG_PARS['c_width']
        integration_direction=INTEG_PARS['integration_direction']
        ord_cus_s=INTEG_PARS['ord_cus_s']
        ss=INTEG_PARS['ss']
        fct=INTEG_PARS['fct']
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
        if INTEG_PARS['integration_direction']=='y':
            peak_width==INTEG_PARS['c_width']/5
        elif INTEG_PARS['integration_direction']=='x':
            peak_width==INTEG_PARS['r_width']/5
        #y=y-np.average(list(y[0:brg_width])+list(y[-brg_width:-1]))
        def _cal_FOM(y,z,peak_width):
            ct=len(y)/2
            lf=ct-peak_width
            rt=ct+peak_width
            sum_temp=np.sum(np.abs(y[rt:-1]-z[rt:-1]))+np.sum(np.abs(y[rt:-1]-z[rt:-1]))
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

    def batch_image_integration(self):
        data_info=self.data_info
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
                print 'processing scan',str(scan_number[i]),'image',images_temp.index(image),"at core ",rank
                I,I_bgr,I_err,s,ord_cus,center_pix,peak_width,r_width,c_width=self.integrate_one_image(image,plot_live=False)
                I_temp.append(I)
                I_bgr_temp.append(I_bgr)
                I_err_temp.append(I_err)
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
        self.data_info=data_info

data=data_integration_mpi(spec_path=spec_path,spec_name=spec_name,scan_number=scan_number)
data_info_temp_temp=data.data_info
#comm.Barrier()

data_info_temp=comm.gather(data_info_temp_temp,root=0)
if rank==0:
    data_info_final={}
    image_type_labels=["H","K","L","chi","mu","nu","eta","del","phi","time","norm","images_path","I","Ierr","F","Ferr","E","s","ord_cus","center_pix","r_width","c_width","peak_width","Ibgr","ctot","transmision","beta","alpha"]
    scan_type_labels=["row_number_range","or0","or1","n_azt","cell","scan_type","lambda","scan_number"]#make sure the scan_number is at the end
    spec_type_labels=["spec_path","col_label"]
    #Initialisation final data info, lib with keys of [] values
    for key in data_info_temp[0].keys():
            data_info_final[key]=[]
    #loop over each sub dataset and extract info to data_info_final
    for i in range(len(data_info_temp)):
        for key in image_type_labels+scan_type_labels:
            if key in image_type_labels:
                for jj in range(len(data_info_temp[i]['scan_number'])):
                    scan_temp=data_info_temp[i]['scan_number'][jj]#scan number
                    images_temp=data_info_temp[i][key][jj]#image-like arrays
                    if scan_temp in data_info_final["scan_number"]:#if already exist, then append results
                        append_index=data_info_final["scan_number"].index(scan_temp)#where to append
                        data_info_final[key][append_index]=data_info_final[key][append_index]+images_temp
                    else:#if not exist then just append it to empty []
                        data_info_final[key].append(data_info_temp[i][key][jj])
            elif key in scan_type_labels:#scan_typle_label has single item for each scan
                for jj in range(len(data_info_temp[i]['scan_number'])):
                    scan_temp=data_info_temp[i]['scan_number'][jj]
                    images_temp=data_info_temp[i][key][jj]
                    if scan_temp in data_info_final["scan_number"]:
                        append_index=data_info_final["scan_number"].index(scan_temp)
                        if key!="row_number_range":#if already exist and not row_number_range, then doing nothing
                            pass
                        else:#otherwise update the row_number_range (firt of original and last of current list)
                            data_info_final[key][append_index]=[data_info_final[key][append_index][0],images_temp[1]]
                    else:
                        data_info_final[key].append(data_info_temp[i][key][jj])

    #spec type has single value for the whole dataset
    for key in spec_type_labels:
        data_info_final[key]=data_info_temp[0][key]

    pickle.dump(data_info_final,open("./data_info_mpi.dump","wb"))
    print "MPI run is completed now!"
