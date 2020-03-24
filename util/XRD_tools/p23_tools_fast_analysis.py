import numpy as np
from PyMca5.PyMcaIO import EdfFile
from nexusformat.nexus import *
from glob import glob
import os
import scipy.optimize as opt
import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import matplotlib as mat
from scipy.optimize import leastsq
import matplotlib.patches as patches
import bg_subtraction as bg
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from numpy.random import rand
from matplotlib.backends.backend_agg import FigureCanvasAgg

CHECK=0
check_image_number=0
save_edf=False
make_movie=True
center_pix=[640,220]
width,length=(150,380)
#Note at the beamline the detector is aligned such that the longer edge is vertical, while the shorter edge is horizontal
top_hor,bottom_hor,left_hor,right_hor=center_pix[1]-length/2,center_pix[1]+length/2,\
                                      center_pix[0]-width/2,center_pix[0]+width/2
top_ver,bottom_ver,left_ver,right_ver=center_pix[1]-width/2,center_pix[1]+width/2,\
                                      center_pix[0]-length/2,center_pix[0]+length/2
if make_movie:
    ref_img=None
    f = plt.figure(frameon=True, figsize=(12, 4), dpi=129)
    canvas_width, canvas_height = f.canvas.get_width_height()
    ax_movie = f.add_axes([0, 0, 1, 1])
    # ax_movie.axis('off')
    rect1=patches.Rectangle((left_hor,top_hor),right_hor-left_hor,bottom_hor-top_hor,linewidth=1,edgecolor='r',facecolor='none')
    rect2=patches.Rectangle((left_ver,top_ver),right_ver-left_ver,bottom_ver-top_ver,linewidth=1,edgecolor='g',facecolor='none')
    ax_movie.add_patch(rect1)
    ax_movie.add_patch(rect2)
    im = ax_movie.imshow(rand(516,1556),cmap='jet')
    # im.set_clim([0,1])
    # Open an ffmpeg process
    outf = 'ffmpeg.mp4'
    cmdstring = ('ffmpeg',
            '-y', '-r', '10', # overwrite, 30fps
    '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
    '-pix_fmt', 'argb', # format
    '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
    '-vcodec', 'mpeg4', outf) # output encoding
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
    agg=f.canvas.switch_backends(FigureCanvasAgg)

#set fio_file to None for continued image collection mode for potential step experiment
fio_file='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/startup/FirstTest_00666.fio'
#fio_file='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/potstep/potstep_00107.fio'
# fio_file=None
#also give the elapsed time and potential_step value during potential step
time_elapse=100
potential_step=0.6
#folder hodling the nex files
nexfile_folder='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/FirstTest_00666/lmbd'
save_fig_header='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/FirstTest_00666/FirstTest_00666'
potstep_number='00037'
# nexfile_folder='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/potstep'
# save_fig_header='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/potstep/potstep_{}'.format(potstep_number)
#nexfile pattern, can be either one file (simply give the name) or a collection of files(then use *.nxs)
nexfile_pattern='potstep_{}.nxs'.format(potstep_number)
if fio_file!=None:
    nexfile_pattern='*.nxs'

cwd=os.getcwd()
#Gaussian function
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
#Extract potential data
def extract_potential(num_header_lines=168,col_pot=10,comment='!'):
    if fio_file!=None:#cv experiment, potential data saved in fio file
        data=np.loadtxt(fname=os.path.join(nexfile_folder,fio_file),skiprows=num_header_lines,comments=comment)
        return data[:,col_pot-1]
    else:#potential step experiment, potential data documented in labbook manually
        return []

def fit_gaussian_peak(img_data,plot_it=True):
    img_data=np.array(img_data)
    img_data_hor=np.sum(img_data[top_hor:bottom_hor,left_hor:right_hor],axis=1)
    img_data_ver=np.sum(img_data[top_ver:bottom_ver,left_ver:right_ver],axis=0)
    slope_hor=(np.sum(img_data_hor[-6:-1])/5.- np.sum(img_data_hor[0:5])/5.)/len(img_data_hor)
    slope_ver=(np.sum(img_data_ver[-6:-1])/5.- np.sum(img_data_ver[0:5])/5.)/len(img_data_ver)
    #do background subtraction using linear regression
    img_data_hor=img_data_hor-slope_hor*(np.array(range(len(img_data_hor))))-np.sum(img_data_hor[0:5])/5.
    img_data_ver=img_data_ver-slope_ver*(np.array(range(len(img_data_ver))))-np.sum(img_data_ver[0:5])/5.
    ver_len,hor_len=img_data.shape
    #estimate mean and standard deviation
    #do the fit!
    # x_ver=range(top_ver,bottom_ver)
    # x_hor=range(left_hor,right_hor)
    x_ver=range(left_ver,right_ver)
    x_hor=range(top_hor,bottom_hor)
    popt_hor, pcov_hor = curve_fit(gauss_function, x_hor, img_data_hor, bounds =([100,center_pix[1]-50,1],[35890,center_pix[1]+50,330]))
    popt_ver, pcov_ver = curve_fit(gauss_function,x_ver,img_data_ver, bounds =([100,center_pix[0]-50,1],[35890,center_pix[0]+50,330]))
    #plot the fit results
    if plot_it:
        mat.pyplot.figure(111)
        mat.pyplot.plot(x_hor,gauss_function(x_hor, *popt_hor))
        mat.pyplot.scatter(x_hor,img_data_hor,label='horizontal')
        mat.pyplot.legend()
        mat.pyplot.figure(222)
        mat.pyplot.plot(x_ver,gauss_function(x_ver, *popt_ver))
        mat.pyplot.scatter(x_ver,img_data_ver,label='vertical')
        mat.pyplot.legend()
        fig,ax=mat.pyplot.subplots()
        mat.pyplot.imshow(img_data,cmap='jet')
        mat.pyplot.colorbar(extend='both',orientation='horizontal')
        ax.set_xlabel('vertical direction')
        ax.set_ylabel('horizontal direction')
        roi_image=img_data[top_ver:bottom_ver,left_hor:right_hor]
        mat.pyplot.clim(roi_image.min()+10, roi_image.max()+10)
        rect1=patches.Rectangle((left_hor,top_hor),right_hor-left_hor,bottom_hor-top_hor,linewidth=1,edgecolor='r',facecolor='none')
        rect2=patches.Rectangle((left_ver,top_ver),right_ver-left_ver,bottom_ver-top_ver,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        mat.pyplot.show()
    return popt_hor,popt_ver

#make a list of nex file path
os.chdir(nexfile_folder)
nexfiles=glob(nexfile_pattern)
nexfiles=sorted(nexfiles)
nexfile_path_list=[os.path.join(nexfile_folder,each) for each in nexfiles]
#container for fit pars
peaks_width_hor=[]
peaks_pos_hor=[]
peaks_width_ver=[]
peaks_pos_ver=[]
pot_container_full=extract_potential()
pot_container_used=[]
#if you want to check first, only fit one nex file
if CHECK:
    nexfile_path_list=nexfile_path_list[0:1]
for nexfile_path in nexfile_path_list:
    print nexfile_path
    num_of_image=0
    if fio_file!=None:
        temp_pot=[pot_container_full[nexfile_path_list.index(nexfile_path)]]
    else:
        temp_pot=[potential_step]
    try:
        data=nxload(nexfile_path)
    except:
        print "loading of "+nexfile_path+"failed!"
    #get the total numbe of images in the nex container
    try:
        num_of_image=len(data.entry.instrument.detector.data.nxdata)
    except:
        num_of_image=data.entry.instrument.detector.data.shape[0]
    #do only one image if you want to check first
    # if CHECK:
        # num_of_image=1
    if CHECK:
        try:
            data_1D=data.entry.instrument.detector.data.nxdata[check_image_number]
            if save_edf:
                f=EdfFile.EdfFile(nexfile_path.replace('.nxs',str(check_image_number)+'.edf'))
                f.WriteImage({"potential":0},data_1D)
        except:
            data_1D=data.entry.instrument.detector.data[check_image_number]
            if save_edf:
                f=EdfFile.EdfFile(nexfile_path.replace('.nxs',str(check_image_number)+'.edf'))
                f.WriteImage({"potential":0},data_1D)
        print "fit frame {} now".format(check_image_number)
        #remove dead pixes first
        data_1D_filtered=(data_1D<10000)*data_1D
        try:
            hor_temp,ver_temp=fit_gaussian_peak(data_1D_filtered,plot_it=CHECK)
            peaks_width_hor.append(hor_temp[2])
            peaks_pos_hor.append(hor_temp[1])
            peaks_width_ver.append(ver_temp[2])
            peaks_pos_ver.append(ver_temp[1])
            pot_container_used=pot_container_used+temp_pot
        except:
            print "Fitting of" +nexfile_path+ "failed!"
    else:
        for i in range(num_of_image):
            try:
                data_1D=data.entry.instrument.detector.data.nxdata[i]
                if save_edf:
                    f=EdfFile.EdfFile(nexfile_path.replace('.nxs',str(i)+'.edf'))
                    f.WriteImage({"potential":0},data_1D)
            except:
                data_1D=data.entry.instrument.detector.data[i]
                if save_edf:
                    f=EdfFile.EdfFile(nexfile_path.replace('.nxs',str(i)+'.edf'))
                    f.WriteImage({"potential":0},data_1D)
            print "fit frame {} now".format(i)
            if make_movie:
                if i==0 and (nexfile_path==nexfile_path_list[0]):
                    ref_img=data_1D
                image_data=data_1D-ref_img
                im.set_data(image_data)
                roi_image=image_data[top_ver:bottom_ver,left_hor:right_hor]
                im.set_clim([roi_image.min(), roi_image.max()])
                # im.set_clim([0,100])
                # plt.draw()
                agg.draw()
                string=agg.tostring_argb()
                # write to pipe
                p.stdin.write(string)
            #remove dead pixes first
            data_1D_filtered=(data_1D<10000)*data_1D
            try:
                hor_temp,ver_temp=fit_gaussian_peak(data_1D_filtered,plot_it=CHECK)
                peaks_width_hor.append(hor_temp[2])
                peaks_pos_hor.append(hor_temp[1])
                peaks_width_ver.append(ver_temp[2])
                peaks_pos_ver.append(ver_temp[1])
                pot_container_used=pot_container_used+temp_pot
            except:
                print "Fitting of" +nexfile_path+ "failed!"
#now plot the results
#first peak width
if not CHECK:
    fig,ax1=mat.pyplot.subplots()
    ax2=ax1.twinx()
    if fio_file!=None:
        lns1=ax1.plot(pot_container_used,peaks_width_hor,'r.-',label='horizontal',markersize=6)
        lns2=ax2.plot(pot_container_used,peaks_width_ver,'b.-',label='vertical',markersize=6)
    else:
        lns1=ax1.plot(np.arange(0,time_elapse,time_elapse/float(len(peaks_width_hor))),peaks_width_hor,'r.-',label='horizontal',markersize=6)
        lns2=ax2.plot(np.arange(0,time_elapse,time_elapse/float(len(peaks_width_ver))),peaks_width_ver,'b.-',label='vertical',markersize=6)
    mat.pyplot.title('Peak width')
    if fio_file!=None:
        ax1.set_xlabel("Potential(V)")
    else:
        ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("horizontal width",color='r')
    ax2.set_ylabel("vertical width",color='b')
    lns=lns1+lns2
    labs=[l.get_label() for l in lns]
    ax1.legend(lns,labs,loc=0)

    mat.pyplot.savefig(save_fig_header+'peak_width.png',dpi=300)

    #then peak position
    fig,ax1_2=mat.pyplot.subplots()
    ax2_2=ax1_2.twinx()
    if fio_file!=None:
        lns1=ax1_2.plot(pot_container_used,peaks_pos_hor,'r.-',label='horizontal',markersize=6)
        lns2=ax2_2.plot(pot_container_used,peaks_pos_ver,'b.-',label='vertical',markersize=6)
    else:
        lns1=ax1_2.plot(np.arange(0,time_elapse,time_elapse/float(len(peaks_pos_hor))),peaks_pos_hor,'r.-',label='horizontal',markersize=6)
        lns2=ax2_2.plot(np.arange(0,time_elapse,time_elapse/float(len(peaks_pos_ver))),peaks_pos_ver,'b.-',label='vertical',markersize=6)
    mat.pyplot.title('Peak pos')
    if fio_file!=None:
        ax1_2.set_xlabel("Potential(V)")
    else:
        ax1_2.set_xlabel("Time(s)")
    ax1_2.set_ylabel("horizontal pos",color='r')
    ax2_2.set_ylabel("vertical pos",color='b')
    lns=lns1+lns2
    labs=[l.get_label() for l in lns]
    ax1_2.legend(lns,labs,loc=0)
    mat.pyplot.savefig(save_fig_header+'peak_position.png',dpi=300)
    #save data to csv file
    data_file=None
    if fio_file!=None:
        data_file=pd.DataFrame(\
            {'potential':pot_container_used,\
            'horizontal_pos':peaks_pos_hor,\
            'vertical_pos':peaks_pos_ver,\
            'horizontal_width':peaks_width_hor,\
            'vertical_width':peaks_width_ver},\
            index=range(1,len(pot_container_used)+1,1))
    else:
        data_file=pd.DataFrame(\
            {'time':np.arange(0,time_elapse,time_elapse/float(len(peaks_pos_ver))),\
            'horizontal_pos':peaks_pos_hor,\
            'vertical_pos':peaks_pos_ver,\
            'horizontal_width':peaks_width_hor,\
            'vertical_width':peaks_width_ver},\
            index=range(1,len(pot_container_used)+1,1))
    data_file.to_csv(save_fig_header+'_processed_results.csv',sep=' ',na_rep='NaN')
    mat.pyplot.show()
#change the working directory back to original
os.chdir(cwd)
print "All imaging fitting is completed!"
