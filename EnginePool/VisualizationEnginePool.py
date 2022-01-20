import sys
import matplotlib
from matplotlib import gridspec
# matplotlib.use("tkAgg")
from numpy import dtype
sys.path.append('./XRD_tools/')
import numpy as np
from scipy.interpolate import griddata
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
# from pyspec import spec
import subprocess
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FFMpegWriter
import pandas as pd
import matplotlib.patches as patches
import scipy
import pyqtgraph as pg
import time

def plot_pxrd_profile_time_scan(fig,data,image,delta_range=None,int_range=None, int_range_bkg=None,plot_final = False):
    if not plot_final:
        fig.clear()
        spec = gridspec.GridSpec(ncols = 3, nrows =2)
        ax_img = fig.add_subplot(spec[:,0])
        ax = fig.add_subplot(spec[0,1:])
        ax2 = fig.add_subplot(spec[1,1:])
        ax2_2 = ax2.twinx()
        peak_number = sum([1 for each in data if each.startswith('intensity_peak')])
        peak_intensity_list = [data['intensity_peak{}'.format(i+1)] for i in range(peak_number)]
        i=0
        for each_peak in peak_intensity_list:
            i=i+1
            ax2.plot(data['frame_number'],each_peak,label = 'peak{}'.format(i))
        ax2_2.plot(data['frame_number'],data['potential'], color = 'blue', label = 'potential')
        ax.plot(delta_range,int_range_bkg)
        ax_img.imshow(image,cmap ='jet',vmax = image.mean()*2,aspect='equal')
        #background profile(current frame)
        ax.plot(delta_range,np.array(int_range_bkg)+int_range)
        max_int = max(np.array(int_range_bkg)+int_range)
        #charateristic lines
        ax.plot([12.83,12.83],[0,max_int],'r:',label='Cu2O(111)')
        ax.plot([14.84,14.84],[0,max_int],color ='r',label='Cu2O(200)')
        ax.plot([15.17,15.17],[0,max_int],'b:',label='Cu(111)')
        ax.plot([17.53,17.53],[0,max_int],color ='b',label='Cu(200)')
        ax.plot([13.41,13.41],[0,max_int],'m:',label='Ag(111)')
        ax.plot([15.5,15.5],[0,max_int],color ='m',label='Ag(200)')

        #plot settings
        ax.set_xlabel('2theta angle')
        ax2.set_xlabel('time')
        ax.set_ylabel('Intensity')
        ax2.set_ylabel('Peak_Intensity')
        ax2_2.set_ylabel('potential(V)',color='blue')
        ax2.legend()
        ax2_2.legend()
        ax.legend()
        # ax.set_ylim([-0.05,0.3])
        plt.pause(.01)
    else:
        fig.clear()
        fig.set_size_inches(7,4)
        ax2 = fig.add_subplot(111)
        ax2_2 = ax2.twinx()
        peak_number = sum([1 for each in data if each.startswith('intensity_peak')])
        peak_intensity_list = [data['intensity_peak{}'.format(i+1)] for i in range(peak_number)]
        i=0
        for each_peak in peak_intensity_list:
            i=i+1
            ax2.plot(data['frame_number'],each_peak,label = 'peak{}'.format(i))
        ax2_2.plot(data['frame_number'],data['potential'], color = 'blue')
        #ax.plot(delta_range,int_range_bkg)
        #background profile(current frame)
        #ax.plot(delta_range,np.array(int_range_bkg)+int_range)
        #max_int = max(np.array(int_range_bkg)+int_range)

        #plot settings
        ax2.set_xlabel('time')
        ax2.set_ylabel('Peak_Intensity')
        ax2_2.set_ylabel('potential(V)',color='blue')
        ax2.legend()
        ax2_2.legend()
        #ax.legend()
        # ax.set_ylim([-0.05,0.3])
        plt.pause(10.01)
    return fig

def plot_pxrd_profile(fig,data,image,delta_range=None,int_range=None, int_range_bkg=None,plot_final = False):
    if not plot_final:
        fig.clear()
        spec = gridspec.GridSpec(ncols = 3, nrows =2)
        ax_img = fig.add_subplot(spec[:,0])
        ax = fig.add_subplot(spec[0,1:])
        ax2 = fig.add_subplot(spec[1,1:])
        # ax_img = fig.add_subplot(121)
        # ax = fig.add_subplot(222)
        # ax2 = fig.add_subplot(224)
        ax_img.imshow(image,cmap ='jet',vmax = image.mean()*2)
        #sort the data column by 2theta angle
        int_intensity_pd = pd.DataFrame(data).sort_values(by = '2theta')
        #background baseline
        ax.plot(data['2theta'],np.zeros(len(data['2theta'])))
        #intensity profile(combined data)
        ax.plot(int_intensity_pd['2theta'],int_intensity_pd['intensity'],color = '0.5')
        #intensity profile(current frame)
        ax2.plot(delta_range,int_range_bkg)
        #background profile(current frame)
        ax2.plot(delta_range,np.array(int_range_bkg)+int_range)

        max_int = 2*np.mean(np.array(int_range_bkg)+int_range)
        #charateristic lines
        ax.plot([12.83,12.83],[0,max_int],'r:',label='Cu2O(111)')
        ax.plot([14.84,14.84],[0,max_int],color ='r',label='Cu2O(200)')
        ax.plot([15.17,15.17],[0,max_int],'b:',label='Cu(111)')
        ax.plot([17.53,17.53],[0,max_int],color ='b',label='Cu(200)')
        ax.plot([13.41,13.41],[0,max_int],'m:',label='Ag(111)')
        ax.plot([15.5,15.5],[0,max_int],color ='m',label='Ag(200)')

        #plot settings
        ax.set_xlabel('2theta angle')
        ax2.set_xlabel('2theta angle')
        ax.set_ylabel('Intensity')
        ax2.set_ylabel('Intensity')
        ax.legend()
        # ax.set_ylim([-0.05,0.3])
        plt.tight_layout()
        plt.pause(.01)
    else:
        fig.clear()
        fig.set_size_inches(7,4)
        ax = fig.add_subplot(111)
        int_intensity_pd = pd.DataFrame(data).sort_values(by = '2theta')
        ax.scatter(int_intensity_pd['2theta'],int_intensity_pd['intensity'],marker='.',s=2,color = '0.55')
        #smooth the intensity
        inten_smoothed = scipy.signal.savgol_filter(int_intensity_pd['intensity'],21,2)

        #charateristic lines
        ax.plot([12.83,12.83],[0,100],'r:',label='Cu2O(111)')
        ax.plot([14.84,14.84],[0,100],color ='r',label='Cu2O(200)')
        ax.plot([15.17,15.17],[0,100],'b:',label='Cu(111)')
        ax.plot([17.53,17.53],[0,100],color ='b',label='Cu(200)')
        ax.plot([13.41,13.41],[0,100],'m:',label='Ag(111)')
        ax.plot([15.5,15.5],[0,100],color ='m',label='Ag(200)')

        #plot smoothed profile
        ax.plot(int_intensity_pd['2theta'],inten_smoothed,color = 'green',lw=2)
        ax.set_xlabel('2theta angle')
        ax.set_ylabel('Intensity')
        ax.legend()
        plt.tight_layout()
        plt.ylim([0,0.3])
        plt.xlim([12,18])
        plt.pause(10.01)
    return fig

def plot_bkg_fit_gui(ax_img, ax_profile, ax_ctr, ax_pot,data, fit_bkg_object, plot_final = False):
    ax_img.clear()
    ax_profile.clear()
    ax_ctr.clear()
    ax_pot.clear()
    z = fit_bkg_object.fit_data['y_bkg']
    n = fit_bkg_object.fit_data['x']
    y = fit_bkg_object.fit_data['y_total']
    img = fit_bkg_object.img
    x_min = fit_bkg_object.x_min
    x_max = fit_bkg_object.x_max
    y_min = fit_bkg_object.y_min
    y_max = fit_bkg_object.y_max
    y_span = fit_bkg_object.y_span
    x_span = fit_bkg_object.x_span
    clip_image_center = [int(y_span/2)+fit_bkg_object.peak_shift,int(x_span/2)+fit_bkg_object.peak_shift]
    peak_l = max([clip_image_center[int(fit_bkg_object.int_direct=='x')]-fit_bkg_object.peak_width,0])#peak_l>0
    peak_r = clip_image_center[int(fit_bkg_object.int_direct=='x')]+fit_bkg_object.peak_width
    # ax_img.imshow(img,cmap ='jet',vmax = clip_img.max())
    ax_img.imshow(img,cmap ='jet',vmax = img[y_min:y_min+y_span,x_min:x_min+x_span].max()*0.7,aspect='equal')
    rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
    rect_bkg = patches.Rectangle((fit_bkg_object.x_min_bkg,fit_bkg_object.y_min_bkg),fit_bkg_object.x_span_bkg,fit_bkg_object.y_span_bkg,linewidth=1,edgecolor='m',facecolor='none')
    ax_img.add_patch(rect)
    ax_img.add_patch(rect_bkg)
    ax_profile.plot(n,y,color='blue',label="data")
    ax_profile.plot(n,z,color="red",label="background")
    ax_profile.plot(n,y-z,color="m",label="data-background")
    ax_profile.plot(n,[0]*len(n),color='black')
    ax_profile.plot([peak_l,peak_l],[0,z[peak_l]],color = 'green')
    ax_profile.plot([peak_r,peak_r],[0,z[peak_r]],color = 'green')
    ax_pot.plot(data['image_no'],data['potential'])
    if 'L' in data:
        L_list, I_list, I_err_list = data['L'],data['peak_intensity'], data['peak_intensity_error']
        if not fit_bkg_object.rod_scan:
            L_list = data['image_no']
        # I_list = list(np.append(I_list,[I_container[index_best]]))
        # I_err_list = list(np.append(I_err_list,[Ierr_container[index_best]]))
        # ax_ctr.plot(L_list, np.array(I_list)/np.array(data['transmission']),label='CTR profile')
        ax_ctr.errorbar(np.array(L_list), np.array(I_list),yerr=np.array(I_err_list),xerr=None,fmt='ro:',markersize=4, label='CTR profile')
        ax_ctr.set_yscale('log',nonposy='clip')
        if plot_final:
            #fig2 = plt.figure(figsize=(8,7))
            ax_final = fig2.add_subplot(211)
            ax_final_pot = fig2.add_subplot(212)
            ax_final_pot.plot(data['image_no'],data['potential'])
            ax_final_pot.set_xlabel('time')
            ax_final_pot.set_ylabel('Potential')
            ax_final_pot.set_title('E (V)')
            scan_nos = list(set(data['scan_no']))
            colors = ['r','g','b','m','black','yellow']+['r','g','b','m','black','yellow']
            plot_x_list,plot_y_list,plot_err_list = [],[],[]
            for scan in scan_nos:
                index_partial_scan = data['scan_no']==scan
                plot_x_list.append(L_list[index_partial_scan])
                plot_y_list.append(I_list[index_partial_scan])
                plot_err_list.append(I_err_list[index_partial_scan])
            #ax_final.errorbar(np.array(L_list), np.array(I_list),yerr=np.array(I_err_list),xerr=None,fmt='rd-',markersize=4, label='CTR profile')
            for i in range(len(scan_nos)):
                potential_label = round(data['potential'][list(data['scan_no']).index(scan_nos[i])],1)
                ax_final.errorbar(plot_x_list[i], plot_y_list[i],yerr=plot_err_list[i],xerr=None,fmt='d-',color = colors[i], markersize=4, label='Scan{}_at {}V'.format(scan_nos[i],potential_label))
            if fit_bkg_object.rod_scan:
                ax_final.set_yscale('log',nonposy='clip')
                ax_final.set_xlabel('L')
            ax_final.set_ylabel('Itensity')
            ax_final.set_title('CTR')
            ax_final.legend()
    plt.tight_layout()
    #fig.canvas.draw()
    #fig.tight_layout()
    #plt.show()

    #return fig
def replot_bkg_profile(ax_profile, data, fit_bkg_object, plot_final = False):
    z = fit_bkg_object.fit_data['y_bkg'][:,0]
    n = fit_bkg_object.fit_data['x']
    y = fit_bkg_object.fit_data['y_total'][:,0]
    y_span = fit_bkg_object.y_span
    x_span = fit_bkg_object.x_span
    clip_image_center = [int(y_span/2)+fit_bkg_object.peak_shift,int(x_span/2)+fit_bkg_object.peak_shift]
    peak_l = max([clip_image_center[int(fit_bkg_object.int_direct=='x')]-fit_bkg_object.peak_width,0])#peak_l>0
    peak_r = clip_image_center[int(fit_bkg_object.int_direct=='x')]+fit_bkg_object.peak_width
    #ax_profile.plot(n,y,pen='b',name="data")
    ax_profile.plot(n,z,pen="r",name="background", clear = True)
    #ax_profile.plot(n,y-z,pen="m",name="data-background")
    #ax_profile.plot(n,[0]*len(n),pen='k')
    ax_profile.plot([peak_l,peak_l],[z[peak_l],y.max()],pen = 'g')
    ax_profile.plot([peak_r,peak_r],[z[peak_r],y.max()],pen = 'g')

def plot_xrv_gui_pyqtgraph(p1,p2, p3, p4, p5, p6, p7, app_ctr, x_channel_potential = False, plot_small_cut_result = True):
    p2,p2_r = p2
    p3,p3_r = p3
    p4,p4_r = p4
    data = app_ctr.data
    current_scan_number = app_ctr.current_scan_number
    index_list = np.where((np.array(data['scan_no'])==current_scan_number)&(np.array(data['mask_cv_xrd'])==True))
    img_number = np.array(data['image_no'])[index_list]
    #potential = np.array(data['potential'])[index_list]
    potential = 0.205+np.array(data['potential'])[index_list]+0.059*np.array(data['phs'])[index_list]
    single_point_handle_1b = None
    single_point_handle_2b = None

    #cut_values_oop=[peak_center[0]-cut_offset['hor'][-1],peak_center[0]+cut_offset['hor'][-1]]
    #cut_values_ip = [peak_center[1]-cut_offset['ver'][-1],peak_center[1]+cut_offset['ver'][-1]]

    if app_ctr.p2_data_source == 'vertical':
        if x_channel_potential:
            p2.plot(potential,np.array(data['strain_oop'])[index_list],clear = True)
            single_point_handle_1 = p2.plot([potential[-1]], [np.array(data['strain_oop'])[index_list][-1]], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            p2_r.clear()
            p2_r.addItem(pg.PlotCurveItem(potential,np.array(data['grain_size_oop'])[index_list], pen='g',clear = True))
            single_point_handle_2 = pg.PlotDataItem(name='name')
            p2_r.addItem(single_point_handle_2)
            single_point_handle_2.setData(x=[potential[-1]], y=[np.array(data['grain_size_oop'])[index_list][-1]],pen=None, symbolBrush=(0,100,100),symbol='o', symbolSize=10)
        else:
            p2.plot(img_number,np.array(data['strain_oop'])[index_list],clear = True)
            single_point_handle_1 = p2.plot([img_number[-1]], [np.array(data['strain_oop'])[index_list][-1]], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            p2_r.clear()
            p2_r.addItem(pg.PlotCurveItem(img_number,np.array(data['grain_size_oop'])[index_list], pen='g',clear = True))
            single_point_handle_2 = pg.PlotDataItem(name='name')
            p2_r.addItem(single_point_handle_2)
            single_point_handle_2.setData(x=[img_number[-1]], y=[np.array(data['grain_size_oop'])[index_list][-1]],pen=None, symbolBrush=(0,100,100),symbol='o', symbolSize=10)
    elif app_ctr.p2_data_source == 'horizontal':
        if x_channel_potential:
            p2.plot(potential,np.array(data['strain_ip'])[index_list],clear = True)
            single_point_handle_1 = p2.plot([potential[-1]], [np.array(data['strain_ip'])[index_list][-1]], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            p2_r.clear()
            p2_r.addItem(pg.PlotCurveItem(potential,np.array(data['grain_size_ip'])[index_list], pen='g',clear= True))
            single_point_handle_2 = pg.PlotDataItem(name='name')
            p2_r.addItem(single_point_handle_2)
            single_point_handle_2.setData(x=[potential[-1]], y=[np.array(data['grain_size_ip'])[index_list][-1]],pen=None, symbolBrush=(0,100,100),symbol='o', symbolSize=10)
            # p2_r.plot(potential[0:1], np.array(data['grain_size_ip'])[index_list][0:1], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
        else:
            p2.plot(img_number,np.array(data['strain_ip'])[index_list],clear = True)
            single_point_handle_1 = p2.plot([img_number[-1]], [np.array(data['strain_ip'])[index_list][-1]], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            p2_r.clear()
            p2_r.addItem(pg.PlotCurveItem(img_number,np.array(data['grain_size_ip'])[index_list], pen='g',clear= True))
            # p2_r.plot(img_number[0:1], np.array(data['grain_size_ip'])[index_list][0:1], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            single_point_handle_2 = pg.PlotDataItem(name='name')
            p2_r.addItem(single_point_handle_2)
            single_point_handle_2.setData(x=[img_number[-1]], y=[np.array(data['grain_size_ip'])[index_list][-1]],pen=None, symbolBrush=(0,100,100),symbol='o', symbolSize=10)
    else:
        if x_channel_potential:
            pass
        else:
            p2.plot(img_number,np.array(data['strain_oop'])[index_list],clear = True)
            single_point_handle_1 = p2.plot([img_number[-1]], [np.array(data['strain_oop'])[index_list][-1]], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            p2_r.clear()
            p2_r.addItem(pg.PlotCurveItem(img_number,np.array(data['grain_size_oop'])[index_list], pen='g',clear = True))
            single_point_handle_2 = pg.PlotDataItem(name='name')
            p2_r.addItem(single_point_handle_2)
            single_point_handle_2.setData(x=[img_number[-1]], y=[np.array(data['grain_size_oop'])[index_list][-1]],pen=None, symbolBrush=(0,100,100),symbol='o', symbolSize=10)

            p2.plot(img_number,np.array(data['strain_ip'])[index_list],clear = False)
            single_point_handle_1b = p2.plot([img_number[-1]], [np.array(data['strain_ip'])[index_list][-1]], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            #p2_r.clear()
            p2_r.addItem(pg.PlotCurveItem(img_number,np.array(data['grain_size_ip'])[index_list], pen='g',clear= True))
            # p2_r.plot(img_number[0:1], np.array(data['grain_size_ip'])[index_list][0:1], pen=None, symbolBrush=(200,200,0), symbolPen='w', symbol='o', symbolSize=10)
            single_point_handle_2b = pg.PlotDataItem(name='name2')
            p2_r.addItem(single_point_handle_2b)
            single_point_handle_2b.setData(x=[img_number[-1]], y=[np.array(data['grain_size_ip'])[index_list][-1]],pen=None, symbolBrush=(0,100,100),symbol='o', symbolSize=10)

    # if app_ctr.p3_data_source == 'peak_intensity':
    if x_channel_potential:
        p3.plot(potential,np.array(data['peak_intensity'])[index_list],clear = True)
        p3_r.clear()
        p3_r.addItem(pg.PlotCurveItem(potential,np.array(data['bkg'])[index_list], pen='g',clear = True))
    else:
        p3.plot(img_number,np.array(data['peak_intensity'])[index_list],clear = True)
        p3_r.clear()
        p3_r.addItem(pg.PlotCurveItem(img_number,np.array(data['bkg'])[index_list], pen='g',clear = True))
    # elif app_ctr.p3_data_source == 'bkg_intensity':
        # p3.plot(img_number,np.array(data['bkg'])[index_list],clear = True)

    # if app_ctr.p4_data_source == 'current':
    p4.plot(img_number,0.205+np.array(data['potential'])[index_list]+0.059*np.array(data['phs'])[index_list],clear = True)
    p4_r.clear()
    p4_r.addItem(pg.PlotCurveItem(img_number, np.array(data['current'])[index_list], pen='g',clear = True))
    # elif app_ctr.p4_data_source == 'potential':
        # p4.plot(img_number,np.array(data['potential'])[index_list],clear = True)
    index_to_plot = None
    _peak_fit_values = {'hor':None, 'ver':None}
    if plot_small_cut_result:
        index_to_plot = 1
        _peak_fit_values['hor'] = app_ctr.peak_fitting_instance.fit_results_plot['hor'][0] 
        _peak_fit_values['ver'] = app_ctr.peak_fitting_instance.fit_results_plot['ver'][0] 
    else:
        index_to_plot = 0
        _peak_fit_values['hor'] = app_ctr.peak_fitting_instance.fit_results_plot_0['hor'][0]
        _peak_fit_values['ver'] = app_ctr.peak_fitting_instance.fit_results_plot_0['ver'][0]
    # p5.plot(app_ctr.peak_fitting_instance.fit_data['hor']['x'][-1],app_ctr.peak_fitting_instance.fit_data['hor']['y'][-1], clear = True)
    # p5.plot(app_ctr.peak_fitting_instance.fit_data['hor']['x'][0],app_ctr.model(app_ctr.peak_fitting_instance.fit_data['hor']['x'][0],*app_ctr.peak_fitting_instance.fit_results_plot['hor'][0]),pen = 'y')

    p5.plot(app_ctr.peak_fitting_instance.fit_data['hor']['x'][index_to_plot],app_ctr.peak_fitting_instance.fit_data['hor']['y'][index_to_plot], clear = True)
    p5.plot(app_ctr.peak_fitting_instance.fit_data['hor']['x'][index_to_plot],app_ctr.model(app_ctr.peak_fitting_instance.fit_data['hor']['x'][index_to_plot],*_peak_fit_values['hor']),pen = 'y')
    q_boundary_ip = [app_ctr.peak_fitting_instance.fit_data['hor']['x'][0].min(), app_ctr.peak_fitting_instance.fit_data['hor']['x'][0].max(),app_ctr.peak_fitting_instance.fit_data['hor']['x'][-1].min(), app_ctr.peak_fitting_instance.fit_data['hor']['x'][-1].max()]
    intensity_boundary_ip = [app_ctr.peak_fitting_instance.fit_data['hor']['y'][-1].min(), app_ctr.peak_fitting_instance.fit_data['hor']['y'][-1].max()]
    for i in range(len(q_boundary_ip)):
        if i in [0,1]:
            color = 'k'
        else:
            color = 'r'
        p5.plot([q_boundary_ip[i],q_boundary_ip[i]],intensity_boundary_ip,pen = color)
    # p6.plot(app_ctr.peak_fitting_instance.fit_data['ver']['x'][-1],app_ctr.peak_fitting_instance.fit_data['ver']['y'][-1], clear = True)
    # p6.plot(app_ctr.peak_fitting_instance.fit_data['ver']['x'][0],app_ctr.model(app_ctr.peak_fitting_instance.fit_data['ver']['x'][0],*app_ctr.peak_fitting_instance.fit_results_plot['ver'][0]),pen = 'y')

    p6.plot(app_ctr.peak_fitting_instance.fit_data['ver']['x'][index_to_plot],app_ctr.peak_fitting_instance.fit_data['ver']['y'][index_to_plot], clear = True)
    p6.plot(app_ctr.peak_fitting_instance.fit_data['ver']['x'][index_to_plot],app_ctr.model(app_ctr.peak_fitting_instance.fit_data['ver']['x'][index_to_plot],*_peak_fit_values['ver']),pen = 'y')
    q_boundary_oop = [app_ctr.peak_fitting_instance.fit_data['ver']['x'][0].min(), app_ctr.peak_fitting_instance.fit_data['ver']['x'][0].max(),app_ctr.peak_fitting_instance.fit_data['ver']['x'][-1].min(), app_ctr.peak_fitting_instance.fit_data['ver']['x'][-1].max()]
    intensity_boundary_oop = [app_ctr.peak_fitting_instance.fit_data['ver']['y'][-1].min(), app_ctr.peak_fitting_instance.fit_data['ver']['y'][-1].max()]
    for i in range(len(q_boundary_oop)):
        if i in [0,1]:
            color = 'k'
        else:
            color = 'r'
        p6.plot([q_boundary_oop[i],q_boundary_oop[i]],intensity_boundary_oop,pen = color)

    #p7.plot(np.array(data['potential'])[index_list],np.array(data['current'])[index_list],clear = True)

    # p7.plot(list(app_ctr.bkg_sub.fit_data['x']),list(app_ctr.bkg_sub.fit_data['y_total'][:,0]),pen='w',clear = True)
    # p7.plot(list(app_ctr.bkg_sub.fit_data['x']),list(app_ctr.bkg_sub.fit_data['y_bkg'][:,0]),pen='r')
    p7.plot(list(app_ctr.bkg_sub.fit_data['x']),list(app_ctr.bkg_sub.fit_data['y_total']),pen='w',clear = True)
    p7.plot(list(app_ctr.bkg_sub.fit_data['x']),list(app_ctr.bkg_sub.fit_data['y_bkg']),pen='r')
    if single_point_handle_1b == None:
        return single_point_handle_1, single_point_handle_2
    else:
        return single_point_handle_1,single_point_handle_1b, single_point_handle_2,single_point_handle_2b


def plot_pxrd_fit_gui_pyqtgraph(ax_profile, ax_ctr, ax_strain, ax_pot,app_ctr):

    ax_profile_mon,ax_profile = ax_profile
    if not app_ctr.time_scan:
        ax_profile.plot(app_ctr.delta_range,app_ctr.int_range_bkg,pen="r",name="background",clear = True)
        ax_profile.plot(app_ctr.delta_range,np.array(app_ctr.int_range) + np.array(app_ctr.int_range_bkg),pen="w")

        #ax_profile_mon.plot(app_ctr.delta_range,app_ctr.int_range_bkg,pen="r",name="background",clear = True)
        #ax_profile_mon.plot(app_ctr.delta_range,np.array(app_ctr.int_range) + np.array(app_ctr.int_range_bkg),pen="w")

        #print(np.array(data['image_no'])[plot_index].shape)
        ax_ctr.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.data[app_ctr.img_loader.scan_number]['intensity'],clear = True)
        ax_profile_mon.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.data[app_ctr.img_loader.scan_number]['intensity'],clear = True)
        ax_pot.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.data[app_ctr.img_loader.scan_number]['potential'])
    else:
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.int_range_bkg,pen="r",name="background",clear = True)
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range) + np.array(app_ctr.int_range_bkg),pen="w")
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range),pen="w")
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range)*0,pen="g")

        ax_profile_mon.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.int_range_bkg,pen="r",name="background",clear = True)
        ax_profile_mon.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range) + np.array(app_ctr.int_range_bkg),pen="w")
        ax_profile_mon.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range),pen="w",clear = True)
        ax_profile_mon.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range)*0,pen="g")

        max_int = max(np.array(app_ctr.int_range))
        y_values = [0,max_int]
        for i in range(len(app_ctr.kwarg_peak_fit['peak_ranges'])):
            line_segment = app_ctr.kwarg_peak_fit['peak_ranges'][i]
            left, right =[line_segment[0]]*2,[line_segment[1]]*2
            ax_profile.plot(left,y_values,pen=app_ctr.kwarg_peak_fit['colors'][i])
            ax_profile.plot(right,y_values,pen=app_ctr.kwarg_peak_fit['colors'][i])
            ax_profile_mon.plot(left,y_values,pen=app_ctr.kwarg_peak_fit['colors'][i])
            ax_profile_mon.plot(right,y_values,pen=app_ctr.kwarg_peak_fit['colors'][i])
            if app_ctr.kwarg_peak_fit['peak_fit'][i]:
                x = np.arange(line_segment[0]-0.2, line_segment[1]+0.2,0.01)
                par_labels = ['_peak_pos','_FWHM','_amp','_lfrac','_bg_slope','_bg_offset']
                pars=[app_ctr.data[app_ctr.img_loader.scan_number][app_ctr.kwarg_peak_fit['peak_ids'][i]+par_label][-1] for par_label in par_labels]
                try:
                    y = app_ctr.model(x,*pars)
                except:
                    y = x
                #print(x,y)
                ax_profile.plot(x,y,pen=app_ctr.kwarg_peak_fit['colors'][i])
                ax_profile_mon.plot(x,y,pen=app_ctr.kwarg_peak_fit['colors'][i])
        clear = True
        for i in range(len(app_ctr.kwarg_peak_fit['peak_ranges'])):
            if app_ctr.kwarg_peak_fit['peak_fit'][i]:
                #y = app_ctr.data[app_ctr.img_loader.scan_number][app_ctr.kwarg_peak_fit['peak_ids'][i]+'_peak_pos']
                y = app_ctr.data[app_ctr.img_loader.scan_number][app_ctr.kwarg_peak_fit['peak_ids'][i]+'_{}'.format(app_ctr.p4_data_source)]
                ax_strain.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],y,pen=app_ctr.kwarg_peak_fit['colors'][i],name = app_ctr.kwarg_peak_fit['peak_ids'][i],clear=clear)
                clear = False
        clear = True
        if app_ctr.p3_data_source == 'peak_intensity':
            for i in range(len(app_ctr.kwarg_peak_fit['peak_ids'])):
                if app_ctr.kwarg_peak_fit['peak_fit'][i]:
                    ax_ctr.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],app_ctr.data[app_ctr.img_loader.scan_number][app_ctr.kwarg_peak_fit['peak_ids'][i]+'_intensity'],name=app_ctr.kwarg_peak_fit['peak_ids'][i]+'_intensity',pen=app_ctr.kwarg_peak_fit['colors'][i],clear = clear)
                    clear = False
        elif app_ctr.p3_data_source == 'bkg_intensity':
            ax_ctr.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],app_ctr.data[app_ctr.img_loader.scan_number]['bkg'],name='bkg_intensity',pen='g',clear = True)

        #ax_ctr.addLegend()
        #ax_pot.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],app_ctr.data[app_ctr.img_loader.scan_number]['potential'],clear = True)
        ax_pot.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],app_ctr.data[app_ctr.img_loader.scan_number][app_ctr.p5_data_source],clear = True)

def plot_pxrd_fit_gui_pyqtgraph_old(ax_profile, ax_ctr, ax_pot,app_ctr):
    if not app_ctr.time_scan:
        ax_profile.plot(app_ctr.delta_range,app_ctr.int_range_bkg,pen="r",name="background",clear = True)
        ax_profile.plot(app_ctr.delta_range,np.array(app_ctr.int_range) + np.array(app_ctr.int_range_bkg),pen="w")
        #print(np.array(data['image_no'])[plot_index].shape)
        ax_ctr.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.data[app_ctr.img_loader.scan_number]['intensity'],clear = True)
        ax_pot.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.data[app_ctr.img_loader.scan_number]['potential'])
    else:
        colors=['b','r','y']
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],app_ctr.int_range_bkg,pen="r",name="background",clear = True)
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range) + np.array(app_ctr.int_range_bkg),pen="w")
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range),pen="w")
        ax_profile.plot(app_ctr.data[app_ctr.img_loader.scan_number]['2theta'],np.array(app_ctr.int_range)*0,pen="g")
        max_int = max(np.array(app_ctr.int_range))
        y_values = [0,max_int]
        for i in range(len(app_ctr.kwarg_peak_fit['peak_ranges'])):
            line_segment = app_ctr.kwarg_peak_fit['peak_ranges'][i]
            left, right =[line_segment[0]]*2,[line_segment[1]]*2
            ax_profile.plot(left,y_values,pen=colors[i])
            ax_profile.plot(right,y_values,pen=colors[i])

        #print(np.array(data['image_no'])[plot_index].shape)
        
        for i in range(len(app_ctr.kwarg_peak_fit['peak_ids'])):
            ax_ctr.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],app_ctr.data[app_ctr.img_loader.scan_number][app_ctr.kwarg_peak_fit['peak_ids'][i]+'_intensity'],name=app_ctr.kwarg_peak_fit['peak_ids'][i]+'_intensity',pen=colors[i],clear = i==0)
        
        #ax_ctr.addLegend()
        ax_pot.plot(app_ctr.data[app_ctr.img_loader.scan_number]['frame_number'],app_ctr.data[app_ctr.img_loader.scan_number]['potential'])

def plot_bkg_fit_gui_pyqtgraph(ax_profile, ax_ctr, ax_pot,app_ctr , processed_frame_index = -1):
    data = app_ctr.data
    fit_bkg_object = app_ctr.bkg_sub
    plot_x_channel = fit_bkg_object.plot_x_channel
    try:
        z = fit_bkg_object.fit_data['y_bkg'][:,0]
    except:
        z = fit_bkg_object.fit_data['y_bkg']
    n = fit_bkg_object.fit_data['x']
    try:
        y = fit_bkg_object.fit_data['y_total'][:,0]
    except:
        y = fit_bkg_object.fit_data['y_total']

    y_span = fit_bkg_object.y_span
    x_span = fit_bkg_object.x_span
    clip_image_center = [int(y_span/2)+fit_bkg_object.peak_shift,int(x_span/2)+fit_bkg_object.peak_shift]
    peak_l = max([clip_image_center[int(fit_bkg_object.int_direct=='x')]-fit_bkg_object.peak_width,0])#peak_l>0
    peak_r = min(clip_image_center[int(fit_bkg_object.int_direct=='x')]+fit_bkg_object.peak_width,len(z)-1)
    #print([peak_l,peak_r],)
    ax_profile.plot(n,y,pen='w',name="data",clear = True)
    ax_profile.plot(n,z,pen="r",name="background")
    # ax_profile.plot(n,y,pen="m",name="signal")
    #ax_profile.plot(n,y-z,pen="m",name="data-background")
    #ax_profile.plot(n,[0]*len(n),pen='k')
    ax_profile.plot([peak_l,peak_l],[z[peak_l],y.max()],pen = 'g')
    ax_profile.plot([peak_r,peak_r],[z[peak_r],y.max()],pen = 'g')
    #plot_index = []
    #for i in range(len(data['mask_ctr'])):
    #    if data['mask_ctr'][i]==True:
    #        plot_index.append(i)
    #plot_index = [i for i in range(len(data['mask_ctr'])) if data['mask_ctr'][i]==True]
    i=len(data['mask_ctr'])-1
    current_HK = [int(round(data['H'][-1],0)),int(round(data['K'][-1],0))]
    plot_index = [len(data['mask_ctr'])-1]
    #only extract the datapoints for current rod
    while 1:
        i=i-1
        if i<0:
            break
        else:
            if ((int(round(data['H'][i],0))==current_HK[0]) and (int(round(data['K'][i],0))==current_HK[1]) and (data['mask_ctr'][i]==True)):
                plot_index.append(i)
            elif ((int(round(data['H'][i],0))==current_HK[0]) and (int(round(data['K'][i],0))==current_HK[1]) and (data['mask_ctr'][i]==False)):
                pass
            else:
                break
    plot_index = plot_index[::-1]

    imge_no = [data['image_no'][i] for i in plot_index]
    L_list = [data['L'][i] for i in plot_index]
    if plot_x_channel != None:
        if plot_x_channel in data:
            X_list = [data[plot_x_channel][i] for i in plot_index]
        else:
            raise Exception('Wrong channel label for X_list!')
    else:
        X_list = None
    """
    offset_L = 0
    L_list = []
    for i in plot_index:
        if i>0:
            #if data['L'][i]<(data['L'][i-1]+0.1):
            if [np.round(data['H'][i],0),np.round(data['K'][i],0)]!=[np.round(data['H'][i-1],0),np.round(data['K'][i-1],0)]:
                offset_L = data['L'][i-1]+offset_L
            else:
                pass
        else:
            pass
        L_list.append(data['L'][i]+offset_L)
    """
    L_list = np.array(L_list)
    try:
        potential = [data['potential'][i] for i in plot_index]
        current = [data['current'][i] for i in plot_index]
    except:
        potential, current = [], []
    peak_intensity = np.array([data['peak_intensity'][i] for i in plot_index])
    peak_intensity_error = np.array([data['peak_intensity_error'][i] for i in plot_index])
    bkg_intensity = np.array([data['bkg'][i] for i in plot_index])

    #print(np.array(data['image_no'])[plot_index].shape)
    if app_ctr.p4_data_source == 'potential':
        if len(potential)==0:
            pass
        else:
            ax_pot.plot(imge_no,potential,clear = True)
    elif app_ctr.p4_data_source == 'current':
        if len(current)==0:
            pass
        else:
            ax_pot.plot(imge_no,current,clear = True)
    #L_list, I_list, I_err_list = np.array(data['L'])[plot_index],np.array(data['peak_intensity'])[plot_index], np.array(data['peak_intensity_error'])[plot_index]
    if not fit_bkg_object.rod_scan:
        if X_list==None:
            ax_ctr.setLabel('bottom','frame_number')
            if app_ctr.p3_data_source == 'peak_intensity':
                ax_ctr.plot(imge_no, peak_intensity,pen={'color': 'y', 'width': 1}, symbolBrush=(255,0,0), symbolSize=5,symbolPen='w', clear = True)
                ax_ctr.setLogMode(x=False,y=False)
            elif app_ctr.p3_data_source == 'bkg_intensity':
                ax_ctr.plot(imge_no, bkg_intensity,pen={'color': 'g', 'width': 1}, symbolBrush=(255,0,0), symbolSize=5,symbolPen='w', clear = True)
                ax_ctr.setLogMode(x=False,y=False)
        else:
            ax_ctr.setLabel('bottom',plot_x_channel)
            if app_ctr.p3_data_source == 'peak_intensity':
                ax_ctr.plot(X_list, peak_intensity,pen={'color': 'y', 'width': 1}, symbolBrush=(255,0,0), symbolSize=5,symbolPen='w', clear = True)
                ax_ctr.setLogMode(x=False,y=False)
            elif app_ctr.p3_data_source == 'bkg_intensity':
                ax_ctr.plot(X_list, bkg_intensity,pen={'color': 'g', 'width': 1}, symbolBrush=(255,0,0), symbolSize=5,symbolPen='w', clear = True)
                ax_ctr.setLogMode(x=False,y=False)
    else:
        ax_ctr.setLabel('bottom','L')
        if app_ctr.p3_data_source == 'peak_intensity':
            ax_ctr.plot(L_list, peak_intensity,pen={'color': 'y', 'width': 1},  symbolBrush=(255,0,0), symbolSize=5,symbolPen='w',clear = True)
            ax_ctr.plot([L_list[processed_frame_index]], [peak_intensity[processed_frame_index]],pen={'color': 'y', 'width': 1},  symbolBrush=(0,255,0), symbolSize=8,symbolPen='r',clear = False)
            #draw error bars
            """
            x = np.append(L_list[:,np.newaxis],L_list[:,np.newaxis],axis = 1)
            y_d = peak_intensity[:,np.newaxis]-peak_intensity_error[:,np.newaxis]/2
            y_u = peak_intensity[:,np.newaxis]+peak_intensity_error[:,np.newaxis]/2
            y = np.append(y_d,y_u,axis=1)
            for ii in range(len(y)):
                ax_ctr.plot(x=x[ii],y=y[ii],pen={'color':'w', 'width':1},clear = False)
            """
            ax_ctr.setLogMode(x=False,y=True)
            ax_ctr.setTitle("{}{}L".format(*current_HK))
        elif app_ctr.p3_data_source == 'bkg_intensity':
            ax_ctr.plot(L_list, bkg_intensity,pen={'color': 'g', 'width': 1},  symbolBrush=(255,0,0), symbolSize=5,symbolPen='w',clear = True)
            ax_ctr.setLogMode(x=False,y=True)
        else:
            ax_ctr.plot(L_list, np.array([data[app_ctr.p3_data_source][i] for i in plot_index]),pen={'color': 'g', 'width': 1},  symbolBrush=(255,0,0), symbolSize=5,symbolPen='w',clear = True)
            ax_ctr.setLogMode(x=False,y=True)

def plot_bkg_fit_gui_pyqtgraph_old(ax_profile, ax_ctr, ax_pot,data, fit_bkg_object, plot_final = False):
    z = fit_bkg_object.fit_data['y_bkg'][:,0]
    n = fit_bkg_object.fit_data['x']
    y = fit_bkg_object.fit_data['y_total'][:,0]
    y_span = fit_bkg_object.y_span
    x_span = fit_bkg_object.x_span
    clip_image_center = [int(y_span/2)+fit_bkg_object.peak_shift,int(x_span/2)+fit_bkg_object.peak_shift]
    peak_l = max([clip_image_center[int(fit_bkg_object.int_direct=='x')]-fit_bkg_object.peak_width,0])#peak_l>0
    peak_r = clip_image_center[int(fit_bkg_object.int_direct=='x')]+fit_bkg_object.peak_width
    #ax_profile.plot(n,y,pen='b',name="data")
    ax_profile.plot(n,z,pen="r",name="background")
    #ax_profile.plot(n,y-z,pen="m",name="data-background")
    #ax_profile.plot(n,[0]*len(n),pen='k')
    ax_profile.plot([peak_l,peak_l],[z[peak_l],y.max()],pen = 'g')
    ax_profile.plot([peak_r,peak_r],[z[peak_r],y.max()],pen = 'g')
    #plot_index = []
    #for i in range(len(data['mask_ctr'])):
    #    if data['mask_ctr'][i]==True:
    #        plot_index.append(i)
    plot_index = [i for i in range(len(data['mask_ctr'])) if data['mask_ctr'][i]==True]
    imge_no = [data['image_no'][i] for i in plot_index]
    L_list = [data['L'][i] for i in plot_index]
    potential = [data['potential'][i] for i in plot_index]
    peak_intensity = [data['peak_intensity'][i] for i in plot_index]
    peak_intensity_error = [data['peak_intensity_error'][i] for i in plot_index]

    #print(np.array(data['image_no'])[plot_index].shape)
    ax_pot.plot(imge_no,potential,clear = True)
    if 'L' in data:
        #L_list, I_list, I_err_list = np.array(data['L'])[plot_index],np.array(data['peak_intensity'])[plot_index], np.array(data['peak_intensity_error'])[plot_index]
        if not fit_bkg_object.rod_scan:
            #L_list = np.array(data['image_no'])[plot_index]
            # I_list = list(np.append(I_list,[I_container[index_best]]))
            # I_err_list = list(np.append(I_err_list,[Ierr_container[index_best]]))
            # ax_ctr.plot(L_list, np.array(I_list)/np.array(data['transmission']),label='CTR profile')
            #err = pg.ErrorBarItem(x=np.array(L_list), y=np.array(I_list), top=np.array(I_err_list), bottom=np.array(I_err_list), beam=0.5)
            #ax_ctr.addItem(err)
            ax_ctr.plot(imge_no, peak_intensity,pen={'color': 'y', 'width': 1}, clear = True)
            ax_ctr.setLogMode(x=False,y=False)
        else:
            ax_ctr.plot(L_list, peak_intensity,pen={'color': 'y', 'width': 1},  symbolBrush=(255,0,0), symbolSize=5,symbolPen='w',clear = True)
            ax_ctr.setLogMode(x=False,y=True)

def plot_bkg_fit(fig,data, fit_bkg_object, plot_final = False):
    fig.clear()
    ax_img = fig.add_subplot(121)
    ax_profile = fig.add_subplot(322)
    ax_ctr = fig.add_subplot(324)
    ax_pot = fig.add_subplot(326)
    z = fit_bkg_object.fit_data['y_bkg']
    n = fit_bkg_object.fit_data['x']
    y = fit_bkg_object.fit_data['y_total']
    img = fit_bkg_object.img
    x_min = fit_bkg_object.x_min
    x_max = fit_bkg_object.x_max
    y_min = fit_bkg_object.y_min
    y_max = fit_bkg_object.y_max
    y_span = fit_bkg_object.y_span
    x_span = fit_bkg_object.x_span
    clip_image_center = [int(y_span/2)+fit_bkg_object.peak_shift,int(x_span/2)+fit_bkg_object.peak_shift]
    peak_l = max([clip_image_center[int(fit_bkg_object.int_direct=='x')]-fit_bkg_object.peak_width,0])#peak_l>0
    peak_r = clip_image_center[int(fit_bkg_object.int_direct=='x')]+fit_bkg_object.peak_width
    # ax_img.imshow(img,cmap ='jet',vmax = clip_img.max())
    ax_img.imshow(img,cmap ='jet',vmax = img[y_min:y_min+y_span,x_min:x_min+x_span].max()*0.7,aspect='equal')
    rect = patches.Rectangle((x_min,y_min),x_span,y_span,linewidth=1,edgecolor='r',facecolor='none')
    rect_bkg = patches.Rectangle((fit_bkg_object.x_min_bkg,fit_bkg_object.y_min_bkg),fit_bkg_object.x_span_bkg,fit_bkg_object.y_span_bkg,linewidth=1,edgecolor='m',facecolor='none')
    ax_img.add_patch(rect)
    ax_img.add_patch(rect_bkg)
    ax_profile.plot(n,y,color='blue',label="data")
    ax_profile.plot(n,z,color="red",label="background")
    ax_profile.plot(n,y-z,color="m",label="data-background")
    ax_profile.plot(n,[0]*len(n),color='black')
    ax_profile.plot([peak_l,peak_l],[0,z[peak_l]],color = 'green')
    ax_profile.plot([peak_r,peak_r],[0,z[peak_r]],color = 'green')
    ax_pot.plot(data['image_no'],data['potential'])
    if 'L' in data:
        L_list, I_list, I_err_list = data['L'],data['peak_intensity'], data['peak_intensity_error']
        if not fit_bkg_object.rod_scan:
            L_list = data['image_no']
        # I_list = list(np.append(I_list,[I_container[index_best]]))
        # I_err_list = list(np.append(I_err_list,[Ierr_container[index_best]]))
        # ax_ctr.plot(L_list, np.array(I_list)/np.array(data['transmission']),label='CTR profile')
        ax_ctr.errorbar(np.array(L_list), np.array(I_list),yerr=np.array(I_err_list),xerr=None,fmt='ro:',markersize=4, label='CTR profile')
        ax_ctr.set_yscale('log',nonposy='clip')
        if plot_final:
            fig2 = plt.figure(figsize=(8,7))
            ax_final = fig2.add_subplot(211)
            ax_final_pot = fig2.add_subplot(212)
            ax_final_pot.plot(data['image_no'],data['potential'])
            ax_final_pot.set_xlabel('time')
            ax_final_pot.set_ylabel('Potential')
            ax_final_pot.set_title('E (V)')
            scan_nos = list(set(data['scan_no']))
            colors = ['r','g','b','m','black','yellow']+['r','g','b','m','black','yellow']
            plot_x_list,plot_y_list,plot_err_list = [],[],[]
            for scan in scan_nos:
                index_partial_scan = data['scan_no']==scan
                plot_x_list.append(L_list[index_partial_scan])
                plot_y_list.append(I_list[index_partial_scan])
                plot_err_list.append(I_err_list[index_partial_scan])
            #ax_final.errorbar(np.array(L_list), np.array(I_list),yerr=np.array(I_err_list),xerr=None,fmt='rd-',markersize=4, label='CTR profile')
            for i in range(len(scan_nos)):
                potential_label = round(data['potential'][list(data['scan_no']).index(scan_nos[i])],1)
                ax_final.errorbar(plot_x_list[i], plot_y_list[i],yerr=plot_err_list[i],xerr=None,fmt='d-',color = colors[i], markersize=4, label='Scan{}_at {}V'.format(scan_nos[i],potential_label))
            if fit_bkg_object.rod_scan:
                ax_final.set_yscale('log',nonposy='clip')
                ax_final.set_xlabel('L')
            ax_final.set_ylabel('Itensity')
            ax_final.set_title('CTR')
            ax_final.legend()
    plt.tight_layout()
    fig.canvas.draw()
    fig.tight_layout()
    plt.show()

    return fig

def draw_lines_on_image(ax_handle,x_y_grid,variable_list,direction = 'horizontal',\
                        color='gray',line_style='-',marker = None,\
                        xlabel=r'$q_\parallel$ / $\AA^{-1}$',\
                        ylabel='$q_\perp$ / $\AA^{-1}$',\
                        fontsize=20,
                        debug = False):
    line_ax_container = []
    x_couples, y_couples = [], []
    if direction == 'horizontal':
        # print(x_y_grid[0,:][0])
        x_couples = [x_y_grid[0,:][[0,-1]]]*len(variable_list)
        y_couples = [[each,each] for each in variable_list]
    elif direction == 'vertical':
        y_couples = [x_y_grid[:,0][[0,-1]]]*len(variable_list)
        x_couples = [[each,each] for each in variable_list]
    for i in range(len(x_couples)):
        temp_line_ax = ax_handle.plot(x_couples[i],y_couples[i],line_style, color = color, marker = marker)
        # line_ax_container.append(temp_line_ax)
    if debug:
        print(x_couples,y_couples)
    ax_handle.set_xlabel(xlabel,fontsize=fontsize)
    ax_handle.set_ylabel(ylabel,fontsize=fontsize)
    return ax_handle

# def show_all_plots_new(fig,grid_intensity,grid_q_ip,grid_q_oop, vmin, vmax, cmap, is_zap_scan, fit_data, model, fit_results, processed_data_container={},bkg_int=None, cut_offset=None,peak_center=None,title = None):
def show_all_plots_new(fig,fit_engine_instance,bkg_int, processed_data_container, title, kwarg):
    for key in kwarg:
        globals()[key] = kwarg[key]
    grid_intensity = fit_engine_instance.img
    grid_q_ip = fit_engine_instance.grid_q_ip
    grid_q_oop = fit_engine_instance.grid_q_oop
    fit_data = fit_engine_instance.fit_data
    fit_results = fit_engine_instance.fit_results_plot
    cut_offset = fit_engine_instance.cut_offset
    peak_center = fit_engine_instance.peak_center
    model = fit_engine_instance.model

    fig.clear()
    if processed_data_container == {}:
        ax_im=fig.add_subplot(131)
        ax_ip=fig.add_subplot(132)
        ax_oop=fig.add_subplot(133)
    else:
        ax_im=fig.add_subplot(341)
        ax_ip=fig.add_subplot(342)
        ax_oop=fig.add_subplot(343)

        ax_cv = fig.add_subplot(345)
        ax_v = fig.add_subplot(349)
        ax_strain_ip=fig.add_subplot(346)
        ax_strain_oop=fig.add_subplot(347)
        ax_width_ip=fig.add_subplot(3,4,10)
        ax_width_oop=fig.add_subplot(3,4,11)

        ax_bkg_profile = fig.add_subplot(344)
        ax_ctr_profile_pot = fig.add_subplot(348)
        ax_ctr_profile_time = fig.add_subplot(3,4,12)

    ax_im.set_title(title)
    # ax_im.pcolormesh(grid_q_ip, grid_q_oop, grid_intensity, vmin = vmin, vmax = vmax, cmap = cmap)
    ax_im.pcolormesh(grid_q_ip, grid_q_oop, grid_intensity, vmin = vmin, vmax = grid_intensity.max()*.5, cmap = cmap)
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [fit_data['ver']['x'][0].min(),fit_data['ver']['x'][0].max()],direction = 'horizontal',color = 'gray')
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [fit_data['ver']['x'][1].min(),fit_data['ver']['x'][1].max()],direction = 'horizontal',color = 'red')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [fit_data['hor']['x'][0].min(),fit_data['hor']['x'][0].max()],direction = 'vertical',color = 'gray')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [fit_data['hor']['x'][1].min(),fit_data['hor']['x'][1].max()],direction = 'vertical',color = 'red')
    # print([fit_data['ver']['x'][0].min(),fit_data['ver']['x'][0].max()])
    cut_values_oop=[peak_center[0]-cut_offset['hor'][-1],peak_center[0]+cut_offset['hor'][-1]]
    cut_values_ip = [peak_center[1]-cut_offset['ver'][-1],peak_center[1]+cut_offset['ver'][-1]]
    # print(cut_values_oop, cut_values_ip)
    # print('sensor',peak_center,cut_values_ip, cut_values_oop)
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [grid_q_oop[each,0] for each in cut_values_oop], direction = 'horizontal',color = 'm')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [grid_q_ip[0,each] for each in cut_values_ip], direction = 'vertical',color = 'm')

    x_span =  abs(grid_q_ip[0,bkg_int.x_min+bkg_int.x_span]-grid_q_ip[0,bkg_int.x_min])
    y_span =  abs(grid_q_oop[-bkg_int.y_min-bkg_int.y_span,0]-grid_q_oop[-bkg_int.y_min,0])
    rect = patches.Rectangle((grid_q_ip[0,bkg_int.x_min],grid_q_oop[-bkg_int.y_min,0]),x_span, y_span,linewidth=1,edgecolor='g',ls='-',facecolor='none')
    ax_im.add_patch(rect)

    ax_ip.plot(fit_data['hor']['x'][-1],fit_data['hor']['y'][-1])
    ax_ip.plot(fit_data['hor']['x'][0],model(fit_data['hor']['x'][0],*fit_results['hor'][0]))
    q_boundary_ip = [fit_data['hor']['x'][0].min(), fit_data['hor']['x'][0].max(),fit_data['hor']['x'][-1].min(), fit_data['hor']['x'][-1].max()]
    intensity_boundary_ip = [fit_data['hor']['y'][-1].min(), fit_data['hor']['y'][-1].max()]
    for i in range(len(q_boundary_ip)):
        if i in [0,1]:
            color = 'gray'
        else:
            color = 'red'
        ax_ip.plot([q_boundary_ip[i],q_boundary_ip[i]],intensity_boundary_ip,color = color)
    ax_ip.set_ylim((0,fit_data['hor']['y'][-1].max()*1.2))

    ax_ip.set_xlabel(r'$q_\parallel$ / $\AA^{-1}$', fontsize=20)
    ax_ip.set_ylabel(r'Intensity / a.u.', fontsize=20)
    ax_oop.plot(fit_data['ver']['x'][-1],fit_data['ver']['y'][-1])
    ax_oop.plot(fit_data['ver']['x'][0],model(fit_data['ver']['x'][0],*fit_results['ver'][0]))
    q_boundary_oop = [fit_data['ver']['x'][0].min(), fit_data['ver']['x'][0].max(),fit_data['ver']['x'][-1].min(), fit_data['ver']['x'][-1].max()]
    intensity_boundary_oop = [fit_data['ver']['y'][-1].min(), fit_data['ver']['y'][-1].max()]
    for i in range(len(q_boundary_oop)):
        if i in [0,1]:
            color = 'gray'
        else:
            color = 'red'
        ax_oop.plot([q_boundary_oop[i],q_boundary_oop[i]],intensity_boundary_oop,color = color)
    ax_oop.set_ylim((0,fit_data['ver']['y'][-1].max()*1.2))
    ax_oop.set_xlabel(r'$q_\perp$ / $\AA^{-1}$', fontsize=20)
    ax_oop.set_ylabel(r'Intensity / a.u.', fontsize=20)

    ax_bkg_profile.plot(bkg_int.fit_data['x'],bkg_int.fit_data['y_total'],color = 'blue', label ='data')
    ax_bkg_profile.plot(bkg_int.fit_data['x'],bkg_int.fit_data['y_bkg'],color = 'red', label ='background')
    ax_bkg_profile.plot(bkg_int.fit_data['x'],bkg_int.fit_data['y_total']-bkg_int.fit_data['y_bkg'],color = 'm', label ='data-background')
    ax_bkg_profile.plot(bkg_int.fit_data['x'],[0]*len(bkg_int.fit_data['x']))

    ax_bkg_profile.set_xlabel('pixel',fontsize=20)
    ax_bkg_profile.set_ylabel('I',fontsize=20)
    ax_bkg_profile.set_title('bkg_profile')

    ax_ctr_profile_pot.set_xlabel('E(V)',fontsize=20)
    ax_ctr_profile_pot.set_ylabel('I',fontsize=20)
    ax_ctr_profile_pot.set_title('ctr_profile_L')

    ax_ctr_profile_time.set_xlabel('t',fontsize=20)
    ax_ctr_profile_time.set_ylabel('I',fontsize=20)
    ax_ctr_profile_time.set_title('CTR_profile_t')

    index_ctr = np.where(np.array(processed_data_container['mask_ctr'])==1)
    if len(index_ctr[0])==0:
        index_ctr = [0]
    ax_ctr_profile_pot.errorbar(np.array(processed_data_container['potential'])[index_ctr], np.array(processed_data_container['peak_intensity'])[index_ctr],xerr=None,yerr=np.array(processed_data_container['peak_intensity_error'])[index_ctr],fmt='ro:', markersize=4, label='CTR profile')

    ax_ctr_profile_time.errorbar(np.array(range(len(processed_data_container['L'])))[index_ctr], np.array(processed_data_container['peak_intensity'])[index_ctr],xerr=None,yerr=np.array(processed_data_container['peak_intensity_error'])[index_ctr],fmt='ro:', markersize=4, label='CTR profile')

    if pot_step:
        ax_strain_oop.plot(processed_data_container['strain_oop'],'r-.')
        ax_strain_ip.plot(processed_data_container['strain_ip'],'g-.')
        ax_width_oop.plot(processed_data_container['grain_size_oop'],'r-.')
        ax_width_ip.plot(processed_data_container['grain_size_ip'],'g-.')
        ax_cv.plot(processed_data_container['potential'], processed_data_container['current'],'k--.')
        ax_v.plot(processed_data_container['potential'],'m:d')
        ax_strain_oop.plot(len(processed_data_container['potential'])-1,processed_data_container['strain_oop'][-1],'g-o')
        ax_strain_ip.plot(len(processed_data_container['potential'])-1,processed_data_container['strain_ip'][-1],'r-o')
        ax_width_oop.plot(len(processed_data_container['potential'])-1,processed_data_container['grain_size_oop'][-1],'g-o')
        ax_width_ip.plot(len(processed_data_container['potential'])-1,processed_data_container['grain_size_ip'][-1],'r-o')
        ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current'][-1],'r--o')
        ax_v.plot(len(processed_data_container['potential'])-1,processed_data_container['potential'][-1],'k:d')
    else:
        # ax_strain_oop.plot(processed_data_container['potential'],processed_data_container['oop_strain'],'r-.')
        # ax_strain_ip.plot(processed_data_container['potential'],processed_data_container['ip_strain'],'g-.')
        # ax_width_oop.plot(processed_data_container['potential'],processed_data_container['oop_grain_size'],'r-.')
        # ax_width_ip.plot(processed_data_container['potential'],processed_data_container['ip_grain_size'],'g-.')
        # ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        # ax_v.plot(processed_data_container['potential'],'m:d')
        # ax_strain_oop.plot(processed_data_container['potential'][-1],processed_data_container['oop_strain'][-1],'g-o')
        # ax_strain_ip.plot(processed_data_container['potential'][-1],processed_data_container['ip_strain'][-1],'r-o')
        # ax_width_oop.plot(processed_data_container['potential'][-1],processed_data_container['oop_grain_size'][-1],'g-o')
        # ax_width_ip.plot(processed_data_container['potential'][-1],processed_data_container['ip_grain_size'][-1],'r-o')
        # ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')
        # index_cv = np.where(np.array(processed_data_container['mask_cv_xrd'])==1)
        # if len(index_cv[0])==0:
            # index_cv = [0]
        ax_strain_oop.plot(processed_data_container['potential'],np.array(processed_data_container['strain_oop']),'r-.')
        ax_strain_ip.plot(processed_data_container['potential'],np.array(processed_data_container['strain_ip']),'g-.')
        ax_width_oop.plot(processed_data_container['potential'],np.array(processed_data_container['grain_size_oop']),'r-.')
        ax_width_ip.plot(processed_data_container['potential'],np.array(processed_data_container['grain_size_ip']),'g-.')
        ax_cv.plot(processed_data_container['potential'], processed_data_container['current'],'k--.')
        ax_v.plot(processed_data_container['potential'],'m:d')
        ax_strain_oop.plot(processed_data_container['potential'][-1],processed_data_container['strain_oop'][-1],'g-o')
        ax_strain_ip.plot(processed_data_container['potential'][-1],processed_data_container['strain_ip'][-1],'r-o')
        ax_width_oop.plot(processed_data_container['potential'][-1],processed_data_container['grain_size_oop'][-1],'g-o')
        ax_width_ip.plot(processed_data_container['potential'][-1],processed_data_container['grain_size_ip'][-1],'r-o')
        ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current'][-1],'r--o')

    ax_v.set_xlabel('Time')
    ax_v.set_title('potential')
    ax_strain_oop.set_title('out-of-plane strain')
    ax_strain_ip.set_title('inplane strain')
    ax_width_oop.set_title('out-of-plane width')
    ax_width_ip.set_title('inplane width')
    ax_cv.set_title('CV')
    if pot_step:
        ax_cv.set_xlabel('E(V)')
        ax_strain_ip.set_xlabel('Time')
        ax_strain_oop.set_xlabel('Time')
        ax_width_ip.set_xlabel('Time')
        ax_width_oop.set_xlabel('Time')
    else:
        ax_cv.set_xlabel('E(V)')
        ax_strain_ip.set_xlabel('E(V)')
        ax_strain_oop.set_xlabel('E(V)')
        ax_width_ip.set_xlabel('E(V)')
        ax_width_oop.set_xlabel('E(V)')

    return fig

def show_all_plots(fig,grid_intensity,grid_q_ip,grid_q_oop, vmin, vmax, cmap, is_zap_scan, fit_data, model, fit_results, processed_data_container={},bkg_int=None, cut_offset=None,peak_center=None,title = None):
    fig.clear()
    if processed_data_container == {}:
        ax_im=fig.add_subplot(131)
        ax_ip=fig.add_subplot(132)
        ax_oop=fig.add_subplot(133)
    else:
        ax_im=fig.add_subplot(341)
        ax_ip=fig.add_subplot(342)
        ax_oop=fig.add_subplot(343)

        ax_cv = fig.add_subplot(345)
        ax_v = fig.add_subplot(349)
        ax_strain_ip=fig.add_subplot(346)
        ax_strain_oop=fig.add_subplot(347)
        ax_width_ip=fig.add_subplot(3,4,10)
        ax_width_oop=fig.add_subplot(3,4,11)

        ax_bkg_profile = fig.add_subplot(344)
        ax_ctr_profile_pot = fig.add_subplot(348)
        ax_ctr_profile_time = fig.add_subplot(3,4,12)

    ax_im.set_title(title)
    # ax_im.pcolormesh(grid_q_ip, grid_q_oop, grid_intensity, vmin = vmin, vmax = vmax, cmap = cmap)
    ax_im.pcolormesh(grid_q_ip, grid_q_oop, grid_intensity, vmin = vmin, vmax = grid_intensity.max()*.5, cmap = cmap)
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [fit_data['ver']['x'][0].min(),fit_data['ver']['x'][0].max()],direction = 'horizontal',color = 'gray')
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [fit_data['ver']['x'][1].min(),fit_data['ver']['x'][1].max()],direction = 'horizontal',color = 'red')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [fit_data['hor']['x'][0].min(),fit_data['hor']['x'][0].max()],direction = 'vertical',color = 'gray')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [fit_data['hor']['x'][1].min(),fit_data['hor']['x'][1].max()],direction = 'vertical',color = 'red')
    # print([fit_data['ver']['x'][0].min(),fit_data['ver']['x'][0].max()])
    cut_values_oop=[peak_center[0]-cut_offset['hor'][-1],peak_center[0]+cut_offset['hor'][-1]]
    cut_values_ip = [peak_center[1]-cut_offset['ver'][-1],peak_center[1]+cut_offset['ver'][-1]]
    # print(cut_values_oop, cut_values_ip)
    # print('sensor',peak_center,cut_values_ip, cut_values_oop)
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [grid_q_oop[each,0] for each in cut_values_oop], direction = 'horizontal',color = 'm')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [grid_q_ip[0,each] for each in cut_values_ip], direction = 'vertical',color = 'm')

    x_span =  abs(grid_q_ip[0,bkg_int.x_min+bkg_int.x_span]-grid_q_ip[0,bkg_int.x_min])
    y_span =  abs(grid_q_oop[-bkg_int.y_min-bkg_int.y_span,0]-grid_q_oop[-bkg_int.y_min,0])
    rect = patches.Rectangle((grid_q_ip[0,bkg_int.x_min],grid_q_oop[-bkg_int.y_min,0]),x_span, y_span,linewidth=1,edgecolor='g',ls='-',facecolor='none')
    ax_im.add_patch(rect)

    ax_ip.plot(fit_data['hor']['x'][-1],fit_data['hor']['y'][-1])
    ax_ip.plot(fit_data['hor']['x'][0],model(fit_data['hor']['x'][0],*fit_results['hor'][0]))
    q_boundary_ip = [fit_data['hor']['x'][0].min(), fit_data['hor']['x'][0].max(),fit_data['hor']['x'][-1].min(), fit_data['hor']['x'][-1].max()]
    intensity_boundary_ip = [fit_data['hor']['y'][-1].min(), fit_data['hor']['y'][-1].max()]
    for i in range(len(q_boundary_ip)):
        if i in [0,1]:
            color = 'gray'
        else:
            color = 'red'
        ax_ip.plot([q_boundary_ip[i],q_boundary_ip[i]],intensity_boundary_ip,color = color)
    ax_ip.set_ylim((0,fit_data['hor']['y'][-1].max()*1.2))

    ax_ip.set_xlabel(r'$q_\parallel$ / $\AA^{-1}$', fontsize=20)
    ax_ip.set_ylabel(r'Intensity / a.u.', fontsize=20)
    ax_oop.plot(fit_data['ver']['x'][-1],fit_data['ver']['y'][-1])
    ax_oop.plot(fit_data['ver']['x'][0],model(fit_data['ver']['x'][0],*fit_results['ver'][0]))
    q_boundary_oop = [fit_data['ver']['x'][0].min(), fit_data['ver']['x'][0].max(),fit_data['ver']['x'][-1].min(), fit_data['ver']['x'][-1].max()]
    intensity_boundary_oop = [fit_data['ver']['y'][-1].min(), fit_data['ver']['y'][-1].max()]
    for i in range(len(q_boundary_oop)):
        if i in [0,1]:
            color = 'gray'
        else:
            color = 'red'
        ax_oop.plot([q_boundary_oop[i],q_boundary_oop[i]],intensity_boundary_oop,color = color)
    ax_oop.set_ylim((0,fit_data['ver']['y'][-1].max()*1.2))
    ax_oop.set_xlabel(r'$q_\perp$ / $\AA^{-1}$', fontsize=20)
    ax_oop.set_ylabel(r'Intensity / a.u.', fontsize=20)

    ax_bkg_profile.plot(bkg_int.fit_data['x'],bkg_int.fit_data['y_total'],color = 'blue', label ='data')
    ax_bkg_profile.plot(bkg_int.fit_data['x'],bkg_int.fit_data['y_bkg'],color = 'red', label ='background')
    ax_bkg_profile.plot(bkg_int.fit_data['x'],bkg_int.fit_data['y_total']-bkg_int.fit_data['y_bkg'],color = 'm', label ='data-background')
    ax_bkg_profile.plot(bkg_int.fit_data['x'],[0]*len(bkg_int.fit_data['x']))

    ax_bkg_profile.set_xlabel('pixel',fontsize=20)
    ax_bkg_profile.set_ylabel('I',fontsize=20)
    ax_bkg_profile.set_title('bkg_profile')

    ax_ctr_profile_pot.set_xlabel('E(V)',fontsize=20)
    ax_ctr_profile_pot.set_ylabel('I',fontsize=20)
    ax_ctr_profile_pot.set_title('ctr_profile_L')

    ax_ctr_profile_time.set_xlabel('t',fontsize=20)
    ax_ctr_profile_time.set_ylabel('I',fontsize=20)
    ax_ctr_profile_time.set_title('CTR_profile_t')




    index_ctr = np.where(np.array(processed_data_container['mask_ctr'])==1)
    if len(index_ctr[0])==0:
        index_ctr = [0]
    ax_ctr_profile_pot.errorbar(np.array(processed_data_container['potential'])[index_ctr], np.array(processed_data_container['peak_intensity'])[index_ctr],xerr=None,yerr=np.array(processed_data_container['peak_intensity_error'])[index_ctr],fmt='ro:', markersize=4, label='CTR profile')

    ax_ctr_profile_time.errorbar(np.array(range(len(processed_data_container['L'])))[index_ctr], np.array(processed_data_container['peak_intensity'])[index_ctr],xerr=None,yerr=np.array(processed_data_container['peak_intensity_error'])[index_ctr],fmt='ro:', markersize=4, label='CTR profile')

    if is_zap_scan:
        ax_strain_oop.plot(processed_data_container['oop_strain'],'r-.')
        ax_strain_ip.plot(processed_data_container['ip_strain'],'g-.')
        ax_width_oop.plot(processed_data_container['oop_grain_size'],'r-.')
        ax_width_ip.plot(processed_data_container['ip_grain_size'],'g-.')
        ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        ax_v.plot(processed_data_container['potential'],'m:d')
        ax_strain_oop.plot(len(processed_data_container['potential'])-1,processed_data_container['oop_strain'][-1],'g-o')
        ax_strain_ip.plot(len(processed_data_container['potential'])-1,processed_data_container['ip_strain'][-1],'r-o')
        ax_width_oop.plot(len(processed_data_container['potential'])-1,processed_data_container['oop_grain_size'][-1],'g-o')
        ax_width_ip.plot(len(processed_data_container['potential'])-1,processed_data_container['ip_grain_size'][-1],'r-o')
        ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')
        ax_v.plot(len(processed_data_container['potential'])-1,processed_data_container['potential'][-1],'k:d')
    else:
        # ax_strain_oop.plot(processed_data_container['potential'],processed_data_container['oop_strain'],'r-.')
        # ax_strain_ip.plot(processed_data_container['potential'],processed_data_container['ip_strain'],'g-.')
        # ax_width_oop.plot(processed_data_container['potential'],processed_data_container['oop_grain_size'],'r-.')
        # ax_width_ip.plot(processed_data_container['potential'],processed_data_container['ip_grain_size'],'g-.')
        # ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        # ax_v.plot(processed_data_container['potential'],'m:d')
        # ax_strain_oop.plot(processed_data_container['potential'][-1],processed_data_container['oop_strain'][-1],'g-o')
        # ax_strain_ip.plot(processed_data_container['potential'][-1],processed_data_container['ip_strain'][-1],'r-o')
        # ax_width_oop.plot(processed_data_container['potential'][-1],processed_data_container['oop_grain_size'][-1],'g-o')
        # ax_width_ip.plot(processed_data_container['potential'][-1],processed_data_container['ip_grain_size'][-1],'r-o')
        # ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')
        index_cv = np.where(np.array(processed_data_container['mask_cv_xrd'])==1)
        if len(index_cv[0])==0:
            index_cv = [0]
        ax_strain_oop.plot(np.array(processed_data_container['oop_strain'])[index_cv],'r-.')
        ax_strain_ip.plot(np.array(processed_data_container['ip_strain'])[index_cv],'g-.')
        ax_width_oop.plot(np.array(processed_data_container['oop_grain_size'])[index_cv],'r-.')
        ax_width_ip.plot(np.array(processed_data_container['ip_grain_size'])[index_cv],'g-.')
        ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        ax_v.plot(processed_data_container['potential'],'m:d')
        ax_strain_oop.plot(processed_data_container['oop_strain'][-1],'g-o')
        ax_strain_ip.plot(processed_data_container['ip_strain'][-1],'r-o')
        ax_width_oop.plot(processed_data_container['oop_grain_size'][-1],'g-o')
        ax_width_ip.plot(processed_data_container['ip_grain_size'][-1],'r-o')
        ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')

    ax_v.set_xlabel('Time')
    ax_v.set_title('potential')
    ax_strain_oop.set_title('out-of-plane strain')
    ax_strain_ip.set_title('inplane strain')
    ax_width_oop.set_title('out-of-plane width')
    ax_width_ip.set_title('inplane width')
    ax_cv.set_title('CV')
    if is_zap_scan:
        ax_cv.set_xlabel('E(V)')
        ax_strain_ip.set_xlabel('Time')
        ax_strain_oop.set_xlabel('Time')
        ax_width_ip.set_xlabel('Time')
        ax_width_oop.set_xlabel('Time')
    else:
        ax_cv.set_xlabel('E(V)')
        ax_strain_ip.set_xlabel('E(V)')
        ax_strain_oop.set_xlabel('E(V)')
        ax_width_ip.set_xlabel('E(V)')
        ax_width_oop.set_xlabel('E(V)')

    return fig

def show_3_plots(fig,grid_intensity,grid_q_ip,grid_q_oop, vmin, vmax, cmap, is_zap_scan, fit_data, model, fit_results, processed_data_container={}, cut_offset=None,peak_center=None,title = None):
    fig.clear()
    if processed_data_container == {}:
        ax_im=fig.add_subplot(131)
        ax_ip=fig.add_subplot(132)
        ax_oop=fig.add_subplot(133)
    else:
        ax_im=fig.add_subplot(331)
        ax_ip=fig.add_subplot(332)
        ax_oop=fig.add_subplot(333)

        ax_cv = fig.add_subplot(334)
        ax_v = fig.add_subplot(337)
        ax_strain_ip=fig.add_subplot(335)
        ax_strain_oop=fig.add_subplot(336)
        ax_width_ip=fig.add_subplot(338)
        ax_width_oop=fig.add_subplot(339)
    ax_im.set_title(title)
    ax_im.pcolormesh(grid_q_ip, grid_q_oop, grid_intensity, vmin = vmin, vmax = vmax, cmap = cmap)
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [fit_data['ver']['x'][0].min(),fit_data['ver']['x'][0].max()],direction = 'horizontal',color = 'gray')
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [fit_data['ver']['x'][1].min(),fit_data['ver']['x'][1].max()],direction = 'horizontal',color = 'red')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [fit_data['hor']['x'][0].min(),fit_data['hor']['x'][0].max()],direction = 'vertical',color = 'gray')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [fit_data['hor']['x'][1].min(),fit_data['hor']['x'][1].max()],direction = 'vertical',color = 'red')
    # print([fit_data['ver']['x'][0].min(),fit_data['ver']['x'][0].max()])
    cut_values_oop=[peak_center[0]-cut_offset['hor'][-1],peak_center[0]+cut_offset['hor'][-1]]
    cut_values_ip = [peak_center[1]-cut_offset['ver'][-1],peak_center[1]+cut_offset['ver'][-1]]
    # print(cut_values_oop, cut_values_ip)
    # print('sensor',peak_center,cut_values_ip, cut_values_oop)
    ax_im = draw_lines_on_image(ax_im, grid_q_ip, variable_list = [grid_q_oop[each,0] for each in cut_values_oop], direction = 'horizontal',color = 'm')
    ax_im = draw_lines_on_image(ax_im, grid_q_oop, variable_list = [grid_q_ip[0,each] for each in cut_values_ip], direction = 'vertical',color = 'm')

    ax_ip.plot(fit_data['hor']['x'][-1],fit_data['hor']['y'][-1])
    ax_ip.plot(fit_data['hor']['x'][0],model(fit_data['hor']['x'][0],*fit_results['hor'][0]))
    q_boundary_ip = [fit_data['hor']['x'][0].min(), fit_data['hor']['x'][0].max(),fit_data['hor']['x'][-1].min(), fit_data['hor']['x'][-1].max()]
    intensity_boundary_ip = [fit_data['hor']['y'][-1].min(), fit_data['hor']['y'][-1].max()]
    for i in range(len(q_boundary_ip)):
        if i in [0,1]:
            color = 'gray'
        else:
            color = 'red'
        ax_ip.plot([q_boundary_ip[i],q_boundary_ip[i]],intensity_boundary_ip,color = color)
    ax_ip.set_ylim((0,fit_data['hor']['y'][-1].max()*1.2))

    ax_ip.set_xlabel(r'$q_\parallel$ / $\AA^{-1}$', fontsize=20)
    ax_ip.set_ylabel(r'Intensity / a.u.', fontsize=20)
    ax_oop.plot(fit_data['ver']['x'][-1],fit_data['ver']['y'][-1])
    ax_oop.plot(fit_data['ver']['x'][0],model(fit_data['ver']['x'][0],*fit_results['ver'][0]))
    q_boundary_oop = [fit_data['ver']['x'][0].min(), fit_data['ver']['x'][0].max(),fit_data['ver']['x'][-1].min(), fit_data['ver']['x'][-1].max()]
    intensity_boundary_oop = [fit_data['ver']['y'][-1].min(), fit_data['ver']['y'][-1].max()]
    for i in range(len(q_boundary_oop)):
        if i in [0,1]:
            color = 'gray'
        else:
            color = 'red'
        ax_oop.plot([q_boundary_oop[i],q_boundary_oop[i]],intensity_boundary_oop,color = color)
    ax_oop.set_ylim((0,fit_data['ver']['y'][-1].max()*1.2))
    ax_oop.set_xlabel(r'$q_\perp$ / $\AA^{-1}$', fontsize=20)
    ax_oop.set_ylabel(r'Intensity / a.u.', fontsize=20)
    if is_zap_scan:
        ax_strain_oop.plot(processed_data_container['oop_strain'],'r-.')
        ax_strain_ip.plot(processed_data_container['ip_strain'],'g-.')
        ax_width_oop.plot(processed_data_container['oop_grain_size'],'r-.')
        ax_width_ip.plot(processed_data_container['ip_grain_size'],'g-.')
        ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        ax_v.plot(processed_data_container['potential'],'m:d')
        ax_strain_oop.plot(len(processed_data_container['potential'])-1,processed_data_container['oop_strain'][-1],'g-o')
        ax_strain_ip.plot(len(processed_data_container['potential'])-1,processed_data_container['ip_strain'][-1],'r-o')
        ax_width_oop.plot(len(processed_data_container['potential'])-1,processed_data_container['oop_grain_size'][-1],'g-o')
        ax_width_ip.plot(len(processed_data_container['potential'])-1,processed_data_container['ip_grain_size'][-1],'r-o')
        ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')
        ax_v.plot(len(processed_data_container['potential'])-1,processed_data_container['potential'][-1],'k:d')
    else:
        # ax_strain_oop.plot(processed_data_container['potential'],processed_data_container['oop_strain'],'r-.')
        # ax_strain_ip.plot(processed_data_container['potential'],processed_data_container['ip_strain'],'g-.')
        # ax_width_oop.plot(processed_data_container['potential'],processed_data_container['oop_grain_size'],'r-.')
        # ax_width_ip.plot(processed_data_container['potential'],processed_data_container['ip_grain_size'],'g-.')
        # ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        # ax_v.plot(processed_data_container['potential'],'m:d')
        # ax_strain_oop.plot(processed_data_container['potential'][-1],processed_data_container['oop_strain'][-1],'g-o')
        # ax_strain_ip.plot(processed_data_container['potential'][-1],processed_data_container['ip_strain'][-1],'r-o')
        # ax_width_oop.plot(processed_data_container['potential'][-1],processed_data_container['oop_grain_size'][-1],'g-o')
        # ax_width_ip.plot(processed_data_container['potential'][-1],processed_data_container['ip_grain_size'][-1],'r-o')
        # ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')
        ax_strain_oop.plot(processed_data_container['oop_strain'],'r-.')
        ax_strain_ip.plot(processed_data_container['ip_strain'],'g-.')
        ax_width_oop.plot(processed_data_container['oop_grain_size'],'r-.')
        ax_width_ip.plot(processed_data_container['ip_grain_size'],'g-.')
        ax_cv.plot(processed_data_container['potential'], processed_data_container['current_density'],'k--.')
        ax_v.plot(processed_data_container['potential'],'m:d')
        ax_strain_oop.plot(processed_data_container['oop_strain'][-1],'g-o')
        ax_strain_ip.plot(processed_data_container['ip_strain'][-1],'r-o')
        ax_width_oop.plot(processed_data_container['oop_grain_size'][-1],'g-o')
        ax_width_ip.plot(processed_data_container['ip_grain_size'][-1],'r-o')
        ax_cv.plot(processed_data_container['potential'][-1], processed_data_container['current_density'][-1],'r--o')
    ax_v.set_xlabel('Time')
    ax_v.set_title('potential')
    ax_strain_oop.set_title('out-of-plane strain')
    ax_strain_ip.set_title('inplane strain')
    ax_width_oop.set_title('out-of-plane width')
    ax_width_ip.set_title('inplane width')
    ax_cv.set_title('CV')
    if is_zap_scan:
        ax_cv.set_xlabel('E(V)')
        ax_strain_ip.set_xlabel('Time')
        ax_strain_oop.set_xlabel('Time')
        ax_width_ip.set_xlabel('Time')
        ax_width_oop.set_xlabel('Time')
    else:
        ax_cv.set_xlabel('E(V)')
        ax_strain_ip.set_xlabel('E(V)')
        ax_strain_oop.set_xlabel('E(V)')
        ax_width_ip.set_xlabel('E(V)')
        ax_width_oop.set_xlabel('E(V)')

    return fig

def plot_after_fit(data,is_zap_scan):
    if not is_zap_scan:
        plt.figure(111)
        plt.plot(data['potential'],data['ip_strain'],'o-',markersize=5)
        plt.title('Inplane strain(%)')
        plt.xlabel('Potential(v)')
        plt.figure(112)
        plt.plot(data['potential'],data['ip_grain_size'])
        plt.title('Inplane crystallite size (nm)')
        plt.xlabel('Potential(v)')
        plt.figure(222)
        plt.plot(data['potential'],data['oop_strain'],'o-',markersize=5)
        plt.title('Out of plane strain(%)')
        plt.xlabel('Potential(v)')
        plt.figure(223)
        plt.plot(data['potential'],data['oop_grain_size'])
        plt.xlabel('Potential(v)')
        plt.title('Out of plane crystallite size(nm)')
        plt.figure(224)
        plt.plot(data['potential'],data['peak_intensity'])
        plt.xlabel('Potential(v)')
        plt.title('Peak intensity (counts)')
        plt.show()
    else:
        plt.figure(111)
        plt.plot(data['cen_ip'],'o-',markersize=5)
        plt.title('Inplane peak position')
        plt.figure(112)
        plt.plot(data['FWHM_ip'])
        plt.title('Inplane peak width')
        plt.figure(222)
        plt.plot(data['cen_oop'],'o-',markersize=5)
        plt.title('Out of plane peak position')
        plt.figure(223)
        plt.plot(data['FWHM_oop'])
        plt.title('Out of plane peak width')
        plt.show()


def movie_creator(fig_handle, movie_name,fps = 5):
    canvas_width, canvas_height = fig_handle.canvas.get_width_height()
    # Open an ffmpeg process
    outf = movie_name
    cmdstring = ('ffmpeg',
            '-y', '-r', str(fps), # overwrite, 30fps
            '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
            '-pix_fmt', 'argb', # format
            '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'mpeg4', outf) # output encoding
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
    agg=fig_handle.canvas.switch_backends(FigureCanvasAgg)
    return p, agg

def update_line(line_ax_handles, atr_value_dic={'set_data':None,'set_cmap':'gnuplot2'}):
    num_ax = len(line_ax_handles)
    updated_lines = []
    for i in range(num_ax):
        line_ax_handle = line_ax_handles[i]
        for key in atr_value_dic.keys():
            getattr(line_ax_handle, key)(atr_value_dic[key])
        updated_lines.append(line_ax_handle)
    return line_ax_handles

