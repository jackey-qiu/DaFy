import numpy as num
import numpy as np
from matplotlib import pyplot
import matplotlib as mpt
import pickle,glob
import sys,os,inspect
from matplotlib import rc
from matplotlib import pyplot
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from color_mate import color_combine_mate as set_color
import os,inspect,copy,sys
SuPerRod_path=['/','\\'][int(os.name=='nt')].join(os.getcwd().rsplit(['/','\\'][int(os.name=='nt')])[0:os.getcwd().rsplit(['/','\\'][int(os.name=='nt')]).index('SuPerRod')+1])
print('SuPerRod_path @ '+ SuPerRod_path)
sys.path.append(SuPerRod_path)
sys.path.append(['/','\\'][int(os.name=='nt')].join([SuPerRod_path,'models']))
sys.path.append(['/','\\'][int(os.name=='nt')].join([SuPerRod_path,'lib']))
sys.path.append(['/','\\'][int(os.name=='nt')].join([SuPerRod_path,'plugins','data_loaders']))
import parameters,diffev,model
import filehandling as io


##fill in the plot items here and then exec in the ipython with 'execfile('create_plots_gx_file.py')'#####
###################################Plot items handle######################################################
plot_e_model=True                                                      ##plot electron density profiles?##
plot_e_FS=True                              ##plot electron density profiles based on Foriour synthesis?##
plot_ctr=True                                                                        ##plot CTR results?##
plot_raxr=True                                                                     ##plot RAXR results?##
<<<<<<< HEAD:accessory_functions/plotting_results/create_plots_gx_file_muscovite.py
plot_AP_Q=False                                                              ##plot Foriour components?##
gx_file_path='/Users/cqiu/model_file/scale_RAXR_Th_mica_LiCl_100mM_superrod_run2_Jun05combined_ran.gx'                         ##where is your gx file##
=======
plot_AP_Q=False                                                               ##plot Foriour components?##
gx_file_path='P:\\temp_model\\scale_RAXR_Th_mica_LiCl_100mM_superrod_run2_Jun05combined_ran.gx'                         ##where is your gx file##
>>>>>>> 23dca1bf07e5598b2c9a615f60e2fd7ae69c68a2:accessory_functions/plotting_results/legacy/create_plots_gx_file.py
##########################################################################################################


def local_func():
    return None

def module_path_locator(func=local_func):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(func))))),'dump_files')

bl_dl_muscovite_old={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '0_0':{'segment':[[0,20]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
    '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
    '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
bl_dl_muscovite={'0_0':{'segment':[[0,20]],'info':[[2,2]]}}

"""
functions to make plots of CTR, RAXR, Electron Density using the dumped files created in GenX script (running_mode=False)
Formates for each kind of dumped files
1. CTR_dumped file:[experiment_data,model],both items in the list is a library of form {'HKL':[L,I,eI]} and {'HKL':[L,I]},respecitvely.
2. RAXR_dumped file: [experiment_data,model],both items in the list is a library of form {'HKL':[E,I,eI]} and {'HKL':[E,I]},respecitvely.
3. e_density_dumped file (model): [e_data, labels], where e_data=[[z,ed1],[z,ed2]...[z,ed_total]],labels=['Domain1A','Domain2A',...,'Total']
4. e_density_dumped file (imaging): [z_plot,eden_plot,eden_domains], where
    z_plot is a list [z1,z2,z3,...,zn]
    eden_plot is a list of [ed1,ed2,...,edn], which is the total e density for all domains
    eden_domains=[[ed_z1_D1,ed_z1_D2,...,ed_z1_Dm],[ed_z2_D1,ed_z2_D2,...,ed_z2_Dm],...,[ed_zn_D1,ed_zn_D2,...,ed_zn_Dm]] considering m domains
"""


def plot_all(path=module_path_locator(),make_offset_of_total_e=False,fit_e_profile=0,plot_e_model=1,plot_e_FS=1,plot_ctr=1,plot_raxr=0,plot_AP_Q=0):
    #set make_offset_of_total_e to True if you want to have total e density replesented by total_e - resonant e in the case when the resonant els are freezed to have no influence on the CTR.
    #At the same time, the total_e - raxs_e - water is actually total_e - 2*raxs_e -water
    PATH=path
    #which plots do you want to create
    plot_e_model,plot_e_FS,plot_ctr,plot_raxr,plot_AP_Q=plot_e_model,plot_e_FS,plot_ctr,plot_raxr,plot_AP_Q

    #specify file paths (files are dumped files when setting running_mode=False in GenX script)
    e_file=os.path.join(PATH,"temp_plot_eden")#e density from model
    e_file_FS=os.path.join(PATH,"temp_plot_eden_fourier_synthesis") #e density from Fourier synthesis
    water_scaling_file=os.path.join(PATH,"water_scaling")
    ctr_file_folder=PATH
    ctr_file_names=["temp_plot"]#you may want to overplot differnt ctr profiles based on differnt models
    raxr_file=os.path.join(PATH,"temp_plot_raxr")
    AP_Q_file=os.path.join(PATH,"temp_plot_raxr_A_P_Q")
    e_den_subtracted=None
    e_den_raxr_MI=None
    water_scaling=None
    #plot electron density profile
    if plot_e_model:
        data_eden=pickle.load(open(e_file,"rb"))
        edata,labels=data_eden[0],data_eden[1]
        water_scaling=pickle.load(open(water_scaling_file,"rb"))[-1]#total water scaling factor to be used in Gaussian fit below
        N=len(labels)
        fig=pyplot.figure(figsize=(6,6))
        if plot_e_FS:
            data_eden_FS=pickle.load(open(e_file_FS,"rb"))
            data_eden_FS_sub=pickle.load(open(e_file_FS+"_sub","rb"))
        for i in range(N-1,N):#here only plot the total e density
            if i==N-1:
                ax=fig.add_subplot(1,1,1)
            else:
                pass
            if make_offset_of_total_e:
                try:
                    edata[i][1,:]=list(np.array(edata[i][1,:])-np.array(edata[i][2,:]))
                except:
                    pass
            else:
                pass
            ax.plot(np.array(edata[i][0,:]),edata[i][1,:],color='black',lw=2.5,linestyle='-',label="Total e density")
            ax.plot([0,0],[0,max(edata[i][1,:])+2],linestyle=':',color='m',lw=3,label='Mineral surface')
            ax.fill_between(np.array(edata[i][0,:]),edata[i][2,:],alpha=0.2,color='m',label="RAXS element e profile (MD)")
            pyplot.title(labels[i],fontsize=11)
            if plot_e_FS:
                ax.plot(data_eden_FS[0],data_eden_FS[1],color='r',label="RAXR imaging (MI)")
                ax.fill_between(data_eden_FS[0],edata[i][3,:],color='green',alpha=0.4,label="LayerWater")
                eden_temp=None
                if make_offset_of_total_e:
                    eden_temp=list(edata[i][1,:]-edata[i][3,:]-2*edata[i][2,:]*0)
                else:
                    eden_temp=list(edata[i][1,:]-edata[i][3,:]-edata[i][2,:]*0)
                eden_temp=(np.array(eden_temp)*(np.array(eden_temp)>0.01))[:,np.newaxis]
                z_temp=np.array(data_eden_FS[0])[:,np.newaxis]
                e_den_subtracted=np.append(z_temp,eden_temp,axis=1)
                e_den_raxr_MI=np.append(np.array(data_eden_FS[0])[:,np.newaxis],(np.array(data_eden_FS[1])*(np.array(data_eden_FS[1])>0.01))[:,np.newaxis],axis=1)
                ax.plot(data_eden_FS_sub[0],data_eden_FS_sub[1],color='blue',label="RAXR imaging (MD)")
            pyplot.xlabel('Z(Angstrom)',axes=ax,fontsize=12)
            pyplot.ylabel('E_density',axes=ax,fontsize=12)
            pyplot.ylim(ymin=0)
            pyplot.xlim(xmin=-10)
            pyplot.xlim(xmax=max(data_eden_FS[0]))
            pyplot.legend(fontsize=11,ncol=1)
        fig.tight_layout()
        fig.savefig(e_file+".png",dpi=300)
    if plot_ctr:
        plotting_many_modelB(save_file=os.path.join(ctr_file_folder,"temp_plot_ctr.png"),head=ctr_file_folder,object_files=ctr_file_names,color=['b','r'],l_dashes=[(None,None)],lw=1)
        plt.show(block=False)
    if plot_raxr:
        #plot raxr profiles
        data_raxr=pickle.load(open(raxr_file,"rb"))
        try:
            plotting_raxr_new(data_raxr,savefile=raxr_file+".png",color=['b','r'],marker=['o'])
            plt.show(block=False)
        except:
            print('No RAXR data is plotted!')
    if plot_AP_Q:
        #plot Q dependence of Foriour components A and P
        colors=['black','r','blue','green','yellow']
        labels=['Domain1','Domain2','Domain3','Domain4']
        data_AP_Q=pickle.load(open(AP_Q_file,"rb"))
        fig1=pyplot.figure(figsize=(9,9))
        ax1=fig1.add_subplot(2,1,1)
        #A over Q
        ax1.plot(data_AP_Q[0][2],data_AP_Q[0][0],color='r')
        ax1.errorbar(data_AP_Q[1][2],data_AP_Q[1][0],yerr=np.transpose(data_AP_Q[1][3]),color='g',fmt='o')
        pyplot.ylabel("A",axes=ax1)
        pyplot.xlabel("Q",axes=ax1)
        pyplot.legend()
        #P over Q
        ax2=fig1.add_subplot(2,1,2)
        ax2.plot(data_AP_Q[0][2],np.array(data_AP_Q[0][1])/np.array(data_AP_Q[0][2])*np.pi*2,color='r')
        ax2.errorbar(data_AP_Q[1][2],np.array(data_AP_Q[1][1])/np.array(data_AP_Q[1][2])*np.pi*2,yerr=np.transpose(data_AP_Q[1][4])*np.pi*2/[data_AP_Q[1][2],data_AP_Q[1][2]],color='g',fmt='o')
        pyplot.ylabel("P/Q(2pi)",axes=ax2)
        pyplot.xlabel("Q",axes=ax2)
        pyplot.legend()
        fig1.savefig(os.path.join(PATH,'temp_APQ_profile.png'),dpi=300)
    #now plot the subtracted e density and print out the gaussian fit results
    if fit_e_profile:
        pyplot.figure()
        print '##############Total e -layer water#################'
        #gaussian_fit(e_den_subtracted,zs=None,water_scaling=water_scaling)
        gaussian_fit_DE(e_den_subtracted,zs=3,water_scaling=water_scaling)
        pyplot.title('Total e - layer water')

        pyplot.figure()
        print '#########################RAXR (MI)########################'
        gaussian_fit(e_den_raxr_MI,zs=None,N=40,water_scaling=water_scaling)
        pyplot.title('RAXR (MI)')
        pyplot.figure()
        '''
        print '#########################RAXR (MD)########################'
        gaussian_fit(np.append([data_eden_FS_sub[0]],[data_eden_FS_sub[1]*(np.array(data_eden_FS_sub[1])>0)],axis=0).transpose(),zs=None,N=40,water_scaling=water_scaling)
        pyplot.title('RAXR (MD)')
        pyplot.show()
        #return e_den_subtracted,data_eden_FS
        '''
    return water_scaling

def gaussian_fit_DE(data,fit_range=[1,40],zs=None,N=8,water_scaling=1):
    x,y=[],[]
    for i in range(len(data)):
        if data[i,0]>fit_range[0] and data[i,0]<fit_range[1]:
            x.append(data[i,0]),y.append(data[i,1])
    x,y=np.array(x),np.array(y)*water_scaling
    plt.plot(x,y)
    plt.show()

    def func_DE(params,*args):
        x=np.array(list(args))
        y_fit = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = abs(params[i])
            wid = abs(params[i+1])
            ctr= abs(params[i+2])
            y_fit = y_fit + amp * np.exp( -((x - ctr)/wid)**2/2)
        return sum(abs(y_fit-y))

    def func(params,*args):
        x=np.array(list(args))
        y_fit = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = abs(params[i])
            wid = abs(params[i+1])
            ctr= abs(params[i+2])
            y_fit = y_fit + amp * np.exp( -((x - ctr)/wid)**2/2)
        return y_fit

    ctrs=[]
    bounds=[]
    if zs==None:
        for i in range(1,len(x)-1):
            if y[i-1]<y[i] and y[i+1]<y[i]:
                ctrs.append(x[i])
    elif type(zs)==int:
        ctrs=[fit_range[0]+(fit_range[1]-fit_range[0])/zs*i for i in range(zs)]+[fit_range[1]]
    else:
        ctrs=np.array(zs)
    for i in range(len(ctrs)):
        #bounds+=[(0,30),(0.2,10),(np.max([ctrs[i]-5,fit_range[0]]),np.min([ctrs[i]+5,fit_range[1]]))]
        bounds+=[(0,30),(0.2,10),(fit_range[0],fit_range[1])]

    result=differential_evolution(func_DE, bounds,args=tuple(x))
    popt=result.x
    print popt
    combinded_set=[]
    #print 'z occupancy*4 U(sigma**2)'
    for i in range(0,len(popt),3):
        combinded_set=combinded_set+[abs(popt[i])/N*(abs(popt[i])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2,popt[i+2]]
        #print '%3.3f\t%3.3f\t%3.3f'%(ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2)
    combinded_set=np.reshape(np.array(combinded_set),(len(combinded_set)/3,3)).transpose()
    print combinded_set
    #combinded_set=combinded_set.transpose()
    #normalized to full surface unit cell
    print 'total_occupancy=',np.sum(combinded_set[0,:]/4)
    #normalized to half unit cell (oc and u have been added to 1 to be used in Matlab input par file)
    print 'OC_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])'
    #the u not U(u**2)
    print 'U_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])'
    print 'X_RAXS_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Y_RAXS_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Z_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])'


    fit = func(popt,*x)

    plt.plot(x, y)
    plt.plot(x, fit , 'r-')
    plt.show()

def gaussian_fit(data,fit_range=[1,40],zs=None,N=8,water_scaling=1):
    x,y=[],[]
    for i in range(len(data)):
        if data[i,0]>fit_range[0] and data[i,0]<fit_range[1]:
            x.append(data[i,0]),y.append(data[i,1])
    x,y=np.array(x),np.array(y)*water_scaling
    plt.plot(x,y)
    plt.show()

    def func(x_ctrs,*params):
        y = np.zeros_like(x_ctrs[0])
        x=x_ctrs[0]
        ctrs=x_ctrs[1]
        for i in range(0, len(params), 2):
            amp = abs(params[i])
            wid = abs(params[i+1])
            ctr=ctrs[int(i/2)]
            y = y + amp * np.exp( -((x - ctr)/wid)**2/2)
        return y

    guess = []
    ctrs=[]
    if zs==None:
        for i in range(1,len(x)-1):
            if y[i-1]<y[i] and y[i+1]<y[i]:
                ctrs.append(x[i])
    elif type(zs)==int:
        ctrs=[fit_range[0]+(fit_range[1]-fit_range[0])/zs*i for i in range(zs)]+[fit_range[1]]
    else:
        ctrs=np.array(zs)
    for i in range(len(ctrs)):
        guess += [0.5, 1]

    popt, pcov = curve_fit(func, [x,ctrs], y, p0=guess)
    combinded_set=[]
    #print 'z occupancy*4 U(sigma**2)'
    for i in range(0,len(popt),2):
        combinded_set=combinded_set+[ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2]
        #print '%3.3f\t%3.3f\t%3.3f'%(ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2)
    combinded_set=np.reshape(np.array(combinded_set),(len(combinded_set)/3,3)).transpose()
    #combinded_set=combinded_set.transpose()
    #normalized to full surface unit cell
    #inputs for GenX refinement
    print 'total_occupancy=',np.sum(combinded_set[1,:]/4)
    print 'OC_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])'
    print 'U_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])'
    print 'X_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Y_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Z_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])'


    fit = func([x,ctrs], *popt)

    plt.plot(x, y)
    plt.plot(x, fit , 'r-')
    plt.show()

def find_A_P_muscovite(q_list,ctrs,amps,wids,wt=0.25):
    #ctrs:z list (in A with reference of surface having 0A)
    #amps:oc list
    #wids:u list(in A)
    Q=q_list
    A_container,P_container=[],[]
    for q_index in range(len(Q)):
        q=Q[q_index]
        complex_sum=0.+1.0J*0.
        for i in range(len(ctrs)):
            complex_sum+=wt*amps[i]*np.exp(-q**2*wids[i]**2/2)*np.exp(1.0J*q*ctrs[i])#z should be plus 1 to account for the fact that surface slab sitting on top of bulk slab
        A_container.append(abs(complex_sum))
        img_complex_sum, real_complex_sum=np.imag(complex_sum),np.real(complex_sum)
        if img_complex_sum==0.:
            P_container.append(0)
        elif real_complex_sum==0 and img_complex_sum==1:
            P_container.append(0.25)#1/2pi/2pi
        elif real_complex_sum==0 and img_complex_sum==-1:
            P_container.append(0.75)#3/2pi/2pi
        else:#adjustment is needed since the return of np.arctan is ranging from -1/2pi to 1/2pi
            if real_complex_sum>0 and img_complex_sum>0:
                P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.)
            elif real_complex_sum>0 and img_complex_sum<0:
                P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+1.)
            elif real_complex_sum<0 and img_complex_sum>0:
                P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)
            elif real_complex_sum<0 and img_complex_sum<0:
                P_container.append(np.arctan(img_complex_sum/real_complex_sum)/np.pi/2.+0.5)

    return np.array(A_container),np.array(P_container),Q

def fourier_synthesis(q_list,P,A,z,N=40,Auc=46.9275):
    ZR=N
    q_list.sort()
    delta_q=np.average([q_list[i+1]-q_list[i] for i in range(len(q_list)-1)])
    z_plot=z
    eden_plot=[]
    for i in range(len(z)):
        z_each=z[i]
        eden=0
        eden=ZR/Auc/np.pi/2*np.sum(A*np.cos(2*np.pi*P-np.array(q_list)*z_each)*delta_q)
        eden_plot.append(eden)
    return eden_plot

def q_list_func(h, k, l,a=5.1988, b=9.0266, c=20.1058, alpha=90,beta=95.782,gamma=90):
    '''Returns the absolute value of (h,k,l) vector in units of
    AA.

    This is equal to the inverse lattice spacing 1/d_hkl.
    '''
    h=np.array(h)
    k=np.array(k)
    l=np.array(l)
    alpha,beta,gamma=np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma)
    dinv = np.sqrt(((h/a*np.sin(alpha))**2 +
                     (k/b*np.sin(beta))**2  +
                     (l/c*np.sin(gamma))**2 +
                    2*k*l/b/c*(np.cos(beta)*
                                         np.cos(gamma) -
                                         np.cos(alpha)) +
                    2*l*h/c/a*(np.cos(gamma)*
                                         np.cos(alpha) -
                                         np.cos(beta)) +
                    2*h*k/a/b*(np.cos(alpha)*
                                         np.cos(beta) -
                                         np.cos(gamma)))
                    /(1 - np.cos(alpha)**2 - np.cos(beta)**2
                      - np.cos(gamma)**2 + 2*np.cos(alpha)
                      *np.cos(beta)*np.cos(gamma)))
    return dinv*np.pi*2

def fit_e_density(path=module_path_locator(),fit_range=[1,40],zs=None,N=8):
    PATH=path
    ##extract hkl values##
    full_dataset=np.loadtxt(os.path.join(PATH,"temp_full_dataset.dat"))
    h,k,l=[],[],[]
    for i in range(len(full_dataset)):
        if full_dataset[i,3]!=0:
            if full_dataset[i,3] not in l:
                h.append(full_dataset[i,1])
                k.append(full_dataset[i,2])
                l.append(full_dataset[i,3])
    ##extract e density data##
    data_file=os.path.join(PATH,"temp_plot_eden_fourier_synthesis")
    data=np.append([pickle.load(open(data_file,"rb"))[0]],[pickle.load(open(data_file,"rb"))[1]],axis=0).transpose()
    ##extract water scaling value##
    water_scaling_file=os.path.join(PATH,"water_scaling")
    water_scaling=pickle.load(open(water_scaling_file,"rb"))[-1]
    x,y=[],[]
    for i in range(len(data)):
        if data[i,0]>fit_range[0] and data[i,0]<fit_range[1]:
            x.append(data[i,0]),y.append(data[i,1])
    x,y=np.array(x),np.array(y)*water_scaling
    plt.plot(x,y)
    plt.show()

    #cal q list
    q_list=q_list_func(h,k,l)

    def func(x_ctrs_qs,*params):
        x=x_ctrs_qs[0]
        ctrs=x_ctrs_qs[1]
        q_list=x_ctrs_qs[2]
        amps=[]
        wids=[]
        for i in range(0, len(params), 2):
            amps.append(abs(params[i]))
            wids.append(abs(params[i+1]))

        #cal A and P list
        A,P,Q=find_A_P_muscovite(q_list,ctrs,amps,wids)
        #Fourier thynthesis
        y=fourier_synthesis(q_list,P,A,z=x,N=40)
        return y

    guess = []
    ctrs=[]
    if zs==None:
        for i in range(1,len(x)-1):
            if y[i-1]<y[i] and y[i+1]<y[i]:
                ctrs.append(x[i])
    elif type(zs)==int:
        ctrs=[fit_range[0]+(fit_range[1]-fit_range[0])/zs*i for i in range(zs)]+[fit_range[1]]
    else:
        ctrs=np.array(zs)
    for i in range(len(ctrs)):
        guess += [0.5, 1]
    #print x,ctrs
    popt, pcov = curve_fit(func, [x,ctrs,q_list], y, p0=guess)
    combinded_set=[]
    #print 'z occupancy*4 U(sigma**2)'
    for i in range(0,len(popt),2):
        combinded_set=combinded_set+[ctrs[i/2],abs(popt[i])*4,abs(popt[i+1])**2]
        #print '%3.3f\t%3.3f\t%3.3f'%(ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2)
    combinded_set=np.reshape(np.array(combinded_set),(len(combinded_set)/3,3)).transpose()
    #combinded_set=combinded_set.transpose()
    print 'total_occupancy=',np.sum(combinded_set[1,:]/4)
    print 'OC_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])'
    print 'U_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])'
    print 'X_RAXS_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Y_RAXS_LIST=[0.5]*',len(combinded_set[1,:])
    print 'Z_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])'


    fit = func([x,ctrs], *popt)

    plt.plot(x, y)
    plt.plot(x, fit , 'r-')
    plt.show()

def plotting_modelB(object=[],fig=None,index=[2,3,1],color=['0.35','r','c','m','k'],l_dashes=[()],lw=3,label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1,data2,data3],multiple dataset with the first one of experimental data and the others model datasets

    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    x_min,x_max=min([min(each[:,0]) for each in object])-0.1,max([max(each[:,0]) for each in object])+0.1
    y_min,y_max=min([min(each[:,1]) for each in object])/10.,max([max(each[:,1]) for each in object])*10.
    ax.scatter(object[0][:,0],object[0][:,1],marker='o',s=3,facecolors='none',edgecolors=color[0],label=label[0],alpha=0.5)
    ax.errorbar(object[0][:,0],object[0][:,1],yerr=object[0][:,2],fmt=None,ecolor=color[0],alpha=0.5)
    for i in range(len(object)-1):#first item is experiment data (L, I, err) while the second one is simulated result (L, I_s)
        l,=ax.plot(object[i+1][:,0],object[i+1][:,1],color=color[i+1],lw=lw,label=label[i+1])
        l.set_dashes(l_dashes[i])
        #ax.scatter(object[i+1][:,0],object[i+1][:,1],color=color[i+1],label=label[i+1],marker='*',s=5,facecolors='none')
    if index[2] in [7,8,9]:
        pyplot.xlabel('L(r.l.u)',axes=ax,fontsize=12)
    if index[2] in [1,4,7]:
        pyplot.ylabel(r'$\rm{|F_{HKL}|}$',axes=ax,fontsize=12)
    #settings for demo showing
    pyplot.title('('+title[0]+')',position=(0.5,0.82),weight=4,size=10,clip_on=True)
    #pyplot.ylim((1,1000))
    #settings for publication
    #pyplot.title('('+title[0]+')',position=(0.5,1.001),weight=4,size=10,clip_on=True)
    """##add arrows to antidote the misfits
    if title[0]=='0 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.25,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
        ax.add_patch(mpt.patches.FancyArrow(0.83,0.5,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='1 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.68,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='3 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.375,0.8,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    """
    if legend==True:
        #ax.legend()
        ax.legend(bbox_to_anchor=(0.2,1.03,3.,1.202),mode='expand',loc=3,ncol=5,borderaxespad=0.,prop={'size':9})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)
    #plot normalized data now
    if index[0]==2 and index[1]==1:
        ax=fig.add_subplot(index[0],index[1],index[2]+1)
        ax.set_yscale('log')
        ax.scatter(object[0][:,0],object[0][:,3],marker='o',s=20,facecolors='none',edgecolors=color[0],label=label[0])
        ax.errorbar(object[0][:,0],object[0][:,3],yerr=object[0][:,4],fmt=None,ecolor=color[0])
        for i in range(len(object)-1):#first item is experiment data (L, I, err) while the second one is simulated result (L, I_s)
            l,=ax.plot(object[i+1][:,0],object[i+1][:,2],color=color[i+1],lw=lw,label=label[i+1])
            l.set_dashes(l_dashes[i])
        if index[2] in [7,8,9]:
            pyplot.xlabel(r'$\rm{L(r.l.u)}$',axes=ax,fontsize=12)
        if index[2] in [1,4,7]:
            pyplot.ylabel(r'$|normalized F_{HKL}|$',axes=ax,fontsize=12)
    #settings for demo showing
    if index[0]==2 and index[1]==1:
        pass
    else:
        if title[0]=='0 0 L':
            pyplot.ylim((0.001,5))
            #pyplot.xlim((0.,4))
            xtick_labels=ax.get_xticks().tolist()
            x_tick_new=[]
            for each in xtick_labels:
                if each in [-1,0,1.0,2.0,3.,4.,5.]:
                    x_tick_new.append(int(each))
                else:
                    x_tick_new.append('')
            #ax.set_xticklabels(x_tick_new)
            #pyplot.xlim((0,20))
        elif title[0]=='3 0 L':
            pyplot.ylim((.001,5))
        elif title[0] in ['2 1 L','2 -1 L']:
            pyplot.ylim((0.001,5))
        else:pyplot.ylim((0.001,5))

    pyplot.xlim((x_min,x_max))
    pyplot.ylim((y_min,y_max))
    if title[0]=='0 0 L':
        pyplot.xlim((0.8,x_max))
    #settings for publication
    #pyplot.title('('+title[0]+')',position=(0.5,1.001),weight=4,size=10,clip_on=True)
    """##add arrows to antidote the misfits
    if title[0]=='0 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.25,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
        ax.add_patch(mpt.patches.FancyArrow(0.83,0.5,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='1 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.68,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='3 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.375,0.8,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    """
    if legend==True:
        #ax.legend()
        ax.legend(bbox_to_anchor=(0.2,1.03,3.,1.202),mode='expand',loc=3,ncol=5,borderaxespad=0.,prop={'size':9})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)

    #ax.set_ylim([1,10000])
#object files are returned from genx when switch the plot on
def plotting_many_modelB(save_file='D://pic.png',head='P:\\My stuff\\Manuscripts\\hematite rcut\\pb on cmp rcut v10\dump_files\\',object_files=['temp_plot_O1O3_O5O7','temp_plot_O1O3_O5O8','temp_plot_O1O4_O5O7','temp_plot_O1O4_O5O8'],index=[3,3],color=['blue','#e41a1c','#4daf4a','#984ea3','#ff7f00'],lw=1.5,l_dashes=[(None,None),(None,None),(None,None),(None,None),(None,None),(None,None)],label=['Experimental data','Model1 results','Model2 results','Model3','Model4','Model5','Model6'],marker=['p'],title=['0 0 L','0 2 L','1 0 L','1 1 L','2 0 L','2 2 L','3 0 L','2 -1 L','2 1 L'],legend=[False,False,False,False,False,False,False,False,False],fontsize=10):
    #plotting model results simultaneously, object_files=[file1,file2,file3] file is the path of a dumped data/model file
    #setting for demo show
    #fig=pyplot.figure(figsize=(10,9))
    #settings for publication
    #fig=pyplot.figure(figsize=(10,7))
    fig=pyplot.figure(figsize=(8,6))
    object_sets=[pickle.load(open(os.path.join(head,file))) for file in object_files]#each_item=[00L,02L,10L,11L,20L,22L,30L,2-1L,21L]
    object=[]
    for i in range(len(object_sets[0])):
        object.append([])
        for j in range(len(object_sets)):
            if j==0:
                object[-1].append(object_sets[j][i][0])
            object[-1].append(object_sets[j][i][1])
    if len(object_sets[0])==1:#case for plotting muscovite (assuming there is one specular rod)
        index=[2,1]
    else:#case for plotting hematite (9 rods in total including one specular rod)
        index=[3,3]

    for i in range(len(object)):
    #for i in range(1):
        order=i
        #print 'abc'
        ob=object[i]
        plotting_modelB(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        #plotting_modelB(object=ob,fig=fig,index=[1,1,i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(save_file,dpi=300)
    return fig

def plotting_raxr_new(data,savefile="D://raxr_temp.png",color=['b','r'],marker=['o']):
    experiment_data,model=data[0],data[1]
    labels=model.keys()
    label_tag=map(lambda x:float(x.split("_")[-1]),labels)
    label_tag.sort()
    labels=map(lambda x:"0_0_"+str(x),label_tag)
    #labels.sort()
    fig=pyplot.figure(figsize=(15,len(labels)/3))
    for i in range(len(labels)):
        rows=None
        if len(labels)%3==0:
            rows=len(labels)/3
        else:
            rows=len(labels)/3+1
        ax=fig.add_subplot(rows,3,i+1)
        ax.scatter(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label="data points")
        ax.errorbar(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1],yerr=experiment_data[labels[i]][:,2],fmt=None,color=color[0])
        ax.plot(model[labels[i]][:,0],model[labels[i]][:,1],color=color[1],lw=3,label='model profile')
        if i!=len(labels)-1:
            ax.set_xticklabels([])
            pyplot.xlabel('')
        else:
            pyplot.xlabel('Energy (kev)',axes=ax,fontsize=12)
        pyplot.ylabel('|F|',axes=ax,fontsize=12)
        pyplot.title(labels[i])
    #fig.tight_layout()
    fig.savefig(savefile,dpi=300)
    return fig
def replace_script_section(str_text,key_text,replace_value,replace_loc=1):
    #loc=str_text.find(key_text,replace_loc-1)
    loc=find_nth_overlapping(str_text,key_text,replace_loc)
    comment_pos=None
    equal_sign_pos=None
    line_end_pos=None
    line_start_pos=loc
    while str_text[loc]!='\n':
        if str_text[loc]=='#':
            comment_pos=loc
        elif str_text[loc]=='=':
            equal_sign_pos=loc
        else:
            pass
        loc+=1
    line_end_pos=loc
    if comment_pos==None:
        comment_pos=line_end_pos
    text_to_be_feed=key_text+'='+str(replace_value)+str_text[comment_pos:line_end_pos]
    str_text=str_text.replace(str_text[line_start_pos:line_end_pos],text_to_be_feed,1)
    return str_text

if __name__=="__main__":
    mod = model.Model()
    config = io.Config()
    opt = diffev.DiffEv()
    io.load_gx(gx_file_path,mod,opt,config)
    mod.script=replace_script_section(mod.script,'running_mode','0')
    print('Simulate and dump files now!')
    mod.simulate()
    print('Plot files are dumpt to pocket!')
    print('Plot the results now!')
    plot_all(plot_e_model=plot_e_model,plot_e_FS=plot_e_FS,plot_ctr=plot_ctr,plot_raxr=plot_raxr,plot_AP_Q=plot_AP_Q)
