import numpy as np
import os
import glob

def qsi_correction(data_path='M:\\fwog\\members\\qiu05\\mica\\nQc_zr_mica_CTR_May19_GenX_formate.dat',L_column=0,I_column=4,correction_factor=0.3125):
    #correction_factor=2pi/c_project, where c-project=volume/a*b
    data=np.loadtxt(data_path)
    I=data[:,I_column]*(correction_factor*data[:,L_column])**2
    data[:,I_column]=I
    data[:,I_column+1]=data[:,I_column+1]*(correction_factor*data[:,L_column])**2
    np.savetxt(data_path.replace('.dat','_Q_corrected.dat'),data,fmt='%.5e')
    return True

def format_hkl(h_,k_,x_):
    new_h,new_k,new_x=[],[],[]
    if np.around(h_[0],0) in [1,2,3] and np.around(k_[0],0)==0:
        for iii in range(len(x_)):
            if x_[iii]>0:
                new_h.append(-h_[iii])
                new_k.append(-k_[iii])
                new_x.append(-x_[iii])
            else:
                new_h.append(h_[iii])
                new_k.append(k_[iii])
                new_x.append(x_[iii])
        return  np.array(new_h),np.array(new_k),np.array(new_x)
    else:
        return np.array(h_),np.array(k_),np.array(x_)

def l_correction(data_path='P:\\apps\\genx_pc_qiu\\dump_files\\temp_full_dataset.dat',L_column=[0,3],I_column=4,correction_factor=0.3125,l_shift=0):
    data=np.loadtxt(data_path)
    raxr_first=None
    for i in range(len(data)):
        if data[i,0]>100:
            raxr_first=i
            break
    data[0:raxr_first,I_column]=(data[0:raxr_first,0]/(data[0:raxr_first,0]+l_shift))**2*data[0:raxr_first,I_column]
    data[raxr_first:len(data),I_column]=(data[raxr_first:len(data),3]/(data[raxr_first:len(data),3]+l_shift))**2*data[raxr_first:len(data),I_column]
    np.savetxt(data_path.replace('.dat','_l_corrected.dat'),data,fmt='%.5e')
    return True

bl_dl_muscovite={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '0_0':{'segment':[[0,20]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
    '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
    '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
def formate_CTR_data(file='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\nQc_Th_LiCl_3mM_191217a',bragg_peaks=bl_dl_muscovite):
    data_formated=None
    f_original=np.loadtxt(file,skiprows=0,comments='%')
    data_points=len(f_original)-1#the first row is not data but some q corr information
    LB=np.array([2]*data_points)[:,np.newaxis]
    dL=np.array([2]*data_points)[:,np.newaxis]
    #print f_original[1:,2]
    #print f_original[1:,2][:,np.newaxis]*f_original[0,0]/2./np.pi
    data_formated=f_original[1:,2][:,np.newaxis]*f_original[0,0]/2./np.pi#recaculate L based on q data (matlab script correct only q column but not update L column as well)
    data_formated=np.append(data_formated,np.array([0]*data_points)[:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,np.array([0]*data_points)[:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,np.array([0]*data_points)[:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,f_original[1:,3][:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,f_original[1:,4][:,np.newaxis],axis=1)
    data_formated=np.append(data_formated,LB,axis=1)
    data_formated=np.append(data_formated,dL,axis=1)
    np.savetxt(file+'_GenX_formate.dat',data_formated,fmt='%.5e')
    #qsi_correction(file+'_GenX_formate.dat',L_column=0,I_column=4,correction_factor=np.pi*2/f_original[0,0])
    return None

def formate_RAXR_data(file_path='M:\\fwog\\members\\qiu05\\1608 - 13-IDC\\schmidt\mica\\files for Li Th mica model\\th_mica_LiCl_',E_range=[16196,16546]):
    full_data=np.zeros((1,8))
    L_list=['0041a','0053a','0061b','0075a','0088a','0115a','0145a','0171a','0231a','0264a','0285a','0321a','0355a','0424a','0455a','0561a','0625a','0731a','0915a','1031a','1115a']

    segment_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #L_list=[L_list[0]]
    def _find_segment_index(data,segment_index):
        current_segment=0
        index_container=[]
        for i in range(len(data)):
            if i!=len(data)-1:
                if data[i+1,0]<data[i,0]:
                    current_segment+=1
            if current_segment==segment_index:
                index_container.append(i)
            else:
                pass
        return index_container

    for i in range(len(L_list)):
        file=file_path+L_list[i]+'_RAXR_R.ipg'
        data=np.loadtxt(file,comments='%')
        index_segment_all=_find_segment_index(data,segment_list[i])
        if E_range==None:
            index_segment=index_segment_all
        else:
            index_segment=[i for i in index_segment_all if data[i,3]*1000<E_range[1] and data[i,3]*1000>E_range[0]]
        E=data[index_segment,3][:,np.newaxis]
        I=(data[index_segment,5]/data[index_segment,4])[:,np.newaxis]
        Ierr=(data[index_segment,6]/data[index_segment,4])[:,np.newaxis]
        BL=np.array([2]*len(index_segment))[:,np.newaxis]
        dL=np.array([2]*len(index_segment))[:,np.newaxis]
        H=np.array([0]*len(index_segment))[:,np.newaxis]
        K=np.array([0]*len(index_segment))[:,np.newaxis]
        L=np.around(data[index_segment,1][:,np.newaxis],2)
        temp_data=E*1000
        temp_data=np.append(temp_data,H,axis=1)
        temp_data=np.append(temp_data,K,axis=1)
        temp_data=np.append(temp_data,L,axis=1)
        temp_data=np.append(temp_data,I,axis=1)
        temp_data=np.append(temp_data,Ierr,axis=1)
        temp_data=np.append(temp_data,BL,axis=1)
        temp_data=np.append(temp_data,dL,axis=1)
        full_data=np.append(full_data,temp_data,axis=0)
    np.savetxt(file_path+'_GenX_formate.dat',full_data[1:],fmt='%.5e')


def split_RAXR_data_file(file='M:\\fwog\\members\\qiu05\\1704_APS_13IDC\\mica\\100mM_NH4Cl_Zr_mica\\s5_100mM_NH4Cl_Zr_1_RAXR_2nd_spot1.ipg'):

    L_container=[]
    L_label_container=[]
    data=np.loadtxt(file,comments='%')
    index_container=[0]
    for i in range(len(data)):
        if i!=len(data)-1:
            if data[i+1,0]<data[i,0]:
                index_container.append(i+1)
    index_container.append(len(data))
    sub_data_sets=[]
    for i in range(len(index_container)-1):
        start,end=index_container[i],index_container[i+1]
        sub_data_sets.append(data[start:end,:])
    for each_data_set in sub_data_sets:
        L=round(each_data_set[0,1],2)
        L_container.append(L)
        temp_L_number=sum([1 for each_L in L_container if each_L==L])
        L_label_container.append("L"+str(L).replace(".","")+"_"+str(temp_L_number))
        header='1             2           3            4            5           6            7                8          9                 10          11              12          13             14           15                   16    17                  18            19                  20          21        22         23         24           25             26         27            28          29        30               31          32       33          34           35        36           37            38\n\
        file_number   L           epoch        Energy       Monitor     Signal_Best  sigSignal_Best   Signal_0d  mse_sSignal_0d    Signal      sigSignal       Signaly     sigSignaly     Background   ccd_num_back_ccd_px  time  time_corr_factor    xbar          ybar                HH          KK        LL         theta      ttheta       chi            phi        transm        alpha       beta      azimuth          mu          nu       fwhm_x      fwhm_y       samx      samy            accept_option   specscan#'
        np.savetxt(file.replace(".ipg","_"+L_label_container[-1]+".ipg"),each_data_set,fmt='%10.5f',header=header,comments='%')

    return L_container,L_label_container

def split_RAXR_data_file_fr_GenX_output(file='P:\\apps\\genx_pc_qiu\\dump_files\\temp_full_dataset.dat',name_head='raxr_data_100mM_NH4Cl_Zr_mica'):
    #delete the CTR data first
    #file is created using domain_creator.combine_all_datasets()
    L_container=[]
    L_label_container=[]
    data=np.loadtxt(file,comments='#')
    index_container=[]
    L_container=[]
    for i in range(len(data)):
        if i!=len(data)-1:
            if data[i,3] not in L_container:
                index_container.append(i)
                L_container.append(data[i,3])
    index_container.append(len(data))
    sub_data_sets=[]
    for i in range(len(index_container)-1):
        start,end=index_container[i],index_container[i+1]
        sub_data_sets.append(data[start:end,:])
    for i in range(len(sub_data_sets)):
        each_data_set=sub_data_sets[i]
        each_data_set[:,0]=each_data_set[:,0]/1000.
        each_data_set[:,-2]=1
        L=L_container[i]
        L_label_container.append("_L"+str(round(L,2)).replace(".",""))
        header='1             2           3            4            5           6            7                8       \nEngergy       H           K            L            I           Ierr         monitor_dummy    BL'
        np.savetxt(os.path.dirname(file)+'\\'+name_head+L_label_container[-1]+".ipg",each_data_set,fmt='%10.5f',header=header,comments='%')

    return L_container,L_label_container

def split_CTR_RAXR_data_file_fr_GenX_output(file='P:\\apps\\genx_pc_qiu\\dump_files\\temp_full_dataset.dat',name_head='100mM_KCl_Zr_mica_ESRF',c_projected=19.977,wal=0.7515):
    #file is created using domain_creator.combine_all_datasets()
    def _find_CTR_index(data):
        index_container=[]
        for i in range(len(data)):
            if data[i][3]==0:
                index_container.append(i)
            else:
                pass
        return index_container[0],index_container[-1]
    L_container=[]
    L_label_container=[]
    data=np.loadtxt(file,comments='#')
    CTR_index_l,CTR_index_r=_find_CTR_index(data)
    index_container=[]
    L_container=[]
    for i in range(CTR_index_r+1,len(data)):
        if i!=len(data)-1:
            if data[i,3] not in L_container:
                index_container.append(i)
                L_container.append(data[i,3])
    index_container.append(len(data))
    sub_data_sets=[]
    for i in range(len(index_container)-1):
        start,end=index_container[i],index_container[i+1]
        sub_data_sets.append(data[start:end,:])
    for i in range(len(sub_data_sets)):
        each_data_set=sub_data_sets[i]
        each_data_set[:,0]=each_data_set[:,0]/1000.
        each_data_set[:,-2]=1
        L=L_container[i]
        L_label_container.append("_L"+str(round(L,2)).replace(".",""))
        header='1             2           3            4            5           6            7                8       \nEngergy       H           K            L            I           Ierr         monitor_dummy    BL'
        np.savetxt(os.path.dirname(file)+'\\RAXR_'+name_head+L_label_container[-1]+".ipg",each_data_set,fmt='%10.5f',header=header,comments='%')
    CTR_data=np.zeros((0,5))
    CTR_data=np.append(CTR_data,[[c_projected,wal,np.pi*2/c_projected*data[CTR_index_l][0],np.pi*2/c_projected*data[CTR_index_r][0],4.363323e-5]],axis=0)
    for i in range(CTR_index_l,CTR_index_r+1):
        CTR_data=np.append(CTR_data,[[i+1,data[i,0],np.pi*2/c_projected*data[i,0],data[i,4],data[i,5]]],axis=0)
    np.savetxt(os.path.dirname(file)+'\\CTR_'+name_head,CTR_data,fmt='%10.7f',comments='%')
    return L_container,L_label_container,CTR_index_l,CTR_index_r

#E_range=[17807,18265] for Zr
#E_range=[14999,15500] for Rb
def formate_RAXR_data_APS(file_path='M:\\fwog\\members\\qiu05\\March_21_2018_APS\\schmidt\\mica\\mica-Zr_KCl_1_RAXR_Zr_1st_spot.ipg',E_range=[17807,18265]):
    full_data=np.zeros((1,8))
    L_list=[]
    data=np.loadtxt(file_path,comments='%')
    for i in range(len(data)):
        temp_data=[]
        if i!=(len(data)-1) and data[i,0]>data[i+1,0]:
            L_list.append(np.around(data[i,1],2))
            temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
        else:
            if np.around(data[i,1],2) not in L_list:
                temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
        if temp_data!=[] and np.around(data[i,3]*1000,0)>E_range[0] and np.around(data[i,3]*1000,0)<E_range[1]:
            full_data=np.append(full_data,temp_data,axis=0)
    for i in range(len(full_data[1:])):
        each_segment=full_data[1:][i]
        if i!=(len(full_data[1:])-1) and each_segment[0]==full_data[1:][i+1,0]:
            print each_segment[0],each_segment[3]#manually delete those duplicate points in the output file
    #print L_list
    np.savetxt(file_path.replace('.ipg','_GenX_formate.dat'),full_data[1:],fmt='%.5e')


def formate_RAXR_data_APS_multiple_datasets(file_head='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data',E_range=[16151,16450]):
    full_data=np.zeros((1,8))
    L_list=[]
    file_paths=glob.glob(os.path.join(file_head,'*.ipg'))
    for file_path in file_paths:
        data=np.loadtxt(os.path.join(file_head,file_path),comments='%')
        for i in range(len(data)):
            temp_data=[]
            if i!=(len(data)-1) and data[i,0]>data[i+1,0]:
                L_list.append(np.around(data[i,1],2))
                temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
            else:
                if np.around(data[i,1],2) not in L_list:
                    temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
            if temp_data!=[] and np.around(data[i,3]*1000,0)>E_range[0] and np.around(data[i,3]*1000,0)<E_range[1]:
                full_data=np.append(full_data,temp_data,axis=0)
    for i in range(len(full_data[1:])):
        each_segment=full_data[1:][i]
        if i!=(len(full_data[1:])-1) and each_segment[0]==full_data[1:][i+1,0]:
            print each_segment[0],each_segment[3]#manually delete those duplicate points in the output file
        #print L_list
    np.savetxt(os.path.join(file_head,'RAXR_data_GenX_formate.dat'),full_data[1:],fmt='%.5e')

def formate_RAXR_data_APS_single_set(file_path='M:\\fwog\\members\\qiu05\\1704_APS_13IDC\mica\\s3_100mM_NH4Cl_Zr_1_RAXR_2nd_spot1.ipg',E_range=[17807,18265],L=0.61):
    full_data=np.zeros((1,8))
    L_list=[]
    data=np.loadtxt(file_path,comments='%')
    for i in range(len(data)):
        temp_data=[]
        if np.around(data[i,1],2)==L:
            temp_data=[[np.around(data[i,3]*1000,0),0,0,np.around(data[i,1],2),data[i,5]/data[i,4]/data[i,26],data[i,6]/data[i,4]/data[i,26],2,2]]
        if temp_data!=[] and np.around(data[i,3]*1000,0)>E_range[0] and np.around(data[i,3]*1000,0)<E_range[1]:
            full_data=np.append(full_data,temp_data,axis=0)
    for i in range(len(full_data[1:])):
        each_segment=full_data[1:][i]
        if i!=(len(full_data[1:])-1) and each_segment[0]==full_data[1:][i+1,0]:
            print each_segment[0],each_segment[3]#manually delete those duplicate points in the output file
    #print L_list
    np.savetxt(file_path.replace('.ipg','single_L_GenX_formate.dat'),full_data[1:],fmt='%.5e')

def formate_RAXR_data_ESRF(file_path='M:\\fwog\\members\\qiu05\\Jul19_2017_ESRF\\ctr7\\S4_Zr_100mM_RbCl_3_RAXR_1st_point_Rb_1.ipg',E_range=[15000,15499],L_shift=0,e_shift=-3):
    #L_shift:after q correction, L should be corrected somehow. For example, it was L=0.3 while it is now L=0.255 after Q correction, then L_shift=-0.045
    full_data=np.zeros((1,8))
    L_list=[]
    data=np.loadtxt(file_path,comments='%')
    if data[0,3]<1000:
        data[:,3]=data[:,3]*1000
    else:
        pass
    for i in range(len(data)):
        temp_data=[]
        if i!=(len(data)-1) and data[i,0]>data[i+1,0]:
            L_list.append(np.around(data[i,1],2))
            temp_data=[[np.around(data[i,3],0)+e_shift,0,0,np.around(data[i,1],2),data[i,5]/data[i,4],data[i,6]/data[i,4],2,2]]
        else:
            if np.around(data[i,1],2) not in L_list:
                temp_data=[[np.around(data[i,3],0)+e_shift,0,0,np.around(data[i,1],2),data[i,5]/data[i,4],data[i,6]/data[i,4],2,2]]
        if temp_data!=[] and np.around(data[i,3],0)>E_range[0] and np.around(data[i,3],0)<E_range[1]:
            full_data=np.append(full_data,temp_data,axis=0)
    full_data[:,0]=full_data[:,0]+L_shift
    for i in range(len(full_data[1:])):
        each_segment=full_data[1:][i]
        if i!=(len(full_data[1:])-1) and each_segment[0]==full_data[1:][i+1,0]:
            print each_segment[0],each_segment[3]#manually delete those duplicate points in the output file
    np.savetxt(file_path.replace('.ipg','_GenX_formate.dat'),full_data[1:],fmt='%.5e')

def scale_RAXS_data_to_CTR(file_ctr='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\nQc_S0_Zr_0mM_NaCl_Dry_CTR_1st_spot1_R_GenX_formate_Q_corrected.dat',file_raxs='M:\\fwog\\members\\qiu05\\1611 - ROBL20\\S0_Zr_0mM_NaCl_Dry_RAXR_1st_spot1_R_GenX_formate.dat',E_col=0,L_col_raxs=3,E_ctr=16000):
    f_ctr=np.loadtxt(file_ctr)
    f_raxs=np.loadtxt(file_raxs)
    current_L=f_raxs[0,L_col_raxs]
    I_new=f_raxs[0,4]
    scaling=None
    for i in range(len(f_raxs)):
        if abs(f_raxs[i,L_col_raxs]-current_L)>0.01:#a sign move to next segment
            current_L=f_raxs[i,L_col_raxs]
            I_new=f_raxs[i,4]
            scaling=f_ctr[np.argmin(abs(f_ctr[:,0]-current_L)),4]/I_new
        elif abs(f_raxs[i,L_col_raxs]-current_L)<0.01 and scaling==None:#a sign of first segment
            current_L=f_raxs[i,L_col_raxs]
            I_new=f_raxs[0,4]
            scaling=f_ctr[np.argmin(abs(f_ctr[:,0]-current_L)),4]/I_new
        else:
            pass
        f_raxs[i,4]=f_raxs[i,4]*scaling#scaling I
        f_raxs[i,5]=f_raxs[i,5]*scaling#scaling error
    np.savetxt(file_raxs.replace('.dat','_scaled_to_CTR.data'),f_raxs,fmt='%.5e')

#Th Li system
#f1f2_file='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\axd_ThLi_mica'
#ipg_file='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\th_3mM_licl_a_L0041a_RAXR_R.ipg'
def formate_F1F2_data(f1f2_file='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\axd_ThLi_mica',ipg_file='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\th_3mM_licl_a_L0041a_RAXR_R.ipg'):
    f1f2=np.loadtxt(f1f2_file)
    ipg=np.loadtxt(ipg_file,comments='%')
    E_list=[]
    for i in range(len(ipg)):
        if ipg[i,0]<ipg[i-1,0] and i!=0:
            break
        E_list.append(round(ipg[i,3]*1000,0))

    f1f2_new=np.zeros((1,3))
    print E_list
    for i in range(len(f1f2)):
        if round(f1f2[i,0]*1000,0) in E_list:
            f1f2_new=np.append(f1f2_new,(f1f2[i]*[1000,1,1])[np.newaxis,:],axis=0)
    f1f2_new=f1f2_new[1:]
    np.savetxt(f1f2_file+'.formated',f1f2_new[:,[1,2,0]])
    #now drop duplicate points
    data_new=np.loadtxt(f1f2_file+'.formated')
    energy=[]
    data_sub=[]
    for i in range(len(data_new)):
        if round(data_new[i,2],0) not in energy:
            energy.append(round(data_new[i,2],0))
            data_sub.append(list(data_new[i]))
    np.savetxt(f1f2_file+'.formated',data_sub,fmt=['%.6e','%.6e','%i'])
    return None

def formate_F1F2_data_new(f1f2_file='P:\\My stuff\\Manuscripts\\Th mica -moritz\\Th_3mM_LiCl_data\\axd_ThLi_mica',E_col=0,f1_col=1,f2_col=2):
    f1f2=np.loadtxt(f1f2_file)
    if f1f2[0,E_col]<1000:
        scale=1000
    else:
        scale=1
    if scale==1000:
        for i in range(len(f1f2)):
            f1f2[i,E_col]=f1f2[i,E_col]*scale
    E_bounds=[int(np.round(f1f2[0,E_col])),int(np.round(f1f2[-1,E_col]))]
    index_container=[0]
    for each in range(E_bounds[0],E_bounds[1]+1):
        offset=10
        for i in range(index_container[-1],len(f1f2)):
            if abs(f1f2[i,0]-each)<offset:
                offset=abs(f1f2[i,0]-each)
            else:
                index_container.append(i)
                break
    index_container=index_container[1:]
    f1f2_new=np.zeros((len(index_container),0))
    f1f2_new=np.append(f1f2_new,f1f2[index_container,f1_col][:,np.newaxis],axis=1)
    f1f2_new=np.append(f1f2_new,f1f2[index_container,f2_col][:,np.newaxis],axis=1)
    f1f2_new=np.append(f1f2_new,np.round(f1f2[index_container,E_col][:,np.newaxis]),axis=1)
    np.savetxt(f1f2_file+'.formated',f1f2_new,fmt=['%.6e','%.6e','%i'])
    return f1f2_new[:,2]


def formate_F1F2_data_ESRF(f1f2_file="/Users/cqiu/data/ESRF/axd_Zr-A01-bkg_b.f1f2.ASC",ipg_file='/Users/cqiu/data/ESRF/March_2017/Zr_ClO4_RAXR_R.ipg'):
    f1f2=np.loadtxt(f1f2_file)
    ipg=np.loadtxt(ipg_file,comments='%')
    E_list=[]
    for i in range(len(ipg)):
        if ipg[i,0]<ipg[i-1,0] and i!=0:
            break
        E_list.append(round(ipg[i,3],0))

    f1f2_new=np.zeros((1,3))
    print E_list
    for i in range(len(f1f2)):
        if round(f1f2[i,0],0) in E_list:
            f1f2_new=np.append(f1f2_new,f1f2[i][np.newaxis,:],axis=0)
    f1f2_new=f1f2_new[1:]
    np.savetxt(f1f2_file+'.formated',f1f2_new[:,[1,2,0]])
    return None

if __name__=='__main__':
    formate_CTR_data()
    formate_RAXR_data(E_range=[17920,18160])
    formate_F1F2_data()
