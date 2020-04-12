import numpy as num
import numpy as np

def format_hkl(h_,k_,x_,h_list = [1,2,3],k_list=[0]):
    """
    func to deal with the symmetrical shape of 10,30 and 20L rod at positive and negative sides
    
    Arguments:
        h_ {[type]} -- [description]
        k_ {[type]} -- [description]
        x_ {[type]} -- [description]
    
    Keyword Arguments:
        h_list {list} -- [description] (default: {[1,2,3]})
        k_list {list} -- [description] (default: {[0]})
    
    Returns:
        [type] -- [description]
    """
    new_h,new_k,new_x=[],[],[]
    if np.around(h_[0],0) in h_list and np.around(k_[0],0) in k_list:
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

hk=['3_0','2_0','2_1','2_2','2_-1','1_1','1_0','0_2','0_0','-1_0','0_-2','-2_-2','-2_-1','-2_0','-2_1','-1_-1','-3_0']
dl_bl={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},
        '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},
        '0_0':{'segment':[[0,12]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},
        '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},
        '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
#segment seperate the data with different D_L (L difference b/t two continual Bragg's peaks) and B_L (L value for the first Bragg's peak)
#info store the information of dl (left) and bl (right)
#{'segment':[[0,5],[5,8]],'info':[[4,-4.8609],[2,-6.8609]]} means for l from 0 to 5, set dl to be 4 and the bl to be -4.8609
#for l from 5 to 8, set dl and bl to to 2 and -6.8609
#as for why use a negative value, that's  because the l will be negative value when depicted in full single rod (eg -30 and 30 merge to 30) according to feder's law
#then the l with the highest magnitude will be on the leftmost

br_l={'3_0':[1,7],'2_0':[2,4,6],'2_1':[0.8609,4.8609,6.8609],'2_2':[1.7218,3.7218,5.7218,7.7218],'2_-1':[3.1391,5.1391],'1_1':[1.8609,3.8609,5.8609],'1_0':[3,5],'0_2':[1.7218,3.7218,5.7218],'0_0':[2,4],\
    '-1_0':[3,5],'0_-2':[0.2782,2.2782,4.2782,6.2782],'-2_-2':[0.2782,2.2782,4.2782,6.2782],'-2_-1':[3.1391,5.1391],'-2_0':[2,4,6],'-2_1':[0.8609,4.8609,6.8609],'-1_-1':[2.1391,4.1391,6.1391],'-3_0':[1,7]}
       
#deal with the dataset exported from HDF integrator
#two things will be done: delete the first index column, append two more columns at the end, one for dL and the other one for L_br   
def creat_genx_datafile_from_HDF(original_file="C:\\Users\\jackey\\Google Drive\\data\\200uM_Pb_rcut_Nov_2013_new4.lst",new_file_name="C:\\Users\\jackey\\Google Drive\\data\\200uM_Pb_rcut_Nov_2013_new4.dat"):
    data=np.loadtxt(original_file)
    data=np.delete(data,0,1)
    data=np.append(data,np.zeros([data.shape[0],2]),1)
    for sub_set in data:
        label=str(int(sub_set[0]))+'_'+str(int(sub_set[1]))
        if label in hk:
            for i in range(len(dl_bl[label]['segment'])):
                if sub_set[2]>dl_bl[label]['segment'][i][0] and sub_set[2]<dl_bl[label]['segment'][i][1]:
                    sub_set[6],sub_set[5]=dl_bl[label]['info'][i][0],dl_bl[label]['info'][i][1]
                    break
            if sub_set[0]<0 or (sub_set[0]==0 and sub_set[1]<0):
                sub_set[0],sub_set[1],sub_set[2]=-sub_set[0],-sub_set[1],-sub_set[2]
    data=num.delete(data,num.where(data[:,3]==0.)[0],0)
    np.savetxt(new_file_name,data,fmt='%5.3f')
    format_from_CTR_to_RaxsCTR_loader(new_file_name)    

#Make the file loadable by using uaf_CTR_raxr_2.py loader in GenX
#The loader accept a data file with following columns:    
#1st column X values (l value for CTR and E for RAXS); 
#2nd column h values; 
#3rd values k values;
#4th column Y (l value for RAXS but a column wont be used for CTR); 
#5th column intensity; 
#6th column The standard deviation of the intensities;
#7th column L of first Bragg peak;
#8th column L spacing of Bragg peaks.
def format_from_CTR_to_RaxsCTR_loader(filename='D:\\Google Drive\\data\\clean_hematite.dat'):
    data_original=num.loadtxt(filename)
    dim=data_original.shape
    data_new=num.zeros((dim[0],dim[1]+1))
    data_new[:,0]=data_original[:,2]
    data_new[:,1]=data_original[:,0]
    data_new[:,2]=data_original[:,1]
    data_new[:,4]=data_original[:,3]
    data_new[:,5]=data_original[:,4]
    data_new[:,6]=data_original[:,5]
    data_new[:,7]=data_original[:,6]
    num.savetxt(filename+'new_formate',data_new,fmt='%5.3f')
    
#l for bragg peak and valley alternates for each rods with head and end both being of bragg peak position
#in other words:even index are for bragg peak positions, odd index are for valley positions
#key define the (H,L)
br_valley={(0,0):[0,1.13,2,2.83,4,5.03,6,6.79,8,9.48,10,11,12],(0,2):[-8.28,-6.8,-6.28,-5.38,-4.28,-3.03,-2.28,-1.39,-0.28,0.67,1.72,2.67,3.72,4.68,5.72,6.81,7.72],\
           (1,1):[-6.14,-4.8,-4.14,-3.26,-2.14,-1.02,-0.14,0.65,1.86,2.62,3.86,4.73,5.86,6.1,7.86],(2,0):[-8,-6.8,-6,-5.1,-4,-2.55,-2,-1.16,0,1.12,2,2.53,4,5,6,7,8],\
           (1,0):[-7,-5.24,-5,-4.24,-3,0,3,4.2,5,5.3,7],(2,2):[-8.28,-6.7,-6.28,-5.28,-4.28,-3.09,-2.28,-1.1,-0.28,-0.66,1.72,2.8,3.72,4.6,5.72,6.93,7.72,8.05,9.72],\
           (3,0):[-9,-7.5,-7,-5.7,-5,-3,-1,0,1,3.2,5,6,7,7.5,9],(2,-1):[-8.86,-7.38,-6.86,-5.86,-4.86,-2.72,-0.86,1.14,3.14,4.13,5.14,6.33,7.14,7.36,9.14],\
           (2,1):[-9.14,-7.36,-7.14,-6.1,-5.14,-4,-3.14,-1.06,0.86,2.76,4.86,6.03,6.86,7.26,8.86]}
#extract the bragg's peak position from br_valley
br_position_full={}
for key in br_valley.keys():
    br_position_full[key]=br_valley[key][::2]
    
#make dummpy error for a full dataset
#errors are scaled such that the valley zone has a weight of 10 while the near-bragg peak has a weight of 1 when using chi2bar fom function for model fitting
#peak_error in the equation is an arbitrary defined error for the bragg peak position
#STD is an arbitrary standard deviation of the offset between fitting value and real value
def make_dummy_error(datafile='D:\\Google Drive\\data\\200uM_Pb_rcut_Nov_2013_new3_spike_deleted.datnew_formate',br_valley_info=br_valley,STD=0.1,peak_error=5,use_prestandard=True):
    data_old=np.loadtxt(datafile)
    data_new=np.zeros(data_old.shape)[0:0]
    def _find_segment(sep_list=[],value=1):
        sep=[]
        for i in range(len(sep_list)):
            if value>=sep_list[i] and value<sep_list[i+1]:
                if i%2==0:#make sure the left item is the bragg peak position
                    sep=[sep_list[i],sep_list[i+1]]
                else:
                    sep=[sep_list[i+1],sep_list[i]]
                break
            else:pass
        return sep          
                
    for key in br_valley_info.keys():
        condition= (data_old[:,1]==key[0]) & (data_old[:,2]==key[1])
        data_subset=data_old[condition]
        
        for i in range(len(data_subset)):
            l=data_subset[i,0]
            F=data_subset[i,4]
            sep=_find_segment(br_valley_info[key],l)
            index_top=np.where(abs(data_subset[:,0]-sep[0])==np.min(abs(data_subset[:,0]-sep[0])))[0][0]
            F_top=data_subset[index_top,4]
            l_top=data_subset[index_top,0]
            ref_unity,scale=0,0
            if use_prestandard:
                ref_unity=(1000*STD/peak_error)**2
                scale=1+abs(sep[0]-l)/abs(sep[0]-sep[1])*9
            else:
                ref_unity=(F_top*STD/peak_error)**2
                scale=1+abs(l_top-l)/abs(l_top-sep[1])*9         
            dummy_error=1/((scale*ref_unity)**0.5)*F*STD
            data_subset[i,5]=dummy_error
        data_new=np.append(data_new,data_subset,axis=0)
    np.savetxt(datafile.replace('.datnew_formate','_dummy_error.datnew_formate'),data_new,fmt='%5.3f')
    return True
    
#this function willl delete spikes near bragg's peak position
#br_positions_lib is a library hodling the bragg's peak positions for each rod
#affected_offset define the width of the affected zone (br_l plus minus this offset is the points to be considered)
#cutoff_ratio define the conditions to identify a spike, ie if F_current/F_adjacent>cutoff_ratio, be wise to choose this value    
def delete_spike(datafile='D:\\Google Drive\\data\\200uM_Pb_rcut_Nov_2013_new3.datnew_formate',br_positions_lib=br_position_full,affected_offset=0.2,cutoff_ratio=1.6):
    full_data_set=np.loadtxt(datafile)
    data_new=np.zeros(full_data_set.shape)[0:0]
    for key in br_positions_lib.keys():
        br_positions=br_positions_lib[key]
        condition= (full_data_set[:,1]==key[0]) & (full_data_set[:,2]==key[1])
        data_set=full_data_set[condition]
        data_set_sorted=data_set[data_set[:,0].argsort()]
        affected_zone_container={}
        deleted_index=[]
        for i in range(len(data_set_sorted)):
            l=data_set_sorted[i,0]
            br_index=list(abs(np.array(br_positions)-l)).index(min(list(abs(np.array(br_positions)-l))))
            br_l=br_positions[br_index]
            position='right'
            if l<br_l:position='left'
            if position=='right' and l<(br_l+affected_offset) and l>br_l:
                if (position,br_l) in affected_zone_container.keys():
                    affected_zone_container[(position,br_l)].append(i)
                else:
                    affected_zone_container[(position,br_l)]=[i]
            elif position=='left' and l<br_l and l>(br_l-affected_offset):
                if (position,br_l) in affected_zone_container.keys():
                    affected_zone_container[(position,br_l)].append(i)
                else:
                    affected_zone_container[(position,br_l)]=[i]
                    
        for key in affected_zone_container.keys():
            if key[0]=='left':
                for i in range(len(affected_zone_container[key])):
                    fetch_index=affected_zone_container[key][i]
                    if (fetch_index-1)>=0:#ensure it wont cause overflow of index
                        if (data_set_sorted[fetch_index,4]/data_set_sorted[fetch_index-1,4])>cutoff_ratio:
                            deleted_index=deleted_index+range(fetch_index,affected_zone_container[key][-1]+1)
                            break
                        else:
                            pass
            if key[0]=='right':
                for i in range(len(affected_zone_container[key]))[::-1]:
                    fetch_index=affected_zone_container[key][i]
                    if (fetch_index+1)<len(data_set_sorted):#ensure it wont cause overflow of index
                        if (data_set_sorted[fetch_index,4]/data_set_sorted[fetch_index+1,4])>cutoff_ratio:
                            deleted_index=deleted_index+range(affected_zone_container[key][0],fetch_index+1)
                            break
                        else:
                            pass
        data_set_sorted_spike_off=np.delete(data_set_sorted,deleted_index,0)
        data_new=np.append(data_new,data_set_sorted_spike_off,axis=0)
    
    np.savetxt(datafile.replace('.datnew_formate','_spike_deleted.datnew_formate'),data_new,fmt='%5.3f')
    
    return True
    
def format_delete_spike_dummy_error_all_in_all(original_file='D:\\Google Drive\\data\\200uM_As_CMP_rcut_reprocessed_Jul_2014.lst',dummy_error=True):
    creat_genx_datafile_from_HDF(original_file,original_file.replace('.lst','.dat'))
    delete_spike(original_file.replace('.lst','.datnew_formate'))
    if dummy_error:
        make_dummy_error(original_file.replace('.lst','_spike_deleted.datnew_formate'))
    return True
    
def format_from_RAXS_HDF_to_RAXSACTR_loader(filename='D:\\Google Drive\\data\\raxs_pb_anl_rcut_00L_Jul_2014.lst'):
    data_original=num.loadtxt(filename)
    dim=data_original.shape
    data_new=num.zeros((dim[0],dim[1]+1))
    data_new[:,0]=data_original[:,4]*1000
    data_new[:,1]=data_original[:,1]
    data_new[:,2]=data_original[:,2]
    data_new[:,3]=data_original[:,3]
    data_new[:,4]=data_original[:,5]
    data_new[:,5]=data_original[:,6]
    num.savetxt(filename.replace('.lst','.datnew_formate'),data_new,fmt='%6.3f')

#individule files is a list of paths for different RAXS ascall txt file
#new file is the file name to be created
#basically, this function will connect each individule files together and format them to be read using RAXS_CTR_2 data loader in genx
def format_from_RAXS_to_RaxsCTR_loader2(individule_files,new_file):
    full_set=num.zeros((1,8))[0:0]
    for file in individule_files:
        temp_data=num.loadtxt(file)
        V,L=temp_data.shape[0],temp_data.shape[1]
        if L==6:
            hk=str(int(temp_data[0][1]))+'_'+str(int(temp_data[0][2]))
            l=temp_data[0][3]
            dl,bl=0,0
            for i in range(len(dl_bl[hk]['segment'])):
                if l<=dl_bl[hk]['segment'][i][1] and l>dl_bl[hk]['segment'][i][0]:
                    dl,bl=dl_bl[hk]['info'][i]
                    break
                else:pass
            dl_array=num.zeros((V,1))+dl
            bl_array=num.zeros((V,1))+bl
            bl_dl_array=num.append(bl_array,dl_array,axis=1)
            temp_data=num.append(temp_data,bl_dl_array,axis=1)
            if ((temp_data[0,1]<0) or ((temp_data[0,1]==0) and (temp_data[0,2]<0))):
                temp_data[:,1],temp_data[:,2],temp_data[:,3]=-temp_data[:,1],-temp_data[:,2],-temp_data[:,3]
        full_set=num.append(full_set,temp_data,axis=0)
    num.savetxt(new_file,full_set,fmt='%5.3f')    

if __name__=="__main__":
    CTR_file_original='C:\\Users\\jackey\\Google Drive\\data\\Sb_400uM_rcut_Nov_2013.lst'
    #do you want to make dummy error for the data set
    dummy_error=True
    format_delete_spike_dummy_error_all_in_all(original_file=CTR_file_original,dummy_error=dummy_error)
    #You should be able to find a bunch of files in the folder holding the original CTR data file with different extention name (files with .datnew_formate are final ones)
        
        
        