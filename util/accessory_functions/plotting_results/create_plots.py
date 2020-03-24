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
from .color_mate import color_combine_mate as set_color

def local_func():
    return None

def module_path_locator(func=local_func):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(func))))),'dump_files')

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

#calcualte the error for pb complex structure
def output_errors(edge_length=2.7,top_angle=70,error_top_angle=1,error_theta=1,error_delta1=0.02,error_delta2=0.03):
    sin_alpha_left=np.sin(np.deg2rad(top_angle-error_top_angle)/2.)
    sin_alpha_right=np.sin(np.deg2rad(top_angle+error_top_angle)/2.)
    tan_alpha_left=np.tan(np.deg2rad(top_angle-error_top_angle)/2.)
    tan_alpha_right=np.tan(np.deg2rad(top_angle+error_top_angle)/2.)
    print('error of PbO1 bond length:',edge_length/4.*(1./sin_alpha_left-1./sin_alpha_right)+error_delta1)
    print('error of pbO2 bond length:',edge_length/4.*(1./sin_alpha_left-1./sin_alpha_right))
    print('error of PbOdistal bond length:',edge_length/4.*(1./sin_alpha_left-1./sin_alpha_right)+error_delta2)
    print('error of O1PbO2 bond angle:',error_top_angle)
    print('error of O1PbOdistal and O2PbOdistal bond angle:',error_top_angle+error_theta)
    print('error of PbFe seperation:',edge_length/4.*(1./tan_alpha_left-1./tan_alpha_right))
    return None


bl_dl_muscovite_old={'3_0':{'segment':[[0,1],[1,9]],'info':[[2,1],[6,1]]},'2_0':{'segment':[[0,9]],'info':[[2,2.0]]},'2_1':{'segment':[[0,9]],'info':[[4,0.8609]]},'2_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,3.1391],[2,3.1391]]},'1_1':{'segment':[[0,9]],'info':[[2,1.8609]]},'1_0':{'segment':[[0,3],[3,9]],'info':[[6,3],[2,3]]},'0_2':{'segment':[[0,9]],'info':[[2,1.7218]]},\
    '0_0':{'segment':[[0,20]],'info':[[2,2]]},'-1_0':{'segment':[[0,3],[3,9]],'info':[[6,-3],[2,-3]]},'0_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},\
    '-2_-2':{'segment':[[0,9]],'info':[[2,-6.2782]]},'-2_-1':{'segment':[[0,3.1391],[3.1391,9]],'info':[[4,-3.1391],[2,-3.1391]]},'-2_0':{'segment':[[0,9]],'info':[[2,-6]]},\
    '-2_1':{'segment':[[0,4.8609],[4.8609,9]],'info':[[4,-4.8609],[2,-6.8609]]},'-1_-1':{'segment':[[0,9]],'info':[[2,-4.1391]]},'-3_0':{'segment':[[0,1],[1,9]],'info':[[2,-1],[6,-1]]}}
bl_dl_muscovite={'0_0':{'segment':[[0,20]],'info':[[2,2]]}}



def generate_plot_files(output_file_path,sample,rgh,data,fit_mode, z_min=0,z_max=29,RAXR_HKL=[0,0,20],bl_dl=bl_dl_muscovite,height_offset=0,version=1,freeze=False):
    plot_data_container_experiment={}
    plot_data_container_model={}
    plot_raxr_container_experiment={}
    plot_raxr_container_model={}
    A_list_Fourier_synthesis=[]
    P_list_Fourier_synthesis=[]
    HKL_list_raxr=[[],[],[]]
    A_list_calculated,P_list_calculated,Q_list_calculated=sample.find_A_P_muscovite(h=RAXR_HKL[0],k=RAXR_HKL[1],l=RAXR_HKL[2])
    i=0
    for data_set in data:
        f=np.array([])
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        I=data_set.y
        eI=data_set.error
        if x[0]>100:
            i+=1
            A_key_list,P_key_list=[key for key in sample.domain['raxs_vars'].keys() if 'A'+str(i)+'_D' in key and 'set' not in key and 'get' not in key],[key for key in sample.domain['raxs_vars'].keys() if 'P'+str(i)+'_D' in key and 'set' not in key and 'get' not in key]
            A_key_list.sort(),P_key_list.sort()
            A_list_Fourier_synthesis.append(sample.domain['raxs_vars'][A_key_list[0]])
            P_list_Fourier_synthesis.append(sample.domain['raxs_vars'][P_key_list[0]])
            if not data_set.use:
                A_list_Fourier_synthesis[-1]=0
            q=np.pi*2*sample.unit_cell.abs_hkl(h,k,y)
            rough = (1-rgh.beta)**2/(1+rgh.beta**2 - 2*rgh.beta*np.cos(q*sample.unit_cell.c*np.sin(np.pi-sample.unit_cell.beta)/2))
            #try:
            #    exp_const,rgh.mu,re,auc=sample.domain['exp_factors']
            #    pre_factor=3e6*np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            #except:
            #    pre_factor=1
            if version>=1.2:
                exp_const,mu,re,auc,ra_conc=sample.domain['exp_factors']
                pre_factor=3e6*np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            elif version>=1.1:
                exp_const,mu,re,auc=sample.domain['exp_factors']
                pre_factor=3e6*np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            else:
                pre_factor=1
            f=abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=fit_mode,height_offset=height_offset,version=version))
            f=rough*pre_factor*f*f
            label=str(int(h[0]))+'_'+str(int(k[0]))+'_'+str(y[0])
            plot_raxr_container_experiment[label]=np.concatenate((x[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_raxr_container_model[label]=np.concatenate((x[:,np.newaxis],f[:,np.newaxis]),axis=1)
            HKL_list_raxr[0].append(h[0])
            HKL_list_raxr[1].append(k[0])
            HKL_list_raxr[2].append(y[0])
        else:
            f=np.array([])
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            I=data_set.y
            eI=data_set.error
            #make dumy hkl and f to make the plot look smoother
            if l[0]>0:
                l_dumy=np.arange(0.22,l[-1]+0.1,0.1)
            else:
                l_dumy=np.arange(l[0],l[-1]+0.1,0.1)
            N=len(l_dumy)
            h_dumy=np.array([h[0]]*N)
            k_dumy=np.array([k[0]]*N)
            q_dumy=np.pi*2*sample.unit_cell.abs_hkl(h_dumy,k_dumy,l_dumy)
            rough_dumy = (1-rgh.beta)**2/(1+rgh.beta**2 - 2*rgh.beta*np.cos(q_dumy*sample.unit_cell.c*np.sin(np.pi-sample.unit_cell.beta)/2))
            q_data=np.pi*2*sample.unit_cell.abs_hkl(h,k,l)

            LB_dumy=[]
            dL_dumy=[]
            f_dumy=[]

            for j in range(N):
                key=None
                if l_dumy[j]>=0:
                    key=str(int(h[0]))+'_'+str(int(k[0]))
                else:key=str(int(-h[0]))+'_'+str(int(-k[0]))
                for ii in bl_dl[key]['segment']:
                    if abs(l_dumy[j])>=ii[0] and abs(l_dumy[j])<ii[1]:
                        n=bl_dl[key]['segment'].index(ii)
                        LB_dumy.append(bl_dl[key]['info'][n][1])
                        dL_dumy.append(bl_dl[key]['info'][n][0])
            LB_dumy=np.array(LB_dumy)
            dL_dumy=np.array(dL_dumy)

            f_dumy=abs(sample.calculate_structure_factor(h_dumy,k_dumy,l_dumy,None,index=0,fit_mode=fit_mode,height_offset=height_offset,version=version))
            #try:
            #    exp_const,mu,re,auc=sample.domain['exp_factors']
            #    pre_factor=3e6*np.exp(-exp_const*mu/q_dumy)*(4*np.pi*re/auc)**2/q_dumy**2
            #except:
            #    pre_factor=1
            if version>=1.2:
                exp_const,mu,re,auc,ra_conc=sample.domain['exp_factors']
                pre_factor=3e6*np.exp(-exp_const*mu/q_dumy)*(4*np.pi*re/auc)**2/q_dumy**2
            elif version>=1.1:
                exp_const,mu,re,auc=sample.domain['exp_factors']
                pre_factor=3e6*np.exp(-exp_const*mu/q_dumy)*(4*np.pi*re/auc)**2/q_dumy**2
            else:
                pre_factor=1
            f_dumy=rough_dumy*pre_factor*f_dumy*f_dumy
            c_projected_on_z=sample.unit_cell.vol()/(sample.unit_cell.a*sample.unit_cell.b*np.sin(sample.unit_cell.gamma))
            f_ctr=lambda q:(q*np.sin(q*c_projected_on_z/4))**2
            #f_ctr=lambda q:(np.sin(q*19.96/4))**2
            f_dumy_norm=f_dumy*f_ctr(q_dumy)
            label=str(int(h[0]))+str(int(k[0]))+'L'
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis],(I*f_ctr(q_data))[:,np.newaxis],(eI*f_ctr(q_data))[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis],f_dumy_norm[:,np.newaxis]),axis=1)
    Q_list_Fourier_synthesis=np.pi*2*sample.unit_cell.abs_hkl(np.array(HKL_list_raxr[0]),np.array(HKL_list_raxr[1]),np.array(HKL_list_raxr[2]))

    #A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=list(HKL_list_raxr[0]),k=list(HKL_list_raxr[1]),l=list(HKL_list_raxr[2]))
    A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=HKL_list_raxr[0][0],k=HKL_list_raxr[1][0],l=HKL_list_raxr[2][-1])

    #dump CTR data and profiles
    hkls=['00L']
    plot_data_list=[]
    for hkl in hkls:
        plot_data_list.append([plot_data_container_experiment[hkl],plot_data_container_model[hkl]])
    pickle.dump(plot_data_list,open(os.path.join(output_file_path,"temp_plot"),"wb"))
    #dump raxr data and profiles
    pickle.dump([plot_raxr_container_experiment,plot_raxr_container_model],open(os.path.join(output_file_path,"temp_plot_raxr"),"wb"))
    pickle.dump([[A_list_calculated,P_list_calculated,Q_list_calculated],[A_list_Fourier_synthesis,P_list_Fourier_synthesis,Q_list_Fourier_synthesis]],open(os.path.join(output_file_path,"temp_plot_raxr_A_P_Q"),"wb"))
    #dump electron density profiles
    #e density based on model fitting
    water_scaling=sample.plot_electron_density_muscovite(sample.domain,file_path=output_file_path,z_min=z_min,z_max=z_max,N_layered_water=100,height_offset=height_offset,version=version,freeze=freeze)#dumpt file name is "temp_plot_eden"
    #e density based on Fourier synthesis
    z_plot,eden_plot,eden_domains=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_Fourier_synthesis).transpose(),np.array(A_list_Fourier_synthesis).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000,water_scaling=water_scaling)
    #z_plot_sub,eden_plot_sub,eden_domains_sub=sample.fourier_synthesis(np.array(HKL_list_raxr),np.array(P_list_calculated_sub).transpose(),np.array(A_list_calculated_sub).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000,water_scaling=water_scaling)
    z_plot_sub,eden_plot_sub,eden_domains_sub=sample.fourier_synthesis(np.array([[HKL_list_raxr[0][0]]*100,[HKL_list_raxr[1][0]]*100,np.arange(0,HKL_list_raxr[2][-1],HKL_list_raxr[2][-1]/100.)]),np.array(P_list_calculated_sub).transpose(),np.array(A_list_calculated_sub).transpose(),z_min=z_min,z_max=z_max,resonant_el=sample.domain['el'],resolution=1000)
    pickle.dump([z_plot,eden_plot,eden_domains],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis"),"wb"))
    pickle.dump([z_plot_sub,eden_plot_sub,eden_domains_sub],open(os.path.join(output_file_path,"temp_plot_eden_fourier_synthesis_sub"),"wb"))
    pickle.dump([water_scaling*0.25,water_scaling*0.75,water_scaling],open(os.path.join(output_file_path,"water_scaling"),"wb"))

#a function to make files to generate vtk files
def generate_plot_files_2(output_file_path,sample,rgh,data,fit_mode, z_min=0,z_max=29,RAXR_HKL=[0,0,20],bl_dl=bl_dl_muscovite,height_offset=0,tag=1):
    plot_data_container_experiment={}
    plot_data_container_model={}
    plot_raxr_container_experiment={}
    plot_raxr_container_model={}
    A_list_Fourier_synthesis=[]
    P_list_Fourier_synthesis=[]
    HKL_list_raxr=[[],[],[]]
    A_list_calculated,P_list_calculated,Q_list_calculated=sample.find_A_P_muscovite(h=RAXR_HKL[0],k=RAXR_HKL[1],l=RAXR_HKL[2])
    i=0
    for data_set in data:
        f=np.array([])
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']
        I=data_set.y
        eI=data_set.error
        if x[0]>100:
            i+=1
            A_key_list,P_key_list=[key for key in sample.domain['raxs_vars'].keys() if 'A'+str(i)+'_D' in key and 'set' not in key and 'get' not in key],[key for key in sample.domain['raxs_vars'].keys() if 'P'+str(i)+'_D' in key and 'set' not in key and 'get' not in key]
            A_key_list.sort(),P_key_list.sort()
            A_list_Fourier_synthesis.append(sample.domain['raxs_vars'][A_key_list[0]])
            P_list_Fourier_synthesis.append(sample.domain['raxs_vars'][P_key_list[0]])
            rough = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5
            f=rough*abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=fit_mode,height_offset=height_offset))
            f=f*f
            label=str(int(h[0]))+'_'+str(int(k[0]))+'_'+str(y[0])
            plot_raxr_container_experiment[label]=np.concatenate((x[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis]),axis=1)
            plot_raxr_container_model[label]=np.concatenate((x[:,np.newaxis],f[:,np.newaxis]),axis=1)
            HKL_list_raxr[0].append(h[0])
            HKL_list_raxr[1].append(k[0])
            HKL_list_raxr[2].append(y[0])
        else:
            f=np.array([])
            h = data_set.extra_data['h']
            k = data_set.extra_data['k']
            l = data_set.x
            LB = data_set.extra_data['LB']
            dL = data_set.extra_data['dL']
            I=data_set.y
            eI=data_set.error
            #make dumy hkl and f to make the plot look smoother
            if l[0]>0:
                l_dumy=np.arange(0.05,l[-1]+0.1,0.1)
            else:
                l_dumy=np.arange(l[0],l[-1]+0.1,0.1)
            N=len(l_dumy)
            h_dumy=np.array([h[0]]*N)
            k_dumy=np.array([k[0]]*N)
            q_dumy=np.pi*2*sample.unit_cell.abs_hkl(h_dumy,k_dumy,l_dumy)
            q_data=np.pi*2*sample.unit_cell.abs_hkl(h,k,l)
            LB_dumy=[]
            dL_dumy=[]
            f_dumy=[]

            for j in range(N):
                key=None
                if l_dumy[j]>=0:
                    key=str(int(h[0]))+'_'+str(int(k[0]))
                else:key=str(int(-h[0]))+'_'+str(int(-k[0]))
                for ii in bl_dl[key]['segment']:
                    if abs(l_dumy[j])>=ii[0] and abs(l_dumy[j])<ii[1]:
                        n=bl_dl[key]['segment'].index(ii)
                        LB_dumy.append(bl_dl[key]['info'][n][1])
                        dL_dumy.append(bl_dl[key]['info'][n][0])
            LB_dumy=np.array(LB_dumy)
            dL_dumy=np.array(dL_dumy)
            rough_dumy = (1-rgh.beta)/((1-rgh.beta)**2 + 4*rgh.beta*np.sin(np.pi*(l_dumy-LB_dumy)/dL_dumy)**2)**0.5
            f_dumy=rough_dumy*abs(sample.calculate_structure_factor(h_dumy,k_dumy,l_dumy,None,index=0,fit_mode=fit_mode,height_offset=height_offset))
            f_dumy=f_dumy*f_dumy
            c_projected_on_z=sample.unit_cell.c*np.sin(np.pi-sample.unit_cell.beta)
            f_ctr=lambda q:(np.sin(q*c_projected_on_z/4))**2
            #f_ctr=lambda q:(np.sin(q*19.96/4))**2
            f_dumy_norm=f_dumy*f_ctr(q_dumy)
            label=str(int(h[0]))+str(int(k[0]))+'L'
            plot_data_container_experiment[label]=np.concatenate((l[:,np.newaxis],I[:,np.newaxis],eI[:,np.newaxis],(I*f_ctr(q_data))[:,np.newaxis],(eI*f_ctr(q_data))[:,np.newaxis]),axis=1)
            plot_data_container_model[label]=np.concatenate((l_dumy[:,np.newaxis],f_dumy[:,np.newaxis],f_dumy_norm[:,np.newaxis]),axis=1)
    Q_list_Fourier_synthesis=np.pi*2*sample.unit_cell.abs_hkl(np.array(HKL_list_raxr[0]),np.array(HKL_list_raxr[1]),np.array(HKL_list_raxr[2]))

    A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=list(HKL_list_raxr[0]),k=list(HKL_list_raxr[1]),l=list(HKL_list_raxr[2]))
    #A_list_calculated_sub,P_list_calculated_sub,Q_list_calculated_sub=sample.find_A_P_muscovite(h=HKL_list_raxr[0][0],k=HKL_list_raxr[1][0],l=HKL_list_raxr[2][-1])

    #output files
    #CTR
    np.savetxt('D://temp_CTR'+str(tag),plot_data_container_model['00L'])
    #RAXR
    keys=plot_raxr_container_model.keys()
    keys.sort()
    np.savetxt('D://temp_RAXR'+str(tag),plot_raxr_container_model[keys[0]])
    #Fourier components
    #print A_list_calculated
    ap_data=np.concatenate((A_list_calculated[:,np.newaxis],P_list_calculated[:,np.newaxis],Q_list_calculated[:,np.newaxis]),axis=1)
    np.savetxt('D://temp_APQ'+str(tag),ap_data)

#this function must be called within the shell of GenX gui and par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q' by default
#The purpose of this function is to append the errors of A and P extracted from the errors displaying inside the tab of GenX gui
#copy and past this command line to the shell for action:
#model.script_module.create_plots.append_errors_for_A_P(par_instance=model.parameters,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs')
def append_errors_for_A_P_original(par_instance,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs'):
    data_AP_Q=pickle.load(open(dump_file,"rb"))
    AP_calculated=data_AP_Q[0]
    A_model_fit,P_model_fit=data_AP_Q[1][0],data_AP_Q[1][1]
    A_error_model_fit,P_error_model_fit=[],[]
    table=np.array(par_instance.data)
    for i in range(len(A_model_fit)):
        A_error_model_fit_domain=[]
        for j in range(len(A_model_fit[i])):
            par_name=raxs_rgh+'.setA'+str(i+1)+'_D'+str(j+1)
            for k in range(len(table)):
                if table[k][0]==par_name:
                    if table[k][5][0]=='(' and table[k][5][-1]==')':
                        error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                        A_error_model_fit_domain.append(error)
                    else:
                        A_error_model_fit_domain.append(np.array([0.1,0.1]))
        A_error_model_fit.append(A_error_model_fit_domain)
    for i in range(len(P_model_fit)):
        P_error_model_fit_domain=[]
        for j in range(len(P_model_fit[i])):
            par_name=raxs_rgh+'.setP'+str(i+1)+'_D'+str(j+1)
            for k in range(len(table)):
                if table[k][0]==par_name:
                    if table[k][5][0]=='(' and table[k][5][-1]==')':
                        error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                        P_error_model_fit_domain.append(error)
                    else:
                        P_error_model_fit_domain.append(np.array([0.1,0.1]))
        P_error_model_fit.append(P_error_model_fit_domain)
    dump_data=[[AP_calculated[0],AP_calculated[1],AP_calculated[2]],[data_AP_Q[1][0],data_AP_Q[1][1],data_AP_Q[1][2],A_error_model_fit,P_error_model_fit]]
    pickle.dump(dump_data,open(dump_file,"wb"))

def append_errors_for_A_P(par_instance,dump_file='D://temp_plot_raxr_A_P_Q',raxs_rgh='rgh_raxs'):
    data_AP_Q=pickle.load(open(dump_file,"rb"))
    AP_calculated=data_AP_Q[0]
    A_model_fit,P_model_fit=data_AP_Q[1][0],data_AP_Q[1][1]
    A_error_model_fit,P_error_model_fit=[],[]
    table=np.array(par_instance.data)
    for i in range(len(A_model_fit)):
        par_name=raxs_rgh+'.setA'+str(i+1)+'_D'+str(1)
        for k in range(len(table)):
            if table[k][0]==par_name:
                if table[k][5][0]=='(' and table[k][5][-1]==')':
                    error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                    A_error_model_fit.append(error)
                else:
                    A_error_model_fit.append(np.array([0.1,0.1]))
    for i in range(len(P_model_fit)):
        par_name=raxs_rgh+'.setP'+str(i+1)+'_D'+str(1)
        for k in range(len(table)):
            if table[k][0]==par_name:
                if table[k][5][0]=='(' and table[k][5][-1]==')':
                    error=[abs(eval(table[k][5])[0]),abs(eval(table[k][5])[1])]
                    P_error_model_fit.append(error)
                else:
                    P_error_model_fit.append(np.array([0.1,0.1]))
    dump_data=[[AP_calculated[0],AP_calculated[1],AP_calculated[2]],[data_AP_Q[1][0],data_AP_Q[1][1],data_AP_Q[1][2],A_error_model_fit,P_error_model_fit]]
    pickle.dump(dump_data,open(dump_file,"wb"))

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

def plotting_raxr_multiple(file_head=module_path_locator(),dump_files=['temp_plot_raxr_0NaCl','temp_plot_raxr_1NaCl','temp_plot_raxr_10NaCl','temp_plot_raxr_100NaCl'],label_marks=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl'],number=9,color_type=1,marker=['o']):
    color=set_color(len(dump_files),color_type)
    datas=[pickle.load(open(os.path.join(file_head,file))) for file in dump_files]
    fig=pyplot.figure(figsize=(15,10))
    for i in range(number):
        ax=fig.add_subplot(3,3,i+1)
        for j in range(len(datas)):
            data=datas[j]
            experiment_data,model=data[0],data[1]
            labels=model.keys()
            labels.sort()
            label_tag=map(lambda x:float(x.split("_")[-1]),labels)
            label_tag.sort()
            labels=map(lambda x:"0_0_"+str(x),label_tag)
            labels_comp=datas[0][1].keys()
            labels_comp.sort()
            offset=model[labels[i]][:,1][0]-datas[0][1][labels_comp[i]][:,1][0]-j*0.2#arbitrary offset between datasets
            #print labels[i],offset
            ax.scatter(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1]-offset,marker=marker[0],s=15,c=color[j],edgecolors=color[j],label=label_marks[j])
            ax.errorbar(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1]-offset,yerr=experiment_data[labels[i]][:,2],fmt=None,ecolor=color[j])
            ax.plot(model[labels[i]][:,0],model[labels[i]][:,1]-offset,color=color[j],lw=1.5)
            if i in [6,7,8]:
                pyplot.xlabel('Energy (ev)',axes=ax,fontsize=12)
            if i in [0,3,6]:
                pyplot.ylabel('|F|',axes=ax,fontsize=12)
            if j==0:
                pyplot.title(labels[i],size=12)
            if i==0:
                ax.legend(loc=2,ncol=2,prop={'size':12})
                pyplot.ylim((1.5,3.5))
    pyplot.subplots_adjust(wspace=0.2, hspace=None)
    #fig.tight_layout()

    fig.savefig(os.path.join(file_head,'multiple_raxrs.png'),dpi=300)
    return fig

def plotting_raxr_multiple_2(file_head=module_path_locator(),dump_files=['temp_plot_raxr_0NaCl','temp_plot_raxr_1NaCl','temp_plot_raxr_10NaCl','temp_plot_raxr_100NaCl'],label_marks=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl'],number_raxr=[0,1,6],color_type=1,marker=['o'],plot_layout=[3,1],fig_size=(4.5,6)):
    color=set_color(len(dump_files),color_type)
    datas=[pickle.load(open(os.path.join(file_head,file))) for file in dump_files]
    fig=pyplot.figure(figsize=fig_size)
    scale=0.3141#q=scale*L, scale=2pi/d[001]
    number=None
    if type(number_raxr)==type(1):
        number=range(number_raxr)
    else:
        number=number_raxr
    for i in number:
        ax=fig.add_subplot(plot_layout[0],plot_layout[1],number.index(i)+1)
        for j in range(len(datas)):
            data=datas[j]
            experiment_data,model=data[0],data[1]
            labels=model.keys()
            labels.sort()
            label_tag=map(lambda x:float(x.split("_")[-1]),labels)
            label_tag.sort()
            labels=map(lambda x:"0_0_"+str(x),label_tag)
            labels_title=map(lambda x:"q="+str(x)+r'$\rm{\ \AA^{-1}}$',np.round(np.array(label_tag)*scale,3))
            #labels=map(lambda x:str(x),list(np.array(label_tag)*scale))
            labels_comp=datas[0][1].keys()
            labels_comp.sort()
            offset=model[labels[i]][:,1][0]-datas[0][1][labels_comp[i]][:,1][0]-j*0.15#arbitrary offset between datasets
            #print labels[i],offset
            ax.scatter(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1]-offset,marker=marker[0],s=15,c=color[j],edgecolors=color[j],label=label_marks[j])
            ax.errorbar(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1]-offset,yerr=experiment_data[labels[i]][:,2],fmt=None,ecolor=color[j])
            ax.plot(model[labels[i]][:,0],model[labels[i]][:,1]-offset,color=color[j],lw=1.5)
            pyplot.ylabel(r'|F|',axes=ax,fontsize=12)
            if i!=number[-1]:
                ax.get_xaxis().set_ticks([])
            if i == number[-1]:
                pyplot.xlabel(r'Energy (ev)',axes=ax,fontsize=12)
            if j==0:
                pyplot.title(labels_title[i],size=12)
            #if i==0:
            #    ax.legend(loc=2,ncol=2,prop={'size':12})
            #    pyplot.ylim((1.5,3.5))
    pyplot.subplots_adjust(wspace=0.2, hspace=None)
    #fig.tight_layout()

    fig.savefig(os.path.join(file_head,'multiple_raxrs.png'),dpi=300)
    return fig

def plotting_raxr_multiple_full_set(file_head=module_path_locator(),dump_files=['temp_plot_raxr_0NaCl','temp_plot_raxr_1NaCl','temp_plot_raxr_10NaCl','temp_plot_raxr_100NaCl'],label_marks=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl'],number_raxr=range(9),color_type=1,marker=['o'],plot_layout=[3,3],fig_size=(12,8)):
    color=set_color(len(dump_files),color_type)
    datas=[pickle.load(open(os.path.join(file_head,file))) for file in dump_files]
    fig=pyplot.figure(figsize=fig_size)
    scale=0.3141#q=scale*L, scale=2pi/d[001]
    number=None
    if type(number_raxr)==type(1):
        number=range(number_raxr)
    else:
        number=number_raxr
    for i in number:
        ax=fig.add_subplot(plot_layout[0],plot_layout[1],number.index(i)+1)
        for j in range(len(datas)):
            data=datas[j]
            experiment_data,model=data[0],data[1]
            labels=model.keys()
            labels.sort()
            label_tag=map(lambda x:float(x.split("_")[-1]),labels)
            label_tag.sort()
            labels=map(lambda x:"0_0_"+str(x),label_tag)
            labels_title=map(lambda x:"q="+str(x)+r'$\rm{\ \AA^{-1}}$',np.round(np.array(label_tag)*scale,3))
            #labels=map(lambda x:str(x),list(np.array(label_tag)*scale))
            labels_comp=datas[0][1].keys()
            labels_comp.sort()
            offset=model[labels[i]][:,1][0]-datas[0][1][labels_comp[i]][:,1][0]-j*0.15#arbitrary offset between datasets
            #print labels[i],offset
            ax.scatter(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1]-offset,marker=marker[0],s=15,c=color[j],edgecolors=color[j],label=label_marks[j])
            ax.errorbar(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1]-offset,yerr=experiment_data[labels[i]][:,2],fmt=None,ecolor=color[j])
            ax.plot(model[labels[i]][:,0],model[labels[i]][:,1]-offset,color=color[j],lw=1.5)
            if number.index(i) in [0,3,6]:
                pyplot.ylabel(r'|F|',axes=ax,fontsize=12)
            if number.index(i) not in [6,7,8]:
                ax.get_xaxis().set_ticks([])
            if number.index(i) in [6,7,8]:
                pyplot.xlabel(r'Energy (ev)',axes=ax,fontsize=12)
            if j==0:
                pyplot.title(labels_title[i],size=12)
            #if i==0:
            #    ax.legend(loc=2,ncol=2,prop={'size':12})
            #    pyplot.ylim((1.5,3.5))
    pyplot.subplots_adjust(wspace=0.2, hspace=None)
    #fig.tight_layout()

    fig.savefig(os.path.join(file_head,'multiple_raxrs.png'),dpi=300)
    return fig

def plot_CTR_multiple_model_muscovite(file_head=module_path_locator(),dump_files=['temp_plot_0NaCl','temp_plot_1NaCl','temp_plot_10NaCl','temp_plot_100NaCl'],labels=['0NaCl','1mM NaCl','10mM NaCl','100mM NaCl'],markers=['.']*20,fontsize=16,lw=1.5,color_type=1):
    colors=set_color(len(dump_files)*2,color_type)
    objects=[pickle.load(open(os.path.join(file_head,file))) for file in dump_files]
    fig=pyplot.figure(figsize=(10,8))
    ax=fig.add_subplot(2,1,1)
    ax.set_yscale('log')
    scale=0.3141
    for i in range(len(objects)):
        object=objects[i][0]
        ax.scatter(object[0][:,0]*scale,object[0][:,1]*(10**i),marker=markers[i],s=20,facecolors='none',edgecolors=colors[i],label='Data_'+labels[i])
        ax.errorbar(object[0][:,0]*scale,object[0][:,1]*(10**i),yerr=object[0][:,2],fmt=None,ecolor=colors[i])
        l,=ax.plot(object[1][:,0]*scale,object[1][:,1]*(10**i),color=colors[i],lw=lw,label='Model_'+labels[i])
        #l.set_dashes(l_dashes[i])
        #pyplot.xlabel('L(r.l.u)',axes=ax,fontsize=fontsize)
        pyplot.ylabel(r'$\rm{|F_{HKL}|}$',axes=ax,fontsize=fontsize)
        pyplot.title('(00L)',weight=4,size=fontsize,clip_on=True)
    ax.legend(prop={'size':fontsize})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)
    ax.plot([0.4,0.4],[0,10000],'--',color='black')
    ax.plot([0.9,0.9],[0,10000],'--',color='black')
    pyplot.xlim((-2*scale,30*scale))
    ax=fig.add_subplot(2,1,2)
    ax.set_yscale('log')
    for i in range(len(objects)):
        object=objects[i][0]
        ax.scatter(object[0][:,0]*scale,object[0][:,3]*(10**i),marker=markers[i],s=20,facecolors='none',edgecolors=colors[i],label='Data_'+labels[i])
        ax.errorbar(object[0][:,0]*scale,object[0][:,3]*(10**i),yerr=object[0][:,4],fmt=None,ecolor=colors[i])
        l,=ax.plot(object[1][:,0]*scale,object[1][:,2]*(10**i),color=colors[i],lw=lw,label='Model_'+labels[i])
        #l.set_dashes(l_dashes[i])
        pyplot.xlabel(r'$\rm{q\ (\AA^{-1})}$',axes=ax,fontsize=fontsize)
        pyplot.ylabel(r'$\rm{|normalized\ F_{HKL}|}$',axes=ax,fontsize=fontsize)
        #pyplot.title('(00L)',weight=4,size=10,clip_on=True)
    ax.plot([0.4,0.4],[0,10000],'--',color='black')
    ax.plot([0.9,0.9],[0,10000],'--',color='black')
    #ax.plot([2.92*scale,2.92*scale],[0,10000],'--',color='black')
    ax.legend(prop={'size':fontsize})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)
    pyplot.xlim((-2*scale,30*scale))
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_ctrs.png'),dpi=300)
    return fig

def plot_CTR_multiple_model_muscovite_2(file_head=module_path_locator(),dump_files=['CTR_100mL_LiCl_Zr_mica_APS','CTR_100mM_KCl_Zr_mica_ESRF','CTR_Zr_case_100mLRbCl_Zr_mica_APS','CTR_100mL_CsCl_Zr_mica_APS'],labels=['100LiCl','100KCl','100RbCl','100CsCl'],markers=['.']*20,fontsize=12,lw=1.5,color_type=1):
    colors=set_color(len(dump_files)*2,color_type)
    objects=[pickle.load(open(os.path.join(file_head,file))) for file in dump_files]
    fig=pyplot.figure(figsize=(5,4))
    ax=fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    scale=0.3141#q=scale*L, scale=2pi/d[001]
    for i in range(len(objects)):
        object=objects[i][0]
        ax.scatter(object[0][:,0]*scale,object[0][:,1]*(10**i),marker=markers[i],s=20,facecolors='none',edgecolors=colors[i],label='Data_'+labels[i])
        ax.errorbar(object[0][:,0]*scale,object[0][:,1]*(10**i),yerr=object[0][:,2],fmt=None,ecolor=colors[i])
        l,=ax.plot(object[1][:,0]*scale,object[1][:,1]*(10**i),color=colors[i],lw=lw,label='Model_'+labels[i])
        #l.set_dashes(l_dashes[i])
        #pyplot.xlabel('L(r.l.u)',axes=ax,fontsize=fontsize)
        pyplot.xlabel(r'$\rm{q\ (\AA^{-1})}$',axes=ax,fontsize=fontsize)
        pyplot.ylabel(r'$\rm{|F_{HKL}|}$',axes=ax,fontsize=fontsize)
        #pyplot.title('(00L)',weight=4,size=fontsize,clip_on=True)
    #ax.legend(prop={'size':fontsize})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)
    ax.plot([0.35,0.35],[0,10000],':',color='black')
    ax.plot([0.87,0.87],[0,10000],':',color='black')
    pyplot.xlim((0,5.5))
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_ctrs.png'),dpi=300)
    return fig
#CTR results for ZrO2 NP aggregation mechanism as a function of ionic strength
file_head_IS=module_path_locator()
dump_file_CTR_IS=[['ctr_data_0mM_NaCl.dat','bestfit_ctr_results_0mM_NaCl.dat'],['ctr_data_1mM_NaCl.dat','bestfit_ctr_results_1mM_NaCl.dat'],['ctr_data_10mM_NaCl.dat','bestfit_ctr_results_10mM_NaCl.dat'],['ctr_data_100mM_NaCl.dat','bestfit_ctr_results_100mM_NaCl.dat']]
lable_CTR_IS=['0NaCl','1mM NaCl','10mM NaCl','100mM NaCl']
c_projected_IS=[19.9347,19.9597,19.9167,19.9803]
#CTR results for cation effect of ZrNP sorption on mica
file_head_CE='P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\Matlab models\\results_in_all\Feb8_2018'
dump_file_CTR_CE=[['ctr_data_100mM_LiCl.dat','bestfit_ctr_results_100mM_LiCl.dat'],['ctr_data_100mM_NaCl.dat','bestfit_ctr_results_100mM_NaCl.dat'],['ctr_data_100mM_KCl.dat','bestfit_ctr_results_100mM_KCl.dat'],['ctr_data_100mM_RbCl.dat','bestfit_ctr_results_100mM_RbCl.dat'],['ctr_data_100mM_CsCl.dat','bestfit_ctr_results_100mM_CsCl.dat']]
lable_CTR_CE=['0.1 M LiCl','0.1 M NaCl','0.1 M KCl','0.1 M RbCl','0.1 M CsCl']
c_projected_ES=[19.9171,19.9803,19.977,19.9573,19.977]
def plot_CTR_multiple_model_muscovite_matlab_outputs(file_head=file_head_CE,c_projected=c_projected_ES,dump_files=dump_file_CTR_CE,labels=lable_CTR_CE,markers=['.']*10,fontsize=13,lw=0.5,color_type=[1,6]):
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    colors=set_color(len(dump_files),color_type[0])
    #colors_2=set_color(len(dump_files)*2,color_type[0])
    colors_2=set_color(len(dump_files),color_type[0])
    objects=[]
    for i in range(len(dump_files)):
        print(i)
        objects.append([np.loadtxt(os.path.join(file_head,dump_files[i][0]),comments='%'),np.loadtxt(os.path.join(file_head,dump_files[i][1]),comments='%')])
    fig=pyplot.figure(figsize=(6,4))
    ax=fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    def _find_index_of_near_bragg_peak(q_list=[],l_bragg=[0,2,4,6,8,10,12,14,16],c=19.9347):
        index_list=[0]
        q_bragg=np.array(l_bragg)*np.pi*2/c
        for i in range(len(q_list)):
            if i!=len(q_list)-1:
                for q in q_bragg:
                    if q_list[i]<q and q_list[i+1]>q:
                        index_list.append(i+1)
                        break
        index_list.append(len(q_list))
        return index_list
    for i in range(len(objects)):
        object_data=objects[i][0]
        max_q=5.588
        # print max_q
        object_model=objects[i][1]
        index_use=np.where(object_model[:,0]<max_q)[0]
        object_model=np.append(object_model[index_use,0][:,np.newaxis],object_model[index_use,1][:,np.newaxis],axis=1)
        index_list=_find_index_of_near_bragg_peak(object_model[:,0],c=c_projected[i])
        ax.scatter(object_data[:,0],object_data[:,1]*(100**i),marker=markers[i],s=10,facecolors='none',edgecolors=colors_2[i],alpha=0.8,label='Data_'+labels[i])
        ax.errorbar(object_data[:,0],object_data[:,1]*(100**i),yerr=object_data[:,2]*(100**i),fmt=None,alpha=0.8,ecolor=colors_2[i])
        ax.annotate(labels[i],xy=(object_data[-1,0]+0.2,object_data[-1,1]*(100**i)),xytext=(object_data[-1,0]+0.2,object_data[-1,1]*(100**i)),fontsize=10,**hfont)
        for j in range(len(index_list)-1):
            l,=ax.plot(object_model[index_list[j]:index_list[j+1],0],object_model[index_list[j]:index_list[j+1],1]*(100**i),color=colors[i],lw=lw,label='Model_'+labels[i])
        pyplot.xlabel(r'$\rm{q\ (\AA^{-1})}$',axes=ax,fontsize=fontsize,**hfont)
        pyplot.ylabel(r'$\rm{Intensity}$',axes=ax,fontsize=fontsize,**hfont)
        #pyplot.title('(00L)',weight=4,size=fontsize,clip_on=True)
    #ax.legend(prop={'size':fontsize})

    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    ax.plot([0.35,0.35],[0,100],':',color='black')
    ax.plot([0.87,0.87],[0,100],':',color='black')
    pyplot.xlim((0,7))
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_ctrs.png'),dpi=300)
    return fig

def plot_CTR_multiple_CTR_datasets_muscovite_matlab(file_head=module_path_locator(),dump_files=['CTR_100mL_LiCl_Zr_mica_APS','CTR_100mM_KCl_Zr_mica_ESRF','CTR_Zr_case_100mLRbCl_Zr_mica_APS','CTR_100mL_CsCl_Zr_mica_APS_Feb07_2018'],labels=['100LiCl','100KCl','100RbCl','100CsCl'],markers=['.']*10,fontsize=13,lw=0.5,color_type=[1,5]):
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    colors=set_color(len(dump_files)*2,color_type[0])
    colors_2=set_color(len(dump_files)*2,color_type[0])
    objects=[]
    for i in range(len(dump_files)):
        objects.append(np.loadtxt(os.path.join(file_head,dump_files[i]),comments='%',skiprows=1))
    fig=pyplot.figure(figsize=(8,5))
    ax=fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    def _find_index_of_near_bragg_peak(q_list=[],l_bragg=[0,2,4,6,8,10,12,14,16],c=19.9347):
        index_list=[0]
        q_bragg=np.array(l_bragg)*np.pi*2/c
        for i in range(len(q_list)):
            if i!=len(q_list)-1:
                for q in q_bragg:
                    if q_list[i]<q and q_list[i+1]>q:
                        index_list.append(i+1)
                        break
        index_list.append(len(q_list))
        return index_list
    for i in range(len(objects)):
        object_data=objects[i]
        max_q=5.588
        print(max_q)
        if object_data[:,2].max()>10:
            print(object_data[:,2].max())
            object_data[:,2]=object_data[:,2]*np.pi*2./19.97
        else:
            pass
        ax.plot(object_data[:,2],object_data[:,3],marker=markers[i],ms=10,mfc='none',ls='-',c=colors_2[i],mec=colors_2[i],alpha=0.8,label='Data_'+labels[i])
        ax.errorbar(object_data[:,2],object_data[:,3],yerr=object_data[:,4],fmt=None,alpha=0.8,ecolor=colors_2[i])
        pyplot.xlabel(r'$\rm{q\ (\AA^{-1})}$',axes=ax,fontsize=fontsize,**hfont)
        pyplot.ylabel(r'$\rm{Intensity}$',axes=ax,fontsize=fontsize,**hfont)
        #pyplot.title('(00L)',weight=4,size=fontsize,clip_on=True)
    ax.legend(prop={'size':fontsize})

    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(2)
    for label in ax.get_xticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    ax.plot([0.35,0.35],[0,100],':',color='black')
    ax.plot([0.87,0.87],[0,100],':',color='black')
    pyplot.xlim((0,6))
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_ctrs.png'),dpi=300)
    return fig

#file arguments for cation effect results
file_heads_list=['P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Li_Zr_mica','M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\100mM_NaCl_Zr_mica','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\K_Zr_mica','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Rb_Zr_mica\\Zr_RAXR','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Cs_Zr_mica','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Rb_Zr_mica_moritz_fit_bestfit_so_far\\Rb_RAXR']
glob_heads_list=['Zr_mica_zr_100mM_RbCl_MD_Feb07_fit','mica_zr_100mM_NaCl_MD_May04_fit','Zr_mica_zr_100mM_KCl_MD_Feb08_fit','Zr_mica_zr_100mM_RbCl_MD_Feb07_fit','Zr_mica_zr_100mM_CsCl_MD_Feb07_fit','Zr_mica_Rb_100mM_RbCl_MD_Feb07_fit']
L_list_full=[0.41,0.53,0.61,0.75,0.88,1.15,1.45,1.71,2.31,2.64,2.85,3.21,3.55,4.24,4.55,5.61,6.25,7.31,9.15,10.31,11.15]
def plot_RAXR_matlab_output_single_data_set(file_head=file_heads_list[-1],glob_head=glob_heads_list[-1],L_list=[0.41,0.53,0.61,0.75,0.88,1.15,1.45,1.71,2.31,2.64,2.85,3.21,3.55,4.24,4.55,5.61,6.25,7.31,9.15,10.31,11.15],c_projected=19.9597,offset=8,start_plot=0,num_plots=20):
    #file_head for 0mM NaCl:'M:\\fwog\\members\\qiu05\\mica\\Zr_files_fit',glob_head:'mica_zr_0NaCl_MD_May04_fit'
    #file_head for 1mM NaCl:'M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\1mM_NaCl_Zr_mica', glob_head:'mica_zr_1mM_NaCl_MD_May03_fit'
    #file_head for 10mM NaCl:'M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\10mM_NaCl_Zr_mica', glob_head:'mica_zr_10mM_NaCl_MD_May04_fit'
    #file_head for 100mM NaCl:'M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\100mM_NaCl_Zr_mica', glob_head:'mica_zr_100mM_NaCl_MD_1peak_May04_fit','mica_zr_100mM_NaCl_MD_May04_fit'
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    files=glob.glob(os.path.join(file_head,glob_head)+'*_norm2')
    files_sorted=[]
    index_list=[]
    for each_file in files:
        index_list.append(int(each_file.split('_')[-2]))
    for i in range(1,len(files)+1):
        for j in range(len(index_list)):
            if i==index_list[j]:
                files_sorted.append(files[index_list[j]-1])
                break
    fig=pyplot.figure(figsize=(7,9))
    ax=fig.add_subplot(1,1,1)
    y_lim_max=170
    y_lim_min=0
    for i in range(len(files_sorted)):
        if i in range(start_plot,num_plots+start_plot):
            file=files_sorted[i]
            data=np.loadtxt(file)
            avg=np.average(data[:,1])
            ax.scatter(data[:,0],data[:,1]-avg+i*offset,facecolors='none')
            ax.errorbar(data[:,0],data[:,1]-avg+i*offset,yerr=data[:,2],fmt=None)
            ax.plot(data[:,0],data[:,3]-avg+i*offset)
            ax.annotate('q='+str(round(L_list[i]*np.pi*2/c_projected,3))+' (+'+str(i*offset)+')',xy=(15.33,data[:,3][-1]-avg+i*offset+2),xytext=(15.33,data[:,3][-1]-avg+i*offset+2),fontsize=10,**hfont)
            y_lim_max=data[:,3][-1]-avg+i*offset+offset
            if i==start_plot:
                y_lim_min=data[:,3][-1]-avg+i*offset-offset
    for label in ax.get_xticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    pyplot.xlim((14.999,15.39))
    pyplot.ylim((y_lim_min,y_lim_max))
    pyplot.xlabel(r'$\rm{Energy(keV)}$',axes=ax,fontsize=13,**hfont)
    pyplot.ylabel(r'$\rm{Intensity}$',axes=ax,fontsize=13,**hfont)
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_raxrs.png'),dpi=300)
    return fig
#arguments for ZrNP aggregation with [NaCl]
file_heads_IS=['M:\\fwog\\members\\qiu05\\mica\\Zr_files_fit','M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\1mM_NaCl_Zr_mica','M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\10mM_NaCl_Zr_mica','M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\100mM_NaCl_Zr_mica']
glob_heads_IS=['mica_zr_0NaCl_MD_May04_fit','mica_zr_1mM_NaCl_MD_May03_fit','mica_zr_10mM_NaCl_MD_May04_fit','mica_zr_100mM_NaCl_MD_May04_fit']
#arguments for ZrNP sorption with [MCl]
file_heads_CE=['P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Li_Zr_mica','M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\100mM_NaCl_Zr_mica','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\K_Zr_mica','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Rb_Zr_mica_old\\Zr_RAXR','P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Cs_Zr_mica']
glob_heads_CE=['Zr_mica_zr_100mM_RbCl_MD_Feb07_fit','mica_zr_100mM_NaCl_MD_May04_fit','Zr_mica_zr_100mM_KCl_MD_Feb08_fit','Zr_mica_zr_100mM_RbCl_MD_Feb07_fit','Zr_mica_zr_100mM_CsCl_MD_Feb07_fit']
def plot_RAXR_matlab_output_multiple_data_sets(file_heads=file_heads_CE,glob_heads=glob_heads_CE,L_list=[0.41,0.53,0.61,0.75,0.88,1.15,1.45,1.71,2.31,2.64,2.85,3.21,3.55,4.24,4.55,5.61,6.25,7.31,9.15,10.31,11.15],c_projected=19.9597,offset=10,start_plot=0,num_plots=3,E_range=[17.92,18.075],color_type=1):
    #file_head for 0mM NaCl:'M:\\fwog\\members\\qiu05\\mica\\Zr_files_fit',glob_head:'mica_zr_0NaCl_MD_May04_fit'
    #file_head for 1mM NaCl:'M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\1mM_NaCl_Zr_mica', glob_head:'mica_zr_1mM_NaCl_MD_May03_fit'
    #file_head for 10mM NaCl:'M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\10mM_NaCl_Zr_mica', glob_head:'mica_zr_10mM_NaCl_MD_May04_fit'
    #file_head for 100mM NaCl:'M:\\fwog\\members\qiu05\\1608 - 13-IDC\\schmidt\\mica\\Zr_files\\100mM_NaCl_Zr_mica', glob_head:'mica_zr_100mM_NaCl_MD_1peak_May04_fit','mica_zr_100mM_NaCl_MD_May04_fit'
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    colors=set_color(num_plots*2,color_type)
    fig=pyplot.figure(figsize=(5,6))
    y_lim_max=170
    y_lim_min=0
    for q_plot in range(num_plots):
        ax=fig.add_subplot(num_plots,1,q_plot+1)
        for i_plot in range(len(file_heads)):
            file_head=file_heads[i_plot]
            glob_head=glob_heads[i_plot]
            files=glob.glob(os.path.join(file_head,glob_head)+'*_norm2')
            files_sorted=[]
            index_list=[]
            for each_file in files:
                index_list.append(int(each_file.split('_')[-2]))
            for i in range(1,len(files)+1):
                for j in range(len(index_list)):
                    if i==index_list[j]:
                        files_sorted.append(files[index_list[j]-1])
                        break

            i=q_plot+start_plot
            file=files_sorted[i]
            data=np.loadtxt(file)
            selected_rows=[list(data[:,0]).index(each) for each in data[:,0] if (each>=E_range[0] and each<=E_range[1])]
            avg=np.average(data[selected_rows,1])
            ax.scatter(data[selected_rows,0],data[selected_rows,1]-avg+i_plot*offset,facecolors='none',edgecolors=colors[i_plot],alpha=0.5)
            ax.errorbar(data[selected_rows,0],data[selected_rows,1]-avg+i_plot*offset,yerr=data[selected_rows,2],fmt=None,alpha=0.5,ecolor=colors[i_plot])
            ax.plot(data[selected_rows,0],data[selected_rows,3]-avg+i_plot*offset,color=colors[i_plot])
            #ax.annotate('(+'+str(i_plot*offset)+')',xy=(18.13,data[:,3][-1]-avg+i_plot*offset),xytext=(18.13,data[:,3][-1]-avg+i_plot*offset),fontsize=10,**hfont)
            if i_plot==0:
                ax.annotate(r'$\rm{(+0)}$',xy=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),xytext=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),fontsize=10,**hfont)
            elif i_plot==1:
                ax.annotate(r'$\rm{(+10)}$',xy=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),xytext=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),fontsize=10,**hfont)
            elif i_plot==2:
                ax.annotate(r'$\rm{(+20)}$',xy=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),xytext=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),fontsize=10,**hfont)
            elif i_plot==3:
                ax.annotate(r'$\rm{(+30)}$',xy=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),xytext=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),fontsize=10,**hfont)
            elif i_plot==4:
                ax.annotate(r'$\rm{(+40)}$',xy=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),xytext=(18.08,data[selected_rows,3][-1]-avg+i_plot*offset),fontsize=10,**hfont)
            y_lim_max=data[:,3][-1]-avg+i_plot*offset+offset*1.5
            if i_plot==0:
                print(i,i_plot)
                y_lim_min=data[:,3][-1]-avg+i_plot*offset-offset

        pyplot.xlim((17.9,18.12))
        pyplot.ylim((y_lim_min,y_lim_max+9))
        q_value=str(round(L_list[q_plot+start_plot]*np.pi*2/c_projected,3))
        if q_plot==0:
            ax.annotate(r'$\rm{q=0.129\ \AA^{-1}}$',xy=(18.,y_lim_max),xytext=(18.,y_lim_max),fontsize=10,**hfont)
        elif q_plot==1:
            ax.annotate(r'$\rm{q=0.167\ \AA^{-1}}$',xy=(18.,y_lim_max),xytext=(18.,y_lim_max),fontsize=10,**hfont)
        elif q_plot==2:
            ax.annotate(r'$\rm{q=0.192\ \AA^{-1}}$',xy=(18.,y_lim_max),xytext=(18.,y_lim_max),fontsize=10,**hfont)
        if q_plot==num_plots-1:pyplot.xlabel(r'$\rm{Energy(keV)}$',axes=ax,fontsize=13,**hfont)
        pyplot.ylabel(r'$\rm{Intensity}$',axes=ax,fontsize=13,**hfont)
        for label in ax.get_xticklabels() :
            label.set_fontproperties(hfont['fontname'])
            label.set_fontsize(12)
        for label in ax.get_yticklabels() :
            label.set_fontproperties(hfont['fontname'])
            label.set_fontsize(12)
        #pyplot.title('q='+str(round(L_list[q_plot+start_plot]*np.pi*2/c_projected,3))+r'$\rm{\ \AA^{-1}}$',fontsize=10,**hfont)
    #pyplot.rcParams.update({'font.size': 10})
    fig.tight_layout()
    fig.savefig(os.path.join(file_heads[0],'multiple_raxrs_multiple_data.png'),dpi=300)
    return fig

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

def plotting_many_modelB_2(save_file='D://pic.png',fig_size=(8,6),head='P:\\My stuff\\Manuscripts\\hematite rcut\\pb on cmp rcut v10\dump_files\\',object_files=['temp_plot_O1O3_O5O7','temp_plot_O1O3_O1O4_no_water_Oct12'],index=[3,3],color=['blue','green','red','#984ea3','#ff7f00'],lw=2.,l_dashes=[(None,None),(None,None),(None,None),(None,None),(None,None),(None,None)],label=['Experimental data','Model1 results','Model2 results','Model3','Model4','Model5','Model6'],marker=['p'],title=['0 0 L','0 2 L','1 0 L','1 1 L','2 0 L','2 2 L','3 0 L','2 -1 L','2 1 L'],legend=[False,False,False,False,False,False,False,False,False],fontsize=10):
    #plotting model results simultaneously, object_files=[file1,file2,file3] file is the path of a dumped data/model file
    fig=pyplot.figure(figsize=fig_size)
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
        pass

    for i in range(len(object)):
        if i<index[0]*index[1]:
            order=i
            ob=object[i]
            plotting_modelB(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        else:
            pass
    fig.tight_layout()
    fig.savefig(save_file,dpi=300)
    return fig

def plotting_single_rod(save_file='D://pic.png',head='C:\\Users\\jackey\\Google Drive\\useful codes\\plotting\\',object_files=['temp_plot_O1O2','temp_plot_O5O6','temp_plot_O1O3','temp_plot_O5O7','temp_plot_O1O4','temp_plot_O5O8'],index=[1,1],color=['0.6','b','b','g','g','r','r'],lw=1.5,l_dashes=[(2,2,2,2),(None,None),(2,2,2,2),(None,None),(2,2,2,2),(None,None)],label=['Experimental data','Model1 results','Model2 results','Model3','Model4','Model5','Model6'],marker=['p'],title=['0 0 L','0 2 L','1 0 L','1 1 L','2 0 L','2 2 L','3 0 L','2 -1 L','2 1 L'],legend=[False,False,False,False,False,False,False,False,False],fontsize=10,rod_index=0):
    #plotting model results simultaneously, object_files=[file1,file2,file3] file is the path of a dumped data/model file
    #setting for demo show
    #fig=pyplot.figure(figsize=(10,9))
    #settings for publication
    #fig=pyplot.figure(figsize=(10,7))
    fig=pyplot.figure(figsize=(8.5,7))
    object_sets=[pickle.load(open(head+file)) for file in object_files]#each_item=[00L,02L,10L,11L,20L,22L,30L,2-1L,21L]
    object=[]
    for i in range(9):
        object.append([])
        for j in range(len(object_sets)):
            if j==0:
                object[-1].append(object_sets[j][i][0])
            object[-1].append(object_sets[j][i][1])

    for i in [rod_index]:
    #for i in range(1):
        order=i
        #print 'abc'
        ob=object[i]
        plotting_modelB(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=None,fontsize=fontsize)
        #plotting_modelB(object=ob,fig=fig,index=[1,1,i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(save_file,dpi=300)
    return fig

#overplotting experimental datas formated with UAF_CTR_RAXS_2 loader in GenX
def plot_many_experiment_data(data_files=['D:\\Google Drive\\data\\400uM_Sb_hematite_rcut.datnew_formate','D:\\Google Drive\\data\\1000uM_Pb_hematite_rcut.datnew_formate'],labels=['Sb 400uM on hematite','Pb 1000uM on hematite'],HKs=[[0,0],[0,2],[1,0],[1,1],[2,0],[2,1],[2,-1],[2,2],[3,0]],index_subplot=[3,3],colors=['b','g','r','c','m','y','w'],markers=['.','*','o','v','^','<','>'],fontsize=10):
    data_container={}
    for i in range(len(labels)):
        temp_data=np.loadtxt(data_files[i])
        sub_set={}
        for HK in HKs:
            label=str(int(HK[0]))+'_'+str(int(HK[1]))
            sub_set[label]=np.array(filter(lambda x:x[1]==HK[0] and x[2]==HK[1],temp_data))
        data_container[labels[i]]=sub_set
    fig=pyplot.figure()
    for i in range(len(HKs)):
        title=str(int(HKs[i][0]))+str(int(HKs[i][1]))+'L'
        ax=fig.add_subplot(index_subplot[0],index_subplot[1],i+1)
        ax.set_yscale('log')
        for label in labels:
            data_temp=data_container[label][str(int(HKs[i][0]))+'_'+str(int(HKs[i][1]))]
            ax.errorbar(data_temp[:,0],data_temp[:,4],data_temp[:,5],label=label,marker=markers[labels.index(label)],ecolor=colors[labels.index(label)],color=colors[labels.index(label)],markerfacecolor=colors[labels.index(label)],linestyle='None',markersize=8)
        pyplot.title(title,position=(0.5,0.85),weight='bold',clip_on=True)
        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(fontsize)
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(fontsize)
        for l in ax.get_xticklines() + ax.get_yticklines():
            l.set_markersize(5)
            l.set_markeredgewidth(2)
        if i==0:
            ax.legend(bbox_to_anchor=(0.5,0.92,0,3),bbox_transform=fig.transFigure,loc='lower center',ncol=4,borderaxespad=0.,prop={'size':14})
        if (i+1)>index_subplot[1]*(index_subplot[0]-1):
            pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        if i%index_subplot[1]==0:
            pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
    return True

lateral_size=[5.4,5.1,7.3,8.6]
lateral_size_errors=[0.5,0.1,0.6,0.5]
vertical_size=[0.974,1.61,1.6,2.48]
vertical_size_error=[0.01,0.01,0.01,0.04]
labels=['IS=1.8 mM','IS=2.9 mM','IS=12 mM', 'IS=102 mM']
def plot_NP_size_evolution(file_head=module_path_locator(),lateral=lateral_size,lateral_error=lateral_size_errors,vertical=vertical_size,vertical_error=vertical_size_error,label=labels,color_type=1):
    colors=set_color(len(lateral),color_type)
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    fig=pyplot.figure(figsize=(8,5))
    ax=fig.add_subplot(1,1,1)
    pyplot.ylabel("Vertical size (nm)",fontsize=12)
    pyplot.xlabel("Lateral size (nm)",fontsize=12)
    for i in range(len(lateral)):
        x,y,x_er,y_er,label,color=lateral[i],vertical[i],lateral_error[i],vertical_error[i],labels[i],colors[i]
        ax.errorbar(x,y,xerr=x_er,yerr=y_er,fmt='o',label=label,color=color)
    #for i in range(len(lateral)-1):
    #    ax.arrow(lateral[i],vertical[i],lateral[i+1]-lateral[i],vertical[i+1]-vertical[i],head_width=0.05, head_length=0.2, fc='k', ec='k',ls=':',color=colors[i])
        ax.annotate(label,xy=(x-0.4,y+0.05),xytext=(x-0.4,y+0.05),fontsize=12)
    #ax.legend(fontsize=12,loc=2)
    ax.set_ylim(0.8,2.8)
    fig.savefig(os.path.join(file_head,'NP_size_evolution.png'),dpi=300)
    return fig

def plot_multiple_e_profiles(file_head=module_path_locator(),dump_files=['temp_plot_eden_0NaCl','temp_plot_eden_1NaCl','temp_plot_eden_10NaCl','temp_plot_eden_100NaCl'],label_marks=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl'],color_type=5):
    colors=set_color(len(dump_files),color_type)
    fig=pyplot.figure(figsize=(6,8))
    ax1=fig.add_subplot(2,1,1)
    pyplot.ylabel("E_density",fontsize=12)
    #pyplot.xlabel("Z(Angstrom)",fontsize=12)
    pyplot.title('Total e profile',fontsize=12)
    ax2=fig.add_subplot(2,1,2)
    pyplot.ylabel("e density",fontsize=12)
    pyplot.xlabel(r"$\rm{Z(\AA)}$",fontsize=12)
    pyplot.title('Zr e profile',fontsize=12)
    for i in range(len(dump_files)):
        data_eden=pickle.load(open(os.path.join(file_head,dump_files[i]),"rb"))
        edata,labels=data_eden[0],data_eden[1]
        ax1.plot(np.array(edata[-1][0,:]),edata[-1][1,:]+i,color=colors[i],label="Total e "+label_marks[i],lw=2)
        ax2.fill_between(np.array(edata[-1][0,:]),edata[-1][2,:]*0+i,edata[-1][2,:]+i,color=colors[i],label="Zr e profile (MD) "+label_marks[i],alpha=0.6)
    ax1.legend(fontsize=12)
    ax2.legend(fontsize=12)
    ax2.set_ylim(0,6)
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_eprofiles.png'),dpi=300)
    return fig

dump_files_genx_0=['temp_plot_eden_0NaCl_0502','temp_plot_eden_1NaCl_0502','temp_plot_eden_10NaCl_0502','temp_plot_eden_100NaCl_0502']
labels_genx_0=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl']
dump_files_genx_1=['temp_plot_eden_100mM_NH4Cl_Jul10_2017','temp_plot_eden_100mM_CH5NCl_Jul23_2017','temp_plot_eden_100mM_LiCl_Jul10_2017','temp_plot_eden_100mM_NaCl_Jul10_2017','temp_plot_eden_100mM_KCl_Jul23_2017','temp_plot_eden_100mM_RbCl_Jul10_2017','temp_plot_eden_32mM_MgCl2_Jul10_2017','temp_plot_eden_32mM_CaCl2_Jul10_2017']
labels_genx_1=['100 mM NH4Cl (2.4 Zr/AUC)','100 mM CH5NCl (4.4 Zr/AUC)','100 mM LiCl (2.1 Zr/AUC)','100 mM NaCl (3.7 Zr/AUC)','100 mM KCl (3.9 Zr/AUC)','100 mM RbCl (3.9 Zr/AUC)','32 mM MgCl2 (2.4 Zr/AUC)','32 mM CaCl2 (3.9 Zr/AUC)']
dump_files_genx_2=['temp_plot_eden_100LiCl_Oct4_2017','temp_plot_eden_100NaCl_Oct4_2017','temp_plot_eden_100KCl_Oct4_2017','temp_plot_eden_100RbCl_Oct4_2017','temp_plot_eden_100CsCl_Oct4_2017']
labels_genx_2=[r'100 mM LiCl [5.04(0.01) Zr/AUC]',r'100 mM NaCl [4.57(0.6) Zr/AUC]',r'100 mM KCl [3.92(1) Zr/AUC]',r'100 mM RbCl [2.26(0.15) Zr/AUC]',r'100 mM CsCl [1.58(0.8) Zr/AUC]']
dump_files_genx_3=['temp_plot_eden_100LiCl_Oct5_2017','temp_plot_eden_100NaCl_Oct5_2017','temp_plot_eden_100RbCl_Zr_Oct5_2017','temp_plot_eden_100RbCl_Rb_Oct5_2017','temp_plot_eden_100CsCl_Oct5_2017']
labels_genx_3=[r'100 mM LiCl [4.40(1) Zr/AUC]',r'100 mM NaCl [3.98(0.68) Zr/AUC]',r'100 mM RbCl [2.23(0.04) Zr/AUC]',r'100 mM RbCl [~1.0(0.1) Rb/AUC]',r'100 mM CsCl [1.56(0.04) Zr/AUC]']
dump_files_genx_4=['temp_plot_eden_100LiCl_Oct5_2017','temp_plot_eden_100NaCl_Oct5_2017','temp_plot_eden_100NH4Cl_Zr_Oct10_2017','temp_plot_eden_100KCl_Zr_Oct10_2017','temp_plot_eden_100RbCl_Zr_Oct5_2017','temp_plot_eden_100RbCl_Rb_Oct10_2017','temp_plot_eden_100CsCl_Oct5_2017']
labels_genx_4=[r'100 mM LiCl [4.40(1) Zr/AUC]',r'100 mM NaCl [3.98(0.68) Zr/AUC]',r'100 mM NH4Cl [3.0(0.8) Zr/AUC]',r'100 mM KCl [3.6(0.1) Zr/AUC]',r'100 mM RbCl [2.23(0.04) Zr/AUC]',r'100 mM RbCl [0.66(0.01) Rb/AUC]',r'100 mM CsCl [1.56(0.04) Zr/AUC]']
dump_files_genx_5=['temp_plot_eden_100LiCl_Oct5_2017','temp_plot_eden_100NaCl_0502','temp_plot_eden_100KCl_Zr_Oct10_2017','temp_plot_eden_100RbCl_Zr_Oct5_2017','temp_plot_eden_100RbCl_Rb_Oct10_2017','temp_plot_eden_100CsCl_Oct5_2017']
labels_genx_5=[r'                  LiCl',r'                  NaCl',r'                  KCl',r'                  RbCl',r'0.66(0.01) Rb/AUC@RbCl',r'                  CsCl']
def plot_multiple_e_profiles_2(file_head=module_path_locator(),dump_files=dump_files_genx_5[::-1],label_marks=labels_genx_5[::-1],color_type=1):
    def _cal_percentage_(data,cutoff=5,label=''):#use to calculate the percentage of resonant element area within the cutoff distance from mineral surface
        z,e=data[0],data[1]
        total_area=0
        target_area=0
        for i in range(len(z))[1:]:
            x1,y1=z[i],e[i]
            x0,y0=z[i-1],e[i-1]
            area=abs(x1-x0)*min([y0,y1])+abs(x1-x0)*abs(y1-y0)/2
            total_area+=area
            if x1<=cutoff:
                target_area+=area
        print('<<Case of '+label+'>>')
        print('The percentage of area plot within '+str(cutoff)+' A is:'+str(target_area/total_area*100)+'%')
        return None
    colors=set_color(len(dump_files),color_type)
    fig=pyplot.figure(figsize=(6.5,9.8))
    ax1=fig.add_subplot(1,1,1)
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    pyplot.ylabel(r"$\rm{Normalized\ Electron\ Density}$",fontsize=20,**hfont)
    pyplot.xlabel(r"$\rm{Height\ from\ the\ Surface(\AA)}$",fontsize=20,**hfont)
    #pyplot.title('Total e profile',fontsize=12)

    #ax2=fig.add_subplot(2,1,2)
    #pyplot.ylabel("E_density",fontsize=12)
    #pyplot.xlabel("Z(Angstrom)",fontsize=12)
    #pyplot.title('Zr e profile',fontsize=12)
    for i in range(len(dump_files)):
        data_eden=pickle.load(open(os.path.join(file_head,dump_files[i]),"rb"))
        edata,labels=data_eden[0],data_eden[1]
        ax1.plot(np.array(edata[-1][0,:]),edata[-1][1,:]+i*5,color=colors[i],label=label_marks[i],lw=2)
        ax1.fill_between(np.array(edata[-1][0,:]),edata[-1][2,:]*0+i*5,edata[-1][2,:]+i*5,color=colors[i],alpha=0.6)
        ax1.annotate(label_marks[i],xy=(18,edata[-1][1,:][-1]+i*5+0.5),xytext=(18,edata[-1][1,:][-1]+i*5+0.5),fontsize=20,**hfont)
        _cal_percentage_([edata[-1][0,:],edata[-1][2,:]],label=label_marks[i])
    #ax1.legend(fontsize=12)
    #ax2.legend(fontsize=12)
    ax1.set_ylim(0,35)
    ax1.set_xlim(-5,50)
    ax1.plot([2.,2.],[0,40],':',color='m')
    ax1.plot([5,5],[0,40],':',color='black')
    #ax1.plot([4.3,4.3],[0,12],':',color='black')
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(20)
    for label in ax1.get_yticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(20)
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_eprofiles_genx_results.png'),dpi=300)
    return fig

dump_file_1=[['total_e_profile_0mM_NaCl','mica_zr_0mM_NaCl_MD_May04_rho'],['total_e_profile_1mM_NaCl','mica_zr_1mM_NaCl_MD_May04_rho'],['total_e_profile_10mM_NaCl','mica_zr_10mM_NaCl_MD_May04_rho'],['total_e_profile_100mM_NaCl','mica_zr_100mM_NaCl_MD_May04_rho']]
label_1=['0 mM NaCl','1 mM NaCl','10 mM NaCl','100 mM NaCl']
dump_file_2=[['total_e_profile_100mM_LiCl','mica_zr_100mM_LiCl_MD_Jun12_rho'],['total_e_profile_100mM_NaCl','mica_zr_100mM_NaCl_MD_May04_rho'],['total_e_profile_100mM_RbCl_Jun29','Zr_mica_zr_100mM_RbCl_MD_Jun29_rho'],['total_e_profile_100mM_RbCl_Jun29','mica_Rb_100mM_RbCl_MD_Jun29_rho'],['total_e_profile_32mM_CaCl2_APS_Jun29','mica_zr_32mM_CaCl2_MD_Jun29_rho_APS'],['total_e_profile_32mM_MgCl2_ESRF','mica_zr_32mM_MgCl2_MD_Jun29_rho_ESRF'],['total_e_profile_100mM_NH4Cl','mica_zr_100mM_NH4Cl_MD_Jun27_rho']]
label_2=['100 mM LiCl (3.4 Zr/AUC)','100 mM NaCl (3.4 Zr/AUC)','100 mM RbCl(3.2 Zr/AUC)','100 mM RbCl(2.4 Rb/AUC)','32 mM CaCl2 (5.1 Zr/AUC)','32 mM MgCl2 (3.7 Zr/AUC)','100 mM NH4Cl (2.4 Zr/AUC)']
dump_file_Jul11=[['total_e_profile_100mM_NH4Cl_Jul11','mica_zr_100mM_NH4Cl_MD_Jul11_rho'],['total_e_profile_100mM_LiCl_Jul11','mica_zr_100mM_LiCl_MD_Jul11_rho'],['total_e_profile_100mM_NaCl_Jul11','mica_zr_100mM_NaCl_MD_Jul11_rho'],['total_e_profile_100mM_KCl_Aug10','mica_zr_100mM_KCl_MD_Aug10_rho'],['total_e_profile_100mM_RbCl_Jun29','Zr_mica_zr_100mM_RbCl_MD_Jun29_rho'],['total_e_profile_100mM_CsCl_Aug10','mica_zr_100mM_CsCl_MD_Aug10_rho'],['total_e_profile_32mM_MgCl2_ESRF_Jul11','mica_zr_32mM_MgCl2_MD_Jul11_rho_ESRF'],['total_e_profile_32mM_CaCl2_APS_Jul11','mica_zr_32mM_CaCl2_MD_Jul11_rho_APS']]
label_Jul11=['100 mM NH4Cl (1.6 Zr/AUC)','100 mM LiCl (2.3 Zr/AUC)','100 mM NaCl (3.4 Zr/AUC)','100 mM KCl (3.3 Zr/AUC)','100 mM RbCl(3.2 Zr/AUC)','100 mM CsCl(3.4 Zr/AUC)','32 mM MgCl2 (3.6 Zr/AUC)','32 mM CaCl2 (5.4 Zr/AUC)',]
dump_file_Jan30=[['total_e_profile_Jan08','Zr_mica_zr_100mM_RbCl_MD_Jan08_rho'],['total_e_profile_10mM_RbCl_Jan08','Zr_mica_zr_10mM_RbCl_MD_Jan08_rho'],['total_e_profile_1mM_RbCl_Jan08','Zr_mica_zr_1mM_RbCl_MD_Jan08_rho']]
label_Jan30=['100 mM RbCl(2.2(1) Zr/AUC)','10 mM RbCl(1.4(1) Zr/AUC)','1 mM RbCl(2.0(1) Zr/AUC)']

dump_file_Feb1_2017=[['total_e_profile_100mM_LiCl_Feb1_2017','100mM_LiCl_Feb1_2017_rho'],['total_e_profile_100mM_NaCl_Feb1_2017','100mM_NaCl_Feb1_2017_rho'],['total_e_profile_100mM_KCl_Feb1_2017','100mM_KCl_Feb1_2017_rho'],['total_e_profile_100mM_RbCl_Feb1_2017','100mM_RbCl_Feb1_2017_rho'],['total_e_profile_100mM_CsCl_Feb1_2017','100mM_CsCl_Feb1_2017_rho']]
label_Feb1_2017=['100 mM LiCl(3.9(3) Zr/AUC)','100 mM NaCl(3.4(1) Zr/AUC)','100 mM KCl(3.0(1) Zr/AUC)','100 mM RbCl(2.2(1) Zr/AUC)','100 mM CsCl(1.6(1) Zr/AUC)']
file_head='P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\results_in_all\\Feb8_2018'
dump_file_Feb8_2018=[['total_e_profile_100mM_LiCl_Feb08_2018','100mM_LiCl_Feb08_2018_rho'],['total_e_profile_100mM_NaCl_Feb08_2018','100mM_NaCl_Feb08_2018_rho'],['total_e_profile_100mM_KCl_Feb08_2018','100mM_KCl_Feb08_2018_rho'],['total_e_profile_100mM_RbCl_Feb08_2018','100mM_RbCl_Feb08_2018_rho','Rb_test_May18b_rho'],['total_e_profile_100mM_CsCl_Feb08_2018','100mM_CsCl_Feb08_2018_rho']]
label_Feb8_2018=['0.1 M LiCl(3.7(1) Zr/AUC)','0.1 M NaCl(3.4(1) Zr/AUC)','0.1 M KCl(3.1(1) Zr/AUC)','0.1 M RbCl(2.4(1) Zr/AUC)','0.1 M CsCl(1.8(1) Zr/AUC)']
def plot_multiple_e_profiles_matlab_output(file_head=module_path_locator(),dump_files=dump_file_Feb8_2018,label_marks=label_Feb8_2018,color_type=1,z_offset_Rb=0.073):
    def _cal_percentage_(data,cutoff=2.2,cutoff_from=0,label=''):#use to calculate the percentage of resonant element area within the cutoff distance from mineral surface
        z,e=data[0],data[1]
        total_area=0
        target_area=0
        for i in range(len(z))[1:]:
            x1,y1=z[i],e[i]
            x0,y0=z[i-1],e[i-1]
            area=abs(x1-x0)*min([y0,y1])+abs(x1-x0)*abs(y1-y0)/2
            total_area+=area
            if x1<=cutoff and x1>=cutoff_from:
                target_area+=area
        print('<<Case of '+label+'>>')
        print('The percentage of area plot within '+str(cutoff)+' A is:'+str(target_area/total_area*100)+'%')
        return None

    def _cal_integration_(data,cutoff_from=1,cutoff_to=3.4,label=''):#use to calculate the percentage of resonant element area within the cutoff distance from mineral surface
        z,e=data[0],data[1]
        total_area=0
        target_area=0
        for i in range(len(z))[1:]:
            x1,y1=z[i],e[i]
            x0,y0=z[i-1],e[i-1]
            area=abs(x1-x0)*min([y0,y1])+abs(x1-x0)*abs(y1-y0)/2
            total_area+=area
            if x1<=cutoff_to and x1>=cutoff_from:
                target_area+=area
        print('<<Case of '+label+'>>')
        print('The total integration of from '+str(cutoff_from)+'A to'+str(cutoff_to)+' A is:'+str(target_area))
        return None
    colors=set_color(len(dump_files),color_type)
    colors=set_color(sum([len(file)-1 for file in dump_files]),color_type)
    fig=pyplot.figure(figsize=(5,6))
    ax1=fig.add_subplot(1,1,1)
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    pyplot.ylabel(r"$\rm{Normalized\ Electron\ Density}$",fontsize=12,**hfont)
    pyplot.xlabel(r"$\rm{Height\ from\ the\ Surface(\AA)}$",fontsize=12,**hfont)
    #pyplot.title('Total e profile',fontsize=12)

    #ax2=fig.add_subplot(2,1,2)
    #pyplot.ylabel("E_density",fontsize=12)
    #pyplot.xlabel("Z(Angstrom)",fontsize=12)
    #pyplot.title('Zr e profile',fontsize=12)
    i_color=0
    for i in range(len(dump_files)):
        edata=np.loadtxt(os.path.join(file_head,dump_files[i][0]))
        ra_data=np.loadtxt(os.path.join(file_head,dump_files[i][1]))
        if len(dump_files[i])==3:
            i_color+=2
            ra_data_2=np.loadtxt(os.path.join(file_head,dump_files[i][2]))
            ax1.fill_between(ra_data_2[:,0]+z_offset_Rb,np.array([0]*len(ra_data_2))+i*5,ra_data_2[:,1]+i*5,color=colors[-i_color],alpha=0.5)
            ax1.plot(ra_data_2[:,0]+z_offset_Rb,ra_data_2[:,1]+i*5,color=colors[-i_color])
            _cal_percentage_([ra_data_2[:,0],ra_data_2[:,1]],cutoff=3.285,cutoff_from=1,label=label_marks[i])
        else:
            pass
        #labels=data_eden[0],data_eden[1]
        ax1.plot(np.array(edata[:,0]),edata[:,1]+i*5,color='0.1',label=label_marks[i],lw=1)
        print('total e density integration start here')
        _cal_integration_([edata[:,0],edata[:,1]],label=label_marks[i])
        print('total e density integration end here')
        ax1.fill_between(ra_data[:,0],np.array([0]*len(ra_data))+i*5,ra_data[:,1]+i*5,color=colors[i],alpha=0.3)
        ax1.annotate(label_marks[i],xy=(17,edata[:,1][-1]+i*5+1.5),xytext=(16,edata[:,1][-1]+i*5+1.5),fontsize=10,**hfont)
        _cal_percentage_([ra_data[:,0],ra_data[:,1]],label=label_marks[i])
    #ax1.legend(fontsize=10)
    #ax2.legend(fontsize=12)
    ax1.set_ylim(0,30)
    ax1.set_xlim(-5,35)
    #ax1.plot([2.5,2.5],[0,42],':',color='black')
    ax1.fill_between([2.2,2.6],[0,0],[42,42],color='black',alpha=0.15)
    #ax1.plot([2.2,2.2],[0,42],':',color='black')
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    for label in ax1.get_yticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(12)
    #ax1.plot([4.3,4.3],[0,20],':',color='black')
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'multiple_eprofiles3.png'),dpi=300)
    return fig
#files for agregation
files_AFM=['P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 0mM NaCl\\processed images\\height_dist_data',\
           'P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 1mM NaCl\\processed ones\\height_dist_data',\
           'P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 10mM NaCl\\processed ones\\height_dist_data',\
           'P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 100mM NaCl\\processed ones\\height_dist_data']
zero_height_offset=2.8
labels=['0 mM NaCl','1 mM NaCl', '10 mM NaCl', '100 mM NaCl']
#files for cation effect
files_AFM_ES=['P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\LiCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\NaCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\KCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\RbCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\CsCl.dat']
labels_ES=['0.1 M LiCl','0.1 M NaCl', '0.1 M KCl', '0.1 M RbCl', '0.1 M CsCl']
zero_height_offset_ES=0

def plot_AFM_height_distribution(files=files_AFM_ES,labels=labels_ES,zero_height_offset=3,offset_index=[1,4],color_type=1):
    colors=set_color(len(files),color_type)
    fig=pyplot.figure(figsize=(5,4))
    ax1=fig.add_subplot(1,1,1)
    hfont = {'fontname':['times new roman','Helvetica'][0]}
    pyplot.ylabel(r"$\rm{Distribution}$",fontsize=15,**hfont)
    pyplot.xlabel(r"$\rm{Height (\AA)}$",fontsize=15,**hfont)
    for i in range(len(files)):
        data1=np.loadtxt(files[i],skiprows=3)
        if i in offset_index:
            data1[:,0]=data1[:,0]*1e10-zero_height_offset
        else:
            data1[:,0]=data1[:,0]*1e10
        data1[:,1]=data1[:,1]/num.max(data1[:,1])
        ax1.plot(data1[:,0],data1[:,1],color=colors[i],label=labels[i])
    for label in ax1.get_xticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(15)
    for label in ax1.get_yticklabels() :
        label.set_fontproperties(hfont['fontname'])
        label.set_fontsize(15)
    l=pyplot.legend(fontsize=15,frameon=False)
    pyplot.xlim(0,80)
    pyplot.setp(l.texts,family='times new roman')
    fig.tight_layout()
    fig.savefig(os.path.join(module_path_locator(),'AFM_height_distribution.png'),dpi=300)

    return fig

#files for ZrO2 NP aggregation mechanism
files_AFM=['P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 0mM NaCl\\processed images\\facet_profile',\
           'P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 1mM NaCl\\processed ones\\waviness_profile',\
           'P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 10mM NaCl\\processed ones\\facet_profile',\
           'P:\\My stuff\\Data\\AFM zr on mica 001 various conditions\\Zr-mica of 100mM NaCl\\processed ones\\facet_profile']
labels=[r"$\rm{0\ mM\ NaCl\ (Avg=5.4 \pm 0.5\ nm)}$",r"$\rm{1\ mM\ NaCl\ (Avg=5.1 \pm 0.1\ nm)}$", r"$\rm{10\ mM\ NaCl\ (Avg=7.3 \pm 0.6\ nm)}$", r"$\rm{100\ mM\ NaCl\ (Avg=8.6 \pm 0.5\ nm)}$"]
#files for cation effect
files_AFM_CE=['P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\waviness_LiCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\waviness_NaCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\waviness_KCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\waviness_RbCl.dat',\
              'P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\AFM images 2\\waviness_CsCl.dat']
labels_CE=[r"$\rm{100\ mM\ LiCl\ (Avg=7.5 \pm 0.5\ nm)}$",r"$\rm{100\ mM\ NaCl\ (Avg=6.5 \pm 0.5\ nm)}$", r"$\rm{100\ mM\ KCl\ (Avg=4.5 \pm 0.4\ nm)}$", r"$\rm{100\ mM\ RbCl\ (Avg=5.1 \pm 0.5\ nm)}$",r"$\rm{100\ mM\ CsCl\ (Avg=4.6 \pm 0.5\ nm)}$"]
def plot_AFM_facet_profiles(files=files_AFM_CE,labels=labels_CE,color_type=1):
    colors=set_color(len(files),color_type)
    fig=pyplot.figure(figsize=(5,10))

    hfont = {'fontname':['times new roman','Helvetica'][0]}

    for i in range(len(files)):
        ax1=fig.add_subplot(5,1,i+1)
        data1=np.loadtxt(files[i],skiprows=3)
        data1[:,0]=data1[:,0]*1e9
        data1[:,1]=data1[:,1]*1e9
        ax1.plot(data1[:,0],data1[:,1],color=colors[i])
        l=pyplot.title(labels[i],fontsize=15)
        #pyplot.setp(l.texts,family='times new roman')
        if 1:
            pyplot.ylabel(r"$\rm{Height (nm)}$",fontsize=15,**hfont)
        if i==4:
            pyplot.xlabel(r"$\rm{Length (nm)}$",fontsize=15,**hfont)
    fig.tight_layout()
    fig.savefig(os.path.join(module_path_locator(),'AFM_facet_profiles.png'),dpi=300)

    return fig

def temp_plot_dG_over_coverage(coverages=[3.7,3.4,3.1,2.4,1.8],coverage_errors=[0.1,0.1,0.1,0.1,0.1],dG=[-475,-365,-295,-275,-250],labels=['LiCl','NaCl','KCl','RbCl','CsCl'],color_map='Greens'):
    hfont = {'fontname':['times new roman','Helvetica'][1]}

    #p1=np.polyfit(dG[0:3],coverages[0:3],1)
    #p1=np.polyfit(dG[0:2],coverages[0:2],1)
    p1=[0,3.55]
    z1=np.poly1d(p1)([dG[0]-50]+dG[0:3]+[dG[4]+50])
    eq1="y=%.5fx+%.3f"%(p1[0],p1[1])
    p2=np.polyfit(dG[2:5],coverages[2:5],1)
    z2=np.poly1d(p2)(dG[1:5]+[dG[-1]+50])
    eq2="y=%.5fx%.3f"%(p2[0],p2[1])

    fig=pyplot.figure(figsize=(6,5))
    ax=fig.add_subplot(1,1,1)
    ax.errorbar(dG[0:3],coverages[0:3],yerr=coverage_errors[0:3],fmt='.',color='blue')
    #ax.plot([dG[0]-50]+dG[0:3]+[dG[4]+50],z1,'-',lw=1,color='blue',label=eq1)

    ax.errorbar(dG[2:5],coverages[2:5],yerr=coverage_errors[2:5],fmt='.',color='m')
    ax.plot(dG[1:5]+[dG[-1]+50],z2,'-',lw=1,color='m',label=eq2)
    for i in range(5):
        if i==1:
            ax.annotate(labels[i],xy=(dG[i],coverages[i]-.2),xytext=(dG[i],coverages[i]-.2),fontsize=10,**hfont)
        elif i==2:
            ax.annotate(labels[i],xy=(dG[i]-20,coverages[i]-.1),xytext=(dG[i]-20,coverages[i]-.1),fontsize=10,**hfont)
        else:
            ax.annotate(labels[i],xy=(dG[i],coverages[i]+.2),xytext=(dG[i],coverages[i]+.2),fontsize=10,**hfont)

    #horizontal guild line
    ax.plot([-524,-313],[3.55,3.55],'--r',lw=2)

    #make a gradient filled color between LiCl and CsCl
    n_fills=100
    x_start,x_end=-358,-224
    normalize = mpt.colors.Normalize(vmin=x_start, vmax=x_end)
    #cmap = mpt.cm.jet
    cmap=plt.get_cmap(color_map)

    cross_pt_x=(p2[1]-p1[1])/(p1[0]-p2[0])
    cross_pt_y=p1[0]*cross_pt_x+p1[1]
    line_1=lambda x:p1[0]*x+p1[1]
    line_2=lambda x:p2[0]*x+p2[1]

    for i in range(n_fills+1):
        x_temp_0=x_start+float(x_end-x_start)/n_fills*i
        x_temp_1=x_start+float(x_end-x_start)/n_fills*(i+1)
        if x_temp_1<=cross_pt_x:
            temp_p1,temp_p2,temp_p3,temp_p4=[x_temp_0,1],[x_temp_1,1],[x_temp_1,line_1(x_temp_1)],[x_temp_0,line_1(x_temp_0)]
            path=Path([temp_p1,temp_p2,temp_p3,temp_p4,temp_p1])
            #pyplot.imshow(path,cmap=pyplot.cm.Greens)
            patch = PathPatch(path, facecolor=cmap(normalize(x_temp_0)),ec=cmap(normalize(x_temp_0)))
            ax.add_patch(patch)
        elif x_temp_0>=cross_pt_x:
            temp_p1,temp_p2,temp_p3,temp_p4=[x_temp_0,1],[x_temp_1,1],[x_temp_1,line_2(x_temp_1)],[x_temp_0,line_2(x_temp_0)]
            path=Path([temp_p1,temp_p2,temp_p3,temp_p4,temp_p1])
            patch = PathPatch(path, facecolor=cmap(normalize(x_temp_0)),ec=cmap(normalize(x_temp_0)))
            ax.add_patch(patch)
        else:
            temp_p1,temp_p2,temp_p3,temp_p4,temp_p5=[x_temp_0,1],[x_temp_1,1],[x_temp_1,line_2(x_temp_1)],[cross_pt_x,cross_pt_y],[x_temp_0,line_1(x_temp_0)]
            path=Path([temp_p1,temp_p2,temp_p3,temp_p4,temp_p5,temp_p1])
            patch = PathPatch(path, facecolor=cmap(normalize(x_temp_0)),ec=cmap(normalize(x_temp_0)))
            ax.add_patch(patch)
        #print normalize(x_temp_0)
    #_find_all_polygon(p1,p2,n_fills)

    ax.annotate('OS',xy=(-430,2.58),xytext=(-430,2.58),fontsize=15,**hfont)
    ax.annotate('IS',xy=(-279,1.5),xytext=(-279,1.5),fontsize=15,**hfont)

    ax.set_ylim(1,5)
    pyplot.ylabel(r"$\rm{Coverage (Zr/A_{UC})}$",fontsize=12,**hfont)
    pyplot.xlabel(r"$\rm{Hydration\/ Energy (kJ/mol)}$",fontsize=12,**hfont)
    leg=pyplot.legend(fontsize=10,loc=2,frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(module_path_locator(),'dG_over_coverage.png'),dpi=300)
    return None

def plot_multiple_APQ_profiles(file_head=module_path_locator(),dump_files=['temp_plot_raxr_A_P_Q_0NaCl','temp_plot_raxr_A_P_Q_1NaCl','temp_plot_raxr_A_P_Q_10NaCl','temp_plot_raxr_A_P_Q_100NaCl'],labels=['free of NaCl','1 mM NaCl','10 mM NaCl','100 mM NaCl'],color_type=5):
    colors=set_color(len(dump_files),color_type)
    fig1=pyplot.figure(figsize=(8,4))
    ax1=fig1.add_subplot(1,2,1)
    pyplot.ylabel(r'$\rm{Partial\ SF\ Amplitude\ (Zr/A_{UC})}$')
    pyplot.xlabel(r'$\rm{q\ (\AA^{-1})}$')
    ax2=fig1.add_subplot(1,2,2)
    pyplot.ylabel(r'$\rm{Partial\ SF\ Phase/q\ (\AA)}$')
    pyplot.xlabel(r'$\rm{q\ (\AA^{-1})}$')

    for i in range(len(dump_files)):
        AP_Q_file=os.path.join(file_head,dump_files[i])
        #plot Q dependence of Foriour components A and P
        data_AP_Q=pickle.load(open(AP_Q_file,"rb"))
        #A over Q
        ax1.plot(data_AP_Q[0][2],data_AP_Q[0][0],color=colors[i],label=labels[i],lw=1.5)
        ax1.errorbar(data_AP_Q[1][2],data_AP_Q[1][0],yerr=np.transpose(data_AP_Q[1][3]),color=colors[i],fmt='o',markersize=4.5)
        #P over Q
        ax2.plot(data_AP_Q[0][2],np.array(data_AP_Q[0][1])/np.array(data_AP_Q[0][2])*np.pi*2,color=colors[i],label=labels[i],lw=1.5)
        ax2.errorbar(data_AP_Q[1][2],np.array(data_AP_Q[1][1])/np.array(data_AP_Q[1][2])*np.pi*2,yerr=np.transpose(data_AP_Q[1][4])*np.pi*2/[data_AP_Q[1][2],data_AP_Q[1][2]],color=colors[i],fmt='o',markersize=4.5)
    ax2.legend(frameon=False,fontsize=12)
    ax1.legend(frameon=False,fontsize=12)
    ax1.set_xlim(0,4)
    ax2.set_xlim(0,4)
    fig1.tight_layout()
    fig1.savefig(os.path.join(file_head,'multiple_APQ_profiles.png'),dpi=300)
    return fig1

def plot_AFM_profiles(file_head='P:\\My stuff\\Manuscripts\\zr on mica\\AFM images',profile_files=['AFM profile for 0mM NaCl.csv','AFM profile for 1mM NaCl.csv','AFM profile for 10mM NaCl.csv','AFM profile for 100mM NaCl.csv'],labels=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl'],color_type=5):
    colors=set_color(len(profile_files),color_type)
    fig=pyplot.figure(figsize=(6,6))
    for i in range(len(profile_files)):
        data=np.loadtxt(os.path.join(file_head,profile_files[i]),delimiter=',')
        ax=fig.add_subplot(2,2,i+1)
        ax.plot(data[:,1],data[:,0])
        pyplot.title(labels[i],fontsize=10)
        pyplot.xlabel("length (nm)")
        pyplot.ylabel("height (nm)")
    fig.tight_layout()
    fig.savefig(os.path.join(file_head,'AFM_profiles.png'),dpi=300)
    return fig


def cal_e_density(z_list,oc_list,u_list,z_min=0,z_max=29,resolution=1000,N=40,wt=0.25,Auc=46.927488088,water_scaling=1):
    height_list=[]
    e_list=[]
    z_min=float(z_min)
    z_max=float(z_max)
    def _density_at_z(z,z_cen,oc,u):
        return wt*N*oc*(2*np.pi*u**2)**-0.5*np.exp(-0.5/u**2*(z-z_cen)**2)/Auc
    for i in range(resolution):
        z_each=z_min+(z_max-z_min)/resolution*i
        e_temp=0
        for j in range(len(z_list)):
            z_cen=z_list[j]
            oc=oc_list[j]
            u=u_list[j]
            e_temp+=_density_at_z(z_each,z_cen,oc,u)
        height_list.append(z_each)
        e_list.append(e_temp/water_scaling)
    pickle.dump([height_list,e_list],open(os.path.join(module_path_locator(),"temp_plot_RAXR_eden_e_fit"),"wb"))
    pyplot.figure()
    pyplot.plot(height_list,e_list)
    return e_list

def fit_e_2(zs=None,water_scaling=1,fit_range=[1,40]):
    total_eden=pickle.load(open(os.path.join(module_path_locator(),"temp_plot_eden"),"rb"))[0][-1]
    raxr_eden=pickle.load(open(os.path.join(module_path_locator(),"temp_plot_RAXR_eden_e_fit"),"rb"))
    pyplot.figure()
    pyplot.plot(raxr_eden[0],raxr_eden[1])
    fit_data=np.append(np.array(raxr_eden[0])[:,np.newaxis],(np.array(total_eden[1,:])-np.array(raxr_eden[1]))[:,np.newaxis],axis=1)
    pyplot.figure()
    print('##############Total e - raxr - water#################')
    gaussian_fit(fit_data,fit_range=fit_range,zs=zs,water_scaling=water_scaling)
    pyplot.title('Total e - raxr -water')
    return None

def overplot_total_raxr_e_density():
    total_eden=pickle.load(open(os.path.join(module_path_locator(),"temp_plot_eden"),"rb"))[0][-1]
    raxr_eden=pickle.load(open(os.path.join(module_path_locator(),"temp_plot_RAXR_eden_e_fit"),"rb"))
    pyplot.figure()
    pyplot.plot(raxr_eden[0],raxr_eden[1],label='RAXR el e density')
    pyplot.plot(total_eden[0,:],total_eden[1,:],label='Total e density')
    pyplot.legend()
    return None

def overplot_raxr_e_density(dump_files=["temp_plot_RAXR_eden_e_fit_0mMNaCl","temp_plot_RAXR_eden_e_fit_1mMNaCl","temp_plot_RAXR_eden_e_fit_10mMNaCl","temp_plot_RAXR_eden_e_fit_100mMNaCl"],labels=['0mM NaCl','1mM NaCl','10mM NaCl','100mM NaCl']):
    fig=pyplot.figure()
    colors=set_color(len(dump_files),1)
    for i in range(len(dump_files)):
        dump_file=dump_files[i]
        label=labels[i]
        raxr_eden=pickle.load(open(os.path.join(module_path_locator(),dump_file),"rb"))
        pyplot.fill_between(raxr_eden[0],np.array(raxr_eden[1])+i,i,color=colors[i],label=label)
    pyplot.legend()
    #fig.savefig(os.path.join(os.path.join(module_path_locator(),'temp_raxr_e_profiles_overlapping_profile.png'),dpi=300))
    return fig

def plot_all(path=module_path_locator(),make_offset_of_total_e=False,fit_e_profile=0):
    #set make_offset_of_total_e to True if you want to have total e density replesented by total_e - resonant e in the case when the resonant els are freezed to have no influence on the CTR.
    #At the same time, the total_e - raxs_e - water is actually total_e - 2*raxs_e -water
    PATH=path
    #which plots do you want to create
    plot_e_model,plot_e_FS,plot_ctr,plot_raxr,plot_AP_Q=1,1,1,1,0

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
            pyplot.xlim(xmax=50)
            pyplot.legend(fontsize=11,ncol=1)
        fig.tight_layout()
        fig.savefig(e_file+".png",dpi=300)
    if plot_ctr:
        plotting_many_modelB(save_file=os.path.join(ctr_file_folder,"temp_plot_ctr.png"),head=ctr_file_folder,object_files=ctr_file_names,color=['b','r'],l_dashes=[(None,None)],lw=1)
        plt.show(block=False)
    if plot_raxr:
        #plot raxr profiles
        data_raxr=pickle.load(open(raxr_file,"rb"))
        plotting_raxr_new(data_raxr,savefile=raxr_file+".png",color=['b','r'],marker=['o'])
        plt.show(block=False)
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
        print('##############Total e -layer water#################')
        #gaussian_fit(e_den_subtracted,zs=None,water_scaling=water_scaling)
        gaussian_fit_DE(e_den_subtracted,zs=3,water_scaling=water_scaling)
        pyplot.title('Total e - layer water')

        pyplot.figure()
        print('#########################RAXR (MI)########################')
        gaussian_fit(e_den_raxr_MI,zs=None,N=40,water_scaling=water_scaling)
        pyplot.title('RAXR (MI)')
        pyplot.figure()
        '''
        print('#########################RAXR (MD)########################')
        gaussian_fit(np.append([data_eden_FS_sub[0]],[data_eden_FS_sub[1]*(np.array(data_eden_FS_sub[1])>0)],axis=0).transpose(),zs=None,N=40,water_scaling=water_scaling)
        pyplot.title('RAXR (MD)')
        pyplot.show()
        #return e_den_subtracted,data_eden_FS
        '''
    return water_scaling

def plot_all_old(path=module_path_locator(),make_offset_of_total_e=False,fit_e_profile=0):
    #set make_offset_of_total_e to True if you want to have total e density replesented by total_e - resonant e in the case when the resonant els are freezed to have no influence on the CTR.
    #At the same time, the total_e - raxs_e - water is actually total_e - 2*raxs_e -water
    PATH=path
    #which plots do you want to create
    plot_e_model,plot_e_FS,plot_ctr,plot_raxr,plot_AP_Q=1,1,1,0,0

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
        fig=pyplot.figure(figsize=(15,6))
        if plot_e_FS:
            data_eden_FS=pickle.load(open(e_file_FS,"rb"))
            data_eden_FS_sub=pickle.load(open(e_file_FS+"_sub","rb"))
        for i in range(N):
            if i==N-1:
                ax=fig.add_subplot(1,2,2)
            else:
                #ax=fig.add_subplot(N/2+1,2,i*2+1)
                ax=fig.add_subplot(N-1,2,i*2+1)
            if make_offset_of_total_e:
                try:
                    edata[i][1,:]=list(np.array(edata[i][1,:])-np.array(edata[i][2,:]))
                except:
                    pass
            else:
                pass
            ax.plot(np.array(edata[i][0,:]),edata[i][1,:],color='black',lw=2.5,linestyle='-',label="Total e density")
            ax.plot([0,0],[0,max(edata[i][1,:])+2],linestyle=':',color='m',lw=3,label='Mineral surface')
            #ax.plot(np.array(edata[i][0,:]),edata[i][2,:],color='blue')
            ax.fill_between(np.array(edata[i][0,:]),edata[i][2,:],alpha=0.2,color='m',label="RAXS element e profile (MD)")
            #try:#some domain may have no raxr element
            #    ax.plot(np.array(edata[i][0,:]),edata[i][2,:],color='g',label="RAXS element e profile (MD)")
            #except:
            #    pass
            pyplot.title(labels[i],fontsize=11)
            if plot_e_FS:
                #if i==0:
                if i!=N-1:
                    #ax.plot(data_eden_FS[0],list(np.array(data_eden_FS[2])[:,i]),color='r',label="RAXR imaging (MI)")
                    #ax.fill_between(data_eden_FS[0],list(np.array(data_eden_FS[2])[:,i]),color='m',alpha=0.6)
                    #clip off negative part of the e density through Fourier thynthesis
                    #ax.fill_between(data_eden_FS[0],list(edata[i][1,:]-edata[i][3,:]-np.array(data_eden_FS[2])[:,i]*(np.array(data_eden_FS[2])[:,i]>0.01)),color='black',alpha=0.6,label="Total e - LayerWater - RAXR")
                    ax.fill_between(data_eden_FS[0],edata[i][3,:],color='green',alpha=0.4,label="LayerWater")
                    #ax.plot(data_eden_FS_sub[0],list(np.array(data_eden_FS_sub[2])[:,i]),color='blue',label="RAXR imaging (MD)")
                    #ax.fill_between(data_eden_FS_sub[0],list(np.array(data_eden_FS_sub[2])[:,i]),color='c',alpha=0.6)
                elif i==N-1:
                    ax.plot(data_eden_FS[0],data_eden_FS[1],color='r',label="RAXR imaging (MI)")
                    #ax.fill_between(data_eden_FS[0],data_eden_FS[1],color='m',alpha=0.6)
                    #ax.fill_between(data_eden_FS[0],edata[i][1,:]-data_eden_FS[1],color='black',alpha=0.6,label="Total e - RAXR(MI)")
                    ax.fill_between(data_eden_FS[0],edata[i][3,:],color='green',alpha=0.4,label="LayerWater")
                    '''
                    if make_offset_of_total_e:
                        ax.fill_between(data_eden_FS[0],list(edata[i][1,:]-edata[i][3,:]-2*edata[i][2,:]),color='black',alpha=0.6,label="Total e - raxr (MD) - LayerWater")
                    else:
                        ax.fill_between(data_eden_FS[0],list(edata[i][1,:]-edata[i][3,:]-edata[i][2,:]),color='black',alpha=0.6,label="Total e - raxr (MD) - LayerWater")
                        #ax.fill_between(data_eden_FS[0],list(edata[i][1,:]-edata[i][3,:]-np.array(data_eden_FS[1])*(np.array(data_eden_FS[1])>0.01)),color='black',alpha=0.6,label="Total e - LayerWater - RAXR")
                        #eden_temp=list(edata[i][1,:]-edata[i][3,:]-np.array(data_eden_FS[1])*(np.array(data_eden_FS[1])>0.01))
                    '''
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
                    #ax.fill_between(data_eden_FS_sub[0],data_eden_FS_sub[1],color='m',alpha=0.6)
            if i==N-1:pyplot.xlabel('Z(Angstrom)',axes=ax,fontsize=12)
            pyplot.ylabel('E_density',axes=ax,fontsize=12)
            pyplot.ylim(ymin=0)
            pyplot.xlim(xmin=-5)
            pyplot.xlim(xmax=50)
            if i==N-1:pyplot.legend(fontsize=11,ncol=1)
        fig.tight_layout()
        fig.savefig(e_file+".png",dpi=300)
    if plot_ctr:
        #plot ctr profiles
        #plotting_single_rod(save_file=ctr_file_folder+"temp_plot_ctr.png",head=ctr_file_folder,object_files=ctr_file_names,color=['w','r'],l_dashes=[(None,None)],lw=2,rod_index=0)
        plotting_many_modelB(save_file=os.path.join(ctr_file_folder,"temp_plot_ctr.png"),head=ctr_file_folder,object_files=ctr_file_names,color=['b','r'],l_dashes=[(None,None)],lw=2)
        plt.show(block=False)
    if plot_raxr:
        #plot raxr profiles
        data_raxr=pickle.load(open(raxr_file,"rb"))
        plotting_raxr_new(data_raxr,savefile=raxr_file+".png",color=['b','r'],marker=['o'])
        plt.show(block=False)
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
        print('##############Total e -layer water#################')
        #gaussian_fit(e_den_subtracted,zs=None,water_scaling=water_scaling)
        gaussian_fit_DE(e_den_subtracted,zs=3,water_scaling=water_scaling)
        pyplot.title('Total e - layer water')

        pyplot.figure()
        print('#########################RAXR (MI)########################')
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
    print(popt)
    combinded_set=[]
    #print 'z occupancy*4 U(sigma**2)'
    for i in range(0,len(popt),3):
        combinded_set=combinded_set+[abs(popt[i])/N*(abs(popt[i])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2,popt[i+2]]
        #print '%3.3f\t%3.3f\t%3.3f'%(ctrs[i/2],abs(popt[i])/N*(abs(popt[i+1])*np.sqrt(np.pi*2)*5.199*9.027)*4,abs(popt[i+1])**2)
    combinded_set=np.reshape(np.array(combinded_set),(len(combinded_set)/3,3)).transpose()
    print(combinded_set)
    #combinded_set=combinded_set.transpose()
    #normalized to full surface unit cell
    print('total_occupancy=',np.sum(combinded_set[0,:]/4))
    #normalized to half unit cell (oc and u have been added to 1 to be used in Matlab input par file)
    print('OC_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])')
    #the u not U(u**2)
    print('U_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])')
    print('X_RAXS_LIST=[0.5]*',len(combinded_set[1,:]))
    print('Y_RAXS_LIST=[0.5]*',len(combinded_set[1,:]))
    print('Z_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])')


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
    print('total_occupancy=',np.sum(combinded_set[1,:]/4))
    print('OC_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])')
    print('U_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])')
    print('X_LIST=[0.5]*',len(combinded_set[1,:]))
    print('Y_LIST=[0.5]*',len(combinded_set[1,:]))
    print('Z_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])')


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
    print('total_occupancy=',np.sum(combinded_set[1,:]/4))
    print('OC_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[1,:]]),'])')
    print('U_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[2,:]]),'])')
    print('X_RAXS_LIST=[0.5]*',len(combinded_set[1,:]))
    print('Y_RAXS_LIST=[0.5]*',len(combinded_set[1,:]))
    print('Z_RAXS_LIST=np.array([',','.join([str(each) for each in combinded_set[0,:]]),'])')


    fit = func([x,ctrs], *popt)

    plt.plot(x, y)
    plt.plot(x, fit , 'r-')
    plt.show()

def cal_effective_charge(unit_step=2.5,NP_height=9*2.5,NP_width=30*2.5,k=2):
    #unit_step=O-O distance in ZrO2 structure (~2.5 A)
    #NP_height: maximum height of NP
    #NP_width: maximum width of NP
    #k:shrink factor between two adjacent layers, if k=2, then lay-1 is 2*unit_step larger in one edge compared to layer-0
    #We make an assumption that the NP has a symmetrical shape. From bottom to middle, lateral size increases; while from middle to top, lateral size decrease!
    l=int(NP_width/unit_step)
    h=int(NP_height/unit_step)
    i_mid=int(h/2)
    m=l-k*i_mid
    print('m=',m)
    effective_charge=0
    effective_charge_container=[]
    for i in range(h):
        if i<i_mid:
            #effective_charge+=m*4*(1+float(k)*i/m)/(i+1.0)**2
            effective_charge+=4./m/(1.+float(k)*i/m)/(i+1.0)**2#has been normalized by the area of each layer
        else:
            #effective_charge+=m*4*(1+i_mid*k+float(-k)*(i-i_mid)/m)/(i+1.0)**2
            effective_charge+=4./m/(1.+i_mid*k+float(-k)*(i-i_mid)/m)/(i+1.0)**2
        effective_charge_container.append(effective_charge)
    print('Nano particle of width %3.2f A and of height %3.2f has effective charge of %6.3f'%(NP_width,NP_height,effective_charge/(1+0)/(1+0)))
    print('Percentage of effective charge during loop')
    for each in effective_charge_container:
        print ('Layer %i has charge contribution of %3.2f in percentrage'%(effective_charge_container.index(each),each/effective_charge_container[-1]*100))
    return None


def cal_electrostatic_potential(d=2.5,H=9*2.5,L_max=30*2.5,L_min=2.5,deltaG_ion=-21.2,Auc=46.7):
    n_l=int((H/d+1.)/2.)
    E_j_container=[]
    E_ele_total=0
    for j in range(1,n_l+1,1):
        E_j=(2*(d*j-d)*(L_max-L_min)/H+L_min)*4/2.5*(H+2*d)/(H+2*d-d*j)/(d*j)/int(L_max**2/Auc+1)
        E_j_container.append(E_j)
        E_ele_total+=E_j
    print('Nano particle of width %3.2f A and of height %3.2f has electrostatic potential energy of %6.3f that is in an equivalence to %6.3f point charge'%(L_max,H,E_ele_total,E_ele_total/(1/2.5)))
    return E_ele_total,E_j_container

def cal_electrostatic_potential_2(d=2.5,d_l=2.5,H=9*2.5,L_max=30*2.5,L_min=2.5*3,deltaG_ion=-21.2,deltaG_ref=-15,Auc=46.7,deltaG_over_deltaz=2.4,damping_factor=0.2,first_z_ion=2.5):
    n_l=int((H/d_l+1.)/2.)
    E_j_container=[]
    E_ele_total=0
    deltaG_over_deltaz_container=[]
    deltaz_container=[]
    deltaG_container=[]
    for j in range(1,n_l+1,1):
        deltaG_over_deltaz_j=deltaG_over_deltaz/np.exp(damping_factor*(j-1))
        deltaG_over_deltaz_container.append(deltaG_over_deltaz_j)
        if j==1:
            deltaz_container.append(d-first_z_ion)
        else:
            deltaz_container.append(d_l)
        def _cal_delta_G(dG_init,dz_container,dG_dz_container):
            return np.sum(np.array(dz_container)*np.array(dG_dz_container))+dG_init
        N_j=(2*(d_l*j)*(L_max-L_min)/H+L_min)*4/2.5
        E_j=N_j*_cal_delta_G(deltaG_ref,deltaz_container,deltaG_over_deltaz_container)/int(L_max**2/Auc+1)
        deltaG_container.append(_cal_delta_G(deltaG_ref,deltaz_container,deltaG_over_deltaz_container))
        E_j_container.append(E_j*int(E_j<0))
        E_ele_total+=E_j*int(E_j<0)
    for j in range(n_l,int(H/d_l+1.),1):
        deltaG_over_deltaz_j=deltaG_over_deltaz/np.exp(damping_factor*(j-1))
        deltaG_over_deltaz_container.append(deltaG_over_deltaz_j)
        if j==1:
            deltaz_container.append(d-first_z_ion)
        else:
            deltaz_container.append(d_l)
        def _cal_delta_G(dG_init,dz_container,dG_dz_container):
            return np.sum(np.array(dz_container)*np.array(dG_dz_container))+dG_init
        N_j=(L_max-2*d_l*(j-n_l)*(L_max-L_min)/H)*4/2.5
        E_j=N_j*_cal_delta_G(deltaG_ref,deltaz_container,deltaG_over_deltaz_container)/int(L_max**2/Auc+1)
        deltaG_container.append(_cal_delta_G(deltaG_ref,deltaz_container,deltaG_over_deltaz_container))
        E_j_container.append(E_j*int(E_j<0))
        E_ele_total+=E_j*int(E_j<0)
    #print deltaG_over_deltaz_container
    #print deltaz_container
    #print deltaG_container
    print('Nano particle of width %3.2f A and of height %3.2f has electrostatic potential energy of %6.3f kJ/mol in comparison to %6.3f kJ/mol of competing ion'%(L_max,H,E_ele_total,deltaG_ion))
    return E_ele_total,E_j_container

def plot_trend_EI_with_size(L_range=np.arange(5*2.5,40*2.5,2.5),h=np.arange(12.5,30,2.5)):
    Li,Na,K,Rb,Cs=-16.7,-14.3,-22.2,-23.5,-21.2
    fig=pyplot.figure(figsize=(9,6))
    ax=fig.add_subplot(1,1,1)
    colors=set_color(len(h)+1,1)
    for i in range(len(h)):
        H=h[i]
        E_all=[]
        for j in L_range:
            E_all.append(cal_electrostatic_potential_2(H=H,L_max=j,damping_factor=0.1)[0])
        ax.plot(L_range,E_all,color=colors[i],label='Height='+str(H)+r'$\rm{\AA}$')
    ax.plot([55.7,55.7],[-50,-21.5],':k')
    ax.plot([54,54],[-50,-22.1],':k')
    ax.plot([51.4,51.4],[-50,-23.1],':k')
    ax.plot([63,63],[-50,-16.7],':k')
    ax.plot([73,73],[-50,-14],':k')
    ax.plot([L_range[0],63],[Li]*2,':k')
    ax.annotate('Li+',xy=(21,Li+0.2),xytext=(21,Li+0.2),fontsize=10,color='black')
    ax.annotate('Na+',xy=(21,Na+0.2),xytext=(21,Na+0.2),fontsize=10,color='black')
    ax.plot([L_range[0],73],[Na]*2,':k')
    ax.annotate('K+',xy=(21,K+0.2),xytext=(21,K+0.2),fontsize=10,color='black')
    ax.plot([L_range[0],53.5],[K]*2,':k')
    ax.annotate('Rb+',xy=(21,Rb+0.2),xytext=(21,Rb+0.2),fontsize=10,color='black')
    ax.plot([L_range[0],51.4],[Rb]*2,':k')
    ax.annotate('Cs+',xy=(21,Cs+0.2),xytext=(21,Cs+0.2),fontsize=10,color='black')
    ax.plot([L_range[0],55.7],[Cs]*2,':k')



    pyplot.xlabel(r'$\rm{Lateral\ size (\AA)}$')
    pyplot.ylabel(r'Adsorption Gibbs free energy (kJ/mol)')
    legend=pyplot.legend(loc=4,fontsize=12,fancybox=True)
    legend.draw_frame(False)
    pyplot.ylim(-50,-5)
    pyplot.xlim(20,90)
    fig.tight_layout()
    fig.savefig(os.path.join(module_path_locator(),'EI_trend_with_size.png'),dpi=300)

    return None


if __name__=="__main__":

    plot_all(make_offset_of_total_e=False)
