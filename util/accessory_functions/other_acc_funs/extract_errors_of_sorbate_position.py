import os,sys
SuPerRod_path=os.path.join(os.getcwd().rsplit('SuPerRod')[0],'SuPerRod')
sys.path.append(SuPerRod_path)
sys.path.append(os.path.join(SuPerRod_path,'models'))
import numpy as np
import model,diffev
import filehandling as io

#This small script is used to extract dxdydz errors of sorbate atoms based on scanning geometrical parameters with the error range from the best fit model file.
#To make it run faster, switch off the bond valence check (USE_BV=0). And switch off all I/O functions (running_mode=1)
#so far it can deal with 4 parameters at most.
path='P://temp_model'
gx_file_path=os.path.join(path,'Best_Pb_anneal_rcut_run1_fitdxdydz_Aug9_ran.gx')
scan_number=2#number of scan steps, 2 is usually good enough. Never set it to a number >5. Time consuming and unnecessary.
#Geometrical parameter you want to scan
scan_pars_list=['rgh_domain2.setPhi_BD_0','rgh_domain2.setTop_angle_BD_0','rgh_domain2.setOffset2_BD_0','rgh_domain2.setAngle_offset_BD_0']
#the atom id of sorbates (metal or O ligands)
monitor_variables=['Pb1_D2A','HO1_Pb1_D2A']
#since you want to extract the dxdydz info, so this list should not be changed.
monitor_pars=['x','y','z']
scan_pars_values=[]#each item has a format of ['best fit value','left boundary','right boundary'], last two values extracted from para table
scan_pars_pos=[]
monitor_variable_pos=[]
best_monitor_pars_values={}#each itme has a format of ['best fit value','left boundary','right boundary'], last two values dynamically changed
mod = model.Model()
config = io.Config()
opt = diffev.DiffEv()
io.load_gx(gx_file_path,mod,opt,config)
print 'Doing simulation the first time.\nBest fit par values is extracted now...'
mod.simulate()
table=np.array(mod.parameters.data)
for scan_par in scan_pars_list:
    temp_pos=np.where(table==scan_par)
    scan_pars_pos.append(temp_pos[0][0])
    boundary_temp=table[temp_pos[0][0],-1].replace('(','').replace(')','').rsplit(',')
    scan_pars_values.append([float(table[temp_pos[0][0],1]),float(boundary_temp[0]),float(boundary_temp[1])])
for monitor_var in monitor_variables:
    best_monitor_pars_values[monitor_var]={}
    monitor_variable_pos.append(list(getattr(getattr(mod.script_module,'domain{}A'.format(monitor_var.rsplit('_')[-1].replace('D','').replace('A',''))),'id')).index(monitor_var))
    for monitor_par in monitor_pars:
        best_monitor_pars_values[monitor_var][monitor_par]=[getattr(getattr(mod.script_module,'domain{}A'.format(monitor_var.rsplit('_')[-1].replace('D','').replace('A',''))),monitor_par)[monitor_variable_pos[-1]],0,0]
print 'Before scan, the values are as follows:'
for i in range(len(scan_pars_list)):
    print scan_pars_list[i],scan_pars_values[i]
for i in range(len(monitor_variables)):
    for j in range(len(monitor_pars)):
        print monitor_variables[i],monitor_pars[j],best_monitor_pars_values[monitor_variables[i]][monitor_pars[j]]
print 'Start scan the pars..'
#let's say the number of scan pars is fewer than 4
def loop_values(value,left,right,step_number,current_step):
    left_boundary=value+left
    right_boundary=value+right
    step=(right_boundary-left_boundary)/(step_number-1)
    current_value=left_boundary+step*current_step
    return current_value
def extract_value(domain_module,ids=[''],tags=['x'],monitor_values={}):
    for id in ids:
        tag_domain='domain'+id.split('_')[-1].replace('D','')
        domain=getattr(domain_module,tag_domain)
        index=list(domain.id).index(id)
        for tag in tags:
            current_temp=getattr(domain,tag)[index]
            current_values=monitor_values[id][tag]
            abs_diff=abs(current_temp-current_values[0])
            if (current_temp-current_values[0])>=0 and (abs_diff>current_values[2]):
                monitor_values[id][tag][2]=abs_diff
            elif (current_temp-current_values[0])<0 and (abs_diff>current_values[1]):
                monitor_values[id][tag][1]=-abs_diff
    return monitor_values

def _formate_values(value,errors):
    import math,decimal
    #eg value='1.245',errors=[-0.1,0.2], will return 1.2(2)
    value=float(value)
    if errors[0]==0 and errors[1]==0:
        return '{:0.3f}()'.format(value)
    error=None
    if abs(errors[0])>=abs(errors[1]):
        error=abs(errors[0])
    else:
        error=abs(errors[1])
    decimal_place=None
    error_tag=''
    try:
        temp='%e'%error
        if '+' in temp:
            error_tag=int(error)
            decimal_place=0
        else:
            decimal_place=int(temp.split('-')[1])
        if decimal_place>0:
            error_tag=int(round(10**decimal_place*error))
    except:
        pass
    if decimal_place>0 and error_tag==10:
        error_tag=9
    if decimal_place==0:
        if value<1:
            return '%i(%i)'%(1,error_tag)
        else:
            return '%2.1f(%i)'%(value,error_tag)
    return '{0:2.{1}f}({2})'.format(value,decimal_place,error_tag)

i_accum=0
if len(scan_pars_list)==1:
    for i in range(scan_number):
        i_accum+=1
        if i_accum%2==0:
            print 'Now doing loop{0} out of {1} loops in total'.format(i_accum,scan_number)
        current_value_i=loop_values(scan_pars_values[0][0],scan_pars_values[0][1],scan_pars_values[0][2],scan_number,i)
        mod.parameters.data[scan_pars_pos[0]][1]=current_value_i
        mod.simulate()
        best_monitor_pars_values=extract_value(domain_module=mod.script_module,ids=monitor_variables,tags=monitor_pars,monitor_values=best_monitor_pars_values)
elif len(scan_pars_list)==2:
    for i in range(scan_number):
        for j in range(scan_number):
            i_accum+=1
            if i_accum%10==0:
                print 'Now doing loop{0} out of {1} loops in total'.format(i_accum,scan_number**2)
            current_value_i=loop_values(scan_pars_values[0][0],scan_pars_values[0][1],scan_pars_values[0][2],scan_number,i)
            current_value_j=loop_values(scan_pars_values[1][0],scan_pars_values[1][1],scan_pars_values[1][2],scan_number,j)
            mod.parameters.data[scan_pars_pos[0]][1]=current_value_i
            mod.parameters.data[scan_pars_pos[1]][1]=current_value_j
            mod.simulate()
            best_monitor_pars_values=extract_value(domain_module=mod.script_module,ids=monitor_variables,tags=monitor_pars,monitor_values=best_monitor_pars_values)
elif len(scan_pars_list)==3:
    for i in range(scan_number):
        for j in range(scan_number):
            for k in range(scan_number):
                i_accum+=1
                if i_accum%20==0:
                    print 'Now doing loop{0} out of {1} loops in total'.format(i_accum,scan_number**3)
                current_value_i=loop_values(scan_pars_values[0][0],scan_pars_values[0][1],scan_pars_values[0][2],scan_number,i)
                current_value_j=loop_values(scan_pars_values[1][0],scan_pars_values[1][1],scan_pars_values[1][2],scan_number,j)
                current_value_k=loop_values(scan_pars_values[2][0],scan_pars_values[2][1],scan_pars_values[2][2],scan_number,k)
                mod.parameters.data[scan_pars_pos[0]][1]=current_value_i
                mod.parameters.data[scan_pars_pos[1]][1]=current_value_j
                mod.parameters.data[scan_pars_pos[2]][1]=current_value_k
                mod.simulate()
                best_monitor_pars_values=extract_value(domain_module=mod.script_module,ids=monitor_variables,tags=monitor_pars,monitor_values=best_monitor_pars_values)
elif len(scan_pars_list)==4:
    for i in range(scan_number):
        for j in range(scan_number):
            for k in range(scan_number):
                for l in range(scan_number):
                    i_accum+=1
                    if i_accum%50==0:
                        print 'Now doing loop{0} out of {1} loops in total'.format(i_accum,scan_number**4)
                    current_value_i=loop_values(scan_pars_values[0][0],scan_pars_values[0][1],scan_pars_values[0][2],scan_number,i)
                    current_value_j=loop_values(scan_pars_values[1][0],scan_pars_values[1][1],scan_pars_values[1][2],scan_number,j)
                    current_value_k=loop_values(scan_pars_values[2][0],scan_pars_values[2][1],scan_pars_values[2][2],scan_number,k)
                    current_value_l=loop_values(scan_pars_values[3][0],scan_pars_values[3][1],scan_pars_values[3][2],scan_number,l)
                    mod.parameters.data[scan_pars_pos[0]][1]=current_value_i
                    mod.parameters.data[scan_pars_pos[1]][1]=current_value_j
                    mod.parameters.data[scan_pars_pos[2]][1]=current_value_k
                    mod.parameters.data[scan_pars_pos[3]][1]=current_value_l
                    mod.simulate()
                    best_monitor_pars_values=extract_value(domain_module=mod.script_module,ids=monitor_variables,tags=monitor_pars,monitor_values=best_monitor_pars_values)
else:
    print 'More than 4 pars is not surported yet!'
print 'After scan, the values are as follows:'
for i in range(len(scan_pars_list)):
    print scan_pars_list[i],scan_pars_values[i]
for i in range(len(monitor_variables)):
    for j in range(len(monitor_pars)):
        temp=best_monitor_pars_values[monitor_variables[i]][monitor_pars[j]]
        print monitor_variables[i],monitor_pars[j],_formate_values(temp[0],temp[1:])
