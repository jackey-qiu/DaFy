import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.integrate as integrate
from scipy import signal

def RHE(E_AgAgCl, pH=13):
    # electrode is 3.4 M KCl
    return 0.205 + E_AgAgCl + 0.059*pH

#data format based on Fouad's potentiostat
def extract_cv_file(file_path='D:\\processed_data\\P23_I20180835\\ids\\x048_S221_CV', which_cycle=1):
    #return:time(s), pot(V), current (mA)
    skiprows = 0
    with open(file_path,'r') as f:
        for each in f.readlines():
            if each.startswith('Time(s)'):
                skiprows+=1
                break
            else:
                skiprows+=1
    data = np.loadtxt(file_path,skiprows = skiprows)
    #nodes index saving all the valley pot positions
    nodes =[0]
    for i in range(len(data[:,1])):
        if i!=0 and i!=len(data[:,1])-1:
            if data[i,1]<data[i+1,1] and data[i,1]<data[i-1,1]:
                nodes.append(i)
    nodes.append(len(data[:,1]))
    if which_cycle>len(nodes):
        print('Cycle number lager than the total cycles! Use the first cycle instead!')
        return data[nodes[1]:nodes[2],0], data[nodes[1]:nodes[2],1],data[nodes[1]:nodes[2],2]
    else:
        return data[nodes[which_cycle]:nodes[which_cycle+1],0],data[nodes[which_cycle]:nodes[which_cycle+1],1],data[nodes[which_cycle]:nodes[which_cycle+1],2]

def plot_cv_from_external(file_name='D:\\processed_data\\P23_I20180835\\ids\\x048_S221_CV', which_cycle=1, ph=10, cv_spike_cut=0.002, cv_scale_factor=30, scan_rate = 0.005,ax=None,color='r',label='',pot_range = [0.98,1.6]):
    pot_left, pot_right = pot_range
    t, pot,current = extract_cv_file(file_name, which_cycle)
    t_filtered, pot_filtered, current_filtered = t, pot, current
    for ii in range(4):
        filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
        filter_index = filter_index+1#index offset by 1
        t_filtered = t_filtered[(filter_index,)]
        pot_filtered = pot_filtered[(filter_index,)]
        current_filtered = current_filtered[(filter_index,)]
    pot_filtered = RHE(pot_filtered,pH=ph)
    if ax==None:
        _,ax = plt.subplots()
    ax.plot(pot_filtered,current_filtered*8*cv_scale_factor,label='',color = 'r')
    current_smooth = signal.savgol_filter(current_filtered*8*cv_scale_factor,5,3)
    ax.plot(pot_filtered,current_smooth,color = color,label=label)
    # ax.plot(RHE(pot,pH=ph),current*8,label='',color = 'r')
    ax.text(1.1,2,'x{}'.format(cv_scale_factor),color='r')
    # _,ax2 = plt.subplots()
    # ax2.plot(t_filtered,current_filtered*8*cv_scale_factor,label='',color = 'r')
    # return t_filtered, pot_filtered, current_filtered
    index_max = np.argmax(current_smooth)
    index_top_left = np.argmin(np.abs(pot_filtered[0:index_max]-pot_left))
    index_top_right = np.argmin(np.abs(pot_filtered[0:index_max]-pot_right))
    index_bottom_right = np.argmin(np.abs(pot_filtered[index_max:]-pot_left))+index_max
    index_bottom_left = np.argmin(np.abs(pot_filtered[index_max:]-pot_right))+index_max
    charge_top = metrics.auc(pot_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
    charge_bottom = metrics.auc(pot_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
    charge_top_2 = compute_area_under_a_curve(t_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
    charge_bottom_2 = compute_area_under_a_curve(t_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
    # charge_top_np = np.trapz(current_smooth[index_top_left:index_top_right],pot_filtered[index_top_left:index_top_right])
    # charge_bottom_np = np.trapz(current_smooth[index_bottom_left:index_bottom_right],pot_filtered[index_bottom_left:index_bottom_right])
    #*200 due to scan rate = 5 mV/s,
    #convert scan rate to time
    v_to_t = 1/scan_rate
    print('Resutls based on sklearn.metrics.auc api func')
    print('Total charge = {} mC/cm2 between {} V and {} V'.format((charge_top-charge_bottom)*v_to_t/cv_scale_factor/2,pot_left,pot_right))
    print('Results basedon my own hard-coded func')
    print('Total charge = {} mC/cm2 between {} V and {} V'.format((charge_top_2-charge_bottom_2)/cv_scale_factor/2, pot_left, pot_right))
    # print('numpy results: Total charge = {} mC/cm2'.format((charge_top_np-charge_bottom_np)*v_to_t/cv_scale_factor/2))
    return ax, t_filtered, pot_filtered, current_smooth

def calculate_charge(t, pot, current, which_cycle=1, ph=10, cv_spike_cut=0.002, cv_scale_factor=30, scan_rate = 0.005, pot_range = [0.98,1.6]):
    pot_left, pot_right = pot_range
    #t, pot,current = extract_cv_file(file_name, which_cycle)
    t_filtered, pot_filtered, current_filtered = t, pot, current
    for ii in range(4):
        filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
        filter_index = filter_index+1#index offset by 1
        t_filtered = t_filtered[(filter_index,)]
        pot_filtered = pot_filtered[(filter_index,)]
        current_filtered = current_filtered[(filter_index,)]
    pot_filtered = RHE(pot_filtered,pH=ph)
    current_smooth = signal.savgol_filter(current_filtered*8*cv_scale_factor,5,3)
    index_max = np.argmax(current_smooth)
    index_top_left = np.argmin(np.abs(pot_filtered[0:index_max]-pot_left))
    index_top_right = np.argmin(np.abs(pot_filtered[0:index_max]-pot_right))
    index_bottom_right = np.argmin(np.abs(pot_filtered[index_max:]-pot_left))+index_max
    index_bottom_left = np.argmin(np.abs(pot_filtered[index_max:]-pot_right))+index_max
    charge_top = metrics.auc(pot_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
    charge_bottom = metrics.auc(pot_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
    charge_top_2 = compute_area_under_a_curve(t_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
    charge_bottom_2 = compute_area_under_a_curve(t_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
    # charge_top_np = np.trapz(current_smooth[index_top_left:index_top_right],pot_filtered[index_top_left:index_top_right])
    # charge_bottom_np = np.trapz(current_smooth[index_bottom_left:index_bottom_right],pot_filtered[index_bottom_left:index_bottom_right])
    #*200 due to scan rate = 5 mV/s,
    #convert scan rate to time
    v_to_t = 1/scan_rate
    print('Resutls based on sklearn.metrics.auc api func')
    print('Total charge = {} mC/cm2 between {} V and {} V'.format((charge_top-charge_bottom)*v_to_t/cv_scale_factor/2,pot_left,pot_right))
    # print('Results basedon my own hard-coded func')
    # print('Total charge = {} mC/cm2 between {} V and {} V'.format((charge_top_2-charge_bottom_2)/cv_scale_factor/2, pot_left, pot_right))
    return (charge_top-charge_bottom)*v_to_t/cv_scale_factor/2

def compute_area_under_a_curve(x, y):
    area = 0
    for i in range(len(x)-1):
        x1, x2, y1, y2 = x[i],x[i+1],y[i],y[i+1]
        if y1*y2<0:
            x0 = -y1*(x1-x2)/(y1-y2)+x1
            area = area + (x0-x1)*y1/2+(x2-x0)*y2/2
        elif y1*y2==0:
            area = area + (y2-y1)*(x2-x1)/2
        else:
            if y1<0:
                area = area - abs((y1-y2)*(x1-x2)/2) - abs(min([abs(y1),abs(y2)])*(x2-x1))
            else:
                area = area + abs((y1-y2)*(x1-x2)/2) + abs(min([abs(y1),abs(y2)])*(x2-x1))
    return area