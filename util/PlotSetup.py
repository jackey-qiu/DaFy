import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

class ScanInfo():

    def __init__(self, scan_id, data, structure_lattice, HKL_position, scan_direction_ranges, color, analog_data_filename=None, analog_data_plot_range=[0,-1], ids_filename=None, analog_data_interval=None, scan_label=None, pH=13):
        self.scan_id = scan_id
        self.data = data
        self.structure_lattice = structure_lattice
        self.HKL_position = HKL_position
        self.scan_direction_ranges = scan_direction_ranges
        self.color = color
        self.analog_data_filename = analog_data_filename
        self.analog_data_plot_range = analog_data_plot_range
        self.ids_filename = ids_filename
        self.analog_data_interval = analog_data_interval
        self.scan_label = scan_label
        self.pH = pH

class ScanInfoContainer(dict):
    def __init__(self):
        return
    def add(self, scan_id, *args, **kwargs):
        self[scan_id] = ScanInfo(scan_id, *args, **kwargs)

#####################################################################
def RHE(E_AgAgCl, pH=13):
    # electrode is 3.4 M KCl
    return 0.205 + E_AgAgCl + 0.059*pH

def POT(E_AgAgCl, plot_vs_RHE=False, pH=13):
    if(plot_vs_RHE):
        return RHE(E_AgAgCl, pH)
    else:
        return E_AgAgCl

def find_tick_ticklables(x, num_ticks = 5, endpoint = True, dec_place = 1):
    x_ticks = np.linspace(min(x),max(x),num = num_ticks, endpoint = endpoint)
    steps_x = np.round(x_ticks[1]-x_ticks[0], dec_place)
    first_x = np.round(x_ticks[0],dec_place)
    x_ticks =[first_x+i*steps_x for i in range(num_ticks)]
    x_tick_labels = [str(np.round(each_x,dec_place)) for each_x in x_ticks]
    x_ticks = [float(each) for each in x_tick_labels]
    return x_ticks, x_tick_labels

def select_cycle_new(pot_result_frame_triple, bin_mode = 'select', return_cycle = 0, n_pots_rebin = 100, tailing_cut = 50):
    #bin_mode = 'average' or 'select'
    #   'average':average every bin_level data points
    #   'select': select every bin_level data point
    #plot_mode = 'CV' or 'pot_step'
    #   'CV': plot CV data
    #   'pot_step': plot potential step data
    pot_bin, result_bin = [], []
    pot, result, frame_numbers = pot_result_frame_triple
    pot, result, frame_numbers = pot[0:len(pot)-tailing_cut], result[0:len(result)-tailing_cut], frame_numbers[0:len(frame_numbers)-tailing_cut]
    pot = np.around(pot,decimals=6)
    pot_offset = pot.max()-pot.min()
    pot_keys_number = n_pots_rebin
    pot_keys = np.array([i*abs(pot[0]-pot[1])+pot.min() for i in range(pot_keys_number)])
    result_lib ={}
    for key in pot_keys:
        result_lib[(key,0)] = []
        result_lib[(key,1)] = []
    for i in range(len(pot)):
        current_pot = pot[i]
        up_or_down = 0
        if i ==0:
            up_or_down = int(pot[i+1]>pot[i])
        elif i == len(pot)-1:
            up_or_down = int(pot[i-1]<pot[i])
        else:
            if pot[i]>pot[i+1] and (frame_numbers[i+1]-frame_numbers[i])==1:
                up_or_down = 0
            elif pot[i]>pot[i-1] and  (frame_numbers[i]-frame_numbers[i-1])==1:
                up_or_down = 1
            else:
                up_or_down = 0
        current_key = (pot_keys[np.argmin(np.abs(pot_keys - current_pot))],up_or_down)
        result_lib[current_key].append(result[i])

    if bin_mode == 'select':
        for key in pot_keys:
            if result_lib[(key,0)]!=[]:
                try:
                    result_bin.append(result_lib[(key,0)][return_cycle])
                except:
                    result_bin.append(result_lib[(key,0)][0])
                pot_bin.append(key)
            else:
                pass
        for key in pot_keys:
            if result_lib[(key,1)]!=[]:
                try:
                    result_bin.append(result_lib[(key,1)][return_cycle])
                except:
                    result_bin.append(result_lib[(key,1)][0])
                pot_bin.append(key)
            else:
                pass
    elif bin_mode == 'average':
        for key in pot_keys:
            if result_lib[(key,0)]!=[]:
                result_bin.append(np.mean(result_lib[(key,0)]))
                pot_bin.append(key)
            else:
                pass
        for key in pot_keys:
            if result_lib[(key,1)]!=[]:
                result_bin.append(np.mean(result_lib[(key,1)]))
                pot_bin.append(key)
            else:
                pass
    return [0,int(len(pot_bin)/2), -2], np.array(pot_bin), np.array(result_bin)

def select_cycle_new2(pot_result_frame_mask_four, bin_mode = 'select', return_cycle = 0, n_pots_rebin = None, tailing_cut = 3):
    #bin_mode = 'average' or 'select'
    #   'average':average every bin_level data points
    #   'select': select every bin_level data point
    #plot_mode = 'CV' or 'pot_step'
    #   'CV': plot CV data
    #   'pot_step': plot potential step data
    #   'return_cycle' will be automatically set based on mask at bin_mode = 'select'
    pot_bin, result_bin = [], []
    pot, result, frame_numbers, mask = pot_result_frame_mask_four
    pot_offset = pot.max()-pot.min()
    total_cycle = int(round(len(pot)/(abs(pot_offset/(pot[1]-pot[0]))*2)))
    best_cycle = 0
    contamination_level = []
    for each_cycle in range(total_cycle):
        number_points_in_normal_range = 0
        points_per_cycle = int(len(pot)/total_cycle)
        index_of_current_cycle = np.array(range(points_per_cycle))+points_per_cycle*each_cycle
        for i in index_of_current_cycle:
            if mask[i]:
                number_points_in_normal_range+=1
            else:
                pass
        contamination_level.append(1-number_points_in_normal_range/float(points_per_cycle))
    best_cycle = contamination_level.index(min(contamination_level))
    #print('best_cycle is ',best_cycle,contamination_level)
    return_cycle = best_cycle
            
    pot, result, frame_numbers, mask = pot[tailing_cut:len(pot)-tailing_cut], result[tailing_cut:len(result)-tailing_cut], frame_numbers[tailing_cut:len(frame_numbers)-tailing_cut],mask[tailing_cut:len(frame_numbers)-tailing_cut]
    pot = np.around(pot,decimals=6)
    
    if n_pots_rebin == None:
        pot_keys_number = int(abs(pot_offset/(pot[1]-pot[0])))*2
    else:
        if n_pots_rebin > abs(list(pot).index(pot.min()) - list(pot).index(pot.max())):
            pot_keys_number = abs(list(pot).index(pot.min()) - list(pot).index(pot.max()))
        else:
            pot_keys_number = n_pots_rebin
    #print(pot_keys_number)
    pot_keys = np.array([i*abs(pot[0]-pot[1])+pot.min() for i in range(pot_keys_number)])
    result_lib ={}
    for key in pot_keys:
        result_lib[(key,0)] = []
        result_lib[(key,1)] = []
    for i in range(len(pot)):
        current_pot = pot[i]
        up_or_down = 0
        if i ==0:
            up_or_down = int(pot[i+1]>pot[i])
        elif i == len(pot)-1:
            up_or_down = int(pot[i-1]<pot[i])
        else:
            if pot[i]>pot[i+1] and (frame_numbers[i+1]-frame_numbers[i])==1:
                up_or_down = 0
            elif pot[i]>pot[i-1] and  (frame_numbers[i]-frame_numbers[i-1])==1:
                up_or_down = 1
            else:
                up_or_down = 0
        current_key = (pot_keys[np.argmin(np.abs(pot_keys - current_pot))],up_or_down)
        if mask[i]:
            result_lib[current_key].append(result[i])
        else:
            pass

    if bin_mode == 'select':
        for key in pot_keys:
            if result_lib[(key,0)]!=[]:
                try:
                    result_bin.append(result_lib[(key,0)][return_cycle])
                except:
                    result_bin.append(result_lib[(key,0)][0])
                pot_bin.append(key)
            else:
                pass
        for key in pot_keys:
            if result_lib[(key,1)]!=[]:
                try:
                    result_bin.append(result_lib[(key,1)][return_cycle])
                except:
                    result_bin.append(result_lib[(key,1)][0])
                pot_bin.append(key)
            else:
                pass
    elif bin_mode == 'average':
        for key in pot_keys:
            if result_lib[(key,0)]!=[]:
                result_bin.append(np.mean(result_lib[(key,0)]))
                pot_bin.append(key)
            else:
                pass
        for key in pot_keys:
            if result_lib[(key,1)]!=[]:
                result_bin.append(np.mean(result_lib[(key,1)]))
                pot_bin.append(key)
            else:
                pass
    return [0,int(len(pot_bin)/2), -2], np.array(pot_bin), np.array(result_bin)

def select_cycle(pot_result_couple, bin_level = 1, bin_mode = 'select', return_cycle = 0, pot_step = 0.11, plot_mode='CV'):
    #bin_mode = 'average' or 'select'
    #   'average':average every bin_level data points
    #   'select': select every bin_level data point
    #plot_mode = 'CV' or 'pot_step'
    #   'CV': plot CV data
    #   'pot_step': plot potential step data
    pot, result = pot_result_couple
    pot = np.around(pot,decimals=6)
    index_of_valley_pot = []
    min_pot_value = np.min(pot)
    temp_index = []
    #get valley points
    for i in range(1,len(pot)):
        if abs(pot[i]-min_pot_value)<pot_step:
            if temp_index!=[]:
                if (i-temp_index[-1])<10:
                    temp_index.append(i)
                else:
                    index_of_valley_pot.append(temp_index[int(len(temp_index)/2)])
                    temp_index = [i]
            else:
                temp_index.append(i)
    if temp_index!=[]:
        index_of_valley_pot.append(temp_index[int(len(temp_index)/2)])

    if plot_mode == 'pot_step':
        index_of_valley_pot = [0, len(pot)]
        bin_level = 1# you dont want to bin for pot_step dataset
        return [0, len(pot), -2], pot, result
    if return_cycle > len(index_of_valley_pot)-2:
        return_cycle = len(index_of_valley_pot)-2

    total_cycles = len(index_of_valley_pot)
    length_of_each_cycle = int(len(pot)/total_cycles)

    pot_partial, result_partial = pot[length_of_each_cycle*return_cycle:length_of_each_cycle*(return_cycle+1)],\
                                 result[length_of_each_cycle*return_cycle:length_of_each_cycle*(return_cycle+1)]
    # pot_partial, result_partial = pot[index_of_valley_pot[return_cycle]:index_of_valley_pot[return_cycle+1]],\
                                 # result[index_of_valley_pot[return_cycle]:index_of_valley_pot[return_cycle+1]]
    # pot_partial, result_partial = pot[0:index_of_valley_pot[return_cycle]],\
                                  # result[0:index_of_valley_pot[return_cycle]]
    def _bin_array(data, bin_level, bin_mode):
        if bin_mode =='average':
            data_bin =np.array(data[0::bin_level])
            for i in range(1,bin_level):
                temp = np.array(data[i::bin_level])
                if len(temp)<len(data_bin):
                    temp = np.append(temp, np.zeros(len(data_bin) - len(temp))+temp[-1])
                    data_bin = data_bin + temp
            return data_bin/bin_level
        elif bin_mode =='select':
            return np.array(data[0::bin_level])


    if bin_level>1:
        pot_bin = _bin_array(pot_partial, bin_level,bin_mode)
        result_bin = _bin_array(result_partial, bin_level,bin_mode)
        return [0, int((index_of_valley_pot[return_cycle+1]-index_of_valley_pot[return_cycle])/(bin_level*2)),-2],pot_bin, result_bin
    else:
        return [0, int((index_of_valley_pot[return_cycle+1]-index_of_valley_pot[return_cycle])/2),-2],pot_partial, result_partial

def set_max_to_0(data_list,slice_index,ref_at_0 = 1):
    if ref_at_0:
        return np.array(data_list)[slice_index[0]:slice_index[1]]-np.array(data_list).max()
    else:
        return np.array(data_list)[slice_index[0]:slice_index[1]]

#data format based on the output of IVIUM potentiostat
def extract_ids_file(file_path,which_cycle=3):
    data = []
    data_lines =[]
    current_cycle = 0
    with open(file_path,encoding="ISO-8859-1") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith('primary_data'):
                print(current_cycle)
                current_cycle=current_cycle+1
                if current_cycle == which_cycle:
                    for j in range(i+3,i+3+int(lines[i+2].rstrip())):
                        data.append([float(each) for each in lines[j].rstrip().rsplit()])
                    break
                else:
                    pass
            else:
                pass
    #return (pot: V, current: mA)
    return np.array(data)[:,0], np.array(data)[:,1]*1000

#data format based on Fouad's potentiostat
def extract_cv_data(file_path='/home/qiu/apps/048_S221_CV', which_cycle=1):
    #return:pot(V), current (mA)
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
        return data[nodes[1]:nodes[2],1],data[nodes[1]:nodes[2],2]
    else:
        return data[nodes[which_cycle]:nodes[which_cycle+1],1],data[nodes[which_cycle]:nodes[which_cycle+1],2]

file_head='/Users/cqiu/data/I20180835_Jul_2019_CVs'                                                                           
file_names_ph13_1=['039_S201_CV','041_S206_CV','043_S214_CV','045_S219_CV']                                                   
file_names_ph10=['047_S220_CV','049_S224_CV','050_S225_CV','051_S226_CV']                                                     
file_names_ph8=['056_S230_CV','057_S231_CV','058_S232_CV','059_S235_CV']                                                      
file_names_ph7=['063_S242_CV','064_S243_CV']                                                                                  
file_names_ph13_2=['053_S228_CV','054_S229_CV']                                                                               
file_names_ph13_3=['060_S236_CV','061_S237_CV']                                                                               
file_names_ph13_4=['065_S244_CV','066_S245_CV']  
file_names_ph13_all=['039_S201_CV','053_S228_CV','060_S236_CV','065_S244_CV']
file_names_different_phs=['048_S221_CV','054_S229_CV','057_S231_CV','060_S236_CV','064_S243_CV','065_S244_CV'] 

def plot_cvs_together(file_head, file_names, phs, labels,func=extract_cv_data, which_cycle=1, which_plot = 'all',cv_scale_factor = 25,cv_spike_cut = 0.01):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'E / V$_{RHE}$')
    ax.set_ylabel(r'j / mAcm$^{-2}$')
    if which_plot=='all':
        plot_index_list = range(len(file_names))
    else:
        plot_index_list = [which_plot]
    for i in plot_index_list:

        file = os.path.join(file_head, file_names[i])
        pot,current = func(file, which_cycle)
        pot_filtered, current_filtered = pot, current
        for ii in range(4):
            filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
            
            filter_index = filter_index+1#index offset by 1
            pot_filtered = pot_filtered[(filter_index,)]
            current_filtered = current_filtered[(filter_index,)]
        pot_filtered = RHE(pot_filtered,pH=phs[i])
        ax.plot(pot_filtered,current_filtered*8*cv_scale_factor,label =labels[i])
        ax.plot(RHE(pot,pH=phs[i]),current*8,label='')
        ax.set_ylim([min(current_filtered*8*cv_scale_factor),max(current*8)])
    plt.legend()
    plt.show()


plot_lib = {
            221:['048_S221_CV',1,0.002,30,'g',10],#data from I20180835, Co3O4
            229:['054_S229_CV',2,0.002,30,'r',13],#data from I20180835, Co3O4
            231:['057_S231_CV',2,0.002,30,'b',8],#data from I20180835, Co3O4
            236:['060_S236_CV',2,0.002,30,'r',13],#data from I20180835, Co3O4
            243:['064_S243_CV',2,0.002,30,'m',7],#data from I20180835, Co3O4
            244:['065_S244_CV',2,0.002,30,'r',13],#data from I20180835, Co3O4
            805:['023.ids',2,0.2,100,'r',13],#data from I20180114, CoOOH
            807:['025.ids',2,0.2,100,'b',8]#data from I20180114, CoOOH
           }

def data_viewer_plot_cv(ax,scan_no,marker_pos,plot_lib=plot_lib):
    file_name,which_cycle,cv_spike_cut,cv_scale_factor,color,ph = plot_lib[scan_no]
    if scan_no not in [805,807]:
        file = os.path.join('/Users/cqiu/data/I20180835_Jul_2019_CVs', file_name)
        pot,current = extract_cv_data(file, which_cycle)
    else:
        file = os.path.join('/Users/cqiu/data/I20180114_CVs', file_name)
        pot,current = extract_ids_file(file, which_cycle)

    pot_filtered, current_filtered = pot, current
    for ii in range(4):
        filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
        filter_index = filter_index+1#index offset by 1
        pot_filtered = pot_filtered[(filter_index,)]
        current_filtered = current_filtered[(filter_index,)]
    pot_filtered = RHE(pot_filtered,pH=ph)
    ax.plot(pot_filtered,current_filtered*8*cv_scale_factor,label='',color = color)
    ax.plot(RHE(pot,pH=ph),current*8,label='',color = color)
    ax.text(1.1,2,'x{}'.format(cv_scale_factor),color=color)
    for each in marker_pos:
        ax.plot([each,each],[-100,100],':k')
    #ax.set_ylim([min(current_filtered*8*cv_scale_factor),max(current*8)])
    return min(current_filtered*8*cv_scale_factor),max(current*8)

def ir_drop_analysis(pot, current,pot_first, half = 1):
    #half:1 or 2 (which half of the CV profile)
    #pot_first: index of the first potential for linear fit
    #return: resistance R
    if half==1:
        pot_fit = pot[0:int(len(pot)/2)]
        current_fit = current[0:int(len(pot)/2)]
    else:
        pot_fit = pot[int(len(pot)/2):len(pot)][::-1]
        current_fit = current[int(len(pot)/2):len(pot)][::-1]
    indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_first)),len(pot_fit)]
    slope,intercept,*others =stats.linregress(current_fit[indx1:indx2],pot_fit[indx1:indx2])
    print('R=',slope)
    # plt.plot(pot,np.log10(current))
    # plt.plot(current*slope+intercept,np.log10(current))
    plt.plot(pot,current)
    plt.plot(current*slope+intercept,current)
    plt.show()

def plot_tafel(file_head='/Users/canrong/Documents/I20180835_Jul_2019_CVs', cv_files = ['054_S229_CV','048_S221_CV','057_S231_CV','064_S243_CV'],phs=[13,10,8,7],pot_starts=[1.6,1.6,1.69,1.7], colors = ['r','g','b','m'],labels=['pH 13', 'pH 10', 'pH 8', 'pH 7'],half = 0, resistance = [60,50,0,30]):
    #half = 0, first half cycle E scan from low to high values
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel(r'E / V$_{RHE}$')
    ax.set_ylabel(r'j / mAcm$^{-2}$')
    # ax2 = fig.add_subplot(212)
    #labels = ['pH {}'.format(ph) for ph in phs]
    for i in range(len(phs)):
        cv = os.path.join(file_head,cv_files[i])
        ph = phs[i]
        pot_start=pot_starts[i]
        pot, current = extract_cv_data(cv,1)
        # ax2.plot(pot)
        if half==1:
            pot_fit = RHE(pot[0:int(len(pot)/2)],ph)
            current_fit = current[0:int(len(pot)/2)]
        else:
            pot_fit = RHE(pot[int(len(pot)/2):len(pot)][::-1],ph)
            current_fit = current[int(len(pot)/2):len(pot)][::-1]
        indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_start)),len(pot_fit)]
        ax.plot(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]*0.001),current_fit[indx1:indx2]*8,label=labels[i],color = colors[i])
    plt.legend()
    plt.show()
    fig.savefig('Tafel.png', dpi=300, bbox_inches='tight')

def plot_tafel_from_formatted_cv_info(cv_info, resistance = [80,70,40,50,90,50], pot_starts= [1.62,1.6,1.72,1.6,1.75,1.6], pot_ends =[1.75,1.65,1.85,1.65,1.83,1.65], potential_for_reaction_order = 1.65, forward_cycle = True):
    #half = 0, first half cycle E scan from low to high values
    if forward_cycle:
        half = 0
    else:
        half = 1
    if type(pot_starts)!=list:
        pot_starts = [pot_starts]*len(cv_info)
    else:
        pass
    if type(pot_ends)!=list:
        pot_ends = [pot_ends]*len(cv_info)
    else:
        pass
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.set_yscale('log')
    ax.set_xlabel(r'E / V$_{RHE}$')
    ax.set_ylabel(r'j / mAcm$^{-2}$')
    ax2.set_xlabel(r'pH')
    over_E = round(potential_for_reaction_order-1.23,2)
    
    ax2.set_ylabel(r'log(j / mAcm$^{-2}$)'+r',$\eta$= {}V'.format(over_E))
    # ax2 = fig.add_subplot(212)
    #labels = ['pH {}'.format(ph) for ph in phs]
    scans = list(cv_info.keys())
    scans = sorted(scans)
    log_current_density = []
    pHs = []
    for i in range(len(scans)):
        #ph = phs[i]
        pHs.append(cv_info[scans[i]]['pH'])
        label = 'scan {}_pH {}'.format(scans[i],cv_info[scans[i]]['pH'])
        color = cv_info[scans[i]]['color']
        pot_start=pot_starts[i]
        pot_end=pot_ends[i]
        print(label,': resistance ={};pot_range between {} and {}'.format(resistance[i],pot_start, pot_end))
        current = cv_info[scans[i]]['current_density']
        pot = cv_info[scans[i]]['potential']
        # ax2.plot(pot)
        if half==1:
            pot_fit = pot[0:int(len(pot)/2)]
            current_fit = current[0:int(len(pot)/2)]
        else:
            pot_fit = pot[int(len(pot)/2):len(pot)][::-1]
            current_fit = current[int(len(pot)/2):len(pot)][::-1]
        indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_start)),np.argmin(abs(np.array(pot_fit)-pot_end))]
        ax.plot(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),current_fit[indx1:indx2],label=label,color = color)
        ax.legend()
        #linear regression
        slope,intercept,r_value, *others =stats.linregress(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),np.log10(current_fit[indx1:indx2]))
        print('Linear fit results: log(current) = {} E + {}, R2 = {}'.format(slope, intercept, r_value**2))
        log_current_density.append(potential_for_reaction_order*slope+intercept)
    ax2.plot(pHs, log_current_density, 'og')
    slope_, intercept_, r_value_, *_ = stats.linregress(pHs, log_current_density)
    f = lambda x: slope_*x + intercept_
    x_min, x_max = min(pHs), max(pHs)
    ax2.plot([x_min,x_max],[f(x_min),f(x_max)],'-r')
    text_label = 'y = {}x{}, R2 = {}'.format(round(slope_,3), round(intercept_,3), round(r_value_**2,3))
    ax2.text(x_min,f(x_min),text_label)
    print('Reaction order fit: log(current) = {}pH + {}, R2 = {}'.format(slope_, intercept_, r_value_**2))
    #plt.legend()
    plt.show()
    #fig.savefig('Tafel.png', dpi=300, bbox_inches='tight')

def overplot_ctr_temp(data,scans=[217,222,233,239],colors=['r','g','b','m']):
    #data in pandas.DataFrame format
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel(r'L')
    ax.set_ylabel(r'F')
    for i in range(len(scans)):
        condition = data['scan_no']==scans[i]
        L = data['L'][condition]
        intensity = data['peak_intensity'][condition]
        ax.plot(L,intensity,color = colors[i],label = 'scan'+str(scans[i]))
    ax.legend()
    plt.show()

#temp to test different resistance for one cv file
def plot_tafel_temp(file_head='D:\\processed_data\\P23_I20180835\\ids', cv_file = 'x064_S243_CV',ph=7,pot_start=1.7, color = 'm',label='pH 7',half = 1, resistances = [0,0,0,0,0]):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.set_xlabel(r'E / V$_{RHE}$')
    ax.set_ylabel(r'j / mAcm$^{-2}$')
    #labels = ['pH {}'.format(ph) for ph in phs]
    phs = [ph]
    pot_starts = [pot_start]
    colors = [color]
    labels = [label]
    cv_files = [cv_file]

    for resistance in resistances:
        for i in range(len(phs)):
            cv = os.path.join(file_head,cv_files[i])
            ph = phs[i]
            pot_start=pot_starts[i]
            pot, current = extract_cv_data(cv,1)
            if half==1:
                pot_fit = RHE(pot[0:int(len(pot)/2)],ph)
                current_fit = current[0:int(len(pot)/2)]
            else:
                pot_fit = RHE(pot[int(len(pot)/2):len(pot)][::-1],ph)
                current_fit = current[int(len(pot)/2):len(pot)][::-1]
            indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_start)),len(pot_fit)]
            ax.plot(pot_fit[indx1:indx2]-0.001*resistance*current_fit[indx1:indx2],current_fit[indx1:indx2]*8, label=labels[i]+', when R = {}'.format(resistance))
    plt.legend()
    plt.show()
    fig.savefig('Tafel.png', dpi=300, bbox_inches='tight')

def strain_ip(q_ip, HKL, lattice):
    q_bulk = lattice.q(HKL)
    q_ip_bulk = np.sqrt(q_bulk[0]**2 + q_bulk[1]**2)
    return (q_ip_bulk/q_ip - 1.0)*100.

def strain_ip_with_uncertainty(q_ip, HKL, lattice, uncertainty_q_ip):
    q_bulk = lattice.q(HKL)
    q_ip_bulk = np.sqrt(q_bulk[0]**2 + q_bulk[1]**2)
    _strain_ip = (q_ip_bulk/q_ip - 1.0)*100.
    uncedrtainty_strain_ip = np.abs(q_ip_bulk/q_ip**2*100*uncertainty_q_ip)
    return (_strain_ip, uncedrtainty_strain_ip)

def strain_oop(q_oop, HKL, lattice):
    return (lattice.q(HKL)[2]/q_oop - 1.0)*100.

def strain_oop_with_uncertainty(q_oop, HKL, lattice, uncertainty_q_oop):
    q_oop_bulk = lattice.q(HKL)[2]
    _strain_oop = (q_oop_bulk/q_oop - 1.0)*100.
    uncertainty_strain_oop = np.abs(q_oop_bulk/q_oop**2*100*uncertainty_q_oop)
    return  (_strain_oop, uncertainty_strain_oop)
#####################################################################################################################################

