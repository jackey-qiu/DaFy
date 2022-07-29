import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
from scipy import signal
from sklearn import metrics
from scipy import stats
try:
    import ConfigParser as configparser
except:
    import configparser
try:
    from . import locate_path
except:
    import locate_path
script_path = locate_path.module_path_locator()
DaFy_path =os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'FilterPool'))
sys.path.append(os.path.join(DaFy_path,'util'))
#from UtilityFunctions import extract_vars_from_config

def extract_vars_from_config(config_file, section_var = None):
    config = configparser.ConfigParser()
    config.read(config_file)
    if section_var == None:
        section_var = config.sections()
    kwarg = {}
    for item in section_var:
        for each in config.items((item)):
            try:
                kwarg[each[0]] = eval(each[1])
            except:
                kwarg[each[0]] = each[1]
    return kwarg

def RHE(E_AgAgCl, pH=13):
    # electrode is 3.4 M KCl
    return 0.205 + E_AgAgCl + 0.059*pH

class cvAnalysis(object):
    def __init__(self, **kwargs):
        self.info = {}
        self.cv_info = {}
        self._extract_info(kwargs)
        #self.info = kwargs

    def _extract_info(self, info_lib):
        if 'config_file' in info_lib:
            self._extract_parameter_from_config(info_lib['config_file'])
        else:
            self.info = info_lib

    def _extract_parameter_from_config(self,config_file, sections = [None]):
        #in Global section you should have the following items (all in list type)
        #ph = [], color =, fmt = , path = , method =, pot_range = , sequence_id =, cv_scale_factor =, cv_spike_cut=, cv_scan_rate=
        self.info = {}
        self.cv_info = {}
        keys = ['sequence_id','ph','fmt','which_cycle','color','method','pot_range','cv_scale_factor',
                'cv_spike_cut','scan_rate','resistance','pot_starts_tafel','pot_ends_tafel','potential_reaction_order',
                'cv_folder','path','current_filter_length','current_filter_order','reaction_order_mode','current_reaction_order']
        for section in sections:
            kwarg_temp = extract_vars_from_config(config_file, section_var = section)
            for each in kwarg_temp:
                print('Extracting {} now'.format(each))
                # if each == 'cv_folder':
                    # self.info[each] = kwarg_temp[each]
                # else:
                self.info[each] = kwarg_temp[each]
        item_missing = []
        for key in keys:
            if key not in self.info:
                item_missing.append(key)
        return item_missing

    def change_potential_range(self, sequence_id, new_range):
        if 'pot_range' not in self.info:
            print('No key of pot_range in self.info')
            return
        try:
            which = self.info['sequence_id'].index(sequence_id)
        except:
            raise '{} is not in the list of sequence_id'.format(sequence_id)
            return
        self.info['pot_range'][which] = new_range

    def change_cv_scale_factor(self, sequence_id, new_cv_scale_factor):
        if 'cv_scale_factor' not in self.info:
            print('No key of cv_scale_factor inf self.info')
            return
        try:
            which = self.info['sequence_id'].index(sequence_id)
        except:
            raise '{} is not in the list of sequence_id'.format(sequence_id)
            return
        self.info['cv_scale_factor'][which] = new_cv_scale_factor 

    def change_cv_spike_cut(self, sequence_id, new_cv_spike_cut):
        if 'cv_spike_cut' not in self.info:
            print('No key of cv_spike_cut inf self.info')
            return
        try:
            which = self.info['sequence_id'].index(sequence_id)
        except:
            raise '{} is not in the list of sequence_id'.format(sequence_id)
            return
        self.info['cv_spike_cut'][which] = new_cv_spike_cut

    def extract_cv_info(self):
        self.cv_info = {}
        for i in range(len(self.info['sequence_id'])):
            pot, current = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            ph = self.info['ph'][i]
            color = self.info['color'][i]
            self.cv_info[self.info['sequence_id'][i]] = {'current_density':current*8, 'potential':RHE(pot,pH=ph), 'pH':ph, 'color':color}

    def calc_charge_all(self):
        self.info['charge'] = []
        outputs = []
        for i in range(len(self.info['sequence_id'])):
            pot, current = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            ph = self.info['ph'][i]
            cv_spike_cut = self.info['cv_spike_cut'][i]
            cv_scale_factor = self.info['cv_scale_factor'][i]
            scan_rate = self.info['scan_rate'][i]
            pot_range = self.info['pot_range'][i]
            print('Processing sequence {} now ... '.format(self.info['sequence_id'][i]))
            outputs.append('Processing sequence {} now ... '.format(self.info['sequence_id'][i]))
            charge, _output = self.calculate_pseudocap_charge(pot, current, ph, cv_spike_cut, cv_scale_factor, scan_rate, pot_range)
            self.info['charge'].append(charge)
            outputs.append(_output)
        return '\n'.join(outputs)

    #filter out the spikes on CV due to beam shutter on/off
    def filter_current_(self, pot, current, cv_spike_cut, times = 4):
        pot_filtered, current_filtered = pot, current
        for ii in range(times):
            filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
            filter_index = filter_index+1#index offset by 1
            #t_filtered = t_filtered[(filter_index,)]
            pot_filtered = pot_filtered[(filter_index,)]
            current_filtered = current_filtered[(filter_index,)]
        return pot_filtered, current_filtered

    #filter out the spikes on CV due to beam shutter on/off using smoothing func
    def filter_current(self, pot, current, filter_length, filter_order):
        # return pot, signal.savgol_filter(current,)
        return pot, signal.savgol_filter(current,filter_length,filter_order)

    #assuming symmetrical potential for one sweep
    #The original pot starts in between pot_min and pot_max
    def format_pot_current(self, pot, current):
        #return formatted_pot, which starts from the lowest value and ends at the lowest value
        #The order of current will change accordingly
        pot, current = np.array(pot), np.array(current)
        idx_max_pot = np.argmin(pot)
        idx_min_pot = np.argmin(pot)
        f = lambda pot, id_min, id_max:list(pot[(id_min+1):])+list(pot[0:id_max]) + list(pot[id_max:(id_min+1)])
        return np.array(f(pot, idx_min_pot, idx_max_pot)), np.array(f(current, idx_min_pot, idx_max_pot))

    def calculate_pseudocap_charge(self, pot, current, ph=10, cv_spike_cut=0.002, cv_scale_factor=30, scan_rate = 0.005, pot_range = [0.98,1.6]):
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
        pot_left, pot_right = pot_range
        pot_filtered, current_filtered = self.filter_current(pot, current,self.info['current_filter_length'],self.info['current_filter_order'])
        pot_filtered = RHE(pot_filtered,pH=ph)
        #extra smoothing step
        current_smooth = signal.savgol_filter(current_filtered*8*cv_scale_factor,5,3)
        index_max = np.argmax(current_smooth)
        index_top_left = np.argmin(np.abs(pot_filtered[0:index_max]-pot_left))
        index_top_right = np.argmin(np.abs(pot_filtered[0:index_max]-pot_right))
        index_bottom_right = np.argmin(np.abs(pot_filtered[index_max:]-pot_left))+index_max
        index_bottom_left = np.argmin(np.abs(pot_filtered[index_max:]-pot_right))+index_max
        charge_top = metrics.auc(pot_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
        charge_bottom = metrics.auc(pot_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
        #convert scan rate to time
        v_to_t = 1/scan_rate
        print('Resutls based on sklearn.metrics.auc api func')
        output = 'Total charge = {} mC/cm2 between {} V and {} V'.format((charge_top-charge_bottom)*v_to_t/cv_scale_factor/2,pot_left,pot_right)
        print(output)
        return (charge_top-charge_bottom)*v_to_t/cv_scale_factor/2, output

    @staticmethod
    def calculate_pseudocap_charge_stand_alone(pot, current, scan_rate = 0.005, pot_range = [0.98,1.6]):
        #pot: RHE potential in V
        #current: filtered and smoothed current without scaling
        #scan_rate: V per second
        #pot_range: list of two item, which define the bounds of potential (in V according to RHE scale)
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
        pot_left, pot_right = pot_range
        index_max = np.argmax(current)
        index_top_left = np.argmin(np.abs(pot[0:index_max]-pot_left))
        index_top_right = np.argmin(np.abs(pot[0:index_max]-pot_right))
        index_bottom_right = np.argmin(np.abs(pot[index_max:]-pot_left))+index_max
        index_bottom_left = np.argmin(np.abs(pot[index_max:]-pot_right))+index_max
        charge_top = metrics.auc(pot[index_top_left:index_top_right],current[index_top_left:index_top_right])
        charge_bottom = metrics.auc(pot[index_bottom_left:index_bottom_right],current[index_bottom_left:index_bottom_right])
        #convert scan rate to time
        v_to_t = 1/scan_rate
        output = 'Total charge = {} mC/cm2 between {} V and {} V'.format((charge_top-charge_bottom)*v_to_t/2,pot_left,pot_right)
        # print(output)
        return (charge_top-charge_bottom)*v_to_t/2, output

    #doesnot work yet!
    def locate_OER_onset(self, current, pot):
        current, pot = np.array(current), np.array(pot)
        idx_half_max = np.argmin(abs(current-max(current)/2))
        idx_max = np.argmax(current)
        index_left_second_line = int(idx_half_max - (idx_max - idx_half_max)*0.3)
        index_right_second_line = idx_half_max
        index_left_first_line = 10
        index_right_first_line = 50 
        print(len(pot),idx_max,index_left_first_line,index_left_second_line,index_right_first_line,index_right_second_line)
        slope_first_line, intercept_first_line, r_value_, *_ = stats.linregress(list(range(index_left_first_line,index_right_first_line)), current[index_left_first_line:index_right_first_line])
        slope_second_line, intercept_second_line, r_value_, *_ = stats.linregress(list(range(index_left_second_line,index_right_second_line)), current[index_left_second_line:index_right_second_line])
        cross_point_x = int((intercept_first_line - intercept_second_line)/(slope_second_line-slope_first_line))
        print('OER pot is {} V_RHE'.format(pot[cross_point_x]))
        return pot[cross_point_x]

    #data format based on the output of IVIUM potentiostat
    #note first cycle corresponds to which_cycle = 1
    #for potential step results, you need to use_all = True
    @staticmethod
    def extract_cv_file_ivium(file_path,which_cycle=3, use_all = False):
        if which_cycle == 0:
            which_cycle = 1
        data = []
        current_cycle = 0
        with open(file_path,encoding="ISO-8859-1") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith('primary_data'):
                    print(current_cycle)
                    current_cycle=current_cycle+1
                    if not use_all:
                        if current_cycle == which_cycle:
                            for j in range(i+3,i+3+int(lines[i+2].rstrip())):
                                data.append([float(each) for each in lines[j].rstrip().rsplit()])
                            break
                        else:
                            pass
                    else:
                        for j in range(i+3,i+3+int(lines[i+2].rstrip())):
                            data.append([float(each) for each in lines[j].rstrip().rsplit()])
                else:
                    pass
        #one more step to format the data so that the starting point is at the lowest potential
        data = np.array(data)
        index_min_pot = np.argmin(data[:,0])
        pot, current_density = list(data[:,0]), list(data[:,1]*1000)
        pot = pot[index_min_pot:]+pot[0:index_min_pot]
        current_density = current_density[index_min_pot:]+current_density[0:index_min_pot]
        #return (pot: V, current: mA)
        return np.array(pot),np.array(current_density)

    #data format based on Fouad's potentiostat
    @staticmethod
    def extract_cv_file_fouad(file_path='/home/qiu/apps/048_S221_CV', which_cycle=1, use_all = False):
        #return: pot(V), current (mA)
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
        if use_all:
            return data[:,1], data[:,2]
        else:
            if which_cycle>len(nodes):
                print('Cycle number lager than the total cycles! Use the first cycle instead!')
                return data[nodes[1]:nodes[2],1],data[nodes[1]:nodes[2],2]
            else:
                return data[nodes[which_cycle]:nodes[which_cycle+1],1],data[nodes[which_cycle]:nodes[which_cycle+1],2]

    #only one cycle, which can be manually exported from BioLogic software
    @staticmethod
    def extract_cv_file_biologic(file_path ='', which_cycle =1,use_all = False):
        #return: time(s), potential(V), current(mA)
        #skip three rows for header info
        def format_pot_current(pot, current):
            #return formatted_pot, which starts from the lowest value and ends at the lowest value
            #The order of current will change accordingly
            pot, current = np.array(pot), np.array(current)
            idx_max_pot = np.argmin(pot)
            idx_min_pot = np.argmin(pot)
            f = lambda pot, id_min, id_max:list(pot[(id_min+1):])+list(pot[0:id_max]) + list(pot[id_max:(id_min+1)])
            return np.array(f(pot, idx_min_pot, idx_max_pot)), np.array(f(current, idx_min_pot, idx_max_pot))
        data = np.loadtxt(file_path,skiprows = 3)
        pot, current = format_pot_current(data[:,1], data[:,2])
        return pot, current

    #all cycles are stored in the file. File format has four columns: time/s, Ewe/V, I/mA, cycle number (starting from 1)
    @staticmethod
    def extract_cv_file_biologic_multi_cycles(file_path ='', which_cycle =1,use_all = False):
        #return: time(s), potential(V), current(mA)
        #skip three rows for header info
        def format_pot_current(pot, current):
            #return formatted_pot, which starts from the lowest value and ends at the lowest value
            #The order of current will change accordingly
            pot, current = np.array(pot), np.array(current)
            idx_max_pot = np.argmin(pot)
            idx_min_pot = np.argmin(pot)
            f = lambda pot, id_min, id_max:list(pot[(id_min+1):])+list(pot[0:id_max]) + list(pot[id_max:(id_min+1)])
            return np.array(f(pot, idx_min_pot, idx_max_pot)), np.array(f(current, idx_min_pot, idx_max_pot))
        data = np.loadtxt(file_path,skiprows = 1)
        rows = data[:,3]==which_cycle
        pot, current = format_pot_current(data[rows,1], data[rows,2])
        return pot, current

    def _update_bounds(self, current_bounds, data):
        min_, max_ = min(data), max(data)
        return [min([min_,current_bounds[0]]), max([max_,current_bounds[1]])]

    def plot_cv_files(self,axs = []):
        if len(self.cv_info)==0:
            self.extract_cv_info()
        pot_bounds = [1000, -1000]
        current_bounds = [1000, -1000]
        # fig, axes1= plt.subplots(len(self.info['sequence_id']), 1)
        if len(axs)!=0:
            axes2 = axs
            fig2 = None
        else:
            fig2, axes2= plt.subplots(2, int(len(self.info['sequence_id'])/2+len(self.info['sequence_id'])%2),figsize=(8,4))
            axes2 = axes2.flatten()
        for i in range(len(self.info['sequence_id'])):
            pot_origin, current_origin = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            ph = self.info['ph'][i]
            color = self.info['color'][i]
            cv_spike_cut = self.info['cv_spike_cut'][i]
            cv_scale_factor = self.info['cv_scale_factor'][i]
            pot, current = self.filter_current(pot_origin, current_origin, self.info['current_filter_length'],self.info['current_filter_order'])
            pot_bounds = self._update_bounds(pot_bounds, RHE(pot_origin,pH=ph))
            current_bounds = self._update_bounds(current_bounds, current*8)
            current_bounds_ = self._update_bounds(current_bounds, current*8*cv_scale_factor)
            current_bounds[0] = current_bounds_[0]
            axes2[i].plot(RHE(pot,pH=ph),current*8*cv_scale_factor,label='seq{}_pH {}'.format(self.info['sequence_id'][i],ph),color = color)
            axes2[i].plot(RHE(pot_origin,pH=ph),current_origin*8,label='',color = color)
            # axes2[i].text(1.4,2.2,'x{}'.format(cv_scale_factor),color=color)
            # axes2[i].plot(RHE(pot,pH=ph),current*8,label='',color = color)

            # axes2[i].legend()
            # axes2[i].set_title('seq{}_pH {}'.format(self.info['sequence_id'][i],ph),fontsize=9)
            '''
            if i!=0:
                axes1[i].plot(RHE(pot,pH=ph),current*8*cv_scale_factor,label='seq{}_pH {}'.format(self.info['sequence_id'][i],ph),color = color)
                axes1[i].plot(RHE(pot,pH=ph),current*8,label='',color = color)
                axes1[i].text(1.1,2,'x{}'.format(cv_scale_factor),color=color)
                axes1[i].legend()
            '''
            '''
            if len(axs)==0:
                if i in [0,4]:
                    axes2[i].set_ylabel(r'j / mAcm$^{-2}$')
                    # axes2[i].set_xlabel(r'E / V$_{RHE}$')
                if i in [4,5,6]:
                    axes2[i].set_xlabel(r'E / V$_{RHE}$')
            else:
                axes2[i].set_ylabel(r'j / mAcm$^{-2}$')
                if i == 6:
                    axes2[i].set_xlabel(r'E / V$_{RHE}$')
                #axes2[i].set_yticklabels([])
            '''
            '''
            if i == len(self.info['sequence_id'])-1:
                axes1[i].set_xlabel(r'E / V$_{RHE}$')
            else:
                if i!=0:
                    axes1[i].set_ylabel(r'j / mAcm$^{-2}$')
                else:
                    pass
            '''
        for i in range(len(self.info['sequence_id'])):
            #axes1[i].set_xlim(*pot_bounds)
            #axes1[i].set_ylim(*current_bounds)
            axes2[i].set_xlim(*pot_bounds)
            axes2[i].set_ylim(-1.5, 8.)
        # plt.tight_layout()
        if fig2!=None:
            fig2.tight_layout()
        # print(self.data_summary)
        # fig2.subplots_adjust(wspace=0.04,hspace=0.04)
        plt.show()

    def plot_cv_files_selected_scans(self,axs = [],scans = []):
        assert len(axs)==len(scans),'The length of axs and scans must match!'
        if len(self.cv_info)==0:
            self.extract_cv_info()
        pot_bounds = [1000, -1000]
        current_bounds = [1000, -1000]
        axes2 = axs
        fig2 = None
        for scan in scans:
            i = self.info['sequence_id'].index(scan)
            pot_origin, current_origin = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            iR = self.info['resistance'][i]*np.array(current_origin)*0.001
            #iR correction
            pot_origin -= iR 
            ph = self.info['ph'][i]
            color = self.info['color'][i]
            cv_spike_cut = self.info['cv_spike_cut'][i]
            cv_scale_factor = self.info['cv_scale_factor'][i]
            pot, current = self.filter_current(pot_origin, current_origin, self.info['current_filter_length'],self.info['current_filter_order'])
            pot_bounds = self._update_bounds(pot_bounds, RHE(pot_origin,pH=ph))
            current_bounds = self._update_bounds(current_bounds, current*8)
            current_bounds_ = self._update_bounds(current_bounds, current*8*cv_scale_factor)
            current_bounds[0] = current_bounds_[0]
            axes2[scans.index(scan)].plot(RHE(pot,pH=ph),current*8*cv_scale_factor,label='seq{}_pH {}'.format(self.info['sequence_id'][i],ph),color = color)
            axes2[scans.index(scan)].plot(RHE(pot_origin,pH=ph),current_origin*8,ls = ':',label='',color = color)
            #remove axis tick lable

    #plot tafel slope for one scan
    def plot_tafel_from_formatted_cv_info_one_scan_2(self,scan, ax, forward_cycle = True, use_marker = True):
        #half = 0, first half cycle E scan from low to high values
        #how many points to be extended beyond the Tafel E range
        return_text = [str(scan)]
        offset = 0
        which = self.info['sequence_id'].index(scan)
        resistance = [self.info['resistance'][which]]
        pot_start = self.info['pot_starts_tafel'][which]
        pot_end = self.info['pot_ends_tafel'][which]
        potential_for_reaction_order = self.info['potential_reaction_order']
        if len(self.cv_info)==0:
            self.extract_cv_info()
        cv_info = self.cv_info
        if forward_cycle:
            half = 0
        else:
            half = 1
        ax.set_yscale('log')
        ax.set_xlabel(r'E / V$_{RHE}$',fontsize = int(self.info['fontsize_axis_label']))
        # ax.set_ylabel(f"pH {cv_info[scan]['pH']}")
        ax.set_ylabel(r'j / mAcm$^{-2}$',fontsize = int(self.info['fontsize_axis_label']))
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        over_E = round(potential_for_reaction_order-1.23,2)
        
        # ax2 = fig.add_subplot(212)
        #labels = ['pH {}'.format(ph) for ph in phs]
        scans = [scan]
        log_current_density = []
        pHs = []
        min_x, max_x, min_y, max_y= 0, 0, 0, 0
        pH13_count = 1
        for i in range(len(scans)):
            #ph = phs[i]
            pHs.append(cv_info[scans[i]]['pH'])
            if cv_info[scans[i]]['pH']==13 and self.pH13_count==1:
                label = 'pH {}'.format(cv_info[scans[i]]['pH'])
            elif cv_info[scans[i]]['pH']==13 and self.pH13_count!=1:
                label = None
            else:
                label = 'pH {}'.format(cv_info[scans[i]]['pH'])
            color = cv_info[scans[i]]['color']
            #pot_start=pot_starts[i]
            #pot_end=pot_ends[i]
            return_text.append('{}: resistance ={};pot_range between {} and {}'.format(label,resistance[i],pot_start, pot_end))
            print(return_text[-1])
            current = signal.savgol_filter(cv_info[scans[i]]['current_density'],21,0)
            pot = cv_info[scans[i]]['potential']
            # ax2.plot(pot)
            if half==1:
                pot_fit = pot[0:int(len(pot)/2)]
                current_fit = current[0:int(len(pot)/2)]
            else:
                pot_fit = pot[int(len(pot)/2):len(pot)][::-1]
                current_fit = current[int(len(pot)/2):len(pot)][::-1]
            indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_start)),np.argmin(abs(np.array(pot_fit)-pot_end))]
            ax.plot(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),current_fit[indx1:indx2],color = color)
            if use_marker:
                if cv_info[scans[i]]['pH']==13:
                    if self.pH13_count==1:
                        ax.text(pot_fit[indx2]-resistance[i]*(current_fit[indx2]/8*0.001),current_fit[indx2]+0.4, 'pH 13 ({})'.format(self.pH13_count),fontsize = int(self.info['fontsize_text_marker']), color = color,rotation =30)
                    elif self.pH13_count==3:
                        ax.text(pot_fit[indx2]-resistance[i]*(current_fit[indx2]/8*0.001),current_fit[indx2]-0.2, 'pH 13 ({})'.format(self.pH13_count),fontsize = int(self.info['fontsize_text_marker']), color = color,rotation=30)
                    else:
                        ax.text(pot_fit[indx2]-resistance[i]*(current_fit[indx2]/8*0.001),current_fit[indx2], 'pH 13 ({})'.format(self.pH13_count),fontsize = int(self.info['fontsize_text_marker']), color = color,rotation=30)
                    self.pH13_count+=1
                elif cv_info[scans[i]]['pH']==10:
                    ax.text(pot_fit[indx2]-resistance[i]*(current_fit[indx2]/8*0.001),current_fit[indx2]-0.2, 'pH '+str(cv_info[scans[i]]['pH']), ha = 'left',fontsize=int(self.info['fontsize_text_marker']), color = color,rotation = 0)
                elif cv_info[scans[i]]['pH']==7:
                    ax.text(pot_fit[indx2]-resistance[i]*(current_fit[indx2]/8*0.001),current_fit[indx2]+0.2, 'pH '+str(cv_info[scans[i]]['pH']), ha = 'right',fontsize=int(self.info['fontsize_text_marker']), color = color,rotation = 0)
                else:
                    ax.text(pot_fit[indx2]-resistance[i]*(current_fit[indx2]/8*0.001),current_fit[indx2], 'pH '+str(cv_info[scans[i]]['pH']), ha = 'right',fontsize=int(self.info['fontsize_text_marker']), color = color)
            # ax.plot(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),current_fit[indx1:indx2],label=label,color = color)
            # ax.plot(pot_fit[indx1-offset:indx1]-resistance[i]*(current_fit[indx1-offset:indx1]/8*0.001),current_fit[indx1-offset:indx1],':',label=label,color = color)
            min_x, max_x = min(pot_fit[indx1-offset:indx2]-resistance[i]*(current_fit[indx1-offset:indx2]/8*0.001)),max(pot_fit[indx1-offset:indx2]-resistance[i]*(current_fit[indx1-offset:indx2]/8*0.001))
            min_y, max_y = min(current_fit[indx1-offset:indx2]),max(current_fit[indx1-offset:indx2])
            # ax.legend()
            #linear regression
            try:
                slope,intercept,r_value, *others =stats.linregress(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),np.log10(current_fit[indx1:indx2]))
                return_text.append('Linear fit results: log(current) = {} E + {}, R2 = {}'.format(slope, intercept, r_value**2))
                print(return_text[-1])
                return_text.append('Tafel slope = {} mV/decade'.format(1/slope*1000))
                print(return_text[-1])
                # ax.text(min_x,min_y+0.1,'b = {} mV/decade'.format(int(round(1/slope*1000,0))),color='k')
                # ax.text(1.65,0.1,'b = {} mV/decade'.format(int(round(1/slope*1000,0))),color='k')
                log_current_density.append(potential_for_reaction_order*slope+intercept)
            except:
                pass
        # ax.legend()
        return min_x, max_x, min_y, max_y, return_text

    #plot tafel slope for one scan
    def plot_tafel_from_formatted_cv_info_one_scan(self,scan, ax, forward_cycle = True):
        #half = 0, first half cycle E scan from low to high values
        #how many points to be extended beyond the Tafel E range
        offset = 0
        which = self.info['sequence_id'].index(scan)
        resistance = [self.info['resistance'][which]]
        pot_start = self.info['pot_starts_tafel'][which]
        pot_end = self.info['pot_ends_tafel'][which]
        potential_for_reaction_order = self.info['potential_reaction_order']
        if len(self.cv_info)==0:
            self.extract_cv_info()
        cv_info = self.cv_info
        if forward_cycle:
            half = 0
        else:
            half = 1

        ax.set_yscale('log')
        ax.set_xlabel(r'E / V$_{RHE}$')
        ax.set_ylabel(f"pH {cv_info[scan]['pH']}")
        # ax.set_ylabel(r'j / mAcm$^{-2}$')
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        over_E = round(potential_for_reaction_order-1.23,2)
        
        # ax2 = fig.add_subplot(212)
        #labels = ['pH {}'.format(ph) for ph in phs]
        scans = [scan]
        log_current_density = []
        pHs = []
        min_x, max_x, min_y, max_y= 0, 0, 0, 0
        for i in range(len(scans)):
            #ph = phs[i]
            pHs.append(cv_info[scans[i]]['pH'])
            label = 'scan {}_pH {}'.format(scans[i],cv_info[scans[i]]['pH'])
            color = cv_info[scans[i]]['color']
            #pot_start=pot_starts[i]
            #pot_end=pot_ends[i]
            print(label,': resistance ={};pot_range between {} and {}'.format(resistance[i],pot_start, pot_end))
            current = signal.savgol_filter(cv_info[scans[i]]['current_density'],21,0)
            pot = cv_info[scans[i]]['potential']
            # ax2.plot(pot)
            if half==1:
                pot_fit = pot[0:int(len(pot)/2)]
                current_fit = current[0:int(len(pot)/2)]
            else:
                pot_fit = pot[int(len(pot)/2):len(pot)][::-1]
                current_fit = current[int(len(pot)/2):len(pot)][::-1]
            indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_start)),np.argmin(abs(np.array(pot_fit)-pot_end))]
            ax.plot(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),current_fit[indx1:indx2],color = color)
            # ax.plot(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),current_fit[indx1:indx2],label=label,color = color)
            ax.plot(pot_fit[indx1-offset:indx1]-resistance[i]*(current_fit[indx1-offset:indx1]/8*0.001),current_fit[indx1-offset:indx1],':',color = color)
            min_x, max_x = min(pot_fit[indx1-offset:indx2]-resistance[i]*(current_fit[indx1-offset:indx2]/8*0.001)),max(pot_fit[indx1-offset:indx2]-resistance[i]*(current_fit[indx1-offset:indx2]/8*0.001))
            min_y, max_y = min(current_fit[indx1-offset:indx2]),max(current_fit[indx1-offset:indx2])
            # ax.legend()
            #linear regression
            try:
                slope,intercept,r_value, *others =stats.linregress(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),np.log10(current_fit[indx1:indx2]))
                print('Linear fit results: log(current) = {} E + {}, R2 = {}'.format(slope, intercept, r_value**2))
                print('Tafel slope = {} mV/decade'.format(1/slope*1000))
                # ax.text(min_x,min_y+0.1,'b = {} mV/decade'.format(int(round(1/slope*1000,0))),color='k')
                ax.text(1.65,0.1,'b = {} mV/decade'.format(int(round(1/slope*1000,0))),color='k')
                log_current_density.append(potential_for_reaction_order*slope+intercept)
            except:
                pass
        return min_x, max_x, min_y, max_y
    
    def plot_tafel_with_reaction_order(self,ax_tafel, ax_order,constant_value = 1,mode = 'constant_current', forward_cycle = True, use_marker = True, use_all = True):
        self.plot_reaction_order_with_pH(constant_value = constant_value, ax = ax_order, mode = mode, forward_cycle = forward_cycle, use_marker = use_marker, use_all = use_all)
        self.pH13_count = 1
        text_log = {}
        if use_all:
            all_scans = self.info['sequence_id']
        else:
            all_scans = self.info['selected_scan']
        for scan in all_scans:
            *dump, return_text = self.plot_tafel_from_formatted_cv_info_one_scan_2(scan=scan, ax=ax_tafel, forward_cycle = forward_cycle, use_marker = use_marker)
            text_log[scan] = '\n'.join(return_text)
        return text_log

    #plot reaction order with pH
    #two modes: 
    # constant_current: current density at the same potential
    # constant_potential: potential at the same current density
    def plot_reaction_order_with_pH(self, constant_value = 1, ax = None, mode = 'constant_current', forward_cycle = True, use_marker = True, use_all = True):
        if forward_cycle:
            half = 0
        else:
            half = 1
        if ax == None:
            fig = plt.figure(figsize=(7,4))
            ax = fig.add_subplot(111)
        else:
            pass
        ax.set_xlabel(r'pH',fontsize = int(self.info['fontsize_axis_label']))
        pHs = []
        values = [] #either pot values or current density values depending on the mode 
        colors = []
        mode = self.info['reaction_order_mode']
        if mode == 'constant_current':
            constant_value = self.info['current_reaction_order']
        elif mode == 'constant_potential':
            constant_value = self.info['potential_reaction_order']
        def _get_pot_current(scan, resistance, half):
            current = self.cv_info[scan]['current_density']
            #ir corrected potential
            pot = self.cv_info[scan]['potential'] - self.cv_info[scan]['current_density']/8*0.001*resistance
            if half==1:
                pot_fit = pot[0:int(len(pot)/2)]
                current_fit = current[0:int(len(pot)/2)]
            else:
                pot_fit = pot[int(len(pot)/2):len(pot)][::-1]
                current_fit = current[int(len(pot)/2):len(pot)][::-1]
            return pot_fit, current_fit

        if mode == 'constant_potential':
            #potential_for_reaction_order = self.info['potential_reaction_order']
            over_E = round(constant_value-1.23,2)
            ax.set_ylabel(r'log(j / mAcm$^{-2}$)'+r',$\eta$= {}V'.format(over_E),fontsize = int(self.info['fontsize_axis_label']))
        elif mode == 'constant_current':
            ax.set_ylabel(r'E / V$_{RHE}$' + f'at j = {constant_value}'+r'mAcm$^{-2}$',fontsize = int(self.info['fontsize_axis_label']))
        for i, scan in enumerate(self.info['sequence_id']):
            if not use_all:
                if scan not in self.info['selected_scan']:
                    continue
            pot_fit, current_fit = _get_pot_current(scan, self.info['resistance'][i], half)
            pHs.append(self.cv_info[scan]['pH'])
            colors.append(self.cv_info[scan]['color'])
            if mode == 'constant_potential':
                pot_start = self.info['pot_starts_tafel'][i]
                pot_end = self.info['pot_ends_tafel'][i]
                potential_for_reaction_order = constant_value
                indx1,indx2 = [np.argmin(abs(np.array(pot_fit)-pot_start)),np.argmin(abs(np.array(pot_fit)-pot_end))]
                #linear regression
                slope,intercept,r_value, *others =stats.linregress(pot_fit[indx1:indx2],np.log10(current_fit[indx1:indx2]))
                print('Linear fit results: log(current) = {} E + {}, R2 = {}'.format(slope, intercept, r_value**2))
                #ax.text(1.65,.1,'b = {} mV/decade'.format(int(round(1/slope*1000,0))),color='k')
                values.append(potential_for_reaction_order*slope+intercept)
            elif mode == 'constant_current':
                which = np.argmin(np.abs(current_fit - constant_value))
                values.append(pot_fit[which])
        pHs_unique = list(set(pHs))
        values_unique = [values[pHs.index(each)] for each in pHs_unique]
        # pHs, values = pHs_unique, values_unique
        pH13_count = 1
        for j,pH in enumerate(pHs):
            # ax.plot(pH, values[j], 'o', color = self.info['color'][j])
            ax.plot(pH, values[j], 'o', color = colors[j])
            if pH ==13:
                if use_marker:
                    ax.text(pH+0.1,values[j],str(pH13_count),fontsize=int(self.info['fontsize_text_marker']))
                pH13_count+=1
        slope_, intercept_, r_value_, *_ = stats.linregress(pHs, values)
        f = lambda x: slope_*x + intercept_
        x_min, x_max = min(pHs), max(pHs)
        ax.plot([x_min,x_max],[f(x_min),f(x_max)],'-k')
        text_label = 'y = {}x + {}, R2 = {}'.format(round(slope_,3), round(intercept_,3), round(r_value_**2,3))
        # ax.text(x_min,f(x_max),text_label)
        print('Reaction order fit: log(current) = {}pH + {}, R2 = {}'.format(slope_, intercept_, r_value_**2))
        #plt.legend()
        try:
            fig.tight_layout()
        except:
            pass
        plt.show()

    #plot tafel slopes for all in one panel
    def plot_tafel_from_formatted_cv_info(self,forward_cycle = True):
        #half = 0, first half cycle E scan from low to high values
        resistance = self.info['resistance']
        pot_starts = self.info['pot_starts_tafel']
        pot_ends = self.info['pot_ends_tafel']
        potential_for_reaction_order = self.info['potential_reaction_order']
        if len(self.cv_info)==0:
            self.extract_cv_info()
        cv_info = self.cv_info
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
            ax.plot(pot_fit[indx1-50:indx1]-resistance[i]*(current_fit[indx1-50:indx1]/8*0.001),current_fit[indx1-50:indx1],':',color = color)

            ax.legend()
            #linear regression
            slope,intercept,r_value, *others =stats.linregress(pot_fit[indx1:indx2]-resistance[i]*(current_fit[indx1:indx2]/8*0.001),np.log10(current_fit[indx1:indx2]))
            print('Linear fit results: log(current) = {} E + {}, R2 = {}'.format(slope, intercept, r_value**2))
            print('Tafel slope = {} mV/decade'.format(1/slope*1000))
            log_current_density.append(potential_for_reaction_order*slope+intercept)
        #pHs_ = sorted(list(set(pHs)))
        #log_current_density_ = [log_current_density[pHs.index(each)] for each in pHs_]
        # print(pHs)
        pHs_, log_current_density_ = [], []
        for i in range(len(pHs)):
            if pHs[i] not in pHs_:
                pHs_.append(pHs[i])
                log_current_density_.append(log_current_density[i])

        ax2.plot(pHs_, log_current_density_, 'og')
        slope_, intercept_, r_value_, *_ = stats.linregress(pHs_, log_current_density_)
        f = lambda x: slope_*x + intercept_
        x_min, x_max = min(pHs), max(pHs)
        ax2.plot([x_min,x_max],[f(x_min),f(x_max)],'-r')
        text_label = 'y = {}x{}, R2 = {}'.format(round(slope_,3), round(intercept_,3), round(r_value_**2,3))
        ax2.text(x_min,f(x_min),text_label)
        print('Reaction order fit: log(current) = {}pH + {}, R2 = {}'.format(slope_, intercept_, r_value_**2))
        #plt.legend()
        fig.tight_layout()
        plt.show()
        #fig.savefig('Tafel.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    cv = cvAnalysis(config_file = '/Users/canrong/apps/DaFy/util/cv_config_I20180835.ini')
    # cv = cvAnalysis(config_file = '/Users/canrong/apps/DaFy/util/cv_config.ini')
    cv.calc_charge_all()
    cv.plot_cv_files()
    cv.plot_tafel_from_formatted_cv_info()