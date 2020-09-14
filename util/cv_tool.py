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

def extract_vars_from_config(config_file, section_var):
    config = configparser.ConfigParser()
    config.read(config_file)
    kwarg = {}
    for each in config.items(section_var):
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

    def _extract_parameter_from_config(self,config_file, sections = ['Global']):
        #in Global section you should have the following items (all in list type)
        #ph = [], color =, fmt = , path = , method =, pot_range = , sequence_id =, cv_scale_factor =, cv_spike_cut=, cv_scan_rate=
        for section in sections:
            kwarg_temp = extract_vars_from_config(config_file, section_var = section)
            for each in kwarg_temp:
                print('Extracting {} now'.format(each))
                if each == 'cv_folder':
                    self.info[each] = kwarg_temp[each]
                else:
                    self.info[each] = kwarg_temp[each]

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
            t, pot, current = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            ph = self.info['ph'][i]
            color = self.info['color'][i]
            self.cv_info[self.info['sequence_id'][i]] = {'current_density':current*8, 'potential':RHE(pot,pH=ph), 'pH':ph, 'color':color}

    def calc_charge_all(self):
        self.info['charge'] = []
        for i in range(len(self.info['sequence_id'])):
            t, pot, current = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            ph = self.info['ph'][i]
            cv_spike_cut = self.info['cv_spike_cut'][i]
            cv_scale_factor = self.info['cv_scale_factor'][i]
            scan_rate = self.info['scan_rate'][i]
            pot_range = self.info['pot_range'][i]
            print('Processing sequence {} now ... '.format(self.info['sequence_id'][i]))
            charge = self.calculate_pseudocap_charge(t, pot, current, ph, cv_spike_cut, cv_scale_factor, scan_rate, pot_range)
            self.info['charge'].append(charge)

    #filter out the spikes on CV due to beam shutter on/off
    def filter_current(self, t, pot, current, cv_spike_cut, times = 4):
        t_filtered, pot_filtered, current_filtered = t, pot, current
        for ii in range(times):
            filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
            filter_index = filter_index+1#index offset by 1
            t_filtered = t_filtered[(filter_index,)]
            pot_filtered = pot_filtered[(filter_index,)]
            current_filtered = current_filtered[(filter_index,)]
        return t_filtered, pot_filtered, current_filtered

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

    def calculate_pseudocap_charge(self, t, pot, current, ph=10, cv_spike_cut=0.002, cv_scale_factor=30, scan_rate = 0.005, pot_range = [0.98,1.6]):
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
        # plt.plot(pot_filtered)
        # return
        index_max = np.argmax(current_smooth)
        index_top_left = np.argmin(np.abs(pot_filtered[0:index_max]-pot_left))
        index_top_right = np.argmin(np.abs(pot_filtered[0:index_max]-pot_right))
        index_bottom_right = np.argmin(np.abs(pot_filtered[index_max:]-pot_left))+index_max
        index_bottom_left = np.argmin(np.abs(pot_filtered[index_max:]-pot_right))+index_max
        charge_top = metrics.auc(pot_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
        charge_bottom = metrics.auc(pot_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
        charge_top_2 = compute_area_under_a_curve(t_filtered[index_top_left:index_top_right],current_smooth[index_top_left:index_top_right])
        charge_bottom_2 = compute_area_under_a_curve(t_filtered[index_bottom_left:index_bottom_right],current_smooth[index_bottom_left:index_bottom_right])
        # self.locate_OER_onset(current_filtered[index_top_left:index_top_right],pot_filtered[index_top_left:index_top_right])
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
    def extract_cv_file_ivium(self,file_path,which_cycle=3):
        data = []
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
    def extract_cv_file_fouad(self,file_path='/home/qiu/apps/048_S221_CV', which_cycle=1):
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

    #only one cycle, which can be manually exported from BioLogic software
    def extract_cv_file_biologic(self, file_path ='', which_cycle =1):
        #return: time(s), potential(V), current(mA)
        #skip three rows for header info
        data = np.loadtxt(file_path,skiprows = 3)
        pot, current = self.format_pot_current(data[:,1], data[:,2])
        # return data[:,0], data[:,1], data[:,2]
        return data[:,0], pot, current

    def _update_bounds(self, current_bounds, data):
        min_, max_ = min(data), max(data)
        return [min([min_,current_bounds[0]]), max([max_,current_bounds[1]])]

    def plot_cv_files(self):
        pot_bounds = [1000, -1000]
        current_bounds = [1000, -1000]
        # fig, axes1= plt.subplots(len(self.info['sequence_id']), 1)
        fig2, axes2= plt.subplots(2, int(len(self.info['sequence_id'])/2+len(self.info['sequence_id'])%2))
        axes2 = axes2.flatten()
        for i in range(len(self.info['sequence_id'])):
            t, pot_origin, current_origin = getattr(self,self.info['method'][i])(file_path = os.path.join(self.info['cv_folder'],self.info['path'][i]), which_cycle = self.info['which_cycle'][i])
            ph = self.info['ph'][i]
            color = self.info['color'][i]
            cv_spike_cut = self.info['cv_spike_cut'][i]
            cv_scale_factor = self.info['cv_scale_factor'][i]
            t, pot, current = self.filter_current(t, pot_origin, current_origin, cv_spike_cut)
            pot_bounds = self._update_bounds(pot_bounds, RHE(pot_origin,pH=ph))
            current_bounds = self._update_bounds(current_bounds, current*8)
            current_bounds_ = self._update_bounds(current_bounds, current*8*cv_scale_factor)
            current_bounds[0] = current_bounds_[0]
            axes2[i].plot(RHE(pot,pH=ph),current*8*cv_scale_factor,label='seq{}_pH {}'.format(self.info['sequence_id'][i],ph),color = color)
            axes2[i].plot(RHE(pot_origin,pH=ph),current_origin*8,label='',color = color)
            axes2[i].text(1.1,1,'x{}'.format(cv_scale_factor),color=color)
            # axes2[i].legend()
            axes2[i].set_title('seq{}_pH {}'.format(self.info['sequence_id'][i],ph),fontsize=9)
            '''
            if i!=0:
                axes1[i].plot(RHE(pot,pH=ph),current*8*cv_scale_factor,label='seq{}_pH {}'.format(self.info['sequence_id'][i],ph),color = color)
                axes1[i].plot(RHE(pot,pH=ph),current*8,label='',color = color)
                axes1[i].text(1.1,2,'x{}'.format(cv_scale_factor),color=color)
                axes1[i].legend()
            '''
            if i in [0,3]:
                axes2[i].set_ylabel(r'j / mAcm$^{-2}$')
                axes2[i].set_xlabel(r'E / V$_{RHE}$')
            else:
                axes2[i].set_xlabel(r'E / V$_{RHE}$')
                axes2[i].set_yticklabels([])
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
            axes2[i].set_ylim(-1, 4.)
        # plt.tight_layout()
        fig2.tight_layout()
        # print(self.data_summary)
        # fig2.subplots_adjust(wspace=0.04,hspace=0.04)
        plt.show()

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
        ax2.plot(pHs, log_current_density, 'og')
        slope_, intercept_, r_value_, *_ = stats.linregress(pHs, log_current_density)
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
    cv.calc_charge_all()
    cv.plot_cv_files()
    cv.plot_tafel_from_formatted_cv_info()