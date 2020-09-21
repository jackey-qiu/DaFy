import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import pandas as pd

class plotCTR(object):
    def __init__(self, data_folder, data_files, colors=['b','g','m','y','k','b','g','m','y','k'],label='P11_'):
        self.data_folder = data_folder
        self.data_files = []
        if len(data_files)==0:
            for file in os.listdir(data_folder):
                if file.endswith('.xlsx'):
                    self.data_files.append(file)
        else:
            self.data_files = data_files
        self.label = label
        self.data = None
        self.hk_list = []
        self.potential_list = []
        self.colors = colors

    def action(self):
        self.extract_data()
        self.get_HK_potential_set()
        self.compute_plot_dimention()
        self.stack_ctr_profiles()

    def extract_data(self):
        for each in self.data_files:
            file = os.path.join(self.data_folder, each)
            data_temp = pd.read_excel(file)
            if self.data_files.index(each)!=0:
                self.data = pd.concat([self.data,data_temp])
            else:
                self.data = data_temp

    def get_HK_potential_set(self):
        self.hk_list = []
        self.potential_list = []
        self.potential_list = sorted(list(set(self.data['potential'])))[::-1]
        self.hk_list = list(set(zip(self.data['H'],self.data['K'])))
        # print(self.hk_list)

    def compute_plot_dimention(self):
        fig, axes = plt.subplots(int(len(self.hk_list)/2)+len(self.hk_list)%2,2,figsize=(10,9))
        self.fig = fig
        self.axes = axes.flatten()

    def stack_ctr_profiles(self):
        for hk in self.hk_list:
            h,k = hk
            ax = self.axes[self.hk_list.index(hk)]
            ax.set_yscale('log')
            if (self.hk_list.index(hk)+1)%2==1:
                ax.set_ylabel('F')
            if hk in self.hk_list[-2:]:
                ax.set_xlabel('L')
            ax.set_title('{}{}L'.format(h,k),fontsize=10)
            L_list = []
            I_list = []
            I_model_list = []
            I_ideal_list = []
            I_error_list = []
            Label_list = []
            for potential in self.potential_list:
                conditions = (self.data["H"] == h) & (self.data["K"] == k) & (self.data["potential"] == potential) & (self.data["use"] == 1)
                Label_list.append('{}:{} V'.format(self.label, potential))
                L_list.append(self.data[conditions]['L'])
                I_list.append(self.data[conditions]['I'])
                I_model_list.append(self.data[conditions]['I_model'])
                I_ideal_list.append(self.data[conditions]['I_bulk'])
                I_error_list.append(self.data[conditions]['error'])
            max_L = 0
            text_y_coordinate_list = []
            text_list = []
            for i in range(len(L_list)):
                ax.errorbar(L_list[i], I_list[i]*10**(i+1), yerr = I_error_list[i],fmt = 'o',color=self.colors[i], markersize = 3, label = Label_list[i])
                ax.plot(L_list[i], I_model_list[i]*10**(i+1),color='r',lw=2)
                ax.plot(L_list[i], I_ideal_list[i]*10**(i+1),':',color = 'k')
                #if max(L_list[i])>max_L
                #if hk==self.hk_list[0]:
                #    ax.legend()
                ax.text(max(L_list[i])+0.1,list(I_model_list[i])[-1]*10**(i+1),Label_list[i])
                ax.set_xlim(right = max(L_list[i])+1)
        self.fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    # ctr = plotCTR(data_folder='/Users/canrong/Documents/CO2RR/ctr_model/bestfit', data_files=['best_model_P11_m1p3V_OCCO_run3C.xlsx','best_model_P11_m0p7V_CO_run9_B.xlsx','best_model_P11_m0p6V_CO3_newrun1.xlsx'])
    ctr = plotCTR(data_folder='/Users/canrong/Documents/CO2RR/ctr_model/bestfit', data_files=[])
    ctr.action()
            
