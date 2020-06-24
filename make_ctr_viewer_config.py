import numpy as np 
import os

#the data file name in excel format
data_file_name = ''
first_scan_no = [9774,9797]
labels = ['-1.3 V', '-1 V']

scan_offsets = [[0,1,2,3,4],[5,6,7,8,9,10],[11,12,13,14,15,16],[17,18,19,20,21]]
config_file_template = 'D:\\DaFy\\projects\\viewer\\plot_viewer_config_standard.ini'
data_file_path = 'D://DaFy//dump_files'
file_name = os.path.join(data_file_path, data_file_name)

def stack_labels(labels,num_stacks = 1):
    all_labels =[]
    for i in range(num_stacks):
        current_lables = '[{}]'.format(';'.join(labels))
        all_labels.append(current_labels)
    return '+'.join(all_labels)

def stack_scans(scans, num_stacks = 1):
    all_scans = []
    for i in range(num_stacks):
        for each in first_scan_no:
            current_set = []
            

with open(config_file_template.replace('.ini','_new.ini'),'w') as f_w:
    with open(config_file_template,'r') as f_r:
        lines = f_r.readlines()
        lines_to_be_writein = []
        for each_line in lines:
            if each_line.startswith('lineEdit_data_file'):
                lines_to_be_writein.append(['lineEdit_data_file:{}\n'.format(file_name))
            if each_line.startswith('lineEdit_labels'):
                lines_to_be_writein.append(['lineEdit_labels:{}\n'.format(stack_labels(labels,len(first_scan_no))))
            if each_line.startswith('scan_numbers_append:'):
                lines_to_be_writein.append(['scan_numbers_append:{}\n'.format(stack_scans(labels,len(first_scan_no))))


