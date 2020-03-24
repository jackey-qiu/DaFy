import numpy as np
from PyMca5.PyMcaIO import specfilewrapper
'''
This class is used to extract information from Fio file, one of the common output file at DESY beamline.
At this moment one scan has a seperated fio file. This class is not designed to work on multiple scans within
one fio file.

1.You may need to extend the field_tag_lib to include other fields. 
2.Note that the first item in header_info is the macro ran to do the scan
3.The lable of data column comes from the third item in the lines starting with 'Col'
4.In principle each parameter has a single value, but when calling get_parameter it returns a list of this value of
  the same size as the data column.

'''
class FioFile(object):
    def __init__(self,fio_path='/home/qiu/data/beamtime/P23_11_18_I20180114/raw/startup/FirstTest_00666.fio',field_tag_lib={'comments':'%c','parameters':'%p','data':'%d'}):
        self.fio_path=fio_path
        self.field_tag_lib=field_tag_lib
        self.parameter_info={}
        self.data_column_size=0
        self.data_info={'col_lable':[],'data_value':[]}
        self.header_info=[]
        self.extract_info()

    def extract_info(self):
        with open(self.fio_path) as f:
            current_tag=None
            for line in f:
                if not line.startswith('!'):
                    if line[0:2] in self.field_tag_lib.values():
                        current_tag=line.rstrip()
                    else:
                        if current_tag==self.field_tag_lib['comments']:
                           self.header_info.append(line.strip())
                        elif current_tag==self.field_tag_lib['parameters']:
                            key_temp,value_temp=line.strip().replace(' ','').rsplit('=')
                            self.parameter_info[key_temp]=float(value_temp)
                        elif current_tag==self.field_tag_lib['data']:
                            if line.strip().rsplit(' ')[0]=='Col':
                                self.data_info['col_lable'].append(line.strip().rsplit(' ')[2])
                            else:
                                temp=[float(value) for value in line.strip().rsplit(' ')]
                                self.data_info['data_value'].append(temp)
                        else:
                            pass
        self.data_info['data_value']=np.array(self.data_info['data_value'])
        self.data_column_size=self.data_info['data_value'].shape[0]

        return None
                            
    def get_datacol(self,col_label):
        return self.data_info['data_value'][:,self.data_info['col_lable'].index(col_label)]

    def get_header(self,which_line=0):
        return self.header_info[which_line].strip()

    def get_parameter(self,par_lable):
        return np.array([self.parameter_info[par_lable]]*self.data_column_size)

    def get_col(self,label):
        try:
            return self.get_datacol(label)
        except:
            return self.get_parameter(label)

    def extract_motor_angle(self,original_motor_angles,scan_number,frame_number,updated_angles):
        for angle in updated_angles:
            original_motor_angles[angle] = self.get_col(angle)[frame_number]
        return original_motor_angles

    def extract_pot_current(self,scan_number,frame_number,labels = ['p23/pilcvfcadc/dev.01/value','p23/pilcvfcadc/dev.02/value']):
        results = []
        for label in labels:
            results.append(self.get_col(label)[frame_number])
        return results

    def extract_pot_profile(self,scan_number,label = 'p23/pilcvfcadc/dev.01/value'):
        results = self.get_col(label)
        return results

class SpecFile(object):
    def __init__(self, spec_file='/home/qiu/data/ma4171/sixcvertical.spec'):
        self.spec_file = spec_file
        self.scan_selector = specfilewrapper.Specfile(spec_file)
        self.motor_angles = {}
        self.trans = 1
        self.mon = 1
        self.potential =None
        self.current = None
        self.scan_no = None
        self.frame_no = None

    def extract_pot_current(self, scan_no, frame_no, is_zap_scan = False):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        if is_zap_scan:
            potential = scan.datacol('zap_Eana')[frame_no]
            current = scan.datacol('zap_Iana')[frame_no]
        else:
            try:
                potential = scan.datacol('Eana')[frame_no]
                current = scan.datacol('Iana')[frame_no]
            except:
                potential =0 
                current = 0
        self.potential, self.current = potential, current
        self.scan_no, self.frame_no = scan_no, frame_no
        return potential, current

    def extract_L(self, scan_no, frame_no, is_zap_scan = False):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        if is_zap_scan:
            L = scan.datacol('zap_Lcnt')[frame_no]
        else:
            try:
                L = scan.datacol('L')[frame_no]
            except:
                L = 0
        
        return L 

    def extract_mon(self, scan_no, frame_no, is_zap_scan = False):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        if is_zap_scan:
            mon = scan.datacol('zap_mon')[frame_no]
        else:
            try:
                mon = scan.datacol('mon')[frame_no]
            except:
                mon = 1
        
        return mon 

    def extract_trans(self, scan_no, frame_no, is_zap_scan = False):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        if is_zap_scan:
            trans = scan.datacol('zap_transm')[frame_no]
        else:
            try:
                trans = scan.datacol('transm')[frame_no]
            except:
                trans = 0
        
        return trans 

    def extract_motor_angle(self, scan_no, frame_no, is_zap_scan = False):
        self.scan_no = scan_no
        self.frame_no = frame_no
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        chi_ = scan.motorpos('Chi')
        phi_ = scan.motorpos('Phi')
        # if False:#zap scan means motor positions not change during the scan
            # th_ = scan.motorpos('Theta')
            # gam_ = scan.motorpos('Gam')
            # del_ = scan.motorpos('Delta')
            # mon_ = scan.datacol('zap_mon')[frame_no]
            # transm_ = scan.datacol('zap_transm')[frame_no]
            # if(transm_ == 0):
                # this is neccessary to avoid division by zero
                # transm_ = 1e-30
            # mu_ = scan.motorpos('Mu')
            # potential = scan.datacol('zap_Eana')[frame_no]
            # current = scan.datacol('zap_Iana')[frame_no]
        # else:
        counter_prefix = ''
        if('ccoscan' in scan.header('S')[0]):
            counter_prefix = 'zap_'
        th_ = scan.datacol('%sthcnt'%(counter_prefix))[frame_no]
        gam_ = scan.datacol('%sgamcnt'%(counter_prefix))[frame_no]
        del_ = scan.datacol('%sdelcnt'%(counter_prefix))[frame_no]
        try:
            mon_ = scan.datacol('%smon'%(counter_prefix))[frame_no]
        except:
            mon_ = 1
        transm_ = scan.datacol('%stransm'%(counter_prefix))[frame_no]
        mu_ = scan.datacol('%smucnt'%(counter_prefix))[frame_no]
        try:
            potential = scan.datacol('Eana')[frame_no]
            current = scan.datacol('Iana')[frame_no]
        except:
            potential = 0
            current = 0
        self.motor_angles = dict(zip(['phi','chi','mu','delta','gamma','omega_t','mon','transm'],[phi_, chi_, th_,gam_,del_, mu_, mon_, transm_]))
        self.trans = transm_
        self.mon = mon_
        self.potential = potential
        self.current = current
        return self.motor_angles

    def extract_header_info(self, scan_no, items={'or0_lib':['G',1, [list(range(12,15)),list(range(18,24)),[30]]],\
                                                         'or1_lib':['G',1, [list(range(15,18)),list(range(24,30)),[31]]],\
                                                         'cell':['G',1,[list(range(0,6))]],\
                                                         'n_azt':['G',0,[list(range(3,6))]],\
                                                         'lambda':['G',2,[[3]]]}):
        #note in sixc geometry motor angles are in this order: del, th, chi, phi, mu, gam
        #note in psci geometry motor angles are in this order: del, eta, chi, phi, mu, nu
        #The mapping is like this: (del,th, chi, phi, mu)_sixc = (del, eta, chi, phi, mu)_psic, but gam_sixc + mu_sixc= nu_psic         
        items_values = {}
        self.scan_no = scan_no
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        for key, item in items.items():
            indexs = sum(item[2],[])
            values = [float(scan.header(item[0])[item[1]].rsplit()[each+1]) for each in indexs]
            if key in ['or0_lib', 'or1_lib']:
                #correct the nu for psic geo by nu = gam + mu
                values[-2] = values[-2] + values[-3]
                #chi equals 90 instead of 0? double check it
                values[5] = values[5] +90
            if key in ['or0_lib','or1_lib']:
                items_values[key]={'h':values[0:3]}
                for mo, val in zip(['delta','eta','chi','phi','mu','nu','lam'],values[3:]):
                    items_values[key][mo] = val
            else:
                items_values[key] = values
        print(items_values)
        return items_values

class SpecFile_APS(object):
    def __init__(self, spec_file='/home/qiu/data/ma4171/sixcvertical.spec'):
        self.spec_file = spec_file
        self.scan_selector = specfilewrapper.Specfile(spec_file)
        self.motor_angles = {}
        self.trans = 1
        self.mon = 1
        self.potential =None
        self.current = None
        self.scan_no = None
        self.frame_no = None

    def extract_L(self, scan_no, frame_no):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        L = scan.datacol('L')[frame_no]
        return L 

    def extract_mon(self, scan_no, frame_no):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        mon = scan.datacol('io')[frame_no]
        return mon 

    def extract_trans(self, scan_no, frame_no):
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        trans = scan.datacol('transm')[frame_no]
        return trans 

    def extract_motor_angle(self, scan_no, frame_no):
        self.scan_no = scan_no
        self.frame_no = frame_no
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        chi_ = scan.datacol('chi')[frame_no]
        phi_ = scan.datacol('phi')[frame_no]
        # if False:#zap scan means motor positions not change during the scan
            # th_ = scan.motorpos('Theta')
            # gam_ = scan.motorpos('Gam')
            # del_ = scan.motorpos('Delta')
            # mon_ = scan.datacol('zap_mon')[frame_no]
            # transm_ = scan.datacol('zap_transm')[frame_no]
            # if(transm_ == 0):
                # this is neccessary to avoid division by zero
                # transm_ = 1e-30
            # mu_ = scan.motorpos('Mu')
            # potential = scan.datacol('zap_Eana')[frame_no]
            # current = scan.datacol('zap_Iana')[frame_no]
        # else:
        try:
            th_ = scan.datacol('theta')[frame_no]
        except:
            th_ = scan.datacol('eta')[frame_no]

        try:
            gam_ = scan.datacol('Nu')[frame_no]
        except:
            gam_ = scan.datacol('nu')[frame_no]

        try:
            del_ = scan.datacol('del')[frame_no]
        except:
            del_ = scan.datacol('TwoTheta')[frame_no]

        try:
            mu_ = scan.datacol('mu')[frame_no]
        except:
            mu_ = scan.datacol('Psi')[frame_no]

        mon_ = scan.datacol('io')[frame_no]
        # transm_ = scan.datacol('transm')[frame_no]
        #io is dependent on transm
        transm_ = 1
        self.motor_angles = dict(zip(['phi','chi','mu','delta','gamma','omega_t','mon','transm'],[phi_, chi_, th_,del_,gam_, mu_, mon_, transm_]))
        self.trans = transm_
        self.mon = mon_
        return self.motor_angles

    def extract_header_info(self, scan_no, items={'or0_lib':['G',1, [list(range(12,15)),list(range(18,24)),[30]]],\
                                                         'or1_lib':['G',1, [list(range(15,18)),list(range(24,30)),[31]]],\
                                                         'cell':['G',1,[list(range(0,6))]],\
                                                         'n_azt':['G',0,[list(range(3,6))]],\
                                                         'lambda':['G',2,[[3]]]}):
        #note in sixc geometry motor angles are in this order: del, th, chi, phi, mu, gam
        #note in psci geometry motor angles are in this order: del, eta, chi, phi, mu, nu
        #The mapping is like this: (del,th, chi, phi, mu)_sixc = (del, eta, chi, phi, mu)_psic, but gam_sixc + mu_sixc= nu_psic         
        items_values = {}
        self.scan_no = scan_no
        scan = self.scan_selector.select('{}.1'.format(scan_no))
        for key, item in items.items():
            indexs = sum(item[2],[])
            values = [float(scan.header(item[0])[item[1]].rsplit()[each+1]) for each in indexs]
            if key in ['or0_lib', 'or1_lib']:
                #correct the nu for psic geo by nu = gam + mu
                values[-2] = values[-2] + values[-3]
                #chi equals 90 instead of 0? double check it
                values[5] = values[5] +90
            if key in ['or0_lib','or1_lib']:
                items_values[key]={'h':values[0:3]}
                for mo, val in zip(['delta','eta','chi','phi','mu','nu','lam'],values[3:]):
                    items_values[key][mo] = val
            else:
                items_values[key] = values
        print(items_values)
        return items_values

if __name__=='__main__':
    test=Fiofile()
    print('datacol of eh_t01',test.get_datacol('eh_t01'))
    print('mu values',test.get_parameter('mu'))
    print('scan macro',test.get_header(0))
    print('parameter keys',test.parameter_info.keys())
