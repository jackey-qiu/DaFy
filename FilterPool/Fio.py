import numpy as np
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
class Fiofile(object):
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

if __name__=='__main__':
    test=Fiofile()
    print('datacol of eh_t01',test.get_datacol('eh_t01'))
    print('mu values',test.get_parameter('mu'))
    print('scan macro',test.get_header(0))
    print('parameter keys',test.parameter_info.keys())
