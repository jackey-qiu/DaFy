from pyspec import spec
from PyMca5.PyMcaIO import EdfFile
import os
import numpy as np
import gzip


class DetImage:
    def __init__(self, img, motors, counters):
        self.img = img
        self.motors = motors
        self.counters = counters

'''
    Loads the images from a spec scan.
'''
class edf_image_loader:
    def __init__(self, spec_filename, image_foldername):
        self.spec_file = spec.SpecDataFile(spec_filename)
        self.image_foldername = image_foldername
        
    def get_no_frames(self, scan_no):
        return self.spec_file[scan_no].data.shape[0]
        #return len(self.spec_file[scan_no].Epoch)
    
    def load_frame(self, scan_no, frame_no, gz_compressed=True, normalize=False, monitor_name=None, monitor_names=None, remove_rows=None, remove_cols=None):
        header = dict()
        for line in self.spec_file[scan_no].header.split('\n'):
            header[line.split(' ')[0]] = line[len(line.split(' ')[0]):]
        comments = []
        for line in self.spec_file[scan_no].comments.split('\n'):
            comments.append(line)
        
        frames = self.get_no_frames(scan_no)
        if(frame_no >= frames):
            raise IndexError("Frame number does not exist.")
        
        if('ccoscan' in header['#S']):
            for comment in comments:
                if('#C DIRECTORY' in comment):        img_folder = comment.split(':')[-1].strip().split('/')[-1]
                if('#C RADIX' in comment):            radix = comment.split(':')[-1].strip()
                if('#C ZAP SCAN NUMBER' in comment):  zap_scan_no = comment.split(':')[-1].strip()
                if('#C ZAP IMAGE NUMBER' in comment):  zap_image_no = comment.split(':')[-1].strip()
            filename = img_folder + '/' + radix + '_mpx-x4_%s_0000_0000.edf'%(str(zap_scan_no).zfill(4))
            edf_frame_no = frame_no
        else:
            first_frame = int(header['#UCCD'].split('#r')[-1].split('.')[0])
            img_folder = header['#UCCD'].split('/')[-2]+'/' 
            filename_template = header['#UCCD'].split('/')[-1].split('#r')[0] + '#r.' + header['#UCCD'].split('/')[-1].split('.')[-1]
            filename = img_folder + filename_template.replace('#n', str(scan_no).zfill(3)).replace('#p', str(frame_no).zfill(3)).replace('#r', str(first_frame+frame_no).zfill(3))
            edf_frame_no = 0
        
        img_filename = os.path.join(self.image_foldername, filename)
        if(not os.path.exists(img_filename)):
            if(os.path.exists(img_filename.replace('.edf', '.edf.gz'))):
                inF = gzip.open(img_filename.replace('.edf', '.edf.gz'), 'rb')
                outF = open(img_filename, 'wb')
                outF.write( inF.read() )
                inF.close()
                outF.close()

        edf = EdfFile.EdfFile(img_filename, 'r')
        

        edf_header = edf.GetHeader(edf_frame_no)
        motors = dict()
        motor_mne = edf_header['motor_mne'].split()
        motor_pos = edf_header['motor_pos'].split()
        for i in xrange(len(motor_mne)):
            motors[motor_mne[i]] = float(motor_pos[i])
        counters = dict()
        if(not('ccoscan' in header['#S'])):
            counter_mne = edf_header['counter_mne'].split()
            counter_pos = edf_header['counter_pos'].split()
            for i in xrange(len(counter_mne)):
                counters[counter_mne[i]] = float(counter_pos[i])        
        
        the_img = np.array(edf.GetData(edf_frame_no), dtype='float')
        if(remove_rows != None):
            the_img = np.delete(the_img, remove_rows, axis=0)
        if(remove_cols != None):
            the_img = np.delete(the_img, remove_cols, axis=1)
        
        if(normalize):
            if(monitor_name):
                mon_count = float(getattr(self.spec_file[scan_no], monitor_name)[frame_no])
                the_img /= mon_count
            if(monitor_names):
                for mon_name in monitor_names:
                    mon_count = float(getattr(self.spec_file[scan_no], mon_name)[frame_no])
                    the_img /= mon_count
        
        return DetImage(the_img, motors, counters)
        
    def load_all_frames(self, scan_no, gz_compressed=True, normalize=False, monitor_name=None, remove_rows=None, remove_cols=None):
        frame_no = self.get_no_frames(scan_no)
        frames = np.zeros(frame_no)
        for i in xrange(frame_no):
            frames[i] = self.load_frame(scan_no, i, gz_compressed, normalize, monitor_name, remove_rows, remove_cols)
        return frames