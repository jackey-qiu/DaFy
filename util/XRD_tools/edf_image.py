from pyspec import spec
from PyMca5.PyMcaIO import EdfFile
import os
import numpy as np

'''
    Loads the images from a spec scan.
'''
class edf_image_loader:
    def __init__(self, spec_filename):
        self.spec_file = spec.SpecDataFile(spec_filename)
        
    def get_no_frames(self, scan_no):
        return len(self.spec_file[scan_no].Epoch)
    
    def load_frame(self, image_foldername, scan_no, frame_no, gz_compressed=True, normalize=False, monitor_name=None, monitor_names=None, remove_rows=None, remove_cols=None):
        header = dict()
        for line in self.spec_file[scan_no].header.split('\n'):
            header[line.split(' ')[0]] = line[len(line.split(' ')[0]):]
        frames = self.get_no_frames(scan_no)
        first_frame = int(header['#UCCD'].split('#r')[-1].split('.')[0])
        img_folder = header['#UCCD'].split('/')[-2]+'/' 
        filename_template = header['#UCCD'].split('/')[-1].split('#r')[0] + '#r.' + header['#UCCD'].split('/')[-1].split('.')[-1]
           
        if(gz_compressed):
            filename_template = filename_template.replace('.edf', '.edf.gz')
        
        if(frame_no >= frames):
            raise IndexError("Frame number does not exist.")
        
        img_filename = os.path.join(image_foldername, img_folder, filename_template.replace('#n', str(scan_no).zfill(3)).replace('#p', str(frame_no).zfill(3)).replace('#r', str(first_frame+frame_no).zfill(3)))
        edf = EdfFile.EdfFile(img_filename, 'r')
        
        the_img = np.array(edf.GetData(0), dtype='float')
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
            return the_img
        
        return the_img
        
    def load_all_frames(self, image_foldername, scan_no, gz_compressed=True, normalize=False, monitor_name=None, remove_rows=None, remove_cols=None):
        frame_no = self.get_no_frames(scan_no)
        frames = np.zeros(frame_no)
        for i in xrange(frame_no):
            frames[i] = self.load_frame(scan_no, i, gz_compressed, normalize, monitor_name, remove_rows, remove_cols)
        return frames