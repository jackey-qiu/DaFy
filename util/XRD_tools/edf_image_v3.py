from pyspec import spec
from PyMca5.PyMcaIO import EdfFile
import os
import numpy as np


class DetImage:
    def __init__(self, img, motors, counters, header=None):
        self.img = img
        self.motors = motors
        self.counters = counters
        self.header = header

'''
    Loads the images from a spec scan.
'''
class edf_image_loader:
    def __init__(self, spec_filename, image_foldername):
        self.spec_file = spec.SpecDataFile(spec_filename)
        self.image_foldername = image_foldername

    def get_no_frames(self, scan_no):
        return self.spec_file[scan_no].data.shape[0]

    def load_frame(self, scan_no, frame_no, gz_compressed=True, normalize=False, monitor_name=None, monitor_names=None, remove_rows=None, remove_cols=None):
        header = dict()
        print self.spec_file[scan_no].header
        for line in self.spec_file[scan_no].header.split('\n'):
            header[line.split(' ')[0]] = line[len(line.split(' ')[0]):]
        comments = dict()
        for line in self.spec_file[scan_no].comments.split('\n'):
            comments[line.split(':')[0].strip()] = line[len(line.split(':')[0])+1:].strip()
        frames = self.get_no_frames(scan_no)

        if('ccoscan' in header['#S'] or 'zapline' in header['#S']):
            img_folder = comments['#C DIRECTORY'].split('/')[-1]+'/'
            if(img_folder == '/'):
                img_folder = comments['#C DIRECTORY'].split('/')[-2]+'/'
            zap_scan_no = int(comments['#C ZAP SCAN NUMBER'])
            radix = comments['#C RADIX']
            filename = radix + '_mpx-x4_%s_0000_0000.edf'%(str(zap_scan_no).zfill(4))
            multiframe_edf_frame_no = frame_no
        else:
            first_frame = int(header['#UCCD'].split('#r')[-1].split('.')[0])
            img_folder = header['#UCCD'].replace('//','/').split('/')[-2]+'/' #in the beginning the path of MA3886 had '//' for some reason
            filename_template = header['#UCCD'].split('/')[-1].split('#r')[0] + '#r.' + header['#UCCD'].split('/')[-1].split('.')[-1]
            filename = filename_template.replace('#n', str(scan_no).zfill(3)).replace('#p', str(frame_no).zfill(3)).replace('#r', str(first_frame+frame_no).zfill(3))
            multiframe_edf_frame_no = 0

        # automatically detect if frame is compressed or not
        if(not os.path.exists(os.path.join(self.image_foldername, img_folder, filename))):
            filename = filename.replace('.edf', '.edf.gz')

        #if(gz_compressed):
        #    filename = filename.replace('.edf', '.edf.gz')

        if(frame_no >= frames):
            raise IndexError("Frame number does not exist.")

        img_filename = os.path.join(self.image_foldername, img_folder, filename)
        edf = EdfFile.EdfFile(img_filename, 'r')


        edf_header = edf.GetHeader(multiframe_edf_frame_no)
        motors = dict()
        motor_mne = edf_header['motor_mne'].split()
        motor_pos = edf_header['motor_pos'].split()
        for i in xrange(len(motor_mne)):
            motors[motor_mne[i]] = float(motor_pos[i])
        counters = dict()
        if(not ('ccoscan' in header['#S'] or 'zapline' in header['#S'])):
            counter_mne = edf_header['counter_mne'].split()
            counter_pos = edf_header['counter_pos'].split()
            for i in xrange(len(counter_mne)):
                counters[counter_mne[i]] = float(counter_pos[i])

        the_img = np.array(edf.GetData(multiframe_edf_frame_no), dtype='float')
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

        return DetImage(the_img, motors, counters, header)

    def load_all_frames(self, scan_no, gz_compressed=True, normalize=False, monitor_name=None, remove_rows=None, remove_cols=None):
        frame_no = self.get_no_frames(scan_no)
        frames = np.zeros(frame_no)
        for i in xrange(frame_no):
            frames[i] = self.load_frame(scan_no, i, gz_compressed, normalize, monitor_name, remove_rows, remove_cols)
        return frames
