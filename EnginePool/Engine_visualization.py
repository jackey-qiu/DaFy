import sys
import matplotlib
matplotlib.use("tkAgg")
from numpy import dtype
sys.path.append('./XRD_tools/')
from P23config import *
from Fio import Fiofile
import numpy as np
from scipy.interpolate import griddata
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter
import p23_tools_debug as p23
import matplotlib.pyplot as plt
# from pyspec import spec
import subprocess
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FFMpegWriter

def draw_lines_on_image(ax_handle,x_y_grid,variable_list,direction = 'horizontal',\
                        colors='gray',line_style='-',marker = None,\
                        xlabel=r'$q_\parallel$ / $\AA^{-1}$',\
                        ylabel='$q_\perp$ / $\AA^{-1}$',\
                        fontsize=20):
    line_ax_container = []
    x_couples, y_couples = [], []
    if direction == 'horizontal':
        x_couples = [x_y_grid[0,:][(0,-1)]]*len(variable_list)
        y_couples = [[each,each] for each in variable_list]
    elif direction == 'vertical':
        y_couples = [x_y_grid[:,0][(0,-1)]]*len(variable_list)
        x_couples = [[each,each] for each in variable_list]
    for i in range(len(x_couples)):
        temp_line_ax, _= ax_handle.plot([x_couples[i],y_couples[i],line_style, color = colors[i], marker = marker)
        line_ax_container.append(temp_line_ax)
    ax_handle.set_xlabel(xlabel,fontsize=fontsize)
    ax_handle.set_ylabel(ylabel,fontsize=fontsize)
    return ax_handle, line_ax_container

def movie_creator(fig_handle, movie_name,fps = 5):
    canvas_width, canvas_height = fig_handle.canvas.get_width_height()
    # Open an ffmpeg process
    outf = movie_name
    cmdstring = ('ffmpeg',
            '-y', '-r', str(fps), # overwrite, 30fps
            '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
            '-pix_fmt', 'argb', # format
            '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'mpeg4', outf) # output encoding
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
    agg=fig_handle.canvas.switch_backends(FigureCanvasAgg)
    return p, agg

def update_line(line_ax_handle, atr_value_dic={'set_data':None,'set_cmap':'gnuplot2'}):
    for key in atr_value_dic.keys():
        getattr(line_ax_handle, key)(atr_value_dic[key])
    return line_ax_handle

