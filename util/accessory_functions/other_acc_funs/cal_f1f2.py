"""
Functions used for resonant dispersion items

Authors/Modifications:
----------------------
Frank Heberling (Frank.Heberling@kit.edu)

"""

import numpy as Num
from pylab import *
from scipy.optimize import leastsq
from scipy.interpolate import interp1d


def f1f2(datafile, expfile, e0, e0shift, output ='exp.f1f2', n=30):

    """
    Function to calculate Differential Kramers Kronig transformation from
    experimental f2 (normalized XANES) to experimental f1

    Literature: Ohta and Ishida (1988) ... integration Methods for Kramers Kronig Transformation, Applied Spectroscopy 42,6
                Cross et al. (1998) ... theoretical x-ray resonant scattering amplitudes ...,Physical Review B, 58, 17

    datafile: Hephaestus *.f1f2 file (1eV steps) --> Cromer Liberman (CL) f1 and f2
    expfile: Athena normalized XANES file (same eV grid as .f1f2 file) --> experimental f2
    output: filename experimental E f1 f2 wll be written to (for use in rasd_menu)
    n = 30: number of datapoints to match f2 CL with f2 exp.
    e0: theoretical edge energy
    e0shift: to match theoretical edge with experimental edge
    """
    

    #f1f2 holds
    #[0] - E exp
    #[1] - f1 theo (->E exp)
    #[2] - f2 theo (->E exp)
    #[3] - diff f1theo f1exp 
    #[4] - f1 exp
    #[5] - f2 exp
    
    e0 = e0 + e0shift
    #read Cromer-Liberman calculated f1f2 from HEPHAESTUS
    f = file(datafile, 'r')
    data = f.readlines()
    f.close()
    theo = Num.ndarray((0,3),float)
    for i in range(len(data)):
        if '#' not in data[i]:
            tmp = str.rsplit(data[i])
            if float(tmp[0])< e0:
                theo = Num.append(theo, [[int(round(float(tmp[0]))),float(tmp[1]),float(tmp[2])]], axis = 0)
            else:
                theo = Num.append(theo, [[int(round(float(tmp[0]))),float(tmp[1]),float(tmp[2])]], axis = 0)
    theo = theo.transpose()
    theo[0] = theo[0] + e0shift
    f1 = interp1d(theo[0], theo[1], 'cubic')
    f2 = interp1d(theo[0], theo[2], 'cubic')

    #read experimental f2 (normalized XANES) and interpolate theoretical f1 and f2 to match the E of f2 exp
    f = file(expfile, 'r')
    data = f.readlines()
    f.close()
    f1f2 = Num.ndarray((0,6),float)
    for i in range(len(data)):
        if '#' not in data[i]:
            tmp = str.rsplit(data[i])
            f1f2 = Num.append(f1f2, [[float(tmp[0]),f1(float(tmp[0])), f2(float(tmp[0])), 0., 0., float(tmp[1])]], axis = 0)
    f1f2 = f1f2.transpose()

    low_cl = 0
    up_cl = 0
    low_exp = 0
    up_exp = 0
    for i in range(n):
        low_cl = low_cl + f1f2[2][i]
        low_exp = low_exp + f1f2[5][i]
        up_cl = up_cl + f1f2[2][len(f1f2[0])-i-1]
        up_exp = up_exp + f1f2[5][len(f1f2[0])-i-1]
    
    low_cl= low_cl /n
    low_exp= low_exp /n
    up_cl = up_cl /n
    up_exp = up_exp /n
    rel = (up_cl-low_cl)/(up_exp-low_exp)
    diff = low_cl - low_exp*rel
    f1f2[5] = f1f2[5]*rel + diff

    #calculate f1exp from f2exp (Kramers-Kronig) (see Ohta 1988/ Cross 1998)
    for i in range(Num.shape(f1f2)[1]):
        sum = 0
        if divmod(float(i),2)[1] == 0:
            for j in range(1, len(f1f2[0]),2):
                sum = sum + (f1f2[5][j]-f1f2[2][j])/(f1f2[0][j] - f1f2[0][i])+(f1f2[5][j]-f1f2[2][j])/(f1f2[0][j] + f1f2[0][i])
        else:
            for j in range(0, len(f1f2[0]),2):
                sum = sum + (f1f2[5][j]-f1f2[2][j])/(f1f2[0][j] - f1f2[0][i])+(f1f2[5][j]-f1f2[2][j])/(f1f2[0][j] + f1f2[0][i])
        f1f2[3][i] = (sum * 2 / Num.pi)


    f1f2[4] = f1f2[1] + f1f2[3]

    #write experimental values to .f1f2 file
    f = file(output,'w')
    f.write('# file: "'+output+'" containing experimental f1 f2 values \n')
    f.write('# calculated using Cromer Liberman f1 f2 from: "'+datafile+'"\n')
    f.write('# and experimental f2 from: "'+expfile+'"\n')
    f.write('# E0 = '+str(e0)+', e0shift = '+str(e0shift)+'\n')
    f.write('# Energy f1exp f2exp \n')
    i=0
    for i in range(len(f1f2[0])):
        f.write(str(f1f2[0][i])+'   '+str(f1f2[4][i])+'   '+str(f1f2[5][i])+' \n')
    f.close()

    #plot results
    figure(1)
    clf()
    plot(f1f2[0],f1f2[1],'b-')
    plot(f1f2[0],f1f2[2],'b-')
    plot(f1f2[0],f1f2[4],'r-')
    plot(f1f2[0],f1f2[5],'g-')
