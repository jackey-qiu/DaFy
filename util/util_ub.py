import numpy as np 
import math

#this func works only for cubic lattice structure
def cal_azimuth_from_hkl(h,k,l, E=25.4, c=3.615):
    wl = E*0.019217594
    cosA = wl/2/c*(h**2+k**2+l**2)/(h**2+k**2)**0.5
    if h!=0:
        B = np.arctan(k/h)/np.pi*180
    else:
        B = 90
    return np.arccos(cosA)/np.pi*180-B
    # return np.arccos(cosA)/np.pi*180

def cal_2theta_from_l(l, E = 25.4, c=3.615):
    wl = E*0.019217594
    sin2theta = wl*l/c
    return np.arcsin(sin2theta)/np.pi*180

def find_limits_for_l(h,k, E=25.4, c=3.615):
    ls=np.arange(0.1,6,0.01)
    max_ang = 0
    max_l =0
    for each_l in ls:
        ang = cal_azimuth_from_hkl(h,k,each_l,E,c)
        if math.isnan(float(ang)):
            return max_ang,max_l
        else:
            max_ang = ang
            max_l = each_l
    return max_ang,max_l

if __name__ == '__main__':
    hks = [[1,0],[2,0],[1,1],[2,2],[2,1],[3,0],[3,1]]
    E = 25.4
    c = 3.615
    print('Energy:{}keV;c:{}A'.format(E,c))
    for each in hks:
        print(each)
        max_angle,max_l = find_limits_for_l(each[0],each[1],E,c)
        print('{}{}L rod, maximum l={} reached at azimuth angle of{}, and 2theta angle of {}'.format(each[0],each[1],max_l,max_angle, cal_2theta_from_l(max_l, E, c)))


