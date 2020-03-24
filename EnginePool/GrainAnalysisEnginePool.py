import numpy as np

def RHE(E_AgAgCl, pH=13):
    # electrode is 3.4 M KCl
    return 0.205 + E_AgAgCl + 0.059*pH

def POT(E_AgAgCl, plot_vs_RHE=False, pH=13):
    if(plot_vs_RHE):
        return RHE(E_AgAgCl, pH)
    else:
        return E_AgAgCl

def strain_ip(q_ip, HKL, lattice):
    q_bulk = lattice.q(HKL)
    q_ip_bulk = np.sqrt(q_bulk[0]**2 + q_bulk[1]**2)
    return (q_ip_bulk/q_ip - 1.0)*100.

def strain_ip_with_uncertainty(q_ip, HKL, lattice, uncertainty_q_ip):
    q_bulk = lattice.q(HKL)
    q_ip_bulk = np.sqrt(q_bulk[0]**2 + q_bulk[1]**2)
    _strain_ip = (q_ip_bulk/q_ip - 1.0)*100.
    uncedrtainty_strain_ip = np.abs(q_ip_bulk/q_ip**2*100*uncertainty_q_ip)
    return (_strain_ip, uncedrtainty_strain_ip)

def strain_oop(q_oop, HKL, lattice):
    return (lattice.q(HKL)[2]/q_oop - 1.0)*100.

def strain_oop_with_uncertainty(q_oop, HKL, lattice, uncertainty_q_oop):
    q_oop_bulk = lattice.q(HKL)[2]
    _strain_oop = (q_oop_bulk/q_oop - 1.0)*100.
    uncertainty_strain_oop = np.abs(q_oop_bulk/q_oop**2*100*uncertainty_q_oop)
    return  (_strain_oop, uncertainty_strain_oop)

def grain_size_calculator(FWHM):
    return 0.2*np.pi/np.array(FWHM)

def cal_strain_and_grain(data, HKL, lattice, key_map = {'ip_pos':'cen_ip','oop_pos':'cen_oop','ip_FWHM':'FWHM_ip','oop_FWHM':'FWHM_oop'}):
    data['strain_ip'] = list(strain_ip(data[key_map['ip_pos']],HKL, lattice))
    data['strain_oop'] = list(strain_oop(data[key_map['oop_pos']],HKL, lattice))
    data['grain_size_ip'] = list(grain_size_calculator(data[key_map['ip_FWHM']]))
    data['grain_size_oop'] = list(grain_size_calculator(data[key_map['oop_FWHM']]))
    return data


