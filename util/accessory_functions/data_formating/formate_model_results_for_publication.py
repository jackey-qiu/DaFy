import numpy as np
import os
import sys
import glob

def formate_results_matlab_output(path_head='P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models\\Li_Zr_mica',ctr_result='param_out_file.dat',raxr_result='Zr_mica_zr_100mM_RbCl_MD_Jan08_parameters',raxr_el='Zr'):
    ctr=np.loadtxt(os.path.join(path_head,ctr_result),comments='%')
    raxr=np.loadtxt(os.path.join(path_head,raxr_result),comments='%')
    #print raxr
    output=open(os.path.join(path_head,'output_for_publication.dat'),'w')
    def _get_info(row_data=[],scale=1):
        return '%s(%s)'%(np.around((row_data[1]-1)*scale,3),np.around(row_data[6]*scale,3))

    def _get_info_2(row_data=[],scale=1):
        return '%s(%s)'%(np.around(row_data[1]*scale,3),np.around(row_data[2]*scale,3))
    all_index_ctr=[10,13,16,45,48,51,54,57,60]
    all_index_raxr=range(0,10)
    ctr_model=[]
    ctr_model.append('\t'.join(('el','zi','ui','ci'))+'\n')
    for i in range(len(all_index_ctr)):
        try:
            this=all_index_ctr[i]
            if (ctr[this+1][1]-1)>0:
                ctr_model.append('\t'.join(['O',_get_info(ctr[this]),_get_info(ctr[this+1]),_get_info(ctr[this+2],scale=2)])+'\n')
            else:
                pass
        except:
            break
    ctr_model.append('\t'.join(['zw','d','uw','du'])+'\n')
    ctr_model.append('\t'.join([_get_info(ctr[3]),_get_info(ctr[3+2]),_get_info(ctr[3+1]),_get_info(ctr[3+3])])+'\n')
    ctr_model.append('\t'.join(['raxr_el','zi','ui','ci'])+'\n')
    for i in range(len(all_index_raxr)):
        try:
            this=all_index_raxr[i]
            if raxr[this][1]>0:
                #print this,ctr[this][1]
                ctr_model.append('\t'.join([raxr_el,_get_info_2(raxr[this+10]),_get_info_2(raxr[this+20]),_get_info_2(raxr[this],scale=2)])+'\n')
            else:
                pass
        except:
            break
    for each in ctr_model:
        output.write(each)
        print (each.rstrip())
    output.close()
    return None

def recursive_glob(rootdir='P:\\My stuff\\Manuscripts\\zr on mica (cation effect)\\Matlab models', pattern='*_parameters'):
    model_number=0
    for looproot, _, filenames in os.walk(rootdir):
        os.chdir(looproot)
        files=glob.glob(pattern)
        raxr_file=sorted( files, key = lambda file: os.path.getctime(file))
        ctr_file=glob.glob('param_out_file.dat')
        if raxr_file!=[] and ctr_file!=[]:
            print('One model is located, and the formatting is starting now ...')
            formate_results_matlab_output(looproot,ctr_file[-1],raxr_file[-1])
            model_number+=1
            print('Done with this model!\n')
    print(model_number,' model results have been formated for publication!')
    os.chdir('P:\\apps\\genx_pc_qiu\\supportive_functions')
