import os,sys
os.chdir("C:\\apps\\genx\\")
import model,diffev,time,fom_funcs
import filehandling as io
import glob
#python create_tab_files D:\\Google Drive\\modeling files\\GenX model file\\fitting with brute force\\RAXS\\ * RAXS
path=sys.argv[1]
files=glob.glob(path+sys.argv[2])
for file in files:
    mod = model.Model()
    config = io.Config()
    opt = diffev.DiffEv()
    io.load_gx(file,mod,opt,config)
    content=mod.parameters.get_ascii_output()
    file_list=file.rsplit('\\')[-1].rsplit('_')
    tab_file_name=sys.argv[3]+'_'+file_list[4]+'_R'+file_list[-1][-4]+'.tab'
    f=open(path+tab_file_name,'w')
    f.write(content)
    f.close()
    
