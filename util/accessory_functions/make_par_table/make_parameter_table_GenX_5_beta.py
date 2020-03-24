import numpy as num
import numpy as np

Al_slab=['O4_O4O3_O7O8','O3_O4O3_O7O8','O5_O3O4_O8O7','Al1_Al4Al3_Al7Al8','Al2_Al3Al4_Al8Al7','O1_O4O3_O7O8','O2_O3O4_O8O7','O6_O3O4_O8O7','Al3_Al4Al3_Al7Al8','Al3_Al6Al5_Al1Al2','O6_O5O6_O2O1','O2_O5O6_O2O1','O1_O6O5_O1O2','Al2_Al5Al6_Al2Al1','Al1_Al6Al5_Al1Al2','O5_O5O6_O2O1','O4_O6O5_O1O2','O3_O6O5_O1O2']
Si_slab=['O4_O4O3_O7O8','O3_O4O3_O7O8','O5_O3O4_O8O7','Si1_Si4Si3_Si7Si8','Si2_Si3Si4_Si8Si7','O1_O4O3_O7O8','O2_O3O4_O8O7','O6_O3O4_O8O7','Al3_Al4Al3_Al7Al8','Al3_Al6Al5_Al1Al2','O6_O5O6_O2O1','O2_O5O6_O2O1','O1_O6O5_O1O2','Si2_Si5Si6_Si2Si1','Si1_Si6Si5_Si1Si2','O5_O5O6_O2O1','O4_O6O5_O1O2','O3_O6O5_O1O2']
"""
################explanation of each componont in structure################
#surface1#
'dxdy'=[8,[0.,-0.05,0.05],'True']:consider the top 8 atoms for inplane movement,start value 0,min -0.05 and max 0.05
'dz'=[14,[0,-0.05,0.05],'True']:consider the top 14 layers for outof plane movement (NOTE here consider in pairs)
'u' and 'oc' are same as 'dz'
#sorbate1#(set to None if clean surface)
'angles':first one is phi angle and second one is top angle
'dxdy':make sure the element is suffixed with number counting from 1
'u' and 'oc' are the same as 'dxdy'
'oc':Note we group all sorbates together to consider occupancy of sorbates
#water1#
'dxdy' have the number of items the same as the water numbers
'dz' and 'oc' and 'u' has half number of items of total water numbers, because we group each two waters at same layer for oc and u consideration
"""
surface1={'dxdy':[4,[0.,-0.05,0.05],'True'],'dz':[0,[0,-0.05,0.05],'False'],'oc':[0,[1,0.6,1],'False'],'u':[0,[0.4,0.32,0.8],'False']}
surface2={'dxdy':[4,[0.,-0.05,0.05],'True'],'dz':[0,[0,-0.05,0.05],'False'],'oc':[0,[1,0.6,1],'False'],'u':[0,[0.4,0.32,0.8],'False']}
surface3={'dxdy':[4,[0.,-0.05,0.05],'True'],'dz':[10,[0,-0.05,0.05],'True'],'oc':[10,[1,0.6,1],'True'],'u':[10,[0.4,0.32,0.8],'True']}
surface4={'dxdy':[2,[0.,-0.05,0.05],'True'],'dz':[10,[0,-0.05,0.05],'True'],'oc':[2,[1,0.6,1],'False'],'u':[2,[0.4,0.32,0.8],'False']}

sorbate1={'angles':[[5.,1,6.28],[1.4,0.7,2],'False'],'dxdy':[['Pb1',[0,-0.1,0.1],'True'],['HO1',[0,-0.1,0.1],'True']],\
         'dz':[['Pb1',[0,-0.1,0.1],'True'],['HO1',[0,-0.1,0.1],'True']],'u':[['Pb1',[0.2,0.2,0.8],'True'],['HO1',[0.4,0.4,0.9],'True']],'oc':[[0.2,0.1,0.8],'True']}
sorbate2={'angles':[[5.1,1,6.28],[1.5,0.7,2],'True'],'dxdy':[['Pb1',[0,-0.5,0.5],'True'],['HO1',[0,-0.5,0.5],'True']],\
         'dz':[['Pb1',[0,-0.5,0.5],'True'],['HO1',[0,-0.5,0.5],'True']],'u':[['Pb1',[0.2,0.2,0.8],'True'],['HO1',[0.4,0.4,0.9],'True']],'oc':[[0.2,0.1,0.8],'True']}
water1={'dxdy':[[0,-0.5,0.5,'False'],[0,-0.5,0.5,'False']],'dz':[[0,-0.5,0.5,'False'],[0,-0.5,0.5,'False']],\
       'oc':[[0.4,0,1,'False'],[0.4,0,1,'False']],'u':[[5,0.5,10,'False'],[5,0.5,10,'False']]}
water2={'dxdy':[[0,-0.5,0.5,'True'],[0,-0.5,0.5,'True']],'dz':[[0,-0.5,0.5,'True'],[0,-0.5,0.5,'True']],\
       'oc':[[0.4,0,1,'True'],[0.4,0,1,'True']],'u':[[5,0.5,10,'True'],[5,0.5,10,'True']]}
##note the domain_tag is the number to lable each domain continually from 1##
"""
structure={'domain1':{'domain_type':'half_layer','domain_tag':1,'surface':surface1,'sorbate':sorbate1,'water':water1},
           'domain2':{'domain_type':'half_layer','domain_tag':2,'surface':surface2,'sorbate':None,'water':water2}}
"""
structure={'domain1':{'domain_type':'half_layer','domain_tag':1,'surface':surface1,'sorbate':sorbate1,'water':water1},\
           'domain2':{'domain_type':'half_layer','domain_tag':2,'surface':surface2,'sorbate':sorbate2,'water':water2},\
           'domain3':{'domain_type':'half_layer','domain_tag':3,'surface':surface3,'sorbate':None,'water':water2}}
#make_structure will be used inside genx script to generate the parameter table
def make_structure(file_path,sorbate_N,O_N,water_N,Domains,Metal,binding_mode=['BD']*3,local_structure='tetrahedral',add_distal_wild=None,use_domains=[1]*10,N_raxr=0,bk_model='Linear',domain_raxr_el=[1,1,0,0],layered_water=None,layered_sorbate=None,dxdy_layer=3,dz_layer=10,u_layer=10,oc_layer=10):
    structure={}
    for i in range(len(Domains)):
        if use_domains[i]==1:
            oc_N=10
            #if sorbate_N[i]==0:oc_N=6
            temp_surface={'dxdy':[dxdy_layer,[0.,-0.04,0.04],'True'],'dz':[dz_layer,[0,-0.03,0.03],'True'],'oc':[oc_layer,[1,0.6,1],'True'],'u':[u_layer,[0.4,0.32,0.8],'True']}
            domain_type=None
            if Domains[i]==1:
                domain_type='Al'
            elif Domains[i]==2:
                domain_type='Si'
            temp_sorbate={}
            if sorbate_N[i]==0:
                temp_sorbate=None
            else:
                temp_sorbate['dxdydz']=[[Metal[i][j]+'_set'+str(2*j+1),[0,-0.5,0.5],'True'] for j in range(sorbate_N[i]/2)]+[['HO'+str(k+1)+'_set'+str(2*j+1),[0,-0.5,0.5],'True'] for j in range(len(O_N[i])) for k in range(O_N[i][j])]
                temp_sorbate['u']=[[Metal[i][j]+'_set'+str(2*j+1),[0.2,0.2,1.],'True'] for j in range(sorbate_N[i]/2)]+[['HO'+str(k+1)+'_set'+str(2*j+1),[0.2,0.2,1.],'True'] for j in range(len(O_N[i])) for k in range(O_N[i][j])]
                temp_sorbate['oc']=[[Metal[i][j]+'_set'+str(2*j+1),[0.3,0.,1.],'True'] for j in range(sorbate_N[i]/2)]+[['HO'+str(k+1)+'_set'+str(2*j+1),[0.3,0.,1.],'True'] for j in range(len(O_N[i])) for k in range(O_N[i][j])]
                temp_sorbate['sorbate_number']=sorbate_N[i]/2
                temp_sorbate['oxygen_number']=O_N[i]
                temp_sorbate['distal_wild']=add_distal_wild[i]
            temp_water={}
            if water_N[i]==0:
                temp_water=None
            else:
                temp_water['dxdy']=[[0,-0.5,0.5,'True'] for j in range(water_N[i])]
                temp_water['dz']=[[0,-0.5,0.5,'True'] for j in range(water_N[i]/2)]
                temp_water['u']=[[3,0.8,10,'True'] for j in range(water_N[i]/2)]
                temp_water['oc']=[[0.3,0,0.6,'True'] for j in range(water_N[i]/2)]
            structure['domain'+str(i+1)]={'binding_mode':binding_mode[i],'domain_type':domain_type,'domain_tag':i+1,'surface':temp_surface,'sorbate':temp_sorbate,'water':temp_water}
        else:
            pass
    return table_maker(table_file_path=file_path+'table.tab',structure_info=structure,local_structure=local_structure,N_raxr=N_raxr,domain_raxr_el=domain_raxr_el,layered_water=layered_water,layered_sorbate=layered_sorbate)

def table_maker(table_file_path='D:\\table.tab',structure_info=structure,local_structure='tetrahedral',N_raxr=0,domain_raxr_el=[1,1,0,0],layered_water=None,layered_sorbate=None):

    f=open(table_file_path,'w')
    f.write('#Parameter Value Fit Min Max Error\n')
    f.write('inst.set_inten\t1\tTrue\t1\t10\t-\n')
    #f.write('rgh.setScale_CTR_specular\t1\tTrue\t0.6\t1.5\t-\n')
    f.write('rgh.setBeta\t0\tTrue\t0\t0.4\t-\n')
    f.write('rgh.setMu\t0\tTrue\t0\t50\t-\n')
    f.write('\t0\tFalse\t0\t0\t-\n')

    domain_N=len(structure_info.keys())
    for i in range(domain_N):
        f.write('rgh_domain'+str(i+1)+'.setWt\t1\tFalse\t0\t1\t-\n')

    keys=structure_info.keys()
    keys.sort()
    for key in keys:
        temp_domain=structure_info[key]
        domain_tag=str(structure_info[key]['domain_tag'])
        domain_index=structure_info[key]['domain_tag']-1
        domain_type=temp_domain['domain_type']
        temp_surface=temp_domain['surface']
        temp_sorbate=temp_domain['sorbate']
        temp_water=temp_domain['water']
        atom_list=[]
        f.write('#############'+key+'############\n')
        if domain_type=='Al':
            atom_list=Al_slab
        elif domain_type=='Si':
            atom_list=Si_slab
        ##do surface
        #out of plane movement together with inplane movement ie dxdydz
        f.write('#dxdydz\n')
        for i in range(temp_surface['dz'][0]):
            if i<temp_surface['dxdy'][0]:
                s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[i]+'_D'+domain_tag+'.setdx',\
                                                      temp_surface['dxdy'][1][0],temp_surface['dxdy'][2],temp_surface['dxdy'][1][1],temp_surface['dxdy'][1][2],'-')
                f.write(s)
                s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[i]+'_D'+domain_tag+'.setdy',\
                                                      temp_surface['dxdy'][1][0],temp_surface['dxdy'][2],temp_surface['dxdy'][1][1],temp_surface['dxdy'][1][2],'-')
                f.write(s)
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[i]+'_D'+domain_tag+'.setdz',\
                                                  temp_surface['dz'][1][0],temp_surface['dz'][2],temp_surface['dz'][1][1],temp_surface['dz'][1][2],'-')
            f.write(s)
        f.write('\t0\tFalse\t0\t0\t-\n')
        f.write('#oc and u\n')
        #occupancy and thermal factor
        for i in range(temp_surface['u'][0]):
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[i]+'_D'+domain_tag+'.setu',\
                                                  temp_surface['u'][1][0],temp_surface['u'][2],temp_surface['u'][1][1],temp_surface['u'][1][2],'-')
            f.write(s)
        f.write('\t0\tFalse\t0\t0\t-\n')
        for i in range(temp_surface['oc'][0]):
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[i]+'_D'+domain_tag+'.setoc',\
                                                  temp_surface['oc'][1][0],temp_surface['oc'][2],temp_surface['oc'][1][1],temp_surface['oc'][1][2],'-')
            f.write(s)
        f.write('\t0\tFalse\t0\t0\t-\n')
        ##do sorbate
        if temp_sorbate!=None:
            #phi and top_angle
            f.write('####sorbate####\n')
            f.write('#angles\n')
            for i in range(len(temp_domain['binding_mode'])):
                binding_mode=temp_domain['binding_mode'][i]
                tag='_'+str(i*2)
                if binding_mode=='BD':
                    s_phi="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi_BD'+tag,0,'True',-50,50,'-')
                    f.write(s_phi)
                    if local_structure[domain_index][i]=='trigonal_pyramid':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_BD'+tag,60,'True',50,90,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setOffset_BD'+tag,0,'True',-0.1,0.1,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR_BD'+tag,2.25,'True',2.25,2.35,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_1_BD'+tag,0.,'True',0.,180.,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_1_BD'+tag,0.,'True',0.,360.,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_1_BD'+tag,2.25,'True',1.9,2.35,'-'))
                        else:
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setOffset2_BD'+tag,0,'True',-0.1,0.1,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setAngle_offset_BD'+tag,0,'True',0.,360.,'-'))
                    elif local_structure[domain_index][i]=='octahedral':
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_BD'+tag,0.,'True',0.,180.,'-')) for ii in range(4)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_BD'+tag,0.,'True',0.,360.,'-')) for ii in range(4)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_BD'+tag,2.26,'True',1.9,2.4,'-')) for ii in range(4)]
                    elif local_structure[domain_index][i]=='tetrahedral':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setAnchor_offset_BD'+tag,0,'True',-0.5,0.5,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_offset_BD'+tag,0,'True',-50.,50.,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_BD'+tag,0.,'True',0.,180.,'-')) for ii in range(2)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_BD'+tag,0.,'True',0.,360.,'-')) for ii in range(2)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_BD'+tag,1.7,'True',1.5,1.9,'-')) for ii in range(2)]
                        else:
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setAngle_offset_BD'+tag,0,'True',-50,50,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setAngle_offset2_BD'+tag,0,'True',-50,50,'-'))
                elif binding_mode=='MD':
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi_BD'+tag,temp_sorbate['angles'][0][0],temp_sorbate['angles'][2],temp_sorbate['angles'][0][1],temp_sorbate['angles'][0][2],'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_BD'+tag,temp_sorbate['angles'][1][0],temp_sorbate['angles'][2],temp_sorbate['angles'][1][1],temp_sorbate['angles'][1][2],'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR_BD'+tag,2.25,'True',2.25,2.35,'-'))
                elif binding_mode=='TD':
                    if local_structure[domain_index][i]=='trigonal_pyramid':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_TD'+tag,90.,'True',50.,110.,'-'))
                    elif local_structure[domain_index][i]=='octahedral':
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_TD'+tag,0.,'True',0.,180.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_TD'+tag,0.,'True',0.,360.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_TD'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(3)]
                    elif local_structure[domain_index][i]=='tetrahedral':
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_TD'+tag,0.,'True',0.,180.,'-')) for ii in range(1)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_TD'+tag,0.,'True',0.,360.,'-')) for ii in range(1)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_TD'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(1)]
                        else:
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setDr_tetrahedral_TD'+tag,0.,'True',-0.5,0.5,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setDr_bc_tetrahedral_TD'+tag,0.,'True',-0.5,0.5,'-'))
                elif binding_mode=='OS':
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setRot_x_OS'+tag,0,'True',0,360,'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setRot_y_OS'+tag,0,'True',0,360,'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setRot_z_OS'+tag,0,'True',0,360,'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setCt_offset_dx_OS'+tag,0,'True',-0.5,0.5,'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setCt_offset_dy_OS'+tag,0,'True',-0.5,0.5,'-'))
                    f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setCt_offset_dz_OS'+tag,0,'True',-0.2,0.2,'-'))
                    if local_structure[domain_index][i]=='trigonal_pyramid':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_OS'+tag,90.,'True',50.,110.,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR0_OS'+tag,2.25,'True',2.0,2.5,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,180.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,360.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_OS'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(3)]
                    elif local_structure[domain_index][i]=='octahedral':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR0_OS'+tag,2.0,'True',1.8,2.2,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,180.,'-')) for ii in range(6)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,360.,'-')) for ii in range(6)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_OS'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(6)]
                    elif local_structure[domain_index][i]=='tetrahedral':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR0_OS'+tag,1.7,'True',1.5,1.9,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,180.,'-')) for ii in range(4)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,360.,'-')) for ii in range(4)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_OS'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(4)]
                f.write('\t0\tFalse\t0\t0\t-\n')
            #dxdydz
            f.write('#dxdydz\n')
            for i in range(len(temp_sorbate['dxdydz'])):
                s_x="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+temp_sorbate['dxdydz'][i][0]+'_D'+domain_tag+'.setdx',temp_sorbate['dxdydz'][i][1][0],temp_sorbate['dxdydz'][i][2],temp_sorbate['dxdydz'][i][1][1],temp_sorbate['dxdydz'][i][1][2],'-')
                s_y="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+temp_sorbate['dxdydz'][i][0]+'_D'+domain_tag+'.setdy',temp_sorbate['dxdydz'][i][1][0],temp_sorbate['dxdydz'][i][2],temp_sorbate['dxdydz'][i][1][1],temp_sorbate['dxdydz'][i][1][2],'-')
                s_z="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+temp_sorbate['dxdydz'][i][0]+'_D'+domain_tag+'.setdz',temp_sorbate['dxdydz'][i][1][0],temp_sorbate['dxdydz'][i][2],temp_sorbate['dxdydz'][i][1][1],temp_sorbate['dxdydz'][i][1][2],'-')
                f.write(s_x)
                f.write(s_y)
                f.write(s_z)
                f.write('\t0\tFalse\t0\t0\t-\n')
            #u
            f.write('#u\n')
            for i in range(len(temp_sorbate['u'])):
                s_u="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+temp_sorbate['u'][i][0]+'_D'+domain_tag+'.setu',temp_sorbate['u'][i][1][0],temp_sorbate['u'][i][2],temp_sorbate['u'][i][1][1],temp_sorbate['u'][i][1][2],'-')
                f.write(s_u)
            f.write('\t0\tFalse\t0\t0\t-\n')
            #oc
            f.write('#oc\n')
            for i in range(len(temp_sorbate['oc'])):
                s_oc="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+temp_sorbate['oc'][i][0]+'_D'+domain_tag+'.setoc',temp_sorbate['oc'][i][1][0],temp_sorbate['oc'][i][2],temp_sorbate['oc'][i][1][1],temp_sorbate['oc'][i][1][2],'-')
                f.write(s_oc)
            f.write('\t0\tFalse\t0\t0\t-\n')
        ##do water
        if temp_water!=None:
            f.write('####water####\n')
            #dxdy
            """
            f.write('#dxdy\n')
            for i in range(len(temp_water['dxdy'])):
                s_x="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_Os'+str(i+1)+'_D'+domain_tag+'.setdx',temp_water['dxdy'][i][0],temp_water['dxdy'][i][-1],temp_water['dxdy'][i][1],temp_water['dxdy'][i][2],'-')
                s_y="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_Os'+str(i+1)+'_D'+domain_tag+'.setdy',temp_water['dxdy'][i][0],temp_water['dxdy'][i][-1],temp_water['dxdy'][i][1],temp_water['dxdy'][i][2],'-')
                f.write(s_x)
                f.write(s_y)
            f.write('\t0\tFalse\t0\t0\t-\n')
            #dz
            f.write('#dz\n')
            for i in range(len(temp_water['dz'])):
                s_oc="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_waters_set'+str(i+1)+'_D'+domain_tag+'.setdz',temp_water['dz'][i][0],temp_water['dz'][i][-1],temp_water['dz'][i][1],temp_water['dz'][i][2],'-')
                f.write(s_oc)
            f.write('\t0\tFalse\t0\t0\t-\n')
            """
            f.write('#rotation angle and vertical shift\n')
            for i in range(len(temp_water['dz'])):
                #s_alpha="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setAlpha_W_'+str(i+1),60.06054,'True',20,160,'-')
                s_vshift="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setV_shift_W_'+str(i+1),1,'True',0.7,3,'-')
                f.write(s_alpha)
                f.write(s_vshift)
            f.write('\t0\tFalse\t0\t0\t-\n')
            #oc and u
            f.write('#oc and u\n')
            for i in range(len(temp_water['oc'])):
                s_oc="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_waters_set'+str(i+1)+'_D'+domain_tag+'.setoc',temp_water['oc'][i][0],temp_water['oc'][i][-1],temp_water['oc'][i][1],temp_water['oc'][i][2],'-')
                f.write(s_oc)
            f.write('\t0\tFalse\t0\t0\t-\n')
            for i in range(len(temp_water['u'])):
                s_u="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_waters_set'+str(i+1)+'_D'+domain_tag+'.setu',temp_water['u'][i][0],temp_water['u'][i][-1],temp_water['u'][i][1],temp_water['u'][i][2],'-')
                f.write(s_u)
            f.write('\t0\tFalse\t0\t0\t-\n')
    for i in range(N_raxr):
        if bk_model=='Linear':
            f.write('rgh_raxr.setA'+str(i+1)+'\t1\tFalse\t0\t5\t-\n')
            f.write('rgh_raxr.setB'+str(i+1)+'\t0\tFalse\t0\t5\t-\n')
        elif bk_model=='Curved':
            f.write('rgh_raxr.setA'+str(i+1)+'\t1\tFalse\t0\t5\t-\n')
            f.write('rgh_raxr.setB'+str(i+1)+'\t0\tFalse\t0\t5\t-\n')
            f.write('rgh_raxr.setC'+str(i+1)+'\t0\tFalse\t0\t5\t-\n')
        for j in range(len(domain_raxr_el)):
            if domain_raxr_el[j]:
                f.write('rgh_raxr.setA_D'+str(j+1)+'_'+str(i+1)+'\t1\tFalse\t0\t5\t-\n')
                f.write('rgh_raxr.setP_D'+str(j+1)+'_'+str(i+1)+'\t1\tFalse\t0\t1\t-\n')
        f.write('\t0\tFalse\t0\t0\t-\n')
    if layered_water!=None:
        for i in range(len(layered_water)):
            if layered_water[i]:
                f.write('rgh_domain'+str(i+1)+'.setU0\t0.4\tFalse\t0\t5\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setUbar\t0.4\tFalse\t0\t5\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setFirst_layer_height\t2\tFalse\t0\t5\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setD_w\t2.0\tFalse\t1\t4\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setDensity_w\t0.033\tFalse\t0\t0.033\t-\n')
                f.write('\t0\tFalse\t0\t0\t-\n')

    if layered_sorbate!=None:
        for i in range(len(layered_sorbate)):
            if layered_sorbate[i]:
                f.write('rgh_domain'+str(i+1)+'.setU0_s\t0.4\tFalse\t0\t5\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setUbar_s\t0.4\tFalse\t0\t5\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setFirst_layer_height_s\t2\tFalse\t0\t5\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setD_s\t2.0\tFalse\t1\t4\t-\n')
                f.write('rgh_domain'+str(i+1)+'.setDensity_s\t0.033\tFalse\t0\t0.033\t-\n')
                f.write('\t0\tFalse\t0\t0\t-\n')
    f.close()

def set_table_input_raxs(container=[],rgh_group_instance=None,rgh_group_instance_name=None,par_range={'a':[],'b':[],'c':[],'A':[],'P':[]},number_spectra=0,number_domain=2):
    for i in range(number_spectra):
        container.append([rgh_group_instance_name+'.setA'+str(i+1),str(getattr(rgh_group_instance,'getA'+str(i+1))()),'False',str(par_range['a'][0]),str(par_range['a'][1]),'-'])
        container.append([rgh_group_instance_name+'.setB'+str(i+1),str(getattr(rgh_group_instance,'getB'+str(i+1))()),'False',str(par_range['b'][0]),str(par_range['b'][1]),'-'])
        container.append([rgh_group_instance_name+'.setC'+str(i+1),str(getattr(rgh_group_instance,'getC'+str(i+1))()),'False',str(par_range['c'][0]),str(par_range['c'][1]),'-'])
        for j in range(number_domain):
            container.append([rgh_group_instance_name+'.setA'+str(i+1)+'_D'+str(j+1),str(getattr(rgh_group_instance,'getA'+str(i+1)+'_D'+str(j+1))()),'False',str(par_range['A'][0]),str(par_range['A'][1]),'-'])
            container.append([rgh_group_instance_name+'.setP'+str(i+1)+'_D'+str(j+1),str(getattr(rgh_group_instance,'getP'+str(i+1)+'_D'+str(j+1))()),'False',str(par_range['P'][0]),str(par_range['P'][1]),'-'])
        container.append(['','0','False','0','0','-'])
    container.append(['','0','False','0','0','-'])
    return container

def set_table_input(container=[],rgh_group_instance=None,rgh_group_instance_name=None,par_range=None):
    set_pars=np.sort([key for key in vars(rgh_group_instance).keys() if key[0:3]=='set'])
    get_pars=np.sort([key for key in vars(rgh_group_instance).keys() if key[0:3]=='get'])
    for i in range(len(set_pars)):
        container.append([rgh_group_instance_name+'.'+set_pars[i],str(getattr(rgh_group_instance,get_pars[i])()),'False',str(par_range[set_pars[i]][0]),str(par_range[set_pars[i]][1]),'-'])
    container.append(['','0','False','0','0','-'])
    return container

def set_table_input_one_by_one(container=[],rgh_group_instance=None,rgh_group_instance_name=None,par_range=None):
    key=par_range.keys()[0]
    if key in ['setWt1','setWt2']:
        container.append([rgh_group_instance_name+'.'+key,str(par_range[key][0]),'False',str(str(par_range[key][1])),str(par_range[key][2]),'-'])
    else:
        container.append([rgh_group_instance_name+'.'+key,str(par_range[key][0]),'True',str(str(par_range[key][1])),str(par_range[key][2]),'-'])
    return container

def set_table_input_one_by_one_new(container=[],rgh_group_instance=None,rgh_group_instance_name=None,par_range=None):
    key=list(par_range.keys())[0]
    container.append([rgh_group_instance_name+'.'+key,str(par_range[key][0]),str(str(par_range[key][3])),str(str(par_range[key][1])),str(par_range[key][2]),'-'])
    return container

def set_table_input_all(container=[],rgh_instance_list=[],rgh_instance_name_list=[],par_file='pars_ranges.txt',sep=True):
    f=open(par_file)
    lines=f.readlines()
    par_range={}
    par_list=[]
    for line in lines:
        items=line.rstrip().rsplit()
        try:
            par_range[items[0]]=[float(items[1]),float(items[2]),float(items[3]),items[4]]
        except:
            par_range[items[0]]=[float(items[1]),float(items[2]),float(items[3]),'True']
        par_list.append(items[0])
    for i in range(len(rgh_instance_list)):
        rgh,rgh_name=rgh_instance_list[i],rgh_instance_name_list[i]
        for par in par_list:
            if hasattr(rgh,par):
                container=set_table_input_one_by_one_new(container,rgh,rgh_name,{par:par_range[par]})
        if sep:
            container.append(['','0','False','0','0','-'])
        else:
            pass
    if not sep:container.append(['','0','False','0','0','-'])
    return container

def set_table_from_a_list(container=[],head='Domain1.setFreezed_el_',el='Th',num_el=6,oc_range=[0,50],u_range=[0.5,50]):
    for i in range(num_el):
        container.append([head+el+'_'+str(i+1)+'_D1oc','0','True',str(oc_range[0]),str(oc_range[1]),'-'])
    container.append(['','0','False','0','0','-'])
    for i in range(num_el):
        container.append([head+el+'_'+str(i+1)+'_D1u','0.5','True',str(u_range[0]),str(u_range[1]),'-'])
    container.append(['','0','False','0','0','-'])
    return container

def make_table(container,file_path='D://tab.tab'):
    f=open(file_path,'w')
    f.write('#Parameter Value Fit Min Max Error\n')
    f.write('inst.set_inten\t1\tTrue\t0\t10\t-\n')
    for i in range(len(container)):
        f.write('\t'.join(container[i])+'\n')
    f.close()
