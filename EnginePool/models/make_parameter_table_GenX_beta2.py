import numpy as num
import numpy as np
#domain1A and domain1B are sequence atom list for half layer, the other two for full layer
domain1A=["O1_0","O2_0","O3_0","O4_0","Fe4_0","Fe6_0","O5_0","O6_0","O7_0","O8_0","Fe8_0","Fe9_0","O9_0","O10_0"]
domain1B=["O7_0","O8_0","O9_0","O10_0","Fe10_0","Fe12_0","O11_0","O12_0","O1_1","O2_1","Fe2_1","Fe3_1","O3_1","O4_1"]
#for old ref full layer domain
domain2A_long=["O11_t","O12_t","O1_0","O2_0","Fe2_0","Fe3_0","O3_0","O4_0","Fe4_0","Fe6_0","O5_0","O6_0","O7_0","O8_0"]
domain2B_long=["O5_0","O6_0","O7_0","O8_0","Fe8_0","Fe9_0","O9_0","O10_0","Fe10_0","Fe12_0","O11_0","O12_0","O1_1","O2_1"]
#for new ref full layer domain
domain2A_short=["O5_0","O6_0","O7_0","O8_0","Fe8_0","Fe9_0","O9_0","O10_0","Fe10_0","Fe12_0","O11_0","O12_0","O1_1","O2_1"]
domain2B_short=["O11_0","O12_0","O1_1","O2_1","Fe2_1","Fe3_1","O3_1","O4_1","Fe4_1","Fe6_1","O5_1","O6_1","O7_1","O8_1"]
"""
################explanation of each componont in structure################
#surface1#
'dxdy'=[8,[0.,-0.05,0.05],'True']:consider the top 8 atoms for inplane movement,start value 0,min -0.05 and max 0.05
'dz'=[14,[0,-0.05,0.05],'True']:consider the top 7 layers for outof plane movement (NOTE here consider in pairs)
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
def make_structure(sorbate_N,O_N,water_N,Domains,Metal,binding_mode=['BD']*3,long_slab=False,local_structure='tetrahedral',add_distal_wild=None,use_domains=[1]*10):
    structure={}
    for i in range(len(Domains)):
        if use_domains[i]==1:
            oc_N=6
            #if sorbate_N[i]==0:oc_N=6
            temp_surface={'dxdy':[0,[0.,-0.05,0.05],'True'],'dz':[10,[0,-0.05,0.05],'True'],'oc':[oc_N,[1,0.6,1],'True'],'u':[0,[0.4,0.32,0.8],'True']}
            domain_type='half_layer'
            full_layer_type='long'
            if Domains[i]==2:
                domain_type='full_layer'
                if long_slab==False:
                    full_layer_type='short'
                else:
                    if long_slab[i]==0:
                        full_layer_type='short'
                    elif long_slab[i]==1:
                        full_layer_type='long'
            else:
                full_layer_type='None'
            temp_sorbate={}
            if sorbate_N[i]==0:
                temp_sorbate=None
            else:
                temp_sorbate['dxdydz']=[[Metal+'_set'+str(2*j+1),[0,-0.5,0.5],'True'] for j in range(sorbate_N[i]/2)]+[['HO'+str(k+1)+'_set'+str(2*j+1),[0,-0.5,0.5],'True'] for j in range(len(O_N[i])) for k in range(O_N[i][j])]          
                temp_sorbate['u']=[[Metal+'_set'+str(2*j+1),[0.2,0.2,1.],'True'] for j in range(sorbate_N[i]/2)]+[['HO'+str(k+1)+'_set'+str(2*j+1),[0.2,0.2,1.],'True'] for j in range(len(O_N[i])) for k in range(O_N[i][j])]
                temp_sorbate['oc']=[[Metal+'_set'+str(2*j+1),[0.3,0.,1.],'True'] for j in range(sorbate_N[i]/2)]+[['HO'+str(k+1)+'_set'+str(2*j+1),[0.3,0.,1.],'True'] for j in range(len(O_N[i])) for k in range(O_N[i][j])]
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
            structure['domain'+str(i+1)]={'binding_mode':binding_mode[i],'full_layer_type':full_layer_type,'domain_type':domain_type,'domain_tag':i+1,'surface':temp_surface,'sorbate':temp_sorbate,'water':temp_water}
        else:
            pass
    return table_maker(structure_info=structure,long_slab=long_slab,local_structure=local_structure)    

def table_maker(table_file_path='D:\\table.tab',structure_info=structure,long_slab=False,local_structure='tetrahedral'):

    f=open(table_file_path,'w')
    f.write('#Parameter Value Fit Min Max Error\n')
    f.write('inst.set_inten\t1\tTrue\t1\t4\t-\n')
    f.write('rgh.setScale_CTR_specular\t1\tTrue\t0.6\t1.5\t-\n')
    f.write('rgh.setBeta\t0\tTrue\t0\t0.4\t-\n')
    f.write('\t0\tFalse\t0\t0\t-\n')
    domain_N=len(structure_info.keys())
    for i in range(domain_N):
        f.write('rgh_domain'+str(i+1)+'.setWt\t1\tFalse\t0\t1\t-\n')
    f.write('\t0\tFalse\t0\t0\t-\n')
    keys=structure_info.keys()
    keys.sort()
    for key in keys:
        temp_domain=structure_info[key]
        domain_tag=str(structure_info[key]['domain_tag'])
        domain_type=temp_domain['domain_type']
        temp_surface=temp_domain['surface']
        temp_sorbate=temp_domain['sorbate']
        temp_water=temp_domain['water']
        atom_list=[]
        f.write('#############'+key+'############\n')
        if domain_type=='half_layer':
            atom_list=[domain1A,domain1B]
            f.write('gp_Fe2Fe3_Fe8Fe9_D'+domain_tag+'.setoc\t0\tFalse\t0\t0\t-\n')
            f.write('\t0\tFalse\t0\t0\t-\n')
        elif domain_type=='full_layer':
            if temp_domain['full_layer_type']=="short":
                atom_list=[domain2A_short,domain2B_short]
            elif temp_domain['full_layer_type']=='long':
                atom_list=[domain2A_long,domain2B_long]
        ##do surface
        #out of plane movement together with inplane movement ie dxdydz
        f.write('#dxdydz\n')
        for i in range(temp_surface['dz'][0]/2):
            index=2*i
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[0][index].rsplit('_')[0]+atom_list[0][index+1].rsplit('_')[0]+'_'+atom_list[1][index].rsplit('_')[0]+atom_list[1][index+1].rsplit('_')[0]+'_D'+domain_tag+'.setdx',\
                                                  temp_surface['dz'][1][0],temp_surface['dz'][2],temp_surface['dz'][1][1],temp_surface['dz'][1][2],'-')
            f.write(s)
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[0][index].rsplit('_')[0]+atom_list[0][index+1].rsplit('_')[0]+'_'+atom_list[1][index].rsplit('_')[0]+atom_list[1][index+1].rsplit('_')[0]+'_D'+domain_tag+'.setdy',\
                                                  temp_surface['dz'][1][0],temp_surface['dz'][2],temp_surface['dz'][1][1],temp_surface['dz'][1][2],'-')
            f.write(s)
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[0][index].rsplit('_')[0]+atom_list[0][index+1].rsplit('_')[0]+'_'+atom_list[1][index].rsplit('_')[0]+atom_list[1][index+1].rsplit('_')[0]+'_D'+domain_tag+'.setdz',\
                                                  temp_surface['dz'][1][0],temp_surface['dz'][2],temp_surface['dz'][1][1],temp_surface['dz'][1][2],'-')
            f.write(s)
        f.write('\t0\tFalse\t0\t0\t-\n')
        f.write('#oc and u\n')
        #occupancy and thermal factor
        for i in range(temp_surface['u'][0]/2):
            index=2*i
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[0][index].rsplit('_')[0]+atom_list[0][index+1].rsplit('_')[0]+'_'+atom_list[1][index].rsplit('_')[0]+atom_list[1][index+1].rsplit('_')[0]+'_D'+domain_tag+'.setu',\
                                                  temp_surface['u'][1][0],temp_surface['u'][2],temp_surface['u'][1][1],temp_surface['u'][1][2],'-')
            f.write(s)
        f.write('\t0\tFalse\t0\t0\t-\n')
        for i in range(temp_surface['oc'][0]/2):
            index=2*i
            s="%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('gp_'+atom_list[0][index].rsplit('_')[0]+atom_list[0][index+1].rsplit('_')[0]+'_'+atom_list[1][index].rsplit('_')[0]+atom_list[1][index+1].rsplit('_')[0]+'_D'+domain_tag+'.setoc',\
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
                    if local_structure=='trigonal_pyramid':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_BD'+tag,60,'True',50,90,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setOffset_BD'+tag,0,'True',-0.5,0.5,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR_BD'+tag,2.25,'True',2.25,2.35,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_1_BD'+tag,0.,'True',0.,180.,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_1_BD'+tag,0.,'True',0.,360.,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_1_BD'+tag,2.25,'True',1.9,2.35,'-'))
                        else:
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setOffset2_BD'+tag,0,'True',-0.5,0.5,'-'))
                            f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setAngle_offset_BD'+tag,0,'True',-0.5,0.5,'-'))
                    elif local_structure=='octahedral':
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_BD'+tag,0.,'True',0.,180.,'-')) for ii in range(4)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_BD'+tag,0.,'True',0.,360.,'-')) for ii in range(4)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_BD'+tag,2.26,'True',1.9,2.4,'-')) for ii in range(4)]
                    elif local_structure=='tetrahedral':
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
                    if local_structure=='trigonal_pyramid':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_TD'+tag,90.,'True',50.,110.,'-'))
                    elif local_structure=='octahedral':
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_TD'+tag,0.,'True',0.,180.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_TD'+tag,0.,'True',0.,360.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_TD'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(3)]
                    elif local_structure=='tetrahedral':
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
                    if local_structure=='trigonal_pyramid':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTop_angle_OS'+tag,90.,'True',50.,110.,'-'))
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR0_OS'+tag,2.25,'True',2.0,2.5,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,180.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,360.,'-')) for ii in range(3)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_OS'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(3)]
                    elif local_structure=='octahedral':
                        f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR0_OS'+tag,2.0,'True',1.8,2.2,'-'))
                        if temp_sorbate['distal_wild'][i*2]:
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setTheta1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,180.,'-')) for ii in range(6)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setPhi1_'+str(ii+1)+'_OS'+tag,0.,'True',0.,360.,'-')) for ii in range(6)]
                            [f.write("%s\t%5.4f\t%s\t%5.4f\t%5.4f\t%s\n"%('rgh_domain'+domain_tag+'.setR1_'+str(ii+1)+'_OS'+tag,2.0,'True',1.8,2.2,'-')) for ii in range(6)]
                    elif local_structure=='tetrahedral':
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
    f.close()