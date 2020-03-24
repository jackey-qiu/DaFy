import numpy as np
import matplotlib
import matplotlib as mpt
matplotlib.use('Agg')
import pylab
from pylab import imshow,show,get_cmap
import matplotlib.pyplot as pyplot
import copy
import pickle
from mpl_toolkits.axes_grid.inset_locator import inset_axes

f1=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))

def _find_shortest_length(coors=np.array([[0.292, 0.471, 7.69],[2.227, 3.188, 7.69],[1.724, 1.507, 7.681],[0.795, 4.224, 7.681]])):
    N=len(coors)
    dist=1000
    for i in range(N-1):
        for j in range(i+1,N):
            temp_dist=f1(coors[i],coors[j])
            if temp_dist<dist:
                dist=temp_dist
    return dist
#find distance for different possible shells
#cluster_range will define the members for each shell, eg. site with r=2.4A and r=2.6A will be assigned to the same shell if set the cluster_range higher than 0.2
#returns
#dist_box_unique is the sorted distance values considering 3by3 super grid(duplicated values has been deleted)
#seperators is a list of radius defining the shell size, it can be passed to dist_limit in the function cal_occ_random()

def _find_length_distribution(coors=np.array([[2.445,3.337,7.471],[0.071,0.62,7.471]]),basis=np.array([5.038,5.434,0]),cluster_range=0.5):
    super_grid=np.zeros((1,3))[0:0]
    dist_box=[]
    group={}
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for each_site in coors:
                super_grid=np.append(super_grid,each_site+basis[np.newaxis,]*[i,j,0],axis=0)
    for each_grid in super_grid:
        if f1(coors[0],each_grid)!=0:
            dist_box.append(f1(coors[0],each_grid))
        else:pass
    dist_box.sort()
    dist_box_unique=[round(dist_box[0],2)]
    for each in dist_box[1:]:
        if abs(each-dist_box_unique[-1])>0.01:
            dist_box_unique.append(round(each,2))
        else:
            pass
    temp_key=dist_box_unique[0]
    temp_group=[]
    for i in range(len(dist_box_unique)):
        if dist_box_unique[i]<=temp_key+cluster_range:
            if i!=len(dist_box_unique)-1:
                pass
            else:
                temp_group.append([temp_key,dist_box_unique[i]])
        else:
            temp_group.append([temp_key,dist_box_unique[i-1]])
            temp_key=dist_box_unique[i]
            if i==len(dist_box_unique)-1:
                temp_group.append([temp_key,dist_box_unique[i]])
    seperators=[round(np.min(each_set)-0.01,2) for each_set in temp_group]
    return dist_box_unique,seperators

#Pb on cmp-rcut hematite case: coors=np.array([[2.24191,3.146286,7.7466057],[0.27709,0.429286,7.7466057],[ 1.768338,1.488916,7.6507866],[ 0.745624,4.21135,7.6507866]]),basis=np.array([5.038,5.434,0]), cluster_range=0.3,sep_range=None,ax_handle=None
#Pb on cmp-rcut hematite case: seperators=[1.68,2.84,3.72,4.65,5.42,5.86,6.47,7.18,7.79,8.2,9.04,10.08]
#Sb on cmp-rcut hematite case:coors=np.array([[2.445,3.337,7.471],[0.071,0.62,7.471]])
#plot 5by5 super grid with different shells identified
def print_fancy_radical_plot(coors=np.array([[2.445,3.337,7.471],[0.071,0.62,7.471]]),basis=np.array([5.038,5.434,0]), cluster_range=0.3,sep_range=12,ax_handle=None):
    seperators=_find_length_distribution(coors,basis,cluster_range)[1]
    if sep_range!=None:
        seperators=_find_length_distribution(coors,basis,cluster_range)[1][0:sep_range]
    #seperators[0]=seperators[0]-0.5
    #seperator for Pb case
    #seperators=[2.84,3.72,4.65,5.42,5.86,6.47,7.18,7.79,8.2,9.04,10.08]
    seperator=[3.60,5.03,5.42,7.40,7.88,8.48,11.01]

    y_grids=np.array([-3,-2,-1,0,1,2,3,4])*basis[1]
    x_grids=np.array([-3,-2,-1,0,1,2,3,4])*basis[0]
    if ax_handle==None:
        fig=pyplot.figure(figsize=(8,5))
        ax=fig.add_subplot(111)
    else:
        ax=ax_handle
    ax.set_aspect(1)
    ax.set_yticks(y_grids,minor=False)
    ax.set_xticks(x_grids,minor=False)
    ax.yaxis.grid(True,linestyle=':',which='major',alpha=0.5)
    ax.xaxis.grid(True,linestyle=':',which='major',alpha=0.5)
    ax.set_xticklabels(range(8))
    ax.set_yticklabels(range(8))
    ax.set_xlim([min(x_grids),max(x_grids)])
    ax.set_ylim([min(y_grids),max(y_grids)])
    colors=['r','b','g','black']
    angles=np.arange(0,np.pi*2,np.pi*2/len(seperators))
    for i in range(-3,4):
        for j in range(-3,4):
            for k in range(len(coors)):
                each=coors[k]+basis*[i,j,0]
                ax.scatter([each[0]],[each[1]],marker='o',color=colors[k],s=6,alpha=0.4)
    for i in range(len(seperators)):
        sep=seperators[i]
        if sep==5.03:
            ax.add_artist(pyplot.Circle((coors[0][0],coors[0][1]),sep,color='m',fill=False,lw=0.8,ls='-'))
        else:
            ax.add_artist(pyplot.Circle((coors[0][0],coors[0][1]),sep,color='g',fill=False,lw=0.8,ls='-'))
        dx,dy=(seperators[-1]+2-sep)*np.cos(angles[i]),(seperators[-1]+2-sep)*np.sin(angles[i])
        dx2,dy2=(seperators[-1]+3-sep)*np.cos(angles[i]),(seperators[-1]+3-sep)*np.sin(angles[i])
        """
        ax.annotate(str(sep),
                  xy=(coors[0][0]+sep*cos(angles[i]),coors[0][1]+sep*np.sin(angles[i])), xycoords='data',
                  xytext=(coors[0][0]+sep*cos(angles[i])+dx,coors[0][1]+sep*np.sin(angles[i])+dy), textcoords='data',
                  size=20, va="center", ha="center",
                  arrowprops=dict(arrowstyle="<|-",
                                  connectionstyle="arc3,rad=0",
                                  fc="w"),
                  )
        """
        if sep==5.03:
            ax.add_patch(mpt.patches.FancyArrow(coors[0][0]+sep*np.cos(angles[i]),coors[0][1]+sep*np.sin(angles[i]),dx,dy,width=0.01,head_width=0.5,head_length=0.5,overhang=0,color='r',length_includes_head=True))
        else:
            ax.add_patch(mpt.patches.FancyArrow(coors[0][0]+sep*np.cos(angles[i]),coors[0][1]+sep*np.sin(angles[i]),dx,dy,width=0.01,head_width=0.5,head_length=0.5,overhang=0,color='black',length_includes_head=True))


        if angles[i]>np.pi:
            if sep==5.03:
                ax.text(coors[0][0]+sep*np.cos(angles[i])+dx2,coors[0][1]+sep*np.sin(angles[i])+dy2,str(sep)+r'$\AA$',fontsize=10,va='center',ha='center',rotation=angles[i]*180/np.pi+90,color='r')
            else:
                ax.text(coors[0][0]+sep*np.cos(angles[i])+dx2,coors[0][1]+sep*np.sin(angles[i])+dy2,str(sep)+r'$\AA$',fontsize=10,va='center',ha='center',rotation=angles[i]*180/np.pi+90)
        else:
            if sep==5.03:
                ax.text(coors[0][0]+sep*np.cos(angles[i])+dx2,coors[0][1]+sep*np.sin(angles[i])+dy2,str(sep)+r'$\AA$',fontsize=10,va='center',ha='center',rotation=angles[i]*180/np.pi-90,color='r')
            else:
                ax.text(coors[0][0]+sep*np.cos(angles[i])+dx2,coors[0][1]+sep*np.sin(angles[i])+dy2,str(sep)+r'$\AA$',fontsize=10,va='center',ha='center',rotation=angles[i]*180/np.pi-90)

    try:
        fig.savefig("D://temp_pic.png",dpi=300)
    except:
        pass
    return ax

#two sites for Sb: [2.445,3.337,7.471],[0.071,0.62,7.471](total site occ=72.5% normalized to HL based on CTR fitting results and occ=58% when normalized to full surface)
#four sites for Pb based on latest fit: [0.286, 0.519, 7.765],[2.233 ,3.238, 7.765],[0.79 ,4.121 ,7.612],[1.724, 1.397, 7.612] (first site type has occ=21% (28% normalized to HL), second site type has occ=26.1% (34.9% normalized to HL), cleanHL=27.7%, cleanFl=25.3%, with total site occ=63% (normalized to HL))
#four sites based on IS and OS:IS [2.21522e+00,3.21226e+00,7.68327e+00],[0.30383,4.95327e-01,7.68335e+00](31% normalized to HL only) and OS [0.79132,4.266,7.69153e+00],[1.72768e+00,1.54921e+00,7.69153e+00] (~100% after normalizing to FL only)
#this function was used to evaluate the occ of a sorbate, which could possibly bind at equivalent sites.
#the distribution of sorbate on the surface is based on random assignment, so that the sorbate-sorbate distance will be larger than
#the predefined minimum distance cut-off limit (distance_limit).
#asym_coors is the coordinates for symmetrical sites.
#site_limit is a number less than the total sites within each unit cell (see ~line 31), we use this number to avoid meaningless searching during the random assignment (higher this number, more searching time, but wont affect the simulation results)
#if you think there is no way for more than two sites (including 2) being occupied within each cell, then set this number to be 2
#probability is used to define the chance for each site being occupied,make sure it has the same length as the asym_coors, you can just use the associated occupancy to feed in the probability, never mess up the order

def cal_random_occ(grid_size=[20,20],asym_coors=np.array([[2.445,3.337,7.471],[0.071,0.62,7.471]]),probability=[1.,1.],basis=np.array([5.038,5.434,7.3707]),dist_limit=5.03,site_limit=2,snapshot=False):
    grid={}
    grid_container=[]
    N_sites=len(asym_coors)
    #shortest_dist=_find_shortest_length(asym_coors)
    shortest_dist=_find_length_distribution(asym_coors)[0][0]
    if len(probability)!=len(asym_coors):
        probability=[1]*len(asym_coors)
    probability_normalized=probability/np.sum(np.array(probability,dtype=float))
    #it will spit out a integer number (starting from 0) as the index specifying the binding site
    def _find_random_index(index_raw,probability=probability_normalized):
        probability_accumulated=[np.sum(probability[0:i+1]) for i in range(len(probability))]
        T_F_index=(np.array(probability_accumulated)-index_raw)>0
        #print index_raw,T_F_index,probability,np.where(T_F_index==True)
        return np.where(T_F_index==True)[0][0]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid[(i,j)]=[[False]*N_sites,asym_coors+basis*[i,j,0]]
    if shortest_dist>=dist_limit:
        for key in grid.keys():
            grid[key][0]=[True]*N_sites
        print "All sites are fully occupied"
        return grid,[grid],N_sites
    else:
        for trial in range(grid_size[0]*grid_size[1]*350):
            if trial%5000==0:print trial
            grid_rand,index=None,None
            while 1:
                grid_rand=(np.random.randint(0,grid_size[0]),np.random.randint(0,grid_size[1]))
                if sum(grid[grid_rand][0])<site_limit:#site limit can be any number less than the total number of sites within each cell
                    while 1:
                        rand_number=np.random.random()
                        index=_find_random_index(rand_number)
                        #index=int((rand_number-rand_number%(1./N_sites))/(1./N_sites))
                        if not grid[grid_rand][0][index]:#if not assigned yet
                            break
                    break
            def _generate_grid(grid_index):
                grid_container=[]
                offsets=np.array([[-1,-1],[1,1],[1,0],[-1,0],[0,1],[0,-1],[1,-1],[-1,1],[0,0]])
                grids_all=grid_index+offsets
                for grid in grids_all:
                    if grid[0]<grid_size[0] and grid[0]>=0 and grid[1]<grid_size[1] and grid[1]>=0:
                        grid_container.append(tuple(grid))
                return grid_container
            grids_for_test=_generate_grid(grid_rand)
            test_good='GOOD'
            for each_grid in grids_for_test:
                try:
                    index_site=np.where(np.array(grid[each_grid][0])==True)[0]
                    for each_index in index_site:
                        dist=f1(grid[each_grid][1][each_index],grid[grid_rand][1][index])
                        if dist<dist_limit:
                            test_good='BAD'
                            break
                    if test_good=='BAD':
                        break
                except:
                    pass
            if test_good=='GOOD':
                grid[grid_rand][0][index]=True
                if snapshot:grid_container.append(copy.deepcopy(grid))
                #print grid_rand
        total_sites=0
        for i in range(N_sites):vars()['site_'+chr(65+i)]=0
        for each in grid.keys():
            total_sites=total_sites+N_sites
            for i in range(N_sites):vars()['site_'+chr(65+i)]=vars()['site_'+chr(65+i)]+int(grid[each][0][i])
        #normalized to the total site number of each site.
        occupied_sites=0
        for i in range(N_sites):
            print "occupancy of site "+chr(65+i)+"=",float(vars()['site_'+chr(65+i)])/(total_sites/N_sites)
            occupied_sites=occupied_sites+float(vars()['site_'+chr(65+i)])
        print "total site occupancy=",occupied_sites/(total_sites/N_sites)
        return grid,grid_container,occupied_sites/(total_sites/N_sites)

#grid_container is a list of different snapshots(second term returned from the previous function) of grid being saved in the process of site assignment
#grid_index is the center index for the final output
#x y_range define the range the smart output will cover, dont sample indexes over boundary
#switch add_circle on to draw circles for sites with N_neighbor neighbor atoms within r A searching range
#The cut-off was used to calculate a average site distance within the cutoff range
#r is the searching range to calculate the sites with N_neighbor neighbors for drawing a circle

def smart_output(grid_container=[],grid_index=(10,10),x_range=[-10,10],y_range=[-10,10],add_circle=True,r=4.,cut_off=5,N_neighbor=7,ax_handle=None):
    N_sites=len(grid_container[0][grid_container[0].keys()[0]][0])
    for n in range(len(grid_container)):
        grid=grid_container[n]
        x_indexs=range(x_range[0],x_range[1])
        y_indexs=range(y_range[0],y_range[1])[::-1]
        x_grids=(grid_index[0]+np.array(x_indexs))*5.038
        y_grids=(grid_index[1]+np.array(y_indexs))*5.434
        fig=pyplot.figure(figsize=(8,5))
        ax=fig.add_subplot(111)
        if ax_handle==None:
            fig=pyplot.figure(figsize=(8,5))
            ax=fig.add_subplot(111)
        else:
            ax=ax_handle
        ax.set_aspect('equal')
        ax.set_yticks(y_grids,minor=False)
        ax.set_xticks(x_grids,minor=False)
        ax.yaxis.grid(True,linestyle=':',which='major',alpha=0.5)
        ax.xaxis.grid(True,linestyle=':',which='major',alpha=0.5)
        x_indexs_tick=[]
        for each in x_indexs:
            if not each%3:
                x_indexs_tick.append(each+grid_index[0])
            else:
                x_indexs_tick.append('')
        y_indexs_tick=[]
        for each in y_indexs:
            if not each%3:
                y_indexs_tick.append(each+grid_index[1])
            else:
                y_indexs_tick.append('')
        ax.set_xticklabels(np.array(x_indexs_tick))
        ax.set_yticklabels(np.array(y_indexs_tick))
        ax.set_xlim([0,max(x_grids)+5.038])
        ax.set_ylim([0,max(y_grids)+5.434])
        distance_container={}
        for i in range(N_sites):
            distance_container['site'+chr(65+i)]=[0,0,0]
        colors=['r','b','g','black']#should expand this items when you have more than four sites within each cell
        markers=['o','o','o','o']#and this also
        def _generate_grid(grid):
            grid_container=[]
            offsets=np.array([[-1,-1],[1,1],[1,0],[-1,0],[0,1],[0,-1],[1,-1],[-1,1],[0,0]])
            grids_all=grid+offsets
            for grid in grids_all:
                if grid[0]<(max(x_indexs)+grid_index[0]) and grid[0]>=0 and grid[1]<(max(y_indexs)+grid_index[1]) and grid[1]>=0:
                    grid_container.append(tuple(grid))
            return grid_container

        for y in y_indexs:
            for x in x_indexs:
                for i in range(N_sites):
                    if grid[tuple(np.array(grid_index)+(x,y))][0][i]:
                        ax.scatter([grid[tuple(np.array(grid_index)+(x,y))][1][i][0]],[grid[tuple(np.array(grid_index)+(x,y))][1][i][1]],marker=markers[i],color=colors[i],s=10,alpha=0.5)
                        if add_circle:
                            current_point=np.array(grid[tuple(np.array(grid_index)+(x,y))][1][i])
                            sub_grid=_generate_grid([x+grid_index[0],y+grid_index[1]])
                            coordinates=[]
                            distance_container_temp=[]
                            for each_grid in sub_grid:
                                spot_check=np.where(np.array(grid_container[n][tuple(each_grid)][0])==True)[0]
                                for each_check in spot_check:
                                    distance_temp=f1(grid_container[n][tuple(each_grid)][1][each_check][:-1],current_point[:-1])
                                    if distance_temp<r:
                                        coordinates.append(grid_container[n][tuple(each_grid)][1][each_check])
                                    if distance_temp<cut_off and distance_temp!=0:
                                        distance_container_temp.append(distance_temp)
                            distance_container['site'+chr(65+i)]=distance_container['site'+chr(65+i)]+np.array([int(distance_container_temp!=[]),len(distance_container_temp),sum(distance_container_temp)])
                            if len(coordinates)>=N_neighbor:
                                ax.add_artist(pyplot.Circle((grid[tuple(np.array(grid_index)+(x,y))][1][i][0],grid[tuple(np.array(grid_index)+(x,y))][1][i][1]),r,color=colors[i],fill=False,lw=4))
        try:
            pylab.savefig('test'+str(n)+'.png',bbox_inches='tight')
        except:
            pass
        print "Cut-off distance limit for circles=",r," A"
        print "Cut-off distance limit for site distance calculation=",cut_off," A\n"
        N_neighbors=0
        distance_accumulated=0
        for key in ['site'+chr(65+i) for i in range(N_sites)]:
            N_neighbors=N_neighbors+distance_container[key][1]
            distance_accumulated=distance_accumulated+distance_container[key][2]
            print "There are ",distance_container[key][0], " ",key, "being tested for the coordination chemistry"
            print key, " has ", distance_container[key][1]," neighbour sites in total with average site distance of ", distance_container[key][2]/distance_container[key][1], " A"
            print "The average coordination number for ",key," is", float(distance_container[key][1]/distance_container[key][0]),"\n"
        print "The average site distance is " ,  distance_accumulated/N_neighbors, "A"

    return distance_container
#for Pb -cmp case:coordinate_container=[[2.24191,3.146286,7.7466057],[0.27709,0.429286,7.7466057],[ 1.768338,1.488916,7.6507866],[ 0.745624,4.21135,7.6507866]],colors=['r','b','g','black'],x_range=[0,20],y_range=[0,20],roll_order=[1,3],direction='r',ax_handle=None,fill_all=False
def build_super_lattice(coordinate_container=[[2.445,3.337,7.471],[0.071,0.62,7.471]],colors=['r','b'],x_range=[0,20],y_range=[0,20],roll_order=[1,3],direction='r',ax_handle=None,fill_all=1):

    x_indexs=range(x_range[0],x_range[1])
    y_indexs=range(y_range[0],y_range[1])[::-1]
    x_grids=(np.array(x_indexs))*5.038
    y_grids=(np.array(y_indexs))*5.434
    if ax_handle==None:
        fig=pyplot.figure(figsize=(8,5))
        ax=fig.add_subplot(111)
    else:
        ax=ax_handle
    ax.set_aspect('equal')
    ax.set_yticks(y_grids,minor=False)
    ax.set_xticks(x_grids,minor=False)
    ax.yaxis.grid(True,linestyle=':',which='major',alpha=0.5)
    ax.xaxis.grid(True,linestyle=':',which='major',alpha=0.5)
    x_indexs_tick=[]
    for each in x_indexs:
        if each in [1,4,7,10,13,16,19]:
            x_indexs_tick.append(each)
        else:
            x_indexs_tick.append('')
    y_indexs_tick=[]
    for each in y_indexs:
        if each in [1,4,7,10,13,16,19]:
            y_indexs_tick.append(each)
        else:
            y_indexs_tick.append('')

    ax.set_xticklabels(np.array(x_indexs_tick))
    ax.set_yticklabels(np.array(y_indexs_tick))
    ax.set_xlim([0,max(x_grids)+5.038])
    ax.set_ylim([0,max(y_grids)+5.434])

    if not fill_all:
        for y in y_indexs:
            for x in x_indexs:
                if direction=='r':#in row direction
                    index=roll_order[x%len(roll_order)]
                    ax.scatter([coordinate_container[index][0]+x*5.038],[coordinate_container[index][1]+y*5.434],marker='o',color=colors[index],s=10)
                elif direction=='c':#in column direction
                    index=roll_order[y%len(roll_order)]
                    ax.scatter([coordinate_container[index][0]+x*5.038],[coordinate_container[index][1]+y*5.434],marker='o',color=colors[index],s=10)
    else:
        for y in y_indexs:
            for x in x_indexs:
                for i in range(len(coordinate_container)):
                    ax.scatter([coordinate_container[i][0]+x*5.038],[coordinate_container[i][1]+y*5.434],marker='o',color=colors[i],s=10,alpha=0.5)
    try:
        pylab.savefig('test.png',bbox_inches='tight')
    except:
        pass

    return True

def make_single_UC(coordinate_container=np.array([[2.24191,3.146286,7.7466057],[0.27709,0.429286,7.7466057],[ 1.768338,1.488916,7.6507866],[ 0.745624,4.21135,7.6507866]]),colors=['r','b','g','black'],ax_handle=None):
    hfont = {'fontname':['times new roman','Helvetica'][1]}
    if ax_handle==None:
        fig=pyplot.figure(figsize=(8,5))
        ax=fig.add_subplot(111)
    else:
        ax=ax_handle
    ax.set_aspect(1)
    #ax.set_yticks(y_grids,minor=False)
    #ax.set_xticks(x_grids,minor=False)
    #ax.yaxis.grid(True,linestyle='-',which='major')
    #ax.xaxis.grid(True,linestyle='-',which='major')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_xlim([0,5.038])
    ax.set_ylim([0,5.434])
    ax.plot([coordinate_container[0][0],coordinate_container[1][0]],[coordinate_container[0][1],coordinate_container[1][1]],ls=':',color='black')
    es1=ax.annotate(r'ES1',xy=((coordinate_container[0][0]+coordinate_container[1][0])/2.-0.65,(coordinate_container[0][1]+coordinate_container[1][1])/2.))
    es1.set_rotation(50)
    ax.plot([coordinate_container[2][0],coordinate_container[3][0]],[coordinate_container[2][1],coordinate_container[3][1]],ls=':',color='black')
    es2=ax.annotate(r'ES2',xy=((coordinate_container[2][0]+coordinate_container[3][0])/2.-0.2,(coordinate_container[2][1]+coordinate_container[3][1])/2.+0.5))
    es2.set_rotation(-70)
    for i in range(len(coordinate_container)):
        ax.scatter([coordinate_container[i][0]],[coordinate_container[i][1]],marker='o',color=colors[i],s=80)
    ax.set_xlabel(r'a($\rm{\AA}$)',fontsize=12)
    ax.set_ylabel(r'b($\rm{\AA}$)',fontsize=12)
    try:
        pylab.savefig('test.png',bbox_inches='tight')
    except:
        pass
    return True

#xlabels=np.arange(2.0,7.0,0.2)
coverage_finer=[[2, 2, 2, 2, 2, 2, 2, 1.76, 1.8, 1.75, 1.75, 1.475, 1.495, 1.605, 1.55, 1.57, 0.99, 0.995, 0.775, 0.75, 0.755, 0.785, 0.815, 0.78, 0.78], [2, 2, 2, 2, 2, 1.745, 1.745, 1.755, 1.75, 1.76, 1.775, 1.76, 1.765, 1.74, 1.745, 1.535, 0.99, 0.985, 0.79, 0.77, 0.81, 0.81, 0.76, 0.69, 0.66], [1.75, 1.755, 1.7425, 1.7475, 1.76, 1.5325, 1.5475, 1.1225, 1.1225, 1.0275, 0.8925, 0.84, 0.8475, 0.8475, 0.7775, 0.6825, 0.6025, 0.57, 0.515, 0.4425, 0.4175, 0.41, 0.415, 0.3725, 0.3575]]
#xlabels=np.arange(2.0,10.0,0.5))
coverage_medium_fine=[[2, 2, 2, 1.765, 1.765, 1.495, 1.59, 0.805, 0.77, 0.765, 0.785, 0.495, 0.51, 0.455, 0.44, 0.435], [2, 2, 1.775, 1.77, 1.725, 1.815, 1.59, 0.795, 0.815, 0.825, 0.65, 0.505, 0.505, 0.445, 0.48, 0.425], [1.745, 1.755, 1.535, 1.1275, 0.8825, 0.84, 0.6875, 0.525, 0.4125, 0.4175, 0.345, 0.27, 0.2775, 0.2675, 0.2425, 0.225]]
#xlabels=np.arange(2,6,0.5)
coverage_coaser=[[2, 2, 2, 1.75, 1.755, 1.515, 1.44, 0.79], [2, 2, 1.755, 1.76, 1.76, 1.765, 1.51, 0.82], [1.7425, 1.7425, 1.5475, 1.15, 0.885, 0.86, 0.6775, 0.52]]
#for sb case xlabels=np.arange(3.5,10,0.5)
coverage_sb=[[2, 1.555/2, 1.545/2, 1.515/2, 0.79/2, 0.79/2, 0.785/2, 0.785/2,0.76/2, 0.6/2, 0.5/2, 0.49/2, 0.44/2]]
#for sb case xlabels=[3.6, 5.03, 5.42, 7.4, 7.88, 8.48, 11.01](calculated from _find_length_distribution())
xlabels_sb_2=[3.60, 5.03, 5.42, 7.40, 7.88, 8.48, 11.01]
coverage_sb_2=[[2, 0.7625, 0.485, 0.3975, 0.3125, 0.2625, 0.215]]
#based on the new run with two edge-sharing sites on HL (not OS on FL)
occ_pb_new=[[1.75, 1.1125, 0.84, 0.5775, 0.4425, 0.42, 0.3475, 0.2725, 0.2575, 0.2475, 0.225]]
dist_pb_new=[2.84,3.72,4.65,5.42,5.86,6.47,7.18,7.79,8.2,9.04,10.08]
coors_pb_new=np.array([[2.18,3.25,7.74],[0.337,0.534,7.74],[1.703,1.513,7.66],[0.81,4.24,7.66]])
def bar_plot(coverage=coverage_sb_2,colors=['b','r','g'],labels=['Half layer termination','Full layer termination','Mixture of half and full layer termination'],xlabels=xlabels_sb_2,ax_handle=None):
    #make a bar plot of coverage at different cutoff limits for different termination patterns
    if ax_handle==None:
        fig=pyplot.figure(figsize=(8,5))
        ax=fig.add_subplot(111)
    else:
        ax=ax_handle
    for i in range(len(coverage)):
        x=np.array(range(len(coverage[i])))+0.8/len(coverage)*i
        rec=ax.bar(x,np.array(coverage[i]),width=0.8/len(coverage),color=colors[i],label=labels[i],align='center',alpha=0.8)
        #p=pyplot.plot(x,np.array(coverage[i])*100+2,'--d'+colors[i],lw=2,markersize=7)
        #ax.legend()
    l2,=ax.plot([-1,100],[0.73,0.73],'r',lw=3)
    #l2,=ax.plot([-1,100],[0.59,0.59],'r',lw=3)
    #ax.annotate(r'0.59 Pb/A$\rm{_U}$$\rm{_C}$ based on bestfit model',xy=(3.5,.67))
    ax.annotate(r'0.73 Sb/A$\rm{_U}$$\rm{_C}$ based on bestfit model',xy=(1.5,0.78))
    #l3,=ax.plot([-1,100],[50,50],'g',lw=3)
    for l in [l2]:
        l.set_dashes((6,6))
    ax.set_xlim(-0.8,len(coverage[0]))
    ax.set_ylim(0,2)
    ax.set_ylabel(r'Simulated site coverage (Sb/A$\rm{_U}$$\rm{_C}$)',fontsize=12)
    ax.set_xlabel(r'Cutoff Sb-Sb distance ($\rm{\AA}$)',fontsize=12)
    ax.set_xticks(np.array(range(len(coverage[0])))+0.4-0.4/len(coverage))
    xtick=ax.set_xticklabels(xlabels)
    #pyplot.setp(xtick,fontsize=10)
    #ax.set_aspect((len(coverage[0])+0.8)/200)
    try:
        fig.savefig("D://temp_pic.png",dpi=300)
    except:
        pass
    pyplot.show()

    return ax


def batch_process(asym_coors=[np.array([[2.21522e+00,3.21226e+00,7.68327e+00],[0.30383,4.95327e-01,7.68335e+00]]),np.array([[0.79132,4.266,7.69153e+00],[1.72768e+00,1.54921e+00,7.69153e+00]]),np.array([[2.21522e+00,3.21226e+00,7.68327e+00],[0.30383,4.95327e-01,7.68335e+00],[0.79132,4.266,7.69153e+00],[1.72768e+00,1.54921e+00,7.69153e+00]])],dist_limits=np.arange(2.0,6.0,0.5),probability=[1,1]):
    #process cal_random_occ in batch with specified sequence of asym_coors (three items for HL, FL and HL&FL respectively) and dist_limit
    coverage=[]
    for each_asym in asym_coors:
        coverage.append([])
        for dist_limit in dist_limits:
            a,b,occ=cal_random_occ(asym_coors=each_asym,dist_limit=dist_limit,probability=probability)
            coverage[-1].append(occ)
    bar_plot(coverage,xlabels=dist_limits)
    return coverage

def create_xyz_super_grid(grid,el='Sb',file="D://super_grid.xyz",distal_oxygens=np.array([[[1.56,1.59,8.11],[2.95,3.53,9.35],[0.69,3.96,8.35]],[[0.962,-1.124,8.11],[-0.426,0.813,9.35],[1.828,1.24,8.35]]]),x_range=[0,5],y_range=[0,5],basis=np.array([5.038,5.434,7.3707])):
    f=open(file,'w')
    N=0
    for key in grid.keys():
        if key[0]>=x_range[0] and key[0]<=x_range[1] and key[1]>=y_range[0] and key[1]<=y_range[1]:
            for i in range(len(grid[key][0])):
                if grid[key][0][i]==True:
                    N=N+1+len(distal_oxygens[0])
    f.write(str(N)+'\n#\n')
    M=0
    ref=N/(1+len(distal_oxygens[0]))
    for key in grid.keys():
        if key[0]>=x_range[0] and key[0]<=x_range[1] and key[1]>=y_range[0] and key[1]<=y_range[1]:
            xyz_offset=list(key)
            xyz_offset.append(0)
            xyz_offset=xyz_offset*basis
            for i in range(len(grid[key][0])):
                if grid[key][0][i]==True:
                    M=M+1
                    f.write('%-5s   %7.5e   %7.5e   %7.5e\n' % (el,grid[key][1][i][0],grid[key][1][i][1],grid[key][1][i][2]))
                    for j in range(len(distal_oxygens[i])):
                        distal_xyz=distal_oxygens[i][j]+xyz_offset
                        if j==len(distal_oxygens[i])-1 and M==ref:
                            f.write('%-5s   %7.5e   %7.5e   %7.5e' % ('O',distal_xyz[0],distal_xyz[1],distal_xyz[2]))
                        else:
                            f.write('%-5s   %7.5e   %7.5e   %7.5e\n' % ('O',distal_xyz[0],distal_xyz[1],distal_xyz[2]))
    f.close()
#for Pb-cmp case
def plot_one_fullset(f_shell=print_fancy_radical_plot,f_bar_plot=bar_plot,f_uc=make_single_UC,f_snapshot=smart_output,f_super_lattice=build_super_lattice,grid_file='P:\\My stuff\\Scripts\\grid.p'):
    grid=[pickle.load(open(grid_file,'rb'))]
    fig=pyplot.figure(figsize=(10,8))
    #ax_shell=fig.add_subplot(2,2,1)
    #ax_bar=fig.add_subplot(2,2,2)
    #ax_snapshot=fig.add_subplot(2,2,3)
    #ax_super=fig.add_subplot(2,2,4)
    ax_bar=pyplot.subplot2grid((2, 3), (0, 1), colspan=2)
    ax_shell=fig.add_subplot(2,3,1)
    #ax_bar=fig.add_subplot(2,2,1)
    ax_snapshot=fig.add_subplot(2,3,5)
    ax_super=fig.add_subplot(2,3,6)
    ax_UC=fig.add_subplot(2,3,4)
    f_shell(ax_handle=ax_shell)
    f_bar_plot(ax_handle=ax_bar)
    f_uc(ax_handle=ax_UC)
    f_snapshot(grid_container=grid,ax_handle=ax_snapshot)
    f_super_lattice(ax_handle=ax_super)
    fig.tight_layout()
    fig.savefig("D://temp_pic.png",dpi=300)
    return fig
#for Sb-cmp case
def plot_one_fullset_Sb(f_shell=print_fancy_radical_plot,f_bar_plot=bar_plot,f_uc=make_single_UC,f_snapshot=smart_output,f_super_lattice=build_super_lattice,grid_file='P:\\My stuff\\Scripts\\grid_sb.p'):
    grid=[pickle.load(open(grid_file,'rb'))]
    fig=pyplot.figure(figsize=(10,8))

    ax_bar=pyplot.subplot2grid((2, 3), (0, 0), colspan=2)
    ax_shell=fig.add_subplot(2,3,4)

    ax_snapshot=fig.add_subplot(2,3,5)
    ax_super=fig.add_subplot(2,3,6)

    f_shell(ax_handle=ax_shell)
    f_bar_plot(ax_handle=ax_bar)

    f_snapshot(grid_container=grid,ax_handle=ax_snapshot)
    f_super_lattice(ax_handle=ax_super)

    fig.savefig("D://temp_pic.png",dpi=300)
    return fig


def plot_one_fullset_Sb2(f_shell=print_fancy_radical_plot,f_bar_plot=bar_plot,f_uc=make_single_UC,f_snapshot=smart_output,f_super_lattice=build_super_lattice,grid_file='P:\\My stuff\\Scripts\\grid_sb.p'):
    grid=[pickle.load(open(grid_file,'rb'))]
    fig=pyplot.figure(figsize=(10,4))

    ax_bar=pyplot.subplot2grid((1, 3), (0, 0), colspan=2)
    #ax_shell=fig.add_subplot(2,3,4)

    ax_snapshot=fig.add_subplot(1,3,3)
    #ax_super=fig.add_subplot(2,3,6)

    #f_shell(ax_handle=ax_shell)
    f_bar_plot(ax_handle=ax_bar)

    f_snapshot(grid_container=grid,ax_handle=ax_snapshot)
    #f_super_lattice(ax_handle=ax_super)
    fig.tight_layout()
    fig.savefig("D://temp_pic.png",dpi=300)
    return fig


if __name__=="__main__":
    from images2gif import writeGif
    from PIL import Image
    import os

    #smart_output(cal_random_occ()[1][0:100])
    #file names in order
    file_names=['test'+str(i)+'.png' for i in range(100)]
    images = [Image.open(fn) for fn in file_names]
    filename = "simulation_of_site_occupancy.GIF"
    #write a gif file from a collection of image files
    writeGif(filename, images, duration=0.2)
