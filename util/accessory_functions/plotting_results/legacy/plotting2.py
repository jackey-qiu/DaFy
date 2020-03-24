import numpy as num
import numpy as np
from matplotlib import pyplot
import matplotlib as mpt
import pickle

#object_container=[[[ctr_00_g,ctr_00_g],[ctr_00_M,ctr_00_M],[ctr_00_i,ctr_00_i]],[[ctr_0m2_g,ctr_02_g],[ctr_0m2_M,ctr_02_M],[ctr_0m2_i,ctr_02_i]],[[ctr_m10_g,ctr_10_g],[ctr_m10_M,ctr_10_M],[ctr_m10_i,ctr_10_i]],[[ctr_m1m1_g,ctr_11_g],[ctr_m1m1_M,ctr_11_M],[ctr_m1m1_i,ctr_11_i]],[[ctr_m20_g,ctr_20_g],[ctr_m20_M,ctr_20_M],[ctr_m20_i,ctr_20_i]],[[ctr_m2m2_g,ctr_22_g],[ctr_m2m2_M,ctr_22_M],[ctr_m2m2_i,ctr_22_i]]]
#object_container2=[[[ctr_00_g,ctr_00_g],[ctr_00_k,ctr_00_k],[ctr_00_M,ctr_00_M],[ctr_00_i,ctr_00_i]],[[ctr_0m2_g,ctr_02_g],[ctr_0m2_k,ctr_02_k],[ctr_0m2_M,ctr_02_M],[ctr_0m2_i,ctr_02_i]],[[ctr_m10_g,ctr_10_g],[ctr_m10_k,ctr_10_k],[ctr_m10_M,ctr_10_M],[ctr_m10_i,ctr_10_i]],[[ctr_m1m1_g,ctr_11_g],[ctr_m1m1_k,ctr_11_k],[ctr_m1m1_M,ctr_11_M],[ctr_m1m1_i,ctr_11_i]],[[ctr_m20_g,ctr_20_g],[ctr_m20_k,ctr_20_k],[ctr_m20_M,ctr_20_M],[ctr_m20_i,ctr_20_i]],[[ctr_m2m2_g,ctr_22_g],[ctr_m2m2_k,ctr_22_k],[ctr_m2m2_M,ctr_22_M],[ctr_m2m2_i,ctr_22_i]]]
#object_container=[[[ctr_00_g],[ctr_00_R],[ctr_00_N],[ctr_00_P]],[[ctr_0m2_g,ctr_02_g],[ctr_0m2_R,ctr_02_R],[ctr_0m2_N,ctr_02_N],[ctr_0m2_P,ctr_02_P]],[[ctr_m10_g,ctr_10_g],[ctr_m10_R,ctr_10_R],[ctr_m10_N,ctr_10_N],[ctr_m10_P,ctr_10_P]],[[ctr_m1m1_g,ctr_11_g],[ctr_m1m1_R,ctr_11_R],[ctr_m1m1_N,ctr_11_N],[ctr_m1m1_P,ctr_11_P]],[[ctr_m20_g,ctr_20_g],[ctr_m20_R,ctr_20_R],[ctr_m20_N,ctr_20_N],[ctr_m20_P,ctr_20_P]],[[ctr_m2m2_g,ctr_22_g],[ctr_m2m2_R,ctr_22_R],[ctr_m2m2_N,ctr_22_N],[ctr_m2m2_P,ctr_22_P]]]
def extract_data(object=[],hk=[]):
#this function will extract data from the full ctr dataset determined by HK
#object is a list of the full ctr datasets (eg different concentration levels),hk=[1,0]
#if hk=[1,0], this function will extract both 10L and -10L
    data_container=[]
    for i in object:
        temp=num.array([[0,0,0,0,0]])
        for j in range(len(i.H)):
            #print 'sensor'
            if (i.H[j]==0)&(i.K[j]==0):
                temp=num.append(temp,[[i.H[j],i.K[j],i.L[j],i.F[j],i.Ferr[j]]],axis=0)
            else:
                if ((i.H[j]==hk[0])&(i.K[j]==hk[1])):
                    #print 'write'
                    temp=num.append(temp,[[i.H[j],i.K[j],i.L[j],i.F[j],i.Ferr[j]]],axis=0)
                elif ((i.H[j]==-hk[0])&(i.K[j]==-hk[1])):
                    #print 'wirte'
                    temp=num.append(temp,[[-i.H[j],-i.K[j],-i.L[j],i.F[j],i.Ferr[j]]],axis=0)
        data_container.append(temp)
    return data_container
    
def plotting(object=[],fig=None,index_subplot=[2,3,1],color=['b'],label=['clean surface'],title=['10L'],marker=['o'],legend=True,fontsize=10,markersize=10):
    #this function will overlap a specific CTR profiles (one HKL set) at different conditions (like concentrations)
    #object=[[ctr_m10_conc1,ctr_10_conc1],[ctr_m10_conc2,ctr_10_conc2]] or object=[[ctr_10_conc1],[ctr_10_conc2]]
    ax=fig.add_subplot(index_subplot[0],index_subplot[1],index_subplot[2])
    ax.set_yscale('log')
    #handles=[]
    for ob in object:
        index=object.index(ob)
        if len(ob)==2:
            error1=ax.errorbar(num.append(-ob[0].L,ob[1].L),num.append(ob[0].F,ob[1].F),yerr=num.append(ob[0].Ferr,ob[1].Ferr),label=label[index],marker=marker[index],ecolor=color[index],color=color[index],markerfacecolor=color[index],linestyle='None',markersize=markersize)
            #handles.append(error1)
        else:
            error1=ax.errorbar(ob[0].L,ob[0].F,yerr=ob[0].Ferr,marker=marker[index],label=label[index],ecolor=color[index],color=color[index],markerfacecolor=color[index],linestyle='None',markersize=markersize)
            #handles.append(error1)
        if index_subplot[2]>index_subplot[1]*(index_subplot[0]-1):
            pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        if (index_subplot[2]-1)%index_subplot[1]==0:
            pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
        pyplot.title(title[0],position=(0.5,0.85),weight='bold',clip_on=True)
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)
    if legend==True:
        ax.legend(bbox_to_anchor=(0.5,0.92,0,3),bbox_transform=fig.transFigure,loc='lower center',ncol=4,borderaxespad=0.,prop={'size':14})
        #ax.legend(bbox_to_anchor=(0.5,1.02,1.,1.102),loc=3,ncol=1,mode='expand',borderaxespad=0.,prop={'size':fontsize})
        #fig.legend(handles,('clean surface','200 uM Pb(II) reacted surface'),loc='upper center',ncol=2,mode=None)
    return True
def plotting_many(object=[],fig=None,index=[2,3],color=['b','r','g'],label=['clean','300uM','1mM'],marker=['o','p','D'],title=['00L','02L','10L','11L','20L','22L'],legend=[True,True,True,False,False,False],fontsize=10,markersize=10):
    #do several overlapping simultaneously
    #object is a container of object that defined in the previous function
    for ob in object:
        order=object.index(ob)
        plotting(object=ob,fig=fig,index_subplot=[index[0],index[1],order+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize,markersize=markersize)
        
def plotting_raxs(object=[],fig=None,index=[2,3,1],color=['b','r'],label=['50 uM','200uM'],title=['10L'],marker=['o','p'],legend=True,xlabel=False,position=(0.5,1.05)):
    #overlapping raxs data
    #object=[raxs10_conc1,raxs10_con2]
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    ends=[min(object[0][:,4])-5,max(object[1][:,4])+50]
    for ob in range(len(object)):
        index=ob
        ax.errorbar(object[index][:,0]/1000,object[index][:,4],yerr=object[index][:,5],marker=marker[index],label=label[index],ecolor=color[index],color=color[index],markerfacecolor=color[index],markeredgecolor=color[index],linestyle='None')
        pyplot.xlabel('Energy (kev)',axes=ax,fontsize=12)
        pyplot.ylabel('|F|',axes=ax,fontsize=12)
        pyplot.title(title[0],position=position,weight='bold',clip_on=True)
        xlabels = ax.get_xticklabels()
        for tick in xlabels:
            tick.set_rotation(30)
        if xlabel==False:
            ax.set_xticklabels([])
            pyplot.xlabel('')
        #ax.yaxis.set_major_locator(mpt.ticker.MaxNLocator())
        #print ax.get_yticks()
        #print ax.get_yticklabels()
    #ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    pyplot.ylim((ends[0]-10,ends[1]+50))
    step=int((ends[1]-ends[0])/5.)
    ax.set_yticks(list(num.arange(int(ends[0]),int(ends[1]),step)))
    ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    if legend==True:
        #ax.legend(bbox_to_anchor=(0.,0.95,1.,1.102),loc=3,ncol=1,mode='expand',borderaxespad=0.,prop={'size':13})
        ax.legend(bbox_to_anchor=(0.15,1.04,1.,1.102),loc=3,ncol=2,borderaxespad=0.,prop={'size':13})
def plotting_many_raxs(object=[],fig=None,index=[2,2,[1,2,3,4]],color=['0.5','0.3'],label=['50 uM Pb(II) reacted surface','200uM Pb(II) reacted surface'],marker=['p','*'],title=['RAXS_00_1.45','RAXS_00_2.77','RAXS_10_1.1','RAXS_-10_1.27'],legend=[True,False,False,False],xlabel=[False,False,True,True],position=[(0.5,0.85),(0.5,0.85),(0.5,1.05),(0.5,1.05)]):
    #plotting several raxs together
    for i in range(len(object)):
        order=i
        print 'abc'
        ob=object[i]
        plotting_raxs(object=ob,fig=fig,index=[index[0],index[1],index[2][order]],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],xlabel=xlabel[i],position=position[i])
        
def plotting_raxr_new(data,savefile="D://raxr_temp.png",color=['b','r'],marker=['o']):
    experiment_data,model=data[0],data[1]
    labels=model.keys()
    labels.sort()
    fig=pyplot.figure(figsize=(15,len(labels)/3))
    for i in range(len(labels)):
        rows=None
        if len(labels)%4==0:
            rows=len(labels)/4
        else:
            rows=len(labels)/4+1
        ax=fig.add_subplot(rows,4,i+1)
        ax_pre=ax
        ax.scatter(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label="data points")
        ax.errorbar(experiment_data[labels[i]][:,0],experiment_data[labels[i]][:,1],yerr=experiment_data[labels[i]][:,2],fmt=None,color=color[0])
        ax.plot(model[labels[i]][:,0],model[labels[i]][:,1],color=color[1],lw=3,label='model profile')
        if i!=len(labels)-1:
            ax.set_xticklabels([])
            pyplot.xlabel('')
        else:
            pyplot.xlabel('Energy (kev)',axes=ax,fontsize=12)
        pyplot.ylabel('|F|',axes=ax,fontsize=12)
        pyplot.title(labels[i])
    fig.tight_layout()
    fig.savefig(savefile,dpi=300)
    return fig
        
        
def plotting_model(object=[],fig=None,index=[2,3,1],color=['b','r'],label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1],singel dataset here
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    
    if len(object)==1:#experiment data and simulated results together (L,I_S,I,err)
        ax.scatter(object[0][:,0],object[0][:,2],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label=label[0])
        ax.errorbar(object[0][:,0],object[0][:,2],yerr=object[0][:,3],fmt=None,color=color[0])
        ax.plot(object[0][:,0],object[0][:,1],color='r',lw=3,label=label[1])
    elif len(object)==2:#first item is experiment data (L, I, err) while the second one is simulated result (L, I_s)
        ax.scatter(object[0][:,0],object[0][:,1],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label=label[0])
        ax.errorbar(object[0][:,0],object[0][:,1],yerr=object[0][:,2],fmt=None,color=color[0])
        ax.plot(object[1][:,0],object[1][:,1],color='r',lw=3,label=label[1])
    
    pyplot.xlabel('L',axes=ax,fontsize=12)
    pyplot.ylabel('|F|',axes=ax,fontsize=12)
    pyplot.title(title[0],position=(0.5,0.85),weight='bold',clip_on=True)
           
    if legend==True:
        ax.legend(bbox_to_anchor=(0.,1.02,1.,1.102),loc=3,ncol=3,mode='expand',borderaxespad=0.,prop={'size':fontsize})
        ax.legend(bbox_to_anchor=(0.2,1.06,3.,1.102),mode='expand',loc=3,ncol=2,borderaxespad=0.,prop={'size':13})
        ax.legend(bbox_to_anchor=(0.5,1.02,2.,1.102),loc=3,ncol=1,borderaxespad=0.,prop={'size':fontsize})
    
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)
        
def plotting_many_model(object=[],fig=None,index=[3,3],color=['b','r'],label=['Experimental data','Model fitting results'],marker=['p'],title=['00L','02L','10L','11L','20L','22L','30L','2-1L','21L'],legend=[True,False,False,False,False,False,False,False,False],fontsize=10):
    #plotting model results simultaneously, object=[data1,data2,data3]
    for i in range(len(object)):
        order=i
        #print 'abc'
        ob=object[i]
        if type(object[0])==type([]):
            plotting_model(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        else:
            plotting_model(object=[ob],fig=fig,index=[index[0],index[1],i+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)

def plotting_modelB(object=[],fig=None,index=[2,3,1],color=['0.35','r','c','m','k'],l_dashes=[()],lw=3,label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1,data2,data3],multiple dataset with the first one of experimental data and the others model datasets
    
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    ax.scatter(object[0][:,0],object[0][:,1],marker='o',s=20,facecolors='none',edgecolors=color[0],label=label[0])
    ax.errorbar(object[0][:,0],object[0][:,1],yerr=object[0][:,2],fmt=None,ecolor=color[0])
    for i in range(len(object)-1):#first item is experiment data (L, I, err) while the second one is simulated result (L, I_s)
        l,=ax.plot(object[i+1][:,0],object[i+1][:,1],color=color[i+1],lw=lw,label=label[i+1])
        l.set_dashes(l_dashes[i])
    if index[2] in [7,8,9]:
        pyplot.xlabel('L(r.l.u)',axes=ax,fontsize=12)
    if index[2] in [1,4,7]:
        pyplot.ylabel(r'$|F_{HKL}|$',axes=ax,fontsize=12)
    #settings for demo showing
    pyplot.title('('+title[0]+')',position=(0.5,0.86),weight=4,size=10,clip_on=True)
    if title[0]=='0 0 L':
        pyplot.ylim((1,1000000))
    elif title[0]=='3 0 L':
        pyplot.ylim((1,1000))
    elif title[0]=='1 0 L':
        pyplot.ylim((1,5000))
    else:pyplot.ylim((1,12000))
    #pyplot.ylim((1,10000))
    #settings for publication
    #pyplot.title('('+title[0]+')',position=(0.5,1.001),weight=4,size=10,clip_on=True)
    """##add arrows to antidote the misfits 
    if title[0]=='0 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.25,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
        ax.add_patch(mpt.patches.FancyArrow(0.83,0.5,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='1 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.68,0.6,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    if title[0]=='3 0 L':
        ax.add_patch(mpt.patches.FancyArrow(0.375,0.8,0,-0.15,width=0.015,head_width=0.045,head_length=0.045,overhang=0,color='k',length_includes_head=True,transform=ax.transAxes))
    """    
    if legend==True:
        #ax.legend()
        ax.legend(bbox_to_anchor=(0.2,1.03,3.,1.202),mode='expand',loc=3,ncol=5,borderaxespad=0.,prop={'size':9})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)

    #ax.set_ylim([1,10000])
#object files are returned from genx when switch the plot on
def plotting_many_modelB(save_file='D://pic.png',head='C:\\Users\\jackey\\Google Drive\\useful codes\\plotting\\',object_files=['temp_plot_O1O2','temp_plot_O5O6','temp_plot_O1O3','temp_plot_O5O7','temp_plot_O1O4','temp_plot_O5O8'],index=[3,3],color=['0.6','b','b','g','g','r','r'],lw=1.5,l_dashes=[(2,2,2,2),(None,None),(2,2,2,2),(None,None),(2,2,2,2),(None,None)],label=['Experimental data','Model1 results','Model2 results','Model3','Model4','Model5','Model6'],marker=['p'],title=['0 0 L','0 2 L','1 0 L','1 1 L','2 0 L','2 2 L','3 0 L','2 -1 L','2 1 L'],legend=[False,False,False,False,False,False,False,False,False],fontsize=10):
    #plotting model results simultaneously, object_files=[file1,file2,file3] file is the path of a dumped data/model file
    #setting for demo show
    #fig=pyplot.figure(figsize=(10,9))
    #settings for publication
    #fig=pyplot.figure(figsize=(10,7))
    fig=pyplot.figure(figsize=(8.5,7))
    object_sets=[pickle.load(open(head+file)) for file in object_files]#each_item=[00L,02L,10L,11L,20L,22L,30L,2-1L,21L]
    object=[]
    for i in range(9):
        object.append([])
        for j in range(len(object_sets)):
            if j==0:
                object[-1].append(object_sets[j][i][0])
            object[-1].append(object_sets[j][i][1])

    for i in range(len(object)):
    #for i in range(1):
        order=i
        #print 'abc'
        ob=object[i]
        plotting_modelB(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        #plotting_modelB(object=ob,fig=fig,index=[1,1,i+1],color=color,l_dashes=l_dashes,lw=lw,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(save_file,dpi=300)
    return fig
    
def plotting_model_2(object=[],fig=None,index=[2,3,1],color=['b','r','g'],label=['Experimental data','Model fit'],title=['10L'],marker=['o'],legend=True,fontsize=10):
    #overlapping the experimental and modeling fit CTR profiles,the data used are exported using GenX,first need to read the data using loadtxt(file,skiprows=3)
    #object=[data1,data2],multiple dataset here in the case you have fitting results under different conditions
    ax=fig.add_subplot(index[0],index[1],index[2])
    ax.set_yscale('log')
    #ends=[min(object[0][:,4])-5,max(object[1][:,4])+50]
    ax.errorbar(object[0][:,0],object[0][:,2],yerr=object[0][:,3],marker=marker[0],label=label[0],ecolor=color[0],color=color[0],markerfacecolor=color[0],linestyle='None')
    #ax.scatter(object[0][:,0],object[0][:,2],marker=marker[0],s=15,c=color[0],edgecolors=color[0],label=label[0])
    #ax.errorbar(object[0][:,0],object[0][:,2],yerr=object[0][:,3],fmt=None,color=color[0])
    for i in range(len(object)):
      ax.plot(object[i][:,0],object[i][:,1],color=color[i+1],lw=3,label=label[i+1])
    pyplot.xlabel('L',axes=ax,fontsize=fontsize)
    pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
    pyplot.title(title[0],position=(0.5,0.85),weight='bold',clip_on=True)
        
        #ax.yaxis.set_major_locator(mpt.ticker.MaxNLocator())
        #print ax.get_yticks()
        #print ax.get_yticklabels()
    #ax.set_yticklabels([str(yl) for yl in ax.get_yticks()])
    
    if legend==True:
        ax.legend(bbox_to_anchor=(0.67,1.06,3.,1.102),mode='expand',loc=3,ncol=2,borderaxespad=0.,prop={'size':13})
        #ax.legend(bbox_to_anchor=(0.,1.02,1.,1.102),loc=3,ncol=1,mode='expand',borderaxespad=0.,prop={'size':13})
    for xtick in ax.xaxis.get_major_ticks():
        xtick.label.set_fontsize(fontsize)
    for ytick in ax.yaxis.get_major_ticks():
        ytick.label.set_fontsize(fontsize)
    for l in ax.get_xticklines() + ax.get_yticklines(): 
        l.set_markersize(5) 
        l.set_markeredgewidth(2)
    return ax
    
def plotting_many_model_2(object=[],fig=None,index=[2,3],color=['b','r','g'],label=['Experimental data','share face mode','share edge mode'],marker=['p'],title=['00L','02L','10L','11L','20L','22L'],legend=[True,True,True,False,False,False],fontsize=10):
    #plotting model results simultaneously, object=[[data1,data11],[data2,data21],[data3,data32]]
    ax_box=[]
    for i in range(len(object)):
        order=i
        #print 'abc'
        ob=object[i]
        ax=plotting_model_2(object=ob,fig=fig,index=[index[0],index[1],i+1],color=color,label=label,title=[title[order]],marker=marker,legend=legend[order],fontsize=fontsize)
        ax_box.append(ax)
    return ax_box
    
def plotting_model_rod(data,models=[],rods=[[0,0],[1,0],[2,0],[0,2],[3,0],[2,-1],[1,1],[2,1],[2,2]],colors=['b','r','g','c','m','y','k'],markers=['.',' ',' ',' ',' ',' ',' '],linestyles=[' ','-','-.','-','-.','-','-.'],labels=['Experimental data','O1O2 modeling results'],fontsize=10):
    #data is the file name for data returned by command "list data" while the items in models are file name for model results returned by "list SUM" in ROD
    def _extract_dataset(data,rod=[0,0]):
        index=[]
        for i in range(len(data[:,0])):
            if (data[i,0]==rod[0])&(data[i,1]==rod[1]):
                index.append(i)
        
        new_data=data[index,:]
        return new_data[:,[2,3,4]]
    fig=pyplot.figure(figsize=(10,10))
    data=num.loadtxt(data)
    for rod in rods:
        data_temp=[_extract_dataset(data,rod)]
        for model in models:
            model=num.loadtxt(model)
            data_temp.append(_extract_dataset(model,rod))
        ax=fig.add_subplot(3,3,rods.index(rod)+1)
        ax.set_yscale('log')
        ax.errorbar(data_temp[0][:,0],data_temp[0][:,1],yerr=data_temp[0][:,2],marker=markers[0],label=labels[0],ecolor=colors[0],color=colors[0],markerfacecolor=colors[0],linestyle=' ')
        for i in range(len(data_temp)-1):
            ax.plot(data_temp[i+1][:,0],data_temp[i+1][:,1],color=colors[i+1],markerfacecolor=colors[i+1],markeredgecolor=colors[i+1],lw=3,label=labels[i+1],marker=markers[i+1],markersize=4,linestyle=linestyles[i+1])
        pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
        pyplot.title(str(rod[0])+str(rod[1])+'L',position=(0.5,0.85),weight='bold',clip_on=True)
        if rod==rods[0]:
            ax.legend(bbox_to_anchor=(0.2,1.06,3.,1.102),mode='expand',loc=3,ncol=3,borderaxespad=0.,prop={'size':13})
        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(fontsize)
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(fontsize)
        for l in ax.get_xticklines() + ax.get_yticklines(): 
            l.set_markersize(5) 
            l.set_markeredgewidth(2)
#note this function is for temperate use only, it will overlap experiment data (info from argument of data) and Rod fitted 
#results (info from models) and genx fitted results (info from object)
#the result is right only when you consider single model (NOT overlapping different models together)
#And you can not change the order of rods (the argument) 
#For more information about the format of object, read the associated text file in quick reflesh folder for ploting with GenX           
def plotting_model_rod_genx(data='D:\\Google Drive\\useful codes\\plotting\\data.dat',object=[],models=['D:\\Google Drive\\useful codes\\plotting\\clean_bestfit_plot.dat'],rods=[[0,0],[0,2],[1,0],[1,1],[2,0],[2,2],[3,0],[2,-1],[2,1]],colors=['b','r','g','c','m','y','k'],markers=['.',' ',' ',' ',' ',' ',' '],linestyles=[' ','-','-.','-','-.','-','-.'],labels=['lead sorbed data','Rod fitting','GenX fitting'],fontsize=10):
    def _extract_dataset(data,rod=[0,0]):
        index=[]
        for i in range(len(data[:,0])):
            if (data[i,0]==rod[0])&(data[i,1]==rod[1]):
                index.append(i)
        
        new_data=data[index,:]
        return new_data[:,[2,3,4]]
    fig=pyplot.figure(figsize=(10,10))
    data=num.loadtxt(data)
    for rod in rods:
        index=rods.index(rod)
        data_temp=[_extract_dataset(data,rod)]
        for model in models:
            model=num.loadtxt(model)
            data_temp.append(_extract_dataset(model,rod))
        ax=fig.add_subplot(3,3,rods.index(rod)+1)
        ax.set_yscale('log')
        ax.errorbar(data_temp[0][:,0],data_temp[0][:,1],yerr=data_temp[0][:,2],marker=markers[0],label=labels[0],ecolor=colors[0],color=colors[0],markerfacecolor=colors[0],linestyle=' ')
        for i in range(len(data_temp)-1):
            ax.plot(data_temp[i+1][:,0],data_temp[i+1][:,1],color=colors[i+1],markerfacecolor=colors[i+1],markeredgecolor=colors[i+1],lw=3,label=labels[i+1],marker=markers[i+1],markersize=4,linestyle=linestyles[i+1])
        ax.plot(object[index][1][:,0],object[index][1][:,1],color=colors[i+2],markerfacecolor=colors[i+2],markeredgecolor=colors[i+2],lw=3,label=labels[i+2],marker=markers[i+1],markersize=4,linestyle=linestyles[i+1])
        
        pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
        pyplot.title(str(rod[0])+str(rod[1])+'L',position=(0.5,0.85),weight='bold',clip_on=True)
        if rod==rods[0]:
            ax.legend(bbox_to_anchor=(0.2,1.06,3.,1.102),mode='expand',loc=3,ncol=3,borderaxespad=0.,prop={'size':13})
        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(fontsize)
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(fontsize)
        for l in ax.get_xticklines() + ax.get_yticklines(): 
            l.set_markersize(5) 
            l.set_markeredgewidth(2)
#overplotting experimental datas formated with UAF_CTR_RAXS_2 loader in GenX            
def plot_many_experiment_data(data_files=['D:\\Google Drive\\data\\400uM_Sb_hematite_rcut.datnew_formate','D:\\Google Drive\\data\\1000uM_Pb_hematite_rcut.datnew_formate'],labels=['Sb 400uM on hematite','Pb 1000uM on hematite'],HKs=[[0,0],[0,2],[1,0],[1,1],[2,0],[2,1],[2,-1],[2,2],[3,0]],index_subplot=[3,3],colors=['b','g','r','c','m','y','w'],markers=['.','*','o','v','^','<','>'],fontsize=10):
    data_container={}
    for i in range(len(labels)):
        temp_data=np.loadtxt(data_files[i])
        sub_set={}
        for HK in HKs:
            label=str(int(HK[0]))+'_'+str(int(HK[1]))
            sub_set[label]=np.array(filter(lambda x:x[1]==HK[0] and x[2]==HK[1],temp_data))
        data_container[labels[i]]=sub_set
    fig=pyplot.figure()
    for i in range(len(HKs)):
        title=str(int(HKs[i][0]))+str(int(HKs[i][1]))+'L'
        ax=fig.add_subplot(index_subplot[0],index_subplot[1],i+1)
        ax.set_yscale('log')
        for label in labels:
            data_temp=data_container[label][str(int(HKs[i][0]))+'_'+str(int(HKs[i][1]))]
            ax.errorbar(data_temp[:,0],data_temp[:,4],data_temp[:,5],label=label,marker=markers[labels.index(label)],ecolor=colors[labels.index(label)],color=colors[labels.index(label)],markerfacecolor=colors[labels.index(label)],linestyle='None',markersize=8)
        pyplot.title(title,position=(0.5,0.85),weight='bold',clip_on=True)
        for xtick in ax.xaxis.get_major_ticks():
            xtick.label.set_fontsize(fontsize)
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label.set_fontsize(fontsize)
        for l in ax.get_xticklines() + ax.get_yticklines(): 
            l.set_markersize(5) 
            l.set_markeredgewidth(2)
        if i==0:
            ax.legend(bbox_to_anchor=(0.5,0.92,0,3),bbox_transform=fig.transFigure,loc='lower center',ncol=4,borderaxespad=0.,prop={'size':14})
        if (i+1)>index_subplot[1]*(index_subplot[0]-1):
            pyplot.xlabel('L',axes=ax,fontsize=fontsize)
        if i%index_subplot[1]==0:
            pyplot.ylabel('|F|',axes=ax,fontsize=fontsize)
    return True
    
    
    
    
    
    
    
    
    