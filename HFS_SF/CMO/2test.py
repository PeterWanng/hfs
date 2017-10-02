#encoding=utf8
import os
import igraph as ig
import numpy as np
import sys
sys.path.append('..\..')
from tools import commontools as gtf
import igraph as ig
from igraph import clustering as clus
from matplotlib import pyplot as plt


gt = gtf()

def getfplist(worksfolder_sim):
    fplist = []
    for filep in os.listdir(worksfolder_sim):
        if os.path.isfile(worksfolder_sim+filep) and filep.split('.')[-1]=='repost':
            fp = filep.split('.')[0]
            fplist.append(fp)
    return fplist

def getForwardGraph(gmlgraph,condition):
    gra = gmlgraph.subgraph_edges(edges=gmlgraph.es.select(retwitype_eq=condition), delete_vertices=False)
    return gra

def draw_SR(mod,worksfolder_sim,worksfolder_real_gml):
    "draw the sim and real graph in different folders"
    mod = 7
    worksfolder_sim = 'G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Mode'+str(mod)+'\\graphs\\'
    worksfolder_real_gml = "G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\"
    fplist = getfplist(worksfolder_sim)
    for fp in fplist:
        adj = worksfolder_sim+'_'+str(mod)+'_'+str(fp)+'.repost_0.adj'
        
        g = ig.Graph.Read_Adjacency(adj)
        g2 = ig.Graph.Read_GML(worksfolder_real_gml+str(fp)+'.coc.gml')
        g2 = clus.VertexClustering.giant(g2.clusters(mode='weak'))
        print g.vcount(),g2.vcount(),g.ecount(),g2.ecount()
        gt.drawgraph(g)
        gt.drawgraph(g2)


def first(x,attfilepath):
        xyz = zip(*x)
        xyzz = gt.normlistlist(listlista=xyz[1:],metacolcount=0,sumormax='max')#xyz[1:]#
        xy = zip(*xyzz)#[1:]
#         z = zip(*xyz[:1])
#         z = zip(*z)[0]
#         import string
#         print z
#         z = string.join(z[0],',')

        xy = np.nan_to_num(xy)
        labels,kmcenter,kmfit = netky.kmeans(xy,k=6,runtimes=1000)
#         print z
#         xy.insert(0,z)
#         print xy
#         for a,b in zip(*(z,xy)):
#             b = list(b)
#             b.insert(0,a[0])
#             print a
#             print b
        
#         xy.append(labels);xy.append(kmcenter);xy.append(kmfit)
#         gt.saveList([xyz,labels,kmcenter,kmfit],attfilepath,writype = 'w')
#         import string
#         resline = string.join(labels.__str__(),kmcenter.__str__())#string.join(string.join(labels.__str__(),kmcenter.__str__()),kmfit.__str__())
#         print resline
#         gt.saveList(xy,attfilepath)
        np.savetxt(fname=attfilepath, X=xy,  delimiter=' ', newline='\n', header='', footer='', comments='# ')#
        np.savetxt(fname=attfilepath+'label', X=labels, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#
        np.savetxt(fname=attfilepath+'center', X=np.asarray(kmcenter), fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#

        return xy,labels,kmcenter
    
def itemcntDis(labels):
    labelen = len(labels)
    distinct_label = np.unique(ar=labels, return_index=True, return_inverse=True)
    distLabel = list(distinct_label[0])
    i = len(distLabel)
    res = []
    oneres = []
    for j in range(i,labelen+1):
        x = list(labels[:j])
        oneres = []
#         oneres = np.bincount(distLabel)
        for k in distLabel:
            oneres.append(x.count(k))#np.append(oneres,[x.count(k)],axis=0)#
        oneres = gt.normalizelist(oneres,sumormax='sum')
        res.append(oneres)#np.append(res,[oneres])#
#     print distLabel,len(res),oneres
    return res,distLabel,oneres

def plot_label_rs(label_r,label_s):
    res_r,distLabel_r,oneres_r = itemcntDis(label_r)
    res_s,distLabel_s,oneres_s = itemcntDis(label_s)
    
    colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
    markerlist = list('x+.*_psv^,><12348hHD|od')
    linestylelist = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted',]

    i,j = 0,0
    for item in zip(*res_r):
        plt.subplot(2,3,i)
        xx = range(len(item))
        plt.plot(xx,item,color=colorlist[i],linestyle=linestylelist[0])#,linestyle=''
        plt.ylim(ymin=-0.1,ymax=1.4)
        i+=1
        
    i,j = 0,0     
    for item in zip(*res_s):
        plt.subplot(2,3,i)
        xx = range(len(item))
        plt.plot(xx,item,color=colorlist[i],linestyle=linestylelist[1])#,linestyle=''
        plt.ylim(ymin=-0.1,ymax=1.4)
        i+=1
#     plt.show()        

def getLabels_sim(fplist,att_sim_filep,allres,att_simreal_filep,netky):
    sim_resultset = []
    if os.path.isfile(att_sim_filep):
        sim_resultset = np.genfromtxt(att_sim_filep)
    else:
        for fp in fplist:
            simf = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Mode"+str(mod)+'\\graphs\\'+'_'+str(mod)+'_'+fp+'.repost_'+str(i)+'.adj'
    #         simf1 = worksfolder_sim+'_'+str(mod)+'_'+fp+'.repost_1.adj'
            g_sim = ig.Graph.Read_Adjacency(simf)
            res_sim = netky.start(g_sim)
            sim_resultset.append(res_sim)
        first(sim_resultset,att_sim_filep)
  
    allres.extend(sim_resultset)
    xy,labels,kmcenter = first(allres,att_simreal_filep)
    allres = []
    return xy,labels,kmcenter

def getLabels_real(fplist,worksfolder_real_gml,att_real_filep,netky):
    real_resultset = []
    for fp in fplist:
        realf = worksfolder_real_gml+fp+'.coc.gml'
        g_real = ig.Graph.Read_GML(realf)
        g_real = clus.VertexClustering.giant(g_real.clusters(mode='weak'))
        print g_real.vcount()
#         gt.drawgraph(g_real)
#         g_real = getForwardGraph(g_real,"8")
#         print g_real.ecount()
#         gt.drawgraph(g_real)
         
        res_real = netky.start(g_real)
        real_resultset.append(res_real)
    xy,labels,kmcenter = first(real_resultset,att_real_filep)
    return real_resultset#xy,labels,kmcenter

def mat_average(listfp):
    for fp in listfp:
        att = np.genfromtxt(fp)
     

                    
if __name__=="__main__":  
#     mod = 2
    experimentimes = 2
    workfolder_out = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
#     worksfolder_sim = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Mode"+str(mod)+'\\graphs\\'#test\\
    worksfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\small\\"
    worksfolder_real_gml = "G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\"
    
    att_real_filep = workfolder_out+'real.att'

    fplist = getfplist(worksfolder_real)
    from NetypesfromGraph import NetypesfromGraph as nety
    netky = nety()        

    real_resultset = list(np.genfromtxt(att_real_filep))#getLabels_real(fplist,worksfolder_real_gml,att_real_filep,netky)
            
    for i in range(experimentimes):
        atts = []
        attall = real_resultset
        att_simall_filep = workfolder_out+'simall_'+'_'+str(i)+'.att'
        for mod in range(2,8):
            att_sim_realall_filep = workfolder_out+'sim_real_all_'+str(mod)+'_'+str(i)+'.att'
            allres = real_resultset[:]
            sim_resultset = []
            att_sim_filep = workfolder_out+'sim_'+str(mod)+'_'+str(i)+'.att'
            att_simreal_filep = workfolder_out+'simreal_'+str(mod)+'_'+str(i)+'.att'
            att = getLabels_sim(fplist,att_sim_filep,allres,att_simreal_filep,netky)[1]#np.genfromtxt(att_sim_filep,dtype=float)#
            print att_sim_filep,len(att)
            atts.extend(att)
            
        attall.extend(atts)
        np.savetxt(att_simall_filep,atts)
        xy,labels,kmcenter = first(attall,att_sim_realall_filep)
        

        lenlabels = len(labels)
        print lenlabels
        for i in range(0,lenlabels,lenlabels/8):
            label_r = labels[:lenlabels/8]
            label_s = labels[i:i+lenlabels/8]
            print len(label_r),len(label_s)
            plot_label_rs(label_r,label_s)

#             labels = np.genfromtxt(att_sim_filep+'label',dtype=int)#getLabels_sim(fplist,att_sim_filep,allres,att_simreal_filep,netky)[1]#

#         labelsall.append(labels)
#         lenlabels = len(labels)
        plt.show()        
 
if __name__=="__main2__":  
#     mod = 2
    experimentimes = 10
    workfolder_out = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
#     worksfolder_sim = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Mode"+str(mod)+'\\graphs\\'#test\\
    worksfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\small\\"
    worksfolder_real_gml = "G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\"
    
    att_real_filep = workfolder_out+'real.att'

    fplist = getfplist(worksfolder_real)
    from NetypesfromGraph import NetypesfromGraph as nety
    netky = nety()        

    real_resultset = list(np.genfromtxt(att_real_filep))#getLabels_real(fplist,worksfolder_real_gml,att_real_filep,netky)
            
    for i in range(experimentimes):
        for mod in range(2,8):
            allres = real_resultset[:]
            sim_resultset = []
            att_sim_filep = workfolder_out+'sim_'+str(mod)+'_'+str(i)+'.att'
            att_simreal_filep = workfolder_out+'simreal_'+str(mod)+'_'+str(i)+'.att'
            print att_sim_filep

##             xy,labels,kmcenter = getLabels_sim(fplist,att_sim_filep,allres,att_simreal_filep,netky)
            labels = np.genfromtxt(att_simreal_filep+'label',dtype=int)#getLabels_sim(fplist,att_sim_filep,allres,att_simreal_filep,netky)[1]#
            lenlabels = len(labels)
             
            label_r = labels[:lenlabels/2]
            label_s = labels[lenlabels/2:]
            print len(label_r),len(label_s)
            plot_label_rs(label_r,label_s)

#             labels = np.genfromtxt(att_sim_filep+'label',dtype=int)#getLabels_sim(fplist,att_sim_filep,allres,att_simreal_filep,netky)[1]#

#         labelsall.append(labels)
#         lenlabels = len(labels)
        plt.show()  
        
        
         

    
    
if __name__=="__main2__":  
    mod = 3
    experimentimes = 1
    workfolder_out = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
    worksfolder_sim = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Mode"+str(mod)+'\\graphs\\'#test\\
    worksfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\small\\"
    worksfolder_real_gml = "G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\"

    fplist = getfplist(worksfolder_sim)
    from NetypesfromGraph import NetypesfromGraph as nety
    netky = nety()        
    real_resultset,sim_resultset = [],[] 
    deg,indeg,outdeg,cc,apl,diam = [],[],[],[],[],[]       
    degr,indegr,outdegr,ccr,aplr,diamr = [],[],[],[],[],[]       
    for fp in fplist:
        realf = worksfolder_real_gml+fp+'.coc.gml'
        g_real = ig.Graph.Read_GML(realf)
        g_real = clus.VertexClustering.giant(g_real.clusters(mode='weak'))
#         print g_real.ecount()
#         gt.drawgraph(g_real)
#         g_real = getForwardGraph(g_real,"8")
#         print g_real.ecount()
#         gt.drawgraph(g_real)
        
#         res_real = netky.start(g_real)
#         real_resultset.append(res_real)
        
        degr.extend(g_real.degree())
        indegr.extend(g_real.indegree())
        outdegr.extend(g_real.outdegree())
        ccr.append(g_real.transitivity_avglocal_undirected(mode="nan"))
        aplr.append(g_real.average_path_length())
        diamr.append(g_real.diameter())
        
    deg.append(degr)
    indeg.append(indegr)
    outdeg.append(outdegr)
    cc.append(ccr)
    apl.append(aplr)
    diam.append(diamr)
            
    for mod in range(2,8):
        degs,indegs,outdegs,ccs,apls,diams = [],[],[],[],[],[]
        for fp in fplist:
            for i in range(experimentimes):
                simf = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Mode"+str(mod)+'\\graphs\\'+'_'+str(mod)+'_'+fp+'.repost_'+str(i)+'.adj'
    #         simf1 = worksfolder_sim+'_'+str(mod)+'_'+fp+'.repost_1.adj'
                g_sim = ig.Graph.Read_Adjacency(simf)
#                 res_sim = netky.start(g_sim)
#                 sim_resultset.append(res_sim)
                
                degs.extend(g_sim.degree())
                indegs.extend(g_sim.indegree())
                outdegs.extend(g_sim.outdegree())
                ccs.append(g_sim.transitivity_undirected(mode='0'))
                apls.append(g_sim.average_path_length())
                diams.append(g_sim.diameter())
        deg.append(degs)
        indeg.append(indegs)
        outdeg.append(outdegs)
        cc.append(ccs)
        apl.append(apls)
        diam.append(diams)

#     print real_resultset
#     print sim_resultset
    print len(deg),deg
    for lista,listaname in zip(*([deg,indeg,outdeg,cc,apl,diam],['deg','indeg','outdeg','cc','apl','diam'])):
        gt.saveList(lista,workfolder_out+listaname+'_sr.txt',)
#         gt.saveList(deg,workfolder_out+'deg_sr.txt',)

    xlabels=['Real','Sim_Mode2','Sim_Mode3','Sim_Mode4','Sim_Mode5','Sim_Mode6','Sim_Mode7']
    [degdis_X,degdis_Y] = gt.list_2_Distribution(deg,xlabels,ylabels=['Frequency',],binseqdiv=0)    
    [indegdis_X,indegdis_Y] = gt.list_2_Distribution(indeg,xlabels,ylabels=['Frequency',])
    [outdegdis_X,outdegdis_Y] = gt.list_2_Distribution(outdeg,xlabels,ylabels=['Frequency',])
    [ccdis_X,ccdis_Y] = gt.list_2_Distribution(cc,xlabels,ylabels=['Frequency',],showfig=False)
    [apldis_X,apldis_Y] = gt.list_2_Distribution(apl,xlabels,ylabels=['Frequency',])
    [diamdis_X,diamdis_Y] = gt.list_2_Distribution(diam,xlabels,ylabels=['Frequency',])

    gt.saveList([degdis_X,degdis_Y],workfolder_out+'deg_sr.fig.data',)
    abc = [[degdis_X,degdis_Y],[indegdis_X,indegdis_Y],[outdegdis_X,outdegdis_Y],[ccdis_X,ccdis_Y],[apldis_X,apldis_Y],[diamdis_X,diamdis_Y]]
    for lista,listaname in zip(*(abc,['deg','indeg','outdeg','cc','apl','diam'])):
        gt.saveList(lista,workfolder_out+listaname+'_sr.fig.data',)
        

        
#         g_sim0 = ig.Graph.Read_Adjacency(simf0)
#         g_sim1 = ig.Graph.Read_Adjacency(simf1)
        
#         resultone = netky.start(g)
#         result.append(resultone)
#              
#     xyz = zip(*result)
#     gt.saveList(xyz,worksfolder_mat+'_'+str(mod)+'_'+casefile+'_'+str(experimenTimes)+'.netattri',writype='w')
    
 

