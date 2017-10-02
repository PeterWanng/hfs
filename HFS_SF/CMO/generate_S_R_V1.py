#encoding=utf8
import os
import igraph as ig
import numpy as np
import sys
sys.path.append('..\..')
from tools import commontools as gtf
import igraph as ig
from igraph import clustering as clus

gt = gtf()

def getfplist(worksfolder_sim,index=0):
    fplist = []
    for filep in os.listdir(worksfolder_sim):
        if os.path.isfile(worksfolder_sim+filep) and filep.split('_')[-1]==str(index)+'.adj':
            fp = filep.split('_')[2].split('.')[0]
            fplist.append(fp)
    return fplist

def getForwardGraph(gmlgraph,condition):
    gra = gmlgraph.subgraph_edges(edges=gmlgraph.es.select(retwitype_eq=condition), delete_vertices=False)
    return gra

if __name__=="__main__":  
    mod = 3
    experimentimes = 1
    workfolder_out = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
    worksfolder_sim = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\test\\Mode"+str(mod)+'\\graphs\\'#
    worksfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\small\\"
    worksfolder_real_gml = "G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\"

    fplist = getfplist(worksfolder_sim,index=2);print len(fplist)
    from NetypesfromGraph import NetypesfromGraph as nety
    netky = nety()        
    real_resultset,sim_resultset = [],[] 
    deg,indeg,outdeg = [],[],[]       
    degr,indegr,outdegr,ccr,aplr,diamr = [],[],[],[],[],[]       
    for fp in fplist:
        realf = worksfolder_real_gml+fp+'.coc.gml'
        g_real = ig.Graph.Read_GML(realf)
#         print g_real.ecount()
#         gt.drawgraph(g_real)
        g_real = getForwardGraph(g_real,"8")
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
            
    for mod in range(2,5):
        degs,indegs,outdegs,ccs,apls,diams = [],[],[],[],[],[]
        for fp in fplist:
            for i in range(experimentimes):
                simf = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\test\\Mode"+str(mod)+'\\graphs\\'+'_'+str(mod)+'_'+fp+'.repost_'+str(i)+'.adj'
    #         simf1 = worksfolder_sim+'_'+str(mod)+'_'+fp+'.repost_1.adj'
                if os.path.exists(simf):
                    g_sim = ig.Graph.Read_Adjacency(simf)
                    res_sim = netky.start(g_sim)
                    sim_resultset.append(res_sim)
                    
                    degs.extend(g_sim.degree())
                    indegs.extend(g_sim.indegree())
                    outdegs.extend(g_sim.outdegree())
                    ccs.append(g_sim.transitivity_undirected(mode='0'))
                    apls.append(g_sim.average_path_length())
                    diams.append(g_sim.diameter())
        deg.append(degs)
        indeg.append(indegs)
        outdeg.append(outdegs)

#     print real_resultset
#     print sim_resultset
    print len(deg),deg
    gt.saveList(deg,workfolder_out+'deg_sr.txt',)
    [degdis_X,disdis_Y] = gt.list_2_Distribution(deg,xlabels=['Real','Sim_Mode2','Sim_Mode3','Sim_Mode4',],ylabels=['Frequency',],binseqdiv=2)    
    gt.list_2_Distribution(indeg,xlabels=['In Degree',],ylabels=['Frequency',])
    gt.list_2_Distribution(outdeg,xlabels=['Out Degree',],ylabels=['Frequency',])
#     gt.list_2_Distribution([ccs,ccr],xlabels=['CC',],ylabels=['Frequency',])
    gt.list_2_Distribution([apls,aplr],xlabels=['APL',],ylabels=['Frequency',])
    gt.list_2_Distribution([diams,diamr],xlabels=['D',],ylabels=['Frequency',])

    gt.saveList([degdis_X,disdis_Y],workfolder_out+'deg_sr.fig.data',)
        

        
#         g_sim0 = ig.Graph.Read_Adjacency(simf0)
#         g_sim1 = ig.Graph.Read_Adjacency(simf1)
        
#         resultone = netky.start(g)
#         result.append(resultone)
#              
#     xyz = zip(*result)
#     gt.saveList(xyz,worksfolder_mat+'_'+str(mod)+'_'+casefile+'_'+str(experimenTimes)+'.netattri',writype='w')
    
    


