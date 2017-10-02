#encoding=utf8
import os
import igraph as ig
import numpy as np
import sys
sys.path.append('..\..')
from tools import commontools as gtf
from weibo_tools import weibo2graph
import igraph as ig
from igraph import clustering as clus

gt = gtf()

def getfplist(workfolder_sim,index=0):
    fplist = []
    for filep in os.listdir(workfolder_sim):
        if os.path.isfile(workfolder_sim+filep) and filep.split('_')[-1]==str(index)+'.adj':
            fp = filep.split('_')[2].split('.')[0]
            fplist.append(fp)
    return fplist

def getForwardGraph(gmlgraph,condition):
    gra = gmlgraph.subgraph_edges(edges=gmlgraph.es.select(retwitype_eq=condition), delete_vertices=False)
    return gra

def feature(graphme):
    g = graphme
    degr,indegr,outdegr,ccr,aplr,diamr = [],[],[],[],[],[] 
    
    bet = g.betweenness(vertices=None, directed=True, cutoff =None, weights=None,nobigint=True)
    pagerank = g.pagerank(vertices=None, directed=True, damping=0.85, weights=None,arpack_options=None)
    #radius = g.radius( mode='IN')

    degr.extend(g.degree())
    indegr.extend(g.indegree())
    outdegr.extend(g.outdegree())
    ccr.extend(list(np.array(g.transitivity_local_undirected(vertices=None,mode="nan")).flat))
    
    apl = list(np.array(g.shortest_paths(target=0)).flat)
    aplr.extend(apl)
    #diamr.append(g.diameter(source=0))
    
    
    
    return degr,indegr,outdegr,ccr,aplr,bet,pagerank#diamr 

def compute_real(workfolder_real_gml,fplist):
    deg,indeg,outdeg = [],[],[]       
    degr,indegr,outdegr,ccr,aplr,diamr = [],[],[],[],[],[]       
    for fp in fplist:
        realf = workfolder_real_gml+fp+'.coc.gml'
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
    
    return degr,indegr,outdegr,ccr,aplr,diamr    
#     deg.append(degr)
#     indeg.append(indegr)
#     outdeg.append(outdegr)
#     
#     return deg,indeg,outdeg

def compute_sim(workfolder_sim,fplist,experimentimes,mod):
    from NetypesfromGraph import NetypesfromGraph as nety
    netky = nety()            
    #for mod in range(2,5):
    degs,indegs,outdegs,ccs,apls,diams = [],[],[],[],[],[]
    for fp in fplist:
        for i in range(experimentimes):
            simf = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\test\\Mode"+str(mod)+'\\graphs\\'+'_'+str(mod)+'_'+fp+'.repost_'+str(i)+'.adj'
#         simf1 = workfolder_sim+'_'+str(mod)+'_'+fp+'.repost_1.adj'
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
                
    return degs,indegs,outdegs,ccs,apls,diams
#     return deg,indeg,outdeg

from weibo_tools import weibo2graph
    
def createGml(gid,workfolder_real,workfolder_real_gml,addcommentlist=False):
    wb2g = weibo2graph() 
    weibofilep = workfolder_real + gt.findfiles(workfolder_real,'*'+str(gid)+'*'+'.repost')[0]
    gmlfilep = workfolder_real_gml+str(gid)+'.gml'
    g = wb2g.start(weibofilep,gmlfilep,timep=None,addcommentlist=addcommentlist)
    return g

def get_simgraph(gid,workfolder_real,workfolder_sim,experimenTimes,mode,addcomment):
    ""
    graphfilepaths = []
    worksfolder_fig = gt.createFolder(workfolder_sim+'figs\\')
    worksfolder_mat = gt.createFolder(workfolder_sim+'graphs\\')
    weibofilep = workfolder_real + gt.findfiles(workfolder_real,'*'+str(gid)+'*'+'.repost')[0]
    from simulate_cmo_growth import cmo_growth_model
    sm = cmo_growth_model()
    
    #[ma_fans,cbm_frs,inv_mention,act_micorcnt] = sm.input_init_real(weibofilep)[0:4]
    [ma_fans,cbm_frs,inv_mention,act_micorcnt] = sm.input_init_real_new(weibofilep,addcomment)[0:4]
    input = [ma_fans,cbm_frs,inv_mention,act_micorcnt]
    print inv_mention
    #     inputm = sp.asmatrix(input, long)
    personcnt =  len(ma_fans)   
    for i in range(0,experimenTimes): 
        print '===============================================',i
        mat = sm.getoutMatrix(personcnt,ma_fans,cbm_frs,inv_mention,act_micorcnt,worksfolder_fig,mode)
        
        g = ig.Graph.Adjacency(mat.tolist())
        graphfilepath = str(workfolder_sim+gid+'_'+str(i)+'.adj')
        print graphfilepath
        g.write_adjacency(graphfilepath)
        graphfilepaths.append(graphfilepath)
        
    return graphfilepaths
    
from tools import graphtools    
grt = graphtools()    
def get_simgraphs_att(feature_real,workfolder_sim,gsim_fp):
    shaper = np.array(feature_real).shape
    print shaper
    feature_sims = np.empty(0)#()#np.zeros([])
    lengsim_fp = len(gsim_fp)
    for gfp in gsim_fp:
        print gfp   
        g_sim = ig.Graph.Read_Adjacency(gfp)#workfolder_sim+
        #print g_sim.indegree()
        #g_sim = grt.getCorePart(g_sim)
        feature_sim = np.array(feature(g_sim))
        if feature_sims.size<1:
            print '999999999999'
            feature_sims = feature_sim
        else:
            feature_sims += feature_sim;print feature_sims.shape
        #feature_sims = np.append(feature_sim,feature_sims,)
        #feature_sims.append(feature_sim)
    feature_sims = np.nan_to_num(feature_sims)
    #feature_sims = np.reshape(feature_sims,(-1,shaper[1]))
    feature_sims_avg = feature_sims/float(lengsim_fp)
    
    return feature_sims_avg

            
def start_one(gid,workfolder_real,workfolder_real_gml,workfolder_sim,fileappendix_real='.gml',fileappendix_sim='.adj',addcomment=False):
    "IN:weibo id; gml file folder of real and sim"
    "OUT:list of atts for comparation"
    feature_real,feature_sim = None,None
    experimenTimes,mode = 1,7
    if not fileappendix_sim:
        fileappendix_sim = fileappendix_real
        
    greal_fp = gt.findfiles(workfolder_real_gml,'*'+str(gid)+'*' + fileappendix_real)    
    gsim_fp =  gt.findfiles(workfolder_sim,'*'+str(gid)+'*' + fileappendix_sim)
    g_real = ig.Graph()
    if len(greal_fp)<1:
        g_real = createGml(gid,workfolder_real,workfolder_real_gml,addcommentlist=addcomment)
    else:        
        g_real = ig.Graph.Read_GML(workfolder_real_gml+greal_fp[0])
    
    if len(gsim_fp)<1:        
        gsim_fp = get_simgraph(gid,workfolder_real,workfolder_sim,experimenTimes,mode,addcomment)
        print gsim_fp
    feature_real = np.array(feature(g_real))
    feature_sim = get_simgraphs_att(feature_real,workfolder_sim,gsim_fp)
    print feature_sim.shape,feature_real.shape  
        

    return feature_real,feature_sim
        

def start(workfolder_real,workfolder_sim,gids,mod,experimentimes):
    workfolder_out = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
    workfolder_sim = gt.createFolder(workfolder_sim+"test2\\Mode"+str(mod)+'\\graphs\\')#
    workfolder_real_gml = gt.createFolder(workfolder_real+"GML\\")
    
    grs,gss = np.empty(0),np.empty(0)
    for gid in gids:
        gr,gs = start_one(gid,workfolder_real,workfolder_real_gml,workfolder_sim,)
        grs = gr if grs.size<1 else grs
        gss = gs if gss.size<1 else gss
        
        grs = np.append(grs,gr,1)
        gss = np.append(gss,gs,1)
    
    for r,s in zip(*(grs,gss)):
        gt.list_2_Distribution([r,s])
    
    
    
        
if __name__=="__main2__":  
    mod = 7
    experimentimes = 1
    workfolder_out = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
    workfolder_sim = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\test2\\Mode"+str(mod)+'\\graphs\\')#
    workfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\"#small\\
    workfolder_real_gml = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeiboGMLNew2\\")
    
    gids = ['2014071400_3731912151816117','3501198583561829','3514416295764354','3510150776647546','3510947052234805','3511312581651670','3428528739801892','3373235549224874','3451840329079188']
    grs,gss = np.empty(0),np.empty(0)
    for gid in gids:
        gr,gs = start_one(gid,workfolder_real,workfolder_real_gml,workfolder_sim,)
        grs = gr if grs.size<1 else grs
        gss = gs if gss.size<1 else gss
        
        grs = np.append(grs,gr,1)
        gss = np.append(gss,gs,1)
    
    for r,s in zip(*(grs,gss)):
        gt.list_2_Distribution([r,s])
    er
    "======================================================="

    fplist = getfplist(workfolder_sim,index=2);print len(fplist)
    
        
    real_resultset,sim_resultset = [],[] 

    deg,indeg,outdeg,cc,apl,diam = [],[],[],[],[],[],
    degr,indegr,outdegr,ccr,aplr,diamr = compute_real(workfolder_real_gml,fplist)
    deg.append(degr);indeg.append(indegr);outdeg.append(outdegr);cc.append(ccr);apl.append(aplr);diam.append(diamr)
    for mod in range(2,7):    
        deg_sim,indeg_sim,outdeg_sim,ccs_sim,apls_sim,diams_sim = compute_sim(workfolder_sim,fplist,experimentimes,mod)

        if deg_sim and indeg_sim and outdeg_sim:
#             deg.append(deg_sim)
#             indeg.append(indeg_sim)
#             outdeg.append(outdeg_sim)
            deg.append(deg_sim);indeg.append(indeg_sim);outdeg.append(outdeg_sim);cc.append(ccs_sim);apl.append(apls_sim);diam.append(diams_sim)
    
#     print real_resultset
#     print sim_resultset
    print len(deg),deg
    gt.saveList(deg,workfolder_out+'deg_sr.txt',)
    [degdis_X,disdis_Y] = gt.list_2_Distribution(deg,xlabels=['Real','Sim_Mode2','Sim_Mode3','Sim_Mode4','Sim_Mode2','Sim_Mode3','Sim_Mode4',],ylabels=['Frequency',],binseqdiv=2)    
    gt.list_2_Distribution(indeg,xlabels=['In Degree',],ylabels=['Frequency',])
    gt.list_2_Distribution(outdeg,xlabels=['Out Degree',],ylabels=['Frequency',])
    gt.list_2_Distribution(cc,xlabels=['CC',],ylabels=['Frequency',])
    gt.list_2_Distribution(apl,xlabels=['APL',],ylabels=['Frequency',])
    gt.list_2_Distribution(diam,xlabels=['D',],ylabels=['Frequency',])

    gt.saveList([degdis_X,disdis_Y],workfolder_out+'deg_sr.fig.data',)
        

        
#         g_sim0 = ig.Graph.Read_Adjacency(simf0)
#         g_sim1 = ig.Graph.Read_Adjacency(simf1)
        
#         resultone = netky.start(g)
#         result.append(resultone)
#              
#     xyz = zip(*result)
#     gt.saveList(xyz,worksfolder_mat+'_'+str(mod)+'_'+casefile+'_'+str(experimenTimes)+'.netattri',writype='w')
    
    


