# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from tools import commontools as gtf
import csv
import re
import os
import time
import numpy as np                             


import igraph as ig
from igraph.drawing.graph import DefaultGraphDrawer as igd
from igraph import clustering as clus
from igraph import statistics as stcs


        
gt=gtf()
# g = ig.read(r'G:\HFS\WeiboData\HFSWeiboGMLNew2\3510150776647546.gml')
# print g.vcount(),g.ecount(),g.vs.attribute_names(),g.vs.get_attribute_values('followerscount')
# gte = g.subgraph_edges(g.es.select(createdtimetos_ge = '1352502966'),delete_vertices=True)
# print g.vcount(),g.ecount(),gte.vcount(),gte.ecount()
# g = g.subgraph(g.vs.select(name_in=gte.vs.get_attribute_values('name')))
# ge = g.subgraph_edges(g.es.select(createdtimetos_ge = '1352502966'),delete_vertices=True)
# 
# print g.vcount(),g.ecount(),ge.vcount(),ge.ecount()
# print g.vs.attribute_names()
# print ge.vs.attribute_names()
# g.subgraph(vs.select(name_in=g.vs.get_attribute_values('name')))
# g.subgraph(g.vs.select(name_in=gte.vs.get_attribute_values('name')))
# er

# if 'bifollowerscount' in g.vs.attribute_names():
#     print 'adsfasdf'
# erw
#print g.get_attribute_values('bifollowerscount')
# bifansumlist = g.vs.get_attribute_values('bifollowerscount')
# bifansumlistFloat = gt.convertype2float(bifansumlist) 
# bifansumlistMedian = np.median(bifansumlistFloat)

    
def assority(g,attribute='degree',directed=False,defaultV=None):
    #g = ig.read(r'G:\HFS\WeiboData\HFSWeiboGMLNew2\3510150776647546.gml')
    vsatt = g.vs.attribute_names()
    bifansumlist = g.vs.get_attribute_values(attribute) if attribute in vsatt else g.es.get_attribute_values(attribute)
    if not defaultV:
        bifansumlistFloat = gt.convertype2float(bifansumlist) 
        defaultV = np.median(bifansumlistFloat)
    bifansumlist = gt.convertype2float(bifansumlist,defaultV) 
    bifansumlist = map(int,bifansumlist)
    
    assorFans=g.assortativity(bifansumlist,directed= directed) 
    #print '------------------------------------',assorFans
    return assorFans

# g = ig.read(r'G:\HFS\WeiboData\HFSWeiboGMLNew2\3510150776647546.gml')
# assority(g,attribute='bifollowerscount',directed=False,defaultV=None)
# er    



def coclist2graph(coclist,g=ig.Graph().as_directed(),listmeta=['name','name','mid','userid','time','plzftype','retwitype','statusid']):
    #IN:edges list/tuple， in each line,the content order is listmeta
    #OUT:the graph of the coc, the attributes of edge is as listmeta
#     print time.clock()
    g.as_directed()   
    for e in coclist:
        e = list(e) 
        g.add_vertex(name=e[0])
        g.add_vertex(name=e[1])
        g.add_edge(e[0], e[1], mid=e[2], userid=e[3], time=e[4], plzftype=e[5], retwitype=e[6], statusid=e[7])   
#         for v,x,m,u,t,ty1,ty2,stid in edge:#itt.izip(list_nameS,list_nameT,list_time,list_type):
#             g.add_vertex(name=v)#{'time':3}
#             g.add_vertex(name=x)#{'time':3}
#             g.add_edge(v, x, mid=m, userid=u, time=t, plzftype=ty1, retwitype=ty2, statusid=stid)
#     print time.clock(),'coclist2graph over'
    return g
        
def coc2gml(cocfilepath,coclineseprator='\t',gmlfilepath='IHaveNoName.gml'):
    ##将图以节点对方式读入,含有其它边属性，输出gml格式
#     print time.clock()
    import networkx as nx
    #单边向DiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.Graph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #单边有向DiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.DiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #多边无向MultiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.MultiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #多边有向MultiDiGraph
    G=nx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiDiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    gmlfile = open(gmlfilepath,'w')
    nx.write_gml(G,gmlfile)
    gmlfile.close()
#     print time.clock(),'coc2gml over'

def drawgml(gmlfilepath):
    try:
        import matplotlib.pyplot as plt
    except:
        raise
    import networkx as nx
#     G=nx.cycle_graph(12)
    G=nx.read_gml(gmlfilepath)
    pos=nx.spring_layout(G,iterations=200)
    nx.draw(G,pos,node_size=800)#,cmap=plt.cm.Blues,style='dotted')
#     plt.savefig("node_colormap.png") # save as png
    plt.show() # display)

def analysisNet(graph):
    try:    
        g=graph
        gg=clus.VertexClustering.giant(g.clusters(mode='weak'))
        vcount=g.vcount()
        ecount=g.ecount()
        degree=gg.degree()
        indegree=gg.indegree()
        outdegree=gg.outdegree()
        degreePowerLawFit=stcs.power_law_fit(degree,method='auto',return_alpha_only=False)
        indegreePowerLawFit=stcs.power_law_fit(indegree, method='auto',return_alpha_only=False)
        outdegreePowerLawFit=stcs.power_law_fit(outdegree,method='auto',return_alpha_only=False)
        assorDeg=gg.assortativity(degree,directed= False)
        assorDegD=gg.assortativity(degree,directed= True)
        assorInDeg=gg.assortativity(indegree,directed= True)
        assorOutDeg=gg.assortativity(outdegree,directed= True)


        assorDegF='1' if assorDeg>0 else '-1'
        assorInDegF='1' if assorInDeg>0 else '-1'
        assorOutDegF= '1' if assorOutDeg>0 else '-1'

        # assority_followerscount = assority(g,attribute='followerscount',directed=False,defaultV=None)
        # assority_followerscount_d = assority(g,attribute='followerscount',directed=True,defaultV=None)
        # assority_bifollowerscount = assority(g,attribute='bifollowerscount',directed=False,defaultV=None)
        # assority_bifollowerscount_d = assority(g,attribute='bifollowerscount',directed=True,defaultV=None)
        # assority_friendscount = assority(g,attribute='friendscount',directed=False,defaultV=None)
        # assority_friendscount_d = assority(g,attribute='friendscount',directed=True,defaultV=None)
        
#         print g.average_path_length()
#         return g.vcount(),g.ecount(),\
#             str(g.average_path_length()),\
#             str(g.diameter()),\
#             str(len(g.clusters(mode='weak'))),\
#             str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
#             str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount())

        return [
                str(vcount),\
        str(ecount),\
        str(g.density()),\
        str(len(g.clusters(mode='weak'))),\
        str(len(g.clusters(mode='strong'))),\
        str(gg.vcount()),\
        str(gg.ecount()),\
        str((ecount*2)/float(vcount)),\
        str(gg.transitivity_undirected(mode='0')) ,\
        str(gg.average_path_length()),\
        str(gg.diameter()),\
        str(assorDeg),\
        str(assorDegD),\
        str(assorInDeg),\
        str(assorOutDeg),\
        str(assorDegF),\
        str(assorInDegF),\
        str(assorOutDegF),\
        str(degreePowerLawFit.alpha),\
        str(degreePowerLawFit.xmin),\
        str(degreePowerLawFit.p),\
        str(degreePowerLawFit.L),\
        str(degreePowerLawFit.D),\
        str(indegreePowerLawFit.alpha),\
        str(indegreePowerLawFit.xmin),\
        str(indegreePowerLawFit.p),\
        str(indegreePowerLawFit.L),\
        str(indegreePowerLawFit.D),\
        str(outdegreePowerLawFit.alpha),\
        str(outdegreePowerLawFit.xmin),\
        str(outdegreePowerLawFit.p),\
        str(outdegreePowerLawFit.L),\
        str(outdegreePowerLawFit.D),\
        # str(assority_followerscount),\
        # str(assority_followerscount_d),\
        # str(assority_bifollowerscount),\
        # str(assority_bifollowerscount_d),\
        # str(assority_friendscount),\
        # str(assority_friendscount_d),\
        
        ]
    except Exception,e:
        print 'error=====',e
        return []

           
def selecTime(timelist,periodcnt):
    timelistPeriodNow = []
    lenyt = len(timelist)/float(periodcnt)
    for j in range(1,periodcnt+1):
        i = int(round(j*lenyt))
        i = i if i>1 else 1
        i = i if i<len(timelist) else len(timelist)-1
        timelistPeriodNow.append(timelist[i])
    return timelistPeriodNow    
   
    
def selecTimelist(findstr,timeSeriesFilepath):
    timel = []
    timefile = open(timeSeriesFilepath,'r')
    for line in csv.reader(timefile):
        if line[0].replace('.repost','')==findstr:
            timel = line[1:]
            break
    return timel
                    
def analyzeNetStat(g):
    "given a graph g with edges attributes, analyze its stat features"
    "IN:graph g"
    "OUT:stat features"
    es = g.es#ig.EdgeSeq(g)
    vs = g.vs#ig.EdgeSeq(g)
    vsatt = vs.attribute_names()
    esatt = es.attribute_names()
    #print es.attribute_names()
    '''reposts_count
    mentioncnt
    city
    verified
    retweeted_status
    attitudes_count
    location
    followers_count
    created_attos
    verified_type
    statuses_count
    statuslasttos
    friends_count
    idstr
    timein
    createdtimetos
    bi_followers_count
    favourites_count
    province
    userid
    comments_count
    gender'''
    from analysisStatFromRepostxt import fansum
    from analysisStatFromRepostxt import echouser
    from analysisStatFromRepostxt import lifespan
    
    timelist = es.get_attribute_values('createdtimetos')
#     if 'followerscount' in esatt:
#         fansumlist = es.get_attribute_values('followerscount')  
#     else:
#         print vsatt
#         fansumlist = vs.get_attribute_values('followerscount')
    fansumlist = es.get_attribute_values('followerscount') if 'followerscount' in esatt else vs.get_attribute_values('followerscount')
    useridlist = es.get_attribute_values('userid')
    mentioncntlist = es.get_attribute_values('mentimes') if 'mentimes' in esatt else vs.get_attribute_values('mentimes')
    bifansumlist = es.get_attribute_values('bifollowerscount')  if 'bifollowerscount' in esatt else vs.get_attribute_values('bifollowerscount')   
    friends_countlist = es.get_attribute_values('friendscount') if 'friendscount' in esatt else vs.get_attribute_values('friendscount')
    reposts_countlist = es.get_attribute_values('repostscount')
    
    fanscnt,fanscntavg = fansum(fansumlist,1)
    echousercnt = echouser(useridlist,1)
    durationlist,durationaddedlist,durationavglist,durationaddedavglist = lifespan(timelist,1)
    mentioncnt,mentioncntavg = fansum(mentioncntlist,1)
    bifansum,bifansumavg = fansum(bifansumlist,1)
    friends_count,friends_countavg = fansum(friends_countlist,1)
    reposts_count,reposts_countavg = fansum(reposts_countlist,1)

#     print fanscnt,echousercnt,fanscntavg,durationlist,durationavglist,mentioncnt,mentioncntavg,bifansum,bifansumavg,friends_count,friends_countavg,reposts_count,reposts_countavg
    return [fanscnt[0],echousercnt[0],fanscntavg[0],durationlist[0],durationavglist[0],mentioncnt[0],mentioncntavg[0],bifansum[0],bifansumavg[0],friends_count[0],friends_countavg[0],reposts_count[0],reposts_countavg[0]]
    
    
    
def getCorePart(g,condition):
#     g.delete_vertices(g.vs.select(_indegree_lt=condition))
    g.vs.select(_indegree_gt=0,_outdegree_gt=0)
    return g
    
def get_netlist(netAttribute,percentNetAttri,netlist):    
    percentNetAttri.extend(netAttribute)
    netlist.append(percentNetAttri)
    return netlist   
    
def analyze_one(cocfilename, coc_folder,gmlfolder,percentlist=[1],timeseriesfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\TimeSeries.txt',periodcnt=1,graphAll = None):
    #IN:one coc file
    #OUT:all the net attributes of this coc by all percent
    #Process:coc2list;percent2timepoint;???
    netlist = []
    netlistcore = []
    statlist = []
    statlistcore = []
    es=ig.EdgeSeq(graphAll)

    cocfilepath=cocfolder+cocfilename+'.coc'

    netAttribute_all = []
    netstat_all = []
    netAttribute_core = []
    netstat_core = []
    for percent in percentlist:
            percentNetAttri = []
            percentNetAttri.append(cocfilename)
            percentNetAttri.append(percent)

            g = graphAll#.subgraph_edges(es.select(createdtimetos_le = timep),delete_vertices=False)
            netAttribute_all = analysisNet(g)#grt.analysisNet(g)
            #netstat_all = analyzeNetStat(g)
            
            gg = clus.VertexClustering.giant(g.clusters(mode='weak'))          
            ggcore = getCorePart(gg,1)
            #print ggcore.vcount(),ggcore.ecount()
            netAttribute_core = analysisNet(ggcore)
            #netstat_core = analyzeNetStat(gg) 
            
            netlist_all = get_netlist(netAttribute_all,percentNetAttri[0:],netlist)    
            netlist_core = get_netlist(netAttribute_core,percentNetAttri[0:],netlistcore) 
            
            netstat_alllist = get_netlist(netstat_all,percentNetAttri[0:],statlist)    
            netstat_corelist = get_netlist(netstat_core,percentNetAttri[0:],statlistcore) 
            #print netstat_alllist
            #print   len(netAttribute_all),len(netAttribute_core),len(netstat_all),len(netstat_core) 

    return [zip(*netlist_all),zip(*netlist_core),zip(*netstat_alllist),zip(*netstat_corelist)]


def openAttributeFiles(netStatpath):
    vcountfile = open(netStatpath+'.vcount','a+')
    ecountfile = open(netStatpath+'.ecount','a+')
    density = open(netStatpath+'.density','a+')
    lenclustersmodeisweak = open(netStatpath+'.lenclustersmodeisweak','a+')
    lenclustersmodeisstrong = open(netStatpath+'.lenclustersmodeisstrong','a+')
    clusVertexClusteringiantclustersmodeisweakvcount = open(netStatpath+'.clusVertexClusteringiantclustersmodeisweakvcount','a+')
    clusVertexClusteringiantclustersmodeisweakecount = open(netStatpath+'.clusVertexClusteringiantclustersmodeisweakecount','a+')
    ecount2vcount = open(netStatpath+'.ecount2vcount','a+')
    transitivity_undirectedmodeis0 = open(netStatpath+'.transitivity_undirectedmodeis0','a+')
    average_path_length = open(netStatpath+'.average_path_length','a+')
    diameter = open(netStatpath+'.diameter','a+')
    assortativitydegreedirectedisfalse = open(netStatpath+'.assortativitydegreedirectedisfalse','a+')
    assortativitydegreedirectedistrue = open(netStatpath+'.assortativitydegreedirectedistrue','a+')
    assortativityindegreedirectedistrue = open(netStatpath+'.assortativityindegreedirectedistrue','a+')
    assortativityoutdegreedirectedistrue = open(netStatpath+'.assortativityoutdegreedirectedistrue','a+')
    degreePowerLawFitalpha = open(netStatpath+'.degreePowerLawFitalpha','a+')
    degreePowerLawFitxmin = open(netStatpath+'.degreePowerLawFitxmin','a+')
    degreePowerLawFitp = open(netStatpath+'.degreePowerLawFitp','a+')
    degreePowerLawFitL = open(netStatpath+'.degreePowerLawFitL','a+')
    degreePowerLawFitD = open(netStatpath+'.degreePowerLawFitD','a+')
    indegreePowerLawFitalpha = open(netStatpath+'.indegreePowerLawFitalpha','a+')
    indegreePowerLawFitxmin = open(netStatpath+'.indegreePowerLawFitxmin','a+')
    indegreePowerLawFitp = open(netStatpath+'.indegreePowerLawFitp','a+')
    indegreePowerLawFitL = open(netStatpath+'.indegreePowerLawFitL','a+')
    indegreePowerLawFitD = open(netStatpath+'.indegreePowerLawFitD','a+')
    outdegreePowerLawFitalpha = open(netStatpath+'.outdegreePowerLawFitalpha','a+')
    outdegreePowerLawFitxmin = open(netStatpath+'.outdegreePowerLawFitxmin','a+')
    outdegreePowerLawFitp = open(netStatpath+'.outdegreePowerLawFitp','a+')
    outdegreePowerLawFitL = open(netStatpath+'.outdegreePowerLawFitL','a+')
    outdegreePowerLawFitD = open(netStatpath+'.outdegreePowerLawFitD','a+')
    assority_followerscount = open(netStatpath+'.assority_followerscount','a+')
    assority_followerscount_d = open(netStatpath+'.assority_followerscount_d','a+')
    assority_bifollowerscount = open(netStatpath+'.assority_bifollowerscount','a+')
    assority_bifollowerscount_d = open(netStatpath+'.assority_bifollowerscount_d','a+')
    assority_friendscount = open(netStatpath+'.assority_friendscount','a+')
    assority_friendscount_d = open(netStatpath+'.assority_friendscount_d','a+')


################################################################################################################################################


    writer_vcountfile = csv.writer(vcountfile)
    writer_ecountfile = csv.writer(ecountfile)
    writer_density = csv.writer(density)
    writer_lenclustersmodeisweak = csv.writer(lenclustersmodeisweak)
    writer_lenclustersmodeisstrong = csv.writer(lenclustersmodeisstrong)
    writer_clusVertexClusteringiantclustersmodeisweakvcount = csv.writer(clusVertexClusteringiantclustersmodeisweakvcount)
    writer_clusVertexClusteringiantclustersmodeisweakecount = csv.writer(clusVertexClusteringiantclustersmodeisweakecount)
    writer_ecount2vcount = csv.writer(ecount2vcount)
    writer_transitivity_undirectedmodeis0 = csv.writer(transitivity_undirectedmodeis0)
    writer_average_path_length = csv.writer(average_path_length)
    writer_diameter = csv.writer(diameter)
    writer_assortativitydegreedirectedisfalse = csv.writer(assortativitydegreedirectedisfalse)
    writer_assortativitydegreedirectedistrue = csv.writer(assortativitydegreedirectedistrue)
    writer_assortativityindegreedirectedistrue = csv.writer(assortativityindegreedirectedistrue)
    writer_assortativityoutdegreedirectedistrue = csv.writer(assortativityoutdegreedirectedistrue)
    writer_degreePowerLawFitalpha = csv.writer(degreePowerLawFitalpha)
    writer_degreePowerLawFitxmin = csv.writer(degreePowerLawFitxmin)
    writer_degreePowerLawFitp = csv.writer(degreePowerLawFitp)
    writer_degreePowerLawFitL = csv.writer(degreePowerLawFitL)
    writer_degreePowerLawFitD = csv.writer(degreePowerLawFitD)
    writer_indegreePowerLawFitalpha = csv.writer(indegreePowerLawFitalpha)
    writer_indegreePowerLawFitxmin = csv.writer(indegreePowerLawFitxmin)
    writer_indegreePowerLawFitp = csv.writer(indegreePowerLawFitp)
    writer_indegreePowerLawFitL = csv.writer(indegreePowerLawFitL)
    writer_indegreePowerLawFitD = csv.writer(indegreePowerLawFitD)
    writer_outdegreePowerLawFitalpha = csv.writer(outdegreePowerLawFitalpha)
    writer_outdegreePowerLawFitxmin = csv.writer(outdegreePowerLawFitxmin)
    writer_outdegreePowerLawFitp = csv.writer(outdegreePowerLawFitp)
    writer_outdegreePowerLawFitL = csv.writer(outdegreePowerLawFitL)
    writer_outdegreePowerLawFitD = csv.writer(outdegreePowerLawFitD)

    writer_assority_followerscount = csv.writer(assority_followerscount)
    writer_assority_followerscount_d = csv.writer(assority_followerscount_d)
    writer_assority_bifollowerscount = csv.writer(assority_bifollowerscount)
    writer_assority_bifollowerscount_d = csv.writer(assority_bifollowerscount_d)
    writer_assority_friendscount = csv.writer(assority_friendscount)
    writer_assority_friendscount_d = csv.writer(assority_friendscount_d)
    

    writerlist = [writer_vcountfile, writer_ecountfile, writer_density, writer_lenclustersmodeisweak, writer_lenclustersmodeisstrong, writer_clusVertexClusteringiantclustersmodeisweakvcount, writer_clusVertexClusteringiantclustersmodeisweakecount, writer_ecount2vcount, writer_transitivity_undirectedmodeis0, writer_average_path_length, writer_diameter, writer_assortativitydegreedirectedisfalse, writer_assortativitydegreedirectedistrue, writer_assortativityindegreedirectedistrue, writer_assortativityoutdegreedirectedistrue, writer_degreePowerLawFitalpha, writer_degreePowerLawFitxmin, writer_degreePowerLawFitp, writer_degreePowerLawFitL, writer_degreePowerLawFitD, writer_indegreePowerLawFitalpha, writer_indegreePowerLawFitxmin, writer_indegreePowerLawFitp, writer_indegreePowerLawFitL, writer_indegreePowerLawFitD, writer_outdegreePowerLawFitalpha, writer_outdegreePowerLawFitxmin, writer_outdegreePowerLawFitp, writer_outdegreePowerLawFitL, writer_outdegreePowerLawFitD, writer_assority_followerscount, writer_assority_followerscount_d, writer_assority_bifollowerscount, writer_assority_bifollowerscount_d,writer_assority_friendscount,writer_assority_friendscount_d]
    #writerlist = [ writer_assority_followerscount, writer_assority_followerscount_d, writer_assority_bifollowerscount, writer_assority_bifollowerscount_d,writer_assority_friendscount,writer_assority_friendscount_d]

    return writerlist

def openStatAttributeFIles(netStatpath):
#     vcountfile = open(netStatpath+'.fansum','a+')
#     ecountfile = open(netStatpath+'.echouser','a+')
#     writer_vcountfile = csv.writer(vcountfile)
#     writer_ecountfile = csv.writer(ecountfile)
#     writerlist = [writer_vcountfile, writer_ecountfile]
    fanscntfile = open(netStatpath+'.fanscnt','a+')
    echousercntfile = open(netStatpath+'.echousercnt','a+')
    fanscntavgfile = open(netStatpath+'.fanscntavg','a+')
    durationlistfile = open(netStatpath+'.durationlist','a+')
    durationavglistfile = open(netStatpath+'.durationavglist','a+')
    mentioncntfile = open(netStatpath+'.mentioncnt','a+')
    mentioncntavgfile = open(netStatpath+'.mentioncntavg','a+')
    bifansumfile = open(netStatpath+'.bifansum','a+')
    bifansumavgfile = open(netStatpath+'.bifansumavg','a+')
    friends_countfile = open(netStatpath+'.friends_count','a+')
    friends_countavgfile = open(netStatpath+'.friends_countavg','a+')
    reposts_countfile = open(netStatpath+'.reposts_count','a+')
    reposts_countavgfile = open(netStatpath+'.reposts_countavg','a+')

    writer_fanscntfile = csv.writer(fanscntfile)
    writer_echousercntfile = csv.writer(echousercntfile)
    writer_fanscntavgfile = csv.writer(fanscntavgfile)
    writer_durationlistfile = csv.writer(durationlistfile)
    writer_durationavglistfile = csv.writer(durationavglistfile)
    writer_mentioncntfile = csv.writer(mentioncntfile)
    writer_mentioncntavgfile = csv.writer(mentioncntavgfile)
    writer_bifansumfile = csv.writer(bifansumfile)
    writer_bifansumavgfile = csv.writer(bifansumavgfile)
    writer_friends_countfile = csv.writer(friends_countfile)
    writer_friends_countavgfile = csv.writer(friends_countavgfile)
    writer_reposts_countfile = csv.writer(reposts_countfile)
    writer_reposts_countavgfile = csv.writer(reposts_countavgfile)
    
    writerlist = [writer_fanscntfile,writer_echousercntfile,writer_fanscntavgfile,writer_durationlistfile,writer_durationavglistfile,writer_mentioncntfile,writer_mentioncntavgfile,writer_bifansumfile,writer_bifansumavgfile,writer_friends_countfile,writer_friends_countavgfile,writer_reposts_countfile,writer_reposts_countavgfile]
    return writerlist
     
def createFolder(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    return folderpath

def deal_netAttribute(netAttribute,writerlist):
    a = zip(*netAttribute[0:2])
    b = netAttribute[2:]
    for metacol, attri in zip(*[a,b]):
        metacol = list(metacol)
        attri = list(attri)
        metacol.extend(attri)
#         print metacol
    
    netAttribute = list(netAttribute)
    namestr = netAttribute[0][0]
#     print 'netAttribute-',netAttribute
#                 netAttributeLine = [netAttribute[0][0],netAttribute[1][0]]
    for item,writer in zip(*[netAttribute[2:],writerlist]):
#                 print item,writer.__getattribute__(__name__)#__str__.__name__
#                 print item[0],item[1]
        itemNew = item#[2:]
        passedlinecnt = 0
        for i in range(len(percent)):
            percentstr = netAttribute[1][i*periodcnt]
            contentcol = list(itemNew[passedlinecnt:(passedlinecnt + periodcnt)])

            contentcol.insert(0,percentstr)
            contentcol.insert(0,namestr)
            passedlinecnt +=periodcnt

            writer.writerow(contentcol)  
#             print 'contentcol',contentcol                      
            
            contentcol = [] 

# gmlfolder = "I:\\dataset\\HFS_XunRen\\User\\GML\\"
# for filename in os.listdir(gmlfolder):
# 	#         filename = '3581866814344587.coc.gml'#G:\HFS\WeiboData\HFSWeiboNoCOC\ttt\3343740313561521.coc.gml
# 	filepath = gmlfolder+filename
# 	if os.path.splitext(filename)[1]=='.gml':
# 		filename = os.path.splitext(filename)[0]
# 		print filepath,time.asctime(),' reading... ',    time.clock()
#
# 		infile=open(gmlfolder+filename+'.gml')
# 		g=ig.Graph.Read_GML(infile)
# 		print analysisNet(g)
# 		er


if __name__ == '__main__':
    print 'All acting'
    time.clock()
    workfolder = "I:\\dataset\\HFS_XunRen\\User\\"#testNew2\\
    netStatpath = gt.createFolder(workfolder+"HFSWeiboStatNet\\")#'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\StatNet\\test\\'
    cocfolder = gt.createFolder(workfolder+"COC\\")#'G:\\HFS\\WeiboData\\HFSWeiboNoCOC\\test\\'#'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'
    #gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\GML\\SimGml\\'#G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\'#workfolder+"GML\\"#
    percent = [1.0]#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]#
    periodcnt = 1#20

    writerlist = openAttributeFiles(createFolder(netStatpath+'netgiant\\'))
    writerlist_core = openAttributeFiles(createFolder(netStatpath+'netcore\\'))
    writerlist_stat = openStatAttributeFIles(createFolder(netStatpath+'statgiant\\'))
    writerlist_statcore = openStatAttributeFIles(createFolder(netStatpath+'statcore\\'))

    filecnt = 0
    gmlfolder = "I:\\dataset\\HFS_XunRen\\User\\GML\\"
    metaline = [[r'userid,type,vcount,ecount,density,lengclustersmodeweak,lengclustersmodestrong,vcount,ecount,ecount2vcount,transitivity_undirectedmode0,straverage_path_length,diameter,assorDeg,assorDegD,assorInDeg,assorOutDeg,assorDegF,assorInDegF,assorOutDegF,degreePowerLawFitalpha,degreePowerLawFitxmin,degreePowerLawFitp,degreePowerLawFitL,degreePowerLawFitD,indegreePowerLawFitalpha,indegreePowerLawFitxmin,indegreePowerLawFitp,indegreePowerLawFitL,indegreePowerLawFitD,outdegreePowerLawFitalpha,outdegreePowerLawFitxmin,outdegreePowerLawFitp,outdegreePowerLawFitL,outdegreePowerLawFitD']]
    atts,atts_core =  metaline,metaline
    for filename in os.listdir(gmlfolder):
#         filename = '3581866814344587.coc.gml'#G:\HFS\WeiboData\HFSWeiboNoCOC\ttt\3343740313561521.coc.gml
        filepath = gmlfolder+filename
        if os.path.splitext(filename)[1]=='.gml' and filecnt<20:#
            filecnt+=1
            filename = os.path.splitext(filename)[0]
            print filecnt,filepath,time.asctime(),' reading... ',    time.clock()

            infile=open(gmlfolder+filename+'.gml')
            g = ig.Graph().as_directed()

            try:
                g = ig.Graph.Read_GML(infile)
                print time.asctime(),' readed'
            except Exception,e:
	            print filepath,e
	            continue


            infile.close()
            g.to_directed(True)
            print g.is_directed()
            if g.vcount()>0:
                # netAttributes = analyze_one(filename,coc_folder=cocfolder,gmlfolder=gmlfolder,percentlist=percent,timeseriesfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\TimeSeries_orgin.txt',periodcnt=periodcnt,graphAll=g)#_igraph 3508278808120380
                # netAttribute = netAttributes[0]
                # netAttribute_core = netAttributes[1]
                # statAttribute = netAttributes[2]
                # statAttribute_core = netAttributes[3]

                netAttribute_all,netAttribute_core = [filename,'all'],[filename,'core']

                netAttribute_all.extend(analysisNet(g))
                gg = clus.VertexClustering.giant(g.clusters(mode='weak'))
                ggcore = getCorePart(gg,1)
                netAttribute_core.extend(analysisNet(ggcore))

                atts.append(netAttribute_all)
                atts_core.append(netAttribute_core)
                print len(netAttribute_all),netAttribute_all
                print len(netAttribute_core),netAttribute_core

	#always save two same lines, I do not know why
    gt.saveList(atts,r'I:\dataset\HFS_XunRen\User\HFSWeiboStatNet\atts_directed.txt')
    gt.saveList(atts_core,r'I:\dataset\HFS_XunRen\User\HFSWeiboStatNet\atts_core_directed.txt')

                # deal_netAttribute(netAttribute,writerlist)
                # deal_netAttribute(netAttribute_core,writerlist_core)
                #
                # deal_netAttribute(statAttribute,writerlist_stat)
                # deal_netAttribute(statAttribute_core,writerlist_statcore)


if __name__ == '__main2__':
    print 'All acting'
    time.clock()
    workfolder = "G:\\HFS\\WeiboData\\HFSWeibo\\"#testNew2\\
    netStatpath = gt.createFolder(workfolder+"HFSWeiboStatNet\\")#'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\StatNet\\test\\'
    cocfolder = gt.createFolder(workfolder+"COC\\")#'G:\\HFS\\WeiboData\\HFSWeiboNoCOC\\test\\'#'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'
    gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\GML\\SimGml\\'#G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\'#workfolder+"GML\\"#
    percent = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]#
    periodcnt = 20
        
    writerlist = openAttributeFiles(createFolder(netStatpath+'netgiant\\'))
    writerlist_core = openAttributeFiles(createFolder(netStatpath+'netcore\\'))
    writerlist_stat = openStatAttributeFIles(createFolder(netStatpath+'statgiant\\'))
    writerlist_statcore = openStatAttributeFIles(createFolder(netStatpath+'statcore\\'))
    
    filecnt = 0
#     if 1:
    for filename in os.listdir(gmlfolder):
#         filename = '3581866814344587.coc.gml'#G:\HFS\WeiboData\HFSWeiboNoCOC\ttt\3343740313561521.coc.gml
        filepath = gmlfolder+filename
        if os.path.splitext(filename)[1]=='.gml':
            filecnt+=1
            filename = os.path.splitext(filename)[0]
            print filecnt,filepath,time.asctime(),' reading... ',    time.clock()

            infile=open(gmlfolder+filename+'.gml')
            g=ig.Graph.Read_GML(infile)
            print time.asctime(),' readed'

            infile.close()
            
            if g.ecount()>2:
                netAttributes = analyze_one(filename,coc_folder=cocfolder,gmlfolder=gmlfolder,percentlist=percent,timeseriesfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\TimeSeries_orgin.txt',periodcnt=periodcnt,graphAll=g)#_igraph 3508278808120380
                netAttribute = netAttributes[0]
                netAttribute_core = netAttributes[1]
                statAttribute = netAttributes[2]
                statAttribute_core = netAttributes[3]
                
                deal_netAttribute(netAttribute,writerlist)
                deal_netAttribute(netAttribute_core,writerlist_core)
                
                deal_netAttribute(statAttribute,writerlist_stat)
                deal_netAttribute(statAttribute_core,writerlist_statcore)

# vcountfile.close()
# ecountfile.close()
# density.close()
# lenclustersmodeisweak.close()
# lenclustersmodeisstrong.close()
# clusVertexClusteringiantclustersmodeisweakvcount.close()
# clusVertexClusteringiantclustersmodeisweakecount.close()
# ecount2vcount.close()
# transitivity_undirectedmodeis0.close()
# average_path_length.close()
# diameter.close()
# assortativitydegreedirectedisfalse.close()
# assortativitydegreedirectedistrue.close()
# assortativityindegreedirectedistrue.close()
# assortativityoutdegreedirectedistrue.close()
# degreePowerLawFitalpha.close()
# degreePowerLawFitxmin.close()
# degreePowerLawFitp.close()
# degreePowerLawFitL.close()
# degreePowerLawFitD.close()
# indegreePowerLawFitalpha.close()
# indegreePowerLawFitxmin.close()
# indegreePowerLawFitp.close()
# indegreePowerLawFitL.close()
# indegreePowerLawFitD.close()
# outdegreePowerLawFitalpha.close()
# outdegreePowerLawFitxmin.close()
# outdegreePowerLawFitp.close()
# outdegreePowerLawFitL.close()
# outdegreePowerLawFitD.close()

print time.asctime(),'all over'