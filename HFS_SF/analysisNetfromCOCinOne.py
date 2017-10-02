# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from tools import commontools as gtf
import csv
import re
import os
import time
#                             


import igraph as ig
from igraph.drawing.graph import DefaultGraphDrawer as igd
from igraph import clustering as clus
from igraph import statistics as stcs

    
gt=gtf()


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
        vcount=g.vcount()   
        ecount=g.ecount() 
#         degree=g.degree() 
#         indegree=g.indegree() 
#         outdegree=g.outdegree() 
#         degreePowerLawFit=stcs.power_law_fit(degree,method='auto',return_alpha_only=False) 
#         indegreePowerLawFit=stcs.power_law_fit(indegree, method='auto',return_alpha_only=False) 
#         outdegreePowerLawFit=stcs.power_law_fit(outdegree,method='auto',return_alpha_only=False)
#         assorDeg=g.assortativity(degree,directed= False) 
#         assorDegD=g.assortativity(degree,directed= True) 
#         assorInDeg=g.assortativity(indegree,directed= True)
#         assorOutDeg=g.assortativity(outdegree,directed= True)
        
    #     assorDegF='1' if assorDeg>0 else '-1'  
    #     assorInDegF='1' if assorInDeg>0 else '-1'   
    #     assorOutDegF= '1' if assorOutDeg>0 else '-1'          
    #     print g.average_path_length()
        return g.vcount(),g.ecount(),\
            str(g.average_path_length()),\
            str(g.diameter()),\
            str(len(g.clusters(mode='weak'))),\
            str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
            str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount())

        return [str(vcount),\
        str(ecount),\
        str(g.density()),\
        str(len(g.clusters(mode='weak'))),\
        str(len(g.clusters(mode='strong'))),\
        str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
        str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount()),\
        str((ecount*2)/float(vcount)),\
        str(g.transitivity_undirected(mode='0')) ,\
        str(g.average_path_length()),\
        str(g.diameter()),\
        str(assorDeg),\
        str(assorDegD),\
        str(assorInDeg),\
        str(assorOutDeg),\
    #     str(assorDegF),\
    #     str(assorInDegF),\
    #     str(assorOutDegF),\
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
        str(outdegreePowerLawFit.D)]
    except:
        return []

def createGml(cocfilepath,gmlfolder,cocfilename,keepold=True):     
    ##IN:gml source coc -cocfilepath, gmlfolder，gml文件名fpf
    ##OUT:gml文件
    gmlfilepath = gmlfolder+cocfilename+'.gml'
    if keepold and os.path.exists(gmlfilepath):
        print gmlfilepath,'has existed'
    else:
        coc2gml(cocfilepath,',',gmlfilepath)
#     drawgml(gmlfilepath) 
    return gmlfilepath

def selectCoc(cocfile,timepoint,newcocfile='temp.coc',edgetype=['01289','01289']):
    "IN:cocfilepath;the condition-timepoint;the newcocfilepath,the edge type selected"
    "OUT:new cocfile path"
    "Caution:the coc edges type should give the first type code and the second, which is 2 dimension list. default is the all,namely ['01289','01289']"
    newcocf = open(newcocfile,'w')
    writer = csv.writer(newcocf)
    for line in csv.reader(file(cocfile)):
        if edgetype!=['01289','01289']:
            if line[5] in list(edgetype[0]) and line[6] in list(edgetype[1]):
#             if line[5]=='2' and line[6]=='1':
                writer.writerow(line)
        else:
            writer.writerow(line)  
        if float(line[4])>float(timepoint) or float(line[4])==float(timepoint):
            break
    newcocf.close()
    return newcocfile

def selectcoc_fromlist_bytime(coclist,timepoint=0):
    coclistmp = []
    linecnt = 0
    for line in coclist:
        if float(line[4])>float(timepoint) or float(line[4])==float(timepoint):
            break
        else:
           linecnt+=1 
    coclistmp = coclist[0:linecnt]
    return coclistmp

           
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
                    

def analyze_one(cocfilename, coc_folder,gmlfolder,percentlist=[1],timeseriesfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\TimeSeries.txt',periodcnt=1,graphAll = None):
    #IN:one coc file
    #OUT:all the net attributes of this coc by all percent
    #Process:coc2list;percent2timepoint;???
    netlist = []
    es=ig.EdgeSeq(graphAll)
    cocfilepath=cocfolder+cocfilename+'.coc'
#     timelist = selecTimelist(findstr=cocfilename,timeSeriesFilepath=timeseriesfile)
    timelist = es.get_attribute_values('createdtimetos')
    timelist.sort()
#     vfg = gt.csv2list(cocfolder+cocfilename)
#     vfg.reverse()
#     timelist = gt.selectColfromList(vfg, 4, 5)
    for percent in percentlist:
        lengthNow = int(round(len(timelist)*percent))
        lengthNow = lengthNow if lengthNow>1 else 1
        timelistPercentNow = timelist[:lengthNow]
        timelistPeriodNow = selecTime(timelistPercentNow,periodcnt)
        for timep in timelistPeriodNow:
            timep = str(timep)
            percentNetAttri = []
            percentNetAttri.append(cocfilename)
            percentNetAttri.append(percent)
            
#             selectedCoc = selectCoc(cocfilepath,timep)    
#             gmlfilepath = createGml(selectedCoc,gmlfolder='',cocfilename='temp',keepold=False)    
#             g=ig.Graph.Read_GML(gmlfilepath)
            
            #选择子网络
        #     print es.attribute_names()
#             print timep
            g = graphAll.subgraph_edges(es.select(createdtimetos_le = timep))
            
#             x = []
#             y = []
#             j = 0
#             goutdegree = g.outdegree()
#             for i in g.indegree():
#                 if i>0:
#                     x.append(i)
#                     y.append(goutdegree[j])
#                 j+=1
#                     
#             plt.scatter(x,y)
#             plt.show()
#             print '==============================='
#             j = ''
#             print  g.vcount(),g.ecount()
#             for i in g.vs:
#                 j+= i['label']+';'
#             print j
#             print g
            netAttribute = analysisNet(g)    
            percentNetAttri.extend(netAttribute)
            netlist.append(percentNetAttri)    
#     print netlist
    return zip(*netlist)
    
# def analyze_one_igraph(cocfilename, coc_folder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\',gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGML\\',percentlist=[1],timeseriesfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\TimeSeries.txt',periodcnt=1):
#     #IN:one coc file
#     #OUT:all the net attributes of this coc by all percent
#     #Process:coc2list;percent2timepoint;???
#     netlist = []
#     cocfilepath=cocfolder+cocfilename+'.coc'
#     coclist = gt.csv2list_new(cocfilepath)
#     timelist = selecTimelist(findstr=cocfilename,timeSeriesFilepath=timeseriesfile)
#     g = ig.Graph() 
#     for percent in percentlist:
#         lengthNow = int(round(len(timelist)*percent))
#         lengthNow = lengthNow-1 if lengthNow>1 else 1
#         timelistPercentNow = timelist[:lengthNow]
#         timelistPeriodNow = selecTime(timelistPercentNow,periodcnt)
#         for timep in timelistPeriodNow:
#             percentNetAttri = []
#             percentNetAttri.append(cocfilename.split('.')[0])
#             percentNetAttri.append(percent)
# #             selectedCoc = selectCoc(cocfilepath,timep)
#             coclistmp = selectcoc_fromlist_bytime(coclist,timep)
# #             print len(coclistmp),'==============================='
#             g = coclist2graph(coclistmp,g = ig.Graph())
# #             j = ''
# #             print  g.vcount(),g.ecount()
# #             for i in g.vs:
# #                 j+= i['name']+';'
# #             print j
#             netAttribute = analysisNet(g)    
#             percentNetAttri.extend(netAttribute)
#             netlist.append(percentNetAttri)    
#     return zip(*netlist)

    
if __name__ == '__main__':
    print 'All acting'
    time.clock()
#     cocfile = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\3343744527348953.coc'
#     timep = selecTime('3343744527348953',percent=0.8,timeseriesfile = r'G:\HFS\WeiboData\Statistics\data4paper\timeseries\.timeline')
#     selectedCoc = selectCoc(cocfile,timep)
#     netAttribute = analyze_one(selectedCoc,cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\')#3508278808120380
#     print netAttribute
#     writer = csv.writer(file('G:\\HFS\\WeiboData\\HFSWeiboStatNet\\NetAttribute.txt','a'))
#     writer.writerow(netAttribute)
#     writer = csv.writer(file('G:\\HFS\\WeiboData\\HFSWeiboStatNet\\NetAttribute.txt','a'))
################################################################################################################################################
    netStatpath = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\test\\'
    cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboNoCOC\\'#'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'
    gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\test\\'
    percent = [1.0]#0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
    periodcnt = 5
        
    vcountfile = open(netStatpath+'.vcount','w')
    ecountfile = open(netStatpath+'.ecount','w')
    density = open(netStatpath+'.density','w')
    lenclustersmodeisweak = open(netStatpath+'.lenclustersmodeisweak','w')
    lenclustersmodeisstrong = open(netStatpath+'.lenclustersmodeisstrong','w')
    clusVertexClusteringiantclustersmodeisweakvcount = open(netStatpath+'.clusVertexClusteringiantclustersmodeisweakvcount','w')
    clusVertexClusteringiantclustersmodeisweakecount = open(netStatpath+'.clusVertexClusteringiantclustersmodeisweakecount','w')
    ecount2vcount = open(netStatpath+'.ecount2vcount','w')
    transitivity_undirectedmodeis0 = open(netStatpath+'.transitivity_undirectedmodeis0','w')
    average_path_length = open(netStatpath+'.average_path_length','w')
    diameter = open(netStatpath+'.diameter','w')
    assortativitydegreedirectedisfalse = open(netStatpath+'.assortativitydegreedirectedisfalse','w')
    assortativitydegreedirectedistrue = open(netStatpath+'.assortativitydegreedirectedistrue','w')
    assortativityindegreedirectedistrue = open(netStatpath+'.assortativityindegreedirectedistrue','w')
    assortativityoutdegreedirectedistrue = open(netStatpath+'.assortativityoutdegreedirectedistrue','w')
    degreePowerLawFitalpha = open(netStatpath+'.degreePowerLawFitalpha','w')
    degreePowerLawFitxmin = open(netStatpath+'.degreePowerLawFitxmin','w')
    degreePowerLawFitp = open(netStatpath+'.degreePowerLawFitp','w')
    degreePowerLawFitL = open(netStatpath+'.degreePowerLawFitL','w')
    degreePowerLawFitD = open(netStatpath+'.degreePowerLawFitD','w')
    indegreePowerLawFitalpha = open(netStatpath+'.indegreePowerLawFitalpha','w')
    indegreePowerLawFitxmin = open(netStatpath+'.indegreePowerLawFitxmin','w')
    indegreePowerLawFitp = open(netStatpath+'.indegreePowerLawFitp','w')
    indegreePowerLawFitL = open(netStatpath+'.indegreePowerLawFitL','w')
    indegreePowerLawFitD = open(netStatpath+'.indegreePowerLawFitD','w')
    outdegreePowerLawFitalpha = open(netStatpath+'.outdegreePowerLawFitalpha','w')
    outdegreePowerLawFitxmin = open(netStatpath+'.outdegreePowerLawFitxmin','w')
    outdegreePowerLawFitp = open(netStatpath+'.outdegreePowerLawFitp','w')
    outdegreePowerLawFitL = open(netStatpath+'.outdegreePowerLawFitL','w')
    outdegreePowerLawFitD = open(netStatpath+'.outdegreePowerLawFitD','w')

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

    writerlist = [writer_vcountfile, writer_ecountfile, writer_density, writer_lenclustersmodeisweak, writer_lenclustersmodeisstrong, writer_clusVertexClusteringiantclustersmodeisweakvcount, writer_clusVertexClusteringiantclustersmodeisweakecount, writer_ecount2vcount, writer_transitivity_undirectedmodeis0, writer_average_path_length, writer_diameter, writer_assortativitydegreedirectedisfalse, writer_assortativitydegreedirectedistrue, writer_assortativityindegreedirectedistrue, writer_assortativityoutdegreedirectedistrue, writer_degreePowerLawFitalpha, writer_degreePowerLawFitxmin, writer_degreePowerLawFitp, writer_degreePowerLawFitL, writer_degreePowerLawFitD, writer_indegreePowerLawFitalpha, writer_indegreePowerLawFitxmin, writer_indegreePowerLawFitp, writer_indegreePowerLawFitL, writer_indegreePowerLawFitD, writer_outdegreePowerLawFitalpha, writer_outdegreePowerLawFitxmin, writer_outdegreePowerLawFitp, writer_outdegreePowerLawFitL, writer_outdegreePowerLawFitD]

    filecnt = 0
#     if 1:
    for filename in os.listdir(gmlfolder):
#         filename = '3581866814344587.coc.gml'#G:\HFS\WeiboData\HFSWeiboNoCOC\ttt\3343740313561521.coc.gml
        filepath = gmlfolder+filename
        if os.path.splitext(filename)[1]=='.gml':
            filecnt+=1
            filename = os.path.splitext(filename)[0]
            print filecnt,filepath,' starting... ',    time.clock()
#             try:                
            infile=open(gmlfolder+filename+'.gml')
            print 'er'
            g=ig.Graph.Read_GML(infile)
            
            print 'ert'
#             except Exception,e:
#                 print infile,'error',e
            infile.close()
            
            if g.ecount()>2:
                netAttribute = analyze_one(filename,coc_folder=cocfolder,gmlfolder=gmlfolder,percentlist=percent,timeseriesfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\TimeSeries_orgin.txt',periodcnt=periodcnt,graphAll=g)#_igraph 3508278808120380
    #                 netAttribute.insert(1,percents/10.0)
    #             print netAttribute
                a = zip(*netAttribute[0:2])
                b = netAttribute[2:]
                for metacol, attri in zip(*[a,b]):
                    metacol = list(metacol)
                    attri = list(attri)
                    metacol.extend(attri)
    #                 print metacol
                
                netAttribute = list(netAttribute)
                namestr = netAttribute[0][0]
    #                 netAttributeLine = [netAttribute[0][0],netAttribute[1][0]]
                for item,writer in zip(*[netAttribute[2:],writerlist]):
    #                 print item,writer.__getattribute__(__name__)#__str__.__name__
    #                 print item[0],item[1]
                    itemNew = item#[2:]
                    passedlinecnt = 0
    #                 for i,writer in zip(*[range(len(percent)),writerlist]):
                    for i in range(len(percent)):
                        percentstr = netAttribute[1][i*periodcnt]
                        contentcol = list(itemNew[passedlinecnt:(passedlinecnt + periodcnt)])
    #                         metacol = netAttributeLine[0:2]
    #                         metacol.extend(contentcol)
                        contentcol.insert(0,percentstr)
                        contentcol.insert(0,namestr)
                        passedlinecnt +=periodcnt
    #                         writer.writerow(contentcol)
    #                     print contentcol,'\n================================================================================='
    #                     for writer in writerlist:
                        writer.writerow(contentcol)
    #                     writer_vcountfile.writerow(contentcol[0])
    #                     writer_ecountfile.writerow(contentcol[1])
    #                     writer_density.writerow(contentcol[2])
    #                     writer_lenclustersmodeisweak.writerow(contentcol[3])
    #                     writer_lenclustersmodeisstrong.writerow(contentcol[4])
    #                     writer_clusVertexClusteringiantclustersmodeisweakvcount.writerow(contentcol[5])
    #                     writer_clusVertexClusteringiantclustersmodeisweakecount.writerow(contentcol[6])
    #                     writer_ecount2vcount.writerow(contentcol[7])
    #                     writer_transitivity_undirectedmodeis0.writerow(contentcol[8])
    #                     writer_average_path_length.writerow(contentcol[9])
    #                     writer_diameter.writerow(contentcol[10])
    #                     writer_assortativitydegreedirectedisfalse.writerow(contentcol[11])
    #                     writer_assortativitydegreedirectedistrue.writerow(contentcol[12])
    #                     writer_assortativityindegreedirectedistrue.writerow(contentcol[13])
    #                     writer_assortativityoutdegreedirectedistrue.writerow(contentcol[14])
    #                     writer_degreePowerLawFitalpha.writerow(contentcol[15])
    #                     writer_degreePowerLawFitxmin.writerow(contentcol[16])
    #                     writer_degreePowerLawFitp.writerow(contentcol[17])
    #                     writer_degreePowerLawFitL.writerow(contentcol[18])
    #                     writer_degreePowerLawFitD.writerow(contentcol[19])
    #                     writer_indegreePowerLawFitalpha.writerow(contentcol[20])
    #                     writer_indegreePowerLawFitxmin.writerow(contentcol[21])
    #                     writer_indegreePowerLawFitp.writerow(contentcol[22])
    #                     writer_indegreePowerLawFitL.writerow(contentcol[23])
    #                     writer_indegreePowerLawFitD.writerow(contentcol[24])
    #                     writer_outdegreePowerLawFitalpha.writerow(contentcol[25])
    #                     writer_outdegreePowerLawFitxmin.writerow(contentcol[26])
    #                     writer_outdegreePowerLawFitp.writerow(contentcol[27])
    #                     writer_outdegreePowerLawFitL.writerow(contentcol[28])
    
                        
                        
                        contentcol = []
                            
    #                     writer.writerow(item)
    #             except Exception,e:
    #                 print e
                
#         print filepath,' over',    time.clock()

vcountfile.close()
ecountfile.close()
density.close()
lenclustersmodeisweak.close()
lenclustersmodeisstrong.close()
clusVertexClusteringiantclustersmodeisweakvcount.close()
clusVertexClusteringiantclustersmodeisweakecount.close()
ecount2vcount.close()
transitivity_undirectedmodeis0.close()
average_path_length.close()
diameter.close()
assortativitydegreedirectedisfalse.close()
assortativitydegreedirectedistrue.close()
assortativityindegreedirectedistrue.close()
assortativityoutdegreedirectedistrue.close()
degreePowerLawFitalpha.close()
degreePowerLawFitxmin.close()
degreePowerLawFitp.close()
degreePowerLawFitL.close()
degreePowerLawFitD.close()
indegreePowerLawFitalpha.close()
indegreePowerLawFitxmin.close()
indegreePowerLawFitp.close()
indegreePowerLawFitL.close()
indegreePowerLawFitD.close()
outdegreePowerLawFitalpha.close()
outdegreePowerLawFitxmin.close()
outdegreePowerLawFitp.close()
outdegreePowerLawFitL.close()
outdegreePowerLawFitD.close()

print 'all over'