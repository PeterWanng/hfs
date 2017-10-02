#encoding=utf8
import sys
sys.path.append("..\..")
from tools import commontools
from matplotlib import pyplot as plt
import numpy as np

               
import igraph as ig
from igraph import clustering as clus


import re
import time
import os
from tools import commontools as gtf
import csv
import scipy as sp 
from scipy import stats
gt=gtf()

class NetypesfromGraph2():
    def geomean(nums):
        return sp.stats.mstats.gmean(a=nums)
#         return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))
         
    nums = (1,2,3,4,5)
    print geomean(nums)
    
class NetypesfromGraph():
    def geomean(self,nums):
        return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))
         
    def mode(self,numbers): 
        '''Return the mode of the list of numbers.''' 
        #Find the value that occurs the most frequently in a data set 
        freq={} 
        for i in range(len(numbers)): 
            try: 
                freq[numbers[i]] += 1 
            except KeyError: 
                freq[numbers[i]] = 1 
        max = 0
        mode = None 
        for k, v in freq.iteritems(): 
            if v > max: 
                max = v 
                mode = k 
        return mode
    
    def kmeans(self,x,k=4,show=False,runtimes=100):
        import numpy as np
        import pylab as pl
        from sklearn.cluster import KMeans,MiniBatchKMeans
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.datasets.samples_generator import make_blobs
        
    
    #     np.random.seed(0)
    #     centers = [[1,1], [-1,-1], [1, -1], [-1, 1]]
    #     k = len(centers)
    #     x , labels = make_blobs(n_samples=3000, centers=centers, cluster_std=.7)
#         print x
        
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init = runtimes)
        t0 = time.time()
        kmeans.fit(x)
        t_end = time.time() - t0
        
#         print kmeans.labels_
        
        if show:
            colors = ['r', 'b', 'g','y','c','m','k','r', 'b', 'g','y','c','m','k']
            for k , col, label in zip( range(k) , colors,kmeans.labels_):
                members = (kmeans.labels_ == k )
        #         pl.plot( x[members, 0] , x[members,1] , 'w', markerfacecolor=col, marker='.')
        #         pl.plot( x[members, 0] , x[members,1] , 'w', markerfacecolor=col, marker='.')
                
                pl.plot(kmeans.cluster_centers_[k,0], kmeans.cluster_centers_[k,1], 'o', markerfacecolor=col,\
                        markeredgecolor='k', markersize=10)
        
                xy = zip(*x)
        #         pl.plot(xy[2],xy[1])
#                 print xy[0],xy[1]
            pl.plot(xy[0],xy[1],linestyle='o',markerfacecolor=colors[label], markeredgecolor=colors[label], markersize=4)
                
            pl.show()
        return kmeans.labels_,kmeans.cluster_centers_,kmeans.fit
        
    
    def insertFig(self,figpath,figInserto):
        import PIL.Image as Image
        
        im = Image.open(figpath)#r'G:\HFS\WeiboData\HFSWeiboGMLNew\shape\3455798066008083.coc.gml.png')
        orginSize = im.size
        height = im.size[1]
    #     im.resize([orginSize[0]/1,orginSize[1]/2])
        im.thumbnail([orginSize[0]/1,orginSize[1]/2])
        # im.show()
        # We need a float array between 0-1, rather than
        # a uint8 array between 0-255
        
        # With newer (1.0) versions of matplotlib, you can 
        # use the "zorder" kwarg to make the image overlay
        # the plot, rather than hide behind it... (e.g. zorder=10)
        
        figInserto.figimage(im, 10, 80)#fig.bbox.ymax - height)
        
        return figInserto
    
    def openfig(self,figpath,label):
        import PIL.Image as Image
        
        im = Image.open(figpath)#r'G:\HFS\WeiboData\HFSWeiboGMLNew\shape\3455798066008083.coc.gml.png')
        im.show()
        time.sleep(1)
    
    
    def indexofavg(self,lista,sorted=True,avg=None,geomean=0,harmean=0):
        "input: number list; need sorted or not; avg value have provided or not"
        "output:avg value index; the distance between avg"
        indexnum,dis = 0,0
        
        if sorted:
            lista.sort()
        if not avg:
            if not geomean and not harmean:
                avg = np.average(lista)
            if geomean:
                avg = geomean(lista)
            if harmean:
                avg = stats.hmean(lista)
        
        for i in range(len(lista)):
            if lista[i]>=avg:
                indexnum = i
                if i<len(lista)-1 and i>0:
                    dis = lista[i]-lista[i-1]
                break
                
        return indexnum,dis 
    
    def getCorePart_deg(self,g,condition):
        g.delete_vertices(g.vs.select(_degree_lt=condition))
        return g
    
    def getCorePart_inoutdeg(self,g,condition):
        gg = g.copy()
    #     g.vs["inoutdeg"] = g.indegree()*g.outdegree()
    #     g.delete_vertices(g.vs.select(_inoutdeg_lt=condition))
        gg.delete_vertices(gg.vs.select(_indegree_lt=condition,_outdegree_lt=condition))
#         g.vs.select(_indegree_gt=condition,_outdegree_gt=condition)
        return gg 

    def netC(self,graphme):
        
        "connectivity"
        print gg.reciprocity(ignore_loops=True, mode="ratio")#one value;mode-default,ratio; Reciprocity defines the proportion of mutual connections in a directed graph.

 
    def netCentrity(self,graphme):
        'degree,betweenness,coreness,closeness,eccentricity,pagerank'
        result = []
        gg = graphme.simplify(multiple=True, loops=True, combine_edges=None)#Simplifies a graph by removing self-loops and/or multiple edges.
        
        "centrility"
        result.append(gg.degree())
         
        result.append(gg.betweenness(vertices=None, directed=True, cutoff =None, weights=None,nobigint=True))
        result.append(gg.coreness(mode='ALL'))#from gg.k_core();same as result.append(gg.shell_index(mode='ALL')---Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
         
        result.append(gg.closeness(vertices=None, mode='ALL', cutoff =None, weights=None))#cutoff
#         result.append(gg.eigenvector_centrality(directed=True, scale=True, weights=None,return_eigenvalue=False))#just shihe undirected graph, pagerank and hits  for directed.
        
        result.append(gg.eccentricity(vertices=None, mode='ALL'))
#         result.append(gg.radius(mode="ALL"))#one value
        
        result.append(gg.pagerank(vertices=None, directed=True, damping=0.85, weights=None))#,arpack_options=None
        
        return result
    
    
    def net_tree_star_line(self,graphme):
        result = []
        gg = graphme.simplify(multiple=True, loops=True, combine_edges=None)#Simplifies a graph by removing self-loops and/or multiple edges.
        indeg = gg.indegree()
        outdeg = gg.outdegree()
        indegavg = np.average(indeg)
        outdegavg = np.average(outdeg) 
        vcnt = gg.vcount()       
        
        a,b1,b2,c1,c2 = 0,0,0,0,0
        "new metrics"
        '(Ind-outd)**2=0;N for tree;N**2'
        'Degree vs average avgdegree; Indegree   Sum (D-d)**2 = 0 ignore indegree <=1; This is N for star; -N for Line'
        for ind,outd in zip(*(indeg,outdeg)):
            a += (ind-outd)**2
            b1 +=((ind-1)-indegavg)**2
            b2 +=((outd-1)-outdegavg)**2  
        
        "For pairs (a,b), sum(indega-indegb)**2=N**3;sum(outdega-outdegb)**2=N;1 for line and ** for tree??"
        edgelists = gg.get_edgelist()
        for pair in edgelists:
        #     c+=(g.indegree(pair[0])-g.indegree(pair[1]))**2
            c1+=(indeg[pair[0]]-indeg[pair[1]])**2
            c2+=(outdeg[pair[0]]-outdeg[pair[1]])**2
        
        a,b1,b2,c1,c2 = a**0.5/vcnt,b1**0.5/vcnt,b2**0.5/vcnt,c1**0.5/vcnt,c2**0.5/vcnt
        result.append([a,b1,b2,c1,c2])        
        return a,b1,b2,c1,c2 #result

    def metrics1(self,gg,ggcore,deg,avgv,deglen):        
        #         fig = plt.figure()
        #         
        #         plt.subplot(243)
        #         plt.semilogy(range(len(deg)),deg)
        #         plt.plot([1,len(deg)],[avgv,geomeanv,])
        #         
        #         "degree ratio"
        #         degratio = list_ratio(deg)
        #         plt.subplot(247)
        #         plt.plot(range(len(degratio)),degratio)
        #         degratio.sort(reverse=True)
        #         plt.semilogx(range(len(degratio)),degratio)
            
            "degree sequence avg,std,nodes cnt"
            avgdegindex = self.indexofavg(deg,sorted=False,avg = avgv,geomean=1)
        #             print avgv,geomean(deg),stats.hmean(deg),avgdegindex[0],len(deg),avgdegindex[1]
            avgdegindex_above,dist_avg_above = avgdegindex[0],avgdegindex[1]
        #             print avgdegindex_above,len(deg)
            deg_abovepart = deg[avgdegindex_above:]
            lendegabove = float(len(deg_abovepart))
        #             degree_about.append([filen,lendegabove,np.average(deg[avgdegindex_above:])/float(deglen),np.std(deg[avgdegindex_above:])/float(deglen)])#len(deg[avgdegindex_above:]),
        #             degree_about.append([filen,np.average(deg[avgdegindex_above:])/float(lendegabove),np.std(deg[avgdegindex_above:])/float(lendegabove),lendegabove,])#len(deg[avgdegindex_above:]),
        #             degree_about.append([filen,geomean(deg[avgdegindex_above:])/float(deglen),np.std(deg[avgdegindex_above:])/float(deglen)])#len(deg[avgdegindex_above:]),
        #             degree_about.append([filen,geomean(deg[avgdegindex_above:]),np.std(deg[avgdegindex_above:])])#len(deg[avgdegindex_above:]),
        #             degree_about.append([filen,len(deg[avgdegindex_above:]),np.std(deg[avgdegindex_above:])])#len(deg[avgdegindex_above:]),
            
            
            "kcore"
        #             print gg.k_core()
            
            "big degree nodes distance"
            ggcore = self.getCorePart_deg(gg,avgv)
        #             gt.drawgraph(ggcore,giantornot=False)
            nodedis = ggcore.average_path_length()
    #         nodedis = 0 if np.isnan(nodedis) else nodedis
    #         nodedis = -1 if np.isinf(nodedis) else nodedis
            
            assor = ggcore.assortativity(ggcore.degree(),directed= False)
    #         assor = 2 if np.isnan(assor) else assor
    #         assor = -2 if np.isinf(assor) else assor
    #         print assor
            
            result1 = [len(deg[avgdegindex_above:]),lendegabove/deglen,np.average(deg_abovepart)/lendegabove,np.std(deg_abovepart)/lendegabove,nodedis/lendegabove,assor]
        #             result.append([filen,lendegabove/deglen,np.average(deg_abovepart)/deglen,np.std(deg_abovepart)/deglen,nodedis])#len(deg[avgdegindex_above:]),
            result1 = sp.nan_to_num(result1)
            return result1
            
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
    #         print g.average_path_length()
    #         return g.vcount(),g.ecount(),\
    #             str(g.average_path_length()),\
    #             str(g.diameter()),\
    #             str(len(g.clusters(mode='weak'))),\
    #             str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
    #             str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount())
    
           "centrility"
           gg.degree()
           
           gg.betweenness(vertices=None, directed=True, cutoff =None, weights=None,nobigint=True)
           
           gg.kcore()
           gg.shell_index(mode=ALL)
           gg.coreness(mode=ALL)#Reference: Vladimir Batagelj, Matjaz Zaversnik: An O(m) Algorithm for Core Decomposition of Networks.
           
           gg.closeness(vertices=None, mode=ALL, cutoff =None, weights=None)#cutoff
           gg.eigenvector_centrality(directed=True, scale=True, weights=None,return_eigenvalue=False, arpack_options=None)
           
           gg.eccentricity(vertices=None, mode=ALL)
           gg.radius(mode=OUT)
           
           gg.pagerank(vertices=None, directed=True, damping=0.85, weights=None,arpack_options=None)
#            gg.hist
           
           return [str(vcount),\
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
           str(outdegreePowerLawFit.D)]
       except:
           return []

    
    def featuresofgraph(self,graphinstance,spanningtree=False):
        result = []
        "metrics vectors"
        degree_about = []
        nodedis = []
        g = graphinstance
    #         g = ig.Graph.Read_GML(gmlfolder+filen+'.coc.gml')
    #     g = ig.Graph.Read_GML(gmlfolder+filen)
        # print analysisNet(g)
        gg = clus.VertexClustering.giant(g.clusters(mode='weak'))          
    #         ggcore = getCorePart_indeg(gg,1)
        ggcore = self.getCorePart_inoutdeg(gg,2)
        ggcore = clus.VertexClustering.giant(ggcore.clusters(mode='weak'))
        if spanningtree:
            ggcore = ig.Graph.spanning_tree(ggcore)          
            ggcore = clus.VertexClustering.giant(ggcore.clusters(mode='weak'))
#             gsp=ig.Graph.spanning_tree(gg)
#         print g.vcount(),gg.vcount(),ggcore.vcount()
        "Degree order list"
        deg = gg.degree()
        deg.sort()
        deglen = float(len(deg))
        geomeanv = round(sp.stats.mstats.gmean(deg),2)#round(self.geomean(deg),2)
        avgv = round(np.average(deg),2)#,np.amax(deg),np.amin(deg)

        result1 = self.metrics1(gg, ggcore, deg, avgv,deglen)#return 6 dimision
        netcentri = self.netCentrity(gg)#return 6 dimision
        result2 = sp.average(netcentri,axis=1)#sp.stats.mstats.gmean(netcentri,axis=1)#
        result3 = self.net_tree_star_line(gg)#return 5 dimision
        result4 = self.net_tree_star_line(ggcore)#return 5 dimision
#         print result1,'\n',result2,'\n',result3,'\n',result4,'\n\n',
        result.extend(result1)
        result.extend(result2)
        result.extend(result3)
        result.extend(result4)
        return result
    
    def define_metrics_gml(self,gmlfolder,pngfolder):
        result = []            
        for filen in os.listdir(pngfolder):
            filen = str(filen).replace('.png','')
    
            if os.path.isfile(gmlfolder+filen) and os.path.splitext(filen)[-1]=='.gml' and os.stat(gmlfolder+filen).st_size<500000:
                resultone = [filen]
                filesize = os.path.getsize(gmlfolder+filen)
                print filen,filesize
                g = ig.Graph.Read_GML(gmlfolder+filen)
                resultone.extend(sp.nan_to_num(self.featuresofgraph(g)))
#                 resultone.extend(sp.nan_to_num(self.net_tree_star_line(g)))
                result.append(resultone)
        return result        

    def submatrix(self,mat,n,colstartindex=0,rowstartindex=0):
        "IN: square matrix; sub rows and cols count;the begin index"
        "OUT: sub square matrix between "
        mat = mat[rowstartindex:(rowstartindex+n)]
        mat = zip(*mat)
        mat = mat[colstartindex:(colstartindex+n)]
        mat = zip(*mat)
        return mat

    def define_metrics_adj_one(self,adjfile,ispanningtree=False,personcnt=100):
        resultone = []
        if os.path.isfile(adjfile) and os.path.splitext(adjfile)[-1]=='.adj':
            resultone = [adjfile]
            mat = np.genfromtxt(adjfile).tolist()
            
            if ispanningtree:
                mat = self.submatrix(mat,personcnt)

            g = ig.Graph.Adjacency(mat)#ig.Graph.Read_Adjacency(adjfile)#.tolist()
            resultone.extend(sp.nan_to_num(self.featuresofgraph(g,spanningtree=ispanningtree)))
#                 resultone.extend(sp.nan_to_num(self.net_tree_star_line(g)))
        return resultone
                
    def define_metrics_adj(self,adjfolder,ispanningtree=False,personcnt=100):
        result = []            
        for filen in os.listdir(adjfolder):
            resultone = self.define_metrics_adj_one(adjfolder+filen,ispanningtree=False,personcnt=personcnt)
            result.append(resultone)
        return result      
    '----------------------------------------------------------------------------------------------------------------------------------'
    
    def classfig(self,labels,x,z):
        # datafig = zip(*(xy,labels))
        # datafig.sort(key=lambda x:x[1])
        # for d,l in datafig:
        #     print d,l
        # print xyz,labels
        # labelsdis = np.histogram(labels, range=None, normed=True, weights=None, density=None)#,bins=(numpy.max(lista)+1)/binsdivide)
        # print labelsdis
        lenlabel = len(labels)
        
        labelfig = zip(*(z,labels))
        labelfig.sort(key=lambda x:x[1])
        
        gt.createFolder(folderName='G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\shape\\tesTypes',keepold=False)
        # figs = plt.figure()
        i = 0
        for fig,label in labelfig:
            i+=1
        #     print fig,label
        #     ax = figs.add_subplot(3,lenlabel/3,i)
        #     plt.subplot(3,lenlabel/3,i)
            figpath = 'G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\test\\'+ fig[0] +'.png'
            gt.copyfile(figpath,'G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\shape\\tesTypes\\'+ str(label)+'_'+fig[0] +'.png')
        #     openfig(figpath,label)
        #     insertFig(figpath,figs)
        
        os.startfile('G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\shape\\tesTypes\\')
        # plt.show()  
    

    def start(self,graphinstance):
        resultone = ['filen']
        g = graphinstance#ig.Graph.Read_GML(gmlfolder+filen)
        resultone.extend(self.featuresofgraph(g))

        return resultone
        

def first(x,attfilepath):
        xyz = zip(*x)
        xyzz = gt.normlistlist(listlista=xyz[1:],metacolcount=0,sumormax='max')#xyz[1:]#
        xy = zip(*xyzz)#[1:]
#         z = zip(*xyz[:1])
#         z = zip(*z)[0]
#         import string
#         print z
#         z = string.join(z[0],',')

        
        labels,kmcenter,kmfit = netky.kmeans(xy,k=6)
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

if __name__=='__main1__': 
    x = []
    personcnt = 200
    experimenTimes = 100
    workfolder_fig = gt.createFolder("N:\\HFS\\WeiboData\\CMO\\Modefigs\\")
    attfilepath = workfolder_fig+str(personcnt)+'_'+str(experimenTimes)+'.att'
    netky = NetypesfromGraph() 
    for mode in range(2,8):
        subx = []
        workfolder = "N:\\HFS\\WeiboData\\CMO\\Mode"+str(mode)+"\\graphs\\"#"G:\\HFS\\WeiboData\\CMO\\test\\"#

        sub_attfilepath = workfolder_fig+str(mode)+'_'+str(personcnt)+'_'+str(experimenTimes)+'.att'
        for modefans in [1,2,4]:#3,,5#]#]:#,2,4
            for modefr in [1,2,4]:##3,,5
                for modeal in [1,2,4]:#,2,4]:#3,,5
                    for modemen in [1,2,4]:#,2,4]:#3,,5
                        for expt in range(experimenTimes):
                            filep = str(personcnt)+'_'+str(experimenTimes)+'_'+str(expt)+'_'+str(modefans)+'__'+str(modefr)+'__'+str(modeal)+'_'+str(modemen)+'.adj'
                            filepath = workfolder+filep
                            print mode,'---------------',filepath
                            filetuple = os.path.splitext(filep)                                   
    
                        #     x = netky.define_metrics_gml(gmlfolder,pngfolder)
                            
                            ispt=True if mode>4 else False
                            resultone = netky.define_metrics_adj_one(filepath,ispanningtree=ispt,personcnt=personcnt)
                            subx.append(resultone)                         
        
        first(subx,sub_attfilepath)
        x.extend(subx)
    first(x,attfilepath) 
                            
# drawtypical = draw_typical()
# centerfilep = r'N:\HFS\WeiboData\CMO\Modefigs\7_200_100.attcenter'
# attfilep = 'N:\\HFS\\WeiboData\\CMO\\Modefigs\\7_200_100.att'
# drawtypical.start(fileinfloder="N:\\HFS\\WeiboData\\CMO\\",personcnt=200,experimentimes=100,k=7,centerfilep=centerfilep,attfilep=attfilep)  
                         
    
if __name__=='__main__':    
    # x = [1,2,3,4,4,5,5,5,6,7,8,8,9]
    # x = [[2, 92.0, 17.0], [2, 36.0, 7.0], [2, 122.5, 62.5], [2, 17.5, 4.5], [2, 36.0, 0.0], [2, 149.5, 3.5], [2, 112.5, 67.5], [2, 111.5, 10.5], [2, 38.0, 11.0], [2, 90.0, 80.0], [2, 187.0, 14.0], [2, 137.0, 46.0], [2, 33.0, 4.0], [2, 44.0, 15.0], [2, 42.5, 21.5]]
    # x = [['3344204856189380.coc.gml', 92.0, 17.0], ['3344631446304834.coc.gml', 36.0, 7.0], ['3347020320429724.coc.gml', 122.5, 62.5], ['3455798066008083.coc.gml', 17.5, 4.5], ['3456392600040737.coc.gml', 36.0, 0.0], ['3486178575789465.coc.gml', 149.5, 3.5], ['3512387527089684.coc.gml', 112.5, 67.5], ['3512638635619787.coc.gml', 111.5, 10.5], ['3512956526933221.coc.gml', 38.0, 11.0], ['3514047335033747.coc.gml', 90.0, 80.0], ['3518864421482109.coc.gml', 187.0, 14.0], ['3519104033490770.coc.gml', 137.0, 46.0], ['3521836014420909.coc.gml', 33.0, 4.0], ['3526708160065364.coc.gml', 44.0, 15.0], ['3582187498347368.coc.gml', 42.5, 21.5]]
    # x = [['3344204856189380.coc.gml', 0.20353982300884957, 0.037610619469026552], ['3344631446304834.coc.gml', 0.19148936170212766, 0.037234042553191488], ['3347020320429724.coc.gml', 0.4766536964980545, 0.24319066147859922], ['3455798066008083.coc.gml', 0.28688524590163933, 0.073770491803278687], ['3456392600040737.coc.gml', 0.35294117647058826, 0.0], ['3486178575789465.coc.gml', 0.33823529411764708, 0.0079185520361990946], ['3512387527089684.coc.gml', 0.21387832699619772, 0.12832699619771862], ['3512638635619787.coc.gml', 0.85769230769230764, 0.080769230769230774], ['3512956526933221.coc.gml', 0.40425531914893614, 0.11702127659574468], ['3514047335033747.coc.gml', 0.3930131004366812, 0.34934497816593885], ['3518864421482109.coc.gml', 0.39534883720930231, 0.029598308668076109], ['3519104033490770.coc.gml', 0.74863387978142082, 0.25136612021857924], ['3521836014420909.coc.gml', 0.13636363636363635, 0.016528925619834711], ['3526708160065364.coc.gml', 0.66666666666666663, 0.22727272727272727], ['3582187498347368.coc.gml', 0.42079207920792078, 0.21287128712871287]]
    
        
    gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\'#shape\\test\\
    pngfolder = 'G:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\'
    attfilepath = pngfolder+'real.att'
#     adjfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\test\\'N:\HFS\WeiboData\CMO\Mode7\test\200_100_0_1__1__1_1.adj
    
    netky = NetypesfromGraph() 
    x = netky.define_metrics_gml(gmlfolder,pngfolder)
#     x = netky.define_metrics_adj(adjfolder)
#     print len(x),x
    # x = [['3342670838100183.coc.gml', 0.05263157894736842, 18.0, 0.0, 0], ['3343740313561521.coc.gml', 0.07865168539325842, 1.9183673469387756, 3.5487912308747034, 0.17857142857142858], ['3343744527348953.coc.gml', 0.0036363636363636364, 267.0, 0.0, 0], ['3343901805640480.coc.gml', 0.10619469026548672, 1.0138888888888888, 1.6859417817226146, 0.13333333333333333], ['3344178035788881.coc.gml', 0.04395604395604396, 5.9375, 9.1299216179548885, 0.3125], ['3344204856189380.coc.gml', 0.10855263157894737, 0.54545454545454541, 0.67066170866769925, 0.07248484848484849], ['3344605319892924.coc.gml', 0.04878048780487805, 2.890625, 4.5789248448052735, 0.17307692307692307], ['3344617189676598.coc.gml', 0.0364963503649635, 5.6399999999999997, 9.8804048500048829, 0.24], ['3344631446304834.coc.gml', 0.17307692307692307, 0.43072702331961593, 0.32572449008846815, 0.06951566951566951], ['3345283975088597.coc.gml', 0.0673076923076923, 2.1224489795918369, 3.7427703459546207, 0.21428571428571427], ['3345341063735706.coc.gml', 0.04918032786885246, 3.5833333333333335, 6.672046440463653, 0.16666666666666666], ['3345672913760585.coc.gml', 0.025477707006369428, 9.6875, 15.48020409264684, 0.3125], ['3346041476969222.coc.gml', 0.03064066852367688, 2.9256198347107438, 8.0741137017537579, 0.09090909090909091], ['3346361808667289.coc.gml', 0.19889502762430938, 0.26774691358024694, 0.19368820951258611, 0.07207854406130268], ['3346671720159783.coc.gml', 0.031578947368421054, 10.666666666666666, 14.142135623730951, 0.3333333333333333], ['3346786119768056.coc.gml', 0.07692307692307693, 12.0, 0.0, 0], ['3347020320429724.coc.gml', 0.00823045267489712, 61.25, 31.25, 0], ['3347114865931646.coc.gml', 0.009615384615384616, 33.666666666666664, 46.197643037521111, 0.3333333333333333], ['3347122272192199.coc.gml', 0.04966887417218543, 1.5911111111111111, 5.0808495449841296, 0.07111111111111111], ['3348202183182981.coc.gml', 0.152317880794702, 0.40831758034026466, 0.43699609154782276, 0.10062111801242236], ['3356800646950624.coc.gml', 0.037037037037037035, 10.0, 12.492960981051453, 0.3333333333333333], ['3356881155164816.coc.gml', 0.015873015873015872, 12.68, 17.832375052134811, 0.2], ['3358716283896811.coc.gml', 0.1864406779661017, 1.2727272727272727, 1.0466216984729966, 0.14545454545454548], ['3363356413828548.coc.gml', 0.025210084033613446, 6.4722222222222223, 12.39863219258131, 0.16666666666666666], ['3367472590570390.coc.gml', 0.10126582278481013, 0.85546875, 1.5339133008903851, 0.11554621848739496], ['3368776344558652.coc.gml', 0.038461538461538464, 6.6875, 9.5759709037778507, 0.25], ['3369168951009868.coc.gml', 0.16304347826086957, 1.1822222222222223, 0.78333254531087237, 0.23055555555555557], ['3369278306978444.coc.gml', 0.10666666666666667, 1.296875, 2.2646549647518053, 0.125], ['3369886157847997.coc.gml', 0.04827586206896552, 3.3469387755102038, 5.0920718934696145, 0.16071428571428573], ['3370126999415642.coc.gml', 0.10232558139534884, 0.62809917355371903, 1.1553485496404718, 0.08645276292335115], ['3370187475368354.coc.gml', 0.20952380952380953, 0.3264462809917355, 0.39971730328605265, 0.06304985337243402], ['3370242220657016.coc.gml', 0.16153846153846155, 0.21712018140589567, 0.63028785398679887, 0.03552532123960696], ['3370848283881337.coc.gml', 0.23958333333333334, 0.55576559546313797, 0.3573728882194952, 0.13000852514919012], ['3371095383919407.coc.gml', 0.0995850622406639, 0.5, 1.4114999737605121, 0.06140350877192982], ['3371320634873316.coc.gml', 0.13756613756613756, 0.34319526627218938, 0.81099094495818724, 0.057692307692307696], ['3371353334212131.coc.gml', 0.03153988868274583, 1.9550173010380623, 6.8644379678601455, 0.07058823529411765], ['3371452671936848.coc.gml', 0.20833333333333334, 0.48749999999999999, 0.53639421137816168, 0.0891891891891892], ['3372030013618565.coc.gml', 0.08776595744680851, 0.39302112029384756, 1.1388429603981345, 0.0361952861952862], ['3372087507207341.coc.gml', 0.09523809523809523, 2.0277777777777777, 3.416214965624679, 0.16666666666666666], ['3372437473180477.coc.gml', 0.014925373134328358, 66.0, 0.0, 0], ['3373230205208545.coc.gml', 0.05555555555555555, 3.6399999999999997, 5.3056950534307941, 0.24], ['3373235549224874.coc.gml', 0.15841584158415842, 0.84765625, 0.80464009265692038, 0.1568627450980392], ['3374640441380743.coc.gml', 0.1619047619047619, 0.90311418685121114, 0.97389939058828379, 0.12802768166089964], ['3376476919830770.coc.gml', 0.04477611940298507, 7.666666666666667, 9.428090415820634, 0.3333333333333333], ['3379543500467226.coc.gml', 0.050359712230215826, 3.1836734693877551, 4.2691614253037979, 0.14285714285714285], ['3380021664523673.coc.gml', 0.1111111111111111, 1.6111111111111109, 2.8571979712497311, 0.19444444444444445], ['3381430950351222.coc.gml', 0.058823529411764705, 2.5306122448979593, 5.4988545246152967, 0.16326530612244897], ['3381830646827771.coc.gml', 0.16666666666666666, 1.6265432098765433, 1.1893453542136936, 0.2148846960167715], ['3384338781194861.coc.gml', 0.1864951768488746, 0.18430439952437574, 0.32827711883233557, 0.03694581280788177], ['3389552931061050.coc.gml', 0.18543046357615894, 0.84693877551020413, 0.96809796042270124, 0.11401743796109994], ['3390492127727272.coc.gml', 0.18478260869565216, 0.52249134948096887, 0.57595568874735348, 0.08144796380090498], ['3391344838326450.coc.gml', 0.12686567164179105, 1.4982698961937715, 1.7941168128716447, 0.1718266253869969], ['3393543211128758.coc.gml', 0.05185185185185185, 1.3979591836734695, 3.0183735563126923, 0.10902255639097745], ['3395625792060467.coc.gml', 0.06726457399103139, 1.0933333333333333, 2.9701527582613334, 0.10303030303030303], ['3397758248007683.coc.gml', 0.17277486910994763, 0.37373737373737376, 0.39557241991530395, 0.057323232323232325], ['3397769022933580.coc.gml', 0.07100591715976332, 1.3611111111111109, 2.6094853094522135, 0.09895833333333333], ['3398499611337996.coc.gml', 0.050314465408805034, 3.21875, 5.4632114811253647, 0.2533783783783784], ['3399163027809032.coc.gml', 0.2185430463576159, 0.38016528925619836, 0.29803886442871774, 0.09672887818583183], ['3400632889381253.coc.gml', 0.22972972972972974, 0.42906574394463665, 0.29585280960486099, 0.11250713877784124], ['3400751558944628.coc.gml', 0.1881720430107527, 0.46693877551020407, 0.41952036237110313, 0.064010989010989], ['3402776098684661.coc.gml', 0.11650485436893204, 2.8194444444444446, 3.7519799299918604, 0.19927536231884058], ['3403267493072603.coc.gml', 0.15151515151515152, 0.66749999999999998, 0.65273941967679561, 0.15459770114942528], ['3403863023733701.coc.gml', 0.07407407407407407, 3.625, 4.4034787384521339, 0.3], ['3404274883505106.coc.gml', 0.20967741935483872, 0.53698224852071008, 0.47996746983708927, 0.08361204013377926], ['3425350770267160.coc.gml', 0.10714285714285714, 0.75111111111111117, 1.6281717384466983, 0.08070175438596491], ['3428308005526158.coc.gml', 0.1728395061728395, 0.36734693877551022, 0.65596452855612941, 0.06689342403628118], ['3428528739801892.coc.gml', 0.08849557522123894, 1.6100000000000001, 2.0265487904316539, 0.1588235294117647], ['3429334025391583.coc.gml', 0.09090909090909091, 1.7959183673469388, 3.0609523688575799, 0.21428571428571427], ['3429588258743940.coc.gml', 0.08379888268156424, 1.1422222222222222, 1.8763031685329836, 0.11891891891891891], ['3430283762648349.coc.gml', 0.0830860534124629, 0.56122448979591832, 1.7855575732984685, 0.06373626373626375], ['3430335716843081.coc.gml', 0.09482758620689655, 1.2892561983471074, 2.2445587204312196, 0.21212121212121213], ['3430668065297419.coc.gml', 0.1276595744680851, 0.83024691358024694, 1.1116468187316051, 0.1111111111111111], ['3431023705030524.coc.gml', 0.18686868686868688, 0.35938641344046746, 0.32131946600325445, 0.07380457380457381], ['3431094353698885.coc.gml', 0.15018315018315018, 0.23795359904818561, 0.73976880788105981, 0.03864428254672157], ['3431737105763161.coc.gml', 0.1597222222222222, 0.21455576559546313, 0.48105833965723965, 0.03595317725752508], ['3432170243211946.coc.gml', 0.10152284263959391, 0.59250000000000003, 1.5129668700933276, 0.07037037037037037], ['3432525189989140.coc.gml', 0.040268456375838924, 4.416666666666667, 5.4863044972947899, 0.22916666666666666], ['3433746454019952.coc.gml', 0.24603174603174602, 0.56919875130072839, 0.47533563803802376, 0.1032258064516129], ['3434292485521437.coc.gml', 0.21, 0.47619047619047616, 0.35966606787419986, 0.1021505376344086], ['3435948283388972.coc.gml', 0.042735042735042736, 5.4399999999999995, 6.7709969723815409, 0.2857142857142857], ['3439812252362083.coc.gml', 0.2222222222222222, 0.27040816326530609, 0.47220759029251375, 0.06578947368421052], ['3440602006168810.coc.gml', 0.05113636363636364, 2.4197530864197532, 4.0095146707722549, 0.1515151515151515], ['3441634194910767.coc.gml', 0.17647058823529413, 0.51700680272108845, 0.51103484895039009, 0.09523809523809523], ['3442111980741300.coc.gml', 0.26229508196721313, 0.2197265625, 0.30623449169518541, 0.052403846153846155], ['3443073797743047.coc.gml', 0.12912087912087913, 0.26708918062471704, 0.74464603825551456, 0.040665751544269046], ['3443355667223867.coc.gml', 0.13584905660377358, 0.27314814814814814, 0.787050653476983, 0.04656862745098039], ['3443416153317033.coc.gml', 0.21296296296296297, 0.35916824196597352, 0.36268303112255912, 0.07275953859804792], ['3443492326385825.coc.gml', 0.19834710743801653, 0.4045138888888889, 0.63184666139088097, 0.08333333333333333], ['3443673498955248.coc.gml', 0.1125, 1.2345679012345678, 2.1586235290837483, 0.1388888888888889], ['3443744265281796.coc.gml', 0.1, 1.0, 2.3610920826366995, 0.09848484848484848], ['3443805103639827.coc.gml', 0.06896551724137931, 2.625, 3.3042349719715758, 0.15], ['3443829309398044.coc.gml', 0.09278350515463918, 1.0462962962962963, 1.3282116136124467, 0.10897435897435898], ['3444219576564892.coc.gml', 0.10077519379844961, 0.80473372781065089, 1.2884471021412873, 0.10526315789473685], ['3445044281384443.coc.gml', 0.1956521739130435, 0.49108367626886146, 0.37805371893086698, 0.07407407407407407], ['3445163550624397.coc.gml', 0.10638297872340426, 0.72444444444444445, 1.6333318216168362, 0.09696969696969697], ['3445566971379762.coc.gml', 0.09803921568627451, 3.5100000000000002, 3.4607658112042197, 0.2103448275862069], ['3447345331395476.coc.gml', 0.0379746835443038, 2.9753086419753085, 6.0776362133462021, 0.13333333333333333], ['3447401585338269.coc.gml', 0.16498316498316498, 0.2099125364431487, 0.38021785158953014, 0.05604850635906537], ['3447504811582619.coc.gml', 0.18023255813953487, 0.38709677419354838, 0.3608887317642987, 0.0565684899485741], ['3447758196911455.coc.gml', 0.13220338983050847, 0.34319526627218938, 0.64318152615344137, 0.042429792429792425], ['3448178085407204.coc.gml', 0.1409090909090909, 0.64412070759625395, 0.91663466982181852, 0.07949308755760369], ['3448186452366671.coc.gml', 0.1165644171779141, 0.32132963988919672, 0.92064015090556806, 0.049877600979192166], ['3448222901573576.coc.gml', 0.17073170731707318, 0.18845663265306123, 0.50335879460640343, 0.032542293233082706], ['3448524715110674.coc.gml', 0.05993690851735016, 0.94182825484764543, 1.5443952381603805, 0.08204334365325078], ['3448580884997872.coc.gml', 0.04639175257731959, 3.1481481481481479, 4.0071951747730337, 0.15873015873015872], ['3449158323300895.coc.gml', 0.22900763358778625, 0.3611111111111111, 0.34438171471798379, 0.06467661691542288], ['3451171018291015.coc.gml', 0.08, 1.3500000000000001, 1.9085334683992314, 0.12727272727272726], ['3451840329079188.coc.gml', 0.05, 3.4166666666666665, 6.2255075652065868, 0.19444444444444445], ['3452192143211197.coc.gml', 0.07975460122699386, 1.0828402366863905, 2.5330696795253158, 0.09615384615384616], ['3452512994803254.coc.gml', 0.1282051282051282, 0.83111111111111113, 1.2328507564398543, 0.09393939393939395], ['3452821297309374.coc.gml', 0.03773584905660377, 9.0625, 8.9989148651379072, 0.35], ['3452834043619851.coc.gml', 0.08737864077669903, 0.72222222222222221, 1.6450240886991743, 0.07333333333333333], ['3455798066008083.coc.gml', 0.2972972972972973, 0.60330578512396693, 0.14644665410470536, 0.2830578512396694], ['3455801434063025.coc.gml', 0.10734463276836158, 0.57340720221606656, 0.97091313951921798, 0.0953058321479374], ['3456392600040737.coc.gml', 0.22, 2.4049586776859502, 0.74801378505684368, 0.21212121212121213], ['3457216382606769.coc.gml', 0.06756756756756757, 1.8199999999999998, 2.8180844557961704, 0.13076923076923078], ['3457458590948980.coc.gml', 0.1532258064516129, 0.51800554016620504, 0.65453728507845621, 0.10141206675224647], ['3458260155974734.coc.gml', 0.10810810810810811, 0.37244897959183676, 1.0356011801114511, 0.047619047619047616], ['3458382231009304.coc.gml', 0.18125, 0.36860879904875149, 0.34638216144788825, 0.05757389162561576], ['3459709472618541.coc.gml', 0.1282051282051282, 1.1699999999999999, 1.5943964375273798, 0.14666666666666667], ['3461761758135160.coc.gml', 0.16891891891891891, 0.61280000000000001, 0.76220480187414197, 0.09055555555555556], ['3462527801769811.coc.gml', 0.08737864077669903, 1.4567901234567902, 3.0222968602096394, 0.1234567901234568], ['3464345767606270.coc.gml', 0.08284023668639054, 1.1836734693877553, 3.1049093879486933, 0.07619047619047618], ['3464705244725034.coc.gml', 0.016286644951140065, 12.640000000000001, 20.083585337284774, 0.24], ['3464854910351381.coc.gml', 0.09900990099009901, 1.29, 2.1252999788265186, 0.17142857142857143], ['3466348208670198.coc.gml', 0.07971014492753623, 2.4628099173553717, 1.9356494028492433, 0.21867321867321865], ['3466382715287291.coc.gml', 0.07534246575342465, 1.3719008264462811, 2.5167969296420534, 0.14285714285714285], ['3466382765276535.coc.gml', 0.08928571428571429, 0.8355555555555555, 1.5370711285822096, 0.07916666666666666], ['3466578031231644.coc.gml', 0.1987179487179487, 0.2351716961498439, 0.51399098970413459, 0.04996837444655282], ['3466699703809713.coc.gml', 0.1258741258741259, 0.71604938271604945, 0.90644107443874156, 0.10069444444444445], ['3466705462858968.coc.gml', 0.17123287671232876, 0.33119999999999999, 0.48325827463169219, 0.058536585365853655], ['3466749821485858.coc.gml', 0.20833333333333334, 0.19506172839506175, 0.33066882014727184, 0.03950617283950617], ['3469573347006484.coc.gml', 0.041666666666666664, 6.125, 9.1660037639093304, 0.25], ['3472429655943907.coc.gml', 0.12598425196850394, 1.796875, 1.9160225933232102, 0.14760638297872342], ['3475083715093089.coc.gml', 0.07407407407407407, 1.074074074074074, 1.819155805961898, 0.08444444444444445], ['3476098195491864.coc.gml', 0.09259259259259259, 2.7600000000000002, 2.7521627858831317, 0.24], ['3476515322550681.coc.gml', 0.14285714285714285, 0.89619377162629754, 1.0113400194564059, 0.10561497326203208], ['3476909163641066.coc.gml', 0.19246861924686193, 0.33790170132325142, 0.34323848806463236, 0.04762945160585537], ['3477038419958946.coc.gml', 0.09734513274336283, 1.0909090909090908, 1.7335683440828951, 0.12987012987012989], ['3478679923748497.coc.gml', 0.14893617021276595, 0.72576530612244905, 1.338215680415326, 0.08048289738430583], ['3479445619333773.coc.gml', 0.04504504504504504, 4.6399999999999997, 6.0191693779125366, 0.2], ['3479624388879132.coc.gml', 0.0755813953488372, 1.2011834319526626, 3.1188764800055164, 0.09615384615384616], ['3481902797536816.coc.gml', 0.11538461538461539, 1.1405895691609977, 1.2421420163518202, 0.14107142857142857], ['3481923777474913.coc.gml', 0.10714285714285714, 0.76000000000000001, 1.4625801655775112, 0.09206349206349206], ['3481928609200009.coc.gml', 0.1792452830188679, 0.445983379501385, 0.62048624839279376, 0.0756578947368421], ['3482159342150289.coc.gml', 0.1834862385321101, 0.67500000000000004, 0.50830601019464638, 0.10500000000000001], ['3483267271112598.coc.gml', 0.0958904109589041, 1.1428571428571428, 1.7240368925858529, 0.10119047619047619], ['3484321828620387.coc.gml', 0.09950248756218906, 0.78000000000000003, 1.0575206853768866, 0.07857142857142857], ['3486178575789465.coc.gml', 0.0880952380952381, 0.49817384952520083, 1.04649837414053, 0.05192833282720923], ['3487349066960837.coc.gml', 0.09615384615384616, 2.6399999999999997, 2.940476151918257, 0.3142857142857143], ['3488143468769060.coc.gml', 0.16721311475409836, 0.23414071510957324, 0.46849160913007198, 0.033655253146034535], ['3489084565802342.coc.gml', 0.16621253405994552, 0.19940876108572964, 0.34078535375129682, 0.02275211127670144], ['3489089556938378.coc.gml', 0.15056818181818182, 0.24136703453186187, 0.49170607858474752, 0.03787158952083616], ['3489092799395971.coc.gml', 0.05660377358490566, 3.9722222222222219, 3.5097966948964685, 0.2142857142857143], ['3489137279411789.coc.gml', 0.18357487922705315, 0.18698060941828257, 0.43951015298468638, 0.03475670307845084], ['3489148290157520.coc.gml', 0.21052631578947367, 0.24691358024691359, 0.49667868844449725, 0.0595679012345679], ['3489586389438248.coc.gml', 0.2032967032967033, 0.30241051862673485, 0.19878987916836433, 0.06187766714082503], ['3489804664943100.coc.gml', 0.12844036697247707, 0.6428571428571429, 1.3681082254770913, 0.08843537414965986], ['3491356209051281.coc.gml', 0.1322314049586777, 1.5859375, 1.8462476461985673, 0.1340909090909091], ['3491608849042650.coc.gml', 0.14457831325301204, 0.71354166666666663, 0.69527511083410021, 0.10101010101010101], ['3492678328811636.coc.gml', 0.16455696202531644, 0.27071005917159763, 0.37222738773266684, 0.05257242757242757], ['3492682624290826.coc.gml', 0.18421052631578946, 0.91581632653061218, 0.95646937382075115, 0.0967741935483871], ['3492684352482154.coc.gml', 0.1388888888888889, 0.36333333333333334, 0.79704779353887245, 0.06519607843137255], ['3492805542177005.coc.gml', 0.23741007194244604, 0.35537190082644626, 0.24142041102156081, 0.09987770077456176], ['3493173752439900.coc.gml', 0.08928571428571429, 1.26, 1.7356266879718114, 0.14375], ['3494236496496836.coc.gml', 0.23214285714285715, 1.4378698224852071, 1.3665436751856521, 0.14219114219114218], ['3495157817228723.coc.gml', 0.16042780748663102, 0.79555555555555557, 0.69605945538588776, 0.06930693069306931], ['3495425338709660.coc.gml', 0.18452380952380953, 0.29448491155046824, 0.28557124132440809, 0.04985337243401759], ['3495908950319195.coc.gml', 0.6666666666666666, 1.25, 0.25, 0.5], ['3497476986739290.coc.gml', 0.14545454545454545, 0.5849609375, 0.73267899132805503, 0.0756340579710145], ['3498227716765728.coc.gml', 0.25225225225225223, 0.5982142857142857, 0.50765145882220697, 0.0888704318936877], ['3501198583561829.coc.gml', 0.3, 8.6666666666666661, 3.6616126785075735, 0.4166666666666667], ['3504355070314763.coc.gml', 0.058823529411764705, 2.7551020408163263, 5.00070798819264, 0.21428571428571427], ['3505165031934094.coc.gml', 0.22580645161290322, 0.17233560090702948, 0.32727774187597236, 0.039047619047619046], ['3506858382217257.coc.gml', 0.18518518518518517, 0.36320000000000002, 0.411501834746821, 0.10400000000000001], ['3507543877973721.coc.gml', 0.125, 0.5855555555555555, 1.0501034634504436, 0.06956521739130435], ['3507607539020444.coc.gml', 0.17054263565891473, 0.60330578512396693, 0.60553759050346179, 0.10265924551638837], ['3507662178094930.coc.gml', 0.14583333333333334, 1.653061224489796, 1.1826174036286059, 0.2380952380952381], ['3507671015760502.coc.gml', 0.09473684210526316, 1.5555555555555556, 2.9574021892855007, 0.1234567901234568], ['3508035156247306.coc.gml', 0.14919354838709678, 0.56756756756756754, 0.71246796804647361, 0.06704494549798418], ['3508278808120380.coc.gml', 0.12962962962962962, 1.1428571428571428, 1.2936264483053452, 0.17857142857142858], ['3509438231211989.coc.gml', 0.11965811965811966, 0.72959183673469385, 1.2712275040017211, 0.09999999999999999], ['3509885691781722.coc.gml', 0.15267175572519084, 0.54749999999999999, 0.72482325431790606, 0.08372093023255814], ['3510108007340190.coc.gml', 0.2421875, 0.71488033298647236, 0.62158249090862083, 0.08770161290322581], ['3510150776647546.coc.gml', 0.10638297872340426, 2.9699999999999998, 2.9254230463302227, 0.2], ['3510947052234805.coc.gml', 0.07462686567164178, 4.6399999999999997, 3.522271994040211, 0.24], ['3511918478199744.coc.gml', 0.11949685534591195, 0.80332409972299168, 1.1417716690211632, 0.09424724602203183], ['3511950958712857.coc.gml', 0.15463917525773196, 0.16194444444444445, 0.28251666071268594, 0.027864583333333335], ['3511983850692431.coc.gml', 0.22813688212927757, 0.2175, 0.33809686591380989, 0.0526071842410197], ['3512192651209611.coc.gml', 0.11666666666666667, 1.6122448979591837, 1.7772686577205914, 0.2619047619047619], ['3512225370844164.coc.gml', 0.09826589595375723, 0.74740484429065734, 1.5672410590914201, 0.08088235294117647], ['3512260125547589.coc.gml', 0.13872832369942195, 0.43055555555555558, 0.96919170535472243, 0.06912878787878789], ['3512261920248384.coc.gml', 0.17142857142857143, 0.63580246913580252, 0.39249746498714894, 0.12014453477868112], ['3512288331952105.coc.gml', 0.08, 1.2013888888888888, 2.9310080614308855, 0.10119047619047618], ['3512343789230980.coc.gml', 0.06060606060606061, 1.2040816326530612, 3.4310001116191851, 0.07653061224489796], ['3512367365458914.coc.gml', 0.09691629955947137, 1.0289256198347108, 1.695031047109739, 0.08070500927643785], ['3512387527089684.coc.gml', 0.1023391812865497, 0.36979591836734693, 0.86537389029973322, 0.04074074074074074], ['3512557425820703.coc.gml', 0.16517857142857142, 0.254200146092038, 0.58138115341278407, 0.03620601733809281], ['3512564992138128.coc.gml', 0.1891891891891892, 1.6377551020408163, 1.6402962039881677, 0.1533613445378151], ['3512586882317341.coc.gml', 0.14012738853503184, 0.47727272727272729, 0.93473641005016939, 0.06598240469208211], ['3512598488183675.coc.gml', 0.11650485436893204, 0.78472222222222221, 1.8510689896168071, 0.09027777777777778], ['3512631170591260.coc.gml', 0.08123791102514506, 0.36054421768707484, 1.469864025862113, 0.029047619047619048], ['3512638635619787.coc.gml', 0.16666666666666666, 2.46875, 2.2841377270755805, 0.1865530303030303], ['3512649558390590.coc.gml', 0.13043478260869565, 0.88888888888888884, 1.4028465329684954, 0.1111111111111111], ['3512661431797880.coc.gml', 0.13157894736842105, 0.84000000000000008, 1.7884071124886529, 0.11000000000000001], ['3512665513423714.coc.gml', 0.09615384615384616, 1.3699999999999999, 2.7155294143131647, 0.13076923076923078], ['3512673221800564.coc.gml', 0.14563106796116504, 0.31444444444444447, 0.67919796565008184, 0.04390243902439025], ['3512681942436390.coc.gml', 0.16393442622950818, 1.1499999999999999, 0.94472218138455921, 0.16842105263157894], ['3512703459106709.coc.gml', 0.1087866108786611, 0.45266272189349116, 1.1573569710477523, 0.05673076923076924], ['3512704620407286.coc.gml', 0.2062780269058296, 0.27173913043478259, 0.21182621655007042, 0.049217391304347824], ['3512722819965166.coc.gml', 0.09803921568627451, 3.7999999999999998, 4.3432706570049264, 0.275], ['3512723826282658.coc.gml', 0.2235294117647059, 0.43490304709141275, 0.40367249267472288, 0.08819345661450924], ['3512731699677616.coc.gml', 0.07718120805369127, 0.63516068052930053, 1.9814517206876603, 0.06649616368286444], ['3512751521888667.coc.gml', 0.08522727272727272, 0.43666666666666665, 1.5662966466111026, 0.045000000000000005], ['3512753392115009.coc.gml', 0.1278772378516624, 0.26800000000000002, 0.74071316985726665, 0.0305], ['3512755191392034.coc.gml', 0.078125, 0.9555555555555556, 2.4897617517037474, 0.09206349206349206], ['3512764419413627.coc.gml', 0.23076923076923078, 0.46296296296296302, 0.37679611017362602, 0.08602150537634408], ['3512767539668338.coc.gml', 0.09090909090909091, 0.80864197530864201, 2.0486388654510161, 0.07555555555555556], ['3512956526933221.coc.gml', 0.17857142857142858, 0.8355555555555555, 0.83616998685290078, 0.184], ['3512965133226559.coc.gml', 0.175, 0.36607142857142855, 0.61856195790930657, 0.08421052631578947], ['3513008209201946.coc.gml', 0.11805555555555555, 0.60553633217993086, 1.4384395600633919, 0.07754010695187165], ['3513054618763335.coc.gml', 0.17894736842105263, 0.72664359861591699, 0.96043047384111757, 0.09803921568627451], ['3513299721572710.coc.gml', 0.1724137931034483, 0.64222222222222214, 0.80764556460410997, 0.10112994350282485], ['3513353957580369.coc.gml', 0.0915032679738562, 0.49107142857142855, 1.3366001986786742, 0.052884615384615384], ['3513457897170153.coc.gml', 0.148, 0.32651570489408327, 0.92374828543733423, 0.05297297297297297], ['3513472585606907.coc.gml', 0.14084507042253522, 2.27, 3.6452846253756368, 0.18641975308641975], ['3513477123020136.coc.gml', 0.1794871794871795, 0.59183673469387765, 0.7425178637643427, 0.12442396313364056], ['3513651849587189.coc.gml', 0.13138686131386862, 0.97530864197530875, 1.1319378697411457, 0.0925925925925926], ['3513665519425522.coc.gml', 0.1, 1.1074380165289257, 1.7817921156038159, 0.12337662337662338], ['3513732346670068.coc.gml', 0.09059233449477352, 0.51331360946745563, 1.2600954326090696, 0.0554561717352415], ['3513732681977292.coc.gml', 0.20833333333333334, 0.47499999999999998, 0.58105507484230789, 0.09268292682926829], ['3513733382475662.coc.gml', 0.056179775280898875, 1.8899999999999999, 4.5704376158087978, 0.10909090909090909], ['3513738009766461.coc.gml', 0.205, 0.34681737061273049, 0.32963767128527172, 0.0917313215311415], ['3513747752676493.coc.gml', 0.12345679012345678, 0.47249999999999998, 1.0090187064668326, 0.06086956521739131], ['3513797614859184.coc.gml', 0.15151515151515152, 1.1040000000000001, 1.5092673719391141, 0.1075438596491228], ['3513821030864621.coc.gml', 0.09443099273607748, 0.31821170282708744, 1.1852860047347402, 0.03780964797913951], ['3514047335033747.coc.gml', 0.07906976744186046, 0.79238754325259519, 2.3024971308731628, 0.07219251336898395], ['3514054737834781.coc.gml', 0.10526315789473684, 1.1599999999999999, 1.8789358690492872, 0.1588235294117647], ['3514061079292261.coc.gml', 0.07647058823529412, 0.6375739644970414, 1.5793808627202552, 0.054945054945054944], ['3514083762166772.coc.gml', 0.06937799043062201, 0.53983353151010705, 2.1079224916700401, 0.041379310344827586], ['3514112790871598.coc.gml', 0.1267605633802817, 1.691358024691358, 3.0696209503031242, 0.06666666666666667], ['3514202721379789.coc.gml', 0.16296296296296298, 0.57644628099173556, 0.54655301987822713, 0.07792207792207792], ['3514207033581502.coc.gml', 0.10655737704918032, 0.50591715976331364, 1.2797080695421461, 0.05098389982110912], ['3514216143529145.coc.gml', 0.22466960352422907, 0.25297962322183776, 0.28092291608425252, 0.05202312138728324], ['3514229367981237.coc.gml', 0.2427536231884058, 0.12296725328580976, 0.2169199292164827, 0.029850746268656716], ['3514287944897392.coc.gml', 0.16071428571428573, 0.49108367626886146, 0.4301062339859768, 0.09144947416552354], ['3514415834044554.coc.gml', 0.17829457364341086, 0.46502835538752363, 0.44808291543593087, 0.08779264214046822], ['3514416295764354.coc.gml', 0.15730337078651685, 0.97448979591836726, 0.89239054863624234, 0.1406926406926407], ['3514448201592653.coc.gml', 0.12015503875968993, 0.18418314255983351, 0.78146810143963086, 0.025466893039049237], ['3514574454119360.coc.gml', 0.06321839080459771, 1.4628099173553719, 3.6773423555717462, 0.09917355371900825], ['3514712136139677.coc.gml', 0.14942528735632185, 0.61538461538461542, 0.50374112460571052, 0.09419152276295134], ['3514721241880789.coc.gml', 0.06153846153846154, 1.5416666666666667, 3.8106970539570626, 0.09027777777777778], ['3516669001647040.coc.gml', 0.205761316872428, 0.33799999999999997, 0.64420183172667256, 0.028415841584158413], ['3517122988859030.coc.gml', 0.17647058823529413, 1.1111111111111112, 2.0563061692579723, 0.13383838383838384], ['3517807143042530.coc.gml', 0.15577889447236182, 0.67741935483870963, 1.2851091347565395, 0.053763440860215055], ['3518018271478014.coc.gml', 0.17261904761904762, 0.30915576694411412, 0.46822726029357581, 0.04490777866880514], ['3518037380208073.coc.gml', 0.20306513409961685, 0.17337130651477395, 0.24065398412732075, 0.06363011006289308], ['3518192234385654.coc.gml', 0.14383561643835616, 0.2284580498866213, 0.68079179884738772, 0.03755868544600939], ['3518864421482109.coc.gml', 0.11185682326621924, 0.27839999999999998, 0.73075128463794026, 0.0316504854368932], ['3518889973774430.coc.gml', 0.21333333333333335, 0.263671875, 0.38913289241464077, 0.05296610169491525], ['3518897993492899.coc.gml', 0.24120603015075376, 0.15755208333333334, 0.27328968392212011, 0.037747524752475246], ['3518902011319515.coc.gml', 0.15725806451612903, 0.24786324786324784, 0.58485909465493868, 0.053946053946053944], ['3519083113115752.coc.gml', 0.12442396313364056, 0.41426611796982166, 0.90092655185386394, 0.053402239448751075], ['3519104033490770.coc.gml', 0.13253012048192772, 1.2954545454545454, 1.8209044697358046, 0.1242807825086306], ['3519173332815242.coc.gml', 0.1956521739130435, 0.61179698216735257, 0.47594904966646356, 0.09579504316346421], ['3519233508278922.coc.gml', 0.16666666666666666, 0.45706371191135736, 0.75057496191878204, 0.10941828254847645], ['3521836014420909.coc.gml', 0.2028985507246377, 0.23696145124716553, 0.16827474796963177, 0.0412008281573499], ['3522707796773995.coc.gml', 0.18045112781954886, 0.40104166666666669, 0.80971857429614724, 0.06060606060606061], ['3523242033543909.coc.gml', 0.2186046511627907, 0.34721593481213214, 0.31579973087493873, 0.0661694816114645], ['3526708160065364.coc.gml', 0.23333333333333334, 1.1173469387755102, 1.011453051868916, 0.1134453781512605], ['3527143084709163.coc.gml', 0.16736401673640167, 0.33875, 0.37951243392015505, 0.048], ['3527449691832220.coc.gml', 0.20238095238095238, 0.27422145328719727, 0.30135731589905534, 0.04555236728837877], ['3558226899367894.coc.gml', 0.19689119170984457, 0.20567867036011081, 0.50966941476530092, 0.04032809295967191], ['3558246365665871.coc.gml', 0.08994708994708994, 1.4221453287197232, 1.5708046834663099, 0.09313725490196079], ['3560576217397500.coc.gml', 0.14, 0.18909438775510204, 0.36129248696130328, 0.027649769585253454], ['3581830525479047.coc.gml', 0.1282051282051282, 0.21454545454545457, 0.77492948226231451, 0.03353808353808354], ['3581866814344587.coc.gml', 0.13657407407407407, 0.22120080436656134, 0.51081929686148952, 0.038528743665909485], ['3582187498347368.coc.gml', 0.21052631578947367, 0.42750000000000005, 0.68565206190895389, 0.0696969696969697]] 
 
     
    xyz = zip(*x)
    xyzz = gt.normlistlist(listlista=xyz[1:],metacolcount=0,sumormax='max')#xyz[1:]#
    xy = zip(*xyzz)#[1:]
    z = zip(*xyz[:1])
    
#     print xy 
    labels,kmcenter,kmfit = netky.kmeans(xy,k=6)
#     netky.classfig(labels,x,z)
    np.savetxt(fname=attfilepath, X=xy,  delimiter=' ', newline='\n', header='', footer='', comments='# ')#
    np.savetxt(fname=attfilepath+'label', X=labels, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#
    np.savetxt(fname=attfilepath+'center', X=np.asarray(kmcenter), fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#



if __name__=='__main4__': 
    x = []
    personcnt = 200
    experimenTimes = 100
    workfolder = gt.createFolder("N:\\HFS\\WeiboData\\CMO\\Modefigs\\")
    workfolder_fig = gt.createFolder("N:\\HFS\\WeiboData\\CMO\\Modefigs\\test\\")
    netky = NetypesfromGraph() 
    from draw_typical_cmo import draw_typical
    for mod in range(2,3):
        cols = [0,1,2,3,4,5,17,18,19,20,21]#]#6,7,8,9,10,11,
 
        attfilepath = workfolder_fig+str(mod)+'_'+str(personcnt)+'_'+str(experimenTimes)+'.att'
        attfilepath2 = workfolder+str(mod)+'_'+str(personcnt)+'_'+str(experimenTimes)+'.att'
        xy = np.genfromtxt(attfilepath2,usecols=cols)
         
        labels,kmcenter,kmfit = netky.kmeans(xy,k=16)
     
        np.savetxt(fname=attfilepath, X=xy,  delimiter=' ', newline='\n', header='', footer='', comments='# ')# 
        np.savetxt(fname=attfilepath+'label', X=labels, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#
        np.savetxt(fname=attfilepath+'center', X=np.asarray(kmcenter), fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')#
                            
        drawtypical = draw_typical()
        centerfilep = attfilepath+'center'#r'N:\HFS\WeiboData\CMO\Modefigs\7_200_100.attcenter'
        attfilep = attfilepath#'N:\\HFS\\WeiboData\\CMO\\Modefigs\\7_200_100.att'
        drawtypical.start(fileinfloder="N:\\HFS\\WeiboData\\CMO\\",personcnt=200,experimentimes=100,k=mod,centerfilep=centerfilep,attfilep=attfilep)  
 

def real_kmdis(real_folder):
    x = []
    workfolder_fig = gt.createFolder("N:\\HFS\\WeiboData\\CMO\\Modefigs\\")
    attfilepath = workfolder_fig+str(personcnt)+'_'+str(experimenTimes)+'.att'
    netky = NetypesfromGraph() 
    for filep in os.listdir(real_folder):
        gfile = real_folder+filep
        g = ig.Graph.Read_Adjacency(gfile)
        x = netky.define_metrics_gml(gmlfolder,pngfolder)
        
        ispt=True if mode>4 else False
        resultone = netky.define_metrics_adj_one(filepath,ispanningtree=ispt,personcnt=personcnt)
        subx.append(resultone)                         
        
        first(subx,sub_attfilepath)
        x.extend(subx)
    first(x,attfilepath) 
    
    
# if __name__=='__main__': 
#     real_kmdis()
    
    