#encoding=utf8
import re
import os
import time
import sys
sys.path.append('..\..')
from tools import commontools as gtf
import igraph as ig
from igraph import clustering as clus
import numpy as np

gt = gtf()

class draw_typical:
    """drawtypical = draw_typical()
    drawtypical.start()"""
    def drawg(self,tag,gfilep):
        import igraph as ig
        graphfilepath = gfilep#r'N:\HFS\WeiboData\CMO\graphs\300_500_40_1_2.adj'
        g = ig.Graph.Read_Adjacency(graphfilepath)
        g.write_gml(graphfilepath+'.gml')
        gt.drawgraph(g,giantornot=True,figpath=graphfilepath+str(tag)+'.png',show=False)#sw
                    
    
    # typical_list = zip(*(gt.csv2list_new(r'G:\HFS\WeiboData\CMO\figs\typical_index')))
    # for tag,filep in zip(*(typical_list[0],typical_list[-1])):
    #     drawg(tag,filep)
    
    def drawall(self,):
        workfolder_root = "N:\\HFS\\WeiboData\\CMO\\"
        for mode in range(2,8):
            workfolder = workfolder_root+"Mode"+str(mode)+"\\graphs\\"
            workfolder_fig = workfolder_root+"Mode"+str(mode)+"\\figs\\"
        #     for filep in os.listdir(workfolder):
        #         print os.path.splitext(filep)[-1],workfolder+filep
        #         filetuple = os.path.splitext(filep)
        #         if filetuple[-1]=='.adj':    
            for modefans in [1,2,4]:#3,,5
                for modefr in [1,2,4]:#3,,5
                    for modeal in [1,2,4]:#3,,5
                        for modemen in [1,2,4]:#3,,5
                            for i in range(10):
                                filep = '200_100_'+str(i)+'_'+str(modefans)+'__'+str(modefr)+'__'+str(modeal)+'_'+str(modemen)+'.adj'
                                filetuple = os.path.splitext(filep)
                                if os.path.isfile(workfolder+filep):
                                    print filep
                                    g = ig.Graph.Read_Adjacency(workfolder+filep)
                                    g.write_gml(workfolder_fig+filetuple[0]+'.gml')
                                    gt.drawgraph(g,giantornot=True,figpath=workfolder_fig+filetuple[0]+'.png',show=False)#sw
            
    
    def dis(self,A,B):
        from scipy.spatial import distance
        
    #     A = np.array([[1,23,2,5,6,2,2,6,2],[12,4,5,5],[1,2,4],[1],[2],[2]], dtype=object )
    #     B = np.array([[1,23,2,5,6,2,2,6,2],[12,4,5,5],[1,2,4],[1],[2],[2]], dtype=object )
        
        Aflat = np.hstack( A )
        Bflat = np.hstack( B )
        
        dist = distance.cosine( Aflat, Bflat )        
        return dist
    
    def distance2center(self,centerfilep,attfilep):        
        center = np.genfromtxt(centerfilep)
        att = np.genfromtxt(attfilep)
        i,j = 0,0
        res,dis_index = [],[]
        for cen in center:
            subres = []
            i+=1
            for at in att:
                j+=1
                subres.append(self.dis(cen,at))
            res.append(subres)
            resindex = [i,np.min(subres),np.argmin(subres)]
            dis_index.append(resindex)
            print resindex
        np.savetxt(centerfilep+'_dis',res)
        np.savetxt(attfilep+'_center_dis_index',dis_index)
        return dis_index
    
    def getfilenames_near_centers(self,fileinfloder,indexlist):
        result = []
        a,b,c,d = 0,0,0,0
        for mode in [0,1,2,3,4,5]:
            m = mode*3**4*100
            for modefans in [0,1,2]:#3,,5
                a=modefans*3**3*100
                for modefr in [0,1,2]:#3,,5
                    b=modefr*3**2*100
                    for modeal in [0,1,2]:#3,,5
                        c=modeal*3*100
                        for modemen in [0,1,2]:#3,,5
                            d=modemen*100
    
                            for i in range(100):
                                index = m+a+b+c+d+i+1
            
                                if index in indexlist:
                                    indexofile = '_'+str(modefans+1)+'__'+str(modefr+1)+'__'+str(modeal+1)+'_'+str(modemen+1)
                                    indexofile = indexofile.replace('_3','_4')
                                    filep = fileinfloder+'Mode'+str(mode+2)+'\\graphs\\200_100_'+str(i)+indexofile+'.adj'
            
                                    result.append([index,filep])
        return result
    
    def start(self,fileinfloder="N:\\HFS\\WeiboData\\CMO\\",personcnt=200,experimentimes=100,k=6,centerfilep=None,attfilep=None):    
#        # fileinfloder = "N:\\HFS\\WeiboData\\CMO\\"
#         centerfilep = fileinfloder+'Modefigs\\'+str(personcnt)+'_'+str(experimentimes)+'.attcenter'
        gt.createFolder(fileinfloder+'Modefigs\\nearCenter\\')
        if not centerfilep:
            centerfilep = fileinfloder+'Modefigs\\fig\\'+str(personcnt)+'_'+str(experimentimes)+'\\kmeans-'+str(k)+'\\shapedis_kmeans_center.center'#+str(personcnt)+'_'+str(experimentimes)+'.attcenter'
        if not attfilep:
            attfilep = fileinfloder+'Modefigs\\'+str(personcnt)+'_'+str(experimentimes)+'.att'
        print centerfilep,'\n',attfilep
        indexlist = self.distance2center(centerfilep,attfilep)
#         indexlist = [[1, 0.00077244313218682858, 34704], [2, 0.00041000346637209972, 39140], [3, 0.00039026611694925606, 15777], [4, 0.00024877962744651594, 17886], [5, 0.0003153829024196142, 41481], [6, 0.0002142365422109771, 6207]]
        indexlist = zip(*indexlist)[2]
        print 'distancce index have done===' ,indexlist   
#         indexlist = [14290, 29587, 20108, 29063, 18682, 13355]
        filist = self.getfilenames_near_centers(fileinfloder,indexlist)                            
        print 'index of min have done===',filist    
        for filen in filist:
            print filen
            tag = str(k)+'_'+str(personcnt)+'_'+str(experimentimes)+'_'+str(indexlist.index(filen[0]))
            g = ig.Graph.Read_Adjacency(filen[1])
            ispt = False
            ispt=True if int(filen[0])>8100*3 else False
            if ispt:
                g = ig.Graph.spanning_tree(g)          
                g = clus.VertexClustering.giant(g.clusters(mode='weak'))
            g.write_gml(fileinfloder+'Modefigs\\nearCenter\\'+str(tag)+'.gml')
            try:
                gt.drawgraph(g,giantornot=True,figpath=fileinfloder+'Modefigs\\nearCenter\\'+str(tag)+'.png',show=False)#s
            except:
                print 'graph has not draw'
            print 'Have draw graph as :',fileinfloder+'Modefigs\\nearCenter\\'+str(tag)+'.png'

if __name__=='__main__': 
    drawtypical = draw_typical()
    centerfilep = r'N:\HFS\WeiboData\CMO\Modefigs\7_200_100.attcenter'
    attfilep = 'N:\\HFS\\WeiboData\\CMO\\Modefigs\\7_200_100.att'
    drawtypical.start(fileinfloder="N:\\HFS\\WeiboData\\CMO\\",personcnt=200,experimentimes=100,k=7,centerfilep=centerfilep,attfilep=attfilep)  
    # for k in range(2,8):
    #     centerfilep = r'N:\HFS\WeiboData\CMO\Modefigs\200_100.attcenter'
    #     attfilep = 'N:\\HFS\\WeiboData\\CMO\\Modefigs\\'+str(k)+'_200_100.att'
    #     drawtypical.start(fileinfloder="N:\\HFS\\WeiboData\\CMO\\",personcnt=200,experimentimes=100,k=k,centerfilep=centerfilep,attfilep=attfilep)  
  