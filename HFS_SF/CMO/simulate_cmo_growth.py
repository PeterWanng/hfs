#encoding=utf8
"fixed the wrong turns of   input = np.array([ma_fans[:i],cbm_frs[:i],inv_mention[:i],act_micorcnt[:i]])===>        input = np.array([ma_fans[:i],cbm_frs[:i],act_micorcnt[:i],inv_mention[:i]])"
"before this version, the simulation process are wrong with above, but the adj file are all right because there is another place have wrong turn, then wrong wrong is right,haha"
"V6 want: add more mode based on the 6modes, which may change the preference functions etc.."
"---add pure repost matrix of mode with invitations--which means two matrix will be produced"
"---+efforts vectors, which is info length each weibo repost"

import os
import sys
sys.path.append('..\..')
from tools import commontools as gtf
gt = gtf()

import weibo_tools
from weibo_tools import weibo2graph
from weibo_tools import wbg
wb2g = weibo2graph()
    
import numpy as np
import pylab
import time
# from numpy import matrix as mx
# from numpy import random
import matplotlib
import matplotlib.pyplot as plt
import igraph as ig

import scipy as sp
from scipy import stats
from scipy import matrix as mx
from scipy import random

class cmo_growth_model():

    def convertype(self,y):
        y = map(int,y)
        y.sort()
        return y
    
    def sigmoid(self,inX):
        return 1.0 / (1 + exp(-inX))
    
        
    def plothem(self,ma_fans,cbm_frs,inv_mention,act_micorcnt):
        x = range(1,101)
        plt.semilogy(x,convertype(ma_fans),marker='o')
        plt.semilogy(x,convertype(cbm_frs),marker='o')
        plt.semilogy(x,convertype(inv_mention),marker='o')
        plt.legend(['fans','frs','inv'])
        plt.show()
    
    def addrow_mat(self,a,rows2addcnt=1):
        ai = a#sp.identity(a.shape[1])
        # aif = ai.flat
        ail = ai.tolist()
        for i in range(rows2addcnt):
            rowtoadd = [0 for i in range(1,len(ail[0])+1)]
            ail.append(rowtoadd)
        
        ailm = sp.asmatrix(ail)
        return ailm
    
    def addrowcol_matrix(self,preoutputmatrix,rowsadd=1,colsadd=1):
        "IN:n*n matrix; how many rows and cols to add" 
        "OUT:(n+rowsadd)*(n+colsadd) matrix with new zero items"
        tempout = self.addrow_mat(preoutputmatrix,rows2addcnt=rowsadd)
        outm = self.addrow_mat(tempout.transpose(),rows2addcnt=colsadd)
        return outm.transpose()
    
    def distance_network(self,adjmatrix,hubnodeindex=0):
        "IN: linjie matrix of a network; the hub node index"
        "OUT: the distance between hub and each node list " 
        if adjmatrix.shape[0]==1:
            dislist = [[0]]
        else:
            g = ig.Graph.Adjacency(adjmatrix.tolist())
            dislist = g.shortest_paths_dijkstra(source=None, target=0, weights=None, mode='OUT') 
    #         gt.drawgraph(g,giantornot=False,figpath='test.png')
        return dislist   
    
    def findindexofvalues(self,lista,value):
        i = 0
        indexreturn = []
        for item in lista:        
            if item==value:
               indexreturn.append(i)
            i+=1
        return indexreturn 
    
    '=====modes======================================================================================================================'
    def pos_fans_C(self,inputmlist,preoutputmatrix):
        fans = inputmlist[0]
        fansum = float(sp.sum(fans))
        fansratio = [i/fansum for i in fans]
        
        preoutputmatrix = self.addrowcol_matrix(preoutputmatrix,1,1)
        outputmatrixDim = preoutputmatrix.shape[0]
        #     adjmatrix = mx('0,0,0;1,0,0;1,0,0')#'0,0,0;1,0,0;1,0,0'#
    #     adjmatrix = inputmatrix#random.randint(low=0,high=2,size=(30,30))
    #     print adjmatrix
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            chance2selected.append(afansr*exp(-adis[0]))#(afansr*10**-adis[0])
        problist = chance2selected/(sp.sum(chance2selected))
        choosed = sp.random.choice(outputmatrixDim-1,1,p=problist)
        pos2change = [(outputmatrixDim-1,choosed[0]),]
        return pos2change
    
    def pos_fans_fr_C(self,inputmlist,preoutputmatrix):
        fans = inputmlist[0]
        fri = inputmlist[1]
        fafr = 1.0*fans/fri
        
        fansum = float(sp.sum(fafr))
        fansratio = [i/fansum for i in fafr]
        
        preoutputmatrix = self.addrowcol_matrix(preoutputmatrix,1,1)
        outputmatrixDim = preoutputmatrix.shape[0]
        #     adjmatrix = mx('0,0,0;1,0,0;1,0,0')#'0,0,0;1,0,0;1,0,0'#
    #     adjmatrix = inputmatrix#random.randint(low=0,high=2,size=(30,30))
    #     print adjmatrix
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            chance2selected.append(afansr*10**-adis[0])
        problist = chance2selected/(sp.sum(chance2selected))
        choosed = sp.random.choice(outputmatrixDim-1,1,p=problist)
        pos2change = [(outputmatrixDim-1,choosed[0]),]
        return pos2change
    
    
    def pos_fans_fr_al_C(self,inputmlist,preoutputmatrix):
        fans = inputmlist[0]
        fri = inputmlist[1]
        al = inputmlist[2]
        fafr = al*1.0*fans/fri
        fafr += 0.0000001 if fafr.all()==0 else fafr
    #     print al,fafr
        
        fansum = float(sp.sum(fafr))
        fansratio = [i/fansum for i in fafr]
        
        preoutputmatrix = self.addrowcol_matrix(preoutputmatrix,1,1)
        outputmatrixDim = preoutputmatrix.shape[0]
        #     adjmatrix = mx('0,0,0;1,0,0;1,0,0')#'0,0,0;1,0,0;1,0,0'#
    #     adjmatrix = inputmatrix#random.randint(low=0,high=2,size=(30,30))
    #     print adjmatrix
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            chance2selected.append(afansr*10**-adis[0])
        problist = chance2selected/(sp.sum(chance2selected))
        choosed = sp.random.choice(outputmatrixDim-1,1,p=problist)
        pos2change = [(outputmatrixDim-1,choosed[0]),]
        return pos2change        
    
    def pos_fans_CI(self,inputmlist,preoutputmatrix,netDim):
        pos2change = []
        choosed = []
        
        outputmatrixDim = preoutputmatrix.shape[0]
        fans = inputmlist[0]
        fansum = float(sp.sum(fans))
        fansratio = [i/fansum for i in fans]
        
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            chance2selected.append(afansr*10**-adis[0])
        problist = chance2selected/(sp.sum(chance2selected))
        nonzerorange = self.findindexofvalues(preoutputmatrix.any(axis=1).getA1(),True)#choosePatMatrix(preoutputmatrix,mode='IN')
    #     nonzerorange = nonzerorange.choose()
        outputmatrixDim = len(nonzerorange)
    #     print outputmatrixDim,len(problist)
        
        lena = len(problist)
        choosed.append(sp.random.choice(nonzerorange,1,p=problist))
        #print outputmatrixDim,'retwit:',choosed
        
        "add mention"
        mention = inputmlist[3]
        mentionsum = float(sp.sum(mention))
        mentionsum = mentionsum if mentionsum>0 else 1
    #     print mentionsum,mention
        mentionratio = [i/mentionsum for i in mention]
        
        mentioncnt = int(mention[-1])
    #     print mentioncnt
        for i in range(1,mentioncnt+1):
            signin = (outputmatrixDim-1)/float(netDim)
            if random.random()>=signin:
    #             preoutputmatrix = addrowcol_matrix(preoutputmatrix,1,1)
                pos2change.append((outputmatrixDim-1,outputmatrixDim+i-1))
            else:
                problist = mentionratio
    #             print outputmatrixDim,len(problist),problist
                choosed.append(sp.random.choice(outputmatrixDim,1,p=problist))
        #print outputmatrixDim,'retwit & invite:',choosed
        
        for cha in choosed:        
            pos2change.append((nonzerorange[-1]+1,cha[0]))
        return pos2change
    
    def pos_fans_fr_CI(self,inputmlist,preoutputmatrix,netDim):
        pos2change = []
        choosed = []
        
        outputmatrixDim = preoutputmatrix.shape[0]
        fans = inputmlist[0]
        fri = inputmlist[1]
        fafr = 1.0*fans/fri
        
        fansum = float(sp.sum(fafr))
        fansratio = [i/fansum for i in fafr]
        
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            chance2selected.append(afansr*10**-adis[0])
        problist = chance2selected/(sp.sum(chance2selected))
        nonzerorange = self.findindexofvalues(preoutputmatrix.any(axis=1).getA1(),True)#choosePatMatrix(preoutputmatrix,mode='IN')
    #     nonzerorange = nonzerorange.choose()
        outputmatrixDim = len(nonzerorange)
    #     print outputmatrixDim,len(problist)
        
        lena = len(problist)
        choosed.append(sp.random.choice(nonzerorange,1,p=problist))
        
        "add mention"
        mention = inputmlist[3];print mention
        mentionsum = float(sp.sum(mention))
        mentionsum = mentionsum if mentionsum>0 else 1
    #     print mentionsum,mention
        mentionratio = [i/mentionsum for i in mention]
        
        mentioncnt = int(mention[-1])
    #     print mentioncnt
        for i in range(1,mentioncnt+1):
            signin = (outputmatrixDim-1)/float(netDim)
            if random.random()>=signin:
    #             preoutputmatrix = addrowcol_matrix(preoutputmatrix,1,1)
                pos2change.append((outputmatrixDim-1,outputmatrixDim+i-1))
            else:
                problist = mentionratio
    #             print outputmatrixDim,len(problist),problist
                choosed.append(sp.random.choice(outputmatrixDim,1,p=problist))
            
        for cha in choosed:        
            pos2change.append((nonzerorange[-1]+1,cha[0]))
        return pos2change
    
    def pos_fans_fr_al_CI(self,inputmlist,preoutputmatrix,netDim):
        pos2change = []
        choosed = []
        
        outputmatrixDim = preoutputmatrix.shape[0]
        fans = inputmlist[0]
        fri = inputmlist[1]
        al = inputmlist[2]
        fafr = al*1.0*fans/fri
        fafr += 0.0000001 if fafr.all()==0 else fafr
        
        fansum = float(sp.sum(fafr))
        fansratio = [i/fansum for i in fafr]
        
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            #chance2selected.append(afansr*10**-adis[0])
            chance2selected.append(afansr*exp(-adis[0]))
        problist = chance2selected/(sp.sum(chance2selected))
        nonzerorange = self.findindexofvalues(preoutputmatrix.any(axis=1).getA1(),True)#choosePatMatrix(preoutputmatrix,mode='IN')
    #     nonzerorange = nonzerorange.choose()
        outputmatrixDim = len(nonzerorange)
    #     print outputmatrixDim,len(problist)
        
        lena = len(problist)
        choosed.append(sp.random.choice(nonzerorange,1,p=problist))
        
        "add mention"
        mention = inputmlist[3]
        mentionsum = float(sp.sum(mention))
        mentionsum = mentionsum if mentionsum>0 else 1
    #     print mentionsum,mention
        mentionratio = [i/mentionsum for i in mention]
        
        mentioncnt = int(mention[-1])
    #     print mentioncnt
        for i in range(1,mentioncnt+1):
            signin = (outputmatrixDim-1)/float(netDim)
            if random.random()>=signin:
    #             preoutputmatrix = addrowcol_matrix(preoutputmatrix,1,1)
                pos2change.append((outputmatrixDim-1,outputmatrixDim+i-1))
            else:
                problist = mentionratio
    #             print outputmatrixDim,len(problist),problist
                choosed.append(sp.random.choice(outputmatrixDim,1,p=problist))
            
        for cha in choosed:        
            pos2change.append((nonzerorange[-1]+1,cha[0]))
        return pos2change
            
    def pos_fans_fr_al_CI2(self,inputmlist,preoutputmatrix,netDim):
        pos2change = []
        choosed = []
        
        outputmatrixDim = preoutputmatrix.shape[0]
        fans = inputmlist[0]
        fansum = float(sp.sum(fans))
        fansratio = [i/fansum for i in fans]
        
        #     adjmatrix = mx('0,0,0;1,0,0;1,0,0')#'0,0,0;1,0,0;1,0,0'#
    #     adjmatrix = inputmatrix#random.randint(low=0,high=2,size=(30,30))
    #     print adjmatrix
        dislist = self.distance_network(preoutputmatrix,hubnodeindex=0)
        chance2selected = []
        for afansr,adis in zip(*(fansratio,dislist)):
            chance2selected.append(afansr*np.e**-adis[0])
        problist = chance2selected/(sp.sum(chance2selected))
        nonzerorange = self.findindexofvalues(preoutputmatrix.any(axis=1).getA1(),True)#choosePatMatrix(preoutputmatrix,mode='IN')
    #     nonzerorange = nonzerorange.choose()
        outputmatrixDim = len(nonzerorange)
    #     print outputmatrixDim,len(problist)
        
        lena = len(problist)
        choosed.append(sp.random.choice(nonzerorange,1,p=problist))
        
        "add mention"
        mention = inputmlist[3]
        mentionsum = float(sp.sum(mention))
        mentionsum = mentionsum if mentionsum>0 else 1
    #     print mentionsum,mention
        mentionratio = [i/mentionsum for i in mention]
        
        mentioncnt = int(mention[-1])
    #     print mentioncnt
        for i in range(1,mentioncnt+1):
            signin = (outputmatrixDim-1)/float(netDim)
            if random.random()>=signin:
    #             preoutputmatrix = addrowcol_matrix(preoutputmatrix,1,1)
                pos2change.append((outputmatrixDim-1,outputmatrixDim+i-1))
            else:
                problist = mentionratio
    #             print outputmatrixDim,len(problist),problist
                choosed.append(sp.random.choice(outputmatrixDim,1,p=problist))
            
        for cha in choosed:        
            pos2change.append((nonzerorange[-1]+1,cha[0]))
        return pos2change

    def pos_fans_fr_al_CI_pref(self,inputmlist,preoutputmatrix,netDim):
        "Mode 11: parameter belta with distance preference"
    
    
    def efforts(self):
        print 'waiting for ...'
    
    '=====modes======================================================================================================================'


    def getPosition2Change(self,inputmlist,preoutputmatrix,netDimAll,mode=1):        
        "set according item based on inputmatrix"
        pos2change = []
        if mode==2:
            pos2change = self.pos_fans_C(inputmlist,preoutputmatrix)
        if mode==3:
            pos2change = self.pos_fans_fr_C(inputmlist,preoutputmatrix)
        if mode==4:
            pos2change = self.pos_fans_fr_al_C(inputmlist,preoutputmatrix)
        if mode==5:
            pos2change = self.pos_fans_CI(inputmlist,preoutputmatrix,netDim=netDimAll)
        if mode==6:
            pos2change = self.pos_fans_fr_CI(inputmlist,preoutputmatrix,netDim=netDimAll)
        if mode==7:
            pos2change = self.pos_fans_fr_al_CI(inputmlist,preoutputmatrix,netDim=netDimAll)
        if mode==8:
            pos2change = self.pos_fans_fr_al_CI2(inputmlist,preoutputmatrix,netDim=netDimAll)
        
        return pos2change
    
    def setVauleofMatrix(self,preoutputmatrix,setpositionlist,valuelist=None): 
        netDimNow = preoutputmatrix.shape[0]
        poslen = mx.max(sp.asmatrix(setpositionlist))#sp.maximum(setpositionlist)+1#len(setpositionlist)
        outmat = self.addrowcol_matrix(preoutputmatrix,poslen-netDimNow+1,poslen-netDimNow+1)
    #     print outmat.shape,
        for pos in setpositionlist:
            if pos:
                outmat.itemset(pos,outmat.item(pos)+1)
        return outmat
          
    def min2mout_mat(self,inputlist,preoutputmatrix,supposeMatrixDim,mode):
        "IN:two matrix,one is input matrix n*4,another is the pre-output matrix n-1*n-1"
        "OUT:a matrix n*n"    
                
        setposition = self.getPosition2Change(inputlist,preoutputmatrix,netDimAll=supposeMatrixDim,mode=mode)#((netDim-1,choosed[0]),)#((netDim-1,0),(netDim-1,netDim-1),)
        outmat = self.setVauleofMatrix(preoutputmatrix,setposition,valuelist=None)
        
    #     poslen = len(setposition)
    #     outmat = addrowcol_matrix(preoutputmatrix,poslen-netDimNow+1,poslen-netDimNow+1)
    #     print outmat.shape,
    #     for pos in setposition:
    #         if pos:
    #             outmat.itemset(pos,outmat.item(pos)+1)
        return outmat
    
    '----------------------------------------------------------------------------------------'
    def chose_rang_list(self,lista,size,low,high):
        "choose the item from lista in range[low,high] "
        "IN:1-d sequence;target select size;range"
        "OUT:selected list"
        result = []    
        for it in lista:
            if len(result)<size:
                if it>=low and it<=high:
                    result.append(it)
            else:
                break
        return result  
    
    def randomlist(self,mode=1,size=1000,low=0,high=100):
        "generate a list followed given distribution"
        "IN:distribution model code which refer to randomlist method; sequence size; sequence range"
        "OUT:a sequence followed distribution like"
        x = []
        if mode == 1:        
            x = random.randint(low,high,size=size)
        if mode == 2:
            x = map(float,random.normal(loc=(low+high)/2.0, scale=(high-low)/6.0, size=size))
        if mode == 3:
            x = map(long,random.exponential(scale=1, size=size)+low)
        if mode == 4:
            x = map(long,random.pareto(1,size=size)+low)
        if mode == 5:
            x = map(long,random.poisson(lam=(low+high)/2.0, size=size))    
    #     x = random.choice(x,size=100)    
        return x
    
    def generateSeq(self,mode=1,size=1000,low=0,high=100):
        "IN:distribution model code which refer to randomlist method; sequence size; sequence range"
        "OUT:a sequence followed distribution like"
        xx = []
        while len(xx)<size:
            x = self.randomlist(mode,1.2*size,low,high)
            xx.extend(self.chose_rang_list(x,size,low,high))
    #         print '+++++++++++++++' ,len(xx)
        return xx[:size] 
    '----------------------------------------------------------------------------------------'
    
    def input_init(self,personcnt,fans_p=(1,1,1000),frs_p=(1,1,1000),men_p=(1,0,50),micor_p=(1,1,1000)):  
        "initiate four sequences:"  
        "IN:sequence length;each sequence's parameter (distributon mode code;range low;range high)"
        "OUT:four sequence [ma_fans,cbm_frs,inv_mention,act_micorcnt]"  
            
        ma_fans = self.generateSeq(mode=fans_p[0],size=personcnt,low=fans_p[1],high=fans_p[2])
        cbm_frs = self.generateSeq(mode=frs_p[0],size=personcnt,low=frs_p[1],high=frs_p[2])
        inv_mention = self.generateSeq(mode=men_p[0],size=personcnt,low=men_p[1],high=men_p[2])
        act_micorcnt = self.generateSeq(mode=micor_p[0],size=personcnt,low=micor_p[1],high=micor_p[2])
        # plothem(ma_fans,cbm_frs,inv_mention,act_micorcnt)
        return [ma_fans,cbm_frs,inv_mention,act_micorcnt] 
    
    def getoutMatrix(self,personcnt,ma_fans,cbm_frs,inv_mention,act_micorcnt,worksfolder,mode):
        preoutmat = mx('1')
        for i in range(1,personcnt+1):
    #         print '-------------------',i
            input = np.array([ma_fans[:i],cbm_frs[:i],act_micorcnt[:i],inv_mention[:i]])
    #         input = [ma_fans[:i],cbm_frs[:i],inv_mention[:i],act_micorcnt[:i]]
            min2mout_pat_mat = self.min2mout_mat(input,preoutmat,personcnt,mode)
            preoutmat = min2mout_pat_mat
        return preoutmat
            
    def itemcntDis(self,labels):
        labelen = len(labels)
        distinct_label = sp.unique(ar=labels, return_index=True, return_inverse=True)
        distLabel = list(distinct_label[0])
        i = len(distLabel)
        res = []
        for j in range(i,labelen+1,i):
            j = labelen if j>labelen else j
            x = list(labels[:j])
            oneres = []
    #         oneres = np.bincount(distLabel)
            for k in distLabel:
                oneres.append(x.count(k))#np.append(oneres,[x.count(k)],axis=0)#
            oneres = gt.normalizelist(oneres,sumormax='sum')
            res.append(oneres)#np.append(res,[oneres])#
    #     print distLabel,res,oneres
        return res
    
    
    def classTypes_km(self,xyz,datapath2save,k=6,runtimes=1000):
        xyzz = gt.normlistlist(listlista=xyz[1:],metacolcount=0,sumormax='sum')#xyz[1:]#
        xy = zip(*xyzz)#[1:]
        z = zip(*xyz[:1])
        
        labels,res = [],[] 
    #             print len(xy),xy
        xy = sp.nan_to_num(xy)
        labels,kmcenter,kmfit = netky.kmeans(xy,k,show=False,runtimes=runtimes)
        #             labels.sort()
        print modefans,modemen,kmcenter,labels
        centername = datapath2save+'.center'#worksfolder_fig+'shapedis_'+str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modemen)+'.center'
    
        
        gt.savefigdata(centername,kmcenter,labels)
        res = itemcntDis(labels)
        fig = plt.figure()
        for item in zip(*res):
            xx = range(1,k*len(item),k)
            plt.plot(xx,item)
        figname = datapath2save+'.png'#worksfolder_fig+'shapedis_'+str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modemen)+'.png'
        fig.dpi = 300
        fig.savefig(figname, dpi=fig.dpi)
    #             pylab.savefig(figname, dpi=fig.dpi)
        gt.savefigdata(datafilepath=figname+'.data',x=xx,y=zip(*res),errorbarlist=None,title='title',xlabel='',ylabel='',leglend='')
        
        plt.close()
    
    def input_init_real(self,casefilepath):
        import re
        ma_fans,cbm_frs,act_micorcnt,inv_mention,bi_followers_count = [],[],[],[],[]
        caselist = gt.txt2list(casefilepath)
        caselistz = zip(*caselist)
        if len(caselistz)<47:
            print '****************',len(caselistz)
        i = 0
    #     for it in caselistz:
    #         print i,it
    #         i+=1
        for fa,fr,mic,men,bifa in zip(*(caselistz[27][1:],caselistz[28][1:],caselistz[29][1:],caselistz[3][1:],caselistz[47][1:])):
            #fa,fr,mic,bifa = fa[0:-1],fr[0:-1],mic[0:-1],bifa[0:-1]
            try:
                fa,fr,mic,bifa = long(fa),long(fr),long(mic),long(bifa)
            except:
                fa,fr,mic,bifa = 1,1,1,1
            
            fa = 1 if fa<1 else fa;fr = 1 if fr<1 else fr;mic = 1 if mic<1 else mic;bifa = 1 if bifa<1 else bifa;
            ma_fans.append(fa)        
            cbm_frs.append(fr)        
            act_micorcnt.append(mic)        
            bi_followers_count.append(bifa)        
            
            mindex = -1
            try:
                mindex = str(men).index(r'//@')
            except:
                pass
            itemn = str(men)[:mindex]#.replace(r'//@','')
            mentioneduser = re.findall(u"@",itemn)
            inv_mention.append(len(mentioneduser))
        return ma_fans,cbm_frs,inv_mention,act_micorcnt,bi_followers_count
    
    def input_new(self,caselistz,inputmetalist=None):
        res = []
        metaline = weibo_tools.wbmetalinelist_new
        if inputmetalist:
            for item in inputmetalist:
                i = metaline.index(item)
                res.append(caselistz[i]) 
        else:
            print 'warning wanng: did not changed anything'
            res = caselistz
        return res
        
    def input_init_real_new(self,casefilepath,addcomment=False):
        import re
        ma_fans,cbm_frs,act_micorcnt,inv_mention,bi_followers_count = [],[],[],[],[]
        casefilepath_cm = None if not addcomment else str(casefilepath).replace('.repost','.comment')
        caselist = wb2g.getxtlist(casefilepath,casefilepath_cm)
        caselistz = zip(*caselist)
        if len(caselistz)<47:
            print '****************',len(caselistz)
        is_oldweibo = weibo_tools.is_oldweibo(weibofilepath_or_list=caselist)
        if is_oldweibo:
            atts = (caselistz[27][1:],caselistz[28][1:],caselistz[29][1:],caselistz[3][1:],caselistz[47][1:])
        else:
            inputmetalist = ['followers_count', 'friends_count', 'statuses_count','text','bi_followers_count', ]
            atts = tuple(self.input_new(caselistz,inputmetalist))
            
        for fa,fr,mic,men,bifa in zip(*atts):
            #fa,fr,mic,bifa = fa[0:-1],fr[0:-1],mic[0:-1],bifa[0:-1]
            try:
                fa,fr,mic,bifa = long(fa),long(fr),long(mic),long(bifa)
            except:
                fa,fr,mic,bifa = 1,1,1,1
            
            fa = 1 if fa<1 else fa;fr = 1 if fr<1 else fr;mic = 1 if mic<1 else mic;bifa = 1 if bifa<1 else bifa;
            ma_fans.append(fa)        
            cbm_frs.append(fr)        
            act_micorcnt.append(mic)        
            bi_followers_count.append(bifa)        
            
            mindex = -1
            try:
                mindex = str(men).index(r'//@')
            except:
                pass
            itemn = str(men)[:mindex]#.replace(r'//@','')
            mentioneduser = re.findall(u"@",itemn)
            inv_mention.append(len(mentioneduser))
        return ma_fans,cbm_frs,inv_mention,act_micorcnt,bi_followers_count
    
      
    def start_sim(self,personcnt,experimenTimes,mode,worksfolder):           
    
        timenow = int(time.strftime("%H",time.localtime(time.time())))
    #     worksfolder = "G:\\HFS\\WeiboData\\CMO\\test\\"
        worksfolder_fig = gt.createFolder(worksfolder+'figs\\')
        worksfolder_mat = gt.createFolder(worksfolder+'graphs\\')
        from NetypesfromGraph import NetypesfromGraph as nety
        netky = nety()
        for modefans in [1,2,4]:#3,,5
            for modefr in [1,2,4]:#3,,5
                for modeal in [1,2,4]:#3,,5
                    for modemen in [1,2,4]:#3,,5
                        result = []
                        startimer = time.clock()
                        for i in range(experimenTimes):        
                            print '===============================================',i,modefans,modemen
                            [ma_fans,cbm_frs,inv_mention,act_micorcnt] = self.input_init(personcnt,fans_p=(modefans,1,10000),frs_p=(modefr,1,1000),micor_p=(modeal,0,1),men_p=(modemen,0,50))
                            input = [ma_fans,cbm_frs,inv_mention,act_micorcnt]
            #                 print act_micorcnt
                        #     inputm = sp.asmatrix(input, long)
                            
                            mat = self.getoutMatrix(personcnt,ma_fans,cbm_frs,inv_mention,act_micorcnt,worksfolder_fig,mode)
                            
                            g = ig.Graph.Adjacency(mat.tolist())
    #                         print g.vcount(),g.ecount()
                            graphfilepath = str(worksfolder_mat+str(personcnt)+'_'+str(experimenTimes)+'_'+str(i)+'_'+str(modefans)+'_'+str(modemen)+'.adj')
            #                 print graphfilepath
                            g.write_adjacency(graphfilepath)
    #                         if g.vcount()>50:
    #                             gt.drawgraph(g,giantornot=False,figpath=worksfolder+'figs\\'+str(modefans)+'_'+str(modemen)+'_'+str(i)+'.png',show=True)#sw
    #                             gt.drawgraph(g,giantornot=True,figpath=worksfolder+'figs\\'+str(modefans)+'_'+str(modemen)+'_'+str(i)+'giant.png',show=True)#sw
                            
                            
                            resultone = netky.start(g)
                            result.append(resultone)
                            
                        xyz = zip(*result)
                        gt.saveList(xyz,worksfolder_mat+str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modefr)+'_'+str(modeal)+'_'+str(modemen)+'.netattri',writype='w')
                        
            #             datapath2save = worksfolder_fig+'shapedis_'+str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modemen)
            #             classTypes_km(xyz,datapath2save,k=6,runtimes=10)
                
        endtimer = time.clock()
        duration = endtimer-startimer
        timestrline = str(time.asctime())+' in '+str(duration/60)+'mins of modefans '+str(modefans)+' modemen '+str(modemen)
        print timestrline
    
          
        
    def start_realcase(self,experimenTimes,mode,worksfolder_repost,workfolder_sim):        
        timenow = int(time.strftime("%H",time.localtime(time.time())))
        worksfolder = workfolder_sim
        worksfolder_fig = gt.createFolder(worksfolder+'figs\\')
        worksfolder_mat = gt.createFolder(worksfolder+'graphs\\')
        from NetypesfromGraph import NetypesfromGraph as nety
        netky = nety()
    
        result = []
        startimer = time.clock()
    #     try:
        for casefile in os.listdir(worksfolder_repost):
            if os.path.splitext(casefile)[-1]!='.repost':
                continue
            
            if os.path.exists(worksfolder_mat+'_'+str(mode)+'_'+casefile+'_9.adj'):
                print '_'+str(mode)+'_'+casefile+'.adj'
                continue
            
            print worksfolder_repost+casefile
            [ma_fans,cbm_frs,inv_mention,act_micorcnt] = self.input_init_real(worksfolder_repost+casefile)[0:4]
            input = [ma_fans,cbm_frs,inv_mention,act_micorcnt]
            print inv_mention
            #     inputm = sp.asmatrix(input, long)
            personcnt =  len(ma_fans)   
            for i in range(0,experimenTimes): 
                print '===============================================',i
                mat = self.getoutMatrix(personcnt,ma_fans,cbm_frs,inv_mention,act_micorcnt,worksfolder_fig,mode)
                
                g = ig.Graph.Adjacency(mat.tolist())
                graphfilepath = str(worksfolder_mat+'_'+str(mode)+'_'+casefile+'_'+str(i)+'.adj')
                print graphfilepath
                g.write_adjacency(graphfilepath)
        #                 gt.drawgraph(g,giantornot=False,figpath=worksfolder+'figs\\'+str(modefans)+'_'+str(modemen)+'_'+str(i)+'.png',show=False)#sw
                
                
    #             resultone = netky.start(g)
    #             result.append(resultone)
    #             
    #     xyz = zip(*result)
    #     gt.saveList(xyz,worksfolder_mat+'_'+str(mode)+'_'+casefile+'_'+str(experimenTimes)+'.netattri',writype='w')
    
    #     except:
    #         pass             
        #             datapath2save = worksfolder_fig+'shapedis_'+str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modemen)
        #             classTypes_km(xyz,datapath2save,k=6,runtimes=10)
                
        endtimer = time.clock()
        duration = endtimer-startimer
        timestrline = str(time.asctime())+' in '+str(duration/60)+'mins of modefans '#+str(modefans)+' modemen '+str(modemen)
        print timestrline
    
            
if __name__=="__main__":    
    personcnt = 10
    experimenTimes = 5
    mode = 6
    worksfolder_sim = gt.createFolder("G:\\HFS\\WeiboData\\HFSWeibo_Sim\\test\\Mode"+str(mode)+'\\')#
    worksfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\test2\\"#"#small\\
    
    sm = cmo_growth_model()
#     sm.start_sim(personcnt,experimenTimes,mode,worksfolder_sim)
    #print sm.input_init_real(r'G:\HFS\WeiboData\HFSWeibo\test\toy.repost')[0:4]
#     try:
    sm.start_realcase(experimenTimes,mode,worksfolder_real,worksfolder_sim)
#     except Exception,e:
#         print e
       
    
    
    
    
    
    
    
