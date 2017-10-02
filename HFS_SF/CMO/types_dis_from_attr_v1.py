#encoding=utf8
import sys
sys.path.append("..\..")
from tools import commontools
from matplotlib import pyplot as plt
from matplotlib import pylab

               
import igraph as ig
from igraph import clustering as clus


import re
import time
import os
from tools import commontools as gtf
import csv
import numpy as np 
import scipy as sp
from scipy import stats
gt=gtf()

def combine2list(lista,listb):
    res = []
    for a,b in zip(*(lista,listb)):
        res.append('T'+str(a)+' '+str(b))
    return res

def one_metric(attfile):
    attlist = np.genfromtxt(fname=attfile, dtype=float, comments=None, delimiter=',', skiprows=1, skip_header=0, skip_footer=0)
    attlistz = zip(*attlist)
    attlistz = np.nan_to_num(attlistz)
    #print attlist.shape
    return attlistz
            
def define_metrics(workfolder):
    att = []
    for filen in os.listdir(workfolder):
        attfile = workfolder+filen#r'G:\HFS\WeiboData\CMO\graphs\300_500_1_1.netattri'
        if os.path.isfile(attfile) and os.path.splitext(attfile)[-1]=='.netattri':
            print attfile
            attlistz = one_metric(attfile)
            att.extend(attlistz)
    return att

def fun_dis (x,y,n):  
    return sum (map (lambda v1,v2:pow(abs(v1-v2),n), x,y))  
 
def distance (x,y):  #adjust
    return fun_dis (x,y,2)

def OsDistance(vector1, vector2):
    sqDiffVector = vector1-vector2
    sqDiffVector=sqDiffVector**2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances**0.5
    return distance

def disformcenter(centerlist,attrlist):
    res = []
    for center in centerlist:
        for att in zip(*attrlist):
#             print len(att),len(center)
            oneres = distance(center,att[1:])#OsDistance(center,att[1:])#
            res.append(oneres)
    return res

def itemcntDis(labels):
    labelen = len(labels)
    distinct_label = sp.unique(ar=labels, return_index=True, return_inverse=True)
    distLabel = list(distinct_label[0])
    i = len(distLabel)
    res = []
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

def loadattributes(workfolder_fig,personcnt,experimenTimes):
    #         mode = 7
    att = []
    for mode in range(2,8):
        workfolder = "N:\\HFS\\WeiboData\\CMO\\Mode"+str(mode)+"\\graphs\\"#"G:\\HFS\\WeiboData\\CMO\\test\\"#
#         workfolder_fig = "N:\\HFS\\WeiboData\\CMO\\Mode"+str(mode)+"\\figs\\"
    #     x = define_metrics(worksfolder+'graphs\\')
    #     x1 = define_metrics(worksfolder+'Mode1\\graphs\\')
    #     x2 = define_metrics(worksfolder+'Mode2\\graphs\\')
        
    #     x1.extend(x2)
    #     x=x1#zip(*x1)
        
        for modefans in [1,2,4]:#3,,5#
            for modefr in [1,2,4]:#3,,5
                for modeal in [1,2,4]:#3,,5
                    for modemen in [1,2,4]:#3,,5
                        filep = str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modefr)+'_'+str(modeal)+'_'+str(modemen)+'.netattri'
                        print workfolder+filep
                        filetuple = os.path.splitext(filep)
                        if os.path.isfile(workfolder+filep):
                            attlistz = one_metric(workfolder+filep)
                            att.extend(attlistz)
        return att

def getLabels(att,workfolder_fig,k=6):
    x = zip(*att)

    xyz = x#zip(*x)
    xyzz = gt.normlistlist(listlista=xyz,metacolcount=0,sumormax='sum')#xyz[1:]#
    xy = zip(*xyzz)#[1:]
    z = zip(*xyz[:1])
    
    from NetypesfromGraph import NetypesfromGraph as nety
    netky = nety()
    
    labels,res = [],[] 
#             print len(xy),xy
    labels,kmcenter,kmfit = netky.kmeans(xy,k=k,runtimes=1)
    #             labels.sort()
#     print len(labels)#kmcenter,labels
    centername = workfolder_fig+'shapedis_kmeans_center.center'
    
    gt.savefigdata(centername,kmcenter,labels,kmfit)
#     gt.saveList(centername,kmcenter)
    return labels  

def start(att,k):
    workfolder_fig = gt.createFolder("N:\\HFS\\WeiboData\\CMO\\Modefigs\\fig\\kmeans-"+str(k)+'\\')
    personcnt = 200
    experimenTimes = 100
    

    labels = getLabels(att,workfolder_fig,k)

#     labels = np.genfromtxt('N:\\HFS\\WeiboData\\CMO\\Modefigs\\'+str(personcnt)+'_'+str(experimenTimes)+'.attlabel',dtype='int')
    xlen = len(labels)
#     figposx = (xlen/experimenTimes)/4
    j = 0
    fig = plt.figure()#figsize=(15, 9.5)
    colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
    markerlist = list('x+.*_psv^,><12348hHD|od')
    leglabels = []
#     linestyle = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted',]
    
    result = []
    subresult = []
    subleglabel = []
    for i in range(0,xlen,experimenTimes):

        print i,xlen,experimenTimes
        label = labels[i:i+experimenTimes]
        res,dislabel,lastres = itemcntDis(label)
        j+=1
        ax = plt.subplot(3,3,j%9)
        ax.set_ylim((0, 1))
        coltype = 0
        fig.set_size_inches(16, 10)
        for item in zip(*res):
            xx = range(len(item))
            plt.plot(xx,item,color=colorlist[dislabel[coltype]],marker=markerlist[dislabel[coltype]], markeredgewidth=0.05,)#,linestyle=''
            plt.ylim(ymin=-0.1,ymax=1.4)
            coltype += 1
            subresult.append(item)
        lastres2 = []
        for it in lastres:
            lastres2.append(format(it, '.1%'))#lambda lastres, n: round(lastres, 2);# round(lastres, 2)
        leglabel = combine2list(dislabel,lastres2)
        subleglabel.append(leglabel)
        plt.legend(leglabel,fontsize=8,loc='upper right',  frameon=True, ncol=1+len(dislabel)/3)

        if j%9==0:
            fig.dpi = 300
            figname = workfolder_fig+'Ashapedis_'+str(personcnt)+'_'+str(experimenTimes)+'_'+str(j)+'.png'
            pylab.savefig(figname, dpi=fig.dpi)
#             fig.savefig(figname, dpi=fig.dpi)
            gt.savefigdata(datafilepath=figname+'.data',x=xx,y=subresult,errorbarlist=None,title='title',xlabel='',ylabel='',leglend=subleglabel)
            leglabels.append(subleglabel)
            result.append(subresult)
            subresult = []
            subleglabel = []
#             plt.show()
            plt.clf()
    gt.savefigdata(datafilepath=figname+'_all.data',x=xx,y=result,errorbarlist=None,title='title',xlabel='',ylabel='',leglend=leglabels)
    gt.saveList(leglabels,figname+'_all.leglends')

 
if __name__=='__main__':
#     att = loadattributes(workfolder_fig,personcnt,experimenTimes)
    cols = [0,1,2,3,4,5,6,7,8,9,10,11,17,18,19,20,21]
    att = np.genfromtxt(fname=r'N:\HFS\WeiboData\CMO\Modefigs\200_100.att', dtype=float, comments=None, delimiter=' ', skip_header=0, skip_footer=0,usecols=cols)
    for k in range(2,8,1):
        print k
        start(att,k)

if __name2__=='__main__':
    workfolder_fig = "N:\\HFS\\WeiboData\\CMO\\Modefigs\\fig\\"
    personcnt = 100
    experimenTimes = 100
    
#     att = loadattributes(workfolder_fig,personcnt,experimenTimes)
#     labels = getLabels(att,workfolder_fig)

    labels = np.genfromtxt('N:\\HFS\\WeiboData\\CMO\\Modefigs\\'+str(personcnt)+'_'+str(experimenTimes)+'.attlabel',dtype='int')
    xlen = len(labels)
#     figposx = (xlen/experimenTimes)/4
    j = 0
    fig = plt.figure()#figsize=(15, 9.5)
    colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
    markerlist = list('x+.*_psv^,><12348hHD|od')
    leglabels = []
#     linestyle = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted',]
    
    result = []

    for i in range(0,xlen,experimenTimes):
        subresult = []
        subleglabel = []
        print i,xlen,experimenTimes
        label = labels[i:i+experimenTimes]
        res,dislabel,lastres = itemcntDis(label)
        j+=1
        ax = plt.subplot(3,3,j%9)
        ax.set_ylim((0, 1))
        coltype = 0
        fig.set_size_inches(15, 9.5)
        for item in zip(*res):
            xx = range(len(item))
            plt.plot(xx,item,color=colorlist[dislabel[coltype]],marker=markerlist[dislabel[coltype]], markeredgewidth=0.05,)#,linestyle=''
            plt.ylim(ymin=-0.1,ymax=1.4)
            coltype += 1
            result.append(item)
        lastres2 = []
        for it in lastres:
            lastres2.append(format(it, '.1%'))#lambda lastres, n: round(lastres, 2);# round(lastres, 2)
        leglabel = combine2list(dislabel,lastres2)
        leglabels.append(leglabel)
        plt.legend(leglabel,fontsize=8,loc='upper right',  frameon=True, ncol=1+len(dislabel)/3)

        if j%9==0:
            fig.dpi = 300
            figname = workfolder_fig+'Ashapedis_'+str(personcnt)+'_'+str(experimenTimes)+'_'+str(j)+'.png'
            pylab.savefig(figname, dpi=fig.dpi)
#             fig.savefig(figname, dpi=fig.dpi)
            gt.savefigdata(datafilepath=figname+'.data',x=xx,y=result,errorbarlist=None,title='title',xlabel='',ylabel='',leglend=leglabels)
#             plt.show()
            result = []
            plt.clf()    
    

if __name__=='__main2__': 
#     resultone = ['filen']
#     g = ig.Graph.Read_GML(gmlfolder+filen)
#     resultone.extend(featuresofgraph(g))
#     result.append(resultone)
    x = define_metrics()
    x = gt.normlistlist(listlista=x,metacolcount=0,sumormax='sum')
    center =np.array([(9.67419916e-05,6.68980119e-05,1.29349492e-04,1.68278901e-04, -4.23595285e-04,3.61408134e-05,4.60954452e-05,3.80404008e-05,  1.03624365e-04,1.57017372e-04,1.39165784e-04)]) 
#     center =np.array([(1.55802783e-04,1.93787511e-04,1.12088361e-04,8.37170455e-05,  1.18811589e-03,2.22485345e-04,2.15465665e-04,2.19719376e-04,  1.50553847e-04,8.80977863e-05,1.14123747e-04)]) 
#     center =np.array([(8.83006580e-05,5.45034478e-05,1.71655478e-04,1.66703867e-04, -1.41980914e-03,2.32674656e-05,2.28648107e-05,2.57748312e-05,  9.73587322e-05,1.64112797e-04,1.38747862e-04)]) 
#     center =np.array([(1.52121986e-04,1.92855799e-04,1.27012909e-04,8.77553475e-05,  6.51677615e-04,2.15781900e-04,2.03160943e-04,2.16451382e-04,  1.48853797e-04,8.98137611e-05,1.13630163e-04)]) 
#     center =np.array([(1.58717805e-04,1.85075721e-04,9.40067252e-05,8.31975689e-05,  1.70614618e-03,2.19630091e-04,2.27089015e-04,2.15533659e-04,  1.50041979e-04,8.88140696e-05,1.14407125e-04)]) 
#     center =np.array([(9.56496515e-05,5.57631895e-05,1.25614504e-04,1.64444607e-04, -9.55601811e-04,2.82328003e-05,3.25076774e-05,3.04794584e-05,  9.90917676e-05,1.62775599e-04,1.33115999e-04)])
    disall = disformcenter(center,x)
    minv = np.min(disall)
    print disall.index(minv),minv
#     print disall
    
  