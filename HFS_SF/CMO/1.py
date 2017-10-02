#coding=utf8

import re
import time
import os
import sys
sys.path.append('..\..')
from tools import commontools as gtf
import csv
import igraph as ig
from igraph import clustering as clus
import numpy as np
from matplotlib import pyplot as plt
 
gt=gtf()

def average_mats(matlist,axisvalue=1):
    x = [[1,2,3],[4,5,6],[7,8,9]]
    y = [[10,20,30],[40,50,60],[70,80,90]]
    z = matlist#[x,y]
    xy = zip(*z)
    xyavg = np.average(xy,1)
    return xyavg
    
def compatts(workfolder,mod):
    atts = []#np.empty(0)

    for exptimes in range(10):
        attfp = workfolder+'sim_'+str(mod)+'_'+str(exptimes)+'.att'
        att = np.genfromtxt(attfp,float)
        print np.shape(att),attfp
        atts.append(att)
#         atts = np.append(atts,att,axis=0)
    atts = np.asarray(atts)
    return atts

def gen_avgatt(workfolder,mod):
    attsavgfp = workfolder+'sim_'+str(mod)+'_avg.att'
    attsfp = workfolder+'sim_'+str(mod)+'.att'
    atts = compatts(workfolder,mod)
    gt.saveList(atts,attsfp)
     
    avgatts = average_mats(atts)
    np.savetxt(attsavgfp,avgatts)
    return avgatts
    
workfolder = gt.createFolder("N:\\HFS\\WeiboData\\HFSWeibo_Sim\\Output\\") 
realattfp = r'N:\HFS\WeiboData\HFSWeibo_Sim\Output\real.att'
real = np.genfromtxt(realattfp)
simatts = [zip(*(real))]
for mod in range(2,8):
    simatt = np.genfromtxt(r'N:\HFS\WeiboData\HFSWeibo_Sim\Output\sim_'+str(mod)+'_avg.att')#gen_avgatt(workfolder,mod)
    simatt = np.nan_to_num(simatt)
    simatts.append(zip(*(simatt)))

simatts = np.nan_to_num(simatts)  

#6 metrics1 = [len(deg[avgdegindex_above:]),lendegabove/deglen,np.average(deg_abovepart)/lendegabove,np.std(deg_abovepart)/lendegabove,nodedis/lendegabove,assor]
#6 netCentrity =  'degree,betweenness,coreness,closeness,eccentricity,pagerank'
#5 net_tree_star_line all
#5 net_tree_star_line core

i = 0
for a in zip(*(simatts)):#zip(*(zip(*(real)),zip(*(simatt)))):
    i+=1
    print i
    try:
        b = a
        x = range(1,len(a)+1)
        b = np.nan_to_num(b)
        a = list(a)
    #     a.sort()
        b = list(b)
    #         b.sort()
#         print a
#         print b
        xlabels=['Real','Sim_Mode2','Sim_Mode3','Sim_Mode4','Sim_Mode5','Sim_Mode6','Sim_Mode7']
        gt.list_2_Distribution(a,xlabels,ylabels=['Frequency',])
    #         plt.plot(x,a)
    #         plt.plot(x,b)
    #         plt.show()
    #         plt.close()
    except Exception,e:
        print 'error' ,e   