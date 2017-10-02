#encoding = utf8
import os
import sys
sys.path.append('..')
from tools import commontools as gtf
# from tools import graphtools
# from weibo_tools import weibo2graph
#import random
import numpy as np
gt = gtf()
# weibo2g = weibo2graph()
# grt = graphtools()
# from cmo_regression import regression_cmo
# import igraph as ig


attfp = r"G:\HFS\WeiboData\HFSWeibo\ATT\stat\startWeibo\repcom_count.txt"
att = gt.csv2list_new(attfp,convertype=float)
atts = []
for it in att:
    atts.append(it)
    
    
dis = gt.list_2_Distribution([atts],binseqdiv=3,showfig=True)
print len(dis[0][0])
print map(int,list(dis[0][0]))
print list(map(dis[1],int))
er


def interval_dis(lista,savefp=None,shown=False):
    att = lista#
    #hist,bin_edges = np.histogram(att)
    #hist = numpy.histogram(lista,bins=(numpy.max(lista)+1)/binsdivide)
    attv = gt.interval_of_list(att,1)
    if savefp:
        gt.saveList([attv],savefp)
    if shown:
        attv = gt.csv2list_new(fp+'.interval')
        gt.list_2_Distribution(attv,binseqdiv=2)
    return attv



attfp = "G:\\HFS\\WeiboData\\HFSWeibo\\testNew\\ATT\\stat\\createdtimeos_interval.txt"
intervals = gt.csv2list_new(attfp)
att = []
for val in intervals:
    att.extend(val)
    
gt.list_2_Distribution([att],binseqdiv=2)
er
intervals = []
workfolder = "G:\\HFS\\WeiboData\\HFSWeibo\\testNew\\"
for fp in os.listdir(workfolder):
    fp = workfolder+fp
    fptypes = os.path.splitext(fp)
    if fptypes[-1] in ['.repost','.comment']:
        att = gt.csv2list_new(fp)#np.genfromtxt(fp)
        timelist = zip(*(att))[2]
        interval = interval_dis(timelist,savefp=None,shown=False)
        #intervals.extend(interval)
        gt.saveList([interval],attfp,writype='a+')
        






"done========================================================================================================================="
def metalist_1():
    picklefp1 = r'N:\dataset\HFS_XunRen_620\2014\meta\hotweibo_metainfo_timestr.dic'
    picklefp2 = r'N:\dataset\HFS_XunRen\2013BC\meta\hotweibo_metainfo_timestr.dic'
    gt.dic2list4meta(picklefp1,picklefp1+'.list')
    gt.dic2list4meta(picklefp2,picklefp2+'.list')        