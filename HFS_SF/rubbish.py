#encoding=utf8
import sys
import numpy
sys.path.append("..")
from tools import commontools as gtf
from tools import specialtools as gts
import time
import re
import csv
import os


gt=gtf()
gt.createFolder(r'c:\a')
gt.createFolder(r'c:\a',keepold=False)


# a = [[1,2,3],[4,5,6],[7,8,9]]
# print numpy.ndarray(shape=(1,),buffer = numpy.array(a[0]))
# x = zip(*a)[0:1]
# y = zip(*a)[1:]
# 
# # x = zip(*x)
# y = zip(*y)
# print x,y
# print numpy.array(x[0]),numpy.array(y)


# vcount   =  [(0.2, 0.16666666666666666, 0.15789473684210525, 0.08163265306122448, 0.09433962264150944, 0.0967741935483871, 0.10294117647058823, 0.1095890410958904, 0.09375, 0.08849557522123894, 0.09243697478991597, 0.0975609756097561, 0.10483870967741936, 0.10606060606060606, 0.10638297872340426, 0.10884353741496598, 0.11409395973154363, 0.1125, 0.1165644171779141, 0.12121212121212122, 0.11864406779661017, 0.12290502793296089, 0.12041884816753927, 0.12060301507537688, 0.12437810945273632, 0.12682926829268293, 0.125, 0.1278538812785388, 0.1318181818181818, 0.13333333333333333, 0.13478260869565217, 0.13675213675213677, 0.13807531380753138, 0.14049586776859505, 0.14344262295081966, 0.1469387755102041, 0.1434108527131783, 0.14615384615384616, 0.1482889733840304, 0.15151515151515152, 0.14642857142857144, 0.14840989399293286, 0.14776632302405499, 0.1461794019933555, 0.1485148514851485, 0.1503267973856209, 0.15309446254071662, 0.1553398058252427, 0.15483870967741936), (64, 86, 99, 168, 175, 190, 211, 221, 291, 364, 406, 438, 438, 501, 548, 584, 590, 726, 737, 755, 880, 906, 1088, 1261, 1360, 1504, 1943, 2019, 2027, 2052, 2127, 2367, 2573, 2730, 2970, 2993, 4811, 5214, 6171, 6253, 12574, 13510, 28319, 59402, 67147, 99037, 100922, 166006, 244245)]
# 
# a = [['3343408888337055.gml_strongGaint',0.1,285,320,320,295,374,288,177,300,279,12,330,352,145,26,198,77,154,86,287,187],['3343408888337055.gml_strongGaint',0.2,320,295,288,300,12,352,26,77,86,358,345,179,283,173,296,163,73,109,321,138]]

    
# from sklearn import metrics
# y = np.array([1,1,1,1,-1,-1,-1,-1])#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
# pred = np.array([0.36636782, -0.09257916,  0.48365673, -0.5077068,   0.54621056, -0.00749949, -0.10816487, -0.09116354])#[0.55768618,-0.57688208,0.69919084,-0.01816183,-0.0683632,-0.1126156,-0.46094844,-0.32206727])#[0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
# print metrics.auc(fpr, tpr)

# import igraph as ig
# g = ig.Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
# g.vs["label"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George", "Georgef"]
# # g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
# g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
# g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
# g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]
# gt.drawgraph(g, giantornot=False)


    
    
# import networkx as nx
# G=nx.path_graph(3)
# cocfilepath = r'G:\HFS\WeiboData\HFSWeiboCOC\3343740313561521.coc'
# G=nx.read_edgelist(cocfilepath, delimiter=',', create_using=nx.MultiDiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
# 
# 
# # path=r'G:\HFS\WeiboData\test\test4batchsetattri.gml'
# # G = nx.read_gml(path,encoding='UTF-8', relabel=True)
# print G
# bb=nx.betweenness_centrality(G)
# # bb = {0: 0.0, 1: 1.0, 2: 0.0}
# nx.set_node_attributes(G,'betweenness',bb)
# print bb
# print G.node[u'\xe6\xb5\x81\xe6\xb5\xaa\xe5\x96\xb5-Ann']['betweenness']

