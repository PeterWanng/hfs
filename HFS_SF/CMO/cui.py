#encoding=utf8
import sys
sys.path.append("..\..")
from tools import commontools as gtf
from matplotlib import pyplot as plt
import numpy as np
import csv

gt = gtf()
def read_weighted_graph(file_path):
        g = igraph.Graph(directed = True)
        g.add_vertices(100000)

        with open(file_path) as f:
                for line in f:
                        tokens = line.split("\t")
                        s_id = int(tokens[0])
                        t_id = int(tokens[1])
                        w = float(tokens[2])
                        g.add_edge(s_id,t_id, weight=w)
        return g
        IG = igraph.load('100k_sample_unique_igraph.txt', format='edgelist')
        
        IG = read_weighted_graph(graph_file_path)
        
        
# attfile = gt.csv2list_new(r'G:\HFS\WeiboData\Cui\att\graph - all [Nodes].csv',passmetacol=1,convertype=float,nan2num=False,passmetarowcnt=0)
attfile = np.genfromtxt(fname=r'G:\HFS\WeiboData\Cui\att\graph - all [Nodes].csv', delimiter=',', skiprows=1, skip_header=0)
att = zip(*attfile)
for it in att:
    print it[1:]
    gt.listDistribution(it,disfigdatafilepath=None,xlabel='Amount',ylabel='Frequency',showfig=True,binsdivide=1)