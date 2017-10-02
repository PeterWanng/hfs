#encoding=utf8
import sys
sys.path.append('..\..')
from tools import commontools as gtf
gt = gtf()

import numpy as np
# from numpy import matrix as mx
# from numpy import random
import matplotlib.pyplot as plt
import igraph as ig

import scipy as sp
from scipy import stats
from scipy import matrix as mx
from scipy import random
def test():
    '''a = mx('1,2,3;0,4,5;9,0,8')
    print a.shape
    print a.I
    print mx.A'''
    a = mx('1,2;3,2')
    b = mx('1,0,0;0,1,1')
    c = mx('1,0;0,1;1,0')
    # print c*(a*b)
    print a.shape[0]
    ai = sp.identity(a.shape[1])
    # aif = ai.flat
    ail = ai.tolist()
    newit = ail[0]
    ail.append(newit)
    print ail#.repeat(2,1)#.reshape((2,))
    ailm = sp.asmatrix(ail)
    print ailm
    
    # c = mx('1,2;0,4;9,2')
    # d = mx('1,2,0;4,9,2')
    # print a*b
    
    #每加入一个节点，得一A同型+1单位阵，，修改Isi+=1，与A相乘得新输出矩阵
    linec = 0
    # for it in inputm:
    #     
    #     linec+=1

def getpowerlaw():
    rvs = sp.random.power(5, 1000000)
    rvsp = np.random.pareto(5, 1000000)
    xx = np.linspace(0,1,100)
    powpdf = stats.powerlaw.pdf(xx,5)
    
    plt.figure()
    plt.hist(rvs, bins=50, normed=True)
    plt.plot(xx,powpdf,'r-')
    plt.title('np.random.power(5)')
    
    plt.figure()
    plt.hist(1./(1.+rvsp), bins=50, normed=True)
    plt.plot(xx,powpdf,'r-')
    plt.title('inverse of 1 + np.random.pareto(5)')
    
    plt.figure()
    plt.hist(1./(1.+rvsp), bins=50, normed=True)
    plt.plot(xx,powpdf,'r-')
    plt.title('inverse of stats.pareto(5)')
    
    plt.show()

def convertype(y):
    y = map(int,y)
    y.sort()
    return y

def plothem(ma_fans,cbm_frs,inv_mention,act_micorcnt):
    x = range(1,101)
    plt.semilogy(x,convertype(ma_fans),marker='o')
    plt.semilogy(x,convertype(cbm_frs),marker='o')
    plt.semilogy(x,convertype(inv_mention),marker='o')
    plt.legend(['fans','frs','inv'])
    plt.show()

def addrow_mat(a,rows2addcnt=1):
    ai = a#sp.identity(a.shape[1])
    # aif = ai.flat
    ail = ai.tolist()
    for i in range(rows2addcnt):
        rowtoadd = [0 for i in range(1,len(ail[0])+1)]
        ail.append(rowtoadd)
    
    ailm = sp.asmatrix(ail)
    return ailm

def addrowcol_matrix(preoutputmatrix,rowsadd=1,colsadd=1):
    "IN:n*n matrix; how many rows and cols to add" 
    "OUT:(n+rowsadd)*(n+colsadd) matrix with new zero items"
    tempout = addrow_mat(preoutputmatrix,rows2addcnt=rowsadd)
    outm = addrow_mat(tempout.transpose(),rows2addcnt=colsadd)
    return outm.transpose()

def distance_network(adjmatrix,hubnodeindex=0):
    "IN: linjie matrix of a network; the hub node index"
    "OUT: the distance between hub and each node list " 
    if adjmatrix.shape[0]==1:
        dislist = [[0]]
    else:
        g = ig.Graph.Adjacency(adjmatrix.tolist())
        dislist = g.shortest_paths_dijkstra(source=None, target=0, weights=None, mode='OUT') 
#         gt.drawgraph(g,giantornot=False,figpath='test.png')
    return dislist   

def pos_fans(inputmlist,preoutputmatrix,outputmatrixDim):
    fans = inputmlist[0]
    fansum = float(sp.sum(fans))
    fansratio = [i/fansum for i in fans]
    
    preoutputmatrix = addrowcol_matrix(preoutputmatrix,1,1)
    outputmatrixDim = preoutputmatrix.shape[0]
    #     adjmatrix = mx('0,0,0;1,0,0;1,0,0')#'0,0,0;1,0,0;1,0,0'#
#     adjmatrix = inputmatrix#random.randint(low=0,high=2,size=(30,30))
#     print adjmatrix
    dislist = distance_network(preoutputmatrix,hubnodeindex=0)
    chance2selected = []
    for afansr,adis in zip(*(fansratio,dislist)):
        chance2selected.append(afansr*10**-adis[0])
    problist = chance2selected/(sp.sum(chance2selected))
    choosed = sp.random.choice(outputmatrixDim-1,1,p=problist)
    pos2change = [(outputmatrixDim-1,choosed[0]),]
    return pos2change
# 
# def choosePatMatrix(mat,mode='IN'):
#     "IN:a matrix;mode is, IN means indegree is zero,which means col i is zero"
#     "OUT: non zero index"
#     nonzeroindex = [0]
#     matlist = mat.tolist()
#     rowindex,colindex = 0,0
#     for row in matlist:
#         rowindex+=1
#         for item in row:
#             colindex+=1
#             if rowindex>1:
#                 if item>0 and colindex!=rowindex:
#                     nonzeroindex.append(rowindex-1)
#                     break
#     return nonzeroindex

def findindexofvalues(lista,value):
    i = 0
    indexreturn = []
    for item in lista:        
        if item==value:
           indexreturn.append(i)
        i+=1
    return indexreturn 
        
    
def pos_fansmention(inputmlist,preoutputmatrix,outputmatrixDim,netDim):
    pos2change = []
    choosed = []
    
    outputmatrixDim = preoutputmatrix.shape[0]
    fans = inputmlist[0]
    fansum = float(sp.sum(fans))
    fansratio = [i/fansum for i in fans]
    
    #     adjmatrix = mx('0,0,0;1,0,0;1,0,0')#'0,0,0;1,0,0;1,0,0'#
#     adjmatrix = inputmatrix#random.randint(low=0,high=2,size=(30,30))
#     print adjmatrix
    dislist = distance_network(preoutputmatrix,hubnodeindex=0)
    chance2selected = []
    for afansr,adis in zip(*(fansratio,dislist)):
        chance2selected.append(afansr*10**-adis[0])
    problist = chance2selected/(sp.sum(chance2selected))
    nonzerorange = findindexofvalues(preoutputmatrix.any(axis=1).getA1(),True)#choosePatMatrix(preoutputmatrix,mode='IN')
#     nonzerorange = nonzerorange.choose()
    outputmatrixDim = len(nonzerorange)
    print outputmatrixDim,len(problist)
    
    lena = len(problist)
    choosed.append(sp.random.choice(nonzerorange,1,p=problist))
    
    "add mention"
    mention = inputmlist[2]
    mentionsum = float(sp.sum(mention))
    mentionsum = mentionsum if mentionsum>0 else 1

    mentionratio = [i/mentionsum for i in mention]
    
    mentioncnt = mention[-1]
    print mentioncnt
    for i in range(1,mentioncnt+1):
        signin = (outputmatrixDim-1)/float(netDim)
        if random.random()>=signin:
#             preoutputmatrix = addrowcol_matrix(preoutputmatrix,1,1)
            pos2change.append((outputmatrixDim-1,outputmatrixDim+i-1))
        else:
            problist = mentionratio
            print outputmatrixDim,len(problist)
            choosed.append(sp.random.choice(outputmatrixDim,1,p=problist))
        
    for cha in choosed:        
        pos2change.append((nonzerorange[-1]+1,cha[0]))
    return pos2change

def getPosition2Change(inputmatrix,preoutputmatrix,outputmatrixDim,netDimAll,mode=1):        
    "set according item based on inputmatrix"
    pos2change = []
    inputmlist = zip(*inputmatrix.tolist())
    if mode==1:
        pos2change = pos_fans(inputmlist,preoutputmatrix,outputmatrixDim)
    if mode==2:
        pos2change = pos_fansmention(inputmlist,preoutputmatrix,outputmatrixDim,netDim=netDimAll)
    
    return pos2change

def setVauleofMatrix(preoutputmatrix,setpositionlist,valuelist=None): 
    netDimNow = preoutputmatrix.shape[0]
    poslen = mx.max(sp.asmatrix(setpositionlist))#sp.maximum(setpositionlist)+1#len(setpositionlist)
    outmat = addrowcol_matrix(preoutputmatrix,poslen-netDimNow+1,poslen-netDimNow+1)
    print outmat.shape,
    for pos in setpositionlist:
        if pos:
            outmat.itemset(pos,outmat.item(pos)+1)
    return outmat
      
def min2mout_mat(inputmatrix,preoutputmatrix,supposeMatrixDim):
    "IN:two matrix,one is input matrix n*4,another is the pre-output matrix n-1*n-1"
    "OUT:a matrix n*n"    
    tempmat = addrowcol_matrix(preoutputmatrix,1,1)
    netDimNow = preoutputmatrix.shape[0]#tempmat.shape[0]
            
    setposition = getPosition2Change(inputmatrix,preoutputmatrix,netDimNow,netDimAll=supposeMatrixDim,mode=2)#((netDim-1,choosed[0]),)#((netDim-1,0),(netDim-1,netDim-1),)
    outmat = setVauleofMatrix(preoutputmatrix,setposition,valuelist=None)
    
#     poslen = len(setposition)
#     outmat = addrowcol_matrix(preoutputmatrix,poslen-netDimNow+1,poslen-netDimNow+1)
#     print outmat.shape,
#     for pos in setposition:
#         if pos:
#             outmat.itemset(pos,outmat.item(pos)+1)
    return outmat

def input_init(personcnt):
    # print random.power([0.1,1,10,9],[10,4])
    # print random.pareto([0.1,1,10,9],[100,4],type=int)
    # print random.pareto(3,100)
    # print random.pareto(30,100)
    
    ma_fans = (random.pareto(0.4,personcnt))*10+1#(random.randint(30,100000000,personcnt))*10+1#
    cbm_frs = random.pareto(0.5,personcnt)*10+1
    inv_mention = random.power(1,personcnt)*10+1#randint(0,2,personcnt)#
    act_micorcnt = random.pareto(3,personcnt)*10+1
    # plothem(ma_fans,cbm_frs,inv_mention,act_micorcnt)
    return [ma_fans,cbm_frs,inv_mention,act_micorcnt] 

def getoutMatrix(personcnt,ma_fans,cbm_frs,inv_mention,act_micorcnt,worksfolder):
    preoutmat = mx('1')
    for i in range(1,personcnt+1):
#         print '-------------------',i
        input = zip(*[ma_fans[:i],cbm_frs[:i],inv_mention[:i],act_micorcnt[:i]])
        inputm = sp.asmatrix(input, long)
        min2mout_pat_mat = min2mout_mat(inputm,preoutmat,supposeMatrixDim=personcnt)
        preoutmat = min2mout_pat_mat
    return preoutmat
        
if __name__=="__main__":    
    personcnt = 200
    experimenTimes = 10
    worksfolder = "G:\\HFS\\WeiboData\\CMO\\"
    
    for i in range(experimenTimes):
        print '===============================================',i
        [ma_fans,cbm_frs,inv_mention,act_micorcnt] = input_init(personcnt)
        input = [ma_fans,cbm_frs,inv_mention,act_micorcnt]
    #     inputm = sp.asmatrix(input, long)
        
        mat = getoutMatrix(personcnt,ma_fans,cbm_frs,inv_mention,act_micorcnt,worksfolder+'figs\\')
        g = ig.Graph.Adjacency(mat.tolist())
        gt.drawgraph(g,giantornot=False,figpath=worksfolder+'figs\\'+str(i)+'.png',show=False)
    







