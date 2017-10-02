
#encoding=utf8
import igraph as ig



import numpy as np
# xfp = r'G:\MyPapers\HFS-Weibo-CSI\x.txt'
#
# x = np.genfromtxt(xfp,dtype=float)
# cumx = np.cumsum(x)
# np.savetxt(xfp+'.cumx',cumx)
# er
import time

usergml_old = "I:\\dataset\\HFS_XunRen\\User\\GML\\"
usergml_new = "I:\\dataset\\HFS_XunRen\\User\\GML2\\"
import os
import igraph as ig
# for fp in os.listdir(usergml_old):
# 	gmlfp = usergml_old+fp#r'I:\dataset\HFS_XunRen\User\GML\3699926871596010.gml'
# 	gmlfp2 = usergml_new+fp#r'I:\dataset\HFS_XunRen\User\GML2\3699926871596010.gml'
#
# 	f = open(gmlfp).read()
# 	# f = f.replace(r'graph [\r\n  node [',r'graph [\n  directed 1\n    node [')
# 	f = f.replace('graph [','graph [\n  directed 1')
# 	open(gmlfp2,'w').write(f)
# 	print gmlfp2,'fininshed'
# er
# for fp in os.listdir(usergml_new):
# 	gmlfp = usergml_old+fp#r'I:\dataset\HFS_XunRen\User\GML\3699926871596010.gml'
# 	gmlfp2 = usergml_new+fp#r'I:\dataset\HFS_XunRen\User\GML2\3699926871596010.gml'
# 	g = ig.Graph.Read_GML(gmlfp2)
# 	if not g.is_directed():
# 		print gmlfp2,g.is_directed()
#
# er

#
# import networkx as nx
# g = nx.read_gml(r'I:\dataset\HFS_XunRen\User\GML\3699928586574386.gml')
#
# print g.is_directed()
# g = g.to_directed()
# print g.degree()
# print g.in_degree()
# print g.out_degree()
# er
gmlfp2 = r'I:\dataset\HFS_XunRen\User\GML2\3699928586574386.gml'
g = ig.Graph.Read_GML(gmlfp2)
# g.directed=1
# g.to_directed(mutual=False)
# ig.Graph.write_gml(g,r'I:\dataset\HFS_XunRen\User\GML\3699928586574386.gml2')
# g = ig.Graph.Read_GML(r'I:\dataset\HFS_XunRen\User\GML\3699928586574386.gml2')
print g.is_directed()
print g.degree()
print g.indegree()
print g.outdegree()
er

import sys
sys.path.append('..\..')
from tools import graphtools as grtf
grt = grtf()


import os
import sys
sys.path.append('..\..')
from tools import commontools as gtf
gt = gtf()

print gt.getfilelistin_folder_new(path="I:\\dataset\\HFS_XunRen\\User\\GML\\",filetype='.gml')
er

def update_wbgml(wbg,udic):
    wbgnew = ig.Graph()
    for v in wbg:
        utype = v['utype']
        if utype in [3,4]:
            uname = v['name']
            udic.get(uname) 
    
    return wbgnew
    


"转换关系gml为用户名gml"
def uidgml2unamegml(uidg,udic):
    unameg = ig.Graph()
    for e in uidg.es():
        print e.attributes()
        print e['source'],e.target()
    for v in uidg.vs():
        print v.attributes()
        uid = v['label']
        print udic.get(str(uid))
        
    
    return unameg
    
    
import igraph as ig  
def one():  
    uidg = ig.read(r'N:\dataset\HFS_XunRen\User\GML\3698492410791306.gml')
    uidunamedic = gt.csv2list_new(r'N:\dataset\HFS_XunRen\2014\2014071107_3698492410791306.repost')
    uidunamedicz = zip(*(uidunamedic))
    uid,uname = uidunamedicz[7],uidunamedicz[8]
    #print uid
    udic = {}
    for id,name in zip(*(uid,uname)):
        udic[id] = name
    
    unamegml = uidgml2unamegml(uidg,udic)
    
one()
er 

uidunamedic = gt.csv2list_new(r'N:\dataset\HFS_XunRen\Meta\udic.txt')#不需要这个目前，只需要从weibo获取就足够
uidunamedicz = zip(*(uidunamedic))
udic = {}
for id,name in uidunamedicz:
    udic[id] = name
    
import cPickle
picklef = file(r'N:\dataset\HFS_XunRen\Meta\udic.pickle','w')
cPickle.dump(udic_names,picklef)
picklef.close()

userfolder = gt.getfilelistin_folder_new(path="N:\\dataset\\HFS_XunRen\\User\\GML\\",filetype='.gml')

for gml in userfolder:
    uidg = ig.read(r'O:\dataset\HFS_XunRen\User\GML\3698492410791306.gml')    
    unamegml = uidgml2unamegml(uidg,udic)
er

    
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
    G=nx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiDiGraph(),data=(('etype',str),),encoding='latin-1')
#     gmlfile = open(gmlfilepath,'w')
#     nx.write_gml(G,gmlfile)
#     gmlfile.close()
    nx.write_pajek(G,cocfilepath+'.eglist', encoding='latin-1')
  
cocfilepath = r'O:\dataset\HFS_XunRen\Example\3728374956619402sc.csv'
gmlfilepath = cocfilepath+'.gml'  
coc2gml(cocfilepath,',',gmlfilepath) 
g = ig.read(gmlfilepath)
print g.vcount(),g.ecount()
er 
        
workfolder = "G:\\HFS\\WeiboData\\HFSWeibo\\"
#fps = os.listdir(workfolder)
# a = gt.getfilepathlistin_folder_new(workfolder,filetype='.repost')
# gt.saveList([a],workfolder+'filelist.txt')

print gt.csv2list_new(workfolder+'filelist.txt')
er

import numpy as np
import scipy as sp
from cmo_regression import regression_cmo


attfp = r'G:\HFS\WeiboData\HFSWeibo\New\sql_out\cmo_m3.txt'
#atts = np.genfromtxt(attfp,dtype=float,delimiter=',')#,usecols=[12,19,20,21,22,23,26,29,38,39,40,41,42,46,49,50,51,52])
atts = gt.csv2list_new(attfp,convertype=float)
attsz = zip(*(atts))
print len(attsz)
# attsz = []
# for i in [12,19,20,21,22,23,26,29,38,39,40,41,42,46,49,50,51,52]:
#     attsz.append(map(float,attszz[i]))
    


#lentext,followers_count,friends_count,statuses_count,favourites_count,created_attos,verified_type,bi_followers_count,,followers_countB,friends_countB,statuses_countB,favourites_countB,created_attosB,verified_typeB,bi_followers_countB,verified_reasonB,len_descriptionB,tag
train_x = list(attsz[:-2])
train_y = gt.totarget(attsz[-1],0)
train_x = zip(*(train_x))
#train_y = zip(*(train_y))
regcmo = regression_cmo()
regcmo.start(train_x, train_y,predict_ornot=True)




er
"=========================="
gt.saveList(a,'N:\\dataset\\HFS_XunRen\\2013BC\\meta\\fp.txt',)

workfolder = "N:\\dataset\\HFS_XunRen\\2013BC\\"
fps = a#gt.csv2list_new(r'N:\\dataset\\HFS_XunRen\\2013BC\\meta\\fp.txt')
fpw = open(workfolder+"meta\\weibocontent.txt",'a+')
for fp in fps:
    fpath = workfolder+fp+'.repost'
    print fpath
    fp =  open(fpath)
    line1 = fp.readline()
    line2 = fp.readline()
    fpw.write(line2)
    fp.close()
fpw.close()
    
    
er

print 'hello world'
import MySQLdb
conn=MySQLdb.connect(host='localhost',user='root',passwd='wutong',db='hfs',port=3306,charset="utf8")

try:
    #conn=MySQLdb.connect(host='localhost',user='root',passwd='wutong',db='hfs',port=3306)
    conn=MySQLdb.connect(host='localhost',user='root',passwd='wutong',db='hfs',port=3306)
    cur=conn.cursor()
    res = cur.execute('select * from user where idstr=\'3393833908403703\'')
    print res
    cur.close()
    conn.close()
except MySQLdb.Error,e:
     print "Mysql Error %d: %s" % (e.args[0], e.args[1])
     
er

import os
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pylab

import sys
sys.path.append('..\..')
from tools import commontools as gtf

gt = gtf()

a = np.genfromtxt(r'G:\\Temp\\wx.txt',delimiter=',',dtype=int)
print a
az = zip(*(a))

plt.loglog(az[0],az[1],marker='o',linestyle='')
plt.show()

def uncompleteList2Mat(listlista,standardlista,standardlistag,delimeter_between_Tag_and_a=' ',):
    "IN:listlist,standard list with default value; standard tags list; delimeter_between_Tag_and_a"
    "OUT:completed listlista like matrix"
    "example:"
    '''listlista = [['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%']]
    standardlista = [0,0,0,0,0,0]
    standardlistag = ['T0','T1','T2','T3','T4','T5']
    print uncompleteList2Mat(listlista,standardlista,standardlistag)
    
    result will be:"[[0.99, 0, 0, 0, 0, 0.01], [0.01, 0, 0, 0, 0, 0.99]]"'''

    a = listlista#[['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%']]
    BB = []
    for aa in a:
#         aa.replace(r'[','')
#         aa = aa.split(',')
        B = standardlista[0:]#[0,0,0,0,0,0]
        T = standardlistag#['T0','T1','T2','T3','T4','T5']
        for i in aa:
            ilist = i.split(delimeter_between_Tag_and_a)
            j = 0
            for b,t in zip(B,T):
                if ilist[0]== t:
                    B[j] = float(ilist[1].replace('%',''))/100.0#ilist[1]
                j+=1    
        BB.append(B[0:])
#         print len(B),np.sum(B),aa,B[0:]
    return BB
        

def print124():
    import string
    a,b,c,d = 0,0,0,0
    for mode in [1,2,3,4,5,6,7]:
        m = mode*3**4*100
        for modefans in [0,1,2]:#3,,5
            a=modefans*3**3*100
            for modefr in [0,1,2]:#3,,5
                b=modefr*3**2*100
                for modeal in [0,1,2]:#3,,5
                    c=modeal*3*100
                    for modemen in [0,1,2]:#3,,5
                        d=modemen*100
                        print string.join(map(str,[mode,modefans+1,modefr+1,modeal+1,modemen+1]))

# print124()
# listlista = gt.csv2list(r'N:\HFS\WeiboData\CMO\Modefigs\fig\kmeans-6\Ashapedis_200_100_486.png_all.leglends',seprator=r'],[')#[[['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%'], ['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%'], ['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%']], [['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T1 1.0%', 'T5 98.0%'], ['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%'], ['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T1 1.0%', 'T5 98.0%']], [['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%'], ['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T1 3.0%', 'T5 96.0%'], ['T0 100.0%'], ['T0 99.0%', 'T5 1.0%'], ['T5 100.0%']]]
listlista = [[['T0 69.0%', 'T3 26.0%', 'T4 3.0%', 'T5 2.0%'],['T0 68.0%', 'T3 26.0%', 'T4 2.0%', 'T5 4.0%'],['T0 75.0%', 'T3 17.0%', 'T4 6.0%', 'T5 2.0%'],['T0 56.0%', 'T3 41.0%', 'T5 3.0%'],['T0 60.0%', 'T3 33.0%', 'T4 4.0%', 'T5 3.0%'],['T0 61.0%', 'T3 36.0%', 'T4 1.0%', 'T5 2.0%'],['T0 61.0%', 'T3 37.0%', 'T4 2.0%'],['T0 62.0%', 'T3 32.0%', 'T4 4.0%', 'T5 2.0%'],['T0 64.0%', 'T3 28.0%', 'T4 5.0%', 'T5 3.0%']],\

[['T0 66.0%', 'T3 28.0%', 'T4 4.0%', 'T5 2.0%'],['T0 63.0%', 'T3 31.0%', 'T4 3.0%', 'T5 3.0%'],['T0 61.0%', 'T3 33.0%', 'T4 3.0%', 'T5 3.0%'],['T0 64.0%', 'T3 31.0%', 'T4 2.0%', 'T5 3.0%'],['T0 61.0%', 'T3 31.0%', 'T4 4.0%', 'T5 4.0%'],['T0 67.0%', 'T3 28.0%', 'T4 3.0%', 'T5 2.0%'],['T0 67.0%', 'T3 27.0%', 'T4 5.0%', 'T5 1.0%'],['T0 60.0%', 'T3 34.0%', 'T4 5.0%', 'T5 1.0%'],['T0 62.0%', 'T3 31.0%', 'T4 3.0%', 'T5 4.0%']],\

[['T0 62.0%', 'T3 35.0%', 'T4 2.0%', 'T5 1.0%'],['T0 62.0%', 'T3 28.0%', 'T4 6.0%', 'T5 4.0%'],['T0 67.0%', 'T3 29.0%', 'T4 3.0%', 'T5 1.0%'],['T0 70.0%', 'T3 25.0%', 'T4 3.0%', 'T5 2.0%'],['T0 65.0%', 'T3 32.0%', 'T4 2.0%', 'T5 1.0%'],['T0 62.0%', 'T3 32.0%', 'T4 4.0%', 'T5 2.0%'],['T0 70.0%', 'T3 21.0%', 'T4 4.0%', 'T5 5.0%'],['T0 66.0%', 'T3 26.0%', 'T4 5.0%', 'T5 3.0%'],['T0 65.0%', 'T3 29.0%', 'T4 1.0%', 'T5 5.0%']],\

[['T0 85.0%', 'T3 13.0%', 'T4 2.0%'],['T0 84.0%', 'T3 13.0%', 'T4 2.0%', 'T5 1.0%'],['T0 89.0%', 'T3 10.0%', 'T4 1.0%'],['T0 84.0%', 'T3 16.0%'],['T0 87.0%', 'T3 12.0%', 'T4 1.0%'],['T0 82.0%', 'T3 17.0%', 'T4 1.0%'],['T0 92.0%', 'T3 8.0%'],['T0 86.0%', 'T3 12.0%', 'T4 1.0%', 'T5 1.0%'],['T0 83.0%', 'T3 16.0%', 'T4 1.0%']],\

[['T0 82.0%', 'T3 18.0%'],['T0 86.0%', 'T3 11.0%', 'T4 3.0%'],['T0 84.0%', 'T3 14.0%', 'T4 2.0%'],['T0 82.0%', 'T3 16.0%', 'T4 2.0%'],['T0 80.0%', 'T3 18.0%', 'T4 2.0%'],['T0 88.0%', 'T3 11.0%', 'T4 1.0%'],['T0 86.0%', 'T3 14.0%'],['T0 89.0%', 'T3 11.0%'],['T0 89.0%', 'T3 9.0%', 'T4 2.0%']],\

[['T0 86.0%', 'T3 10.0%', 'T4 3.0%', 'T5 1.0%'],['T0 87.0%', 'T3 12.0%', 'T4 1.0%'],['T0 85.0%', 'T3 14.0%', 'T5 1.0%'],['T0 84.0%', 'T3 14.0%', 'T4 2.0%'],['T0 88.0%', 'T3 10.0%', 'T4 2.0%'],['T0 87.0%', 'T3 13.0%'],['T0 87.0%', 'T3 10.0%', 'T4 3.0%'],['T0 82.0%', 'T3 17.0%', 'T4 1.0%'],['T0 87.0%', 'T3 11.0%', 'T4 2.0%']],\

[['T0 8.0%', 'T3 51.0%', 'T4 21.0%', 'T5 20.0%'],['T0 12.0%', 'T3 39.0%', 'T4 19.0%', 'T5 30.0%'],['T0 12.0%', 'T3 46.0%', 'T4 15.0%', 'T5 27.0%'],['T0 11.0%', 'T3 48.0%', 'T4 16.0%', 'T5 25.0%'],['T0 11.0%', 'T3 41.0%', 'T4 20.0%', 'T5 28.0%'],['T0 14.0%', 'T3 38.0%', 'T4 17.0%', 'T5 31.0%'],['T0 16.0%', 'T3 40.0%', 'T4 15.0%', 'T5 29.0%'],['T0 4.0%', 'T3 53.0%', 'T4 20.0%', 'T5 23.0%'],['T0 18.0%', 'T3 41.0%', 'T4 16.0%', 'T5 25.0%']],\

[['T0 17.0%', 'T3 32.0%', 'T4 20.0%', 'T5 31.0%'],['T0 12.0%', 'T3 48.0%', 'T4 16.0%', 'T5 24.0%'],['T0 9.0%', 'T3 43.0%', 'T4 14.0%', 'T5 34.0%'],['T0 17.0%', 'T3 46.0%', 'T4 15.0%', 'T5 22.0%'],['T0 13.0%', 'T3 43.0%', 'T4 14.0%', 'T5 30.0%'],['T0 16.0%', 'T3 51.0%', 'T4 15.0%', 'T5 18.0%'],['T0 10.0%', 'T3 40.0%', 'T4 21.0%', 'T5 29.0%'],['T0 11.0%', 'T3 42.0%', 'T4 14.0%', 'T5 33.0%'],['T0 15.0%', 'T3 39.0%', 'T4 15.0%', 'T5 31.0%']],\

[['T0 19.0%', 'T3 44.0%', 'T4 13.0%', 'T5 24.0%'],['T0 11.0%', 'T3 49.0%', 'T4 13.0%', 'T5 27.0%'],['T0 14.0%', 'T3 52.0%', 'T4 16.0%', 'T5 18.0%'],['T0 7.0%', 'T3 45.0%', 'T4 19.0%', 'T5 29.0%'],['T0 13.0%', 'T3 46.0%', 'T4 13.0%', 'T5 28.0%'],['T0 22.0%', 'T3 41.0%', 'T4 13.0%', 'T5 24.0%'],['T0 14.0%', 'T3 38.0%', 'T4 13.0%', 'T5 35.0%'],['T0 10.0%', 'T3 41.0%', 'T4 11.0%', 'T5 38.0%'],['T0 18.0%', 'T3 35.0%', 'T4 19.0%', 'T5 28.0%']],\

[['T0 17.0%', 'T3 43.0%', 'T4 12.0%', 'T5 28.0%'],['T0 13.0%', 'T3 46.0%', 'T4 25.0%', 'T5 16.0%'],['T0 16.0%', 'T3 37.0%', 'T4 18.0%', 'T5 29.0%'],['T0 15.0%', 'T3 46.0%', 'T4 11.0%', 'T5 28.0%'],['T0 12.0%', 'T3 44.0%', 'T4 9.0%', 'T5 35.0%'],['T0 13.0%', 'T3 39.0%', 'T4 21.0%', 'T5 27.0%'],['T0 13.0%', 'T3 36.0%', 'T4 22.0%', 'T5 29.0%'],['T0 12.0%', 'T3 31.0%', 'T4 21.0%', 'T5 36.0%'],['T0 15.0%', 'T3 32.0%', 'T4 19.0%', 'T5 34.0%']],\

[['T0 59.0%', 'T3 26.0%', 'T4 10.0%', 'T5 5.0%'],['T0 57.0%', 'T3 34.0%', 'T4 5.0%', 'T5 4.0%'],['T0 54.0%', 'T3 35.0%', 'T4 7.0%', 'T5 4.0%'],['T0 59.0%', 'T3 34.0%', 'T4 2.0%', 'T5 5.0%'],['T0 48.0%', 'T3 42.0%', 'T4 8.0%', 'T5 2.0%'],['T0 54.0%', 'T3 39.0%', 'T4 6.0%', 'T5 1.0%'],['T0 60.0%', 'T3 29.0%', 'T4 5.0%', 'T5 6.0%'],['T0 56.0%', 'T3 36.0%', 'T4 6.0%', 'T5 2.0%'],['T0 57.0%', 'T3 29.0%', 'T4 8.0%', 'T5 6.0%']],\

[['T0 46.0%', 'T3 35.0%', 'T4 10.0%', 'T5 9.0%'],['T0 34.0%', 'T3 46.0%', 'T4 13.0%', 'T5 7.0%'],['T0 43.0%', 'T3 39.0%', 'T4 9.0%', 'T5 9.0%'],['T0 38.0%', 'T3 48.0%', 'T4 12.0%', 'T5 2.0%'],['T0 38.0%', 'T3 46.0%', 'T4 12.0%', 'T5 4.0%'],['T0 42.0%', 'T3 42.0%', 'T4 6.0%', 'T5 10.0%'],['T0 46.0%', 'T3 33.0%', 'T4 9.0%', 'T5 12.0%'],['T0 45.0%', 'T3 41.0%', 'T4 8.0%', 'T5 6.0%'],['T0 34.0%', 'T3 47.0%', 'T4 5.0%', 'T5 14.0%']],\

[['T0 24.0%', 'T3 39.0%', 'T4 12.0%', 'T5 25.0%'],['T0 12.0%', 'T3 57.0%', 'T4 13.0%', 'T5 18.0%'],['T0 15.0%', 'T3 40.0%', 'T4 14.0%', 'T5 31.0%'],['T0 24.0%', 'T3 36.0%', 'T4 18.0%', 'T5 22.0%'],['T0 13.0%', 'T3 46.0%', 'T4 15.0%', 'T5 26.0%'],['T0 12.0%', 'T3 50.0%', 'T4 12.0%', 'T5 26.0%'],['T0 17.0%', 'T3 44.0%', 'T4 14.0%', 'T5 25.0%'],['T0 12.0%', 'T3 41.0%', 'T4 17.0%', 'T5 30.0%'],['T0 18.0%', 'T3 44.0%', 'T4 12.0%', 'T5 26.0%']],\

[['T0 63.0%', 'T3 33.0%', 'T4 2.0%', 'T5 2.0%'],['T0 67.0%', 'T3 30.0%', 'T4 3.0%'],['T0 72.0%', 'T3 24.0%', 'T4 4.0%'],['T0 67.0%', 'T3 30.0%', 'T4 3.0%'],['T0 74.0%', 'T3 19.0%', 'T4 7.0%'],['T0 68.0%', 'T3 27.0%', 'T4 5.0%'],['T0 68.0%', 'T3 28.0%', 'T4 4.0%'],['T0 71.0%', 'T3 25.0%', 'T4 4.0%'],['T0 75.0%', 'T3 23.0%', 'T4 1.0%', 'T5 1.0%']],\

[['T0 55.0%', 'T3 38.0%', 'T4 5.0%', 'T5 2.0%'],['T0 61.0%', 'T3 31.0%', 'T4 2.0%', 'T5 6.0%'],['T0 56.0%', 'T3 33.0%', 'T4 7.0%', 'T5 4.0%'],['T0 55.0%', 'T3 36.0%', 'T4 6.0%', 'T5 3.0%'],['T0 52.0%', 'T3 43.0%', 'T4 4.0%', 'T5 1.0%'],['T0 42.0%', 'T3 47.0%', 'T4 9.0%', 'T5 2.0%'],['T0 54.0%', 'T3 35.0%', 'T4 10.0%', 'T5 1.0%'],['T0 48.0%', 'T3 38.0%', 'T4 9.0%', 'T5 5.0%'],['T0 51.0%', 'T3 44.0%', 'T4 3.0%', 'T5 2.0%']],\

[['T0 2.0%', 'T3 42.0%', 'T4 20.0%', 'T5 36.0%'],['T0 5.0%', 'T3 39.0%', 'T4 15.0%', 'T5 41.0%'],['T0 5.0%', 'T3 40.0%', 'T4 14.0%', 'T5 41.0%'],['T0 1.0%', 'T3 33.0%', 'T4 23.0%', 'T5 43.0%'],['T0 4.0%', 'T3 38.0%', 'T4 21.0%', 'T5 37.0%'],['T0 3.0%', 'T3 33.0%', 'T4 20.0%', 'T5 44.0%'],['T0 5.0%', 'T3 32.0%', 'T4 22.0%', 'T5 41.0%'],['T0 2.0%', 'T3 36.0%', 'T4 21.0%', 'T5 41.0%'],['T0 3.0%', 'T3 37.0%', 'T4 19.0%', 'T5 41.0%']],\

[['T0 14.0%', 'T3 42.0%', 'T4 12.0%', 'T5 32.0%'],['T0 7.0%', 'T3 48.0%', 'T4 16.0%', 'T5 29.0%'],['T0 10.0%', 'T3 44.0%', 'T4 19.0%', 'T5 27.0%'],['T0 11.0%', 'T3 43.0%', 'T4 16.0%', 'T5 30.0%'],['T0 9.0%', 'T3 46.0%', 'T4 14.0%', 'T5 31.0%'],['T0 13.0%', 'T3 40.0%', 'T4 20.0%', 'T5 27.0%'],['T0 9.0%', 'T3 43.0%', 'T4 16.0%', 'T5 32.0%'],['T0 11.0%', 'T3 37.0%', 'T4 18.0%', 'T5 34.0%'],['T0 4.0%', 'T3 50.0%', 'T4 11.0%', 'T5 35.0%']],\

[['T0 9.0%', 'T3 42.0%', 'T4 22.0%', 'T5 27.0%'],['T0 9.0%', 'T3 38.0%', 'T4 16.0%', 'T5 37.0%'],['T0 8.0%', 'T3 40.0%', 'T4 13.0%', 'T5 39.0%'],['T0 8.0%', 'T3 34.0%', 'T4 20.0%', 'T5 38.0%'],['T0 13.0%', 'T3 45.0%', 'T4 18.0%', 'T5 24.0%'],['T0 10.0%', 'T3 42.0%', 'T4 14.0%', 'T5 34.0%'],['T0 11.0%', 'T3 42.0%', 'T4 9.0%', 'T5 38.0%'],['T0 9.0%', 'T3 31.0%', 'T4 17.0%', 'T5 43.0%'],['T0 11.0%', 'T3 41.0%', 'T4 18.0%', 'T5 30.0%']],\

[['T0 100.0%'],['T0 99.0%', 'T3 1.0%'],['T0 100.0%'],['T0 16.0%', 'T3 34.0%', 'T4 20.0%', 'T5 30.0%'],['T0 13.0%', 'T3 52.0%', 'T4 14.0%', 'T5 21.0%'],['T0 12.0%', 'T3 51.0%', 'T4 9.0%', 'T5 28.0%'],['T3 11.0%', 'T4 40.0%', 'T5 49.0%'],['T3 6.0%', 'T4 54.0%', 'T5 40.0%'],['T3 12.0%', 'T4 36.0%', 'T5 52.0%']],\

[['T0 98.0%', 'T3 1.0%', 'T4 1.0%'],['T0 99.0%', 'T3 1.0%'],['T0 99.0%', 'T3 1.0%'],['T0 53.0%', 'T3 35.0%', 'T4 8.0%', 'T5 4.0%'],['T0 55.0%', 'T3 32.0%', 'T4 12.0%', 'T5 1.0%'],['T0 49.0%', 'T3 33.0%', 'T4 9.0%', 'T5 9.0%'],['T0 1.0%', 'T3 9.0%', 'T4 56.0%', 'T5 34.0%'],['T0 1.0%', 'T3 7.0%', 'T4 45.0%', 'T5 47.0%'],['T0 1.0%', 'T3 6.0%', 'T4 50.0%', 'T5 43.0%']],\

[['T0 98.0%', 'T3 2.0%'],['T0 97.0%', 'T3 2.0%', 'T4 1.0%'],['T0 99.0%', 'T3 1.0%'],['T0 41.0%', 'T3 42.0%', 'T4 8.0%', 'T5 9.0%'],['T0 41.0%', 'T3 40.0%', 'T4 13.0%', 'T5 6.0%'],['T0 41.0%', 'T3 46.0%', 'T4 9.0%', 'T5 4.0%'],['T3 7.0%', 'T4 46.0%', 'T5 47.0%'],['T0 1.0%', 'T3 13.0%', 'T4 40.0%', 'T5 46.0%'],['T3 13.0%', 'T4 36.0%', 'T5 51.0%']],\

[['T0 99.0%', 'T3 1.0%'],['T0 100.0%'],['T0 100.0%'],['T0 18.0%', 'T3 43.0%', 'T4 17.0%', 'T5 22.0%'],['T0 15.0%', 'T3 44.0%', 'T4 11.0%', 'T5 30.0%'],['T0 12.0%', 'T3 45.0%', 'T4 17.0%', 'T5 26.0%'],['T3 8.0%', 'T4 41.0%', 'T5 51.0%'],['T3 6.0%', 'T4 37.0%', 'T5 57.0%'],['T3 7.0%', 'T4 41.0%', 'T5 52.0%']],\

[['T0 98.0%', 'T3 1.0%', 'T4 1.0%'],['T0 98.0%', 'T3 2.0%'],['T0 98.0%', 'T3 2.0%'],['T0 70.0%', 'T3 29.0%', 'T4 1.0%'],['T0 56.0%', 'T3 34.0%', 'T4 9.0%', 'T5 1.0%'],['T0 65.0%', 'T3 24.0%', 'T4 9.0%', 'T5 2.0%'],['T0 1.0%', 'T3 6.0%', 'T4 56.0%', 'T5 37.0%'],['T0 3.0%', 'T3 6.0%', 'T4 56.0%', 'T5 35.0%'],['T0 3.0%', 'T3 7.0%', 'T4 47.0%', 'T5 43.0%']],\

[['T0 100.0%'],['T0 99.0%', 'T3 1.0%'],['T0 98.0%', 'T3 2.0%'],['T0 55.0%', 'T3 36.0%', 'T4 6.0%', 'T5 3.0%'],['T0 52.0%', 'T3 33.0%', 'T4 10.0%', 'T5 5.0%'],['T0 40.0%', 'T3 45.0%', 'T4 10.0%', 'T5 5.0%'],['T0 1.0%', 'T3 10.0%', 'T4 48.0%', 'T5 41.0%'],['T0 1.0%', 'T3 8.0%', 'T4 48.0%', 'T5 43.0%'],['T3 8.0%', 'T4 57.0%', 'T5 35.0%']],\

[['T0 98.0%', 'T3 2.0%'],['T0 98.0%', 'T3 2.0%'],['T0 99.0%', 'T3 1.0%'],['T0 3.0%', 'T3 37.0%', 'T4 14.0%', 'T5 46.0%'],['T0 5.0%', 'T3 41.0%', 'T4 12.0%', 'T5 42.0%'],['T0 5.0%', 'T3 44.0%', 'T4 16.0%', 'T5 35.0%'],['T3 19.0%', 'T4 29.0%', 'T5 52.0%'],['T3 6.0%', 'T4 35.0%', 'T5 59.0%'],['T3 9.0%', 'T4 36.0%', 'T5 55.0%']],\

[['T0 98.0%', 'T3 1.0%', 'T4 1.0%'],['T0 98.0%', 'T3 1.0%', 'T4 1.0%'],['T0 99.0%', 'T3 1.0%'],['T0 8.0%', 'T3 42.0%', 'T4 19.0%', 'T5 31.0%'],['T0 10.0%', 'T3 53.0%', 'T4 13.0%', 'T5 24.0%'],['T0 11.0%', 'T3 38.0%', 'T4 13.0%', 'T5 38.0%'],['T3 9.0%', 'T4 29.0%', 'T5 62.0%'],['T3 8.0%', 'T4 33.0%', 'T5 59.0%'],['T2 1.0%', 'T3 7.0%', 'T4 40.0%', 'T5 52.0%']],\

[['T0 99.0%', 'T3 1.0%'],['T0 97.0%', 'T3 2.0%', 'T4 1.0%'],['T0 97.0%', 'T3 3.0%'],['T0 8.0%', 'T3 37.0%', 'T4 17.0%', 'T5 38.0%'],['T0 7.0%', 'T3 43.0%', 'T4 16.0%', 'T5 34.0%'],['T0 11.0%', 'T3 37.0%', 'T4 17.0%', 'T5 35.0%'],['T3 8.0%', 'T4 28.0%', 'T5 64.0%'],['T3 11.0%', 'T4 31.0%', 'T5 58.0%'],['T3 9.0%', 'T4 46.0%', 'T5 45.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 3.0%', 'T2 97.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 98.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 96.0%', 'T3 3.0%'],['T1 100.0%'],['T1 100.0%'],['T0 3.0%', 'T2 97.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 96.0%', 'T3 2.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 98.0%'],['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 98.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T2 99.0%', 'T3 1.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 99.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 99.0%', 'T3 1.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 99.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 99.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 97.0%', 'T3 2.0%'],['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 99.0%'],['T1 100.0%'],['T1 100.0%'],['T0 3.0%', 'T2 97.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 98.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 98.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 99.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 99.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 97.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 98.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T0 1.0%', 'T2 99.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 96.0%', 'T3 2.0%']],\

[['T1 100.0%'],['T1 100.0%'],['T2 99.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T0 2.0%', 'T2 97.0%', 'T3 1.0%'],['T1 100.0%'],['T1 100.0%'],['T2 100.0%']]]


standardlista = [0,0,0,0,0,0]
standardlistag = ['T0','T1','T2','T3','T4','T5']
matlist = []
# for item in listlista:
#     str(item).replace('[','')
#     matone = uncompleteList2Mat(item,standardlista,standardlistag,delimeter_between_Tag_and_a=' ')
#     print matone
# 
#     matlist.append(matone)
    

#     
# #     print matlist
#     mat2plot = zip(*matone)
#     mat2plotlen = len(mat2plot[0])
#     x = range(1,len(matone[0])+1)
#     w = 1.0/mat2plotlen
#     i = 0 
#     for it in mat2plot:    
#     #     plt.plot(x,it) 
#         j = mat2plotlen
#         for iit in it:
# #             print iit,mat2plotlen-j
#             plt.bar(x[i]-j*w, iit,width=w,color='b',align='center')
#             j-=1
#         i+=1
#     plt.xlim(0,6)
#     plt.ylim(0,1.2)
#     plt.show()   

#-------------------------------
# ax = plt.subplot(111)
# w = 0.3
# ax.bar(x-w, y,width=w,color='b',align='center')
# ax.bar(x, z,width=w,color='g',align='center')
# ax.bar(x+w, k,width=w,color='r',align='center')
# ax.xaxis_date()
# ax.autoscale(tight=True)
# 
# plt.show()


# a = [['T0 99.0%', 'T5 1.0%'], ['T0 1.0%', 'T5 99.0%']]
# BB = []
# for aa in a:
#     b1,b2,b3,b4,b5,b6 = 0,0,0,0,0,0
#     t1,t2,t3,t4,t5,t6 = 'T0','T1','T2','T3','T4','T5'
#     B = [b1,b2,b3,b4,b5,b6]
#     T = [t1,t2,t3,t4,t5,t6]
#     for i in aa:
#         ilist = i.split(' ')
#         j = 0
#         for b,t in zip(B,T):
#             if ilist[0]== t:
#                 B[j] = float(ilist[1].replace('%',''))/100.0
#             j+=1
# 
# #     print B
#     BB.append(B)
# print BB

# workfolder_fig = "D:\\HFS\\WeiboData\\CMO\\Modefigs\\fig\\"
# personcnt = 100
# experimenTimes = 100
# kmax = 25
# figname = r'D:\HFS\WeiboData\CMO\Modefigs\100_100.att.BIC'+str(kmax)+'.png'
# #x = loadattributes(workfolder_fig,personcnt,experimenTimes)
# x = np.genfromtxt(fname='D:\\HFS\\WeiboData\\CMO\\Modefigs\\'+str(personcnt)+'_'+str(experimenTimes)+'.att', dtype=float, comments=None, delimiter=' ', skip_header=1, skip_footer=0)
# xyz = zip(*x)
# xyzz = gt.normlistlist(listlista=xyz[1:],metacolcount=0,sumormax='max')#xyz[1:]#
# xy = zip(*xyzz)#[1:]
# z = zip(*xyz[:1])
# 
# 
# # # IRIS DATA
# # iris = sklearn.datasets.load_iris()
# # X = iris.data[:, :4]  # extract only the features
# # print X
# # #Xs = StandardScaler().fit_transform(X)
# # Y = iris.target
# 
# X = np.asarray(xy)
# ks = range(1,kmax)
# print 'data has parepared.'
# 
# # run 9 times kmeans and save each result in the KMeans object
# KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
# 
# # now run for each cluster the BIC computation
# BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]
# 
# plt.plot(ks,BIC,'r-o')
# print ks,BIC
# plt.title("cluster vs BIC")
# plt.xlabel("# clusters")
# plt.ylabel("# BIC")
# pylab.savefig(figname, dpi=300)
# 
# plt.show()
# gt.savefigdata(datafilepath=figname+'.data',x=ks,y=BIC,errorbarlist=None,title='cluster vs BIC',xlabel='# clusters',ylabel='# BIC',leglend='subleglabel')
# 
# plt.close()