#encoding=utf8
import igraph as ig
import sys
sys.path.append('../..')
from tools import commontools as gtf
import random
import os  
import numpy as np
gt = gtf()

from weibo_tools import deal_weibo
dwb = deal_weibo()

def getAttofNodesFromGraphES(es,attLabelList=None):
    res = []
    if not attLabelList:
        attLabelList = es.attributes()
        print attLabelList
    
    for lab in attLabelList:
        one = es.get_attribute_values(lab)
        res.append(one)
     
    useratts = res
    import operator
    useratts.sort(key=operator.itemgetter(0))
    
    userattsz = zip(*(res))
    print len(userattsz)
    userattsz = np.unique(userattsz)
    print len(userattsz)
#     for it in userattsz:
#         print it[0],it[1],it[2]
    return zip(*(userattsz))

def getDis(g,firstlabel):
    es=ig.EdgeSeq(g)
    vs = ig.VertexSeq(g)
    
    i = -1
    sourceid = 0
    for v in g.vs:
        i+=1
        if v['label']==firstlabel:
            sourceid = i 
            break
#     sor = vs.select(label_eq=firstlabel)
#     sourceid = int(sor.get_attribute_values('id')[0])
#     mids = es.get_attribute_values('mid');    #timelist.sort()
#     print sourceid
    dis = g.shortest_paths_dijkstra(source=None, target=sourceid, weights=None, mode='ALL') 

    
    dis = list(np.mat(dis).flat)
    return dis   

        
def pamChoice(gmlf):
    '''The supporters and invitees of initiator connect to the initiator. However, the supporters and invitees of participator pi have two choices: join the initiator directly, or join his or her master pi. Each individual k has its preference, someone may like join CMO directly, while others may be not. For the member of CMO, the ability of appealing new members is obviously in proportion to its mobilizing ability. Specifically speaking, the preference probability of each member m who are connected is:
    p_mk=β_k  〖MA〗_m/(∑_(k=0)^i?〖MA〗_k )                                (5)
    In which MA represents the mobilizing ability, and the β_k represents the preference coefficient of  m_k .'''
    graphAll = ig.Graph.Read_GML(gmlf)
    es=ig.EdgeSeq(graphAll)
    timelist = es.get_attribute_values('createdtimetos');    timelist.sort()
    sourceLabel = '醉联盟'
    print len(timelist)
    tp = timelist[-1]
    if 1:
#     for tp in timelist[60:]:
        g = graphAll.subgraph_edges(es.select(createdtimetos_le = tp, retwitype_ge='0'))
        print g.vcount()
        "['reposts_count', 'avatar_large', 'retwitcnt', 'text', 'mid', 'visible', 'statuslast', 'mentioncnt', 'description', 'city', 'verified', 'retweeted_status', 'thumbnail_pic', 'truncated', 'plzftype', 'follow_me', 'verified_reason', 'attitudes_count', 'location', 'followers_count', 'retwitype', 'created_attos', 'verified_type', 'username', 'favorited', 'statuses_count', 'statuslasttos', 'friends_count', 'online_status', 'allow_all_act_msg', 'profile_image_url', 'idstr', 'timein', 'allow_all_comment', 'geo_enabled', 'geo', 'createdtimetos', 'lang', 'bi_followers_count', 'remark', 'favourites_count', 'screen_name', 'url', 'province', 'created_at', 'mlevel', 'userid', 'comments_count', 'profile_url', 'gender', 'following']"
        username,followers_count,friends_count = getAttofNodesFromGraphES(es,['username','followers_count','friends_count'])
        followers_count = map(int,followers_count)
        dis = getDis(g,sourceLabel)
        
        vsNetAtt = zip(*(g.vs['label'],dis))
        nodeAtt = zip(*(username,followers_count,friends_count))
        vslabels = g.vs['label']
#         print vslabels
        atts = gt.connectlist(nodeAtt,vsNetAtt,passcol=0,sameposition_a=2)
        print len(atts)
#         vslabels = np.unique(vslabels)
        import operator
        vslabels.sort(key=operator.itemgetter(0))
        
        for v,un,fo,fr,di in zip(*(vslabels,username,followers_count,friends_count,dis)):
            print v,un,fo,fr,di#.get_attribute_values('label') 
        for v,di in zip(*(vslabels,dis)):
            print v,di
        print len(followers_count)
        print len(dis)
        er
        
        
#         print np.mat(dis).flat

        
    

    fansr = np.array(fansr)#np.mat(fansr).flat
    dis = np.array(dis)#
    
    train_x = [fansr,dis]
    train_y = g.shortest_paths_dijkstra(source=None, target=35, weights=None, mode='ALL')
    train_x, train_y = np.mat(train_x), np.mat(train_y).transpose()
    print len(train_x), len(train_y)

    from regression import logisticReg
    opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}

    lr = logisticReg()
    lr.start(train_x, train_y,opts)
    
    


gmlf=r'G:\HFS\WeiboData\HFSWeiboGMLNew\3513472585606907.coc.gml'
pamChoice(gmlf)




def start(no):
    workfolder = "H:\\DataSet\\HFS_XunRen_620\\2014\\"
    weibof=r'H:\DataSet\HFS_XunRen_620\2014\2014061812_3722035160951920.repost'
    cocf=r'G:\HFS\WeiboData\HFSWeiboGMLNew\3513472585606907.coc.gml'
    gmlf=r'G:\HFS\WeiboData\HFSWeiboGMLNew\3513472585606907.coc.gml'
    repost = gt.csv2list_new(fp)
    
    dwb.txt2coc_main(gmlf)
    
    pamChoice(gmlf)


