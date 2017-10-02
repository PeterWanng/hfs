#coding=utf8

import re
import time
import os
import sys
sys.path.append('..')
from tools import commontools as gtf
import csv
 
gt=gtf()


def beretwittuser(line):
#     return re.findall(u"//@+.*?[~ :  ：, .;'，\\。：\r？\n!//@\t@\\[\\]\'：]", line)#.replace(u'//',' '))
    return re.findall(u"@(.*?)[~ :  ：, .;'，\\。：\r？\n!//@\t@\\[\\]\'：]", line)#.replace(u'//',' '))
def writefromcommt(lines,user,beretwitype):
    return [(lines[8] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),lines[11],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'1',beretwitype,lines[9]]

def writefromrepost(lines,user,beretwitype):
    return [(lines[10] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),lines[9],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'2',beretwitype,lines[11]]

def writefromcommt2(lines,user,beretwitype):
    return [user[0],user[1],lines[0].replace(u'\"','').replace(u'\'',''),lines[11],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'1',beretwitype,lines[9]]

def writefromrepost2(lines,user,beretwitype):
    return [user[0],user[1],lines[0].replace(u'\"','').replace(u'\'',''),lines[9],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'2',beretwitype,lines[11]]

def writefromcommt_all(lines,user,beretwitype, secondRetwituserlen,mentioneduserlen):
    cocpart =  [(lines[8] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),'1',beretwitype, secondRetwituserlen,mentioneduserlen]
    cocpart.extend(lines)
    return cocpart 
   
def writefromrepost_all(lines,user,beretwitype, secondRetwituserlen,mentioneduserlen):
    cocpart = [(lines[8] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),'2',beretwitype, secondRetwituserlen,mentioneduserlen]
    cocpart.extend(lines)
    return cocpart 
   
def writefromcommt_all2(lines,user,beretwitype, secondRetwituserlen,mentioneduserlen):
    cocpart = [user[0],user[1],lines[0].replace(u'\"','').replace(u'\'',''),'1',beretwitype, secondRetwituserlen,mentioneduserlen]
    cocpart.extend(lines)
    return cocpart

def writefromrepost_all2(lines,user,beretwitype, secondRetwituserlen,mentioneduserlen):
    cocpart = [user[0],user[1],lines[0].replace(u'\"','').replace(u'\'',''),'2',beretwitype, secondRetwituserlen,mentioneduserlen]
    cocpart.extend(lines)
    return cocpart

def txt2user(lines,username,txtype='.repost',cocontentindex=16):
    "IN:text list line of orginal repost or comment; the orginal weibo's username; textype; and the text content index in the list"
    "The final coc is like this:'Shawn_Max,Beijing,3519173999095079,1823908385,1354519925.0,8,8,3519173332815242"
    " the format is 'source,target,mid,userid,time,retwitype,plzftype,statusid', in which the retwitype in {0189},0-be mentioned user line,1-mention/@ line,8-retwit,9-orginal weibo mention line;plzftype in {1289},1-comment,2-repost,8-retwitte,9-the orginal weibo @ line/mention line"
    result = []
    mention_user = []
    forword_user = []
    #新增将转发用户单线链接
    linestr = str(lines[cocontentindex])
    secondRetwituser=beretwittuser(linestr)
#     mentioneduser = re.findall(u"@+.*?[\s :  ~：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr.replace(u'//@',' '))#user(linestr)
    mentioneduser = re.findall(u"@(.*?)[\s :  ~：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr.replace(u'//@',' '))#user(linestr)
    for item in  secondRetwituser:
        item = item.replace('//@','').replace(u':','').replace(u'/','').strip()
        forword_user.append(item)
    for item in  mentioneduser:
        item = re.sub(u"[\s :  ~：,！ .;'，，\\。：\r？\n!//@ @\\[\\]\']",'',str(item)).strip()
        mention_user.append(item)
    result.append(mention_user)
    result.append(forword_user)
    return result
    
def txt2coc(lines,username,txtype='.repost',cocontentindex=16):
    "IN:text list line of orginal repost or comment; the orginal weibo's username; textype; and the text content index in the list"
    "The final coc is like this:'Shawn_Max,Beijing,3519173999095079,1823908385,1354519925.0,8,8,3519173332815242"
    " the format is 'source,target,mid,userid,time,retwitype,plzftype,statusid', in which the retwitype in {0189},0-be mentioned user line,1-mention/@ line,8-retwit,9-orginal weibo mention line;plzftype in {1289},1-comment,2-repost,8-retwitte,9-the orginal weibo @ line/mention line"
    cocotent = []
#     return [[lines[0].replace('\'',''),lines[11]]]
    #新增将转发用户单线链接
    linestr = str(lines[cocontentindex]).decode('utf-8')
    forword_twice_index = linestr.find('//@')
    linestr_one = linestr[:forword_twice_index]
    linestr_twice = linestr[forword_twice_index:]
    
    secondRetwituser = []
    mentioneduser = []
    if forword_twice_index!=-1:            
        secondRetwituser=beretwittuser(linestr_twice) 
    if forword_twice_index!=0:    
#         mentioneduser = re.findall(u"@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_one)#.replace(u'//@',' ').replace('@','@@'))#user(linestr)
        mentioneduser = re.findall(u"@(.*?)[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_one)#.replace(u'//@',' ').replace('@','@@'))#user(linestr)
    secondRetwituserlen=len(secondRetwituser)
    mentioneduserlen = len(mentioneduser)

    i=1
    retweetuserstr=[]
    "it should be just link 1 edge from retwitter to the first //@"
    if secondRetwituserlen>0:
        "if the line has //@,it means the user did not retwit the orginal weibo, he retwited the first //@ user's weibo, so it is just one line linked the user and the first //@ user"
        usera = lines[8]#username
        userblist = []
        for item in secondRetwituser:
            userblist.append(item.replace('//@','').replace(u':','').replace(u'/','').strip())
#         userb = secondRetwituser[0].replace('//@','').replace(u':','').replace(u'/','').strip()#just the first one 
        for userb in userblist:
            if usera and userb:        #.replace(u'\"','').replace(u'\'','').replace(u'/','')
                retweetuserstr=[usera,userb]
                if txtype == '.comment':           
                    retweetuserline = writefromcommt_all2(lines,retweetuserstr,'0', secondRetwituserlen,mentioneduserlen)
                if txtype == '.repost':           
                    retweetuserline = writefromrepost_all2(lines,retweetuserstr,'0', secondRetwituserlen,mentioneduserlen)
                cocotent.append(retweetuserline)
            usera = userb
        i+=1   
    
    if mentioneduserlen>0: 
        "if the user mentioned someone, all the mentioned user should have one line from the user"   
        if mentioneduser:
            for user in mentioneduser:
                user=re.sub(u"[\s :  。：,！ .;'，，\\。：\r？\n!//@ @\\[\\]\']",'',str(user))                
                if user.strip(): 
                    if txtype == '.comment':           
                        content = writefromcommt_all(lines,user,'1', secondRetwituserlen,mentioneduserlen)#原为lines[4]
                    if txtype == '.repost':           
                        content = writefromrepost_all(lines,user,'1', secondRetwituserlen,mentioneduserlen)
    #                     fw.write(content)#lines[3]+'\t'+user+'\t'+lines[0]+'\t'+lines[1]+'\t'+lines[2]+'\n'
                    cocotent.append(content)
                                             
    if secondRetwituserlen==0:
        " if there is no //@,the user retwit line should be added, or else, it has no link"         
        retwitline = [lines[8].replace(u' ',''),username,lines[0].replace(u'\"','').replace(u'\'',''),'8','8', secondRetwituserlen,mentioneduserlen]
        retwitline.extend(lines)
        cocotent.append(retwitline)
    return cocotent



def txt2coc_main(txtfile_basename,txt_list,cocfilepath):
    cocfile = open(cocfilepath,'w')#'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'+fpf+'.coc'#'r'G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc.cc'
    cocwriter = csv.writer(cocfile)#,dialect="excel-tab"
    cocmeta = txt_list[1]#findmetacoc(txtfile_basename)
#     cocwriter.writerow(cocmeta[1:])
#     for it in cocmeta[1:]:
#         cocwriter.writerow(it)
    for line in txt_list[1:]:
        if line[0]=='idstr':
            pass
        else:
            cocontent = txt2coc(line,cocmeta[8])        
            for it in cocontent:
                cocwriter.writerow(it)
    cocfile.close()
    
def createCoc(repost_list,cocfolder,fpf):    
    ##IN:coc source text -repost_list, coc输出文件夹cocfolder，coc文件名fpf
    ##OUT:coc文件
    cocfilepath = cocfolder+fpf+'.coc'
    if os.path.exists(cocfilepath):
        print cocfilepath,'has existed'
    else:
        txt2coc_main(fpf,repost_list,cocfilepath) 
    return cocfilepath 

def analyze_one(filepath,cocfolder):
    fp=filepath#r'G:\HFS\WeiboData\HFSWeibo\3508278808120380.repost'
    fpf = os.path.splitext(os.path.basename(fp))[0]
    repost = gt.csv2list_new(fp)
#     repost.reverse()#此处已修改为先把转发的直接按时间先后排序了，如果repost文件还有元数据行就可能存在问题， 可能引发连锁反应，后面的reverse可能都不需要了
    
    return createCoc(repost,cocfolder,fpf)  
     



def coc2gml_hfstxt2gml(cocfilepath,coclineseprator='\t',gmlfilepath='IHaveNoName.gml'):
    ##将图以节点对方式读入,含有其它边属性，输出gml格式
#     print time.clock()
    import networkx as nx
    import edgelist_fromnx as mynx
    #单边向DiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.Graph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #单边有向DiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.DiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #多边无向MultiGraph
    #             G=nx.read_edgelist(inpath, delimiter='\t', create_using=nx.MultiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    #多边有向MultiDiGraph
#     G=nx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiDiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
#     cocfilepath = csv.reader(cocfilepath)
#     dataformat = (('mid',int),('plzftype',str),('retwitype',str),('retwitcnt',int),('mentioncnt',int),('idstr',str),('created_at',str),('createdtimetos',str),('text',str),('source',str),('favorited',str),('truncated',str),('thumbnail_pic',str),('geo',str),('userid',str),('screen_name',str),('retweeted_status',str),('reposts_count',str),('comments_count',str),('attitudes_count',str),('mlevel',str),('visible',str),('idstr',str),('username',str),('province',str),('city',str),('location',str),('description',str),('url',str),('profile_image_url',str),('profile_url',str),('gender',str),('followers_count',str),('friends_count',str),('statuses_count',str),('favourites_count',str),('created_at',str),('created_attos',str),('timein',str),('following',str),('allow_all_act_msg',str),('geo_enabled',str),('verified',str),('verified_type',str),('remark',str),('statuslast',str),('statuslasttos',str),('allow_all_comment',str),('avatar_large',str),('verified_reason',str),('follow_me',str),('online_status',str),('bi_followers_count',str),('lang',str))
    dataformat = (('mid',int),('plzftype',str),('retwitype',str),('retwitcnt',int),('mentioncnt',int),( 'id',str),('created_at',str),('created_attos',str),('favorited',int),('truncated',int),('thumbnail_pic',str),('geo',str),('userid',str),('screen_name',str),('retweeted_status',str),('reposts_count',str),('comments_count',str),('attitudes_count',str),('mlevel',str),('visible',str),('source',str),('text',str), ('idstr',str),('username',str),('province',str),('city',str),('location',str) ,  ('gender',str),('followers_count',str),('friends_count',str),('statuses_count',str),('favourites_count',str),('created_attos',str),('timein',str),('following',int),('allow_all_act_msg',int),('geo_enabled',int),('verified',int),('verified_type',int),('remark',str),('statuslasttos',str),('allow_all_comment',int),('follow_me',str),('online_status',str),('bi_followers_count',str),('lang',str),('url',str),('profile_image_url',str),('profile_url',str),('avatar_large',str),('verified_reason',str),('description',str))    
    G=mynx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiDiGraph(),data=dataformat,encoding='utf8')#encoding='latin-1'
    print G.number_of_nodes()
    gmlfile = open(gmlfilepath,'w')
    nx.write_gml(G,gmlfile)
    gmlfile.close()

def analysisNet(graph):
    try:    
        g=graph
        vcount=g.vcount()   
        ecount=g.ecount() 
        degree=g.degree() 
        indegree=g.indegree() 
        outdegree=g.outdegree() 
        degreePowerLawFit=stcs.power_law_fit(degree,method='auto',return_alpha_only=False) 
        indegreePowerLawFit=stcs.power_law_fit(indegree, method='auto',return_alpha_only=False) 
        outdegreePowerLawFit=stcs.power_law_fit(outdegree,method='auto',return_alpha_only=False)
        assorDeg=g.assortativity(degree,directed= False) 
        assorDegD=g.assortativity(degree,directed= True) 
        assorInDeg=g.assortativity(indegree,directed= True)
        assorOutDeg=g.assortativity(outdegree,directed= True)
        
    #     assorDegF='1' if assorDeg>0 else '-1'  
    #     assorInDegF='1' if assorInDeg>0 else '-1'   
    #     assorOutDegF= '1' if assorOutDeg>0 else '-1'          
    #     print g.average_path_length()
        return [str(vcount),\
        str(ecount),\
        str(g.density()),\
        str(len(g.clusters(mode='weak'))),\
        str(len(g.clusters(mode='strong'))),\
        str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
        str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount()),\
        str((ecount*2)/float(vcount)),\
        str(g.transitivity_undirected(mode='0')) ,\
        str(g.average_path_length()),\
        str(g.diameter()),\
        str(assorDeg),\
        str(assorDegD),\
        str(assorInDeg),\
        str(assorOutDeg),\
    #     str(assorDegF),\
    #     str(assorInDegF),\
    #     str(assorOutDegF),\
        str(degreePowerLawFit.alpha),\
        str(degreePowerLawFit.xmin),\
        str(degreePowerLawFit.p),\
        str(degreePowerLawFit.L),\
        str(degreePowerLawFit.D),\
        str(indegreePowerLawFit.alpha),\
        str(indegreePowerLawFit.xmin),\
        str(indegreePowerLawFit.p),\
        str(indegreePowerLawFit.L),\
        str(indegreePowerLawFit.D),\
        str(outdegreePowerLawFit.alpha),\
        str(outdegreePowerLawFit.xmin),\
        str(outdegreePowerLawFit.p),\
        str(outdegreePowerLawFit.L),\
        str(outdegreePowerLawFit.D)]
    except:
        return []


def getCorePart_indeg(g,condition):
    g.delete_vertices(g.vs.select(_indegree_lt=condition))
    return g

def getCorePart_inoutdeg(g,condition):
#     g.vs["inoutdeg"] = g.indegree()*g.outdegree()
#     g.delete_vertices(g.vs.select(_inoutdeg_lt=condition))
    g.vs.select(_indegree_gt=0,_outdegree_gt=0)
    return g 
         
import igraph as ig
from igraph import clustering as clus

# filen = '3346646386115884'
hfscasesfolder = 'H:\\DataSet\\HotWeibo\\'#test\\
cocfolder = 'H:\\DataSet\\HotWeiboCOC\\'
gmlfolder = 'H:\\DataSet\\HotWeiboGML\\'
filecnt = 0
for filen in os.listdir(hfscasesfolder):
    if os.path.isfile(hfscasesfolder+filen) and os.path.splitext(filen)[-1]=='.repost':
        filecnt+=1
        print filecnt,'============================================================================\n',hfscasesfolder+filen,' starting......'
        filen = filen.split('.')[0]
        analyze_one(hfscasesfolder+filen+'.repost',cocfolder)#
        print 'coc done'
        gmlfilepath = gmlfolder+filen+'.gml'
        if os.path.exists(gmlfilepath):
            print gmlfilepath,'has existed'
        else:
            coc2gml_hfstxt2gml(cocfolder+filen+'.coc',gmlfilepath=gmlfilepath)
        print 'gml done'

for filen in os.listdir(gmlfolder):
    if os.path.isfile(gmlfolder+filen) and os.path.splitext(filen)[-1]=='.gml':
        print filen
#         g = ig.Graph.Read_GML(gmlfolder+filen+'.coc.gml')
        g = ig.Graph.Read_GML(gmlfolder+filen)
        # print analysisNet(g)
        gg = clus.VertexClustering.giant(g.clusters(mode='weak'))          
        ggcore = getCorePart_indeg(gg,1)
        ggcore2 = getCorePart_inoutdeg(gg,1)
         
        gsp=ig.Graph.spanning_tree(gg)
         
#         print g.eccentricity(vertices=15, mode='IN')
#         print g.vs(15)
        print g.vcount(),g.ecount(),\
                str(g.average_path_length()),\
                str(g.diameter()),\
                str(len(g.clusters(mode='weak'))),\
                str(clus.VertexClustering.giant(g.clusters(mode='weak')).vcount()),\
                str(clus.VertexClustering.giant(g.clusters(mode='weak')).ecount())
        print ggcore.vcount(),ggcore.ecount(),\
                str(ggcore.average_path_length()),\
                str(ggcore.diameter()),\
                str(len(ggcore.clusters(mode='weak'))),\
                str(clus.VertexClustering.giant(ggcore.clusters(mode='weak')).vcount()),\
                str(clus.VertexClustering.giant(ggcore.clusters(mode='weak')).ecount())
        print ggcore2.vcount(),ggcore2.ecount(),\
                str(ggcore2.average_path_length()),\
                str(ggcore2.diameter()),\
                str(len(ggcore2.clusters(mode='weak'))),\
                str(clus.VertexClustering.giant(ggcore2.clusters(mode='weak')).vcount()),\
                str(clus.VertexClustering.giant(ggcore.clusters(mode='weak')).ecount())
        print gsp.vcount(),gsp.ecount(),\
                str(gsp.average_path_length()),\
                str(gsp.diameter()),\
                str(len(gsp.clusters(mode='weak'))),\
                str(clus.VertexClustering.giant(gsp.clusters(mode='weak')).vcount()),\
                str(clus.VertexClustering.giant(ggcore.clusters(mode='weak')).ecount())
         
         
        figpath = str(gmlfolder+filen+'.png')
        try:
            # gt.drawgraph(g,giantornot=False)
            # gt.drawgraph(ggcore,giantornot=False)
            gt.drawgraph(g,giantornot=False,figpath=figpath)
    #         gt.drawgraph(gsp,giantornot=False)
        except Exception,E:
            print e
            