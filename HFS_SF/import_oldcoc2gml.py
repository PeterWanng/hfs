#encoding=utf8
import sys
sys.path.append("..")
from tools import commontools as gtf
from tools import specialtools as gts
import time
import re
import csv
import os 

def coc2gml_hfstxt2gml(cocfilepath,coclineseprator='\t',gmlfilepath='IHaveNoName.gml',bigml=False):
    ##将图以节点对方式读入,含有其它边属性，输出gml格式
    "mynx the justkeepcore premeter is important for big coc which ignore the edges which have no @ and //@"
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
    dataformat = (('mid',int),('plzftype',str),('retwitype',str),('retwitcnt',int),('mentioncnt',int),('idstr',str),('created_at',str),('createdtimetos',str),('text',str),('source',str),('favorited',str),('truncated',str),('thumbnail_pic',str),('geo',str),('userid',str),('screen_name',str),('retweeted_status',str),('reposts_count',str),('comments_count',str),('attitudes_count',str),('mlevel',str),('visible',str),('idstr',str),('username',str),('province',str),('city',str),('location',str),('description',str),('url',str),('profile_image_url',str),('profile_url',str),('gender',str),('followers_count',str),('friends_count',str),('statuses_count',str),('favourites_count',str),('created_at',str),('created_attos',str),('timein',str),('following',str),('allow_all_act_msg',str),('geo_enabled',str),('verified',str),('verified_type',str),('remark',str),('statuslast',str),('statuslasttos',str),('allow_all_comment',str),('avatar_large',str),('verified_reason',str),('follow_me',str),('online_status',str),('bi_followers_count',str),('lang',str))
#     dataformat = (('mid',int),('plzftype',str),('retwitype',str),('retwitcnt',int),('mentioncnt',int),('idstr',str),('created_at',str),('createdtimetos',str),('favorited',str),('userid',str),('screen_name',str),('retweeted_status',str),('reposts_count',str),('comments_count',str),('attitudes_count',str),('idstr',str),('username',str),('province',str),('city',str),('location',str),('gender',str),('followers_count',str),('friends_count',str),('statuses_count',str),('favourites_count',str),('created_attos',str),('timein',str),('following',str),('allow_all_act_msg',str),('geo_enabled',str),('verified',str),('verified_type',str),('statuslasttos',str),('allow_all_comment',str),('verified_reason',str),('follow_me',str),('online_status',str),('bi_followers_count',str),('lang',str))
#     if bigml:
#         dataformat = (('mid',int),('plzftype',str),('retwitype',str),('retwitcnt',int),('mentioncnt',int),('idstr',str),('created_at',str),('createdtimetos',str),('favorited',str),('truncated',str),('thumbnail_pic',str),('geo',str),('userid',str),('screen_name',str),('retweeted_status',str),('reposts_count',str),('comments_count',str),('attitudes_count',str),('visible',str),('idstr',str),('username',str),('province',str),('city',str),('location',str),('gender',str),('followers_count',str),('friends_count',str),('statuses_count',str),('favourites_count',str),('created_at',str),('created_attos',str),('timein',str),('following',str),('allow_all_act_msg',str),('geo_enabled',str),('verified',str),('verified_type',str),('statuslast',str),('statuslasttos',str),('allow_all_comment',str),('verified_reason',str),('follow_me',str),('online_status',str),('bi_followers_count',str))
    G=mynx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiDiGraph(),data=dataformat,encoding='utf8',justkeepcore=False)#encoding='latin-1'
    print G.number_of_nodes()
    gmlfile = open(gmlfilepath,'w')
    nx.write_gml(G,gmlfile)
    gmlfile.close()
print u'\xe5\x9e\x83\xe5\x9c\xbe\xe7\xb4\xa0\xe8\xb4\xa8'
cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboNoCOC\\test\\'     
gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGMLNew\\test\\'     
filecnt = 0
for filename in os.listdir(cocfolder):
    filecnt+=1
    filepath = cocfolder+filename
    print filecnt,'============================================================================\n',filepath,' starting......'

    cocfilepath = filepath#r'G:\HFS\WeiboData\HFSWeiboNoCOC\3343740313561521.coc' 
    gmlfilepath = gmlfolder+filename+'.gml'#r'G:\HFS\WeiboData\HFSWeiboNoCOC\3343740313561521.gml' 
    if not os.path.exists(gmlfilepath) and os.path.splitext(filename)[1]=='.coc':
        coc2gml_hfstxt2gml(cocfilepath,',',gmlfilepath,bigml=True)