# -*- coding=utf-8 -*-
'''update in v2.0:
20130114
add one output (avgfansum) of fansum as a list
modified the echouser ratio again as ratio not absoluted value
add the function lifespan()
use fansum() to caculate any attribute list
each function return sum and avg'''

'''update in v3.0:
20130117
rewrite analysisStat() totally, add more stat attributes to stat,and changed the write mode
maybe a bug: the lifespan(),range(0,periodcnt) has been changed to range(1,periodcnt+1) to avoid always 0 for 1 period
add the mention and secondforward attributes'''

import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from tools import commontools as gtf
import csv
import re
import os
import time
 
gt=gtf()

def lifespan(timelist,periodcnt):
    "IN:timelist;periodcnt"
    "OUT:time duration of each period; average time duration of each person"
    durationlist,durationaddedlist,durationavglist,durationaddedavglist = [],[],[],[]
    try:
        timelist.remove('nm')
        timelist.remove('nan')
        timelist.remove('inf')
    except:
        pass
    
    timelist = map(float,timelist)
    timelist.sort()
    lenyt = len(timelist)/float(periodcnt)
    ipast = 0
    duration,durationadded,durationavg,durationaddedavg = 0,0,0,0
    if len(timelist)>0:
        for j in range(1,periodcnt+1):
            i = int(round(j*lenyt))
            i = i if i<len(timelist) else len(timelist)-1
            i = i if i>0 else 0
            
            try:
                duration = timelist[i]-timelist[0] 
                durationadded = timelist[i]-timelist[ipast]
                durationavg = duration/float(i+1)
                durationaddedavg = durationadded/float(lenyt) 
            except:
                pass             
            ipast = i
            
            durationlist.append(duration)
            durationaddedlist.append(durationadded)
            durationavglist.append(durationavg)
            durationaddedavglist.append(durationaddedavg)
    else:
        for j in range(1,periodcnt+1):
            i = int(round(j*lenyt))
            i = i if i<len(timelist) else len(timelist)-1
            i = i if i>1 else 1            
            
            durationlist.append(duration)
            durationaddedlist.append(durationadded)
            durationavglist.append(durationavg)
            durationaddedavglist.append(durationaddedavg)            
    return durationlist,durationaddedlist,durationavglist,durationaddedavglist
        
   
def fansum(repostfanslist,periodcnt):
    lenyt = len(repostfanslist)/float(periodcnt)
#     leny = lenyt if lenyt>1 else 1
    y = []
    yavg = []
    fansum = 0
    for j in range(1,periodcnt+1):
        i = int(round(j*lenyt))
        i = i if i<len(repostfanslist) else len(repostfanslist)-1
        i = i if i>1 else 1                
        fansum = gt.listSum(repostfanslist[0:i])
        
        fansumavg = fansum/float(i+1)
        
        y.append(fansum)
        yavg.append(fansumavg)
    if len(y)<periodcnt:
        y = gt.repairY(y,periodcnt)
        yavg = gt.repairY(yavg,periodcnt)
    
    return y,yavg

def echousercnt(lista):
    lena = len(lista)
    lena = 1 if lena==0 else lena
    lenadist = len({}.fromkeys(lista).keys())
#     print lena-lenadist
    return (lena-lenadist)/float(lena)#lena-lenadist#

def echouser(lista,periodcnt):
    lenyt = len(lista)/float(periodcnt)
#     leny = lenyt if lenyt>1 else 1
    y = []
    echouserratio = 0.0
    for j in range(1,periodcnt+1):
        i = int(round(j*lenyt))
        i = i if i<len(lista) else len(lista)-1 
        i = i if i>1 else 1       
        echouserratio = echousercnt(lista[0:i])
        y.append(echouserratio)
    if len(y)<periodcnt:
        y = gt.repairY(y,periodcnt)    
    return y



def beretwittuser(line):
    return re.findall(u"//@+.*?[ :  ：, .;'，\\。：\r？\n!//@\t@\\[\\]\'：]", line.replace('：',':'))#.replace(u'//',' '))
def writefromcommt(lines,user,beretwitype):
    return [(lines[8] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),lines[11],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'1',beretwitype,lines[9]]

def writefromrepost(lines,user,beretwitype):
    return [(lines[10] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),lines[9],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'2',beretwitype,lines[11]]

def writefromcommt2(lines,user,beretwitype):
    return [user[0],user[1],lines[0].replace(u'\"','').replace(u'\'',''),lines[11],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'1',beretwitype,lines[9]]

def writefromrepost2(lines,user,beretwitype):
    return [user[0],user[1],lines[0].replace(u'\"','').replace(u'\'',''),lines[9],float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'2',beretwitype,lines[11]]

def txt2coc(lines,username,txtype='.repost',cocontentindex=3):
    "IN:text list line of orginal repost or comment; the orginal weibo's username; textype; and the text content index in the list"
    "The final coc is like this:'Shawn_Max,Beijing,3519173999095079,1823908385,1354519925.0,8,8,3519173332815242"
    " the format is 'source,target,mid,userid,time,retwitype,plzftype,statusid', in which the retwitype in {0189},0-be mentioned user line,1-mention/@ line,8-retwit,9-orginal weibo mention line;plzftype in {1289},1-comment,2-repost,8-retwitte,9-the orginal weibo @ line/mention line"
    cocotent = []
    #新增将转发用户单线链接
    linestr = str(lines[cocontentindex])
    secondRetwituser=beretwittuser(linestr)
    mentioneduser = re.findall(u"@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr.replace(u'//@',' ').replace('@','@@'))#user(linestr)

    i=1
    secondRetwituserlen=len(secondRetwituser)
    retweetuserstr=[]
    "it should be just link 1 edge from retwitter to the first //@"
    if secondRetwituserlen>0:
        "if the line has //@,it means the user did not retwit the orginal weibo, he retwited the first //@ user's weibo, so it is just one line linked the user and the first //@ user"
        usera = lines[10]#username
        userb = secondRetwituser[0].replace('//@','').replace(u':','').replace(u'/','').strip()
        if usera and userb:        #.replace(u'\"','').replace(u'\'','').replace(u'/','')
            retweetuserstr=[usera,userb]
            if txtype == '.comment':           
                retweetuserline = writefromcommt2(lines,retweetuserstr,'0')
            if txtype == '.repost':           
                retweetuserline = writefromrepost2(lines,retweetuserstr,'0')
            cocotent.append(retweetuserline)
        i+=1   
    
    mentioneduserlen = len(mentioneduser)
    if mentioneduserlen>0: 
        "if the user mentioned someone, all the mentioned user should have one line from the user"   
        if mentioneduser:
            for user in mentioneduser:
                user=re.sub(u"[\s :  ：,！ .;'，，\\。：\r？\n!//@ @\\[\\]\']",'',str(user))                
                if user.strip(): 
                    if txtype == '.comment':           
                        content = writefromcommt(lines,user,'1')#原为lines[4]
                    if txtype == '.repost':           
                        content = writefromrepost(lines,user,'1')
    #                     fw.write(content)#lines[3]+'\t'+user+'\t'+lines[0]+'\t'+lines[1]+'\t'+lines[2]+'\n'
                    cocotent.append(content)
                                             
    if secondRetwituserlen==0:
        " if there is no //@,the user retwit line should be added, or else, it has no link"         
        retwitline = [lines[10].replace(u' ',''),username,lines[0].replace(u'\"','').replace(u'\'',''),str(lines[9]),float(time.mktime(time.strptime(lines[1],'%a %b %d %H:%M:%S +0800 %Y'))),'8','8',lines[11]]
        cocotent.append(retwitline)
    return cocotent

def findmetacoc(searchmid,metafilepath=r'G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc'):
#     r = gt.csv2list('G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc',',')
    r = csv.reader(file(metafilepath))
    metacocline = [searchmid]
    for line in r:
        line0 = str(line[0]).replace(u'\xef\xbb\xbf','')
        if str(line0)==str(searchmid):
            metacocline = [line[10].replace(u' ','')]
            for it in line[12:]:
                timeos = line[11]
                try:
                    timeos = float(time.mktime(time.strptime(timeos,'%Y-%m-%d %H:%M')))#'%a %b %d %H:%M:%S +0800 %Y'))
                except:
                    pass
                if it!='\N':
                    metacocline.append([line[10].replace(u' ',''),it.replace(u'@','').replace(' ',''),line0,line[9],timeos,'9','9',line0])
            break
    return metacocline

def txt2coc_main(txtfile_basename,txt_list,cocfilepath):
    cocfile = open(cocfilepath,'w')#'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'+fpf+'.coc'#'r'G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc.cc'
    cocwriter = csv.writer(cocfile)
    
    cocmeta = findmetacoc(txtfile_basename)
    for it in cocmeta[1:]:
        cocwriter.writerow(it)
    for line in txt_list:
        cocontent = txt2coc(line,cocmeta[0])        
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

def createMentionSeries(textlist,flag='@'):
    listr = []
    for line in textlist:
#         print line
        line = line.replace(r'//@','')
        s = gt.findall_instr(line,flag,start=0)
        listr.append(len(s))
#         print len(s),line
    return listr

def createNameFansFriends_List(repost,fdf) :
    listr = [['mid','userid','name','followers','friends']]
    repostlist = list(repost)
    for line in repostlist:
        s = [fdf,line[9],line[10],line[27],line[28]]
#         print s
        listr.append(s)
#         print len(s),line
    return listr
    

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
    G=nx.read_edgelist(cocfilepath, delimiter=coclineseprator, create_using=nx.MultiDiGraph(),data=(('mid',int),('userid',int),('time',str),('plzftype',str),('retwitype',str),('statusid',str)),encoding='latin-1')
    gmlfile = open(gmlfilepath,'w')
    nx.write_gml(G,gmlfile)
    gmlfile.close()
    
def createGml(cocfilepath,gmlfolder,cocfilename,keepold=True):     
    ##IN:gml source coc -cocfilepath, gmlfolder，gml文件名fpf
    ##OUT:gml文件
    gmlfilepath = gmlfolder+cocfilename+'.gml'
    if keepold and os.path.exists(gmlfilepath):
        print gmlfilepath,'has existed'
    else:
        coc2gml(cocfilepath,',',gmlfilepath)
#     drawgml(gmlfilepath) 
    return gmlfilepath

def getmentionuser_inline(line):
    "IN: str line of repost"
    "OUT: the mentioned user list and secondrepost chanels list"   
    
    linestr = str(line)
    forword_twice_index = linestr.find('//@')
    linestr_one = linestr[:forword_twice_index]
    linestr_twice = linestr[forword_twice_index:]
    
    secforwarduser = re.findall(u"//@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_twice)
    mentioneduser = re.findall(u"@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_one)
    
    return mentioneduser,secforwarduser

def getmentionuser(contentxt):
    "IN: a list of str line of repost"
    "OUT: the mentioned user list and secondrepost chanels list length, just length"
    i,j = 0,0
    lenc = len(contentxt)
    lenc = 1 if lenc<1 else lenc
    for line in contentxt:
        a,b = getmentionuser_inline(line)
        i+=len(a)
        j+=len(b)
    return i,i/float(lenc),j,j/float(lenc)

def analysisStat(repostlist,lengthNow,periodcnt):
    "given a repostlist with stat attributes, analyze its stat features"
    "IN:repostlist and the analyze range length,periodcnt"
    "OUT:stat features"
#     es = ig.EdgeSeq(g)
    "attributeA = es.get_attribute_values('attribute name')"
#     print es.attribute_names()
    '''reposts_count
    mentioncnt
    city
    verified
    retweeted_status
    attitudes_count
    location
    followers_count
    created_attos
    verified_type
    statuses_count
    statuslasttos
    friends_count
    idstr
    timein
    createdtimetos
    bi_followers_count
    favourites_count
    province
    userid
    comments_count
    gender'''
    
#     [3]
    lengthNow = int(lengthNow)
    timelist = list(repostlist[2])[:lengthNow]#es.get_attribute_values('createdtimetos')
    fansumlist = repostlist[27][:lengthNow]#es.get_attribute_values('followers_count')
    useridlist = repostlist[9][:lengthNow]#es.get_attribute_values('userid')
    mentioncntlist = list(repostlist[3])[:lengthNow]#es.get_attribute_values('mentioncnt')
    bifansumlist = repostlist[47][:lengthNow]#es.get_attribute_values('bi_followers_count')    
    friends_countlist = repostlist[28][:lengthNow]#es.get_attribute_values('friends_count')
    reposts_countlist = repostlist[12][:lengthNow]#es.get_attribute_values('reposts_count')
    
    
    periodcnt = 1
    fanscnt,fanscntavg = fansum(fansumlist,periodcnt)
    echousercnt = echouser(useridlist,periodcnt)
    durationlist,durationaddedlist,durationavglist,durationaddedavglist = lifespan(timelist,periodcnt)
    mentioncnt,mentioncntavg,secforwarduser,secforwarduseravg = getmentionuser(mentioncntlist) 
    bifansum,bifansumavg = fansum(bifansumlist,periodcnt)
    friends_count,friends_countavg = fansum(friends_countlist,periodcnt)
    reposts_count,reposts_countavg = fansum(reposts_countlist,periodcnt)#from 1 is to ensure exclude the orginal weibo

#     print [fanscnt[0],echousercnt[0],fanscntavg[0],durationlist[0],durationavglist[0],bifansum[0],bifansumavg[0],friends_count[0],friends_countavg[0],reposts_count[0],reposts_countavg[0]]#mentioncnt[0],mentioncntavg[0],
    return [fanscnt[0],echousercnt[0],fanscntavg[0],durationlist[0],durationavglist[0],mentioncnt,mentioncntavg,secforwarduser,secforwarduseravg,bifansum[0],bifansumavg[0],friends_count[0],friends_countavg[0],reposts_count[0],reposts_countavg[0]]#mentioncnt[0],mentioncntavg[0],
#     return [fanscnt,echousercnt,fanscntavg,durationlist,durationavglist,bifansum,bifansumavg,friends_count,friends_countavg,reposts_count,reposts_countavg]#mentioncnt,mentioncntavg,
  
def get_netlist(netAttribute,percentNetAttri,netlist):    
    percentNetAttri.extend(netAttribute)
    netlist.append(percentNetAttri) 
    return netlist 

def selecTime(timelist,periodcnt):
    timelistPeriodNow = []
    lenyt = len(timelist)/float(periodcnt)
    for j in range(1,periodcnt+1):
        i = int(round(j*lenyt))
        i = i if i>1 else 1
        i = i if i<len(timelist) else len(timelist)-1
        timelistPeriodNow.append(i)#(timelist[i])
    return timelistPeriodNow 

###################################################################################################################################    
###################################################################################################################################    
###################################################################################################################################    
def analyze_one(filepath,cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\',periodcnt=20,percentlist = [1.0],savecontent=False,createCocF=False,createTimelist=False,createMentionlist=False,createNameFansFriendsList=False,timeSeriesPath=''):
    fp=filepath#r'G:\HFS\WeiboData\HFSWeibo\3508278808120380.repost'
    fpf = os.path.splitext(os.path.basename(fp))[0]
    repost = gt.txt2list(fp)
    repost.reverse()#此处已修改为先把转发的直接按时间先后排序了，如果repost文件还有元数据行就可能存在问题， 可能引发连锁反应，后面的reverse可能都不需要了
    if createCocF:
        createCoc(repost,cocfolder,fpf)
    repostlist = zip(*repost)
    repostimelist = list(repostlist[2])
    
    Mentionlist = []
    if createMentionlist:
        Mentionlist = createMentionSeries(list(repostlist[3]))

    NameFansFriendsList = []    
    if createNameFansFriendsList:
        NameFansFriendsList = createNameFansFriends_List(repost,fpf)   

    
    statlist = []    
    statAttribute = []
    netstat_alllist = []   
    if savecontent:
        for percent in percentlist:
            lengthNow = int(round(len(repostimelist)*percent))
            lengthNow = lengthNow if lengthNow>1 else 1
            timelistPercentNow = repostimelist[:lengthNow]
            timelistPeriodNow = selecTime(timelistPercentNow,periodcnt)
            for timep in timelistPeriodNow:
                timep = str(timep)
                percentNetAttri = []
                percentNetAttri.append(fpf)
                percentNetAttri.append(percent)
                netstat_all = analysisStat(repostlist,timep,periodcnt)
#             for it in statOne:
#                 [it].insert(0,percent)                
#                 [it].insert(0,fpf)
#                 statAttribute.append(it) 
#                 netstat_alllist = get_netlist(statAttribute,percentNetAttri[0:],statlist)
#                 statAttribute = get_netlist(statOne,percentNetAttri[0:],statlist)
#                 netstat_alllist.append(statAttribute)
                netstat_alllist = get_netlist(netstat_all,percentNetAttri[0:],statlist)
                
#                 print netstat_alllist
    
    return [repostimelist,zip(*netstat_alllist),Mentionlist,NameFansFriendsList]

def generaTimeSeries(timeseries,periodcnt,percentlist,metacol=1):
    "create 20 periods and each percent data for time"
    result = []
    for item in timeseries:
        for percent in percentlist:
            lengthNow = int(round(len(item[0])*percent))
            lengthNow = lengthNow-metacol if lengthNow>1 else 1
            oneresult = []
            timeindexlist = gt.listdivide(item[metacol:lengthNow],periodcnt)
            for timeindex in timeindexlist:
                oneresult.append(float(item[timeindex])-float(item[metacol]))
            oneresult.insert(0,percent)
            oneresult.insert(0,item[0])
            result.append(oneresult) 
    return result   
                
            
def openStatAttributeFIles(netStatpath):
#     vcountfile = open(netStatpath+'.fansum','a+')
#     ecountfile = open(netStatpath+'.echouser','a+')
#     writer_vcountfile = csv.writer(vcountfile)
#     writer_ecountfile = csv.writer(ecountfile)
#     writerlist = [writer_vcountfile, writer_ecountfile]
    fanscntfile = open(netStatpath+'.fanscnt','a+')
    echousercntfile = open(netStatpath+'.echousercnt','a+')
    fanscntavgfile = open(netStatpath+'.fanscntavg','a+')
    durationlistfile = open(netStatpath+'.durationlist','a+')
    durationavglistfile = open(netStatpath+'.durationavglist','a+')
    mentioncntfile = open(netStatpath+'.mentioncnt','a+')
    mentioncntavgfile = open(netStatpath+'.mentioncntavg','a+')#mentioncnt,mentioncntavg,secforwarduser,secforwarduseravg
    secforwarduserfile = open(netStatpath+'.secforwarduser','a+')
    secforwarduseravgfile = open(netStatpath+'.secforwarduseravg','a+')
    bifansumfile = open(netStatpath+'.bifansum','a+')
    bifansumavgfile = open(netStatpath+'.bifansumavg','a+')
    friends_countfile = open(netStatpath+'.friends_count','a+')
    friends_countavgfile = open(netStatpath+'.friends_countavg','a+')
    reposts_countfile = open(netStatpath+'.reposts_count','a+')
    reposts_countavgfile = open(netStatpath+'.reposts_countavg','a+')

    writer_fanscntfile = csv.writer(fanscntfile)
    writer_echousercntfile = csv.writer(echousercntfile)
    writer_fanscntavgfile = csv.writer(fanscntavgfile)
    writer_durationlistfile = csv.writer(durationlistfile)
    writer_durationavglistfile = csv.writer(durationavglistfile)
    writer_mentioncntfile = csv.writer(mentioncntfile)
    writer_mentioncntavgfile = csv.writer(mentioncntavgfile)
    writer_secforwarduserfile = csv.writer(secforwarduserfile)
    writer_secforwarduseravgfile = csv.writer(secforwarduseravgfile)
    writer_bifansumfile = csv.writer(bifansumfile)
    writer_bifansumavgfile = csv.writer(bifansumavgfile)
    writer_friends_countfile = csv.writer(friends_countfile)
    writer_friends_countavgfile = csv.writer(friends_countavgfile)
    writer_reposts_countfile = csv.writer(reposts_countfile)
    writer_reposts_countavgfile = csv.writer(reposts_countavgfile)
    
    writerlist = [writer_fanscntfile,writer_echousercntfile,writer_fanscntavgfile,writer_durationlistfile,writer_durationavglistfile,writer_mentioncntfile,writer_mentioncntavgfile,writer_secforwarduserfile,writer_secforwarduseravgfile,writer_bifansumfile,writer_bifansumavgfile,writer_friends_countfile,writer_friends_countavgfile,writer_reposts_countfile,writer_reposts_countavgfile]#
    return writerlist
        
def deal_Attribute(netAttribute,writerlist,percent,periodcnt):
    a = zip(*netAttribute[0:2])
    b = netAttribute[2:]
    for metacol, attri in zip(*[a,b]):
        metacol = list(metacol)
        attri = list(attri)
        metacol.extend(attri)
#         print metacol
    
    netAttribute = list(netAttribute)
    namestr = netAttribute[0][0]
#     print 'netAttribute-',netAttribute
#                 netAttributeLine = [netAttribute[0][0],netAttribute[1][0]]
    for item,writer in zip(*[netAttribute[2:],writerlist]):
#                 print item,writer.__getattribute__(__name__)#__str__.__name__
#                 print item[0],item[1]
        itemNew = item#[2:]
        passedlinecnt = 0
        for i in range(len(percent)):
            percentstr = netAttribute[1][i*periodcnt]
            contentcol = list(itemNew[passedlinecnt:(passedlinecnt + periodcnt)])

            contentcol.insert(0,percentstr)
            contentcol.insert(0,namestr)
            passedlinecnt +=periodcnt

            writer.writerow(contentcol)  
#             print 'contentcol',contentcol                      
            
            contentcol = []     
    
def analyzeStat_main(savecontent=False,createCocF=False,createTimelist=False,createMentionlist=False,createNameFansFriendsList=False):
    #run the hfs statistics factors(timeline,fans,echouser etc.) and create coc file
    print time.clock(),'All beginning'
    
    statFolder = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Stat\\'#test\\
    hfscasesfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\test\\'#test\\
    cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\test\\'
    
    filecnt = 0
    percent = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]#
    periodcnt = 20

    if createTimelist:
        writer_timelist = csv.writer(file(statFolder+'TimeSeries.txt','a'))
    if createMentionlist:
        writer_mentionlist = csv.writer(file(statFolder+'Mentioncntlist.txt','a'))
    if savecontent:
        writerlist_stat = openStatAttributeFIles(statFolder)
#         writer_fans = csv.writer(file(statFolder+'Fans.txt','a'))
#         writer_echouser = csv.writer(file(statFolder+'Echouser.txt','a'))
    if createNameFansFriendsList:
        writer_NameFansFriendslist = csv.writer(file(statFolder+'NameFansFriends.txt','a'))
    ErrorFilelist = []
    

    for filename in os.listdir(hfscasesfolder):
        filepath = hfscasesfolder+filename
        print filecnt,'============================================================================\n',filepath,' starting......'

        if os.path.splitext(filename)[1]=='.repost':
#             try:                
                filecnt+=1
                statAttribute = analyze_one(filepath,cocfolder = cocfolder,periodcnt=periodcnt,percentlist=percent,savecontent=savecontent,createCocF=createCocF,createTimelist=createTimelist,createMentionlist=createMentionlist,createNameFansFriendsList=createNameFansFriendsList, timeSeriesPath='')
                
                if createTimelist:
                    statAttribute[0].insert(0,str(filename))
                    writer_timelist.writerow(statAttribute[0])
                    generaTimeSeries(statAttribute[0],periodcnt,percent,metacol=1)
                if createMentionlist:
                    statAttribute[2].insert(0,str(filename))
                    writer_mentionlist.writerow(statAttribute[2])
                if createNameFansFriendsList:
                    lista = statAttribute[3]
                    for line in lista:
                        writer_NameFansFriendslist.writerow(line)
#                 for item in statAttribute[1]:
#                     print item
                if savecontent:
                    deal_Attribute(statAttribute[1],writerlist_stat,percent,periodcnt)
                        
#                         item[0].insert(0,str(filename))
#                         writer_fans.writerow(item[0])
#                         
#                         item[1].insert(0,str(filename))
#                         writer_echouser.writerow(item[1])
#             except Exception,e:
#                 ErrorFilelist.append(filepath)
#                 print 'ERROR:',filepath,e        

    print filecnt, 'files Finished.\n Total time:', time.clock(),'\nERROR file:'
    for line in ErrorFilelist:
        print line
    print 'Error Files Count:',len(ErrorFilelist)

def createGmlAll(cocfolder,gmlfolder):
    for filename in os.listdir(cocfolder):
        infilepath = cocfolder+filename
        if os.path.splitext(filename)[1]=='.coc':
            filename = os.path.splitext(filename)[0]
            outfilepath = gmlfolder+filename+'.gml'
            print infilepath
            createGml(infilepath,gmlfolder,filename)

def wriTimePercnt():
    timelist = gt.csv2list_new('G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Stat\\TimeSeries.txt')
    timewriter = csv.writer(file('G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Stat\\TimeSeries_20_percent.txt','w'))
    times = generaTimeSeries(timelist,periodcnt=20,percentlist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],metacol=1)
    for line in times:
        timewriter.writerow(line)
        
if __name__ == '__main__':
    analyzeStat_main(savecontent=True,createCocF=False,createTimelist=False,createMentionlist=False,createNameFansFriendsList=False)
   
#     wriTimePercnt()
#     
#     cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'
#     gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGML\\'
#     createGmlAll(cocfolder,gmlfolder)
    print 'all over'