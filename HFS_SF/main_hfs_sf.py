# -*- coding=utf-8 -*-
#! /usr/bin/python
__author__='wangtao'

import matplotlib.pyplot as plt
from tools import commontools as gtf
import csv
import re
import os
import time

    
gt=gtf()

def fansum(repostfanslist,periodcnt):
    lenyt = int(len(repostfanslist)/periodcnt)
    leny = lenyt if lenyt>1 else 1
    y = []
    fansum = 0
    for i in range(leny,len(repostfanslist),leny):            
        fansum += gt.listSum(repostfanslist[0:i])
        y.append(fansum)
    if len(y)<periodcnt:
        y = gt.repairY(y,periodcnt)
    
    return y

def echousercnt(lista):
    lena = len(lista)
    lenadist = len({}.fromkeys(lista).keys())
#     print lena-lenadist
    return lena-lenadist#(lena-lenadist)/float(lena)

def echouser(lista,periodcnt):
    lenyt = int(len(lista)/periodcnt)
    leny = lenyt if lenyt>1 else 1
    y = []
    echouserratio = 0.0
    for i in range(leny,len(lista),leny):            
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
        cocotent = []
        #新增将转发用户单线链接
        linestr = str(lines[cocontentindex])
        linestr2=beretwittuser(linestr)
        i=1
        listlen=len(linestr2)
        retweetuserstr=''
        while listlen>1 and i<listlen:
            usera = linestr2[i-1].replace('//@','').replace(':','').replace('/','').strip()
            userb = linestr2[i].replace('//@','').replace(':','').replace('/','').strip()
            if usera and userb:
                retweetuserstr=[usera,userb]
                if txtype == '.comment':           
                    retweetuserline = writefromcommt2(lines,retweetuserstr,'0')
                if txtype == '.repost':           
                    retweetuserline = writefromrepost2(lines,retweetuserstr,'0')
                cocotent.append(retweetuserline)
            i+=1
            
        #原来模样只需把下面replace(u'//@',' ')中的@去掉
        m = re.findall(u"@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr.replace(u'//@',' ').replace('@','@@'))#user(linestr)
        if m:
            for user in m:
                user=re.sub(u"[\s :  ：,！ .;'，，\\。：\r？\n!//@ @\\[\\]\']",'',str(user))                
                if user.strip(): 
                    if txtype == '.comment':           
                        content = writefromcommt(lines,user,'1')#原为lines[4]
                    if txtype == '.repost':           
                        content = writefromrepost(lines,user,'1')
#                     fw.write(content)#lines[3]+'\t'+user+'\t'+lines[0]+'\t'+lines[1]+'\t'+lines[2]+'\n'
                    cocotent.append(content)
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

def coc2gml(cocfilepath,coclineseprator='\t',gmlfilepath='IHaveNoName.gml'):
    ##将图以节点对方式读入,含有其它边属性，输出gml格式
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

def drawgml(gmlfilepath):
    try:
        import matplotlib.pyplot as plt
    except:
        raise
    import networkx as nx
#     G=nx.cycle_graph(12)
    G=nx.read_gml(gmlfilepath)
    pos=nx.spring_layout(G,iterations=200)
    nx.draw(G,pos,node_size=800,cmap=plt.cm.Blues,style='dotted')
#     plt.savefig("node_colormap.png") # save as png
    plt.show() # display)

    
###################################################################################################################################    
###################################################################################################################################    
###################################################################################################################################    
def analyze_one(filepath,cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\',gmlfolder = 'G:\\HFS\\WeiboData\\HFSWeiboGML\\'):
    fp=filepath#r'G:\HFS\WeiboData\HFSWeibo\3508278808120380.repost'
    fpf = os.path.splitext(os.path.basename(fp))[0]
    print fpf
    repost = gt.txt2list(fp)
    repost.reverse()#此处已修改为先把转发的直接按时间先后排序了，如果repost文件还有元数据行就可能存在问题， 可能引发连锁反应，后面的reverse可能都不需要了
    # repostlist = zip(*repost)
    # repostimelist = list(repostlist[2])
    # repostimelist.reverse()
    # 
    # repostuserlist = list(repostlist[0])
    # repostuserlist.reverse()
    # repostfanslist = list(repostlist[27])
    # repostfanslist.reverse()
    # 
    # 
    # periodcnt = 20
    # y1 = fansum(repostfanslist,periodcnt)#fans
    # y2 = echouser(repostuserlist,periodcnt)#echouser
    # print y1,y2
    
    ##############coc
    cocfilepath = cocfolder+fpf+'.coc'
    if os.path.exists(cocfilepath):
        print cocfilepath,'has existed'
    else:
        txt2coc_main(fpf,repost,cocfilepath)
    
    gmlfilepath = gmlfolder+fpf+'.gml'
    if os.path.exists(gmlfilepath):
        print gmlfilepath,'has existed'
    else:
        coc2gml(cocfilepath,',',gmlfilepath)
#     drawgml(gmlfilepath)
    ##############NetAnalysis##############################
    
    



if __name__ == '__main__':
    print 'All acting'
#     analyze_one('G:\\HFS\\WeiboData\\HFSWeibo\\3343744527348953.repost')#3508278808120380
    
    hfscasesfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\'
    for filename in os.listdir(hfscasesfolder):
        filepath = hfscasesfolder+filename
        if os.path.splitext(filename)[1]=='.repost':
            print filepath
            try:
                analyze_one(filepath)
                print filepath,' over'
            except Exception,e:
                print e
        




    























































###############################################################################
filepathin = r'G:\HFS\WeiboData\Statistics\data4paper\SVM\.average_path_length'

def connectallist():
    p1 = r'G:\HFS\WeiboData\Statistics\data4paper\SVM\.timeline.point_20.norm_100'
    list0 = list(csv.reader(file(p1)))#gt.csv2list(p1,seprator=',')
    lista = gt.connectlist_sf(list0,0)
    path = r'G:\HFS\WeiboData\Statistics\data4paper\SVM'
    
    lista = gt.connectlist(lista,gt.csv2list(path+'\\.average_path_length'),0,2,passcol=3)
    lista = gt.connectlist(lista,gt.csv2list(path+'\\.lenclustersmodeisstrong'),0,2,passcol=3)
    lista = gt.connectlist(lista,gt.csv2list(path+'\\.lenclustersmodeisweak'),0,2,passcol=3)
    lista = gt.connectlist(lista,gt.csv2list(path+'\\.density'),0,2,passcol=3)
    lista = gt.connectlist(lista,gt.csv2list(path+'\\.diameter'),0,2,passcol=3)
    lista = gt.connectlist(lista,gt.csv2list(path+'\\.vcount'),0,2,passcol=3)
    
    fw = open(path+'\\svmall.svm','w')
    writer = csv.writer(fw)
    for line in lista:
        writer.writerow(line)
    fw.close()
    
# connectallist()
# gt.csv2libsvmformat(r'G:\HFS\WeiboData\Statistics\data4paper\SVM\svmall.svm',csvdialect='excel')