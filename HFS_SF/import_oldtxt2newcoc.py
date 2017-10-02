#encoding=utf8
import sys
sys.path.append("..")
import re
import time
import os
from tools import commontools as gtf
import csv
 
gt=gtf()


def beretwittuser(line):
    return re.findall(u"//@+.*?[~ :  ：, .;'，\\。：\r？\n!//@\t@\\[\\]\'：]", line)#.replace(u'//',' '))
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
    cocpart = [(lines[10] or 'NM'),user,lines[0].replace(u'\"','').replace(u'\'','').replace(u'/',''),'2',beretwitype, secondRetwituserlen,mentioneduserlen]
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

def txt2user(lines,username,txtype='.repost',cocontentindex=3):
    "IN:text list line of orginal repost or comment; the orginal weibo's username; textype; and the text content index in the list"
    "The final coc is like this:'Shawn_Max,Beijing,3519173999095079,1823908385,1354519925.0,8,8,3519173332815242"
    " the format is 'source,target,mid,userid,time,retwitype,plzftype,statusid', in which the retwitype in {0189},0-be mentioned user line,1-mention/@ line,8-retwit,9-orginal weibo mention line;plzftype in {1289},1-comment,2-repost,8-retwitte,9-the orginal weibo @ line/mention line"
    result = []
    mention_user = []
    forword_user = []
    #新增将转发用户单线链接
    linestr = str(lines[cocontentindex])
    secondRetwituser=beretwittuser(linestr)
    mentioneduser = re.findall(u"@+.*?[\s :  ~：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr.replace(u'//@',' '))#user(linestr)
    for item in  secondRetwituser:
        item = item.replace('//@','').replace(u':','').replace(u'/','').strip()
        forword_user.append(item)
    for item in  mentioneduser:
        item = re.sub(u"[\s :  ~：,！ .;'，，\\。：\r？\n!//@ @\\[\\]\']",'',str(item)).strip()
        mention_user.append(item)
    result.append(mention_user)
    result.append(forword_user)
    return result
    
def txt2coc_old(lines,username,txtype='.repost',cocontentindex=3):
    "IN:text list line of orginal repost or comment; the orginal weibo's username; textype; and the text content index in the list"
    "The final coc is like this:'Shawn_Max,Beijing,3519173999095079,1823908385,1354519925.0,8,8,3519173332815242"
    " the format is 'source,target,mid,userid,time,retwitype,plzftype,statusid', in which the retwitype in {0189},0-be mentioned user line,1-mention/@ line,8-retwit,9-orginal weibo mention line;plzftype in {1289},1-comment,2-repost,8-retwitte,9-the orginal weibo @ line/mention line"
    cocotent = []
    #新增将转发用户单线链接
    linestr = str(lines[cocontentindex])
    forword_twice_index = linestr.find('//@')
    linestr_one = linestr[:forword_twice_index]
    linestr_twice = linestr[forword_twice_index:]
    
    secondRetwituser = []
    mentioneduser = []
    if forword_twice_index!=-1:            
        secondRetwituser=beretwittuser(linestr_twice) 
    if forword_twice_index!=0:    
        mentioneduser = re.findall(u"@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_one)#.replace(u'//@',' ').replace('@','@@'))#user(linestr)
    secondRetwituserlen=len(secondRetwituser)
    mentioneduserlen = len(mentioneduser)

    i=1
    retweetuserstr=[]
    "it should be just link 1 edge from retwitter to the first //@"
    if secondRetwituserlen>0:
        "if the line has //@,it means the user did not retwit the orginal weibo, he retwited the first //@ user's weibo, so it is just one line linked the user and the first //@ user"
        usera = lines[10]#username
        userb = secondRetwituser[0].replace('//@','').replace(u':','').replace(u'/','').strip()
        if usera and userb:        #.replace(u'\"','').replace(u'\'','').replace(u'/','')
            retweetuserstr=[usera,userb]
            if txtype == '.comment':           
                retweetuserline = writefromcommt_all2(lines,retweetuserstr,'0', secondRetwituserlen,mentioneduserlen)
            if txtype == '.repost':           
                retweetuserline = writefromrepost_all2(lines,retweetuserstr,'0', secondRetwituserlen,mentioneduserlen)
            cocotent.append(retweetuserline)
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
        retwitline = [lines[10].replace(u' ',''),username,lines[0].replace(u'\"','').replace(u'\'',''),'8','8', secondRetwituserlen,mentioneduserlen]
        retwitline.extend(lines)
        cocotent.append(retwitline)
    return cocotent

def txt2coc(lines,username,txtype='.repost',cocontentindex=3):
    "IN:text list line of orginal repost or comment; the orginal weibo's username; textype; and the text content index in the list"
    "The final coc is like this:'Shawn_Max,Beijing,3519173999095079,1823908385,1354519925.0,8,8,3519173332815242"
    " the format is 'source,target,mid,userid,time,retwitype,plzftype,statusid', in which the retwitype in {0189},0-be mentioned user line,1-mention/@ line,8-retwit,9-orginal weibo mention line;plzftype in {1289},1-comment,2-repost,8-retwitte,9-the orginal weibo @ line/mention line"
    cocotent = []
#     return [[lines[0].replace('\'',''),lines[11]]]
    #新增将转发用户单线链接
    linestr = str(lines[cocontentindex])
    forword_twice_index = linestr.find('//@')
    linestr_one = linestr[:forword_twice_index]
    linestr_twice = linestr[forword_twice_index:]
    
    secondRetwituser = []
    mentioneduser = []
    if forword_twice_index!=-1:            
        secondRetwituser=beretwittuser(linestr_twice) 
    if forword_twice_index!=0:    
        mentioneduser = re.findall(u"@+.*?[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_one)#.replace(u'//@',' ').replace('@','@@'))#user(linestr)
    secondRetwituserlen=len(secondRetwituser)
    mentioneduserlen = len(mentioneduser)

    i=1
    retweetuserstr=[]
    "it should be just link 1 edge from retwitter to the first //@"
    if secondRetwituserlen>0:
        "if the line has //@,it means the user did not retwit the orginal weibo, he retwited the first //@ user's weibo, so it is just one line linked the user and the first //@ user"
        usera = lines[10]#username
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
        retwitline = [lines[10].replace(u' ',''),username,lines[0].replace(u'\"','').replace(u'\'',''),'8','8', secondRetwituserlen,mentioneduserlen]
        retwitline.extend(lines)
        cocotent.append(retwitline)
    return cocotent

def findmetacoc(searchmid,metafilepath=r'G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc'):
#     r = gt.csv2list('G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc',',')
    r = csv.reader(file(metafilepath))
    metacocline = [searchmid]
    for line in r:
        line0 = str(line[0]).replace(u'\xef\xbb\xbf','').replace('#^#~#^#~##','')
        if str(line0)==str(searchmid):
            metacocline = [line[10].replace(u' ','')]
            for it in line[12:]:
                timeos = line[11]
                try:
                    timeos = float(time.mktime(time.strptime(timeos,'%Y-%m-%d %H:%M')))#'%a %b %d %H:%M:%S +0800 %Y'))
                except:
                    pass
                if it!='\N':
                    metacocline.extend([line[10].replace(u' ',''),it.replace(u'@','').replace(' ',''),line0,'9','9','-1','-1'])
                    metacocline.extend(['nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm','nm'])
                    break
            break
    return metacocline

def txt2coc_main(txtfile_basename,txt_list,cocfilepath):
    cocfile = open(cocfilepath,'w')#'G:\\HFS\\WeiboData\\HFSWeiboCOC\\'+fpf+'.coc'#'r'G:\HFS\WeiboData\Statistics\data4paper\Meta\startweibo_all_shougong.txt.metasuc.cc'
    cocwriter = csv.writer(cocfile)#,dialect="excel-tab"
    cocmeta = findmetacoc(txtfile_basename)
    cocwriter.writerow(cocmeta[1:])
#     for it in cocmeta[1:]:
#         cocwriter.writerow(it)
    for line in txt_list:
        cocontent = txt2coc(line,cocmeta[0])        
        for it in cocontent:
            cocwriter.writerow(it)
    cocfile.close()
    
def createCoc(repost_list,cocfolder,fpf):    
    ##IN:coc source text -repost_list, coc输出文件夹cocfolder，coc文件名fpf
    ##OUT:coc文件
    cocfilepath = cocfolder+fpf+'.coc'
#     if os.path.exists(cocfilepath):
#         print cocfilepath,'has existed'
#     else:
    txt2coc_main(fpf,repost_list,cocfilepath) 
    return cocfilepath 

def analyze_one(filepath,cocfolder):
    fp=filepath#r'G:\HFS\WeiboData\HFSWeibo\3508278808120380.repost'
    fpf = os.path.splitext(os.path.basename(fp))[0]
    repost = gt.txt2list(fp,bigfile=True)
    repost.reverse()#此处已修改为先把转发的直接按时间先后排序了，如果repost文件还有元数据行就可能存在问题， 可能引发连锁反应，后面的reverse可能都不需要了
     
    createCoc(repost,cocfolder,fpf)  
     

hfscasesfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\'#test\\
cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboNoCOC\\test\\'     
filecnt = 0
for filename in os.listdir(hfscasesfolder):
    filecnt+=1
    filepath = hfscasesfolder+filename
#     print filecnt,'============================================================================\n',filepath,' starting......'
 
    if os.path.splitext(filename)[1]=='.repost':
#         try:                
        analyze_one(filepath,cocfolder = cocfolder)
#         except:
#             print 'e'
# analyze_one(r'G:\HFS\WeiboData\HFSWeibo\test\toy.repost',cocfolder)