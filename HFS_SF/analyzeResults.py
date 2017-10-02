#encoding=utf8
import numpy
import matplotlib.pylab as plt
from tools import commontools as gtf
import csv
import time

import os
gt = gtf()

def mentionlistDis(mentionlistpath=r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\test\Mentioncntlist.txt',savefigpath=r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\test\Mentioncntlist.png'):
    a = ['3342670838100183.repost',0,1,1,2,3,0,0,0,0,0,0,0,0,0,0,0,0,0]
#     a = gt.csv2list_new(r'G:\HFS\WeiboData\CommonWeiboStatNet\Stat\Mentioncntlist.txt')#_test
    b = gt.csv2list_new(mentionlistpath)#_test
    lineall = []
#     for line in a:
#         lineNew = map(int,line[1:])
#         lineall.extend(lineNew)
#         lineNew = []
    # print  lineall 
    for line in b:
        lineNew2 = map(int,line[1:])
        lineall.extend(lineNew2)
        lineNew2 = []
         
    return gt.listDistribution(lineall, disfigdatafilepath=savefigpath, xlabel='Mentioned User Count', ylabel='Frequence',showfig=True,binsdivide=1) 

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

def exportTestData(inpath,outpath,filename):
    
    rlist = gt.txt2list(inpath)
    "userid, screen_name,followers_count, friends_count,createdtimetos, "
    rlistnow = zip(*rlist)
    
    userid = list(rlistnow[9])
    screen_name = list(rlistnow[10])
    followers_count = list(rlistnow[27])
    friends_count = list(rlistnow[28])
    createdtimetos = list(rlistnow[2])
    
    userid.reverse()
    screen_name.reverse()
    followers_count.reverse()
    friends_count.reverse()
    createdtimetos.reverse()  
      
    rlistnew = zip(userid, screen_name,followers_count, friends_count,createdtimetos)
#     print rlistnow[2],'\n',rlistnow[9],'\n',rlistnow[10],'\n',rlistnow[27],'\n',rlistnow[28],
    f = open(outpath,'w')
    rwriter = csv.writer(f)
    rwriter.writerow(['userid', ' screen_name', 'followers_count', ' friends_count', 'createdtimetos'])
#     metadata = findcoc(filename)
#     rwriter.writerow(metadata)
    for line in rlistnew:
#         one = (line[9],line[10],line[27],line[28],line[2])
#         print one
        rwriter.writerow(line)
    
def exportTestDataNomention(inpath,outpath):
    fw = open(outpath,'w')
    writer = csv.writer(fw)
    reader = csv.reader(file(inpath))
    for line in reader:
        if line[5] in  list('89'):
            writer.writerow(line)


def get_colfromReopst(repostcollist,colname):
    "idstr, created_at, createdtimetos, text, source, favorited, truncated, thumbnail_pic, geo,userid, screen_name, retweeted_status, reposts_count, comments_count,  attitudes_count, mlevel, visible, idstr, username, province, city, location, description, url, profile_image_url, profile_url, gender, followers_count, friends_count, statuses_count, favourites_count, created_at, created_attos, timein, following, allow_all_act_msg, geo_enabled, verified, verified_type, remark, statuslast, statuslasttos, allow_all_comment, avatar_large, verified_reason, follow_me, online_status, bi_followers_count, lang"

    metaline = ['idstr', 'created_at', 'createdtimetos', 'text', 'source', 'favorited', 'truncated', 'thumbnail_pic', 'geo', 'userid', 'screen_name', 'retweeted_status', 'reposts_count', 'comments_count', ' attitudes_count', 'mlevel', 'visible', 'idstr', 'username', 'province', 'city', 'location', 'description', 'url', 'profile_image_url', 'profile_url', 'gender', 'followers_count', 'friends_count', 'statuses_count', 'favourites_count', 'created_at', 'created_attos', 'timein', 'following', 'allow_all_act_msg', 'geo_enabled', 'verified', 'verified_type', 'remark', 'statuslast', 'statuslasttos', 'allow_all_comment', 'avatar_large', 'verified_reason', 'follow_me', 'online_status', 'bi_followers_count', 'lang']
    colindex = 0
    for it in metaline:
        if it==colname:
            break
        else:
            colindex += 1
    return  repostcollist[colindex]
                
         
def draw4inone(listfansall,listfriendsall,statusesall,fans_friends_ratio,outpathfolder):       
    fansa = gt.listDistribution(listfansall,disfigdatafilepath=outpathfolder+'\\Fans.figdata',xlabel='Fans',ylabel='Frequency',showfig=False,binsdivide=10)
    friendsa = gt.listDistribution(listfriendsall,disfigdatafilepath=outpathfolder+'\\Friends.figdata',xlabel='Friends',ylabel='Frequency',showfig=False,binsdivide=10)
    statusesa = gt.listDistribution(statusesall,disfigdatafilepath=outpathfolder+'\\MicroblogAmount.figdata',xlabel='MicroblogAmount',ylabel='Frequency',showfig=False,binsdivide=10)
    ffr = gt.listDistribution(fans_friends_ratio,disfigdatafilepath=outpathfolder+'\\Fans Friends Ratio.figdata',xlabel='Fans Friends Ratio',ylabel='Frequency',showfig=False,binsdivide=10)
    mention = mentionlistDis(mentionlistpath=r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\test\Mentioncntlist.txt',savefigpath=r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\test\Mentioncntlist.data') 
    
    plt.subplot(2,2,1)
    xlabel='(a) Mobilizing ability (Fans)'
    ylabel='Frenquency'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig1 = plt.loglog(fansa[0],fansa[1],marker='o',linestyle='')
    
    plt.subplot(2,2,2)
    xlabel='(b) Costs of be mobilized (Fans/Friends)'
    ylabel='Frenquency'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig2 = plt.loglog(ffr[0],ffr[1],marker='o',linestyle='')
    
    
    plt.subplot(2,2,3)
    xlabel='(c) Invited individuals unit (Mentioned  user)'
    ylabel='Frenquency'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig3 = plt.semilogy(mention[0],mention[1],marker='o',linestyle='')
    
    plt.subplot(2,2,4)
    xlabel='(d) Activity degree (Microblogs amount)'
    ylabel='Frenquency'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig4 = plt.loglog(statusesa[0],statusesa[1],marker='o',linestyle='')
    plt.show()
    plt.savefig(outpathfolder+'\\4in1.png')
    plt.close()

def drawfigure4in1():
    inpathfolder = r'G:\HFS\WeiboData\HFSWeibo'
    outpathfolder = r'G:\HFS\WeiboData\HFSWeiboStatNet'
    
    filecnt = 0
    listfansall = []
    listfriendsall = []
    statusesall = []
    fans_friends_ratio = []
    for file in os.listdir(inpathfolder):
        if os.path.splitext(file)[1]=='.repost':
            filecnt += 1
            
            repost = gt.txt2list(inpathfolder+'\\'+file)
            ziprepost = zip(*repost) 
            fans = get_colfromReopst(ziprepost,'followers_count')
            friends = get_colfromReopst(ziprepost,'friends_count')
            statuses = get_colfromReopst(ziprepost,'statuses_count')
            try:
                listfansall.extend(map(int,list(fans)))
            except:
                print 'ERROR:',file
                  
            try:
                listfriendsall.extend(map(int,list(friends)))
            except:
                print 'ERROR:',file
                 
            try:
                statusesall.extend(map(int,list(statuses)))
            except:
                print 'ERROR:',file
                 
            print filecnt,file
    try:            
        for fa,fr in zip(*[listfansall, listfriendsall]):
            fr = round(fr,2) if fr>0 else 1.0
            ratio = fa/fr
            fans_friends_ratio.append(ratio)
    except:
        pass 
    draw4inone(listfansall,listfriendsall,statusesall,fans_friends_ratio,outpathfolder)

############################################################################################
"analyzing svm"
def analyzeSvmResult(svmpredict,sucrange,failrange): 
    suclist = svmpredict[0:sucrange]
    failist = svmpredict[sucrange:failrange+1]
#     print suclist,failist
    
    errs=0
    rights=0
    errf=0
    rightf=0
    for it in suclist:
        if str(it)=='-1':
            errs+=1
        else:
            rights+=1
    for it in failist:
        if str(it)=='1':
            errf+=1
        else:
            rightf+=1
    return [errs,rights,errf,rightf]

from tools import commontools as gtf
import csv
import time

import os
gt = gtf()
import numpy 
def accurate(lista):
    return (lista[1]+lista[3])/float(numpy.sum(lista))

def recall(lista):
    return (lista[0]+lista[2])/float(numpy.sum(lista))

import os
def analyzeSVMall(svmpath,prefix,sindex,findex):
    result = []
    svmpath = r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\results'
    for file in os.listdir(svmpath):
        predictlist = []
        if os.path.splitext(file)[1]=='.predict' and str(file).startswith(prefix):
            path = svmpath+'\\'+file
            svmfile = open(path)
            for line in svmfile:
                predictlist.append(line.replace('\n',''))
            svma = analyzeSvmResult(predictlist,sindex,findex)
            result.append(accurate(svma))#,recall(svma)
    return result


##############################################findDistinctValue###########################################################
def findDistinctvalue(lista,metacolcnt=2,stdtimes=3):
    "Find the value which beyond 3 std in one list"
    "IN:list;the meta col to be ignore; the distinct value beyond how mang times of std"
    "OUT:the meta col[0] list of beyond"
    result = []
    avgstd = gt.averageLongtitude(lista)
    avg = avgstd[0]
    std = avgstd[1]
    j = 0
    for item in lista:
        j+=1
        i = metacolcnt
        discnt = 0
        for it in item[metacolcnt:]:
            if float(it)>avg[i]+stdtimes*std[i]:# or float(it)<avg[i]-stdtimes*std[i]:
                discnt+=1
            i+=1
        if discnt:
            print lista[j-1][0],discnt
            result.append(lista[j-1][0])
    return result 

def selectPercent(lista,percent,percentindex=1):
    result = []
    for item in lista:
        if str(item[percentindex])==str(percent):
            result.append(item)
    return result

def findDistValueall(filefolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\Net',percent=1.0,stdtimes=3):    
    "Find the value of all files in filefolder which beyond 3 std in one list"
#     filefolder = r'G:\HFS\WeiboData\HFSWeiboStatNet\Net'
    dismids = []
    dismidf = []
    for filef in os.listdir(filefolder):
        filepath = filefolder+'\\'+filef
        if os.path.isfile(filepath):
            print filepath
#             try:
            a = gt.csv2list_new(filepath)
            a = gt.normlistlist(gt.connectlist_sf(selectPercent(a,percent),0,sorted=False), metacolcount=2, sumormax='max')
            alla = gt.departlist(a,'1','-1',1)
            sa = alla[0]
            sb = alla[1]
            
#             dismids.extend(findDistinctvalue(sa,metacolcnt=2, stdtimes=stdtimes)) 
            dismidf.extend(findDistinctvalue(sb,metacolcnt=2, stdtimes=stdtimes)) 
            print len(gt.get_distinct_inlist(dismids)),len(gt.get_distinct_inlist(dismidf))
#             except Exception,e:
#                 print e
    dismids.extend(dismidf)
    return gt.get_distinct_inlist(dismids)
        
def removeDistFromSuc308(suc308path=r'G:\HFS\WeiboData\Statistics\meta_successed308.txt',sucoutpath=r'G:\HFS\WeiboData\Statistics\meta_successed308_noDis.txt'):
    "just remove distinct cases from suc and fail cases respectively , and create a new suc meta files"
    ""
    fw = open(sucoutpath,'w') 
    lista = gt.csv2list(suc308path, seprator='\t', start_index=0)
    dislist = findDistValueall(filefolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN',percent=1.0,stdtimes=3)
    for it in lista:
        if it[0] in dislist:
            lista.remove(it)
        else:
            stri=''
            for item in it:
                stri+=item+'\t'
            fw.write(str(stri)+'\n')
    fw.close()
    return lista
##############################################ENDfindDistinctValue###########################################################
#############################################################################

# inpathfolder = r'G:\HFS\WeiboData\HFSWeibo'
# outpathfolder = r'G:\HFS\WeiboData\HFSWeiboNodeData'
# for file in os.listdir(inpathfolder):
#     filename = os.path.splitext(file)[0]
#     filesuf = os.path.splitext(file)[1]
#     if filesuf=='.repost':
#         try:
#             exportTestData(inpathfolder+'\\'+file,outpathfolder+'\\'+file,filename)#\3343408888337055.repost')
#         except:
#             pass

    
# inpathfolder = r'G:\MyPapers\CMO\testData4Cui\nomention'
# outpathfolder = r'G:\MyPapers\CMO\testData4Cui\nomention'
# for file in os.listdir(inpathfolder):
#     exportTestDataNomention(inpathfolder+'\\'+file,outpathfolder+'\\'+file+'.nomention')#\3343408888337055.repost')

# drawfigure4in1()
draw4inone

result = []
perclist=[0.2,0.4,0.6,0.8,0.9,1.0]#
featurelist=['_.average_path_length','_.diameter','_.lenclustersmodeisstrong','_.lenclustersmodeisweak','_Fans.txt','_TimeSeries_20_percent.txt','']
for per in perclist:
    i = 0
    for fea in featurelist:
        i+=1
        tem = [per,fea,i]
        prefix = 'vd'+str(per)+str(fea)
        print prefix
        res = analyzeSVMall(svmpath=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\results',prefix=prefix,sindex=9,findex=18)
#         print res
#         for it in res:
        tem.extend(res)
        result.append(tem)
print result
import numpy
itr = []
for item in result:
    itr = item[0:2]
    itr.append(numpy.average(item[2:]))
    print itr
    
# vd1.0_.average_path_length_1.svm.ts
# vd1.0_.average_path_length_5.svm.ts
# vd1.0_.diameter_1.svm.tr
# vd1.0_.lenclustersmodeisstrong_2.svm.ts
# vd1.0_.lenclustersmodeisweak_2.svm.ts
# vd1.0_Fans.txt_10.svm.tr
# vd1.0_TimeSeries_20_percent.txt_1.svm.tr.scale
# vdall1.0_1.svm.tr

# analyzeSVMall(svmpath=r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\results',prefix='vd0.9')
# removeDistFromSuc308()


#         
# 
# 'followers_count', 'friends_count'  