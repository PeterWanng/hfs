# -*- coding=utf-8 -*-
import os
import sys
sys.path.append('..\..')
from tools import commontools as gtf
from matplotlib import pyplot as plt

gt = gtf()

def draw(averageresult):
    x = x=[500,1000,5000,10000,50000,150000,250000]
#range(len(averageresult[0]))    
#     print len(averageresult[0]),len(averageresult[1]),'\n',len(x)

#     plt.title(fname)
    try:
        fig1=plt.loglog(x,averageresult[0][0:],marker='o')
        fig2=plt.loglog(x,averageresult[1][0:],marker='x')
#         plt.show()
    except:
        pass
        

    plt.xlabel('Period')
    plt.ylabel('Fans')
    plt.legend(('Success','Failture'),loc=2)
    plt.show()
#     figname=outfilepath+'.png'
    # pylab.savefig(figname)
    plt.close()    
    print 'over'

def draw_repostrange(): 
    x=[500,1000,5000,10000,50000,150000,250000]
    y1 = [217,729,2196,7462,22251,71380,235073,]
    y2 = [250,786,2208,5962,17892,80533,160104,]
    draw([y1,y2])

def getimeAddlist(timelist):
    try:
        timelist.remove('nm')
        timelist.remove('nan')
        timelist.remove('inf')
    except:
        pass
    
    timelist = map(float,timelist)
    timelist.sort()  
    
    lentimelist = len(timelist) 
    timeadded = []
    if lentimelist>0:
        for i in range(1,lentimelist):
            durationadded = timelist[i]-timelist[i-1]
            timeadded.append(durationadded)
    return timeadded
    
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


    
def draw_repostDis():
    statFolder = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Stat\\test\\'#
    hfscasesfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\'#test\\
    cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\test\\'
    
    filecnt = 0
    reposts_count = []     
    comments_count = []      
    attitudes_count = []
    
    fans_count = []
    bifans_count = []
    
    timelist_cnt = []

    for filename in os.listdir(hfscasesfolder):
        filepath = hfscasesfolder+filename
        print filecnt,'============================================================================\n',filepath,' starting......'
    
        if os.path.splitext(filename)[1]=='.repost':
            try:                
                filecnt+=1
                repost = gt.txt2list(filepath)
                repost = zip(*repost)
                reposts_count.extend(map(float,repost[12]))
                comments_count.extend(map(float,repost[13]))
                attitudes_count.extend(map(float,repost[14]))
                a,b=[],[]
                for ita in repost[27]:
                    try:
                        a.append(int(ita))
                    except:
                        pass
                for itb in repost[47]:
                    try:
                        b.append(int(itb))
                    except:
                        pass
                     
                fans_count.extend(a)
                bifans_count.extend(b)
                
                c = []
                for ita in repost[2]:
                    try:
                        c.append(float(ita))
                    except:
                        pass
                timelist_cnt.extend(c)
                
            except Exception,e:
                print e

    print gt.list_2_Distribution([fans_count,bifans_count],xlabels=['fans','bi-fans',],binseqdiv=5)#'reposts','comments',
    print gt.list_2_Distribution([reposts_count,comments_count],xlabels=['reposts','comments',])#'reposts','comments',binseqdiv=1
#     gt.listDistribution(comments_count)
#     gt.listDistribution(comments_count)


#     durationaddedavglist = getimeAddlist(timelist_cnt)
#     [x,y] = gt.list_2_Distribution([durationaddedavglist],xlabels=['duration',],binseqdiv=0)#'reposts','comments',
# 
#     plt.loglog(x,y)
#     plt.show()
    
def mentioncntdis():
#     lista = gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Stat\backup\Mentioncntlist.txt',passmetacol=1) 
#     b = []
#     for a in lista:
#          b.extend(a)          
#     [x,y] = list_2_Distribution([[b]],xlabels=['mention',],showfig=False)#'reposts','comments',
#     print 'x:',x
#     print 'y:',y

    x = [  0.1,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21., 22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,33.]
    y = [1780942,  154430,   31505,   17477,   12527,   12442,    8550, 5736,    3584,    2348,    1837,    1152,    1083,     714, 478,     538,     305,     235,     234,     188,     144,133,     110,      66,      27,      30,      12,      16,21,       7,       4,       1,       1,       1]

    
    plt.xlabel('Mentioned Users')
    plt.ylabel('Frequency')
    plt.semilogy(x,y,color='b',linestyle='-',marker='o',markersize=8, markerfacecolor='b',markeredgecolor='b',markeredgewidth=1.5,label='mentioned users', linewidth=1.5)

    plt.legend()
    plt.show()
    
# draw_repostDis()

draw_repostDis()
# mentioncntdis()

"-------------------------------------------------------------------"
# combinefile = ['.bifansum','.bifansumavg','.durationavglist','.durationlist','.echousercnt','.fanscnt','.fanscntavg','.friends_count','.friends_countavg','.mentioncnt','.mentioncntavg','.reposts_count','.reposts_countavg','meta_successed_530.txt',]
# folder = r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\statgiant\\'
# # connect_all(folder,0.1,combinefile,save=True, saveFolder=folder+'combine\\')
# for percent in [1.0]:#0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
#     gt.connect_allfileinlist_infolder(folder, percent,lista=combinefile, save=True, saveFolder=folder+'combine\\',passmetacol=21)
#     gt.connectlistinfolder(folder, percent)
"-------------------------------------------------------------------"
