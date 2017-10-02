# -*- coding=utf-8 -*-
import os
import sys
sys.path.append('..')
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
    
def draw_repostDis():
    statFolder = 'G:\\HFS\\WeiboData\\HFSWeiboStatNet\\Stat\\'#test\\
    hfscasesfolder = 'G:\\HFS\\WeiboData\\HFSWeibo\\'#test\\
    cocfolder = 'G:\\HFS\\WeiboData\\HFSWeiboCOC\\test\\'
    
    filecnt = 0
    reposts_count = []     
    comments_count = []      
    attitudes_count = []
    
    fans_count = []
    bifans_count = []

    for filename in os.listdir(hfscasesfolder):
        filepath = hfscasesfolder+filename
        print filecnt,'============================================================================\n',filepath,' starting......'
    
        if os.path.splitext(filename)[1]=='.repost':
            try:                
                filecnt+=1
                repost = gt.txt2list(filepath)
                repost = zip(*repost)
#                 reposts_count.extend(map(float,repost[12]))
#                 comments_count.extend(map(float,repost[13]))
#                 attitudes_count.extend(map(float,repost[14]))
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
            except Exception,e:
                print e

    gt.list_2_Distribution([fans_count,bifans_count],xlabels=['fans','bi-fans',],binseqdiv=5)#'reposts','comments',
#     gt.list_2_Distribution([reposts_count,comments_count],xlabels=['reposts','comments',])#'reposts','comments',binseqdiv=1
#     gt.listDistribution(comments_count)
#     gt.listDistribution(comments_count)


# draw_repostDis()

"-------------------------------------------------------------------"
# combinefile = ['.bifansum','.bifansumavg','.durationavglist','.durationlist','.echousercnt','.fanscnt','.fanscntavg','.friends_count','.friends_countavg','.mentioncnt','.mentioncntavg','.reposts_count','.reposts_countavg','meta_successed_530.txt',]
# folder = r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\statgiant\\'
# # connect_all(folder,0.1,combinefile,save=True, saveFolder=folder+'combine\\')
# for percent in [1.0]:#0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
#     gt.connect_allfileinlist_infolder(folder, percent,lista=combinefile, save=True, saveFolder=folder+'combine\\',passmetacol=21)
#     gt.connectlistinfolder(folder, percent)
"-------------------------------------------------------------------"
