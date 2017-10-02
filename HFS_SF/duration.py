#encoding=utf-8
import sys
sys.path.append('..')
from tools import commontools as gtf
import numpy as np
from matplotlib import pyplot as plt
import operator
import os

gt = gtf()

import copy
def timeduration(mid,timeseriespath):
    timeserieslist = timeseriespath#gt.csv2list_new(csvfilepath=timeseriespath, delimitertype='excel', passmetacol=0, convertype=str)#read timeseries
    results = []
    item = timeserieslist
    if 1:#for item in timeserieslist:
        result = []        
        if item:
            itemnew = item#[1:]
            itemnew.sort()
#             duration = np.absolute(float(itemnew[-1])-float(itemnew[0]))
#             duration_avg = duration/len(itemnew)
#             result_one = [duration,duration_avg]
#             result.append(result_one)
            
            itbefore = item[0]
            for it in item:
                dua = it - itbefore
                itbefore = copy.deepcopy(it)
                
                result.append([mid,dua])
            results.extend(result)
    return results

def lifespan4sql(midpure,timelist,periodcnt,idlist,uidlist):
    "IN:timelist;periodcnt"
    "OUT:time duration of each period; average time duration of each person"
    durationlist,durationaddedlist,durationavglist,durationaddedavglist = [],[],[],[]
    wbids,mids,uids = [],[],[]
    
    timelist = map(float,timelist)
#     timelist.sort()
    [timelist,idlist,uidlist].sort(key=operator.itemgetter(1))
    lenyt = len(timelist)/float(periodcnt)
    ipast = 0
    duration,durationadded,durationavg,durationaddedavg = 0,0,0,0
    mid,uid = -1,-1
    if len(timelist)>0:
        for j in xrange(1,periodcnt+1):
            i = int(round(j*lenyt))
            i = i if i<len(timelist) else len(timelist)-1
            i = i if i>0 else 0
            
            try:
                duration = timelist[i]-timelist[0] 
                durationadded = timelist[i]-timelist[ipast]
                durationavg = duration/float(i+1)
                durationaddedavg = durationadded/float(lenyt) 
                
                wbid,mid,uid = midpure,idlist[i],uidlist[i]
            except:
                pass             
            ipast = i
            
            durationlist.append(duration)
            durationaddedlist.append(durationadded)
            durationavglist.append(durationavg)
            durationaddedavglist.append(durationaddedavg)
            
            wbids.append(wbid)
            mids.append(mid)
            uids.append(uid)
    else:
        for j in range(1,periodcnt+1):
            i = int(round(j*lenyt))
            i = i if i<len(timelist) else len(timelist)-1
            i = i if i>1 else 1            
            
            durationlist.append(duration)
            durationaddedlist.append(durationadded)
            durationavglist.append(durationavg)
            durationaddedavglist.append(durationaddedavg)  
                
            wbid,mid,uid = midpure,idlist[i],uidlist[i] 
            
            wbids.append(wbid)
            mids.append(mid)
            uids.append(uid)          
    return wbids,mids,uids,durationlist,durationaddedlist,durationavglist,durationaddedavglist

def lifespan4kmeans(timelist,periodcnt):
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
        
         
 
def duration_4dis(): 
    workfolder = "I:\\dataset\\HFS_XunRen\\HFSWeibo\\"#"I:\\dataset\\HFS_XunRen\\2014\\"#"I:\\dataset\\HFS_XunRen\\2013BC\\"
    durationfilep = 'I:\\dataset\\HFS_XunRen\\meta\\duration_hfs.csv'
    periodcnt = -1#100#-1  if all
    repostimelistfs = []
    results = []
    for fpp in os.listdir(workfolder):
        if os.path.splitext(fpp)[-1]=='.repost':
            mid = fpp.replace('.repost','')
            print mid
    #         for mid in ['2014070600_3375213297672784','2014070600_3414815978945201']:
            fp = workfolder + fpp
            midpure = mid.split('_')[-1]
            repost = gt.csv2list_new(fp)
            # gt.csv2list_new()
            repostlist = zip(*repost)
            repostimelist = list(repostlist[2])
            idstrlist = repostlist[0]
            useridlist = repostlist[7]
            repostimelistf,idlist,uidlist = [],[],[]
            for item,idstr,uid in zip(*(repostimelist,idstrlist,useridlist)):
                item_new = item
                try:
                    item_new = float(item)
                    repostimelistf.append(item_new) 
                    idlist.append(idstr)
                    uidlist.append(uid)
                except:
                    pass
                #repostimelistfs.append(repostimelistf)
            if len(repostimelistf)>1:
                if periodcnt == -1:
                    periodcnt = len(repostimelistf)
            #     duration = timeduration(midpure,repostimelistf) 
                durationz = zip(*(lifespan4sql(midpure,repostimelistf,periodcnt,idlist,uidlist)))
                #results.extend(durationz)
                gt.saveList(durationz,durationfilep,writype='a+')  

 
def duration_4kmeans(workfolder,durationfilep,periodcnt): 
#     workfolder = "I:\\dataset\\HFS_XunRen\\HFSWeibo\\"#"I:\\dataset\\HFS_XunRen\\2014\\"#"I:\\dataset\\HFS_XunRen\\2013BC\\"
#     durationfilep = 'I:\\dataset\\HFS_XunRen\\meta\\duration_hfs.csv'
#     periodcnt = -1#100#-1  if all
    repostimelistfs = []
    results = []
    for fpp in os.listdir(workfolder):
        if os.path.splitext(fpp)[-1]=='.repost':
            mid = fpp.replace('.repost','')
            print mid
    #         for mid in ['2014070600_3375213297672784','2014070600_3414815978945201']:
            fp = workfolder + fpp
            midpure = mid.split('_')[-1]
            repost = gt.csv2list_new(fp)
            if len(repost)<100:
                continue
            # gt.csv2list_new()
            repostlist = zip(*repost)
            repostimelist = list(repostlist[2])
            idstrlist = repostlist[0]
            useridlist = repostlist[7]
#             repostcntlist = repostlist[10]
#             commentcntlist = repostlist[11]
#             attitudecntlist = repostlist[12]
            repostimelistf,idlist,uidlist = [],[],[]
            for item,idstr,uid in zip(*(repostimelist,idstrlist,useridlist)):
                item_new = item
                try:
                    item_new = float(item)
                    repostimelistf.append(item_new) 
                    idlist.append(idstr)
                    uidlist.append(uid)
                except:
                    pass
                #repostimelistfs.append(repostimelistf)
            if len(repostimelistf)>1:
                if periodcnt == -1:
                    periodcnt = len(repostimelistf)
            #     duration = timeduration(midpure,repostimelistf) 
                durationz = lifespan4kmeans(repostimelistf,periodcnt)[0]
                durationz.insert(0,midpure)
                #print durationz
                #results.extend(durationz)
                gt.saveList([durationz],durationfilep,writype='a+')  


if __name__=='__main__':
#     duration_4dis()

 
    workfolder = "I:\\dataset\\HFS_XunRen\\2014\\"#"I:\\dataset\\HFS_XunRen\\HFSWeibo\\"#"I:\\dataset\\HFS_XunRen\\2013BC\\"#
    periodcnt = 50#100#-1  if all
    durationfilep = 'I:\\dataset\\HFS_XunRen\\meta\\duration_2014_'+str(periodcnt)+'_mt100.csv'
    duration_4kmeans(workfolder,durationfilep,periodcnt)    