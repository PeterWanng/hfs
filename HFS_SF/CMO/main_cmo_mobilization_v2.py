#encoding=utf8
'the main frame of mobilization model'
'basic model-training-model-simulating real case-comparing SR'
import sys
sys.path.append('../..')
from tools import commontools as gtf
from tools import graphtools
from weibo_tools import weibo2graph
import random
import os  
import numpy as np
gt = gtf()
weibo2g = weibo2graph()
grt = graphtools()
print 'hello'
from cmo_regression import regression_cmo
import igraph as ig

           
def selecTime(timelist,periodcnt):
    timelistPeriodNow = []
    lenyt = len(timelist)/float(periodcnt)
    print len(timelist)
    for j in range(1,periodcnt+1):
        i = int(round(j*lenyt))
        i = i if i>1 else 1
        i = i if i<len(timelist) else len(timelist)-1
        if len(timelist)<1:
            timelistPeriodNow.append('0') 
        else:
            timelistPeriodNow.append(timelist[i])
    return timelistPeriodNow 

def get_netlist(netAttribute, percentNetAttri, netlist):
    percentNetAttri.extend(netAttribute)
    netlist.append(percentNetAttri)
    return netlist

def lifespan(timelist,periodcnt):
    "IN:timelist;periodcnt"
    "OUT:time duration of each period; average time duration of each person"
    durationlist,durationaddedlist,durationavglist,durationaddedavglist = [],[],[],[]
#     try:
#         for it in timelist:
#             timelist.remove(u'createdtimetos')
#             timelist.remove('nm')
#             timelist.remove('nan')
#             timelist.remove('inf')
#     except:
#         pass
    
    #gt.convertype2float(timelist)
    timelist = gt.convertype2float(timelist)#map(float,timelist)
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

def analyzeNetStat(g):
    "mid,percent,fanscnt,echousercnt,fanscntavg,durationlist,durationavglist, bifansum,bifansumavg,friends_count,friends_countavg,reposts_count,reposts_countavg"
    "given a graph g with edges attributes, analyze its stat features"
    "IN:graph g"
    "OUT:stat features"
    es = ig.EdgeSeq(g)
    vs = ig.VertexSeq(g)
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
    

    timelist = es.get_attribute_values('createdtimetos')

    fansumlist = vs.get_attribute_values('followerscount')
    useridlist = es.get_attribute_values('userid')
    #mentioncntlist = es.get_attribute_values('username2')
    bifansumlist = vs.get_attribute_values('bifollowerscount')    
    friends_countlist = vs.get_attribute_values('friendscount')
    reposts_countlist = es.get_attribute_values('repostscount')
           
    fanscnt,fanscntavg = fansum(fansumlist,1)
    echousercnt = echouser(useridlist,1)
    durationlist,durationaddedlist,durationavglist,durationaddedavglist = lifespan(timelist,1)
    #mentioncnt,mentioncntavg = fansum(mentioncntlist,1)
    bifansum,bifansumavg = fansum(bifansumlist,1)
    friends_count,friends_countavg = fansum(friends_countlist,1)
    reposts_count,reposts_countavg = fansum(reposts_countlist,1)

#     print fanscnt,echousercnt,fanscntavg,durationlist,durationavglist,mentioncnt,mentioncntavg,bifansum,bifansumavg,friends_count,friends_countavg,reposts_count,reposts_countavg
    #return [fanscnt[0],echousercnt[0],fanscntavg[0],durationlist[0],durationavglist[0],mentioncnt[0],mentioncntavg[0],bifansum[0],bifansumavg[0],friends_count[0],friends_countavg[0],reposts_count[0],reposts_countavg[0]]
    return [fanscnt[0],echousercnt[0],fanscntavg[0],durationlist[0],durationavglist[0], bifansum[0],bifansumavg[0],friends_count[0],friends_countavg[0],reposts_count[0],reposts_countavg[0]]
    #return [durationlist[0],durationaddedlist[0],durationavglist[0],durationaddedavglist[0]]

def analyzeNet_time(workfolder_att,fname,g):
    if 1:
                
            "add time slice function"
            attsfp = workfolder_att+str(fname[0])+'.atts'
            gt.createFiles(attsfp)
            stat_attsfp_percent = workfolder_att+'percent_stat.att'
            gt.createFiles(stat_attsfp_percent)
            net_attsfp_percent = workfolder_att+'percent_net.att'
            gt.createFiles(net_attsfp_percent)
            periodcnt = 1
            percentlist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            netlist = []
            statlist = []
            
            es=ig.EdgeSeq(g)
            vs=ig.VertexSeq(g)
            timelist = es.get_attribute_values('createdtimetos')
            timelist.sort()
            for percent in percentlist:
                lengthNow = int(round(len(timelist)*percent))
                lengthNow = lengthNow if lengthNow>1 else 1
                timelistPercentNow = timelist[:lengthNow]
                timelistPeriodNow = selecTime(timelistPercentNow,periodcnt)
                for timep in timelistPeriodNow:
                    timep = str(timep)
                    percentNetAttri = []
                    percentNetAttri.append(fname[0])
                    percentNetAttri.append(percent)
                    percentNetAttri.append(timep)
                    
                    subg = g.subgraph_edges(es.select(createdtimetos_le = timep))

                    #grt.analyzeNetNodes(g,workfolder_att,str(fname[0]))
                    netAttribute_all = grt.analysisNet(subg)
                    netlist_all = get_netlist(netAttribute_all,percentNetAttri[0:],netlist)
                    gt.saveList(netlist_all,net_attsfp_percent, writype='a+')    
                    netlist = []
                    
                    
                    netstat_all = analyzeNetStat(subg)
                    netstat_alllist = get_netlist(netstat_all,percentNetAttri[0:],statlist) 
                    gt.saveList(netstat_alllist,stat_attsfp_percent, writype='a+')    
                    statlist = []   


def start_create_gml(workfolder,fpspath=None):
    workfolder_gml = gt.createFolder(workfolder+"GML\\")
    workfolder_att = gt.createFolder(workfolder+"ATT\\")
    workfolder_att_simp = gt.createFolder(workfolder+"ATTSimp\\")
    workfolder_att_core = gt.createFolder(workfolder+"ATTCore\\")
    i = 0
    fps = os.listdir(workfolder)
    if fpspath:
        fps = gt.csv2list_new(fpspath)
    for fp in fps:#os.listdir(workfolder):
        fp = os.path.basename(fp)
#         if i>1:
#             break
        fname = fp.split('_')[-1]
        fname = os.path.splitext(fp)
        
        i+=1
        if fname[-1]=='.repost':# or fname[-1]=='.comment':
            wbfilep = workfolder+fp
            
            gmlfp = workfolder_gml+fname[0]+'.gml'
            print i,wbfilep
            try:
                #g = ig.Graph(directed=True)
                if not os.path.exists(gmlfp):
                    weibo2g.start(wbfilep,gmlfp,addcommentlist=True)#ig.read(gmlfp)#
            except:
                pass
            if not os.path.exists(workfolder_att+str(fname[0])+'.atts'):
                g = ig.read(gmlfp)

                # print g.vcount(),gg.vcount(),g.ecount(),gg.ecount(),
                #print g.is_connected(mode='WEAK'),gg.is_connected(mode='STRONG')
                #print len(g.clusters(mode='STRONG')),len(g.clusters(mode='WEAK'))

                grt.analyzeNetNodes_New(g,workfolder_att,str(fname[0]),keepold=False)
                analyzeNet_time(workfolder_att,fname,g)
                
                gg = grt.getCorePart(g,1)
                grt.analyzeNetNodes_New(gg,workfolder_att_core,str(fname[0]),keepold=False)
                analyzeNet_time(workfolder_att_core,fname,gg)                    


  
            
            

workfolder ="N:\\dataset\\HFS_XunRen\\2013BC\\"#"G:\\HFS\\WeiboData\\HFSWeibo\\"#"N:\\dataset\\HFS_XunRen_620\\2014\\"# "H:\\DataSet\\HFS7\\"
fpspath = workfolder+"meta\\success2.0_realXR.txt"#_test#None#_2fix
start_create_gml(workfolder,fpspath)

er  
def train():
    mentimes_tag,mentimes_after_tag,degree_tag, = 1,1,1,
    predict_ornot = False
    

    print "predict_ornot = ",predict_ornot           
    attpath = r'G:\HFS\WeiboData\HFSWeibo\ATT\test4lr_all.txt'#_small#_all
    if not os.path.exists(attpath):
        hfsmallfn = gt.getfilelistin_folder_new(path="G:\\HFS\\WeiboData\\HFSWeibo\\ATT\\",filetype='.attvs',filesize_min=0,filesize_max=None)
#         workfolder = "G:\\HFS\\WeiboData\\HFSWeibo\\"
#         hfsmallfn = []
#         fileappendix = '.repost'
#         hfsmallfn = gt.getfilelistin_folder_new(path=workfolder,filetype=fileappendix,filesize_min=0,filesize_max=500000)

                    
        print len(hfsmallfn),time.sleep(1)
        gatt = get_atts()
        gatt.start(hfsmallfn=hfsmallfn)
    atts = np.genfromtxt(attpath,dtype=str,delimiter=',')
    attsz = zip(*(atts))
    name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type, distance,indegree,outdegree,betweenness,coreness_in,coreness_out,clossess_in,clossess_out,eccentricity_in,eccentricity_out,pagerank = gt.convertype2float(attsz[0]),gt.convertype2float(attsz[1]),gt.convertype2float(attsz[2]),gt.convertype2float(attsz[3]),gt.convertype2float(attsz[4]),gt.convertype2float(attsz[5]),gt.convertype2float(attsz[6]),gt.convertype2float(attsz[7]),gt.convertype2float(attsz[8]),gt.convertype2float(attsz[9]),gt.convertype2float(attsz[10]),gt.convertype2float(attsz[11]),gt.convertype2float(attsz[12]),gt.convertype2float(attsz[13]),gt.convertype2float(attsz[14]),gt.convertype2float(attsz[15]),gt.convertype2float(attsz[16]),gt.convertype2float(attsz[17]),gt.convertype2float(attsz[18]),gt.convertype2float(attsz[19]),gt.convertype2float(attsz[20]),gt.convertype2float(attsz[21]),gt.convertype2float(attsz[22]),gt.convertype2float(attsz[23]),gt.convertype2float(attsz[24]),gt.convertype2float(attsz[25]),
    # name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type, distance = np.nan_to_num(name),np.nan_to_num(created_attos),np.nan_to_num( color),np.nan_to_num( orderatio),np.nan_to_num( verified),np.nan_to_num( utype),np.nan_to_num( mentimes),np.nan_to_num( mentimes_after),np.nan_to_num( followers_count),np.nan_to_num( bi_followers_count),np.nan_to_num( province),np.nan_to_num( statuses_count),np.nan_to_num( friends_count),np.nan_to_num( userid),np.nan_to_num( verified_type),np.nan_to_num( distance)

    if mentimes_tag:
        print "---mentimes--------------------------------"
        targ = gt.totarget(np.nan_to_num(mentimes),0)        
        train_x = [ verified,utype,followers_count, friends_count,created_attos, verified_type,mentimes_after, bi_followers_count, statuses_count,distance ]# 
        train_x = zip(*(train_x))
        train_y = targ
         
        regcmo = regression_cmo()
        regcmo.start(train_x, train_y,predict_ornot=predict_ornot)
    
    if mentimes_after_tag:
        print "---mentimes_after--------------------------------"
        targ = gt.totarget(np.nan_to_num(mentimes_after),0);
        for it,itm in zip(*(targ,mentimes)):
            if it>0 and itm>0:
                print it,itm
        haveinvited =  []       
        train_x = [ followers_count,gt.totarget(np.nan_to_num(mentimes),0) ]# verified,utype, friends_count,created_attos, verified_type, bi_followers_count, statuses_count,distance
        train_x = zip(*(train_x))
        train_y = targ
        
        regcmo = regression_cmo()
        regcmo.start(train_x, train_y) 
       
    if degree_tag:
        print "---degree-preference--------------------------------"
        targ = indegree#gt.totarget(np.nan_to_num(indegree),0);
        #followers_count = np.nan_to_num(np.log(followers_count))
        train_x = [ followers_count,distance, ]# verified,utype, friends_count,created_attos, verified_type, bi_followers_count, statuses_count,distance

        train_x = zip(*(train_x))
        train_y = targ
        
#         print [y for y in followers_count if y>0]
#         print [y for y in train_y if y>0]
        regcmo = regression_cmo()
        regcmo.start(train_x, train_y)
        
if __name__=='__main__': 
    import generate_S_R as genSR
    
    workfolder_sim = "G:\\HFS\\WeiboData\\HFSWeibo_Sim\\"
    workfolder_real = "G:\\HFS\\WeiboData\\HFSWeibo\\"#small\\
    gids = ['3513472585606907','3501198583561829','3514416295764354','3510150776647546','3510947052234805','3511312581651670','3428528739801892','3373235549224874','3451840329079188']
 
    mod = 7
    experimentimes = 1    
    genSR.start(workfolder_real,workfolder_sim,gids,mod,experimentimes)
    
          