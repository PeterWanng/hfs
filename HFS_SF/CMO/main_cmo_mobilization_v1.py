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

def start_create_gml(workfolder):
    workfolder_gml = gt.createFolder(workfolder+"GML\\")
    workfolder_att = gt.createFolder(workfolder+"ATT\\")
    i = 0
    for fp in os.listdir(workfolder):
        fname = os.path.splitext(fp)
        i+=1
        if fname[-1]=='.repost':# or fname[-1]=='.comment':
            wbfilep = workfolder+fp
            print i,wbfilep
            gmlfp = workfolder_gml+fname[0]+'.gml'
            g = ig.Graph()
            if not os.path.exists(gmlfp):
                g = weibo2g.start(wbfilep,gmlfp,addcommentlist=True)#ig.read(gmlfp)#
            else:
                g = ig.read(gmlfp)
            
            "add time slice function"
            for percent in percentlist:
                lengthNow = int(round(len(timelist)*percent))
                lengthNow = lengthNow if lengthNow>1 else 1
                timelistPercentNow = timelist[:lengthNow]
                timelistPeriodNow = selecTime(timelistPercentNow,periodcnt)
                for timep in timelistPeriodNow:
                    g = g.subgraph_edges(es.select(createdtimetos_le = timep))

            #grt.analyzeNetNodes(g,workfolder_att,str(fname[0]))
            grt.analyzeNetNodes_New(g,workfolder_att,str(fname[0]))
            print grt.analysisNet(g)
            
            
            

workfolder = "G:\\HFS\\WeiboData\\HFSWeibo\\testNew\\"#"H:\\DataSet\\HFS7\\"        
start_create_gml(workfolder)

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
    
          