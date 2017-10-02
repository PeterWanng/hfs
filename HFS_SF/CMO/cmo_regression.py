#coding=utf8
""""K-NN classifier;Support Vector Machine;lasso;Bayesian classifier; CART ( Classification and Regression Tree ), ID3 Algorithm. bootstrapping;"
"If you want to try svm and the usual rbf-kernel does not work well, try a kernel with a lower capacity.Less compleex kernels such as linear or lower degree polynomial kernels might be a solution in those cases. Anyway, svm hyperparameters should be learned with care, using cross validation or some other bootstrap technique. "
You may also consider other popular methods like Naive Bayes, or Decision Trees which usually work good on that kind of data.
For small datasets also one nearest neighbor works well
"""
import sys
sys.path.append('../..')
from tools import commontools as gtf
import random
import os  
import numpy as np
gt = gtf()

from matplotlib import pyplot as plt
import sklearn as skl
from sklearn import datasets, linear_model 
from sklearn import metrics
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
import ntpath
import re
import string
import operator

import igraph as ig
import time


class get_atts():
        def filter(lista):
            res = []
            #metaline = 'names,fans,spth,friends,activ,invited,mentimes_after,utype = '
            for item in lista:
                print item[4],item[6]
                if item[6]<3  and item[6]>0 and item[4]>0:
                    res.append(item)
            return res
        
        def add_others_att_g(self,g,vsatts,vsatt):
            
             
            vsid = [-1]
            vsid.extend(g.vs.indices)
            vsatts.append(vsid)   
            vsatt.append('vsid')
            
            distance = [-1]
            distance.extend(g.shortest_paths_dijkstra(source=0)[0])
            vsatts.append(distance)   
            vsatt.append('distance')
            
            indegree = [-1]
            indegree.extend(g.indegree())
            vsatts.append(indegree)   
            vsatt.append('indegree')
            
            outdegree = [-1]
            outdegree.extend(g.outdegree())
            vsatts.append(outdegree)   
            vsatt.append('outdegree')
            
            betweenness = [-1]
            betweenness.extend(g.betweenness(vertices=None, directed=True, cutoff =None, weights=None,nobigint=True))
            vsatts.append(betweenness)   
            vsatt.append('betweenness')
            
            coreness_in = [-1]
            coreness_in.extend(g.coreness(mode='IN'))
            vsatts.append(coreness_in)   
            vsatt.append('coreness_in')
            
            coreness_out = [-1]
            coreness_out.extend(g.coreness(mode='OUT'))
            vsatts.append(coreness_out)   
            vsatt.append('coreness_out')
            
            clossess_in = [-1]
            clossess_in.extend(g.closeness(vertices=None, mode='IN', cutoff =None, weights=None))
            vsatts.append(clossess_in)   
            vsatt.append('clossess_in')
            
            clossess_out = [-1]
            clossess_out.extend(g.closeness(vertices=None, mode='OUT', cutoff =None, weights=None))
            vsatts.append(clossess_out)   
            vsatt.append('clossess_out')
            
            eccentricity_in = [-1]
            eccentricity_in.extend(g.eccentricity(vertices=None, mode='IN'))
            vsatts.append(eccentricity_in)   
            vsatt.append('eccentricity_in')
            
            eccentricity_out = [-1]
            eccentricity_out.extend(g.eccentricity(vertices=None, mode='OUT'))
            vsatts.append(eccentricity_out)   
            vsatt.append('eccentricity_out')
            
            pagerank = [-1]
            pagerank.extend(g.pagerank(vertices=None, directed=True, damping=0.85, weights=None,arpack_options=None))
            vsatts.append(pagerank)   
            vsatt.append('pagerank')
            
#             assorDegID = [-1]
#             assorDegID.extend(g.assortativity(g.indegree(),directed= True))
#             vsatts.append(assorDegID)   
#             vsatt.append('assorDegID')
#             
#             assorDegOD = [-1]
#             assorDegOD.extend(g.assortativity(g.outdegree(),directed= True))
#             vsatts.append(assorDegOD)   
#             vsatt.append('assorDegOD')
            

            
            return vsatts,vsatt          
        
        def getatts(self,g,attfp_vs=None,attfp_es=None):
            "['username', 'avatar_large', 'created_attos', 'color', 'orderatio', 'statuslast', 'label', 'description', 'city', 'verified', 'utype', 'mentimes', 'follow_me', 'mentimes_after', 'verified_reason', 'followers_count', 'location', 'timein', 'bi_followers_count', 'profile_url', 'province', 'statuses_count', 'statuslasttos', 'friends_count', 'online_status', 'allow_all_act_msg', 'profile_image_url', 'allow_all_comment', 'geo_enabled', 'lang', 'remark', 'favourites_count', 'name', 'url', 'gender', 'created_at', 'userid', 'uidstr', 'verified_type', 'following']"
            "['source_wb', 'favorited', 'createdtimetos', 'retweeted_status', 'truncated', 'thumbnail_pic', 'text', 'created_at', 'mlevel', 'userid', 'idstr', 'color', 'attitudes_count', 'screen_name', 'comments_count', 'visible', 'geo', 'reposts_count']"
            vsatts,esatts = [],[]
        
            vsatt = g.vs.attributes()
          
            for it in vsatt:
                vsatts.append(g.vs[it])
            
            vsatts,vsatt = self.add_others_att_g(g,vsatts,vsatt)
            vsatts = zip(*vsatts)  
            vsatts.insert(0,vsatt)
        #     vsatts.sort(key=operator.itemgetter(vsatt.index(1)))
            if attfp_vs:
                gt.saveList(vsatts,attfp_vs)
            
            esatt = g.es.attributes()
            esids = [-1]
            esids.extend(g.es.indices)
            esatts.append(esids)
            for it in esatt[1:]:
                esatts.append(g.es[it])
            esatts = zip(*esatts)
            esatts.insert(0,esatt)
            if attfp_es:
                gt.saveList(esatts,attfp_es)
                
            return vsatts,esatts#np.genfromtxt(attfp,dtype=float,delimiter=',',usecols=(1,2,3,4,5,6,7))
        
        
        def get_graph(self,weibofilep,gmlfilep=None,timep=None,addcommentlist=True):
            from weibo_tools import weibo2graph
            wb2g = weibo2graph() 
            g = wb2g.start(weibofilep,gmlfilep,timep=timep,addcommentlist=addcommentlist)
            return g
        
        def get_att_inlist(self,att,metalist=None,metalist_abstract=None):
            if not metalist_abstract:
                return att
                
            res = []
            attz = zip(*(att))
            metalist = att[0] if not metalist else metalist
        
        #     i = 0
        #     for meta in metalist:
        #         if meta in metalist_abstract:
        #             res.append(attz[i])
        #         i+=1
            defaultv = ('None' for it in range(len(attz[0])))
            for meta in metalist_abstract:
                if meta in metalist:
                    res.append(attz[metalist.index(meta)])
                else:
                    res.append(defaultv)
                
            res = zip(*(res))    
            return res
                
                
        def get_att_temp(g,line=None,oldweibo=False):
            if g.vcount()>0:
                fans = g.vs['followers_count']
                friends = g.vs['friends_count']
                activ = g.vs['statuses_count']
                invited = g.vs['mentimes']
                mentimes_after = g.vs['mentimes_after']
                utype = g.vs['utype']
                try:
                    pass
                    #fans.remove(None)#._del_('None')#np.nan_to_num(fans)
                except:
                    pass
                names = g.vs['name']
                spth = g.shortest_paths(0)[0]
                
        #         return names,fans,spth,friends,activ,invited,mentimes_after,utype
            
        #         print '==============='
        #         for a,b,c,d,e,f,g,h in zip(*(names,fans,spth,friends,activ,invited,mentimes_after,utype)):
        #             if f>0 and b:#h in [1,2] and 
        #                 print 'att:',a,b,c,d,e,f,g,h
                    
                           
                #g.vcount()
        

 
            
        def get_att_es_selected(self,workfolder_att,attselected=None,attpath2save=None):
            "['source_wb', 'favorited', 'createdtimetos', 'retweeted_status', 'truncated', 'thumbnail_pic', 'text', 'created_at', 'mlevel', 'userid', 'idstr', 'color', 'attitudes_count', 'screen_name', 'comments_count', 'visible', 'geo', 'reposts_count']"

            atts = []
            if not attselected:
                attselected = ['source_wb', 'createdtimetos', 'retweeted_status',  'userid', 'idstr', 'color', 'attitudes_count', 'screen_name', 'comments_count', 'reposts_count']
            i = 0
            for  vsattsfp in os.listdir(workfolder_att):
                i+=1
                if  str(vsattsfp).endswith('attvs'):
                    print vsattsfp
                    vsatts = gt.csv2list_new(workfolder_att+vsattsfp,nan2num=False)
                    att = self.get_att_inlist(vsatts,None,attselected)#get_att_temp(g)
                    atts.extend(att)
            
            attsz = zip(*atts)
            # name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type, distance = attsz[0],attsz[1],attsz[2],attsz[3],attsz[4],attsz[5],attsz[6],attsz[7],attsz[8],attsz[9],attsz[10],attsz[11],attsz[12],attsz[13],attsz[14],attsz[15],
            #'created_attos', 'color', 'orderatio', 'verified', 'utype', 'mentimes', 'mentimes_after', 'followers_count', 'bi_followers_count', 'province', 'statuses_count', 'friends_count', 'name', 'uidstr', 'verified_type', 'distance') ('verified', '1', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'verified', 'verified', 'verified', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'verified', 'verified', 'verified', 'verified', 'False', 'False', 'False', 'False', 'True', 'True', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'verified', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'True', 'True', 'False', 'True', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'False', 'verified', 'verified', 'verified', 
            print len(attsz),len(atts)
            
            att4Invit = []
            for  name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type,distance in atts:
                 #print utype,mentimes
                 mentimes = -1 if mentimes=='mentimes' else mentimes
                 if utype in  ['1','2'] and int(mentimes)>-1:
            #         print name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type,distance
                    att4Invit.append([name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type,distance])
            #         att4Invit.append([verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count,statuses_count, friends_count, verified_type,distance])
 
            if not attpath2save:
                attpath = r'G:\HFS\WeiboData\HFSWeibo\ATT\test4lr_all.txt'#_small#_all
                gt.saveList(att4Invit,attpath,writype='a+')
                            
        def get_att_vs_selected(self,workfolder_att,attselected=None,attpath2save=None):
            atts = []
            if not attselected:
                attselected = [ 'name','created_attos', 'color', 'orderatio','verified', 'utype', 'mentimes','mentimes_after','followers_count', 'bi_followers_count','province', 'statuses_count',  'friends_count','uidstr', 'verified_type','distance','indegree','outdegree','','betweenness','coreness_in','coreness_out','clossess_in','clossess_out','eccentricity_in','eccentricity_out','pagerank']
            i = 0
            for  vsattsfp in os.listdir(workfolder_att):
                i+=1
                if  str(vsattsfp).endswith('attvs'):
                    print vsattsfp
                    vsatts = gt.csv2list_new(workfolder_att+vsattsfp,nan2num=False)
                    att = self.get_att_inlist(vsatts,None,attselected)#get_att_temp(g)
                    atts.extend(att)
            
            attsz = zip(*atts)
            # name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type, distance = attsz[0],attsz[1],attsz[2],attsz[3],attsz[4],attsz[5],attsz[6],attsz[7],attsz[8],attsz[9],attsz[10],attsz[11],attsz[12],attsz[13],attsz[14],attsz[15],
            #'created_attos', 'color', 'orderatio', 'verified', 'utype', 'mentimes', 'mentimes_after', 'followers_count', 'bi_followers_count', 'province', 'statuses_count', 'friends_count', 'name', 'uidstr', 'verified_type', 'distance') ('verified', '1', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'verified', 'verified', 'verified', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'verified', 'verified', 'verified', 'verified', 'False', 'False', 'False', 'False', 'True', 'True', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'verified', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'True', 'True', 'False', 'True', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'False', 'False', 'False', 'True', 'False', 'False', 'False', 'verified', 'verified', 'verified', 
            print len(attsz),len(atts)#,attsz
            
            att4Invit = []
            for ita in atts:
                'name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type,distance,indegree,outdegree,betweenness,coreness_in,coreness_out,clossess_in,clossess_out,eccentricity_in,eccentricity_out,pagerank'
                #print utype,mentimes
                utype,mentimes = ita[5],ita[6]
                mentimes = -1 if mentimes=='mentimes' else mentimes
#                 if utype in  ['1','2'] and int(mentimes)>-1:
                if utype in  ['1','2','1.0','2.0'] and float(mentimes)>-1:
                #         print name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type,distance
#                    att4Invit.append([name,created_attos, color, orderatio, verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count, province, statuses_count, friends_count, userid, verified_type,distance,indegree,outdegree,betweenness,coreness_in,coreness_out,clossess_in,clossess_out,eccentricity_in,eccentricity_out,pagerank])
                   att4Invit.append(ita)
                #         att4Invit.append([verified, utype, mentimes, mentimes_after, followers_count, bi_followers_count,statuses_count, friends_count, verified_type,distance])
 
            if not attpath2save:
                attpath = r'G:\HFS\WeiboData\HFSWeibo\ATT\test4lr_all.txt'#_small#_all
                gt.saveList(att4Invit,attpath,writype='a+')

            
        def start(self,workfolder=None,hfsmallfn=None,attselected=None,att_vpath2save=None,att_epath2save=None):
            
            if not workfolder:
                workfolder = "G:\\HFS\\WeiboData\\HFSWeibo\\"
            workfolder_gml = gt.createFolder(workfolder+"GML\\")
            workfolder_att = gt.createFolder(workfolder+"ATT\\")
            
            if not hfsmallfn:
                hfsmall = gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeibo\small\small235mids4CMO.content_all')
                hfsmallfn = zip(*(hfsmall))[0]
            for filen in hfsmallfn: 
                filen = str(filen)#'2014061812_3717147479998656'
                wbfilep = workfolder+filen+'.repost'#r'G:\HFS\WeiboData\HFSWeibo\test\test_newformat.repost'# r'G:\HFS\WeiboData\HFSWeibo\3385409596124201.repost' #          
            #     wbfilep = r'H:\DataSet\HFS_XunRen_620\2014\2014071400_3731913761707254.repost'
                gmlfp = workfolder_gml+filen+'.gml'#'G:\HFS\WeiboData\HFSWeibo\test\3482476628770294.gml' 
                attfp_vs =  workfolder_att+filen+'.attvs'#None#r'G:\HFS\WeiboData\HFSWeibo\test\test_newformat.attvs'#
                attfp_es =  workfolder_att+filen+'.attes'#None#r'G:\HFS\WeiboData\HFSWeibo\test\test_newformat.attes'#
                
                vsatts,esatts = [],[]
                if os.path.exists(attfp_vs):#0:#
                #     vsatts = np.genfromtxt(attfp_vs,dtype=str,delimiter=',',)#usecols=(1,2,3,4,5,6,7)
                #     esatts = np.genfromtxt(attfp_es,dtype=str,delimiter=',',)#usecols=(1,2,3,4,5,6,7)
#                     vsatts = gt.csv2list_new(attfp_vs,nan2num=False)#usecols=(1,2,3,4,5,6,7)
#                     esatts = gt.csv2list_new(attfp_es,nan2num=False)#usecols=(1,2,3,4,5,6,7)
                    pass
                else:
                    if os.path.exists(gmlfp):
                        g = ig.read(gmlfp)
                    else:
                        try:
                            g = self.get_graph(weibofilep=wbfilep,gmlfilep=gmlfp,timep=None,addcommentlist=True,)
                        except:
                            print 'bad file',wbfilep
                            continue
                    vsatts,esatts = self.getatts(g,attfp_vs,attfp_es)
            #         gt.drawgraph(g)  
            self.get_att_vs_selected(workfolder_att,attselected,att_vpath2save)  
#             self.get_att_es_selected(workfolder_att,attselected,att_epath2save) 

class regression_cmo():
   
#     def gt.convertype2float(self,lista,defaultv=0):
#         res = []
#         for it in lista:
#             try:
#                 it = float(it)
#             except:
#                 it = defaultv
#             res.append(it)
#         return res

    
    def regr_one(self,train_x, train_y, test_size,predict_ornot):
        if predict_ornot:
            train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=test_size, random_state=10)#
        else:
            test_x, test_y = train_x, train_y
        regr = linear_model.LogisticRegression()
#         regr = linear_model.LinearRegression()
        regr.fit(X=train_x, y=train_y)  
        predict_result = regr.predict(X=test_x)
        i,j = 0,0
        for a,b in zip(*(predict_result,test_y)):
            i+=1
            if a!=b:
                j+=1
                #print a,b
        print 'accuracy:',(i-j)/(i*1.0)
        score, permutation_scores, pvalue = permutation_test_score(regr,train_x,train_y, scoring="accuracy")
        print 'score,  pvalue = ',score,  pvalue
        
        return regr,predict_result,test_y      
    
    def print_metrics(self,test_size,regr,test_y,predict_result): 
        res,reslist = {},[]  
    #     scores, pv = chi2(train_x, train_y)
    #     print pv ,scores
        #print test_y,predict_result,
        coef = regr.coef_#,regr.raw_coef_   
    
        r2 = metrics.r2_score(test_y, predict_result)
        explv = metrics.explained_variance_score(test_y, predict_result)
        pv = metrics.precision_score
        
        res['test_size'] = test_size
        res['coef'] = coef
        res['r2'] = r2
        res['explv'] = explv 
        
        reslist.append([test_size,coef,r2,explv])
        return reslist
    
    def start(self,train_x,train_y,predict_ornot=False):
    
        train_x = np.nan_to_num(train_x)
        train_y = np.nan_to_num(train_y)   
        
        if 0:
            train_x, train_y = np.mat(train_x), np.mat(train_y).transpose()
            #print train_x, train_y
            print len(train_x), len(train_y)
            
            
            from regression import logisticReg
            opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
            
            lr = logisticReg()
            lr.start(train_x, train_y,opts)
        
        if 1:
            train_x  = np.array(train_x);train_y  = np.array(train_y);
            train_x = StandardScaler().fit_transform(train_x)
            train_x = np.nan_to_num(train_x)            
            
            res = []
            train_test_ratio = 1 if not predict_ornot else 9
            for test_size in range(train_test_ratio,0,-1):
                test_size = test_size*0.1
                regr,predict_result,test_y = self.regr_one(train_x, train_y, test_size,predict_ornot)
                res_one = self.print_metrics(test_size,regr,test_y,predict_result)
                
                res_regr_self = self.print_metrics(test_size,regr,test_y,predict_result)
                
                print res_one[0]
                res.extend(res_one)
            
            resz = zip(*res)
            x = list(resz[0]);x.reverse()
            y1 = resz[1]
            y2 = resz[2]
            y3 = resz[3]
            
            plt.plot(x,y2)
            plt.plot(x,y3)
            plt.show()
            for it in resz:
                print it
                    
            #resz = zip(*res)
        #     rest = []
        #     for item in res:
        #         for k,v in zip(*(item[0].keys(),item[0].values())):#['test_size']
        #             print k,v
        #             rest.append(v)
            
          

def getdata_one(weibofp):
    from weibo_tools import weibo2graph
    from weibo_tools import wbg
    wb2g = weibo2graph() 
    
    wbfilep_fw=r'G:\HFS\WeiboData\HFSWeibo\test\toy.repost'
    wbfilep_cm=None
    txtlist = wb2g.getxtlist_old(wbfilep,wbfilep_comt)
    txtlist = wb2g.transoldweibo2new(txtlist)
    searchmid = os.path.basename(wbfilep).split('.')[0]
    firstline = wb2g.findmetacoc(searchmid=searchmid)
    if firstline == -1:
        pass
    else:
        txtlist.insert(0,firstline)    
    
    res = []
    for wb in txtlist:
        
        wb4g = wbg(line,oldweibo)
        res.append(wb4g.reposts_count)

if __name__=='__main__': 
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

# if __name__=='__main3__':
#     "choice"
#     
#     getdata_one(weibofp)        
# er
















#     
# def mentionUserfromText(text):
#     linestr = text.decode('utf-8')
#     forword_twice_index = linestr.find('//@')
#     linestr_one = linestr[:forword_twice_index]
#     linestr_twice = linestr[forword_twice_index:]
#     topic_index = linestr.find('#')
#     
#     res = []
#     secondRetwituser = []
#     mentioneduser = []
#     topic =  []
#     if forword_twice_index!=-1:            
#         secondRetwituser=re.findall(u"//@(.*?)[~ :  ：, .;'，\\。：\r？\n!//@\t@\\[\\]\'：]", linestr_twice) 
#     if forword_twice_index!=0:    
#         mentioneduser = re.findall(u"@(.*?)[\s :  ：,！ .;'，\\。：\r？\n!//@\t@\\[\\]\']", linestr_one)
#     if topic_index!=-1:    
#         topic = re.findall(u"#(.*?)[#]", linestr)
#     
#     return mentioneduser,secondRetwituser
# 
#     result_m = string.join(mentioneduser,sep=';')
#     result_sm = string.join(secondRetwituser,sep=';')
#     result_t = formatxt(string.join(topic,sep=';'))
#     
#     result = string.join([result_m,result_sm,result_t],sep='|')
#     
#     return result#,secondRetwituser
# 
# def get_wordcount(wordlist):
#     wordcnt ={}
#     for i in wordlist:
#         if i in wordcnt:
#             wordcnt[i] += 1
#         else:
#             wordcnt[i] = 1
#     worddict = wordcnt.items()
#     worddict.sort(key=lambda a: -a[1])
#     #    for word,cnt in worddict:
#     #        word.encode('gbk'), cnt
#     return wordcnt
# 
# def invitePam2(ma_fans,cbm_frs,act_micorcnt,bi_followers_count,inv_mention,repostlen):
# #     import Auto_Norm_Mat
#     mat = gt.normlistlist([ma_fans,cbm_frs,act_micorcnt,bi_followers_count,repostlen])#Auto_Norm_Mat.AutoNorm([ma_fans,cbm_frs,act_micorcnt,bi_followers_count,repostlen])
#     inv_mention = gt.normalizelist(inv_mention,sumormax='max')#Auto_Norm_Mat.AutoNorm([inv_mention])
# #     inv_mention = [(i=1 if i>0 else i) for i in inv_mention]
#     #x = zip(*(ma_fans,cbm_frs,act_micorcnt,bi_followers_count,repostlen))
#     x = zip(*(mat))
#     
#     ## step 1: load data
#     print "step 1: load data..."
#     train_x, train_y = loadData()
#     test_x = train_x; test_y = train_y
#     
#     ## step 2: training...
#     print "step 2: training..."
#     opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
#     optimalWeights = trainLogRegres(train_x, train_y, opts)
#     
#     ## step 3: testing
#     print "step 3: testing..."
#     accuracy = testLogRegres(optimalWeights, test_x, test_y)
#     
#     ## step 4: show the result
#     print "step 4: show the result..."    
#     print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
#     showLogRegres(optimalWeights, train_x, train_y) 
#     
#     "-------------------regression model-------------------------------" 
#     regr = linear_model.LinearRegression(normalize=True)
# #     regr = linear_model.LogisticRegression()
#    
# #     regr = linear_model.ARDRegression()#     0.839344262295     0.66485042735
# #     regr = linear_model.BayesianRidge()#     0.852459016393     0.651709401709
# #     regr = linear_model.ElasticNet()#     0.852459016393     0.5
# #     regr = linear_model.ElasticNetCV()#0.852459016393     0.552884615385
# #     regr = linear_model.Lars()#     0.606557377049     0.495085470085
# #     regr = linear_model.LarsCV()#     0.850819672131     0.521260683761
# #     regr = linear_model.Lasso()#     0.852459016393     0.5
# #     regr = linear_model.LassoCV()#     0.852459016393     0.537072649573
# #     regr = linear_model.LassoLarsCV()#     0.852459016393     0.507905982906
# #     regr = linear_model.LassoLarsIC()#     0.834426229508     0.640277777778
# #     regr = linear_model.LogisticRegression()#     0.852459016393     0.5
# #     regr = linear_model.MultiTaskElasticNet()#     0.852459016393     0.5
# #     regr = linear_model.MultiTaskLasso()#     0.852459016393     0.5
# #     regr = linear_model.OrthogonalMatchingPursuit()#     0.839344262295     0.646367521368
# #     regr = linear_model.PassiveAggressiveClassifier()#     0.852459016393     0.5
# #     regr = linear_model.PassiveAggressiveRegressor()#     0.852459016393     0.520085470085
# #     regr = linear_model.Perceptron()#     0.852459016393     0.5
# #     regr = linear_model.Ridge()#     0.847540983607     0.623290598291
# #     regr = linear_model.RidgeClassifier()#     0.847540983607     0.497115384615
# #     regr = linear_model.RidgeClassifierCV()#     0.847540983607     0.497115384615
# #     regr = linear_model.RidgeCV()#     0.847540983607     0.640598290598
# #     regr = linear_model.SGDClassifier()#     0.855737704918     0.511111111111
# #     regr = linear_model.SGDRegressor()#     0.852459016393     0.44188034188
# #     from sklearn import svm
# #     regr =svm.SVC()
# #     regr = linear_model.LinearRegression()
#     regfit = regr.fit(x, inv_mention)
#     predict_result = regr.predict(X=x)
#     from sklearn import metrics
#     from sklearn.feature_selection import chi2
# #     scores, pv = chi2(x, inv_mention)
# #     print pv ,scores
#     print 'r2:',metrics.r2_score(inv_mention, predict_result)
#     print metrics.explained_variance_score(inv_mention, predict_result)   
#     
# 
# featureFilepath = r'G:\HFS\WeiboData\HFSWeibo\3553712926158978.repost'
# repost = gt.csv2list_new(featureFilepath)
# from weibo_tools import deal_weibo
# dwb = deal_weibo()
# ma_fans,cbm_frs,inv_mention,act_micorcnt,bi_followers_count,mid,repostlen = dwb.weibo2list(featureFilepath)
# print inv_mention
# # gt.listDistribution(inv_mention,showfig=True)
# invitePam2(ma_fans,cbm_frs,act_micorcnt,bi_followers_count,inv_mention,repostlen)
# er
# 
# # print dwb.txt2coc_main(None,repost,featureFilepath+'.coc')
# 
# 
# 
# 
# 
# print ma_fans
# fansum = float(np.sum(ma_fans))
# ma_fans = [i/fansum for i in ma_fans]
# print ma_fans
# 
# 
# er
# 
# featlist = gt.csv2list_new(featureFilepath,passmetacol=0,convertype=str,nan2num=False)
# featlist.reverse()
# featlistz = zip(*(featlist))
# txt = featlistz[3]
# users = featlistz[10]
# 
# res = []
# for item,user in zip(*(txt,users)):
#     musers = mentionUserfromText(item)[0]
#     res.extend(musers)
# #     print musers,user
# print get_wordcount(res)#.unique(res)
#  
# er
# "====================================================="
# 
# def getsindex(sflist,flag):
#     result=[]
# #     f = gt.csv2list(filep)
#     linecnt = 0
#     flagindex = 0
#     for line in sflist:      
#         linecnt+=1
#         if line[1]==flag and ~flagindex:
#             flagindex = linecnt
#             break
#         else:
#             pass  
#     
#     result.append(flagindex)        
#     result.append(len(sflist))        
#             
#     return result
# 
# def list2libsvmformat(lista,percindex=2):
#     listResult = ''
#     for item in lista:
#         i = 0
#         litem = str(item[1])+'\t'#item[0]+'\t'+
#         for it in item[percindex+1:]:
#             i+=1
#             litem+=str(it)+'\t'#(str(i)+':'+
#             listResult+=litem
#             litem = ''
#         listResult+='\n' 
#     return listResult.replace('\t\n','\n')
#     
# def selectPercent(lista,percent,percentindex=1):
#     result = []
#     for item in lista:
#         if str(item[percentindex])==str(percent):
#             result.append(item)
#     return result
#     
# def sampline(listfeature,svmoutfile,lista,listb,listc,listd):
#     traindata = []
#     testdata = []
#             
# #     f = open(svmfile)
# #     fw1 = open(svmoutfile+'.scl','w')
# #         fw2 = open(svmoutfile+'.ts','w')
#     lincnt = 0
#     for line in listfeature:
#         lincnt+=1
#         if (lincnt in lista) or (lincnt in listc):
#             traindata.append(line)
#             continue        
# #     f.close()
# #     f = open(svmfile)    
#     lincnt = 0
#     for line in listfeature:
#         lincnt+=1
#         if (lincnt in listd) or (lincnt in listb):
#             testdata.append(line)
#             continue
# #     f.close()
#     return [traindata,testdata]    
#     
#     
# def sample2(lista,listsindex,listfindex,fratio,sratio,tssratio,tsfratio,svmoutfile):
#     sucnt = len(listsindex)
#     failcnt = len(listfindex)
#     
#     scnt=int(round(sucnt*sratio))
#     fcnt=int(round(failcnt*fratio))#scnt#
#     stscnt=int(round(sucnt*tssratio-0.01))
#     ftscnt=int(round(failcnt*tsfratio-0.01))#stscnt#
# #         print '===============',scnt, stscnt
#     s = gt.divideList(listsindex, scnt, stscnt)
#     f = gt.divideList(listfindex, fcnt, ftscnt)
#  
#     datasample = sampline(lista,svmoutfile,s[0],s[1],f[0],f[1])
#         
#     return datasample#[s,f]
#     
#     
# def delcoc(lista,delstr,colindex):
#     result=[]
#     for item in lista:
#         item0 = str(item[colindex]).replace(delstr,'')
#         it = list(item[1:])
#         it.insert(0,item0)
#         result.append(it)
#     return result
# 
# def svmfeatures(vd,filename,rangemin,rangemax):
#     result_tr = []
#     result_ts = []
#     vdsf = gt.connectlist_sf(vd, sameposition_be=0,suclista=gt.csv2list(r'G:\HFS\WeiboData\Statistics\meta_successed308.txt'),rangemin=rangemin,rangemax=rangemax,keysuffix=None)
#     
# 
# #     vdsvm = list2libsvmformat(vdsf)
#     svmfilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat\vd'+filename+'_km.skl'
# #     
# #     fw = open(svmfilepath,'w')
# #     fw.write(vdsvm)
# #     fw.close()
#     for i in range(1,11):
#         svmoutfilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat\vd'+filename+'_km_'+str(i)+'.skl' 
#         sfindex = getsindex(vdsf,'-1')
#         scnt = sfindex[0]
#         fcnt = sfindex[1]
#         print '=============================',scnt,fcnt
# #         a.sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.8*scnt)/(fcnt-scnt),sratio=(0.8*scnt)/scnt,tssratio=(0.2*scnt)/scnt,tsfratio=(0.2*scnt)/(fcnt-scnt),svmoutfile=svmoutfilepath)
# #         a.sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=44/262.0,sratio=44/48.0,tssratio=4/48.0,tsfratio=4/262.0,svmoutfile=svmoutfilepath)
# #         resultone = sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.8*fcnt)/(fcnt),sratio=(0.8*scnt)/scnt,tssratio=(0.2*scnt)/scnt,tsfratio=(0.2*fcnt)/(fcnt),svmoutfile=svmoutfilepath)
#         resultone = sample2(vdsf,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.8*fcnt)/(fcnt),sratio=(0.8*scnt)/scnt,tssratio=(0.2*scnt)/scnt,tsfratio=(0.2*fcnt)/(fcnt),svmoutfile=svmoutfilepath)
#         result_tr.append(resultone[0])
#         result_ts.append(resultone[1])
#     print '=============================',scnt,fcnt
#     return [result_tr,result_ts]
# 
# "======================================================================================================================="
# def generateSklData(featureFilepath,perc):
#     "IN:feature file path"
#     "OUT:skl training and testing data list"
#     filepath = featureFilepath
#     filep = os.path.basename(filepath)
#     rangemin=0
#     rangemax=100000
#     
#     dataTrain = []
#     dataTest = []
#     
#     featlist = gt.csv2list_new(filepath,passmetacol=2,convertype=float,nan2num=True)
#     featlist = delcoc(featlist,delstr='.coc',colindex=0)
#     vd = gt.normlistlist(selectPercent(featlist,perc,1),metacolcount=2,sumormax='max')
# 
#     [dataTrain,dataTest] = svmfeatures(vd=vd,filename=filep,rangemin=rangemin,rangemax=rangemax)#filename=str(k)+'_'+str(perc)+'_'+filep
# #     feature = gt.normlistlist(selectPercent(gt.csv2list_new(filepath,passmetacol=2,convertype=float,nan2num=True),perc,1),metacolcount=2,sumormax='max')#gt.csv2list_new(filepath)#
# #     vd = gt.connectlist(vd, feature, 0, 0, passcol=2)
#     print 'finished one:',filep
#     
#     return [dataTrain,dataTest]
# "======================================================================================================================="
# 
# def getarget(dataset,xrange,yrange):
#     "IN: a listlist with "
#     "OUT: "
#     temp_tr = zip(*dataset[0])
#     temp_ts = zip(*dataset[1])
#     xtr = temp_tr[xrange[0]:xrange[1]]
#     ytr = temp_tr[yrange[0]:]
#     
#     xtr = zip(*xtr)
#     
#     
#     xts = temp_ts[xrange[0]:xrange[1]]
#     yts = temp_ts[yrange[0]:]
#     
#     xts = zip(*xts)
#     return [numpy.array(xtr),numpy.array(xts),numpy.array(ytr),numpy.array(yts)]# [xtr,xts,ytr,yts]
#        
# def regression(dataset,regressionType):
#     "IN:dataset which include training and testing data; regression type:1-linear;2-logistic;"
#     "OUT:predict Result"
#     result = []
#     
#     
#     dataset = getarget(dataset,[0,1],[1,])
#     
# #     test_index = -61 
#     diabetes_x_train = dataset[0]#[:test_index] #训练样本
#     diabetes_x_test = dataset[1]#[test_index:] #检测样本
#     
#     diabetes_y_train = dataset[2]#diabetes.target[:test_index]
#     diabetes_y_test = dataset[3]#diabetes.target[test_index:]
#      
#     # print len(diabetes_y_test),len(diabetes_x_test),len(diabetes_x_train),len(diabetes_y_train),#
#      
#     regr = linear_model.LinearRegression()
# #     regr = linear_model.LogisticRegression()
# #     regr = linear_model.Lasso()
#     print diabetes_x_train, '\n',diabetes_y_train 
#     regr.fit(diabetes_x_train, diabetes_y_train)
#     predict_result = regr.predict(X=diabetes_x_test)
#     print predict_result
#     print diabetes_y_test
#     wr = 0
#     rr = 0    
# 
#     from sklearn import metrics
#     y = np.array(diabetes_y_test)#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
#     pred = np.array(predict_result)
#     fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
#     aucvalue = metrics.auc(fpr, tpr)
#     print sklfile,"AUC:",aucvalue
#     
#     return result
#     
# if __name__=='__main__':
#     
#     "One feature"
#     featureFilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\test\statcore\.bifansumavg'
#     dataset = generateSklData(featureFilepath,perc='1.0')
#     print dataset
#     predictResult = regression(dataset,regressionType=1)
#     
#     
#     