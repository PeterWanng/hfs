# -*- coding=utf-8 -*-
#encoding=utf8

##将案例先按kmeans分为几类，对每类进行正负例的平均进行比较
##input：案例某特征的list；kmeans分类根据的list,类数k；成败标记文件
##output：每类案例的正负均值 

#可调整之处：均带有adjust标记，如聚类的距离函数distance (x,y)

import os
import string
import matplotlib.pyplot as plt
import pylab
import numpy
import sys
sys.path.append('..')
from tools import commontools as gtf

gt=gtf()


def repairline(csvfile,seprator):
#     return csvfile.readline().replace('\n','').replace(' ','')
    return csvfile.readline().replace(seprator+'\n','').replace('\n','')#.replace(' ','').replace(seprator+'\n'+seprator,'')

def csv2list(csvfilepath,seprator='\t',start_index=0):
    print 'convert the file:',csvfilepath
    csvfile = open(csvfilepath)
    resultlistemp = []
    line=repairline(csvfile,seprator)
    if line=='':
        line=repairline(csvfile,seprator) 
    i = 0    
    while line:
        linelist = line.split(seprator)
        resultlistemp.append(linelist[start_index:])
#         datalistemp.append(map(float,datalist[i][1:]))
    #     datalist.insert(0,linelist[0])    
        line=repairline(csvfile,seprator)
        i+=1
    csvfile.close()
    return resultlistemp

def fun_dis (x,y,n):  
    return sum (map (lambda v1,v2:pow(abs(v1-v2),n), x,y))  
 
def distance (x,y):  #adjust
    return fun_dis (x,y,2)  
#     return fun_dis (x,y,1)  
      
 
def min_dis_center(center, node):  
    min_index = 0 
    min_dista = distance (center[0][1],node)  
    for i in range (1,len(center)):  
        tmp = distance (center[i][1],node)  
        if (tmp < min_dista):  
            min_dista = tmp  
            min_index = i  
    return min_index  

def k_means (info,k=1000):  
#     print 'kmeans'
    center = [[1,info[i]] for i in range(k)]  
    result = [[i] for i in range(k)]  
    width  = len (info[0])  
    for i in range(k,len(info)):  
        min_center = min_dis_center (center,info[i])  
        for j in range(width):  
            center[min_center][1][j] = (center[min_center][1][j] * center[min_center][0] + info[i][j])/ (1.0+center[min_center][0])  
        center[min_center][0] += 1 
        result[min_center].append (i)  
    return result,center      

def convertype2ids(indexlist,listb,idposition=0):
    #print 'input:下标list；处理的list；欲取出list中的元素所在的索引'
    #print 'out：与indexlist同构的list中对应与index那一列 '
    resulist = [[],[]]
    for item in indexlist:        
        resulist[0].append([(listb[item][idposition])])
        resulist[1].append(listb[item])
    return resulist

def connectlist(lista,belista,sameposition_a,sameposition_be):
    # 'input:欲连接的两个list，连接条件所在的索引位置'
    # 'out：连接好的list，belista在后 '
    listout=[]   
    for i in lista:
        for j in belista:
            if i[sameposition_a].split('.')[0]==j[sameposition_be].split('.')[0]:
                if len(j)<len(belista[2]):#避免空行，但以第三行为标准了，必须保证第三行正确
                    print 'error line when connectlist:',j
                    break
                i.extend(j[1:])
                listout.append(i)
                break
    return listout

def connectlist_sf(lista,belista,sameposition_a,sameposition_be):
    # 'input:欲连接的两个list，连接条件所在的索引位置'
    # 'out：连接好的list，belista在后 '
    listout=[]   
    for i in belista:
        for j in lista:
            if i[sameposition_a].split('.')[0]==j[sameposition_be].split('.')[0]:#.repost
#                     j.insert(1,str(i[3]).replace(' ','')+'_'+str(j[0]).replace(' ',''))
                j.insert(1,str(i[3]).replace(' ',''))
                j.insert(2,str(j[0]).replace(' ',''))
                j=j[1:]
                #print j
                listout.append(j)
                break
    return listout

def departlist(listb,s,f,sfposition,passn=0):
    #IN:list;OUT:将list按sfposition对应的值以条件s，f分割开分别作为list的元素加入  ,忽略前passn项 
    lists = []
    listf = []
    listsf = []
    for item in listb:
        if item[sfposition] == s:
            lists.append(item[passn:])
        if item[sfposition] == f:
            listf.append(item[passn:])
    listsf.append(lists)
    listsf.append(listf)
    return listsf

def Sum(lista):#adjustable-可调整是所有累加还是最大的
    r=0.0
#     for i in lista:
#         j = float(i)
#         r+=j
    #################
    r = numpy.max(map(float,lista))#
    #################
    return r#round(result,3)

def normalizelist(lista):
    #IN:list,OUT:guiyihua   
    fanslen = len(lista)
     
    usersumlistNorm = []
    listsum = Sum(lista)
    for item in  lista:
        usersumlistNorm.append(float(item)/listsum)  
    return usersumlistNorm 

def repair(kmeans_list,metacolcnt):
    a = zip(*zip(*kmeans_list)[metacolcnt:])
    result = []
    for item in a:
        result.append(map(float,list(item)))
    return result
     
def sflist_norm(lista):
    flists = []
    if lista!=[]:
        for item in lista:
            flist = normalizelist(item)
            flists.append(flist) 
    return flists 

def Average(lista):
    r=0.0
    for i in lista:
        r+=float(i)
    result=r/len(lista)
    return result#round(result,3)

def averageLongtitude(lista):
    #IN:list matrix;OUT:longtitude average
    averageresults = []
    stdresults = []
    for item in zip(*lista):
        try:            
            averageresults.append(numpy.average(map(float,item)))
            stdresults.append(numpy.std(map(float,item)))#/len(item)
        except:
            print  'error:',item 
    return [averageresults,stdresults]
                
def draw(averageresult):
    x = range(len(averageresult[0]))    
#     print len(averageresult[0]),len(averageresult[1]),'\n',len(x)

#     plt.title(fname)
    try:
        fig1=plt.plot(x,averageresult[0][0:],marker='o')
        fig2=plt.plot(x,averageresult[1][0:],marker='x')
#         plt.show()
    except:
        pass
        
    plt.xlabel('Period')
    plt.ylabel('Fans')
    # plt.legend('Success','Failture')
#     plt.show()
#     figname=outfilepath+'.png'
    # pylab.savefig(figname)
#     plt.close()    
    print 'over'

def createFolder(foldname):
    if os.path.exists(foldname):
        pass 
    else:
        os.mkdir(foldname)
    return foldname

def add_metafig(metalist,figname,savefigure=False):
#     x = range(len(metalist[0][0])-3)
    lenstr = []
    plt.suptitle('Period:'+str(os.path.basename(figname)))
    i = 0
    for item in metalist:
        lenstr.append(len(item))
        y = averageLongtitude(item)[0]
        x = range(len(y))
        stdresults = None#metalist[1]
        fig1=plt.semilogy(x,y,marker='o')#,color='r'
        i+=1
    plt.legend(loc='upper left')
    plt.legend((str(lenstr[0]),str(lenstr[1]),str(lenstr[2])))#(fig1,),
    
    plt.xlabel('Period')
    plt.ylabel(os.path.basename(figname))
#     plt.show()
    if savefigure==False:
        plt.show()
    if savefigure==True:
        pylab.savefig(figname)
    plt.close()

def selectPercent(lista,percent,percentindex=1):
    result = []
    for item in lista:
        if str(item[percentindex])==str(percent):
            result.append(item)
    return result
    
def main_dynamics(feature_file='',kmeans_file=r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear',metacolcnt=3,kmeans_k=4,has_yerr=None,savefigure=False):
    if feature_file=='':
        casefeature_file = r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'#kmean-test2
    else:
        casefeature_file = feature_file 
        
    figfolder = createFolder(os.path.dirname(casefeature_file)+'\\fig')
    
    type_file = kmeans_file#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'
    sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed308.txt'
    #kmeans_k = 4
    
    #convert all file to list
    kmeans_list = selectPercent(gt.csv2list_new(type_file),1.0)#selectPercent(gt.normlistlist(gt.csv2list_new(type_file),2,'max'),1.0)#csv2list(type_file)
    casefeature_list = selectPercent(gt.csv2list_new((casefeature_file)),1.0)#selectPercent(gt.normlistlist(gt.csv2list_new((casefeature_file)),2,'max'),1.0)
    sucmeta_list = csv2list(sucmeta_file)
    
    #kmeans
#     kmeans_listemp = csv2list(type_file,'\t',1)
    metasflist = []#做元数据图用
    
    kmeans_list_sf = repair(kmeans_list,metacolcnt)
    typelist = k_means (kmeans_list_sf,kmeans_k)[0]
#     print typelist 
    idlists = []
    for item in typelist:  
        kmeans_byid = convertype2ids(item,kmeans_list,0) #2     
        idlists.append(kmeans_byid[0])#kmeans_list对应与item的位置2
        print 'kmeans_idlists==========:',connectlist(kmeans_byid[1],sucmeta_list,0,0)
    i = 0

    print idlists
    for group in idlists:
        try:
            print ''
            for it in group:
                if it[0] in []:#['3512192651209611', '3582675862318360', '3346786119768056', '3530377388318988', '3512261920248384', '3514517281419758', '3513485109002235', '3581083603782299', '3517880442503213', '3512320883828568', '3518037380208073', '3513786944578870', '3514712136139677', '3581619170077512', '3488298427647276', '3517122988859030', '3504590328512715', '3519173332815242', '3455798066008083', '3510725307943432', '3513472585606907', '3512467252803282', '3510108007340190', '3518192234385654', '3464705244725034', '3507543877973721', '3369278306978444']:
# ['3517807143042530', '3511850572507320', '3558246365665871', '3344204856189380', '3513054618763335','3510108007340190', '3513299721572710', '3512261920248384', '3514517281419758', '3367472590570390', '3517880442503213', '3518037380208073', '3370848283881337', '3512704620407286', '3369168951009868', '3514712136139677', '3348968449285798', '3517122988859030', '3519173332815242', '3518554643491425', '3455798066008083', '3512764419413627', '3504590328512715', '3513524855215093', '3504252389771186', '3344947345156943', '3513472585606907', '3507607539020444']:
                    group.remove(it)
        except:
            pass
        
        i+=1
        try:
            print len(group),len(casefeature_list)
            featureslist = gt.connectlist(group,casefeature_list,0,0,2)#若为未clear的为2
            featureslist = featureslist[0:]
            sflist = connectlist_sf(featureslist,sucmeta_list,0,0)            
            sflistall = departlist(sflist,'1','-1',0,2)#若为未clear的为4
            slist = sflistall[0]#sflist_norm(sflistall[0])#视情况决定是否需要归一化
            flist = sflistall[1]#sflist_norm(sflistall[1])#
#             print sflistall
            averageresults = averageLongtitude(slist)[0]
            averageresultf = averageLongtitude(flist)[0]
            print 'averageresults:',averageresults,'\naverageresultf',averageresultf
            stdresults = None
            stdresultf = None
            if has_yerr:
                stdresults = averageLongtitude(slist)[1]
                stdresultf = averageLongtitude(flist)[1]
            
            
            print 'ss:ff==',len(slist),':',len(flist)
#             draw([averageresults,averageresultf])
 
            x = range(len(averageresultf))    
#             print len(averageresultf),'\n',len(averageresults),len(x)
          
 
            colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
            try:
                plt.suptitle('Period:'+str(os.path.basename(casefeature_file)))
                plt.subplot((len(idlists)+1)/kmeans_k,kmeans_k if len(idlists)>1 else 1,i)
#                 plt.subplot(1,1,1)
                if len(averageresults)>0:
                    fig1=plt.errorbar(x,averageresults,marker='o',color='r',yerr=stdresults)#,label=str(len(slist)))
                    #,label=str(len(slist)))
                if len(averageresultf)>0:
#                     fig2=plt.errorbar(x,averageresultf,marker='x',color='b',yerr=stdresultf)#,label=str(len(flist)))#colorlist[i+1]
                    fig2=plt.errorbar(x,averageresultf,marker='x',color='b',yerr=stdresultf)#,label=str(len(slist)))
#                 plt.legend((fig1,fig2,),(str(len(slist)),str(len(flist)),))
#                 plt.legend(loc='upper left')
                plt.xlabel('S-'+str(len(slist))+':F-'+str(len(flist)))
#                 plt.ylabel(os.path.basename(casefeature_file))
#                 plt.show()
#                 plt.close()
            except Exception,e:
                print 'error in plotting:',e
                
            metatemp = flist
            metatemp.extend(slist)
            metasflist.append(metatemp)  
        except Exception,e:
            print 'error:',e
            

#     plt.title('Period:'+str(os.path.basename(casefeature_file)))
#     plt.legend(str(len(slist)),str(len(flist)))

    figname=figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+str(kmeans_k)+'_'+str(has_yerr)+'.png'

    if savefigure==False:
        plt.show()
    if savefigure==True:
        pylab.savefig(figname)
        print figname
    plt.close()
    
    if casefeature_file == kmeans_file:
        metafig = add_metafig(metasflist,figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+'meta.png',savefigure)          
    

#动态指标的处理  
if __name__=='__main__':
    filefolder = r'G:\HFS\WeiboData\HFSWeiboStatNet\NetCore'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20'#clear
    filep = r'G:\HFS\WeiboData\HFSWeiboStatNet\Net\.diameter'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20\.transitivity_undirectedmodeis0'#r'G:\HFS\WeiboData\Statistics\fanspeed\noNorm\fanspeed20bytime_leijia.txt'#.transitivity_undirectedmodeis0
    kmeans_file=r'G:\HFS\WeiboData\HFSWeiboStatNet\Net\.vcount'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20\.vcount'#.clear
    sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed308.txt'
#     kmeans_file = sucmeta_file
 
    k = 3
    main_dynamics(filep,kmeans_file,metacolcnt=2,kmeans_k=k,has_yerr=False,savefigure=False)
    main_dynamics(filep,kmeans_file,metacolcnt=3,kmeans_k=k,has_yerr=True,savefigure=False)
     
    main_dynamics(kmeans_file,kmeans_file,metacolcnt=1,kmeans_k=k,has_yerr=True,savefigure=True)
    main_dynamics(kmeans_file,kmeans_file,metacolcnt=1,kmeans_k=k,has_yerr=False,savefigure=True)
    for file in os.listdir(filefolder):
  
        filepath = filefolder+'\\'+file
        print filepath
        if os.path.isfile(filepath):
            main_dynamics(filepath,kmeans_file,metacolcnt=2,kmeans_k=k,has_yerr=True,savefigure=True)
            main_dynamics(filepath,kmeans_file,metacolcnt=2,kmeans_k=k,has_yerr=False,savefigure=True) 
         
    
#静态指标的处理  
def main_static(feature_file='',kmeans_file=r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear',metacolcnt=3,kmeans_k=4,has_yerr=None,savefigure=False):
    if feature_file=='':
        casefeature_file = r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'#kmean-test2
    else:
        casefeature_file = feature_file 
        
    figfolder = createFolder(os.path.dirname(casefeature_file)+'\\fig')
    
    type_file = kmeans_file#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'
    sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed.txt'
    #kmeans_k = 4
    
    #convert all file to list
    kmeans_list = csv2list(type_file)
    casefeature_list = csv2list(casefeature_file)
    sucmeta_list = csv2list(sucmeta_file)
    
    #kmeans
#     kmeans_listemp = csv2list(type_file,'\t',1)
    metasflist = []#做元数据图用
    
    kmeans_list_sf = repair(kmeans_list,metacolcnt)
    typelist = k_means (kmeans_list_sf,kmeans_k)[0]
    idlists = []
    for item in typelist:        
        kmeans_byid = convertype2ids(item,kmeans_list,0)      
        idlists.append(kmeans_byid[0])#kmeans_list对应与item的位置2
#         print 'kmeans-idlists==========:',kmeans_byid[1] 

    i = 0
    for group in idlists:
        i+=1
        try:
            featureslist = connectlist(group,casefeature_list,0,1)#若为未clear的为2
            featureslist = featureslist[1:]
            sflist = connectlist_sf(featureslist,sucmeta_list,0,1)            
            sflistall = departlist(sflist,'1','-1',0,2)#若为未clear的为4
            slist = sflistall[0]#sflist_norm(sflistall[0])#视情况决定是否需要归一化
            flist = sflistall[1]#sflist_norm(sflistall[1])#
            print 'slist',len(slist),'flist',len(flist)
            averageresults = averageLongtitude(slist)[0]
            averageresultf = averageLongtitude(flist)[0]
            print 'averageresults:',averageresults,'\naverageresultf',averageresultf
            stdresults = None
            stdresultf = None
            if has_yerr:
                stdresults = averageLongtitude(slist)[1]
                stdresultf = averageLongtitude(flist)[1]


            metasflist.append([averageresults,averageresultf])  
        except Exception,e:
            print 'error:',e
    
    print metasflist,zip(*metasflist)[0],'\n',zip(*metasflist)[1],len(zip(*metasflist)[0]),len(zip(*metasflist)[1]) 
#     metasflist[i][]       
    x = range(len(averageresultf))    
#             print len(averageresultf),'\n',len(averageresults),len(x)
 

    colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
    try:
        plt.suptitle('Period:'+str(os.path.basename(casefeature_file)))
        plt.subplot((len(idlists)+1)/3,3 if len(idlists)>2 else 1,i)
    #                 plt.subplot(1,1,1)
        if len(averageresults)>0:
            x = range(len(averageresults)-1)
            fig1=plt.errorbar(x,averageresults[1:],marker='o',color='r',yerr=stdresults)#,label=str(len(slist)))
            #,label=str(len(slist)))
        if len(averageresultf)>0:
            x = range(len(averageresultf)-1)
            fig2=plt.errorbar(x,averageresultf[1:],marker='x',color='b',yerr=stdresultf)#,label=str(len(slist)))
    #                 plt.legend((fig1,fig2,),(str(len(slist)),str(len(flist)),))
    #                 plt.legend(loc='upper left')
        plt.xlabel('S-'+str(len(slist))+':F-'+str(len(flist)))
    #                 plt.ylabel(os.path.basename(casefeature_file))
    #                 plt.show()
    #                 plt.close()
    except Exception,e:
        print 'error in plotting:',e
                
#     plt.title('Period:'+str(os.path.basename(casefeature_file)))
#     plt.legend(str(len(slist)),str(len(flist)))

    figname=figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+str(kmeans_k)+'_'+str(has_yerr)+'.png'

    if savefigure==False:
        plt.show()
    if savefigure==True:
        pylab.savefig(figname)
        print figname
    plt.close()
    
    if casefeature_file == kmeans_file:
        metafig = add_metafig(metasflist,figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+'meta.png',savefigure)          

# if __name__=='__main__':
#     filefolder = r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20'#clear
#     filep = r'G:\HFS\WeiboData\Statistics\20130622new\stat\stat_all_kmeans_test.txt'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20\.transitivity_undirectedmodeis0'#.transitivity_undirectedmodeis0
#     kmeans_file=r'G:\HFS\WeiboData\Statistics\20130622new\stat\stat_all_kmeans_test_typefile.txt'#.clear
#     sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed.txt'
# #     kmeans_file = sucmeta_file
# 
#     k = 3
# #     main(filep,kmeans_file,kmeans_k=k,has_yerr=True,savefigure=False)
#     main_static(filep,kmeans_file,metacolcnt=0,kmeans_k=k,has_yerr=False,savefigure=False)