#encoding=utf8

# -*- coding=utf-8 -*-
#encoding=utf8

##将案例先按kmeans分为几类，对每类进行正负例的平均进行比较
##input：案例某特征的list；kmeans分类根据的list,类数k；成败标记文件
##output：每类案例的正负均值 

#可调整之处：均带有adjust标记，如聚类的距离函数distance (x,y)
"V2 update:\
1 kmeans group function is seprated, which can get as a prameter\
2 in this version,we give the idlist groups seprated by the repost amount which have 5 groups"

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
            itemnew = []
            for it in item:
                if it=='nan' or it=='inf':
                    continue
                    it = 0
                itemnew.append(it)
            itemnew=[0] if len(itemnew)<1 else itemnew
            item = itemnew
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

def add_metafig(metalist,figname,savefigure=False,kmeans_k=3):
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
    for i in range(kmeans_k):
        lenstr.append(str(i))
#     plt.legend((str(lenstr[0]),str(lenstr[1]),str(lenstr[2])))#(fig1,),
    plt.legend(lenstr)#(fig1,),    
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
    
def main_dynamics(feature_file='',kmeans_file=r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear',metacolcnt=3,kmeans_k=4,has_yerr=None,savefigure=False,idlists=None):
    if feature_file=='':
        casefeature_file = r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'#kmean-test2
    else:
        casefeature_file = feature_file 
        
    figfolder = createFolder(os.path.dirname(casefeature_file)+'\\fig')
    
    sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed308_ordered.txt'#_delsmallr'G:\HFS\WeiboData\Statistics\meta_successed308.txt'
    
    #convert all file to list
    casefeature_list = selectPercent(gt.csv2list_new((casefeature_file)),1.0)#selectPercent(gt.normlistlist(gt.csv2list_new((casefeature_file)),2,'max'),1.0)
    sucmeta_list = csv2list(sucmeta_file)
    
    #kmeans
#     kmeans_listemp = csv2list(type_file,'\t',1)
    metasflist = []#做元数据图用

    if not idlists:    
        type_file = kmeans_file#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'
        kmeans_list = selectPercent(gt.csv2list_new(type_file),1.0)#selectPercent(gt.normlistlist(gt.csv2list_new(type_file),2,'max'),1.0)#csv2list(type_file)
        kmeans_list_sf = repair(kmeans_list,metacolcnt)
        typelist = k_means(kmeans_list_sf,kmeans_k)[0]
    #     print typelist 
        idlists = []
        for item in typelist:  
            kmeans_byid = convertype2ids(item,kmeans_list,0) #2     
            idlists.append(kmeans_byid[0])#kmeans_list对应与item的位置2
            print 'kmeans_idlists==========:',connectlist(kmeans_byid[1],sucmeta_list,0,0)

    i = 0
    print idlists
    for group in idlists:
#         try:
#             print ''
#             for it in group:
#                 if it[0] in []:#['3512192651209611', '3582675862318360', '3346786119768056', '3530377388318988', '3512261920248384', '3514517281419758', '3513485109002235', '3581083603782299', '3517880442503213', '3512320883828568', '3518037380208073', '3513786944578870', '3514712136139677', '3581619170077512', '3488298427647276', '3517122988859030', '3504590328512715', '3519173332815242', '3455798066008083', '3510725307943432', '3513472585606907', '3512467252803282', '3510108007340190', '3518192234385654', '3464705244725034', '3507543877973721', '3369278306978444']:
# # ['3517807143042530', '3511850572507320', '3558246365665871', '3344204856189380', '3513054618763335','3510108007340190', '3513299721572710', '3512261920248384', '3514517281419758', '3367472590570390', '3517880442503213', '3518037380208073', '3370848283881337', '3512704620407286', '3369168951009868', '3514712136139677', '3348968449285798', '3517122988859030', '3519173332815242', '3518554643491425', '3455798066008083', '3512764419413627', '3504590328512715', '3513524855215093', '3504252389771186', '3344947345156943', '3513472585606907', '3507607539020444']:
#                     group.remove(it)
#         except:
#             pass
        
        i+=1
        try:
            print len(group),len(casefeature_list)
            featureslist = gt.connectlist(group,casefeature_list,0,0,2, belistakey_suffix='.coc')#若为未clear的为2
            featureslist = featureslist[0:]
            sflist = connectlist_sf(featureslist,sucmeta_list,0,0)            
            sflistall = departlist(sflist,'1','-1',0,metacolcnt)#若为未clear的为4
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
        metafig = add_metafig(metasflist,figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+'meta.png',savefigure,kmeans_k)          
    

#动态指标的处理  
if __name__=='__main__':
    filefolder = r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\netcore'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20'#clear
#     filep = r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\statgiant\.diameter'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20\.transitivity_undirectedmodeis0'#r'G:\HFS\WeiboData\Statistics\fanspeed\noNorm\fanspeed20bytime_leijia.txt'#.transitivity_undirectedmodeis0
    kmeans_file=r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\meta308.txt'#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20\.vcount'#.clear
    sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed308_ordered.txt'
#     kmeans_file = sucmeta_file
    filexclude = ['.degreePowerLawFitD','.degreePowerLawFitL','.degreePowerLawFitp','.degreePowerLawFitxmin','.indegreePowerLawFitD','.indegreePowerLawFitL','.indegreePowerLawFitp','.indegreePowerLawFitxmin','.outdegreePowerLawFitD','.outdegreePowerLawFitL','.outdegreePowerLawFitp','.outdegreePowerLawFitxmin']
    
#     "<156"
#     idlista = [['3508278808120380.coc'],['3455798066008083.coc'],['3507662178094930.coc'],['3512722819965166.coc'],['3512192651209611.coc'],['3512681942436390.coc'],['3512764419413627.coc'],['3513472585606907.coc'],['3513477123020136.coc'],['3512723826282658.coc'],['3510947052234805.coc'],['3507607539020444.coc'],['3512261920248384.coc'],['3517122988859030.coc'],['3512661431797880.coc'],['3513054618763335.coc'],\
#                ['3513732681977292.coc'],['3514054737834781.coc'],['3512956526933221.coc'],['3510150776647546.coc'],['3512638635619787.coc'],['3512598488183675.coc'],['3514416295764354.coc'],['3512649558390590.coc'],['3507671015760502.coc'],['3370187475368354.coc'],['3582187498347368.coc'],['3369278306978444.coc'],['3344178035788881.coc'],['3356800646950624.coc'],['3358716283896811.coc'],['3369168951009868.coc'],['3370848283881337.coc'],['3513665519425522.coc'],['3346786119768056.coc'],['3514087109494803.coc'],['3514202721379789.coc'],['3345283975088597.coc'],['3345341063735706.coc'],['3464854910351381.coc'],['3512564992138128.coc'],['3506858382217257.coc'],['3519173332815242.coc'],['3510108007340190.coc'],['3513651849587189.coc'],['3343901805640480.coc'],['3344631446304834.coc'],['3514415834044554.coc'],['3518018271478014.coc'],['3513299721572710.coc'],['3513008209201946.coc'],['3346361808667289.coc'],['3512586882317341.coc'],['3512965133226559.coc']]
#     "<1000"
#     idlistb = [['3519104033490770.coc'],['3514712136139677.coc'],['3512665513423714.coc'],['3479304597792637.coc'],['3344605319892924.coc'],['3558226899367894.coc'],['3513747752676493.coc'],['3512288331952105.coc'],['3513738009766461.coc'],['3367472590570390.coc'],['3517807143042530.coc'],['3514574454119360.coc'],['3348202183182981.coc'],['3512260125547589.coc'],['3464345767606270.coc'],['3512225370844164.coc'],['3514216143529145.coc'],['3514287944897392.coc'],['3513797614859184.coc'],['3558246365665871.coc'],['3368776344558652.coc'],['3371320634873316.coc'],['3345672913760585.coc'],['3489137279411789.coc'],['3516669001647040.coc'],['3513733382475662.coc'],['3512755191392034.coc'],['3512704620407286.coc'],['3512673221800564.coc'],['3514721241880789.coc'],['3370126999415642.coc'],['3508035156247306.coc'],['3512767539668338.coc'],['3519083113115752.coc'],['3514047335033747.coc'],['3507543877973721.coc'],['3512703459106709.coc'],['3514229367981237.coc'],['3512367365458914.coc'],['3347020320429724.coc'],['3518037380208073.coc'],['3511983850692431.coc'],['3514207033581502.coc'],['3512557425820703.coc'],['3582182788767024.coc'],['3363356413828548.coc'],['3512343789230980.coc'],['3345716399866144.coc'],['3513457897170153.coc'],['3513732346670068.coc'],['3513684632475119.coc'],['3511566865123223.coc'],['3518192234385654.coc'],['3511312581651670.coc'],['3371095383919407.coc'],['3464705244725034.coc'],['3513353957580369.coc'],['3512731699677616.coc'],['3343828640519374.coc'],['3369886157847997.coc'],['3560576217397500.coc'],['3489084565802342.coc'],['3514061079292261.coc'],['3512751521888667.coc'],['3356881155164816.coc'],['3344204856189380.coc'],['3344947345156943.coc'],['3514517281419758.coc'],['3370242220657016.coc'],['3511950958712857.coc'],['3464119577576394.coc'],['3514083762166772.coc'],['3346041476969222.coc'],['3347122272192199.coc'],['3581866814344587.coc'],['3514112790871598.coc'],['3581830525479047.coc'],['3513821030864621.coc'],['3512371488398141.coc'],['3513708254962820.coc'],['3343744527348953.coc'],['3512320883828568.coc'],['3582141898089800.coc'],['3513761854134169.coc'],['3518374388216553.coc'],['3346337229522031.coc'],['3512693292397369.coc'],['3348968449285798.coc'],['3512387527089684.coc'],['3344251610958634.coc'],['3512944329886751.coc'],['3502012958680889.coc'],['3512593978848674.coc'],['3516665499181717.coc'],['3517378317183344.coc'],['3514058537701986.coc'],['3514448201592653.coc'],['3511850572507320.coc'],['3560817721137817.coc'],['3367269376657349.coc'],['3560740088421769.coc'],['3512631170591260.coc'],['3371353334212131.coc'],['3512224943485819.coc'],['3513027452433705.coc'],['3344200536278976.coc'],['3371247520439214.coc'],['3513262479906932.coc'],['3504252389771186.coc'],['3581029350321431.coc'],['3370943210679558.coc'],['3514725432613389.coc'],['3513737430369962.coc'],['3512346909867070.coc'],['3513871362653946.coc'],['3554653150827755.coc'],['3514793086497284.coc'],['3514074715123866.coc'],['3517351528480026.coc'],['3513524855215093.coc'],['3512391692548104.coc'],['3512220862117440.coc'],['3514409529974363.coc'],['3512207839497881.coc'],['3518554643491425.coc'],['3514484653860068.coc'],['3345285264425627.coc'],['3514033757910710.coc'],['3367745213249038.coc'],['3517374844008722.coc'],['3513786944578870.coc'],['3511953756712121.coc'],['3345943583930879.coc'],['3431812342844428.coc'],['3517880442503213.coc'],['3489743314991378.coc'],['3518889122334776.coc'],['3513795827993634.coc'],['3581874289297524.coc'],['3513485109002235.coc'],['3513170419502667.coc'],['3343770315269085.coc'],['3371270547035409.coc']]

    "<86"
    idlista = [['3508278808120380.coc'],['3455798066008083.coc'],['3507662178094930.coc'],['3512722819965166.coc'],['3512192651209611.coc'],['3512681942436390.coc'],['3512764419413627.coc'],['3513472585606907.coc'],['3513477123020136.coc'],['3512723826282658.coc'],['3510947052234805.coc'],['3507607539020444.coc'],['3512261920248384.coc'],['3517122988859030.coc'],['3512661431797880.coc'],['3513054618763335.coc'],\
               ]
    "<1000--21:158"
    #['3513732681977292.coc'],['3514054737834781.coc'],['3512956526933221.coc'],['3510150776647546.coc'],['3512638635619787.coc'],['3512598488183675.coc'],['3514416295764354.coc'],['3512649558390590.coc'],['3507671015760502.coc'],['3370187475368354.coc'],['3582187498347368.coc'],['3369278306978444.coc'],['3344178035788881.coc'],['3356800646950624.coc'],['3358716283896811.coc'],['3369168951009868.coc'],['3370848283881337.coc'],['3513665519425522.coc'],['3346786119768056.coc'],['3514087109494803.coc'],['3514202721379789.coc'],['3345283975088597.coc'],['3345341063735706.coc'],['3464854910351381.coc'],['3512564992138128.coc'],['3506858382217257.coc'],['3519173332815242.coc'],['3510108007340190.coc'],['3513651849587189.coc'],['3343901805640480.coc'],['3344631446304834.coc'],['3514415834044554.coc'],['3518018271478014.coc'],['3513299721572710.coc'],['3513008209201946.coc'],['3346361808667289.coc'],['3512586882317341.coc'],['3512965133226559.coc']
    idlistb = [['3513732681977292.coc'],['3514054737834781.coc'],['3512956526933221.coc'],['3510150776647546.coc'],['3512638635619787.coc'],['3512598488183675.coc'],['3514416295764354.coc'],['3512649558390590.coc'],['3507671015760502.coc'],['3370187475368354.coc'],['3582187498347368.coc'],['3369278306978444.coc'],['3344178035788881.coc'],['3356800646950624.coc'],['3358716283896811.coc'],['3369168951009868.coc'],['3370848283881337.coc'],['3513665519425522.coc'],['3346786119768056.coc'],['3514087109494803.coc'],['3514202721379789.coc'],['3345283975088597.coc'],['3345341063735706.coc'],['3464854910351381.coc'],['3512564992138128.coc'],['3506858382217257.coc'],['3519173332815242.coc'],['3510108007340190.coc'],['3513651849587189.coc'],['3343901805640480.coc'],['3344631446304834.coc'],['3514415834044554.coc'],['3518018271478014.coc'],['3513299721572710.coc'],['3513008209201946.coc'],['3346361808667289.coc'],['3512586882317341.coc'],['3512965133226559.coc'],['3519104033490770.coc'],['3514712136139677.coc'],['3512665513423714.coc'],['3479304597792637.coc'],['3344605319892924.coc'],['3558226899367894.coc'],['3513747752676493.coc'],['3512288331952105.coc'],['3513738009766461.coc'],['3367472590570390.coc'],['3517807143042530.coc'],['3514574454119360.coc'],['3348202183182981.coc'],['3512260125547589.coc'],['3464345767606270.coc'],['3512225370844164.coc'],['3514216143529145.coc'],['3514287944897392.coc'],['3513797614859184.coc'],['3558246365665871.coc'],['3368776344558652.coc'],['3371320634873316.coc'],['3345672913760585.coc'],['3489137279411789.coc'],['3516669001647040.coc'],['3513733382475662.coc'],['3512755191392034.coc'],['3512704620407286.coc'],['3512673221800564.coc'],['3514721241880789.coc'],['3370126999415642.coc'],['3508035156247306.coc'],['3512767539668338.coc'],['3519083113115752.coc'],['3514047335033747.coc'],['3507543877973721.coc'],['3512703459106709.coc'],['3514229367981237.coc'],['3512367365458914.coc'],['3347020320429724.coc'],['3518037380208073.coc'],['3511983850692431.coc'],['3514207033581502.coc'],['3512557425820703.coc'],['3582182788767024.coc'],['3363356413828548.coc'],['3512343789230980.coc'],['3345716399866144.coc'],['3513457897170153.coc'],['3513732346670068.coc'],['3513684632475119.coc'],['3511566865123223.coc'],['3518192234385654.coc'],['3511312581651670.coc'],['3371095383919407.coc'],['3464705244725034.coc'],['3513353957580369.coc'],['3512731699677616.coc'],['3343828640519374.coc'],['3369886157847997.coc'],['3560576217397500.coc'],['3489084565802342.coc'],['3514061079292261.coc'],['3512751521888667.coc'],['3356881155164816.coc'],['3344204856189380.coc'],['3344947345156943.coc'],['3514517281419758.coc'],['3370242220657016.coc'],['3511950958712857.coc'],['3464119577576394.coc'],['3514083762166772.coc'],['3346041476969222.coc'],['3347122272192199.coc'],['3581866814344587.coc'],['3514112790871598.coc'],['3581830525479047.coc'],['3513821030864621.coc'],['3512371488398141.coc'],['3513708254962820.coc'],['3343744527348953.coc'],['3512320883828568.coc'],['3582141898089800.coc'],['3513761854134169.coc'],['3518374388216553.coc'],['3346337229522031.coc'],['3512693292397369.coc'],['3348968449285798.coc'],['3512387527089684.coc'],['3344251610958634.coc'],['3512944329886751.coc'],['3502012958680889.coc'],['3512593978848674.coc'],['3516665499181717.coc'],['3517378317183344.coc'],['3514058537701986.coc'],['3514448201592653.coc'],['3511850572507320.coc'],['3560817721137817.coc'],['3367269376657349.coc'],['3560740088421769.coc'],['3512631170591260.coc'],['3371353334212131.coc'],['3512224943485819.coc'],['3513027452433705.coc'],['3344200536278976.coc'],['3371247520439214.coc'],['3513262479906932.coc'],['3504252389771186.coc'],['3581029350321431.coc'],['3370943210679558.coc'],['3514725432613389.coc'],['3513737430369962.coc'],['3512346909867070.coc'],['3513871362653946.coc'],['3554653150827755.coc'],['3514793086497284.coc'],['3514074715123866.coc'],['3517351528480026.coc'],['3513524855215093.coc'],['3512391692548104.coc'],['3512220862117440.coc'],['3514409529974363.coc'],['3512207839497881.coc'],['3518554643491425.coc'],['3514484653860068.coc'],['3345285264425627.coc'],['3514033757910710.coc'],['3367745213249038.coc'],['3517374844008722.coc'],['3513786944578870.coc'],['3511953756712121.coc'],['3345943583930879.coc'],['3431812342844428.coc'],['3517880442503213.coc'],['3489743314991378.coc'],['3518889122334776.coc'],['3513795827993634.coc'],['3581874289297524.coc'],['3513485109002235.coc'],['3513170419502667.coc'],['3343770315269085.coc'],['3371270547035409.coc']]

    
    "<14000--18:73"
    idlistc = [['3367696964256940.coc'],['3367612479773055.coc'],['3559702166286240.coc'],['3512362638453577.coc'],['3512228487367164.coc'],['3343408888337055.coc'],['3512568150742963.coc'],['3586578800272420.coc'],['3506429741008735.coc'],['3517300592201464.coc'],['3513762864278058.coc'],['3488816159775764.coc'],['3524708206766693.coc'],['3429328731908395.coc'],['3443510244101746.coc'],['3512681036745319.coc'],['3369068477131339.coc'],['3513671424206849.coc'],['3517012276302505.coc'],['3430490826135856.coc'],['3510725307943432.coc'],['3512651290338120.coc'],['3513009873831054.coc'],['3489558450803933.coc'],['3488842327677557.coc'],['3516941173798600.coc'],['3482476628770294.coc'],['3494489962794555.coc'],['3584031612169073.coc'],['3507953124535865.coc'],['3517263451924815.coc'],['3518876082450868.coc'],['3477892518336048.coc'],['3512228747718413.coc'],['3518924249182559.coc'],['3474593925041592.coc'],['3512365087957339.coc'],['3512027492461885.coc'],['3512227489306229.coc'],['3504590328512715.coc'],['3513361775532449.coc'],['3494152501426914.coc'],['3581083603782299.coc'],['3558051988500239.coc'],['3346646386115884.coc'],['3516368059904172.coc'],['3555764297312504.coc'],['3581880941207958.coc'],['3513435633981893.coc'],['3513822322955028.coc'],['3513784817583701.coc'],['3524644067598719.coc'],['3487580337482630.coc'],['3506067055046956.coc'],['3553712926158978.coc'],['3512642997758015.coc'],['3464210090965697.coc'],['3365285795421520.coc'],['3512570026130918.coc'],['3488968551195859.coc'],['3557998624957120.coc'],['3512725802294904.coc'],['3466248535071107.coc'],['3505779032316582.coc'],['3435254721283066.coc'],['3512675969861915.coc'],['3514069312670878.coc'],['3497540102476487.coc'],['3481617253114874.coc'],['3512633108174568.coc'],['3376346833399388.coc'],['3580448250376461.coc'],['3512597141818367.coc'],['3511661669904764.coc'],['3506978007041225.coc'],['3489462720299193.coc'],['3513739158587782.coc'],['3512846728070462.coc'],['3431235789559517.coc'],['3508256699661280.coc'],['3506452277059335.coc'],['3512241854636226.coc'],['3516311047353690.coc'],['3512265636249516.coc'],['3553179003983639.coc'],['3440833729050905.coc'],['3509097477346600.coc'],['3527628663048707.coc'],['3553858343177336.coc'],['3511585550779014.coc'],['3513645327614227.coc'],['3524530280557536.coc']]
    "<110000--5:17"
    idlistd = [['3512467252803282.coc'],['3385409596124201.coc'],['3528870550951587.coc'],['3512407622499003.coc'],['3550786464348915.coc'],['3509751591473526.coc'],['3582675862318360.coc'],['3488298427647276.coc'],['3383683287711280.coc'],['3518734310070023.coc'],['3513330587596297.coc'],['3590965983991304.coc'],['3591398001249115.coc'],['3506843546638885.coc'],['3497517021192038.coc'],['3503979856274851.coc'],['3581833155041015.coc'],['3530377388318988.coc'],['3591379278050636.coc'],['3581619170077512.coc'],['3573993774557103.coc'],['3571815701857951.coc']]
    "<250000"
    idliste = [['3512409568346880.coc']]#,['3552170865080083.coc']]
    
#     idlistb.extend(idlista)
#     idlistd.extend(idliste)
    idlist = [idlistb,idlistc,idlistd]#] #idlista,,idliste
    k = len(idlist)#3
#     main_dynamics(filep,kmeans_file,metacolcnt=2,kmeans_k=k,has_yerr=False,savefigure=False)
#     main_dynamics(filep,kmeans_file,metacolcnt=3,kmeans_k=k,has_yerr=True,savefigure=False)
     
#     main_dynamics(kmeans_file,kmeans_file,metacolcnt=2,kmeans_k=k,has_yerr=True,savefigure=True)
#     main_dynamics(kmeans_file,kmeans_file,metacolcnt=2,kmeans_k=k,has_yerr=False,savefigure=True,idlists=idlist)
    for file in os.listdir(filefolder):
   
        filepath = filefolder+'\\'+file
        print filepath
        
        if os.path.isfile(filepath) and file not in filexclude:
#             main_dynamics(filepath,kmeans_file,metacolcnt=3,kmeans_k=k,has_yerr=True,savefigure=True,idlists=idlist)
            main_dynamics(filepath,kmeans_file,metacolcnt=3,kmeans_k=k,has_yerr=False,savefigure=True,idlists=idlist) 
         
    
#静态指标的处理  
# def main_static(feature_file='',kmeans_file=r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear',metacolcnt=3,kmeans_k=4,has_yerr=None,savefigure=False):
#     if feature_file=='':
#         casefeature_file = r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'#kmean-test2
#     else:
#         casefeature_file = feature_file 
#         
#     figfolder = createFolder(os.path.dirname(casefeature_file)+'\\fig')
#     
#     type_file = kmeans_file#r'G:\HFS\WeiboData\Statistics\HFSWeibo_MultiDiGraph\network\avguser20clear\.vcount.clear'
#     sucmeta_file = r'G:\HFS\WeiboData\Statistics\meta_successed.txt'
#     #kmeans_k = 4
#     
#     #convert all file to list
#     kmeans_list = csv2list(type_file)
#     casefeature_list = csv2list(casefeature_file)
#     sucmeta_list = csv2list(sucmeta_file)
#     
#     #kmeans
# #     kmeans_listemp = csv2list(type_file,'\t',1)
#     metasflist = []#做元数据图用
#     
#     kmeans_list_sf = repair(kmeans_list,metacolcnt)
#     typelist = k_means (kmeans_list_sf,kmeans_k)[0]
#     idlists = []
#     for item in typelist:        
#         kmeans_byid = convertype2ids(item,kmeans_list,0)      
#         idlists.append(kmeans_byid[0])#kmeans_list对应与item的位置2
# #         print 'kmeans-idlists==========:',kmeans_byid[1] 
# 
#     i = 0
#     for group in idlists:
#         i+=1
#         try:
#             featureslist = connectlist(group,casefeature_list,0,1)#若为未clear的为2
#             featureslist = featureslist[1:]
#             sflist = connectlist_sf(featureslist,sucmeta_list,0,1)            
#             sflistall = departlist(sflist,'1','-1',0,2)#若为未clear的为4
#             slist = sflistall[0]#sflist_norm(sflistall[0])#视情况决定是否需要归一化
#             flist = sflistall[1]#sflist_norm(sflistall[1])#
#             print 'slist',len(slist),'flist',len(flist)
#             averageresults = averageLongtitude(slist)[0]
#             averageresultf = averageLongtitude(flist)[0]
#             print 'averageresults:',averageresults,'\naverageresultf',averageresultf
#             stdresults = None
#             stdresultf = None
#             if has_yerr:
#                 stdresults = averageLongtitude(slist)[1]
#                 stdresultf = averageLongtitude(flist)[1]
# 
# 
#             metasflist.append([averageresults,averageresultf])  
#         except Exception,e:
#             print 'error:',e
#     
#     print metasflist,zip(*metasflist)[0],'\n',zip(*metasflist)[1],len(zip(*metasflist)[0]),len(zip(*metasflist)[1]) 
# #     metasflist[i][]       
#     x = range(len(averageresultf))    
# #             print len(averageresultf),'\n',len(averageresults),len(x)
#  
# 
#     colorlist = list('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')#['r','g','b',]
#     try:
#         plt.suptitle('Period:'+str(os.path.basename(casefeature_file)))
#         plt.subplot((len(idlists)+1)/3,3 if len(idlists)>2 else 1,i)
#     #                 plt.subplot(1,1,1)
#         if len(averageresults)>0:
#             x = range(len(averageresults)-1)
#             fig1=plt.errorbar(x,averageresults[1:],marker='o',color='r',yerr=stdresults)#,label=str(len(slist)))
#             #,label=str(len(slist)))
#         if len(averageresultf)>0:
#             x = range(len(averageresultf)-1)
#             fig2=plt.errorbar(x,averageresultf[1:],marker='x',color='b',yerr=stdresultf)#,label=str(len(slist)))
#     #                 plt.legend((fig1,fig2,),(str(len(slist)),str(len(flist)),))
#     #                 plt.legend(loc='upper left')
#         plt.xlabel('S-'+str(len(slist))+':F-'+str(len(flist)))
#     #                 plt.ylabel(os.path.basename(casefeature_file))
#     #                 plt.show()
#     #                 plt.close()
#     except Exception,e:
#         print 'error in plotting:',e
#                 
# #     plt.title('Period:'+str(os.path.basename(casefeature_file)))
# #     plt.legend(str(len(slist)),str(len(flist)))
# 
#     figname=figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+str(kmeans_k)+'_'+str(has_yerr)+'.png'
# 
#     if savefigure==False:
#         plt.show()
#     if savefigure==True:
#         pylab.savefig(figname)
#         print figname
#     plt.close()
#     
#     if casefeature_file == kmeans_file:
#         metafig = add_metafig(metasflist,figfolder+'\\'+str(os.path.basename(kmeans_file))+'_'+str(os.path.basename(casefeature_file))+'_'+'meta.png',savefigure)          

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
    
    