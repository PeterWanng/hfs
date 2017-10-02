#coding=utf8
a = ['a','b',]
b = ['a','b','c']
c = ['c','d','e']
import numpy as np

def get_wordcount(wordlist):
   wordcnt ={}
   for i in wordlist:
       if i in wordcnt:
           wordcnt[i] += 1
       else:
           wordcnt[i] = 1
   worddict = wordcnt.items()
   worddict.sort(key=lambda a: -a[1])
   for word,cnt in worddict:
       word.encode('gbk'), cnt
   return wordcnt

print get_wordcount(c)
    
er
#inout:文本文件输入出位置，列名list, 标签位置列名
#output：svm格式的文本
import sys
sys.path.append('../..')
from tools import commontools as gtf
import random
import os  
import numpy
gt = gtf()

import sklearn as skl
from sklearn import datasets, linear_model
import ntpath


def getsindex(sflist,flag):
    result=[]
#     f = gt.csv2list(filep)
    linecnt = 0
    flagindex = 0
    for line in sflist:      
        linecnt+=1
        if line[1]==flag and ~flagindex:
            flagindex = linecnt
            break
        else:
            pass  
    
    result.append(flagindex)        
    result.append(len(sflist))        
            
    return result

def list2libsvmformat(lista,percindex=2):
    listResult = ''
    for item in lista:
        i = 0
        litem = str(item[1])+'\t'#item[0]+'\t'+
        for it in item[percindex+1:]:
            i+=1
            litem+=str(it)+'\t'#(str(i)+':'+
            listResult+=litem
            litem = ''
        listResult+='\n' 
    return listResult.replace('\t\n','\n')
    
def selectPercent(lista,percent,percentindex=1):
    result = []
    for item in lista:
        if str(item[percentindex])==str(percent):
            result.append(item)
    return result
    
def sampline(listfeature,svmoutfile,lista,listb,listc,listd):
    traindata = []
    testdata = []
            
#     f = open(svmfile)
#     fw1 = open(svmoutfile+'.scl','w')
#         fw2 = open(svmoutfile+'.ts','w')
    lincnt = 0
    for line in listfeature:
        lincnt+=1
        if (lincnt in lista) or (lincnt in listc):
            traindata.append(line)
            continue        
#     f.close()
#     f = open(svmfile)    
    lincnt = 0
    for line in listfeature:
        lincnt+=1
        if (lincnt in listd) or (lincnt in listb):
            testdata.append(line)
            continue
#     f.close()
    return [traindata,testdata]    
    
    
def sample2(lista,listsindex,listfindex,fratio,sratio,tssratio,tsfratio,svmoutfile):
    sucnt = len(listsindex)
    failcnt = len(listfindex)
    
    scnt=int(round(sucnt*sratio))
    fcnt=int(round(failcnt*fratio))#scnt#
    stscnt=int(round(sucnt*tssratio-0.01))
    ftscnt=int(round(failcnt*tsfratio-0.01))#stscnt#
#         print '===============',scnt, stscnt
    s = gt.divideList(listsindex, scnt, stscnt)
    f = gt.divideList(listfindex, fcnt, ftscnt)
 
    datasample = sampline(lista,svmoutfile,s[0],s[1],f[0],f[1])
        
    return datasample#[s,f]
    
    
def delcoc(lista,delstr,colindex):
    result=[]
    for item in lista:
        item0 = str(item[colindex]).replace(delstr,'')
        it = list(item[1:])
        it.insert(0,item0)
        result.append(it)
    return result

def svmfeatures(vd,filename,rangemin,rangemax):
    result_tr = []
    result_ts = []
    vdsf = gt.connectlist_sf(vd, sameposition_be=0,suclista=gt.csv2list(r'G:\HFS\WeiboData\Statistics\meta_successed308.txt'),rangemin=rangemin,rangemax=rangemax,keysuffix=None)
    

#     vdsvm = list2libsvmformat(vdsf)
    svmfilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat\vd'+filename+'_km.skl'
#     
#     fw = open(svmfilepath,'w')
#     fw.write(vdsvm)
#     fw.close()
    for i in range(1,11):
        svmoutfilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat\vd'+filename+'_km_'+str(i)+'.skl' 
        sfindex = getsindex(vdsf,'-1')
        scnt = sfindex[0]
        fcnt = sfindex[1]
        print '=============================',scnt,fcnt
#         a.sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.8*scnt)/(fcnt-scnt),sratio=(0.8*scnt)/scnt,tssratio=(0.2*scnt)/scnt,tsfratio=(0.2*scnt)/(fcnt-scnt),svmoutfile=svmoutfilepath)
#         a.sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=44/262.0,sratio=44/48.0,tssratio=4/48.0,tsfratio=4/262.0,svmoutfile=svmoutfilepath)
#         resultone = sample2(svmfilepath,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.8*fcnt)/(fcnt),sratio=(0.8*scnt)/scnt,tssratio=(0.2*scnt)/scnt,tsfratio=(0.2*fcnt)/(fcnt),svmoutfile=svmoutfilepath)
        resultone = sample2(vdsf,listsindex=range(1,scnt),listfindex=range(scnt,fcnt),fratio=(0.8*fcnt)/(fcnt),sratio=(0.8*scnt)/scnt,tssratio=(0.2*scnt)/scnt,tsfratio=(0.2*fcnt)/(fcnt),svmoutfile=svmoutfilepath)
        result_tr.append(resultone[0])
        result_ts.append(resultone[1])
    print '=============================',scnt,fcnt
    return [result_tr,result_ts]

"======================================================================================================================="
def generateSklData(featureFilepath,perc):
    "IN:feature file path"
    "OUT:skl training and testing data list"
    filepath = featureFilepath
    filep = os.path.basename(filepath)
    rangemin=0
    rangemax=100000
    
    dataTrain = []
    dataTest = []
    
    featlist = gt.csv2list_new(filepath,passmetacol=2,convertype=float,nan2num=True)
    featlist = delcoc(featlist,delstr='.coc',colindex=0)
    vd = gt.normlistlist(selectPercent(featlist,perc,1),metacolcount=2,sumormax='max')

    [dataTrain,dataTest] = svmfeatures(vd=vd,filename=filep,rangemin=rangemin,rangemax=rangemax)#filename=str(k)+'_'+str(perc)+'_'+filep
#     feature = gt.normlistlist(selectPercent(gt.csv2list_new(filepath,passmetacol=2,convertype=float,nan2num=True),perc,1),metacolcount=2,sumormax='max')#gt.csv2list_new(filepath)#
#     vd = gt.connectlist(vd, feature, 0, 0, passcol=2)
    print 'finished one:',filep
    
    return [dataTrain,dataTest]
"======================================================================================================================="

def getarget(dataset,xrange,yrange):
    "IN: a listlist with "
    "OUT: "
    temp_tr = zip(*dataset[0])
    temp_ts = zip(*dataset[1])
    xtr = temp_tr[xrange[0]:xrange[1]]
    ytr = temp_tr[yrange[0]:]
    
    xtr = zip(*xtr)
    
    
    xts = temp_ts[xrange[0]:xrange[1]]
    yts = temp_ts[yrange[0]:]
    
    xts = zip(*xts)
    return [numpy.array(xtr),numpy.array(xts),numpy.array(ytr),numpy.array(yts)]# [xtr,xts,ytr,yts]
       
def regression(dataset,regressionType):
    "IN:dataset which include training and testing data; regression type:1-linear;2-logistic;"
    "OUT:predict Result"
    result = []
    
    
    dataset = getarget(dataset,[0,1],[1,])
    
#     test_index = -61 
    diabetes_x_train = dataset[0]#[:test_index] #训练样本
    diabetes_x_test = dataset[1]#[test_index:] #检测样本
    
    diabetes_y_train = dataset[2]#diabetes.target[:test_index]
    diabetes_y_test = dataset[3]#diabetes.target[test_index:]
     
    # print len(diabetes_y_test),len(diabetes_x_test),len(diabetes_x_train),len(diabetes_y_train),#
     
    regr = linear_model.LinearRegression()
#     regr = linear_model.LogisticRegression()
#     regr = linear_model.Lasso()
     
    regr.fit(diabetes_x_train, diabetes_y_train)
    predict_result = regr.predict(X=diabetes_x_test)
    print predict_result
    print diabetes_y_test
    wr = 0
    rr = 0    

    from sklearn import metrics
    y = np.array(diabetes_y_test)#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
    pred = np.array(predict_result)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    aucvalue = metrics.auc(fpr, tpr)
    print sklfile,"AUC:",aucvalue
    
    return result
    
if __name__=='__main__':
    
    "One feature"
    featureFilepath = r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\test\statcore\.bifansumavg'
    dataset = generateSklData(featureFilepath,perc='1.0')
    print dataset
    predictResult = regression(dataset,regressionType=1)
    
    
    