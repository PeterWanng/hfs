#encoding=utf8
import sys
sys.path.append("G:\\MyCode\\MyCode\\")
from tools import commontools as gtf
import os
import csv
import numpy
import random
gt = gtf()


"============================================================================================"
import pylab as pl
import numpy as np
import sklearn as skl
from sklearn import datasets, linear_model
import ntpath


def load_diabetes():
    txtfile = r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN\SKLearn\vd0_0.1_.assortativitydegreedirectedisfalse_km_1.svm.tr'
    txtfilets = r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN\SKLearn\ts.txt'
#     base_dir = ntpath.join(ntpath.dirname(__file__), 'data')
#     print base_dir
#     data = np.loadtxt(ntpath.join(base_dir, r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN\SKLearn\diabetes_data.csv.gz'))
#     target = np.loadtxt(ntpath.join(base_dir, r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN\SKLearn\diabetes_target.csv.gz'))
#     data = gt.csv2list_new(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN\SKLearn\.diameter',passmetacol=2,convertype=int)
#     data = loadtxtdata(r'G:\HFS\WeiboData\HFSWeiboStatNet\Svm\IN\SKLearn\.diameter')
    data = numpy.genfromtxt(fname=txtfile, dtype=int, delimiter='\t', usecols = range(2,20),skip_header=0, skip_footer=0)#(1, 4, 5)
    target = numpy.genfromtxt(fname=txtfilets, dtype=int)#(1, 4, 5)
    print data
    return datasets.base.Bunch(data=data, target=target)

def load_diabetes2(datafilepath,txtfilets = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\target.txt'):
    txtfile = datafilepath#r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl'
    txtfilets = txtfilets#r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\target.txt'
#     target = zip(*(gt.csv2list_new(datafilepath, delimitertype='excel_tab')))[0] 
    data = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(1,21),skip_header=0, skip_footer=0)#(1, 4, 5)
#     target = numpy.genfromtxt(fname=txtfilets, dtype=float)#(1, 4, 5)
    target = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(0,1),skip_header=0, skip_footer=0)#(1, 4, 5)zip(*(data))[0]
    return datasets.base.Bunch(data=data, target=target)

def deal_result(lista,experiments_times):
    "IN:list [filename,accucy];experiments_times for each feature"
    "OUT:average accucy of each feature"
    listb = zip(*(lista))
    i = 1
    tempone = []
    while i< len(listb[0]):
        tempone.append(listb[1][i-1])
        if i%experiments_times==0:
            print len(tempone),'\t',listb[0][i-1],'\t',numpy.average(tempone)
            tempone = []
        
        i+=1
    
def skldata(sklfile):
    diabetes = load_diabetes2(sklfile,txtfilets = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\target.txt')#(r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl') #载入数据datasets.
     
    diabetes_x = diabetes.data#[:, np.newaxis]
    diabetes_x_temp = diabetes_x#[:, :, 2]
     
    diabetes_x_train = diabetes_x_temp[:-54] #训练样本
    diabetes_x_test = diabetes_x_temp[-54:] #检测样本
    diabetes_y_train = diabetes.target[:-54]
    diabetes_y_test = diabetes.target[-54:]
     
#     print len(diabetes_y_test),len(diabetes_x_test),len(diabetes_x_train),len(diabetes_y_train),#
     
    regr = linear_model.LinearRegression()
#     regr = linear_model.LogisticRegression()
     
    regr.fit(diabetes_x_train, diabetes_y_train)
    predict_result = regr.predict(X=diabetes_x_test)
    wr = 0
    rr = 0    
#     print numpy.array(predict_result),'\n',diabetes_y_test
    
    from sklearn import metrics
    y = np.array(diabetes_y_test)#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
    pred = np.array(predict_result)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    print metrics.auc(fpr, tpr)

    for a,b in zip(*(predict_result,diabetes_y_test)):
        
        if a*b>=0:
            rr+=1
    
#     for item in predict_result[0:4]:
#         if float(item)>=0:
#           wr+=1 
#     for item in predict_result[4:8]:
#         if float(item)<0:
#           rr+=1 
#     print sklfile,' #Accucy:\t',float(wr+rr)/len(predict_result)
    return [sklfile,float(wr+rr)/len(predict_result)]
    # print 'Coefficients :\n', regr.coef_ 
    # print ("Residual sum of square: %.2f" %np.mean((regr.predict(diabetes_x_test) - diabetes_y_test) ** 2)) 
    # print ("variance score: %.2f" % regr.score(diabetes_x_test, diabetes_y_test))
     
    # pl.scatter(diabetes_x_test,diabetes_y_test, color = 'black')
    # pl.plot(diabetes_x_test, regr.predict(diabetes_x_test),color='blue',linewidth = 3)
    # pl.xticks(())
    # pl.yticks(())
    # pl.show()


res = skldata(sklfolder+'\\'+filep)
result.append(res)
# print result
deal_result(result,100)

er

    
sklfolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat'#'r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Netcore5p'
result = []
for filep in os.listdir(sklfolder):
    if os.path.splitext(filep)[1]=='.scl':
#         print filep
        res = skldata(sklfolder+'\\'+filep)
        result.append(res)
# print result
deal_result(result,100)
