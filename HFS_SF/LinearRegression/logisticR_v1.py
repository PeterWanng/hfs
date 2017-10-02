#encoding=utf8
import sys
sys.path.append('../..')
from tools import commontools as gtf
import os
import csv
import numpy
import random
from matplotlib import pyplot as plt
gt = gtf()
#lasso


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

# def load_diabetes2(datafilepath,txtfilets = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\target.txt'):
#     txtfile = datafilepath#r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl'
#     txtfilets = txtfilets#r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\target.txt'
# 
#     data = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(1,6),skip_header=0, skip_footer=0)#(1, 4, 5)
#     target = numpy.genfromtxt(fname=txtfilets, dtype=float)#(1, 4, 5)
#     return datasets.base.Bunch(data=data, target=target)

def load_diabetes2(datafilepath,target,xrange):
    lenx = len(gt.csv2list(datafilepath,seprator='\t')[0])
    print lenx
    txtfile = datafilepath#r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl'
    data = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(xrange[0],lenx),skip_header=0, skip_footer=0)#(1, 4, 5)
    target = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(0,1),skip_header=0, skip_footer=0)#(1, 4, 5)
    "need to replace nan and inf etc."
    data = numpy.nan_to_num(data)
        
  
    return datasets.base.Bunch(data=data, target=target)

def plot_result(result,featurecnt):
    lenlistb = len(result)
    x = range((lenlistb+1)/featurecnt)
    a = ['x']
    a.extend(x)
    result_matrix = [a]
    for start_index in range(1,featurecnt+1):
        y = []
        for feature_index in range(start_index,lenlistb+1,featurecnt):
            print result[feature_index-1]
#             x.append(feature_index)
            y.append(float(result[feature_index-1][2]))
#         plt.plot(x,y,marker='o')
        y.insert(0,str(result[feature_index-1][1]).split('\\')[-1])
        result_matrix.append(y)
#         plt.title(str(result[feature_index-1][1]).split('\\')[-1])
    plt.plot(x,[0.5]*len(x),linewidth=4.0,color='r')
    plt.show()
    plt.close()
    print result_matrix
    return result_matrix
        
def deal_result(lista,experiments_times,featurecnt):
    "IN:list [filename,accucy];experiments_times for each feature"
    "OUT:average accucy of each feature"
    result = []
    listb = zip(*(lista))
    i = 1
    tempone = []
    lenlistb = len(listb[0])
    while i< lenlistb+1:
        tempone.append(listb[1][i-1])
        if i%experiments_times==0:
            result.append([len(tempone),listb[0][i-1],numpy.average(tempone)])
#             print len(tempone),'\t',listb[0][i-1],'\t',numpy.average(tempone)
            tempone = []        
        i+=1
    result_matrix = plot_result(result,featurecnt)
    return result_matrix
    
def skldata(sklfile,target,xrange):
    diabetes = load_diabetes2(sklfile,target,xrange)#(r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl') #载入数据datasets.
     
    diabetes_x = diabetes.data#[:, np.newaxis]
    diabetes_x_temp = diabetes_x#[:, :, 2]
    
    test_index = -61 
    diabetes_x_train = diabetes_x_temp[:test_index] #训练样本
    diabetes_x_test = diabetes_x_temp[test_index:] #检测样本
    diabetes_y_train = diabetes.target[:test_index]
    diabetes_y_test = diabetes.target[test_index:]
     
    # print len(diabetes_y_test),len(diabetes_x_test),len(diabetes_x_train),len(diabetes_y_train),#
    
    "-------------------regression model-------------------------------" 
#     regr = linear_model.LinearRegression()
    regr = linear_model.LogisticRegression()
   
#     regr = linear_model.ARDRegression()#     0.839344262295     0.66485042735
#     regr = linear_model.BayesianRidge()#     0.852459016393     0.651709401709
#     regr = linear_model.ElasticNet()#     0.852459016393     0.5
#     regr = linear_model.ElasticNetCV()#0.852459016393     0.552884615385
#     regr = linear_model.Lars()#     0.606557377049     0.495085470085
#     regr = linear_model.LarsCV()#     0.850819672131     0.521260683761
#     regr = linear_model.Lasso()#     0.852459016393     0.5
#     regr = linear_model.LassoCV()#     0.852459016393     0.537072649573
#     regr = linear_model.LassoLarsCV()#     0.852459016393     0.507905982906
#     regr = linear_model.LassoLarsIC()#     0.834426229508     0.640277777778
#     regr = linear_model.LogisticRegression()#     0.852459016393     0.5
#     regr = linear_model.MultiTaskElasticNet()#     0.852459016393     0.5
#     regr = linear_model.MultiTaskLasso()#     0.852459016393     0.5
#     regr = linear_model.OrthogonalMatchingPursuit()#     0.839344262295     0.646367521368
#     regr = linear_model.PassiveAggressiveClassifier()#     0.852459016393     0.5
#     regr = linear_model.PassiveAggressiveRegressor()#     0.852459016393     0.520085470085
#     regr = linear_model.Perceptron()#     0.852459016393     0.5
#     regr = linear_model.Ridge()#     0.847540983607     0.623290598291
#     regr = linear_model.RidgeClassifier()#     0.847540983607     0.497115384615
#     regr = linear_model.RidgeClassifierCV()#     0.847540983607     0.497115384615
#     regr = linear_model.RidgeCV()#     0.847540983607     0.640598290598
#     regr = linear_model.SGDClassifier()#     0.855737704918     0.511111111111
#     regr = linear_model.SGDRegressor()#     0.852459016393     0.44188034188
    
    
    
#     from sklearn import svm
#     regr =svm.SVC()#     0.852459016393     0.5
#     regr =svm.SVR()#     0.852459016393     0.494871794872
    
    "-------------------regression model-------------------------------" 
     
    regfit = regr.fit(diabetes_x_train, diabetes_y_train)
    predict_result = regr.predict(X=diabetes_x_test)
#     print predict_result
#     print diabetes_y_test
    wr = 0
    rr = 0    

    from sklearn import metrics
    y = np.array(diabetes_y_test)#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
    pred = np.array(predict_result)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    aucvalue = metrics.auc(fpr, tpr)
    
    print 'r2:',metrics.r2_score(y, pred)
    print metrics.explained_variance_score(y, pred)
    from sklearn.feature_selection import chi2
    scores, pv = chi2(diabetes_x_train, diabetes_y_train)
    print pv
#     print sklfile,"AUC:\t",aucvalue

    
    # Plot ROC curve
#     pl.clf()
#     pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % aucvalue)
#     pl.plot([0, 1], [0, 1], 'k--')
#     pl.xlim([0.0, 1.0])
#     pl.ylim([0.0, 1.0])
#     pl.xlabel('False Positive Rate')
#     pl.ylabel('True Positive Rate')
#     pl.title('Receiver operating characteristic example')
#     pl.legend(loc="lower right")
#     pl.show()
    
    for a,b in zip(*(predict_result,diabetes_y_test)):
        if a*b>=0:
            rr+=1
    
#     for item in predict_result[0:4]:
#         if float(item)>=0:
#           wr+=1 
#     for item in predict_result[4:8]:
#         if float(item)<0:
#           rr+=1 
#     print sklfile,' #Accucy:',float(wr+rr)/len(predict_result)
#     print 'Coefficients :\n', regr.coef_ 
#     print ("Residual sum of square: %.2f" %np.mean((regr.predict(diabetes_x_test) - diabetes_y_test) ** 2)) 
#     print ("variance score: %.2f" % regr.score(diabetes_x_test, diabetes_y_test))
     
    # pl.scatter(diabetes_x_test,diabetes_y_test, color = 'black')
    # pl.plot(diabetes_x_test, regr.predict(diabetes_x_test),color='blue',linewidth = 3)
    # pl.xticks(())
    # pl.yticks(())
    # pl.show()
    return [sklfile,float(wr+rr)/len(predict_result),aucvalue]


    
def regression(sklfolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet-5p\netcore\skl',xrange=[1,6]):   
#     sklfolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat\test'
    result = []
    txtfilets = r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Netcore5p\target.txt'
    target = numpy.genfromtxt(fname=txtfilets, dtype=float)#(1, 4, 5)
    for filep in os.listdir(sklfolder):
        if os.path.splitext(filep)[1]=='.scl':
            "IN:give a file formatted as Array, the data part should not contains NaN or infinity.the first col is the flag, p.s. -1,1 etc"
            "OUT:predict results"
            try:
                res = skldata(sklfolder+'\\'+filep,target,xrange)
                result.append(res)
            except Exception,e:
                print e
    [meta,accurate,auc] = zip(*(result))
    print '#','\t',numpy.average(accurate),'\t',numpy.average(auc)
#     accurate_matrix = deal_result(result,100,30)
#     print accurate_matrix
    # csvwriter = csv.writer(file(sklfolder+'\\result\\accurate_matrix.csv','w'))
    # for line in accurate_matrix:
    #     csvwriter.writerow(line)

if __name__=='__main__':
    regression()