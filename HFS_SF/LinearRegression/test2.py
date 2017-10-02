#coding=utf8
import numpy as np

a = np.array([[1,2,3,4],[1,2,3,4]])
b = np.array([[1,20,3,40],[1,2,3,4]])
c = np.array([1,2,33,4])
res = np.append(a,b,1)#a+b#
# res = np.reshape(res,(-1,4))
print res
er
"================================================================================================="
import sys
sys.path.append('../..')
from tools import commontools as gtf
gt = gtf()
import os
import csv
import numpy as np
import random
from matplotlib import pyplot as plt





def pca(dataset):
    import pylab as pl
    import numpy as np
    import sklearn as skl
    from sklearn import datasets, linear_model
    import ntpath
    print(__doc__)
    
    
    
    from sklearn import linear_model, decomposition, datasets
    from sklearn.pipeline import Pipeline
    from sklearn.grid_search import GridSearchCV
    
    logistic = linear_model.LogisticRegression()
    
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    #digits = datasets.load_iris()#.load_diabetes()#.load_digits()
    X_digits = dataset.data
    y_digits = dataset.target
    
    ###############################################################################
    # Plot the PCA spectrum
    pca.fit(X_digits)
    # print pca.components_
    for a,b in  zip(*(pca.explained_variance_ratio_,pca.explained_variance_)):
        print '%.2f,%.2f' %(a,b)
    
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=1)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    
    ###############################################################################
    # Prediction
    
    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)
    
    #Parameters of pipelines can be set using ‘__’ separated parameter names:
    
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  logistic__C=Cs))
    estimator.fit(X_digits, y_digits)
    
    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    plt.show()


dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\ATT\test4lr.txt',usecols=(1,3,5,6,7,8,9,10,11,12,), dtype=float, delimiter=',',)
tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\ATT\test4lr.txt',usecols=(8), dtype=float, delimiter=',',)

from tools import loadata as ldata
ld = ldata()
dataset = ld.load_skl(dat,tar)
pca(dataset)
er


def load_diabetes2(datafilepath,target,xrange):
    lenx = len(gt.csv2list(datafilepath,seprator='\t')[0])
    print lenx
    txtfile = datafilepath#r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl'
    data = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(xrange[0],lenx),skip_header=0, skip_footer=0)#(1, 4, 5)
    target = numpy.genfromtxt(fname=txtfile, dtype=float, delimiter='\t', usecols = range(0,1),skip_header=0, skip_footer=0)#(1, 4, 5)
    "need to replace nan and inf etc."
    data = numpy.nan_to_num(data)
        
  
    return datasets.base.Bunch(data=data, target=target)

def skldata(sklfile,target,xrange):
    diabetes = load_diabetes2(sklfile,target,xrange)#(r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\vd0_0.1_all_0.1_km_1.skl.scl') #载入数据datasets.
     
    diabetes_x = diabetes.data#[:, np.newaxis]
    diabetes_x_temp = diabetes_x#[:, :, 2]
    
    test_index = -61 
    diabetes_x_train = diabetes_x_temp[:test_index] #训练样本
    diabetes_x_test = diabetes_x_temp[test_index:] #检测样本
    diabetes_y_train = diabetes.target[:test_index]
    diabetes_y_test = diabetes.target[test_index:]
    "-------------------regression model-------------------------------" 
    #     regr = linear_model.LinearRegression()
    regr = linear_model.LogisticRegression()
    
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
    print sklfile,"AUC:\t",aucvalue
    print 'r2:',metrics.r2_score(y, pred)
    print 'explained_variance_score',metrics.explained_variance_score(y, pred)
#     from sklearn.feature_selection import chi2
#     scores, pv = chi2(diabetes_x_train, diabetes_y_train)
#     print pv      
    for a,b in zip(*(predict_result,diabetes_y_test)):
        if a*b>=0:
            rr+=1
    
    return [sklfile,float(wr+rr)/len(predict_result),aucvalue]

def regression(sklfolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\StatNet\netcore\combine\10\skl',xrange=[1,6]):   
    sklfolder=r'G:\HFS\WeiboData\HFSWeiboStatNet\SKL\Stat\test\test'
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
    print result
    [meta,accurate,auc] = zip(*(result))
    print '#','\t',numpy.average(accurate),'\t',numpy.average(auc)
    #accurate_matrix = deal_result(result,100,30)
#     print accurate_matrix
    # csvwriter = csv.writer(file(sklfolder+'\\result\\accurate_matrix.csv','w'))
    # for line in accurate_matrix:
    #     csvwriter.writerow(line)

if __name__=='__main__':
    regression()