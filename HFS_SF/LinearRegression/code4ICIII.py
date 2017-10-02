#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
zhucheng yidali international conference
=====================
random forest to predict the succeed or failed hfs eposides
small dataset of 307


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV,Ridge,LogisticRegression

h = .02  # step size in the mesh

names = [
#          "Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
#           "AdaBoost", "Naive Bayes", 
#          "LDA", 
         "Random Forest",
         #"QDA",
         #"SGDClassifier",
         #"Lasso",
         #"RidgeCV",
         #"Ridge",
         #"LogisticRegression"
         ]

classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=16, n_estimators=5, max_features=7),
#     RandomForestClassifier(max_depth=15, n_estimators=100, max_features=3),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     LDA(),
#     QDA(),
#     SGDClassifier(loss="hinge", penalty="l2"),
#     Lasso(alpha = 0.1),
#     RidgeCV(alphas=[0.1, 1.0, 10.0]),
#     Ridge (alpha = .5),
#     LogisticRegression()
    ]

X, y = make_classification(n_features=49, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# datasets = [make_moons(n_samples=10, noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]
"------------------------------------"
def start(dat,tar,usecols):
    if 1:
        import sys
        sys.path.append('../..')
        from tools import loadata as ldata
        ld = ldata()
        dataset = ld.load_skl(dat,tar)
        
        dat,tar = np.nan_to_num(dat),np.nan_to_num(tar)
        datasets = [[dat,tar],]#[dat,tar],[dat,tar]
        #print datasets
    "------------------------------------"
    
    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds in datasets:
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        print xx.shape,yy.shape,xx.ravel().shape, yy.ravel().shape
    
        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
    
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            
            predict_result = clf.predict(X=X_test)
            print y_test,'\n',predict_result
            wr = 0
            rr = 0    
        #     print numpy.array(predict_result),'\n',diabetes_y_test
            
            from sklearn import metrics
            y = np.array(y_test)#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
            pred = np.array(predict_result)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            print metrics.auc(fpr, tpr)
            
            print name, clf, score
            
            from sklearn import metrics
            y_pred = predict_result#[0, 1, 0, 0]
            y_true = y_test#[0, 1, 0, 1]
            print 'precision_score',metrics.precision_score(y_true, y_pred)
            print 'recall_score',metrics.recall_score(y_true, y_pred)
            print 'f1_score',metrics.f1_score(y_true, y_pred)  
            print metrics.fbeta_score(y_true, y_pred, beta=0.5)  
            print metrics.fbeta_score(y_true, y_pred, beta=1)  
            print metrics.fbeta_score(y_true, y_pred, beta=2) 
            print metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5) 
            print metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0) 
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
    #         if hasattr(clf, "decision_function"):
    #             Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #         else:
    #             Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    #   
    #         # Put the result into a color plot
    #         Z = Z.reshape(xx.shape)
    #         ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6)
    
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
    
    figure.subplots_adjust(left=.02, right=.98)
    plt.show()


if __name__=='__main2__':
    #     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_stat_1.0.txt',usecols=(9,10,7,8,5,6,3,4,11,), dtype=float, delimiter='\t',)#
    #     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_stat_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)
    
    #     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_net_1.0.txt',usecols=(9,10,7,8,5,6,4,11,), dtype=float, delimiter='\t',)#
    #     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_net_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)
    
    #     usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50)
    #     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statcore_1.0.txt',usecols=usecols, dtype=float, delimiter='\t',)#usecols=(9,10,7,8,5,6,4,11,),
    #     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statcore_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)
        
    #     usecols=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50)#)#
    #     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_stat-530_1.0.txt',usecols=usecols, dtype=float, delimiter='\t',)#
    #     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_stat-530_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)
    
    
        
    usecols=(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50)#)#
    dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statnetallsuccessedWeak-530_1.0.txt',usecols=usecols, dtype=float, delimiter='\t',)#
    tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statnetallsuccessedWeak-530_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)
    

    start(dat,tar,usecols)

