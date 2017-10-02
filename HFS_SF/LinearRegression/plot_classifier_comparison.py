#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

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
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
h = .01  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", 
         "LDA", 
        "QDA",
        "SGDClassifier",
        "GradientBoostingClassifier"
        "Lasso",
        #"RidgeCV",
        #"Ridge",
#         "LogisticRegression"
         ]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=6),
    RandomForestClassifier(max_depth=5),#, n_estimators=10, max_features=1
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA(),
    SGDClassifier(loss="hinge", penalty="l2"),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    Lasso(alpha = 0.1),
#     RidgeCV(alphas=[0.1, 1.0, 10.0]),
#     Ridge (alpha = .5),
#     LogisticRegression(),
               ]

classfierIndex = 3
names = [names[classfierIndex-1],]
classifiers = [classifiers[classfierIndex-1]]
# X, y = make_classification(n_features=49, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# # rng = np.random.RandomState(2)
# # X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)

# datasets = [make_moons(n_samples=10, noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]
"------------------------------------"
test_size = 0.8
if 1:
    featurePath = 'featurePath'
#     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_stat_1.0.txt',usecols=(9,10,7,8,5,6,3,4,11,), dtype=float, delimiter='\t',)#
#     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_stat_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)

#     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_net_1.0.txt',usecols=(9,10,7,8,5,6,4,11,), dtype=float, delimiter='\t',)#
#     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_net_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)

#     usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50)
#     dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statcore_1.0.txt',usecols=usecols, dtype=float, delimiter='\t',)#usecols=(9,10,7,8,5,6,4,11,),
#     tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statcore_1.0.txt',usecols=(1), dtype=float, delimiter='\t',)

    
#     
    usecols=(5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)
    #usecols=list(np.random.choice(usecols,64,False))#64)#
    dat = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statnetallsuccessedWeak-530_1.0.txt',usecols=usecols, dtype=float, delimiter='\t',)#
    tar = np.genfromtxt(r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statnetallsuccessedWeak-530_1.0.txt',usecols=(1), dtype=int, delimiter='\t',)
 
#     usecols=(5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)
#     #usecols=list(np.random.choice(usecols,64,False))#64)#
#     featurePath = r'G:\HFS\WeiboData\HFSWeibo\Stat_Sql\percent_statnetall-530_1.0-0to1.txt'
#     dat = np.genfromtxt(featurePath,usecols=usecols, dtype=float, delimiter='\t',)#
#     tar = np.genfromtxt(featurePath,usecols=(1), dtype=int, delimiter='\t',)

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

metrics_v = []
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#     # and testing points
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    #ax.text(0,0,featurePath.split('\\')[-1],fontsize=7)
    ax.set_title(featurePath.split('\\')[-1],fontsize=7)    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        #print name, clf
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        for j in range(1,10):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            #scores = cross_val_score(clf, X, y)
            
            predict_result = clf.predict(X=X_test)
            wr = 0
            rr = 0    
        #     print numpy.array(predict_result),'\n',diabetes_y_test
            
            from sklearn import metrics
            y = np.array(y_test)#[1,1,1,1,2,2,2,2])#[1,1,1,1,-1,-1,-1,-1])#[1, 1, 2, 2])
            pred = np.array(predict_result)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            
            print name, clf, score, auc
            print y_test,'\n',predict_result
            
            
            from sklearn import metrics
            y_pred = predict_result#[0, 1, 0, 0]
            y_true = y_test#[0, 1, 0, 1]
            precision = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)
            f1 = metrics.f1_score(y_true, y_pred)  
            print metrics.fbeta_score(y_true, y_pred, beta=0.5)  
            print metrics.fbeta_score(y_true, y_pred, beta=1)  
            print metrics.fbeta_score(y_true, y_pred, beta=2) 
            print metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5) 
            
            
    #         import numpy as np
    #         from sklearn.metrics import precision_recall_curve
    #         y_true = y_test#np.array([0, 0, 1, 1])
    #         y_scores = scores#np.array([0.1, 0.4, 0.35, 0.8])
    #         precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    #         print precision, recall, threshold
     
             
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
            ax.text(xx.max() - .3, yy.min() + .3, ('pro %.2f' % test_size).lstrip('0')+('\nscore %.2f' % score).lstrip('0')+('\nauc %.2f' % auc).lstrip('0')+('\nprec %.2f' % precision).lstrip('0')+('\nrecall %.2f' % recall).lstrip('0')+('\nf1 %.2f' % f1).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
            metrics_v.append([test_size,score,auc,precision,recall,f1])

figure.subplots_adjust(left=.02, right=.98)
print np.mean(metrics_v,0)
plt.show()
