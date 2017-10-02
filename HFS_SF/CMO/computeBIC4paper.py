#encoding=utf8
import os
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pylab

import sys
sys.path.append('..\..')
from tools import commontools as gtf

gt = gtf()
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = [(1.0 / (n[i] - m)) * sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2)  for i in xrange(m)]

    const_term = 0.5 * m * np.log10(N)

    BIC = np.sum([n[i] * np.log10(n[i]) -
           n[i] * np.log10(N) -
         ((n[i] * d) / 2) * np.log10(2*np.pi) -
          (n[i] / 2) * np.log10(cl_var[i]) -
         ((n[i] - m) / 2) for i in xrange(m)]) - const_term

    return(BIC)

def one_metric(attfile):
    attlist = np.genfromtxt(fname=attfile, dtype=float, comments=None, delimiter=',', skip_header=1, skip_footer=0)
    attlistz = zip(*attlist)
    attlistz = np.nan_to_num(attlistz)
    #print attlist.shape
    return attlistz

def loadattributes(workfolder_fig,personcnt,experimenTimes):
    #         mode = 7
    att = []
    for mode in range(2,8):
        workfolder = "N:\\HFS\\WeiboData\\CMO\\Mode"+str(mode)+"\\graphs\\"#"G:\\HFS\\WeiboData\\CMO\\test\\"#
#         workfolder_fig = "N:\\HFS\\WeiboData\\CMO\\Mode"+str(mode)+"\\figs\\"
    #     x = define_metrics(worksfolder+'graphs\\')
    #     x1 = define_metrics(worksfolder+'Mode1\\graphs\\')
    #     x2 = define_metrics(worksfolder+'Mode2\\graphs\\')
        
    #     x1.extend(x2)
    #     x=x1#zip(*x1)
        
        for modefans in [1,2,4]:#3,,5#
            for modefr in [1,2,4]:#3,,5
                for modeal in [1,2,4]:#3,,5
                    for modemen in [1,2,4]:#3,,5
                        filep = str(personcnt)+'_'+str(experimenTimes)+'_'+str(modefans)+'_'+str(modefr)+'_'+str(modeal)+'_'+str(modemen)+'.netattri'
                        print workfolder+filep
                        filetuple = os.path.splitext(filep)
                        if os.path.isfile(workfolder+filep):
                            attlistz = one_metric(workfolder+filep)
                            att.extend(attlistz)
        return att


workfolder_fig = "D:\\HFS\\WeiboData\\CMO\\Modefigs\\fig\\"
personcnt = 100
experimenTimes = 100
kmax = 25
figname = r'D:\HFS\WeiboData\CMO\Modefigs\100_100.att.BIC'+str(kmax)+'.png'
#x = loadattributes(workfolder_fig,personcnt,experimenTimes)
x = np.genfromtxt(fname='D:\\HFS\\WeiboData\\CMO\\Modefigs\\'+str(personcnt)+'_'+str(experimenTimes)+'.att', dtype=float, comments=None, delimiter=' ', skip_header=1, skip_footer=0)
xyz = zip(*x)
xyzz = gt.normlistlist(listlista=xyz[1:],metacolcount=0,sumormax='max')#xyz[1:]#
xy = zip(*xyzz)#[1:]
z = zip(*xyz[:1])


# # IRIS DATA
# iris = sklearn.datasets.load_iris()
# X = iris.data[:, :4]  # extract only the features
# print X
# #Xs = StandardScaler().fit_transform(X)
# Y = iris.target

X = np.asarray(xy)
ks = range(1,kmax)
print 'data has parepared.'

# run 9 times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]

# now run for each cluster the BIC computation
BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]

plt.plot(ks,BIC,'r-o')
print ks,BIC
plt.title("cluster vs BIC")
plt.xlabel("# clusters")
plt.ylabel("# BIC")
pylab.savefig(figname, dpi=300)

plt.show()
gt.savefigdata(datafilepath=figname+'.data',x=ks,y=BIC,errorbarlist=None,title='cluster vs BIC',xlabel='# clusters',ylabel='# BIC',leglend='subleglabel')

plt.close()