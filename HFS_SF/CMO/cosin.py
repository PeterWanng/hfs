#encoding=utf-8
import sys
sys.path.append("G:\MyCode\MyCode\jieba-0.32\\")
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import decomposition
import time




def cosine_similarity(X, Y=None):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    kernel matrix : array_like
        An array with shape (n_samples_X, n_samples_Y).
    """
    # to avoid recursive import

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = linear_kernel(X_normalized, Y_normalized)

    return K



n_topic = 30
n_top_words = 25

doc1 = open(r'G:\MyCode\MyCode\HFS_SF\CMO\doc1.txt').read()
doc2 = open(r'G:\MyCode\MyCode\HFS_SF\CMO\doc2.txt').read()
doc3 = open(r'G:\MyCode\MyCode\HFS_SF\CMO\doc3.txt').read()

#print doc1.encode('utf8')
seg_list1 = jieba.cut(doc1, cut_all=False)#"可能是个飞机"
seg_list2 = jieba.cut(doc2, cut_all=False)
seg_list3 = jieba.cut(doc3, cut_all=False)
##print "Full Mode:", "/ ".join(seg_list1) 

docs = seg_list1
#random.shuffle(docs)

print "read done."

print "transform"
count_vect = CountVectorizer()

counts = count_vect.fit_transform(docs)
tfidf = TfidfTransformer().fit_transform(counts)
print tfidf.shape


t0 = time.time()
print "training..."

nmf = decomposition.NMF(n_components=n_topic).fit(tfidf)
print("done in %0.3fs." % (time.time() - t0))

# Inverse the vectorizer vocabulary to be able
feature_names = count_vect.get_feature_names()
for it in feature_names:
    print it
er

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print("")
    
 
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix1 = tfidf_vectorizer.fit_transform(seg_list1)
tfidf_matrix2 = tfidf_vectorizer.fit_transform(seg_list2)
tfidf_matrix3 = tfidf_vectorizer.fit_transform(seg_list3)
print tfidf_matrix1
cosine = cosine_similarity(tfidf_matrix1.data, tfidf_matrix2.data)
print cosine 
er


train_set  = seg_list
tfidf_vectorizer = TfidfVectorizer()
#train_set = ['a','b','b','bn']
tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)
print tfidf_matrix.data
length = tfidf_matrix.getnnz()
cosine = cosine_similarity(tfidf_matrix[length-10], tfidf_matrix)
print cosine
#er#

import sklearn 

from sklearn.metrics.pairwise import cosine_similarity  as cs

sklearn.metrics.pairwise.distance_metrics
x = [1,2,3]
y = [4,5,0]

print cs(x,y)