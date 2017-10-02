from numpy import *
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../..')
from tools import commontools as gtf
import random
import os  
import numpy
gt = gtf()
import numpy as np

class logisticReg():
    # calculate the sigmoid function
    def sigmoid(self,inX):
        return 1.0 / (1 + exp(-inX))
    
    
    # train a logistic regression model using some optional optimize algorithm
    # input: train_x is a mat datatype, each row stands for one sample
    #         train_y is mat datatype too, each row is the corresponding label
    #         opts is optimize option include step and maximum number of iterations
    def trainLogRegres(self,train_x, train_y, opts):
        # calculate training time
        startTime = time.time()
    
        numSamples, numFeatures = shape(train_x)
        alpha = opts['alpha']; maxIter = opts['maxIter']
        weights = ones((numFeatures, 1))
        
    
        # optimize through gradient descent algorilthm
        for k in range(maxIter):
            fitResult = []
            if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm
                output = self.sigmoid(train_x * weights)
                error = train_y - output
                weights = weights + alpha * train_x.transpose() * error
                fitResult.append(output[0][0])
            elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent
                for i in range(numSamples):
                    output = self.sigmoid(train_x[i, :] * weights)
                    error = self.train_y[i, 0] - output
                    weights = weights + alpha * train_x[i, :].transpose() * error
                    fitResult.append(output[0][0])
            elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent
                # randomly select samples to optimize for reducing cycle fluctuations 
                dataIndex = range(numSamples)
                for i in range(numSamples):
                    alpha = 4.0 / (1.0 + k + i) + 0.01
                    randIndex = int(random.uniform(0, len(dataIndex)))
                    output = self.sigmoid(train_x[randIndex, :] * weights)
                    error = train_y[randIndex, 0] - output
                    weights = weights + alpha * train_x[randIndex, :].transpose() * error
                    del(dataIndex[randIndex]) # during one interation, delete the optimized sample
                    fitResult.append(np.matrix.tolist(output)[0][0])
            else:
                raise NameError('Not support optimize method type!')
            
        
    
        print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
        print len(weights),len(fitResult)
        return weights,fitResult
    
    
    # test your trained Logistic Regression model given test set
    def testLogRegres(self,weights, test_x, test_y):
        numSamples, numFeatures = shape(test_x)
        matchCount = 0
        for i in xrange(numSamples):
            predict = self.sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
            if predict == bool(test_y[i, 0]):
                matchCount += 1
        accuracy = float(matchCount) / numSamples
        return accuracy
    
    
    # show your trained logistic regression model only available with 2-D data
    def showLogRegres(self,weights, train_x, train_y):
        # notice: train_x and train_y is mat datatype
        numSamples, numFeatures = shape(train_x)
        print numSamples, numFeatures
        if numFeatures != 3:
            print "Sorry! I can not draw because the dimension of your data is not 2!"
            return 1
    
        # draw all samples
        for i in xrange(numSamples):
            if int(train_y[i, 0]) == 0:
                plt.plot(train_x[i, 1], train_x[i, 2], 'or')
            elif int(train_y[i, 0]) == 1:
                plt.plot(train_x[i, 1], train_x[i, 2], 'ob')
    
        # draw the classify line
        min_x = min(train_x[:, 1])[0, 0]
        max_x = max(train_x[:, 1])[0, 0]
        weights = weights.getA()  # convert mat to array
        y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
        y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
        plt.xlabel('X1'); plt.ylabel('X2')
        plt.show()
        
    
    def r2_score(self,y_true, y_pred):
        """R2 (coefficient of determination) regression score function.
    
        Best possible score is 1.0, lower values are worse.
    
        Parameters
        ----------
        y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
            Ground truth (correct) target values.
    
        y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
            Estimated target values.
    
        Returns
        -------
        z : float
            The R2score.
    
        Notes
        -----
        This is not a symmetric function.
    
        Unlike most other scores, R2 score may be negative (it need not actually
        be the square of a quantity R).
    
        References
        ----------
        .. [1] `Wikipedia entry on the Coefficient of determination
                <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    
        Examples
        --------
        >>> from sklearn.metrics import r2_score
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
        0.948...
        >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
        >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
        >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
        0.938...
    
        """
    #     y_type, y_true, y_pred = _check_reg_targets(y_true, y_pred)
    
        if len(y_true) == 1:
            raise ValueError("r2_score can only be computed given more than one"
                             " sample.")
        numerator = ((y_true - y_pred) ** 2).sum(dtype=np.float64)
        denominator = ((y_true - y_true.mean(axis=0)) ** 2).sum(dtype=np.float64)
    
        if denominator == 0.0:
            if numerator == 0.0:
                return 1.0
            else:
                # arbitrary set to zero to avoid -inf scores, having a constant
                # y_true is not interesting for scoring a regression anyway
                return 0.0
        print 'numerator',numerator
        return 1 - numerator / denominator
    
    def deal_y(self,lista):
        i = 0
        for it in lista:
            if it>0:
                lista[i] = 1
            i+=1
        return lista    
    
    def start(self,train_x, train_y, opts):
        ## step 2: training...
        print "step 2: training..."
        opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
        optimalWeights,output = self.trainLogRegres(train_x, train_y, opts)
        print optimalWeights
        
        ## step 3: testing
        print "step 3: testing..."
        
        test_x = train_x; test_y = train_y
        accuracy = self.testLogRegres(optimalWeights, test_x, test_y)
        print accuracy,np.array(train_y.flat),np.array(output)
        trainy = np.array(train_y.flat)
        # trainy = deal_y(trainy)
        r2 = self.r2_score(trainy,np.array(output))
        
        ## step 4: show the result
        print "step 4: show the result..."    
        print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
        print 'The classify r2 is: %.3f%%' % (r2 * 100)
        self.showLogRegres(optimalWeights, train_x, train_y) 
"============================================"



def loadData():
    featureFilepath = r'G:\HFS\WeiboData\HFSWeibo\test\3513472585606907.repost'#r'G:\HFS\WeiboData\HFSWeibo\test\test_newformat.repost'#3513472585606907#3482476628770294
    repost = gt.csv2list_new(featureFilepath)
    from weibo_tools import deal_weibo
    dwb = deal_weibo()
    ma_fans,cbm_frs,inv_mention,act_micorcnt,bi_followers_count,mid,repostlen = dwb.weibo2list(featureFilepath)
    mat2 = [ma_fans,cbm_frs,act_micorcnt,bi_followers_count,repostlen]
    train_x = zip(*(mat2))
#     train_x = np.ndarray.tolist(train_x)

#     inv_mention = deal_y(inv_mention)
    train_y = [inv_mention]
#     print train_x
    return  mat(train_x), mat(train_y).transpose()
    
    train_x = []
    train_y = []
    fileIn = open(r'G:\HFS\WeiboData\HFSWeibo\test\testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    print train_x
#     print mat(train_x), mat(train_y).transpose()


if __name__=='__main2__':
    lr = logisticReg()
    ## step 1: load data
    print "step 1: load data..."
    train_x, train_y = loadData()
    test_x = train_x; test_y = train_y
    
    ## step 2: training...
    print "step 2: training..."
    opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
    optimalWeights,output = lr.trainLogRegres(train_x, train_y, opts)
    print optimalWeights
    
    ## step 3: testing
    print "step 3: testing..."
    accuracy = lr.testLogRegres(optimalWeights, test_x, test_y)
    print np.array(train_y.flat),np.array(output)
    trainy = np.array(train_y.flat)
    # trainy = deal_y(trainy)
    r2 = lr.r2_score(trainy,np.array(output))
    
    ## step 4: show the result
    print "step 4: show the result..."    
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    print 'The classify r2 is: %.3f%%' % (r2 * 100)
    lr.showLogRegres(optimalWeights, train_x, train_y) 
        
        