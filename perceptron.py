#encoding:UTF-8

import unittest
from numpy import *

#----------------------------------------------------------------------
def createDataSet(w, b):
    """
    数据产生器
    输入：w
          b
    输出：group
          labels
    """
    if type(w).__name__ != 'array':
        w = array(w)
    group = []
    labels = []
    i, j = 0, 0
    while i < 5000 or j < 5000:
        a = 10 * random.random(map(int, w.shape)) - 5
        if i < 5000 and dot(w.T, a) + b > 0:
            group.append(a.tolist())
            labels.append(1)
            i += 1
        elif j < 5000 and dot(w.T, a) + b < 0:
            group.append(a.tolist())
            labels.append(-1)
            j += 1
    return group, labels
    
#----------------------------------------------------------------------
def perceptron(trainingD, labels, n):
    """
    感知机：
    输入: tainingD,训练数据集
          n, 学习率
    输出: w, 权值向量
          b，偏置
          f, 感知机
    """
    trainingD = array(trainingD)
    dataSetSize, w_len, _ = trainingD.shape
    w = zeros((w_len, 1))
    b = 0
    i = 0
    while i < dataSetSize:
        if labels[i] * (dot(w.T, trainingD[i]) + b) > 0:
            i += 1
        else:
            w = w + n * labels[i] * trainingD[i]
            b = b + n * labels[i]
            if labels[i] * (dot(w.T, trainingD[i]) + b) > 0:
                i = 0
    f = lambda x: 1 if sum([w[i] * x[i] for i in range(w_len)]) + b > 0 \
        else -1
    return w, b, f


########################################################################
class PerceptronTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_0(self):
        """"""
        w = random.random() - 0.5
        w = array(w).reshape(1, 1)
        b = random.random() - 0.5
        group, labels = createDataSet(w, b)
        w_opt, b_opt, f = perceptron(group, labels, 0.01)
        for i, val in enumerate(group):
            self.assertEqual(f(val), labels[i])
        print '\nw:', w
        print 'b:', b
        print 'w_opt:', w_opt
        print 'b_opt:', b_opt
        print labels
        print len(labels)
        print 'test_0 end'

    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        w = random.random((2, 1)) - 0.5
        b = random.random() - 0.5
        group, labels = createDataSet(w, b)
        w_opt, b_opt, f = perceptron(group, labels, 0.01)
        for i, val in enumerate(group):
            self.assertEqual(f(val), labels[i])
        print '\nw:', w
        print 'b:', b
        print 'w_opt:', w_opt
        print 'b_opt:', b_opt
        print labels
        print len(labels)
        print 'test_1 end'


if __name__ == '__main__':
    unittest.main()
        
    
    