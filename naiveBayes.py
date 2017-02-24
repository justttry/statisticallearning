#encoding:UTF-8

import unittest
from numpy import *


#----------------------------------------------------------------------
def calcPy(Y):
    """
    计算先验概率P(Y=ck)
    Parameter:
    Y:类别矩阵
    Return:
    Py:先验概率矩阵
    """
    #类别
    cls = set(Y.A[:, 0])
    n = float(shape(Y)[0])
    Py = zeros((len(cls), 1))
    for i, j in enumerate(cls):
        Py[i, 0] = shape(Y[Y[:, 0]==j])[1] / n
    return mat(Py)

#----------------------------------------------------------------------
def calcPx_y(X, Y, j, lamda=0):
    """
    计算条件概率
    Parameter:
    X:特征变量矩阵
    Y:类别矩阵
    j:特征变量
    Return:
    Px_y:第j个特征变量的条件概率矩阵
    """
    #计算第j个特征变量的取值范围
    a = sorted(set(X.A[:, j]))
    #计算类别的取值范围
    cls = sorted(set(Y.A[:, 0]))
    #计算k和l
    n = shape(X)[0]
    k = len(cls)
    l = len(a)
    #索引
    adict = dict(zip(a, range(l)))
    kdict = dict(zip(cls, range(k)))
    Px_y = zeros((l, k))
    for i in range(n):
        Px_y[adict[X[i, j]], kdict[Y[i, 0]]] += 1.0
    Px_y[Px_y[:, :]!=0] += lamda
    return mat(Px_y / sum(Px_y, axis=0))

#----------------------------------------------------------------------
def calcP(X, Y):
    """
    计算先验概率以及条件概率
    Parameter:
    X:特征变量
    Y:类别
    Return:
    Py:先验概率
    Px_y:条件概率
    """
    X = mat(X)
    Y = mat(Y)
    N, n = shape(X)
    Py = calcPy(Y)
    Px_y = []
    for i in range(n):
        Px_y.append(calcPx_y(X, Y, i))
    return Py, Px_y

#----------------------------------------------------------------------
def calcXx(X, x):
    """
    计算实例x在特征空间中对应的特征坐标
    Parameter:
    X:特征矩阵
    x:实例
    Return:
    y:特征坐标
    """
    n = len(x)
    X = X.A.T
    sets = map(set, X)
    sets = map(list, sets)
    sets = map(sorted, sets)
    return [sets[i].index(x[i]) for i in range(n)]
    

########################################################################
class NaiveBayesTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_calcPy(self):
        """"""
        Y = mat([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]).T
        self.assertListEqual(calcPy(Y).tolist(), (mat([[1, 2, 3, 4]]).T/10.0).tolist())
        
    #----------------------------------------------------------------------
    def test_calcPx_y(self):
        """"""
        X = mat([[4, 4, 4, 4, 3, 3, 3, 2, 2, 1]]).T
        Y = mat([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]).T
        j = 0
        result = mat([[0, 0, 0, 1],
                      [0, 0, 0, 2],
                      [0, 0, 2, 1],
                      [1, 2, 1, 0]]) / mat([[1.0, 2.0, 3.0, 4.0]])
        self.assertListEqual(calcPx_y(X, Y, j).tolist(), result.tolist())
        
    #----------------------------------------------------------------------
    def test_calcPx_y1(self):
        """"""
        X = mat([[4, 4, 4, 4, 3, 3, 3, 2, 2, 1]]).T
        Y = mat([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]).T
        j = 0
        result = mat([[0, 0, 0, 2],
                      [0, 0, 0, 3],
                      [0, 0, 3, 2],
                      [2, 3, 2, 0]]) / mat([[2.0, 3.0, 5.0, 7.0]])
        self.assertListEqual(calcPx_y(X, Y, j, 1).tolist(), result.tolist())
        
    #----------------------------------------------------------------------
    def test_calcXx(self):
        """"""
        X = mat([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
        x = [4, 5, 6]
        result = [1, 1, 1]
        self.assertListEqual(calcXx(X, x), result)
    
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(NaiveBayesTest('test_calcPy'))
    suite.addTest(NaiveBayesTest('test_calcPx_y'))
    suite.addTest(NaiveBayesTest('test_calcPx_y1'))
    suite.addTest(NaiveBayesTest('test_calcXx'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')