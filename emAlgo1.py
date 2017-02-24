#encoding:UTF-8

import unittest
from numpy import *

#----------------------------------------------------------------------
def calcGauss(dataMat, thetas):
    """
    计算高斯矩阵
    Parameter:
    dataMat:观测变量
    thetas:高斯分布参数
    Return:
    高斯联合分布矩阵
    """
    return exp(-power(dataMat - thetas[0], 2) / 2.0 / thetas[1]) /\
           sqrt(2 * pi * thetas[1])

#----------------------------------------------------------------------
def calcEStep(dataMat, ak, thetas):
    """
    E步
    parameter:
    dataMat:观测变量
    ak:高斯模型加权系数
    thetas:高斯分布参数
    Return:
    rjk:模型对观测数据的响应度矩阵
    """
    gaussMat = calcGauss(dataMat, thetas)
    return multiply(ak, gaussMat) / (gaussMat * ak.T)

#----------------------------------------------------------------------
def calcMStep(dataMat, rjk):
    """
    M步
    Parameter:
    dataMat:观测变量
    rjk:分模型对观测数据的响应度矩阵
    Return:
    ak:高斯模型加权系数矩阵
    thetas:高斯分布参数
    """
    m = shape(dataMat)[0]
    oneMat = mat(ones((m, 1)))
    sumRjk = oneMat.T * rjk
    #计算mu
    uk = (dataMat.T * rjk) / sumRjk
    #计算sigma
    sigma = diag((power(dataMat - uk, 2).T * rjk) / sumRjk)
    #计算ak
    ak = sumRjk / m
    #计算高斯分布系数矩阵
    thetas = concatenate((uk, mat(sigma)))
    return ak, thetas

#----------------------------------------------------------------------
def emAlgo(dataArr, ak, thetas, delta=0.0001):
    """
    EM算法
    dataArr:观测变量输入
    ak:高斯模型加权系数
    thetas:高斯分布参数
    Return:
    ak:最终收敛的加权系数
    thetas:最终收敛的高斯分布参数
    """
    dataMat = mat(dataArr).T
    ak = mat(ak)
    thetas = mat(thetas)
    error = 1.0
    while error > delta:
        rjk = calcEStep(dataMat, ak, thetas)
        newak, thetas = calcMStep(dataMat, rjk)
        error = abs(newak[0, 0] - ak[0, 0])
        ak = newak
        print ak
        print thetas
    return ak, thetas


########################################################################
class EMAlgoTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def setUp(self):
        """"""
        def calcGaussVal(x, mu, sigma):
            return exp(-(x-mu)**2/2.0/sigma)/sqrt(2 * pi * sigma)
        self.calcGaussVal = calcGaussVal
    
    #----------------------------------------------------------------------
    def test_0(self):
        dataMat = mat([[1], [2], [3]])
        thetas = mat([[1, 2], [3, 4]])
        result = [[self.calcGaussVal(1, 1, 3), self.calcGaussVal(1, 2, 4)],
                  [self.calcGaussVal(2, 1, 3), self.calcGaussVal(2, 2, 4)],
                  [self.calcGaussVal(3, 1, 3), self.calcGaussVal(3, 2, 4)]]
        res = calcGauss(dataMat, thetas)
        self.assertListEqual(res.tolist(), result)
        
    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        dataMat = mat([[1], [2], [3]])
        thetas = mat([[1, 2], [3, 4]])
        ak = mat([[5, 6]])
        result = [[self.calcGaussVal(1, 1, 3), self.calcGaussVal(1, 2, 4)],
                  [self.calcGaussVal(2, 1, 3), self.calcGaussVal(2, 2, 4)],
                  [self.calcGaussVal(3, 1, 3), self.calcGaussVal(3, 2, 4)]]
        numerator = [[result[0][0]*ak[0, 0], result[0][1]*ak[0, 1]],
                     [result[1][0]*ak[0, 0], result[1][1]*ak[0, 1]],
                     [result[2][0]*ak[0, 0], result[2][1]*ak[0, 1]]]
        demoninator = [[numerator[0][0] + numerator[0][1]],
                       [numerator[1][0] + numerator[1][1]],
                       [numerator[2][0] + numerator[2][1]]]
        result = [[numerator[0][0]/demoninator[0][0], numerator[0][1]/demoninator[0][0]],
                  [numerator[1][0]/demoninator[1][0], numerator[1][1]/demoninator[1][0]],
                  [numerator[2][0]/demoninator[2][0], numerator[2][1]/demoninator[2][0]]]
        res = calcEStep(dataMat, ak, thetas)
        self.assertListEqual(res.tolist(), result)
        
    #----------------------------------------------------------------------
    def test_2(self):
        """"""
        dataMat = mat([[1], [2], [3]])
        thetas = mat([[1, 2], [3, 4]])
        ak = mat([[5, 6]])
        rjk = calcEStep(dataMat, ak, thetas)
        sumRjk = sum(rjk, axis=0)
        uk0 = [[rjk[0, 0]*dataMat[0, 0] + rjk[1, 0]*dataMat[1, 0] + rjk[2, 0]*dataMat[2, 0],
                rjk[0, 1]*dataMat[0, 0] + rjk[1, 1]*dataMat[1, 0] + rjk[2, 1]*dataMat[2, 0]]]
        uk = [[uk0[0][0]/sumRjk[0, 0], uk0[0][1]/sumRjk[0, 1]]]
        sigma0 = [[rjk[0, 0]*(dataMat[0, 0] - uk[0][0])**2 + rjk[1, 0]*(dataMat[1, 0] - uk[0][0])**2 + rjk[2, 0]*(dataMat[2, 0] - uk[0][0])**2,
                   rjk[0, 1]*(dataMat[0, 0] - uk[0][1])**2 + rjk[1, 1]*(dataMat[1, 0] - uk[0][1])**2 + rjk[2, 1]*(dataMat[2, 0] - uk[0][1])**2]]
        sigma = [[sigma0[0][0]/sumRjk[0, 0], sigma0[0][1]/sumRjk[0, 1]]]
        newthetas = [uk[0], sigma[0]]
        newak = [[sumRjk[0, 0]/3, sumRjk[0, 1]/3]]
        a, t = calcMStep(dataMat, rjk)
        self.assertListEqual(a.tolist(), newak)
        self.assertListEqual(t[0].tolist(), uk)
        self.assertListEqual(t[1].tolist(), sigma)
        self.assertListEqual(t.tolist(), newthetas)
        
    #----------------------------------------------------------------------
    def test_3(self):
        """"""
        dataArr = [-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
        ak = [0.5, 0.5]
        thetas = [[-1, 1], [10000, 10000]]
        newak, newthetas = emAlgo(dataArr, ak, thetas)
    
    
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(EMAlgoTest('test_0'))
    suite.addTest(EMAlgoTest('test_1'))
    suite.addTest(EMAlgoTest('test_2'))
    suite.addTest(EMAlgoTest('test_3'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')