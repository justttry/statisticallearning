#encoding:UTF-8

import unittest
from numpy import *

#----------------------------------------------------------------------
def calcEStep(dataMat, theta):
    """
    E步
    Parameter:
    dataMat:参测变量矩阵
    theta:模型参数
    Return:
    ui:第i次的E步参数
    """
    theta0 = theta[0]
    theta1 = theta[1]
    theta2 = theta[2]
    p_b = theta0 * multiply(power(theta1, dataMat), 
                            power((1 - theta1), 1 - dataMat))
    p_c = (1 - theta0) * multiply(power(theta2, dataMat),
                                  power(1 - theta2, 1 - dataMat))
    return p_b / (p_b + p_c)

#----------------------------------------------------------------------
def calcMStep(dataMat, ui):
    """
    M步
    Parameter:
    dataMat:观测变量矩阵
    ui:第i次计算的E步值
    Return:
    thetai:第i次计算的模型参数
    """
    n = shape(dataMat)[0]
    nMat = mat(ones((n, 1)))
    theta0 = (ui.T * nMat) / n
    theta1 = (ui.T * dataMat) / (ui.T * nMat)
    theta2 = (dataMat.T * (1 - ui)) / (nMat.T * (1 - ui))
    return (theta0[0, 0], theta1[0, 0], theta2[0, 0])
    

#----------------------------------------------------------------------
def emAlgo(dataArr, theta, delta=0.0001):
    """
    EM算法
    Parameter:
    dataArr:观测变量数据
    theta:初始化模型参数
    delta:停止条件
    Return:
    theta:最终收敛的模型参数
    """
    dataMat = mat(dataArr).T
    error = 1.0
    while error > delta:
        #计算E步
        ui = calcEStep(dataMat, theta)
        #计算M步
        newTheta = calcMStep(dataMat, ui)
        #计算误差
        error = abs(newTheta[0] - theta[0])
        theta = newTheta
        print theta
    return theta

########################################################################
class EMAlgoTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_0(self):
        dataArr = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
        theta = [0.5, 0.5, 0.5]
        emAlgo(dataArr, theta)

    #----------------------------------------------------------------------
    def test_1(self):
        dataArr = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
        theta = [0.4, 0.6, 0.7]
        emAlgo(dataArr, theta)

    #----------------------------------------------------------------------
    def test_2(self):
        dataArr = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
        theta = [0.46, 0.55, 0.67]
        emAlgo(dataArr, theta)
        
        
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(EMAlgoTest('test_0'))
    suite.addTest(EMAlgoTest('test_1'))
    suite.addTest(EMAlgoTest('test_2'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')