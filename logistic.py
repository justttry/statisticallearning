#encoding:UTF-8

import unittest
from numpy import *

#----------------------------------------------------------------------
def sigmoid(inX):
    """
    sigmoid函数
    P(y=1|x)
    Parameter:
    inX:输入数据矩阵
    Return:
    P(y=1|x)
    """
    return 1 / (1 + exp(-inX))

#----------------------------------------------------------------------
def gradAsent(dataMatIn, classLabels, maxIter=500):
    """
    梯度上升算法
    Parameter:
    dataMatIn:输入特征列表
    classLabels:输入类列表
    maxIter:迭代次数
    Return:
    weights:权重系数
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones((n, 1))
    for i in range(maxIter):
        h = sigmoid(dataMatrix*weights)
        weights += alpha * dataMatrix.T * (labelMat - h)
    return weights

#----------------------------------------------------------------------
def newton(dataMatIn, classLabels, maxIter=50):
    """"""
    dataMatrix = mat(dataMatIn)
    labelMatrix = mat(classLabels).T
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = zeros((n, 1))
    for _ in range(maxIter):
        p1 = sigmoid(dataMatrix*weights)
        gradient = dataMatrix.T * (labelMatrix - p1)
        p_mat = p1 * (p1 - 1) * eye(m)
        hessian = dataMatrix.T * p_mat * dataMatrix
        weights -= linalg.inv(hessian) * gradient
    return weights