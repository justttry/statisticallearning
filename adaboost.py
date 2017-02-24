#encoding:UTF-8

"""
1.初始化训练数据的权值分布
  D1 = (w11, w12, ..., w1i, ..., wnN) w1i = 1/N, i = 1,2,...,N
2.对m=1,2,...,M
  (a)根据权值分布Dm得到基本分类器
  (b)在生成Gm(x)的同时得到分类误差率em
  (c)计算Gm(x)的系数am
  (d)更新权值分布
3.构建基本分类器的线性组合f(x)
4.判断f(x)是否满足停止条件,不满足回到2，满足到5
5.得到最终分类器G(x)
"""

import unittest
from numpy import *

class Composable(object):
    
    #----------------------------------------------------------------------
    def __init__(self, function):
        """"""
        self.function = function
        
    #----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        self.function(*args, **kwargs)
    
    #----------------------------------------------------------------------
    def __mul__(self, other):
        """"""
        def multi(*args, **kwargs):
            return self.function(*args, **kwargs) * other
        return multi
    
    #----------------------------------------------------------------------
    def __rmul__(self, other):
        """"""
        def rmulti(*args, **kwargs):
            return self.function(*args, **kwargs) * other
        return rmulti
    
    
#----------------------------------------------------------------------
def stumpClassify(dimen, threshVal, threshIneq):
    """
    构造单层决策树桩函数
    Parameter:
    dimen:特征值
    threshVal:划分点
    threshIneq:大于/小于
    Return:
    classify:分类函数
    """
    @Composable
    def classify(x):
        if threshIneq == 'lt' and x[dimen] < threshVal:
            return -1.0
        elif threshIneq == 'gt' and x[dimen] > threshVal:
            return -1.0
        else:
            return 1.0
    return classify

#----------------------------------------------------------------------
def calcStumpClassify(dataArr, dimen, threshVal, threshIneq):
    """
    计算基于单层决策树的分类
    Parameter:
    dataArr:输入训练数据集
    dimen:特征值对应的标量
    threshVal:划分点
    threshIneq:大于/小于
    Return:
    retArray:训练数据集的类预测
    """
    retArray = ones((shape(dataArr)[0], 1))
    if threshIneq == 'lt':
        retArray[dataArr[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataArr[:, dimen] >= threshVal] = -1.0
    return retArray


#----------------------------------------------------------------------
def buildStump(dataMatrix, labelMat, D):
    """
    构造单层决策树桩
    Parameter:
    dataMatrix:输入数据
    labelMat:输入类别
    D:权值分布
    Return:
    单层决策树
    bestClasEst:单层决策树对应的预测类别
    minError:决策树对应的分类误差率
    """
    n, m = shape(dataMatrix)
    #划分间隔数
    numSteps = 10
    #最小误差
    minError = inf
    #最小误差单层决策树的参数
    dim = None
    thresh = None
    ineq = None
    #最小误差单层决策树的预测类别矩阵
    bestClasEst = {}
    #对于每一个特征值
    for i in range(m):
        #特征列中的最大最小值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        #步长
        stepLength = (rangeMax - rangeMin) / numSteps
        #对于每一个划分点
        for j in range(numSteps+1):
            #计算划分点
            threshVal = rangeMin + j * stepLength
            #对于每一个划分方向
            for oprt in ['lt', 'gt']:
                #预测类别
                predictVals = calcStumpClassify(dataMatrix, i, threshVal, 
                                               oprt)
                #计算错分类矩阵
                errArr = mat(ones((n, 1)))
                errArr[predictVals == labelMat] = 0
                #计算误分类误差
                weightedError = D.T * errArr
                #更新最小误差对应的参数
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictVals.copy()
                    dim = i
                    thresh = threshVal
                    ineq = oprt
        #返回单层决策树和预测类别矩阵
        return stumpClassify(dim, thresh, ineq), bestClasEst, minError
    
#----------------------------------------------------------------------
def calcAlpha(Em):
    """
    计算a系数
    Parameter:
    Em:第m个基本函数对应的分类误差率
    Return:
    am:Gm(x)的系数
    """
    return log((1 - Em) / Em) / 2.0

#----------------------------------------------------------------------
def updateD(Dm, alphas, labelMat, Gmx):
    """"""
    #计算规范化因子
    signs = multiply(labelMat, Gmx)
    scalars = exp(-multiply(alphas, signs))
    Zm = Dm * scalars
    return multiply(scalars, Dm.T) / Zm

#----------------------------------------------------------------------
def Gx(fx):
    """
    生成最终分类器
    Parameter:
    fx:基本分类器的线性组合
    Return:
    输入x的类别
    """
    def signFx(x):
        if fx(x) > 0:
            return 1.0
        elif fx(x) < 0:
            return -1.0
        else:
            return 0
    return signFx

#----------------------------------------------------------------------
def adaBoost(dataArr, classLabels):
    """
    AdaBoost算法
    Parameter:
    dataArr:训练数据集
    classLabels:类别
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    n, m = shape(dataMatrix)
    #初始化权值分布
    D = array([1/float(n)] * n)
    #分类误差率
    em = 1.0
    fx = None
    while em:
        gmx, clasEst, em = buildStump(dataMatrix, labelMat, D)
        #计算a系数
        am = calcAlpha(em)
        #更新权值分布
        gmxi = gmx(dataMatrix)
        D = updateD(D, am, labelMat, gmxi)
        #构建分类器的线性组合
        if fx is None:
            fx = am * gmx
        else:
            fx += am * gmx
    return Gx(fx)


#----------------------------------------------------------------------
def loadDataSet(filename):
    """
    加载数据
    Parameter:
    filename:文件名
    Return:
    dataMat:训练数据集
    labelMat:类别
    """
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


#----------------------------------------------------------------------
def testAdaBoost():
    """
    测试数据
    """
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    Gx = adaBoost(dataArr, labelArr)
    m = len(labelArr)
    error = 0
    for num, data in enumerate(dataArr):
        y = Gx(data)
        if y != labelArr[num]:
            error += 1
    print 'the train set error is %f' %(error/float(m))
    dataArr, labelArr = loadDataSet('horseColicTest2.txt')
    m = len(labelArr)
    error = 0
    for num, data in enumerate(dataArr):
        y = Gx(data)
        if y != labelArr[num]:
            error += 1
    print 'the test set error is %f' %(error/float(m))
    

########################################################################
class BoostTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        """"""
        Dm = mat([[1, 2]])
        alphas = mat([[4], [3]])
        labelMat = mat([[5], [6]])
        Gmx = mat([[8], [7]])
        Zm = exp(-(4 * 5 * 8)) + 2 * exp(-(3 * 6 * 7))
        result = [exp(-(4 * 5 * 8)) / Zm, 2 * exp(-(3 * 6 * 7)) / Zm]
        result = [[i] for i in result]
        self.assertListEqual(result, updateD(Dm, alphas, labelMat, Gmx).tolist())
    
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(BoostTest('test_0'))
    return suite


if __name__ == '__main__':
    #unittest.main(defaultTest='suite')
    testAdaBoost()