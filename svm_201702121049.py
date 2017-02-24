#encoding:UTF-8

import unittest
from numpy import *
from svm_assist import differences as cmp

linearKernel = lambda i, j: i * j

########################################################################
class optStruct(object):
    """"""
    
    #----------------------------------------------------------------------
    def __init__(self, dataInMat, classLabels, C, toler):
        """Constructor"""
        self.X = dataInMat
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataInMat)[0]
        self.alphas = zeros((self.m, 1))
        self.b = 0
        self.E = zeros((self.m, 1))
        
#----------------------------------------------------------------------
def calcPrimary(oS):
    """
    求解最优问题的值
    Parameter:
    oS:optStruct
    Return:
    最优值
    """
    tmp = multiply(oS.alphas, oS.labelMat)
    tmp = multiply(tmp, oS.X)
    return sum(tmp * tmp.T)/2.0 - sum(oS.alphas)

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
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

#----------------------------------------------------------------------
def calcEi(oS, i):
    """
    计算并存储Ei
    Parameter:
    oS:optStruct
    i:特征值参数
    Return:
    Ei
    """
    fXi = float(multiply(oS.alphas, oS.labelMat).T * \
                (oS.X * oS.X[i].T)) + oS.b
    Ei = fXi - oS.labelMat[i]
    oS.E[i] = Ei[0, 0]
    return Ei

#----------------------------------------------------------------------
def calcG(oS, x):
    """
    计算预测类别
    Parameter:
    oS:optStruct
    x:输入
    Return:
    g(x)
    """
    return float(multiply(oS.alphas, oS.labelMat).T * \
                 (oS.X * x.T)) + oS.b

#----------------------------------------------------------------------
def calcEta(oS, i, j, kernel=linearKernel):
    """
    计算Eta = K11 + K22 - 2 * K12
    Parameter:
    oS:optStruct
    i:第一个变量的序号
    j:第二个变量的序号
    kernel:核函数，缺省为线性函数
    Return:
    eta:Eta
    """
    x1 = oS.X[i, :]
    x2 = oS.X[j, :]
    return kernel(x1, x1.T) + kernel(x2, x2.T) - 2 * kernel(x1, x2.T)

#----------------------------------------------------------------------
def updateAlphas(oS, i, j, kernel=linearKernel):
    """
    计算并更新optStruct中的alpha值
    Parameter:
    oS:optStruct
    i:第一个变量的序号
    j:第二个变量的序号
    kernel:核函数，缺省为线性函数
    Return:
    """
    #查找a1和a2
    alphas = oS.alphas.copy()
    a1 = oS.alphas[i].copy()
    a2 = oS.alphas[j].copy()
    #查找y1和y2
    y1 = oS.labelMat[i]
    y2 = oS.labelMat[j]
    #计算边界
    if y1 == y2:
        H1 = min(oS.C, a1 + a2)
        L1 = max(0, a1 + a2 - oS.C)
        H2 = min(oS.C, a1 + a2)
        L2 = max(0, a1 + a2 - oS.C)
    else:
        H1 = min(oS.C, oS.C + a1 - a2)
        L1 = max(0, a1 - a2)
        H2 = min(oS.C, oS.C + a2 - a1)
        L2 = max(0, a2 - a1)
    #查找E1和E2
    E1 = oS.E[i].copy()
    E2 = oS.E[j].copy()
    #计算eta
    eta = calcEta(oS, i, j, kernel)
    #计算a2new
    a2new = a2 + y2 * (E1 - E2) / eta[0, 0]
    if a2new >= H2:
        a2new = H2
    elif a2new <= L2:
        a2new = L2
    #如果未进化，则返回False
    if a2new == a2:
        return False
    #计算a1new
    a1new = a1 + y1 * y2 * (a2 - a2new)
    if a1new >= H1:
        a1new = H1
    elif a1new <= L1:
        a1new = L1
    #更新a1和a2
    oS.alphas[j] = a2new
    oS.alphas[i] = a1new
    #glocals()['_debug'] = True
    if _debug:
        print cmp(mat(alphas), mat(oS.alphas))
        print 'result:' 
        print calcPrimary(oS)
        print '\n'
    #计算E1和E2
    x1 = oS.X[i]
    x2 = oS.X[j]
    #计算b1和b2
    b1 = -E1 - y1 * kernel(x1, x1.T) * (a1new - a1) - \
        y2 * kernel(x1, x2.T) * (a2new - a2) + oS.b
    b2 = -E2 - y1 * kernel(x1, x2.T) * (a1new - a1) - \
        y2 * kernel(x2, x2.T) * (a2new - a2) + oS.b
    if 0 < a1new < oS.C:
        oS.b = b1
    elif 0 < a2new < oS.C:
        oS.b = b2
    else:
        oS.b = (b1 + b2) / 2
    #计算E1new和E2new
    E1new = calcEi(oS, i)
    E2new = calcEi(oS, j)
    return True

#----------------------------------------------------------------------
def selectJrand(oS, alphaList, i):
    """
    在非边界集合中随机寻找一个特征值
    Parameter:
    oS:optStruct
    alphaList:特征列表
    i:第一个特征值的坐标
    Return:
    e:Ei误差
    j:对应的第二个特征值的坐标
    """
    j = random.choice(alphaList[alphaList!=i])
    return oS.E[i] - oS.E[j], j
    

#----------------------------------------------------------------------
def selectJInList(oS, alphaList, i):
    """
    在列表中查找使Ei下降最快的特征值
    Parameter:
    oS:optStruct
    alphaList:特征列表
    i:第一个特征值的坐标
    Return:
    maxE:最大Ei误差
    maxJ:对应的第二个特征值的坐标
    """
    if len(alphaList) == 0:
        return 0, None
    maxE = 0
    alphaList = alphaList[alphaList!=i]
    maxJ = random.choice(alphaList)
    for j in alphaList:
        if i == j:
            continue
        E1 = oS.E[i]
        E2 = oS.E[j]
        error = abs(E1 - E2)
        if maxE < error:
            maxE = error
            maxJ = j
    return maxE, maxJ

#----------------------------------------------------------------------
def selectIJInList(oS, alphaList):
    """"""
    for i in alphaList:
        ai = oS.alphas[i]
        yi = oS.labelMat[i]
        Ei = calcEi(oS, i)
        #如果不满足KKT条件，查找第二个参数
        if (ai < oS.C and yi * Ei < -oS.tol) or \
           (ai > 0 and yi * Ei > oS.tol):
            a2 = selectJ(oS, i)
            if a2 is not None:
                return (i, a2)
    return (None, None)
    

#----------------------------------------------------------------------
def selectJ(oS, i):
    """
    启发式方式查找第二个参数
    Parameter:
    oS:optStruct
    i:第一个参数的序号
    Return:
    maxJ:第二个参数的序号
    """
    maxJ = 0
    maxE = 0
    E1 = oS.E[i]
    #在间隔边界上的支持向量点
    svmList = nonzero((oS.alphas>0)*(oS.alphas<oS.C))[0]
    #不在间隔边界上的支持向量点
    nonsvmList = nonzero((oS.alphas==0)+(oS.alphas==oS.C))[0]
    #查找在间隔边界上的支持向量点
    maxE, maxJ = selectJInList(oS, svmList, i)
    #a2使目标函数有足够的下降
    if maxE > oS.tol:
        return maxJ
    #若在支持向量点中，没有足够的下降，遍历训练数据集
    maxE, maxJ = selectJrand(oS, nonsvmList, i)
    if maxE > oS.tol:
        return maxJ
    #若整个训练数据集中找不到合适的第二个特征，则返回None
    else:
        return None
        
#----------------------------------------------------------------------
def selectIJ(oS):
    """
    查找a1和a2
    Parameter:
    oS:optStruct
    Return:
    (a1, a2):a1/a2对应的参数序号
    """
    #在间隔边界上的支持向量点
    svmList = nonzero((oS.alphas>0)*(oS.alphas<oS.C))[0]
    #不在间隔边界上的支持向量点
    svmListC = nonzero((oS.alphas==oS.C))[0]
    svmListZero = nonzero((oS.alphas==0))[0]
    #首先查找间隔边界上的支持向量点
    for alphaList in [svmList, svmListC, svmListZero]:
        (a1, a2) = selectIJInList(oS, alphaList)
        if a1 is not None:
            #print 'find %d:%f %d:%f' %(a1, oS.alphas[a1], a2, oS.alphas[a2])
            return (a1, a2)
    if a1 is not None:
        #print 'find %d:%f %d:%f' %(a1, oS.alphas[a1], a2, oS.alphas[a2])
        pass
    else:
        print 'find none!'
    return (a1, a2)


#----------------------------------------------------------------------
def smoP(dataInMat, labelMat):
    """
    SMO算法
    Parameter:
    dataInMat:训练数据
    labelMat:类别
    C:C
    Return:
    oS:优化后的optStruct
    """
    dataInMat = mat(dataInMat)
    labelMat = mat(labelMat).T
    C = 200
    toler = 0.001
    oS = optStruct(dataInMat, labelMat, C, toler)
    a1 = True
    while a1 is not None:
        a1, a2 = selectIJ(oS)
        if a1 is not None:
            updateAlphas(oS, a1, a2)
    return oS

#----------------------------------------------------------------------
def predictFunc(oS):
    """
    决策函数
    """
    def func(x):
        return sign(calcG(oS, mat(x)))
    return func

#----------------------------------------------------------------------
def testRbf():
    """
    测试数据
    """
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    oS = smoP(dataArr, labelArr)
    predict = predictFunc(oS)
    m = len(labelArr)
    error = 0
    for num, data in enumerate(dataArr):
        y = predict(data)
        if y != labelArr[num]:
            error += 1
    print 'the train set error is %f' %(error/float(m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    m = len(labelArr)
    error = 0
    for num, data in enumerate(dataArr):
        y = predict(data)
        if y != labelArr[num]:
            error += 1
    print 'the test set error is %f' %(error/float(m))

if __name__ == '__main__':
    testRbf()