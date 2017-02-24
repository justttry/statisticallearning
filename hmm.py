#encoding:UTF-8

import unittest
from numpy import *

#----------------------------------------------------------------------
def calcForwardP(a, t, O, A, B):
    """
    计算前向概率
    Parameter:
    t:t时刻
    O:观测变量矩阵
    a:t时的前向概率向量
    A:状态转移概率矩阵
    B:观测概率分布矩阵
    Return:
    a:t+1时的前向概率向量
    """
    return multiply(A.T * a, B[:, O[0, t]])

#----------------------------------------------------------------------
def calcForwardPMat(O, A, B, pi0):
    """
    计算前向概率矩阵
    Parameter:
    O:观测变量向量
    A:状态转移概率矩阵
    B:观测概率分布矩阵
    pi0:初始状态概率向量
    Return:
    alphas:前向概率矩阵
    """
    t = shape(O)[1]
    n = shape(A)[1]
    alpha = multiply(pi0, B[:, O[0, 0]])
    alphas = mat(zeros((n, t)))
    alphas[:, 0] = alpha
    for i in range(1, t):
        alpha = calcForwardP(alpha, i, O, A, B)
        alphas[:, i] = alpha
    return alphas

#----------------------------------------------------------------------
def calcForwardPMats(O, A, B, pi0):
    """
    计算前向概率矩阵
    Parameter:
    O:观测变量矩阵
    A:状态转移概率矩阵
    B:观测概率分布矩阵
    pi0:初始状态概率向量
    Return:
    alphas:前向概率矩阵
    """
    s, t = shape(O)
    n = shape(A)[1]
    alphasMat = zeros((s, n, t))
    for i in range(s):
        alphas = calcForwardPMat(O, A, B, pi0)
        alphasMat[i] = alphas
    return alphasMat

#----------------------------------------------------------------------
def calcBackwardP(b, t, O, A, B):
    """
    计算后向概率向量
    Parameter:
    b:t+1时刻的后向概率
    t:时刻t
    O:观测变量矩阵
    A:隐马尔科夫模型状态转移矩阵
    B:隐马尔科夫模型观测概率矩阵
    Return:
    b:t时刻的后向概率
    """
    return A * multiply(b, B[:, O[0, t]])

#----------------------------------------------------------------------
def calcBackwardPMat(O, A, B, pi0):
    """
    计算后向概率矩阵
    Parameter:
    O:观测变量向量
    A:状态转移概率矩阵
    B:观测概率分布矩阵
    pi0:初始状态概率向量
    Return:
    betas:后向概率矩阵
    """
    t = shape(O)[1]
    n, m = shape(B)
    beta = mat(ones((n, 1)))
    betas = mat(zeros((n, t)))
    betas[:, t-1] = beta
    for i in range(t-1, 0, -1):
        beta = calcBackwardP(beta, i, O, A, B)
        betas[:, i-1] = beta
    return betas

#----------------------------------------------------------------------
def calcBackwardPMats(O, A, B, pi0):
    """
    计算后向概率矩阵
    Parameter:
    O:观测变量矩阵
    A:状态转移概率矩阵
    B:观测概率分布矩阵
    pi0:初始状态概率向量
    Return:
    betasMat:多观测变量序列的后向概率矩阵
    """
    s, t = shape(O)
    n, m = shape(B)
    betasMat = zeros((s, n, t))
    for i in range(s):
        betas = calcBackwardPMat(O, A, B, pi0)
        betasMat[i] = betas
    return betasMat
    

#----------------------------------------------------------------------
def calcGamma0(alpha, beta):
    """
    计算gamma=P(i=qi|O, lamda)
    Parameter:
    alpha:t时刻的前向概率向量
    beta:t时刻的后向概率向量
    Return:
    numerator:Gamma的分子
    demoninator:Gamma的分母
    """
    numerator = multiply(alpha, beta)
    return numerator, sum(numerator)


#----------------------------------------------------------------------
def calcGamma(alpha, beta):
    """
    计算gamma=P(i=qi|O, lamda)
    Parameter:
    alpha:t时刻的前向概率向量
    beta:t时刻的后向概率向量
    Return:
    gamma:给定观测序列情况下，t时刻i状态出现的概率
    """
    gamma = calcGamma0(alpha, beta)
    return gamma[0] / gamma[1]

#----------------------------------------------------------------------
def calcGammaMat(alphas, betas):
    """
    计算gamma分子矩阵，即P(it=qi, O|lamda)矩阵
    Parameter:
    alphas:前向概率矩阵
    betas:后向概率矩阵
    Return:
    P(it=qi, O|lamda)矩阵
    """
    return multiply(alphas, betas)


#----------------------------------------------------------------------
def calcGammaMats(alphasMat, betasMat):
    """
    计算gamma分子矩阵，即P(it=qi, O|lamda)矩阵
    Parameter:
    alphasMat:前向概率矩阵
    betasMat:后向概率矩阵
    Return:
    P(it=qi, O|lamda)矩阵
    """
    return multiply(alphasMat, betasMat)


#----------------------------------------------------------------------
def calcEpsilon0(alpha, beta, A, B, O, t):
    """
    计算Epsilon=P(i=qi, j=qj|O, lamda)
    Parameter:
    alpha:t时刻的前向概率向量
    beta:t+1时刻的后向概率向量
    A:隐马尔可夫模型状态转移矩阵
    B:隐马尔可夫模型概率分布矩阵
    O:观测序列
    t:时刻t
    Return:
    numerator:Epsilon的分子
    demoninator:Epsilon的分母
    """
    b = B[:, O[0, t]]
    result = multiply(multiply(alpha, A), multiply(b, beta).T)
    return result, sum(result)
    

#----------------------------------------------------------------------
def calcEpsilon(alpha, beta, A, B, O, t):
    """
    计算Epsilon=P(i=qi, j=qj|O, lamda)
    Parameter:
    alpha:t时刻的前向概率向量
    beta:t+1时刻的后向概率向量
    A:隐马尔可夫模型状态转移矩阵
    B:隐马尔可夫模型概率分布矩阵
    O:观测序列
    t:时刻t
    Return:
    epsilon:给定观测序列的情况下，t时刻为i状态，t+1时刻为j状态的概率
    """
    epsilon = calcEpsilon0(alpha, beta, A, B, O, t)
    return epsilon[0] / epsilon[1]

#----------------------------------------------------------------------
def calcEpsilonMat(alphas, betas, A, B, O):
    """
    计算Epsilon分子矩阵，即P(it=qi, it_1=qj, O| lamda)矩阵
    Parameter:
    alphas:前向概率矩阵
    betas:后向概率矩阵
    A:隐马尔可夫模型状态转移矩阵
    B:隐马尔可夫模型概率分布矩阵
    O:观测序列向量
    Return:
    P(it=qi, it_1=qj, O| lamda)矩阵
    """
    t = shape(O)[1]
    n, m = shape(B)
    epsilons = zeros((t-1, n, n))
    for i in range(1, t):
        epsilon = multiply(multiply(alphas[:, i-1], A), 
                           multiply(B[:, O[0, i]], betas[:, i]).T)
        epsilons[i-1] = epsilon
    return epsilons

#----------------------------------------------------------------------
def calcEpsilonMats(alphasMat, betasMat, A, B, O):
    """
    计算Epsilon分子矩阵，即P(it=qi, it_1=qj, O| lamda)矩阵
    Parameter:
    alphasMat:前向概率矩阵
    betasMat:后向概率矩阵
    A:隐马尔可夫模型状态转移矩阵
    B:隐马尔可夫模型概率分布矩阵
    O:观测序列矩阵
    Return:
    P(it=qi, it_1=qj, O| lamda)矩阵
    """
    s, t = shape(O)
    n, m = shape(B)
    epsilonsMat = zeros((s, t-1, n, n))
    for i in range(s):
        epsilons = calcEpsilonMat(mat(alphasMat[i]), mat(betasMat[i]), A, B, O[i])
        epsilonsMat[i] = epsilons
    return epsilonsMat

#----------------------------------------------------------------------
def forwardAlgo(pi0, A, B, O):
    """
    前向算法
    pi0:隐马尔可夫模型的初始状态概率向量
    A:隐马尔可夫模型状态转移概率矩阵
    B:隐马尔可夫模型观测概率分布矩阵
    O:观测序列
    Return:
    p:观测序列的概率
    """
    #计算观测次数
    t = shape(O)[1]
    #计算前向概率的初值
    a = multiply(pi0, B[:, O[0, 0]])
    for i in range(1, t):
        a = calcForwardP(a, i, O, A, B)
    return sum(a)

#----------------------------------------------------------------------
def backwardAlgo(O, A, B, pi0):
    """
    后向算法
    Parameter:
    A:隐马尔科夫模型状态转移矩阵
    B:隐马尔科夫模型观测概率矩阵
    O:观测变量矩阵
    pi0:初始概率分布
    Return:
    观测序列的概率
    """
    O = mat(O)
    A = mat(A)
    B = mat(B)
    pi0 = mat(pi0)
    n = shape(A)[0]
    t = shape(O)[1]
    #初始化后向概率
    b = mat(ones((n, 1)))
    #计算b1(i)
    for i in range(t-1, 0, -1):
        b = calcBackwardP(b, i, O, A, B)
    #返回P(O|lamda)
    return pi0.T * multiply(b, B[:, O[0, 0]])

#----------------------------------------------------------------------
def calcPi0(gammaMat):
    """
    根据状态条件概率分布计算pi0
    Parameter:
    gammaMat:状态条件概率分布
    Return:
    pi0:隐马尔可夫模型初始概率
    """
    tmp = mat(sum(gammaMat, axis=0)[:, 0]).T
    return tmp / sum(tmp)

#----------------------------------------------------------------------
def calcA(epsilonMat):
    """
    根据联合条件概率分布计算A
    Parameter:
    epsilonMat:联合条件概率分布
    Return:
    A:状态转移概率矩阵
    """
    tmp = sum(epsilonMat, axis=0)
    tmp = sum(tmp, axis=0)
    sums = sum(tmp, axis=0)
    return mat(tmp / sums)

#----------------------------------------------------------------------
def calcI(O, m):
    """
    计算指示函数I(ot=vk)
    Parameter:
    O:观测变量
    m:变量可能的取值数
    Return:
    I:指示函数I(ot=vk)
    """
    t = shape(O)[1]
    I = zeros((t, m))
    #当ot==vk时,I[t, k] = 1
    #O中元素的坐标（时刻）和值确定I中的横纵坐标
    I[[range(t), O]] = 1
    return mat(I)

#----------------------------------------------------------------------
def calcIs(O, m):
    """
    计算指示函数集
    Parameter:
    O:观测变量矩阵
    m:变量可能的取值数
    Return:
    Is:指示函数集
    """
    s, t = shape(O)
    Is = zeros((s, t, m))
    for i in range(s):
        instr = calcI(O[i], m)
        Is[i] = instr
    return Is
    

#----------------------------------------------------------------------
def calcB(gammaMat, O, m):
    """
    根据状态条件概率分布计算B
    Parameter:
    gammaMat:状态条件概率分布矩阵
    O:观测变量矩阵
    m:观测变量的可能取值数
    Return:
    B:观测概率分布矩阵
    """
    s, n, t = shape(gammaMat)
    #计算指示函数矩阵
    Is = calcIs(O, m)
    numeratorMat = zeros((s, n, m))
    denominatorMat = zeros((s, n, m))
    I = ones((t, m))
    for i in range(s):
        numerator = dot(gammaMat[i], Is[i])
        denominator = dot(gammaMat[i], I)
        numeratorMat[i] = numerator
        denominatorMat[i] = denominator
    numer = sum(numeratorMat, axis=0)
    denom = sum(denominatorMat, axis=0)
    return mat(numer / denom)


#----------------------------------------------------------------------
def BaumWelch(O, N, M, theta=1e-7):
    """
    Baum-Welch算法
    Parameter:
    O:观测数据矩阵
    N:可能的隐状态数
    M:可能的观测数
    Return:
    A:隐马尔可夫模型状态转移矩阵
    B:隐马尔可夫模型概率分布矩阵
    pi0:隐马尔可夫模型初始概率
    """
    #选取模型初值
    A = mat(random.random((N, N)))
    A = A / sum(A, axis=0)
    B = mat(random.random((N, M)))
    B = B / sum(B, axis=0)
    pi0 = mat(random.random((N, 1)))
    pi0 = pi0 / sum(pi0, axis=0)
    error = sum(abs(A))
    cnt = 0
    #停止条件
    while error > theta:
        #计算前向概率矩阵
        alphasMat = calcForwardPMats(O, A, B, pi0)
        #计算后向概率矩阵
        betasMat = calcBackwardPMats(O, A, B, pi0)
        #计算状态条件概率矩阵
        gammaMat = calcGammaMats(alphasMat, betasMat)
        #计算联合条件概率矩阵
        epsilonMat = calcEpsilonMats(alphasMat, betasMat, A, B, O)
        #更新pi0
        pi0 = calcPi0(gammaMat)
        #更新A
        newA = calcA(epsilonMat)
        #更新B
        B = calcB(gammaMat, O, M)
        #计算error
        error = sum(abs(A - newA))
        A = newA
        #验证，计算P(O|lamda)并打印
        prob = 1.0
        for i in O:
            prob *= forwardAlgo(pi0, A, B, i)
        print '-----start------------'
        print 'prob: ', prob
        print 'cnt:  ', cnt
        print 'error:', error
        cnt += 1
    return A, B, pi0
        
    

########################################################################
class HMMTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        A = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        B = [[1.1, 2.1,  3.1,  4.1],
             [5.1, 6.1,  7.1,  8.1],
             [9.1, 10.1, 11.1, 12.1]]
        a = [[1.2], [2.2], [3.2]]
        t1 = 3
        O = [[0, 1, 2, 3]]
        tmp = [[1 * 1.2 + 4 * 2.2 + 7 * 3.2],
               [2 * 1.2 + 5 * 2.2 + 8 * 3.2],
               [3 * 1.2 + 6 * 2.2 + 9 * 3.2]]
        result = [[tmp[0][0] * B[0][t1]],
                  [tmp[1][0] * B[1][t1]],
                  [tmp[2][0] * B[2][t1]]]
        self.assertListEqual(calcForwardP(mat(a), 
                                          t1, 
                                          mat(O),
                                          mat(A), 
                                          mat(B)).tolist(), 
                             result)
        
    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        A = [[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]]
        B = [[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]]
        pi0 = [[0.2], [0.4], [0.4]]
        O = [[0, 1, 0]]
        self.assertAlmostEqual(forwardAlgo(mat(pi0),
                                           mat(A),
                                           mat(B),
                                           mat(O)),
                               0.13022, delta=0.00001)

    #----------------------------------------------------------------------
    def test_2(self):
        A = [[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]]
        B = [[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]]
        pi0 = [[0.2], [0.4], [0.4]]
        b = [[1], [1], [1]]
        O = [[0, 1, 0]]
        tmp = [[b[0][0]*B[0][0], b[0][0]*B[0][1]],
               [b[1][0]*B[1][0], b[1][0]*B[1][1]],
               [b[2][0]*B[2][0], b[2][0]*B[2][1]]]
        result = [[A[0][0]*tmp[0][0] + A[0][1]*tmp[1][0] + A[0][2]*tmp[2][0]],
                  [A[1][0]*tmp[0][0] + A[1][1]*tmp[1][0] + A[1][2]*tmp[2][0]],
                  [A[2][0]*tmp[0][0] + A[2][1]*tmp[1][0] + A[2][2]*tmp[2][0]]]
        b = calcBackwardP(mat(b), 
                          2, 
                          mat(O), 
                          mat(A), 
                          mat(B))
        self.assertListEqual(b.tolist(), result)
    
    #----------------------------------------------------------------------
    def test_3(self):
        """"""
        A = [[0.5, 0.2, 0.3],
             [0.3, 0.5, 0.2],
             [0.2, 0.3, 0.5]]
        B = [[0.5, 0.5],
             [0.4, 0.6],
             [0.7, 0.3]]
        pi0 = [[0.2], [0.4], [0.4]]
        O = [[0, 1, 0]]
        prob = backwardAlgo(O, A, B, pi0)
        self.assertAlmostEqual(prob[0, 0], 0.13022, delta=0.00001)
        
    #----------------------------------------------------------------------
    def test_calcGamma(self):
        """"""
        alpha = mat([[1], [2], [3]])
        beta = mat([[4], [5], [6]])
        result = mat([[alpha[0, 0]*beta[0, 0]], 
                      [alpha[1, 0]*beta[1, 0]],
                      [alpha[2, 0]*beta[2, 0]]])
        result = result / sum(result)
        self.assertListEqual(calcGamma(alpha, beta).tolist(),
                             result.tolist())
        
    #----------------------------------------------------------------------
    def test_calcEpsilon(self):
        """"""
        A = mat([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
        B = mat([[10.0, 11.0],
                 [12.0, 13.0],
                 [14.0, 15.0]])
        O = mat([[0, 1, 0, 1, 0]])
        t = 4
        alpha = mat([[16.0],
                     [17.0],
                     [18.0]])
        beta = mat([[19.0],
                    [20.0],
                    [21.0]])
        b = mat([[10.0],
                 [12.0],
                 [14.0]])
        e11 = alpha[0, 0] * A[0, 0] * b[0, 0] * beta[0, 0]
        e12 = alpha[0, 0] * A[0, 1] * b[1, 0] * beta[1, 0]
        e13 = alpha[0, 0] * A[0, 2] * b[2, 0] * beta[2, 0]
        e21 = alpha[1, 0] * A[1, 0] * b[0, 0] * beta[0, 0]
        e22 = alpha[1, 0] * A[1, 1] * b[1, 0] * beta[1, 0]
        e23 = alpha[1, 0] * A[1, 2] * b[2, 0] * beta[2, 0]
        e31 = alpha[2, 0] * A[2, 0] * b[0, 0] * beta[0, 0]
        e32 = alpha[2, 0] * A[2, 1] * b[1, 0] * beta[1, 0]
        e33 = alpha[2, 0] * A[2, 2] * b[2, 0] * beta[2, 0]
        e = mat([[e11, e12, e13],
                 [e21, e22, e23],
                 [e31, e32, e33]])
        e = e / (e11 + e12 + e13 + e21 + e22 + e23 + e31 + e32 + e33)
        result = calcEpsilon(alpha, beta, A, B, O, t)
        self.assertListEqual(e.tolist(), 
                             result.tolist())
    
    #----------------------------------------------------------------------
    def test_calcForwardPMat(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0]])
        alphas = mat(zeros((3, 3)))
        alpha = multiply(pi0, B[:, O[0, 0]])
        alphas[:, 0] = alpha
        for i in range(1, 3):
            alpha = calcForwardP(alpha, i, O, A, B)
            alphas[:, i] = alpha
        newalphas = calcForwardPMat(O, A, B, pi0)
        self.assertListEqual(alphas.tolist(), newalphas.tolist())
    
    #----------------------------------------------------------------------
    def test_calcForwardPMats(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0],
                 [0, 1, 0]])
        alphas = mat(zeros((3, 3)))
        alpha = multiply(pi0, B[:, O[0, 0]])
        alphas[:, 0] = alpha
        for i in range(1, 3):
            alpha = calcForwardP(alpha, i, O, A, B)
            alphas[:, i] = alpha
        newalphas = calcForwardPMats(O, A, B, pi0)
        self.assertListEqual([alphas.tolist(), alphas.tolist()], newalphas.tolist())
        
    #----------------------------------------------------------------------
    def test_calcBackwardPMat(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0]])
        betas = calcBackwardPMat(O, A, B, pi0)
        result = multiply(pi0, B[:, O[0, 0]]).T * betas[:, 0]
        self.assertAlmostEqual(result[0, 0], 0.13022, delta=0.00001)
        alphas = calcForwardPMat(O, A, B, pi0)
        self.assertEqual(result[0, 0], sum(alphas, axis=0)[0, shape(O)[1]-1])
        
    #----------------------------------------------------------------------
    def test_calcBackwardPMats(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0],
                 [0, 1, 0],
                 [0, 1, 0]])
        n, m = shape(B)
        s, t = shape(O)
        betas = calcBackwardPMat(O, A, B, pi0)
        betasMat = calcBackwardPMats(O, A, B, pi0)
        self.assertEqual(betasMat.tolist(), [betas.tolist()]*s)
        
    #----------------------------------------------------------------------
    def test_calcGammaMat(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0]])
        alphas = calcForwardPMat(O, A, B, pi0)
        betas = calcBackwardPMat(O, A, B, pi0)
        t = shape(O)[1]
        n = shape(A)[1]
        gammas0 = mat(zeros((n, t)))
        for i in range(1, t+1):
            gamma = multiply(alphas[:, i-1], betas[:, i-1])
            gammas0[:, i-1] = gamma
        gammas = calcGammaMat(alphas, betas)
        self.assertListEqual(gammas0.tolist(), gammas.tolist())
        
    #----------------------------------------------------------------------
    def test_calcGammaMats(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0]])
        O1 = mat([[0, 1, 0],
                  [0, 1, 0]])
        alphas = calcForwardPMat(O, A, B, pi0)
        betas = calcBackwardPMat(O, A, B, pi0)
        t = shape(O)[1]
        n = shape(A)[1]
        gammas0 = mat(zeros((n, t)))
        for i in range(1, t+1):
            gamma = multiply(alphas[:, i-1], betas[:, i-1])
            gammas0[:, i-1] = gamma
        gammas = calcGammaMat(alphas, betas)
        self.assertListEqual(gammas0.tolist(), gammas.tolist())
        alphasMat = calcForwardPMats(O1, A, B, pi0)
        betasMat = calcBackwardPMats(O1, A, B, pi0)
        gammasMat = calcGammaMats(alphasMat, betasMat)
        self.assertListEqual(gammasMat.tolist(), [gammas.tolist()]*2)
        
    #----------------------------------------------------------------------
    def test_calcEpsilonMat(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0]])
        alphas = calcForwardPMat(O, A, B, pi0)
        betas = calcBackwardPMat(O, A, B, pi0)
        t = shape(O)[1]
        n, m = shape(B)
        epsilons0 = zeros((t-1, n, n))
        for i in range(1, t):
            epsilon = calcEpsilon0(alphas[:, i-1], 
                                   betas[:, i], 
                                   A, 
                                   B, 
                                   O, 
                                   i)[0]
            epsilons0[i-1] = epsilon
        epsilons = calcEpsilonMat(alphas, betas, A, B, O)
        self.assertListEqual(epsilons.tolist(), epsilons0.tolist())
        
    #----------------------------------------------------------------------
    def test_calcEpsilonMats(self):
        """"""
        A = mat([[0.5, 0.2, 0.3],
                 [0.3, 0.5, 0.2],
                 [0.2, 0.3, 0.5]])
        B = mat([[0.5, 0.5],
                 [0.4, 0.6],
                 [0.7, 0.3]])
        pi0 = mat([[0.2], [0.4], [0.4]])
        O = mat([[0, 1, 0]])
        alphas = calcForwardPMat(O, A, B, pi0)
        betas = calcBackwardPMat(O, A, B, pi0)
        t = shape(O)[1]
        n, m = shape(B)
        epsilons0 = zeros((t-1, n, n))
        for i in range(1, t):
            epsilon = calcEpsilon0(alphas[:, i-1], 
                                   betas[:, i], 
                                   A, 
                                   B, 
                                   O, 
                                   i)[0]
            epsilons0[i-1] = epsilon
        epsilons = calcEpsilonMat(alphas, betas, A, B, O)
        self.assertListEqual(epsilons.tolist(), epsilons0.tolist())
        O1 = mat([[0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0]])
        alphasMat = calcForwardPMats(O1, A, B, pi0)
        betasMat = calcBackwardPMats(O1, A, B, pi0)
        epsilonsMat = calcEpsilonMats(alphasMat, betasMat, A, B, O1)
        self.assertListEqual(epsilonsMat.tolist(),
                             [epsilons.tolist()]*3)
        
    #----------------------------------------------------------------------
    def test_calcPi0(self):
        """"""
        s = 2
        n = 3
        t = 4
        gammas = random.random((s, n, t))
        gammas0 = gammas[0, 0, 0] + gammas[1, 0, 0]
        gammas1 = gammas[0, 1, 0] + gammas[1, 1, 0]
        gammas2 = gammas[0, 2, 0] + gammas[1, 2, 0]
        pi0 = mat([[gammas0, gammas1, gammas2]]).T / (gammas0+gammas1+gammas2)
        newpi = calcPi0(gammas)
        self.assertListEqual(pi0.tolist(), newpi.tolist())
        
    #----------------------------------------------------------------------
    def test_calcA(self):
        """"""
        s = 2
        t = 2
        n = 3
        eps = random.random((2, 2, 3, 3))
        a11 = eps[0, 0, 0, 0] + eps[1, 0, 0, 0] + eps[0, 1, 0, 0] + eps[1, 1, 0, 0]
        a12 = eps[0, 0, 0, 1] + eps[1, 0, 0, 1] + eps[0, 1, 0, 1] + eps[1, 1, 0, 1]
        a13 = eps[0, 0, 0, 2] + eps[1, 0, 0, 2] + eps[0, 1, 0, 2] + eps[1, 1, 0, 2]
        a21 = eps[0, 0, 1, 0] + eps[1, 0, 1, 0] + eps[0, 1, 1, 0] + eps[1, 1, 1, 0]
        a22 = eps[0, 0, 1, 1] + eps[1, 0, 1, 1] + eps[0, 1, 1, 1] + eps[1, 1, 1, 1]
        a23 = eps[0, 0, 1, 2] + eps[1, 0, 1, 2] + eps[0, 1, 1, 2] + eps[1, 1, 1, 2]
        a31 = eps[0, 0, 2, 0] + eps[1, 0, 2, 0] + eps[0, 1, 2, 0] + eps[1, 1, 2, 0]
        a32 = eps[0, 0, 2, 1] + eps[1, 0, 2, 1] + eps[0, 1, 2, 1] + eps[1, 1, 2, 1]
        a33 = eps[0, 0, 2, 2] + eps[1, 0, 2, 2] + eps[0, 1, 2, 2] + eps[1, 1, 2, 2]
        result = mat([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        result = result / mat([a11+a21+a31, a12+a22+a32, a13+a23+a33])
        A = calcA(eps)
        #self.assertListEqual(result.tolist(), A.tolist())
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(result[i, j], A[i, j], delta=0.000001)
                
    #----------------------------------------------------------------------
    def test_calcI(self):
        """"""
        #om = 0, 1, 2, 3
        m = 4
        O = mat([[1, 2, 3, 2, 1]])
        result = mat([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0]])
        self.assertListEqual(calcI(O, m).tolist(), result.tolist())
                
    #----------------------------------------------------------------------
    def test_calcIs(self):
        """"""
        #om = 0, 1, 2, 3
        m = 4
        O = mat([[1, 2, 3, 2, 1],
                 [2, 3, 1, 0, 0]])
        result0 = [[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]]
        result1 = [[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0]]
        result = array([result0, result1])
        self.assertListEqual(calcIs(O, m).tolist(), result.tolist())
                
    #----------------------------------------------------------------------
    def test_calcB(self):
        """"""
        gammaMat = random.random((2, 3, 4))
        s, n, t = shape(gammaMat)
        m = 2
        O = mat([[0, 1, 1, 0],
                 [1, 0, 0, 0]])
        n111 = gammaMat[0, 0, 0] + gammaMat[0, 0, 3]
        n112 = gammaMat[0, 0, 1] + gammaMat[0, 0, 2]
        n121 = gammaMat[0, 1, 0] + gammaMat[0, 1, 3]
        n122 = gammaMat[0, 1, 1] + gammaMat[0, 1, 2]
        n131 = gammaMat[0, 2, 0] + gammaMat[0, 2, 3]
        n132 = gammaMat[0, 2, 1] + gammaMat[0, 2, 2]

        n211 = gammaMat[1, 0, 1] + gammaMat[1, 0, 2] + gammaMat[1, 0, 3]
        n212 = gammaMat[1, 0, 0]
        n221 = gammaMat[1, 1, 1] + gammaMat[1, 1, 2] + gammaMat[1, 1, 3]
        n222 = gammaMat[1, 1, 0]
        n231 = gammaMat[1, 2, 1] + gammaMat[1, 2, 2] + gammaMat[1, 2, 3]
        n232 = gammaMat[1, 2, 0]        
        
        n11 = n111 + n211
        n12 = n112 + n212
        n21 = n121 + n221
        n22 = n122 + n222
        n31 = n131 + n231
        n32 = n132 + n232
        
        n = array([[n11, n12],
                   [n21, n22],
                   [n31, n32]])
        
        d111 = gammaMat[0, 0, 0] + gammaMat[0, 0, 1] + gammaMat[0, 0, 2] + gammaMat[0, 0, 3]
        d112 = gammaMat[0, 0, 0] + gammaMat[0, 0, 1] + gammaMat[0, 0, 2] + gammaMat[0, 0, 3]
        d121 = gammaMat[0, 1, 0] + gammaMat[0, 1, 1] + gammaMat[0, 1, 2] + gammaMat[0, 1, 3]
        d122 = gammaMat[0, 1, 0] + gammaMat[0, 1, 1] + gammaMat[0, 1, 2] + gammaMat[0, 1, 3]
        d131 = gammaMat[0, 2, 0] + gammaMat[0, 2, 1] + gammaMat[0, 2, 2] + gammaMat[0, 2, 3]
        d132 = gammaMat[0, 2, 0] + gammaMat[0, 2, 1] + gammaMat[0, 2, 2] + gammaMat[0, 2, 3]

        d211 = gammaMat[1, 0, 0] + gammaMat[1, 0, 1] + gammaMat[1, 0, 2] + gammaMat[1, 0, 3]
        d212 = gammaMat[1, 0, 0] + gammaMat[1, 0, 1] + gammaMat[1, 0, 2] + gammaMat[1, 0, 3]
        d221 = gammaMat[1, 1, 0] + gammaMat[1, 1, 1] + gammaMat[1, 1, 2] + gammaMat[1, 1, 3]
        d222 = gammaMat[1, 1, 0] + gammaMat[1, 1, 1] + gammaMat[1, 1, 2] + gammaMat[1, 1, 3]
        d231 = gammaMat[1, 2, 0] + gammaMat[1, 2, 1] + gammaMat[1, 2, 2] + gammaMat[1, 2, 3]
        d232 = gammaMat[1, 2, 0] + gammaMat[1, 2, 1] + gammaMat[1, 2, 2] + gammaMat[1, 2, 3] 
        
        d11 = d111 + d211
        d12 = d112 + d212
        d21 = d121 + d221
        d22 = d122 + d222
        d31 = d131 + d231
        d32 = d132 + d232
        
        d = array([[d11, d12],
                   [d21, d22],
                   [d31, d32]])
        
        result = n / d
        
        self.assertListEqual(calcB(gammaMat, O, m).tolist(), result.tolist())
        
    #----------------------------------------------------------------------
    def test_BaumWelch(self):
        """"""
        #O = mat([[1, 2, 3, 3, 2, 1, 0, 3],
                 #[0, 1, 2, 0, 2, 3, 1, 0]])
        O = mat(random.choice(range(4), (1000, 8)))
        M = 4
        N = 5
        print BaumWelch(O, N, M)

#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(HMMTest('test_0'))
    suite.addTest(HMMTest('test_1'))
    suite.addTest(HMMTest('test_2'))
    suite.addTest(HMMTest('test_3'))
    suite.addTest(HMMTest('test_calcGamma'))
    suite.addTest(HMMTest('test_calcEpsilon'))
    suite.addTest(HMMTest('test_calcForwardPMat'))
    suite.addTest(HMMTest('test_calcForwardPMats'))
    suite.addTest(HMMTest('test_calcBackwardPMat'))
    suite.addTest(HMMTest('test_calcBackwardPMats'))
    suite.addTest(HMMTest('test_calcGammaMat'))
    suite.addTest(HMMTest('test_calcGammaMats'))
    suite.addTest(HMMTest('test_calcEpsilonMat'))
    suite.addTest(HMMTest('test_calcEpsilonMats'))
    suite.addTest(HMMTest('test_calcPi0'))
    suite.addTest(HMMTest('test_calcA'))
    suite.addTest(HMMTest('test_calcI'))
    suite.addTest(HMMTest('test_calcIs'))
    suite.addTest(HMMTest('test_calcB'))
    suite.addTest(HMMTest('test_BaumWelch'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')