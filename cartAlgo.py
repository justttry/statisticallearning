#encoding:UTF-8

import unittest
from numpy import *
import operator
import copy

########################################################################
class NodeError(Exception):
    """"""
    pass

########################################################################
class Node(object):
    """"""
    
    #----------------------------------------------------------------------
    def __init__(self, col=-1, value=None, result=None, tb=None, fb=None,
                 subset=None):
        """Constructor"""
        self.col = col        # 特征值的序号
        self.value = value    # 判定值,该节点为叶节点时,value为None
        self.result = result  # 节点的值
        self.tb = tb          # TRUE分支
        self.fb = fb          # FALSE分支
        
        self.subset = subset  # 当前节点的训练数据集
        self.leaves = 0       # 叶节点数
        

#----------------------------------------------------------------------
def calcClass(dataset):
    """
    多数表决法决定类
    Parameter:
    dataset:训练数据集
    Return:
    maxcls:数据集的类别
    """
    maxcnt = 0
    maxcls = dataset[0, -1]
    clsdict = {}
    for i in dataset[:, -1]:
        clscnt = clsdict.get(i, 0)
        clscnt += 1
        clsdict[i] = clscnt
        if clscnt > maxcnt:
            maxcnt = clscnt
            maxcls = i
    return maxcls
    
    
#----------------------------------------------------------------------
def calcGini(dataset):
    """
    计算训练数据集的基尼指数(以训练数据集为单结点树的预测误差)
    Parameter:
    dataset:训练数据集
    Return:
    gini:基尼指数
    """
    counts = {}
    nums = len(dataset)
    gini = 1
    for i in dataset:
        ck = counts.get(i[-1], 0)
        ck += 1
        counts[i[-1]] = ck
    return 1 - reduce(operator.add, 
                      map(lambda i:i**2, counts.values()))/(float(nums**2))
    
#----------------------------------------------------------------------
def calcCondGini(dataset, col, a):
    """
    计算指定特征下的基尼指数
    该函数用于CART树生成
    Parameter:
    dataset:训练数据集
    col:特征系数
    a:特征值
    Return:
    gini:基尼指数
    """
    d1 = dataset[dataset[:, col]==a]
    d2 = dataset[dataset[:, col]!=a]
    length = float(len(dataset))
    return len(d1)/length * calcGini(d1) + len(d2)/length * calcGini(d2)

#----------------------------------------------------------------------
def calcNodeGini(tree):
    """
    计算结点基尼指数
    Parameter:
    tree:CART树
    Return:
    gini:基尼指数
    """
    #若为单结点树/叶结点树
    if tree.value is None:
        return calcGini(tree.subset)
    #若为内部结点树/根结点树
    else:
        length = len(tree.subset)
        return float(len(tree.tb.subset))/length*calcNodeGini(tree.tb) +\
               float(len(tree.fb.subset))/length*calcNodeGini(tree.fb)
    
#----------------------------------------------------------------------
def calcGt(tree):
    """
    计算内部结点的g(t)
    Parameter:
    tree:CART决策树
    Return:
    g(t)
    """
    #若为单结点树/叶结点树
    if tree.value is None:
        raise NodeError('THE NODE is leaf node!')
    else:
        return (calcGini(tree.subset) - calcNodeGini(tree)) /\
               (getLeaves(tree) - 1)
    
#----------------------------------------------------------------------
def getMinestGtNode(tree):
    """
    遍历CART树，计算最小g(t)和其对应的内部结点
    Parameter:
    tree:CART决策树
    Return:
    gt:最小g(t)
    internode:最小g(t)对应的内部结点
    """
    if tree.value is None:
        raise NodeError('THE NODE is leaf node!')
    else:
        gt = calcGt(tree)
        internode = tree
        if tree.tb is not None and tree.tb.value is not None:
            tbgt, tbnode = getMinestGtNode(tree.tb)
            if tbgt < gt:
                gt = tbgt
                internode = tbnode
        if tree.fb is not None and tree.fb.value is not None:
            fbgt, fbnode = getMinestGtNode(tree.fb)
            if fbgt < gt:
                gt = fbgt
                internode = fbnode
        return gt, internode
        
#----------------------------------------------------------------------
def prune(tree):
    """
    CART剪枝算法
    Parameter:
    tree:CART决策树
    Return:
    tree:剪枝后的决策树
    """
    #计算决策树最小g(t)及其对应的内部结点
    gt, internalnode = getMinestGtNode(tree)
    #将获取的内部结点单结点化
    internalnode.value = None
    internalnode.tb = None
    internalnode.fb = None
    internalnode.col = -1
    return tree

#----------------------------------------------------------------------
def CartPrune(tree):
    """
    CART剪枝算法
    Parameter:
    tree:CART决策树
    Return:
    optimaltree:最优决策树
    """
    #子树序列
    trees = []
    #当前需要剪枝的决策树
    prunetree = tree
    trees.append(copy.deepcopy(prunetree))
    #生成子树序列
    while prunetree.value is not None:
        prunetree = prune(prunetree)
        trees.append(copy.deepcopy(prunetree))
    #交叉验证最优子树
    optimalGini = inf
    optimalTree = trees[0]
    for i, val in enumerate(trees):
        print '\n'
        printtree(val)
        currGini = calcNodeGini(val)
        if currGini < optimalGini:
            optimalGini = currGini
            optimalTree = val
    return optimalTree
        

#----------------------------------------------------------------------
def calcBestSplit(dataset, algo=calcCondGini):
    """
    计算最优特征和最优切分点
    Parameter:
    dataset:训练数据集
    algo:基尼指数/熵
    Return:
    bestCol:最优特征系数
    bestA:最优特征值
    """
    bestCol = -1
    bestA = None
    bestGini = inf
    for col in range(len(dataset[0])-1):
        for a in set(dataset[:, col]):
            condGini = algo(dataset, col, a)
            if condGini < bestGini:
                bestGini = condGini
                bestA = a
                bestCol = col
    return bestCol, bestA

#----------------------------------------------------------------------
def splitDataSet(dataset, col, a):
    """
    划分训练数据集
    Parameter:
    dataset:训练数据集
    col:特征系数
    a:特征值
    Return:
    eqSet:指定特征值等于a时的子数据集
    neSet:指定特征值不为a时的子数据集
    """
    eqSet = dataset[dataset[:, col]==a]
    neSet = dataset[dataset[:, col]!=a]
    length = len(dataset[0])
    eqSet = eqSet[:, [i for i in range(length) if i != col]]
    neSet = neSet[:, [i for i in range(length) if i != col]]
    return eqSet, neSet
    

#----------------------------------------------------------------------
def cartGenAlgo(dataset, delta=0.0001):
    """
    CART生成算法
    Parameter:
    dataset:训练数据集
    delta:停止计算条件
    Return:
    cartTree:CART决策树
    """
    #训练数据集的基尼指数小于阈值时，返回节点，多数表决法决定类别
    if calcGini(dataset) < delta:
        return Node(result=calcClass(dataset), subset=dataset)
    #训练数据集的类别完全相同时，返回节点
    if len(set(dataset[:, -1])) == 1:
        return Node(result=dataset[0, -1], subset=dataset)
    #用完所有特征时，返回节点
    if len(dataset[0]) == 1:
        return Node(result=calcClass(dataset), subset=dataset)
    #计算最优特征和最优特征值
    bestCol, bestA = calcBestSplit(dataset)
    #划分训练数据集
    eqSet, neSet = splitDataSet(dataset, bestCol, bestA)
    #计算类别
    cls = calcClass(dataset)
    #递归计算TRUE分支
    tb = cartGenAlgo(eqSet)
    #递归计算FALSE分支
    fb = cartGenAlgo(neSet)
    #返回节点
    return Node(col=bestCol, value=bestA, result=cls, tb=tb, fb=fb, 
                subset=dataset)


#----------------------------------------------------------------------
def getLeaves(tree):
    """
    计算节点的叶节点数目
    Parameter:
    tree:CART决策树
    Return:
    leaves:决策树的叶节点数
    """
    #当前节点为叶节点，返回1
    if tree.value is None:
        return 1
    else:
        tree.leaves = getLeaves(tree.tb) + getLeaves(tree.fb)
        return tree.leaves

#----------------------------------------------------------------------
def splitInstance(instance, col):
    """
    删除第col个特征
    Parameter:
    instance:实例
    col:需要删除的特征系数
    Return:
    删除第col个特征后的实例
    """
    if type(instance).__name__ == 'list':
        return instance[:col] + instance[col+1:]
    elif type(instance).__name__ == 'ndarray':
        return instance[[i for i in range(len(instance)) if i != col]]
    else:
        raise TypeError('The instance type is not right!')
    

#----------------------------------------------------------------------
def treeclassify(tree, instance):
    """
    CART决策树对实例进行分类
    Parameter:
    tree:CART决策树
    instance:实例
    Return:
    cls:实例的预测类别
    """
    #当前节点为叶结点
    if tree.value is None:
        return tree.result
    else:
        col = tree.col
        if instance[col] == tree.value:
            return treeclassify(tree.tb, splitInstance(instance, col))
        else:
            return treeclassify(tree.fb, splitInstance(instance, col))

    
#----------------------------------------------------------------------
def classify(dataset, instance):
    """"""
    carttree = cartGenAlgo(dataset)
    optimalCartTree = CartPrune(carttree)
    return treeclassify(optimalCartTree, instance)

    
#----------------------------------------------------------------------
def printtree(tree, indent=0):
    """
    打印树
    """
    #这是一个节点吗？
    if tree.value is None:
        print str(tree.result) + ':leafGini:' + str(calcNodeGini(tree))
    else:
        #打印判断条件
        print str(tree.col) + ':' + str(tree.value) + '?' + 'interGini:' + str(calcGt(tree))
        #打印分支
        print ' ' * indent+'T->',
        printtree(tree.tb, indent+2)
        print ' ' * indent+'F->',
        printtree(tree.fb, indent+2)
    


########################################################################
class FunctionTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def setUp(self):
        dataset = [
            ("青年", "否", "否", "一般", "否"),
            ("青年", "否", "否", "好", "否"),
            ("青年", "是", "否", "好", "是"),
            ("青年", "是", "是", "一般", "是"),
            ("青年", "否", "否", "一般", "否"),
            ("中年", "否", "否", "一般", "否"),
            ("中年", "否", "否", "好", "否"),
            ("中年", "是", "是", "好", "是"),
            ("中年", "否", "是", "非常好", "是"),
            ("中年", "否", "是", "非常好", "是"),
            ("老年", "否", "是", "非常好", "是"),
            ("老年", "否", "是", "好", "是"),
            ("老年", "是", "否", "好", "是"),
            ("老年", "是", "否", "非常好", "是"),
            ("老年", "否", "否", "一般", "否")]
        dataset = array(dataset)
        dataset[dataset[:, :]=='青年'] = 1
        dataset[dataset[:, :]=='中年'] = 2
        dataset[dataset[:, :]=='老年'] = 3
        dataset[dataset[:, :]=='是'] = 1
        dataset[dataset[:, :]=='否'] = 2
        dataset[dataset[:, :]=='好'] = 2
        dataset[dataset[:, :]=='一般'] = 3
        dataset[dataset[:, :]=='非常好'] = 1
        self.dataset = dataset
        
    #----------------------------------------------------------------------
    def test_0(self):
        """"""
        self.assertAlmostEqual(calcCondGini(self.dataset, col=0, a='1'), 0.44, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=0, a='2'), 0.48, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=0, a='3'), 0.44, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=1, a='1'), 0.32, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=2, a='1'), 0.27, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=3, a='1'), 0.36, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=3, a='2'), 0.47, delta=0.01)
        self.assertAlmostEqual(calcCondGini(self.dataset, col=3, a='3'), 0.32, delta=0.01)
        
    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        self.assertTupleEqual(calcBestSplit(self.dataset), (2, '1'))
        
    #----------------------------------------------------------------------
    def test_2(self):
        """"""
        self.assertEqual(calcClass(self.dataset), '1')
        self.assertEqual(calcClass(self.dataset[self.dataset[:, 0]=='1']), '2')
        self.assertEqual(calcClass(self.dataset[self.dataset[:, 0]=='2']), '1')
        self.assertEqual(calcClass(self.dataset[self.dataset[:, 0]=='3']), '1')
        
    #----------------------------------------------------------------------
    def test_3(self):
        """"""
        dataset = self.dataset
        eqSet = dataset[dataset[:, 0]=='1']
        neSet = dataset[dataset[:, 0]!='1']
        eqSet = eqSet[:, 1:len(dataset[0])]
        neSet = neSet[:, 1:len(dataset[0])]
        splitEqSet, splitNeSet = splitDataSet(dataset, 0, '1')
        self.assertListEqual(eqSet.tolist(), splitEqSet.tolist())
        self.assertListEqual(neSet.tolist(), splitNeSet.tolist())
        
    #----------------------------------------------------------------------
    def test_4(self):
        """"""
        dataset = [
            ("Yonth", "NO", "NO", "normal", "NO"),
            ("Yonth", "NO", "NO", "good", "NO"),
            ("Yonth", "YES", "NO", "good", "YES"),
            ("Yonth", "YES", "YES", "normal", "YES"),
            ("Yonth", "NO", "NO", "normal", "NO"),
            ("middle", "NO", "NO", "normal", "NO"),
            ("middle", "NO", "NO", "good", "NO"),
            ("middle", "YES", "YES", "good", "YES"),
            ("middle", "NO", "YES", "excellent", "YES"),
            ("middle", "NO", "YES", "excellent", "YES"),
            ("old", "NO", "YES", "excellent", "YES"),
            ("old", "NO", "YES", "good", "YES"),
            ("old", "YES", "NO", "good", "YES"),
            ("old", "YES", "NO", "excellent", "YES"),
            ("old", "NO", "NO", "normal", "NO")]
        dataset = array(dataset)
        cartTree = cartGenAlgo(dataset)
        print '\n'
        printtree(cartTree)
        
        #copycart0 = copy.deepcopy(cartTree)
        #copycart1 = copy.copy(copycart0)
        #copycart0.fb.tb = Node(result='tb0')
        #copycart0.fb.fb = Node(result='fb0')
        #print '\n-------cartTree--------'
        #printtree(cartTree)
        #print '\n-------copycart0-------'
        #printtree(copycart0)
        #copycart1.fb.tb = Node(result='tb1')
        #copycart1.fb.fb = Node(result='fb1')
        #print '\n-------cartTree--------'
        #printtree(copycart0)
        #print '\n-------copycart1-------'
        #printtree(copycart1)
        
    #----------------------------------------------------------------------
    def test_5(self):
        """"""
        cartTree = cartGenAlgo(self.dataset)
        self.assertEqual(getLeaves(cartTree), 3)
        
    #----------------------------------------------------------------------
    def test_6(self):
        """"""
        dataset = [
            ("Yonth", "NO", "NO", "normal", "NO"),
            ("Yonth", "NO", "NO", "good", "NO")]
        dataset = array(dataset)
        cartTree = cartGenAlgo(dataset)
        with self.assertRaises(NodeError) as e:
            calcGt(cartTree)
    
    #----------------------------------------------------------------------
    def test_7(self):
        """"""
        cartTree = cartGenAlgo(self.dataset)
        optimalTree = CartPrune(cartTree)
        
    #----------------------------------------------------------------------
    def test_8(self):
        """"""
        a = [1, 2, 3, 4, 5, 6, 7]
        col = 3
        self.assertListEqual(splitInstance(a, col), [1, 2, 3, 5, 6, 7])
        a = array([1, 2, 3, 4, 5, 6, 7])
        col = 3
        self.assertListEqual(splitInstance(a, col).tolist(), [1, 2, 3, 5, 6, 7])

#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(FunctionTest('test_0'))
    suite.addTest(FunctionTest('test_1'))
    suite.addTest(FunctionTest('test_2'))
    suite.addTest(FunctionTest('test_3'))
    suite.addTest(FunctionTest('test_4'))
    suite.addTest(FunctionTest('test_5'))
    suite.addTest(FunctionTest('test_6'))
    suite.addTest(FunctionTest('test_7'))
    suite.addTest(FunctionTest('test_8'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')




    