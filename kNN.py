#encoding:UTF-8

from numpy import *
import unittest
from treePlotter import *
import matplotlib.pyplot as plt
import bisect
import operator
import os


########################################################################
class Node(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, val):
        """Constructor"""
        self.val = val
        self.left = None
        self.right = None
    

########################################################################
class KdTree(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, val=None):
        """Constructor"""
        if val is None:
            self.node = None
            self.numChild = -1
        else:
            self.node = Node(val)
            self.numChild = 0
        
    #----------------------------------------------------------------------
    @property
    def left(self):
        """"""
        return self.node.left
    
    #----------------------------------------------------------------------
    @left.setter
    def left(self, val):
        """"""
        self.node.left = val
    
    #----------------------------------------------------------------------
    @property
    def right(self):
        """"""
        return self.node.right
    
    #----------------------------------------------------------------------
    @right.setter
    def right(self, val):
        """"""
        self.node.right = val
    
    #----------------------------------------------------------------------
    def _isleaf(self):
        """"""
        return self.numChild == 0
    
    #----------------------------------------------------------------------
    def get_numChild(self):
        """"""
        if self._isNonNode:
            return 0
        else:
            return self.left.get_numchild
    #----------------------------------------------------------------------
    def _isNonNode(self):
        """"""
        return self.node is None
    
    #----------------------------------------------------------------------
    def get_numleafs(self):
        """"""
        if self._isNonNode():
            return 0
        elif self._isleaf():
            return 1
        else:
            return self.left.get_numleafs() + self.right.get_numleafs()
    
    #----------------------------------------------------------------------
    def get_treedepth(self):
        """"""
        if self.node is None:
            return 0
        else:
            return max(1 + self.left.get_treedepth(), 
                       1 + self.right.get_treedepth())
        
    #----------------------------------------------------------------------
    @property
    def height(self):
        """"""
        return self.get_treedepth()
    
    #----------------------------------------------------------------------
    @classmethod
    def genTree(cls, dataSet, r=0):
        """"""
        if not dataSet.any():
            return KdTree()
        else:
            leftSet, midPoint, rightSet = cls.getMid(dataSet, r)
            newTree = KdTree(midPoint)
            r = (r+1) % len(midPoint)
            newTree.left = newTree.genTree(leftSet, r)
            newTree.right = newTree.genTree(rightSet, r)
            if leftSet.any():
                newTree.numChild += 1
            if rightSet.any():
                newTree.numChild += 1
            return newTree
    
    #----------------------------------------------------------------------
    @staticmethod
    def getMid(dataSet, r):
        """
        查找第r个坐标的中间值
        Parameter: dataSet, 数据集
                   r, 第r个坐标
        Return: leftSet, 比第r个坐标中间值小的集合
                midPoint, 中间值
                rightSet, 比第r个坐标中间值大的集合
        """
        if len(dataSet) == 1:
            return array([]), dataSet[0], array([])
        dataSet = sorted(dataSet, key=lambda x: x[r])
        mid = len(dataSet) / 2
        return array(dataSet[:mid]), dataSet[mid], array(dataSet[mid+1:])
    
    #----------------------------------------------------------------------
    def getNearest(self, target, r=0, init=True):
        """
        最近邻搜索
        Parameter: target, 目标点
        Return: ret, 最近邻
                dis, 最近距离
        """
        if self._isleaf():
            return [self.node.val], self.distance(target, self.node.val)
        else:
            if target[r] < self.node.val[r]:
                return self.getBranchDis(self.node.val, 'left', 'right', target, r)
            else:
                return self.getBranchDis(self.node.val, 'right', 'left', target, r)
            
    #----------------------------------------------------------------------
    def getBranchDis(self, mid, b0, b1, target, r):
        """"""
        tree0 = self.__getattribute__(b0)
        tree1 = self.__getattribute__(b1)
        if tree0.node is not None:
            p, dis = tree0.getNearest(target, (r+1)%len(mid))
        else:
            p, dis = None, inf
        boundDis = abs(target[r] - mid[r])
        if dis < boundDis:
            return p, dis
        elif dis == boundDis:
            if dis == self.distance(target, mid):
                p.append(mid)
            return p, dis
        else:
            p1, dis1 = [mid], self.distance(target, mid)
            p, dis = self.compareDists(p, dis, p1, dis1)
            if tree1.node is not None:
                p2, dis2 = tree1.getNearest(target, (r+1)%len(mid))
                p, dis = self.compareDists(p, dis, p2, dis2)
            return p, dis
    
    #----------------------------------------------------------------------
    def getKNN(self, target, k):
        """"""
        knn = []
        r = 0
        return self.getKNearest(target, r, knn, k, init=True)
    
    #----------------------------------------------------------------------
    def getKNearest(self, target, r, knn, k, init=True):
        """"""
        if self._isleaf():
            bisect.insort(knn, (self.distance(target, self.node.val), self.node.val))
            knn = knn[:k]
            return knn
        mid = self.node.val
        if target[r] < self.node.val[r]:
            knn = self.getKNNBranchDis(mid, 'left', 'right', target, r, knn, k)
        else:
            knn = self.getKNNBranchDis(mid, 'right', 'left', target, r, knn, k)
        return knn
            
    #----------------------------------------------------------------------
    def getKNNBranchDis(self, mid, b0, b1, target, r, knn, k):
        """"""
        tree0 = self.__getattribute__(b0)
        tree1 = self.__getattribute__(b1)
        if tree0.node is not None:
            knn = tree0.getKNearest(target, (r+1)%len(mid), knn, k)
        boundDis = abs(target[r] - mid[r])
        if len(knn) < k or knn[-1][0] >= boundDis:
            bisect.insort(knn, (self.distance(target, mid), mid))
            knn = knn[:k]
            if tree1.node is not None:
                knn = tree1.getKNearest(target, (r+1)%len(mid), knn, k)
        return knn
        
    
    #----------------------------------------------------------------------
    @staticmethod
    def compareDists(ap, ad, bp, bd):
        """"""
        if ad == bd:
            ap.extend(bp)
            return ap, ad
        elif ad > bd:
            return bp, bd
        else:
            return ap, ad
        
        
    #----------------------------------------------------------------------
    @staticmethod
    def distance(a, b):
        """"""
        c = array(a) - array(b)
        return sqrt(dot(c.T, c))
    

#----------------------------------------------------------------------
def file2matrix(filename):
    """"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for num, line in enumerate(arrayOLines):
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[num, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat, classLabelVector

#----------------------------------------------------------------------
def autoNorm(dataSet):
    """"""
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals    

#----------------------------------------------------------------------
def classify(inX, dataSet, labels, k):
    """"""
    dataSetSize = dataSet.shape[0]
    kdtree = KdTree()
    kdtree = kdtree.genTree(dataSet)
    knn = kdtree.getKNN(inX, k)
    classCnt = {}
    dataSet = dataSet.tolist()
    for i in knn:    
        index = dataSet.index(i[1].tolist())
        voteLabel = labels[index]
        classCnt[voteLabel] = classCnt.get(voteLabel, 0) + 1
    sortedClassCnt = sorted(classCnt.items(), key=lambda i:i[1])
    return sortedClassCnt[-1][0]
    

#----------------------------------------------------------------------
def img2vector(filename):
    """"""
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect[0]
        

    
########################################################################
class NearestTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_0(self):
        """"""
        lists = []
        disdict = {}
        for _ in range(7):
            lists.append([random.randint(0, 1000), random.randint(0, 1000)])
        target = [random.randint(450, 550) + 0.5, random.randint(450, 550) + 0.5]
        for i in lists:
            distance = KdTree.distance(i, target)
            j = disdict.get(distance, [])
            j.append(i)
            disdict[distance] = j
        mindis = min(disdict.keys())
        minp = disdict[mindis]
        kdTree = KdTree.genTree(lists)
        #createPlot(kdTree)
        p, d = kdTree.getNearest(target)
        self.assertEqual(mindis, d)
        self.assertListEqual(sorted(p), sorted(disdict[d]))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in lists:
            ax.plot(i[0], i[1], marker='o', color='b')
        ax.plot(target[0], target[1], marker='*', color='r')
        for i in p:
            ax.plot((i[0], target[0]), (i[1], target[1]))
        plt.show()

    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        lists = []
        disdict = {}
        for _ in range(52):
            lists.append([random.randint(0, 1000), random.randint(0, 1000)])
        target = [random.randint(450, 550) + 0.5, random.randint(450, 550) + 0.5]
        kdTree = KdTree.genTree(lists)
        knn = kdTree.getKNN(target, k=50)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in lists:
            ax.plot(i[0], i[1], marker='o', color='b')
        ax.plot(target[0], target[1], marker='*', color='r')
        for i in knn:
            ax.plot((i[1][0], target[0]), (i[1][1], target[1]))
        plt.show()
        
        
########################################################################
class File2matrixTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
        normMat, ranges, minVals = autoNorm(datingDataMat)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
                   15.0*array(datingLabels), 15.0*array(datingLabels))
        plt.show()
        
    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
        normMat, ranges, minVals = autoNorm(datingDataMat)
        hoRatio = 0.1
        m = normMat.shape[0]
        numTestVecs = int(m * hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :],\
                                        datingLabels[numTestVecs:m], 3)
            print 'the classifier came back with: %d, the real answer is: %d'\
                  %(classifierResult, datingLabels[i])
            if classifierResult != datingLabels[i]:
                errorCount += 1.0
        print 'the total error rate is: %f' %(errorCount/numTestVecs)
        
    #----------------------------------------------------------------------
    def test_2(self):
        """"""
        hwlabels = []
        trainingFileList = os.listdir('trainingDigits')
        m = len(trainingFileList)
        trainingMat = zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwlabels.append(classNumStr)
            trainingMat[i, :] = list(img2vector('trainingDigits/%s' %fileNameStr))
        testFileList = os.listdir('testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
            classifierResult = classify(vectorUnderTest,
                                        trainingMat, hwlabels, 3)
            print 'the classifier came back with: %d, the real answer is: %d'\
                  % (classifierResult, classNumStr)
            if classifierResult != classNumStr:
                errorCount += 1.0
        print '\nthe total number of errors is: %d' %errorCount
        print '\nthe total error rate is: %f' % (errorCount/float(mTest))
    

#----------------------------------------------------------------------
def suite():
    """构造测试集"""
    suite = unittest.TestSuite()
    suite.addTest(File2matrixTest('test_1'))
    suite.addTest(File2matrixTest('test_2'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
    lists = [(2, 1, 3), (5, 6, 4), (9, 7, 6), (4, 2, 7), (8, 3, 1), (7, 5, 2),
             (9, 3, 4), (1, 2, 3), (4, 5, 6), (2, 3, 4), (6, 5, 4), (3, 3, 3)]
    myTree = KdTree.genTree(lists)
    p, dis = myTree.getNearest((5, 5, 5))
    createPlot(myTree)