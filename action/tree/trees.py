#!/usr/bin/env python3
# coding=utf-8

"""
决策树
优点: 计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据
缺点: 可能产生过度匹配问题
适用数据类型: 数值型和标称型
"""

from math import log
import operator
import pickle

def calcShannonEnt(dataSet):
    # 求熵
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
        
    return shannonEnt

    
    
def createDataSet():
    dataSet = [ [1, 1, 'yes'], 
                [1, 1, 'yes'], 
                [1, 0, 'no'], 
                [0, 1, 'no'], 
                [0, 1, 'no'] ]
                
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
    
    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
            # 提取出第axis个特征为value的数据, 并去掉第axis个特征项
            """
            a = [1,2,3], b = [4,5,6]
            a.append(b) -> [1,2,3, [4,5,6] ]
            a.extend(b) -> [1,2,3,4,5,6]
            """
            
    return retDataSet


    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1    # 只看特征列,不算标签列
    baseEntropy = calcShannonEnt(dataSet)
    baseInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 列出所有数据的第i个特征属性值
        
        uniqueVals = set(featList)
        # 剔除重复的, 只留下特征种类
        
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            """
            A划分成A1和A2
            新的熵为 p(A1)*H(A1)+p(A2)*H(A2)
            """
            
        infoGain = baseEntropy - newEntropy
        # 信息增益(新熵越小越好, 越小表示越整齐, 混乱程度越低)
        if infoGain > baseInfoGain:
            baseInfoGain = infoGain
            bestFeature = i
            # 标记最优划分特征
            
    return bestFeature
        
    
def majorityCnt(classList):
    # 当划分到最后, 消耗了所有的特征, 但子集中还存在不同的类别时, 用此方法选出多数作为该类
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
        
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]
    
    
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # count用于计算list中指定元素数量
        # 当其中全部元素都相同时, 计数等于长度, 此时表面该子集中同属一类
        # 此对应递归终止的第二个条件
        return classList[0]
        
    if len(dataSet[0]) == 1:
        # 当数据集长度为1, 也就是特征都被消耗完了, 只剩一列标签列
        # 此对应递归终止的第一个条件
        return majorityCnt(classList)
        
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #print(labels, bestFeat)
    bestFeatLabel = labels[bestFeat]
    
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])   # 消耗了一个特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        
    return myTree
    
    
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 查找对应特征所在位置, 在哪列
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
                
    return classLabel

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
        
def grabTree(filename):
    with open(filename, 'rb') as fr:
        return pickle.load(fr)
        
    
def file2matrix(filename):
    
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        returnMat = []
        for line in arrayOLines:
            line = line.strip() # 删除字符串开头结尾处的空白符, 包括 \n(换行), \t(制表符), \r(回车)
            # \n, (newline)下一行, 不移动光标位置
            # \r, (return)移动到当前行的最左边
            # windows系统中 \r\n表示回车加换行
            # Linux系统中 \n表示回车加换行
            listFromLine = line.split('\t') # 以'\t'分割字符串, 返回list (默认以空格分)
            returnMat.append(listFromLine)
        return returnMat

        
def createLenseTree():
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses = file2matrix('lenses.txt')
    lensesTree = createTree(lenses, labels)
    return lensesTree
    
    
    