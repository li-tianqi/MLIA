#!/usr/bin/env python3
# coding=utf-8

"""
K-近邻
优点: 精度高, 对异常值不敏感, 无数据输入假定
缺点: 计算复杂度高, 空间复杂度高
适用数据类型: 数值型和标称型
"""

from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)   # axis = 0, 每列相加(纵向相加), axis = 1, 每行相加(横向相加)
    distances = sqDistances ** 0.5
    # print("distances=", distances)
    sortedDistIndicies = distances.argsort()    # 排序, 返回索引值, 如对[12, 38, 5, 24]排序, 返回[2, 0, 3, 1] (意思是最小的是第2个, 第二小的是第0个, ...)
    # print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # dict.get('a', def), 取key = 'a' 的value, 如果没有, 返回def
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]
    
    
def file2matrix(filename):
    mapdict = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines,3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip() # 删除字符串开头结尾处的空白符, 包括 \n(换行), \t(制表符), \r(回车)
            # \n, (newline)下一行, 不移动光标位置
            # \r, (return)移动到当前行的最左边
            # windows系统中 \r\n表示回车加换行
            # Linux系统中 \n表示回车加换行
            listFromLine = line.split('\t') # 以'\t'分割字符串, 返回list (默认以空格分)
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(mapdict[listFromLine[-1]])
            index += 1
        return returnMat, classLabelVector
        
        
def autoNorm(dataSet):
    minVals = dataSet.min(0)    # 选取每一列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]    # 行数
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    
    return normDataSet, ranges, minVals
    
    
def datingClassTest():
    hoRatio = 0.08
    k = 4
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k)
        realAnswer = datingLabels[i]
        
        if classifierResult != realAnswer:
            errorCount += 1
            print("classifier result is ", classifierResult, "real answer is ", realAnswer)
            
    errorRate = errorCount/float(numTestVecs)
    print("error rate is ", errorRate)
    print(errorCount, numTestVecs)

        
        
def show():
    dataMat, labels = file2matrix('datingTestSet.txt')
    dataMat, rang, min = autoNorm(dataMat)
    colorlist = ['g', 'b', 'r']
    markerlist = ['x', '^', '*']
    colorValue = []
    markerValue = []
    for i in labels:
        colorValue.append(colorlist[i-1])
        markerValue.append(markerlist[i-1])
        
    fig = plt.figure()
    ax = fig.add_subplot(131)   # 将画布分割为1行1列, 画在第1个位置 (也可用逗号分隔)
    ax.scatter(dataMat[:,0], dataMat[:,1], c = colorValue, marker = '.', s=15 * array(labels))
    ax = fig.add_subplot(132)
    ax.scatter(dataMat[:,0], dataMat[:,2], c = colorValue, marker = '.', s = 15*array(labels))
    ax = fig.add_subplot(133)
    ax.scatter(dataMat[:,1], dataMat[:,2], c = colorValue, marker = '.', s = 15*array(labels))
    plt.show()
    
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per years?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 4)
    print("you will probably like this person: ", resultList[classifierResult - 1])
    
    
    
def img2vector(filename):
    returnVect = zeros((1, 1024))
    with open(filename) as fr:
        
        for i in range(32):
            lineStr = fr.readline() # readline()是读取下一行, 返回str; readlines()是读取所有行, 返回list
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
                
        return returnVect
    
    
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits')   # 用于列出目录中的文件名
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("./digits/trainingDigits/"+fileNameStr)
        
    testFileList = listdir('./digits/testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    k = 3
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("./digits/testDigits/"+fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        if classifierResult != classNumStr:
            errorCount += 1
            print("file name: ", fileStr, "classifier result is: ", classifierResult, "real result is: ", classNumStr)
            
    errorRate = errorCount / float(mTest)
    print("error count is: ", errorCount)
    print("error rate is: ", errorRate)
    
    
def main():
    a, b = createDataSet()
    plt.scatter(a.T[0], a.T[1])
    plt.show()
