#!/usr/bin/env python3
# coding=utf-8

"""
逻辑回归
优点: 计算代价不高, 易于理解和实现
缺点: 容易欠拟合, 分类精度可能不高
适用数据类型: 数值型和标称型
"""

"""
sigmoid函数
sigmoid(z) = 1/(1+exp(-z))
"""

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 这里是把x0置为1.0, 接下来是x1, x2
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat
        
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))
    
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    # mat()将list转换为numpy中的matrix类型
    labelMat = mat(classLabels).transpose()# 转置
    m, n = shape(dataMatrix)    # 返回(行数, 列数)
    alpha = 0.001   # 步长
    maxCycles = 500 # 迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   # m*n * n*1 -> m*1 并对每一项求sigmoid函数
        error = (labelMat - h)  # h为归一化后的假设函数, 当大于0.5, 判为1类, 反之判0类
        # error相当于将每个样本的预测类别与实际类别比较, 
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
    
    
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))   # 都是是array类型
        # h = x0w0+x1w1+x2w2 一个样本的
        """
        a = array([[1,2,3]
                   [4,5,6]
                   [7,8,9]
                   [2,4,6]])
        b = array([1,2,3])
        a * b = array([[1, 4, 9]
                       [4,10,18]
                       [7,16,27]
                       [2, 8,18]])
        sum(a*b) = array([14,38,72])
        每列相加
        """
        error = classLabels[i] - h  # 这个样本的预测与实际差
        weights = weights + alpha * error * dataMatrix[i]   # w'=w+alpha*(y-h)*x
        # 相当于每进来一个样本, 更新一次权重
    return weights
    
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
    
    
def plotBestFit(wei):
    weights = wei.getA()    # wei 是numpy matrix类型, getA()返回与本身相同的array类型
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [] # 1类x1
    ycord1 = [] # 1类x2
    xcord2 = [] # 0类
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2] # w0x0+w1x1+w2x2=0 -> x2=(-w0-w1x1)/w2    (x0=1)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    

    
    
    
    
    
    
    
    
    