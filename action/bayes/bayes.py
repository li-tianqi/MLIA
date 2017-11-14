#!/usr/bin/env python3
# coding=utf-8

"""
朴素贝叶斯
优点: 在数据较少的情况下仍然有效, 可以处理多类别问题
缺点: 对于输入数据的准备方式较为敏感
适用数据类型: 标称型
"""

"""
P(A|B) = P(AB) / P(B)
P(C|X) = P(X|C)P(C) / P(X)
"""

"""
贝叶斯分类准则
if P(C1|x,y) > P(C2|x,y) then (x,y) in C1
if P(C1|x,y) < P(C2|x,y) then (x,y) in C2

P(Ci|x,y) = P(x,y|Ci)P(Ci) / P(x,y)
P(Ci|x,y): 给定点(x,y), 来自Ci的概率
P(x,y|Ci): 已知来自Ci, 是(x,y)的概率
"""

"""
假设特征之间相互独立

词集(set-of-words)模型: 每个单词出现一次
词袋(bag-of-words)模型: 每个单词可出现多次
"""

from numpy import *
import re
import operator
import feedparser

def loadDataSet():
    # 生成实验样本
    postingList=[
                 ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                 ['maybe', 'not', 'take', 'stupid'], 
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], 
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                ]
    classVec = [0, 1, 0, 1, 0, 1]
    # 带stupid的为侮辱性文字, 用 1 表示
    return postingList, classVec
    
def createVocabList(dataSet):
    # 词汇表
    vocabSet = set([])  # 空集(set([1,2,3])={1,2,3})
    # set()函数会将重复元素去掉, 如set([1,2,2,3])={1,2,3}
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 并集(相当于留下不重复的元素)
        
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    # 词集模型
    returnVec = [0] * len(vocabList)    # [a]*3 -> [a,a,a]
    for word in inputSet:
        # 将输入集合转换为词向量, 向量尺寸与词汇表相同, 出现的词在对应词汇表的位置设为 1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
            
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    # 词袋模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
                
    return returnVec
    

def trainNB0(trainMatrix, trainCategory):
    # 输入的训练矩阵由词向量构成
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 标签 1 的概率, 也就是P(c1)
    p0Num = ones(numWords) # 0类别中每个单词在几篇文档中出现过
    p1Num = ones(numWords) # 1类别中每个单词在几篇文档中出现过
    p0Denom = 2.0   # 0类别中每篇文档中出现的单词个数统计
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 两个向量相加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            
    p1Vect = log(p1Num / p1Denom)    # 每个单词出现的次数除以总的单词出现次数, 得到该类别中每个单词出现的概率, 也就是P(wi|ci)
    p0Vect = log(p0Num / p0Denom)
    
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # 这里应该是只计算了p(w|ci)p(ci), 没有除以p(w), 因为不影响大小关系
    # 因为取了对数, 所以用的加法, log(ab) = log(a)+log(b)
    if p1 > p0:
        return 1
    else:
        return 0
        
def textParse(bigString):
    # 分词
    listOfTokens = re.split(r'\W*', bigString)
    # \W 匹配任何非单词字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    # 筛选长度大于2的单词片段, 并转换为小写字母
        
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 读取一个文件, 切分, 留下长度大于2的单词, 形成一个list
        wordList = textParse(open('email/spam/%d.txt' % i, encoding = 'ISO-8859-2').read())
        # docList中一个元素是一个list, 用append
        docList.append(wordList)
        # fullText中一个元素是一个单词, 用extend
        fullText.extend(wordList)
        # spam中是垃圾邮件, 标签是1
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='ISO-8859-2').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        # 随机构建训练集, 一共50封邮件, 随机去掉10个索引
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        # 在剩下的40个文档中构建词向量矩阵和标签向量
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('error: ', docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))
    
    
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token) # 统计每个词出现的次数
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True) # 对次数按从大到小排序, 结果是list, 元素是tuple [(key1,value1),(key2,value2)]
    return sortedFreq[:30]
    
def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V
        
        
def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse = True)
    print("SF**" * 10)
    for item in sortedSF:
        print(item[0])
        
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse = True)
    print("NY**" * 10)
    for item in sortedNY:
        print(item[0])
        
        
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocaList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocaList, postinDoc))
        
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocaList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocaList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))



