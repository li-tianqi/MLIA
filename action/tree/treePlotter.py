#!/usr/bin/env python3
# coding=utf-8

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict (boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
    
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
            
        if thisDepth > maxDepth:
            maxDepth = thisDepth
            
    return maxDepth

    
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0:'no', 1:'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]
    

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
    
""" 
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("a decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("a leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
"""
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
    
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # 利用叶子节点的位置确定决策节点的位置, 决策节点始终位于子树叶子节点的中间
    """
    假如树长这样: (D:决策节点, L:叶子节点)
     ######################## 
    #                        #        
    #         D              #  -- 0   
    #      /     \           #   |   -> 1.0/D (一份)
    #   L           D        #  --  
    #             /    \     #   |            
    #            D        L  #  --                    
    #          /   \         #   |           
    #         L     L        #  -- 1
    #                        #
     ########################
     
    |--.--|--.--|--.--|--.--|
    0                       1
    |<>| -> 1.0/W/2 (一份的一半)
    |<--->| -> 1.0/W (一份)
       |<-|------->|  -> 这里是同层中决策点和叶子点的差距
         |       \       
        1.0/W/2     w'*(1.0/W/2)  (w'为子树的叶子数, 有几个叶子, 中点之前就有几个一份的一半)   
       L           D    <- 对应第三层
       ^           ^           
       |           |     
   (这里有        (这里的决策     
    个叶子         节点的位置                    
    节点,           xOff+1.0/W/2+w'*(1.0/W/2)            
    xOff=           )
    1.0/W/2)
    
    """
    # 以上解释了横向两种节点的位置计算方法
    
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 首次画第一个决策点时, 目标位置(cntrPt)和注释位置(parentPt)相同, 所以不画箭头, 只显示框
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    # 因为已经画了一个节点, 在纵向上坐标要减去纵向将[0,1]分成D份后每份的值
    for key in secondDict.keys():
        # 继续绘制, 如果是dict, 说明是决策节点, 对应子树, 用递归
        # 如果不是dict, 说明是叶子节点, 此时就要更新叶子节点的位置xOff
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
            
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    # 对与纵向位置, 画完一个决策节点后(也就是调用一次plotTree), 说明要画该决策节点的子节点了, 此时要更新yOff, 做减法到下一层; 
    # 当一个plotTree运行完, 当退出时, 表明遇到了全是叶子的子树, 此时要回到上一层, 接着画上一次的其他分支(比如上面第三层的D子树完成, 要画L), 所以要更新yOff到上一层, 做加法回到上一层
    
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])    # 坐标轴空的
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    # 画树的范围在横竖都是[0,1]的范围
    # 根节点永远画在树的中间位置, 包括子树
    # 一共有W个叶子节点, 则横向就将[0,1]分为W份 
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    
    
    
    
    
    
    
    
    
    