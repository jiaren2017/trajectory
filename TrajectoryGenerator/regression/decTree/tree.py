from math import log
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import operator
import pickle       # 使用pickle.dump存储决策树


# ******************    function    ***********************

def createDataSet():
    dataSet =   [[0, 0, 0, 0, 'no'],         #数据集
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
                
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #分类属性
    return dataSet, labels                #返回数据集和分类属性
    
def createDataSet_easy():
    dataSet =   [[1, 1, 'no'],         #数据集
                [0, 0, 'no'],
                [0, 1,'yes'],
                [1, 0, 'yes']]
                
    labels = ['no surfacing', 'flippers']        #分类属性
    return dataSet, labels                #返回数据集和分类属性
    
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                               #   返回数据集的行数
    labelCounts = {}                                        #   保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                                 #   对每组特征向量进行统计
        currentLabel = featVec[-1]                          #   提取标签(Label)信息
        if currentLabel not in labelCounts.keys():          #   如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                      #   Label计数
#        print('\nlabelCounts:\n', labelCounts)
    shannonEnt = 0.0                                        #   经验熵(香农熵)
#    print('\nlabelCounts:\n', labelCounts) 
    for key in labelCounts:                                 #   计算香农熵
        prob = float(labelCounts[key]) / numEntries         #   选择该标签(Label)的概率
        shannonEnt -= prob * log(prob,2)                    #   利用公式计算
    return shannonEnt                                       #   返回经验熵(香农熵)    
    
    
def splitDataSet(dataSet, axis, value):       
    retDataSet = []                                         #   创建返回的数据集列表
    for featVec in dataSet:                                 #   遍历数据集
        if featVec[axis] == value:
#            print('\nfeatVec:\n', featVec)
            reducedFeatVec = featVec[:axis]                 #   去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])         #   将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet                                       #   返回划分后的数据集
            

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                       #   dataSet[0]: '[0, 0, 0, 0, 'no'] -  1(除去'no') 的特征数量为 4
    baseEntropy = calcShannonEnt(dataSet)                   #   计算数据集的香农熵
    bestInfoGain = 0.0                                      #   信息增益
    bestFeature = -1                                        #   最优特征的索引值
    
    for i in range(numFeatures):                            #   遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]      #   为每行，按照i列的信息建立一个特征list
        print('\nfeatList:\n', featList)    
        uniqueVals = set(featList)                          #   根据特征listim的元素来创建set集合{},元素不可重复
        print('\nuniqueVals:\n', uniqueVals) 
        
        newEntropy = 0.0                                    #   条件经验熵
        for value in uniqueVals:                            #   计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)    #   subDataSet划分后的子集
            print('\nsubDataSet:\n', subDataSet) 
            prob = len(subDataSet) / float(len(dataSet))    #   计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) #   根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                 #   信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))     #   打印每个特征的信息增益
        if (infoGain > bestInfoGain):                       #   计算信息增益
            bestInfoGain = infoGain                         #   更新信息增益，找到最大的信息增益
            bestFeature = i                                 #   记录信息增益最大的特征的索引值
    return bestFeature                                      #   返回信息增益最大的特征的索引值

    
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0   
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)     #   根据字典的值降序排序
    return sortedClassCount[0][0]                                                                   #   返回classList中出现次数最多的元素

    
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]            #   取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):         #   如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                    #   遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #   选择最优特征
    bestFeatLabel = labels[bestFeat]                            #   最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                                 #   根据最优特征的标签生成树
    del(labels[bestFeat])                                       #   删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]     #   得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #   去掉重复的属性值
    for value in uniqueVals:                                    #   遍历特征，创建决策树。                       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


#   获取决策树叶结点的数目
def getNumLeafs(myTree):
    numLeafs = 0                                                #初始化叶子
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不再是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs
    
    
#   获取决策树 的 层数
def getTreeDepth(myTree):
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth            #更新层数
    return maxDepth
    
    
#   绘制带箭头注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):                        
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)
        
#   在父子节点之间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置                   
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    
        
  # 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):                    
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                      #设置结点格式，  定义文本框和箭头模式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


#  绘制图像
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()                                                                                 #显示绘制结果     

    
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))                                                        #获取决策树结点
    secondDict = inputTree[firstStr]                                                        #下一个字典
    featIndex = featLabels.index(firstStr)                                               
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel
    

#   将决策树保存在硬盘上
def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
        
#   将硬盘上的决策树载入  
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr) 
    
    
# ******************   end  function    ***********************          
            
dataSet, labels = createDataSet()       
     
#print('\ndataSet:\n', dataSet)     
      
#print('\ncalcShannonEnt(dataSet):\n', calcShannonEnt(dataSet))

#print('\nsplitDataSet(dataSet,0,1):\n', splitDataSet(dataSet,0,1))    
#print('\nsplitDataSet(dataSet,1,0):\n', splitDataSet(dataSet,1,0))         
            
#print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))    

featLabels = []
myTree = createTree(dataSet, labels, featLabels)
storeTree(myTree, 'classifierStorage.txt')          #   save the decision tree
myTree = grabTree('classifierStorage.txt')          #   grab the decision tree
print(myTree)



testVec = [1,0]     # 测试数据
result = classify(myTree,featLabels,testVec)
if result == 'yes':
    print('\n放贷\n')
else:
    print('\n不放贷\n')
    
createPlot(myTree)

# = [11,22,11,33,33,44]
#a = set(a)
#print('\na:\t', a)        
            
            
            
            
            
            
            
            
            
            
            
            
            
            