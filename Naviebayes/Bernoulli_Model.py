import numpy as np
import random
import re
import os

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):                                   
    vocabSet = set([])  					#创建一个空的不重复列表
    for document in dataSet:				
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)






"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)						#创建一个其中所含元素都为0的向量
    for word in inputSet:							#遍历每个词条
        if word in vocabList:							#如果词条存在于词汇表中，则置1，其中vocabList.index(word)返回word在vocabList中的位置
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec								#返回文档向量





"""
函数说明:朴素贝叶斯分类器训练函数------使用多项式模型
Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainClass - 训练类别标签向量
        ClassNum - 训练类别数
Returns:
	pVect - 各个种类文件的条件概率数组
	pClass - 文档属于各个类的概率数组
"""
def trainNB0(trainMatrix, trainClass, ClassNum):                                                 
    numTrainDocs = len(trainMatrix)						#计算训练的文档数目
    numWords = len(trainMatrix[0])                                              #统计训练数据所出现的所有单词种类数
    ClassDocNum = np.zeros((ClassNum,1))                                            #定义ClassDocNum用来存储各个种类的文档数量
    WordDocNum = np.ones((ClassNum,numWords))                                   #定义WordDocNum用来存储各个类别文档中包含某个单词的文档数目例[0,0]表示第0类别的文档中包含单词表中第0个单词的文档的数目

    for i in range(ClassNum):                                                          
        for j in range(numTrainDocs):
            if trainClass[j] == i:
                ClassDocNum[i] += 1                                       
                for l in range(numWords):
                    if trainMatrix[j,l] == 1:
                       WordDocNum[i,l] += 1


    
    pClass = ClassDocNum/numTrainDocs							#计算文档属于各个类的概率，计算公式：P(C) = 类C下单词总数/整个训练样本的单词总数                                                            
        	                                       
    pVect = np.log(WordDocNum/(ClassDocNum + 2))							#取对数，防止下溢出，概率公式：P（word|C）= (类C下单词word在各个文档中出现的次数之和+1)/类C下单词总数+训练样本中出现的单词种类数                    
    return pVect, pClass							#返回属于条件概率数组，文档属于各类的概率









"""

函数说明:朴素贝叶斯分类器分类函数
Parameters:
	vec2Classify - 待分类的词条数组
	pVect -  各类的条件概率数组 	
	pClass - 文档属于各个类别的概率
Returns:
	Class - 测试文件所属类别
"""
def classifyNB(vec2Classify, pVect, pClass, ClassNum):
    p = []
    for i in range(ClassNum):
        p.append(sum(vec2Classify * pVect[i]) + np.log(pClass[i]))		#对数运算化乘法为加法
    Class = p.index(max(p))							#计算测试文本最大可能属于的类别
    return Class








"""
函数说明:接收一个大字符串并将其解析为字符串列表
Parameters:
    无
Returns:
    无
"""
def textParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W+', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写






"""
函数说明:测试朴素贝叶斯分类器
Parameters:
    无
Returns:
    无
"""
def spamTest():
    docList = []
    classList = []
    #fullText = []
    TrainNum = []
    SumNumber = 0
    DIR_Train = 'Train'                                                                                                 #数据集所在文件夹名
    ClassNum = len([name for name in os.listdir(DIR_Train) if os.path.isdir(os.path.join(DIR_Train, name))])		#统计文件夹下文件夹的个数，即统计分类的类数		
    for i in range(ClassNum):
        DIR = DIR_Train + '/%d' %i											#分类的文件夹设置为0,1,2....作为不同类别
        TrainNum.append(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))		#统计各个类别文件中文档的个数
        for j in range(TrainNum[i]):                                                 
            wordList = textParse(open(DIR + '/%d.txt' %j, 'r', encoding="utf8").read())					#读取文档，列表形式     
            docList.append(wordList)											#docList为存储元素为列表的列表
            #fullText.append(wordList)
            classList.append(i)                                                                                         #截取的文档所述类别                                           
    vocabList = createVocabList(docList)										#获取所有出现的单词列表
    for i in range(ClassNum):
        SumNumber += TrainNum[i]											#统计所有数据文档的个数    
    trainingSet = list(range(SumNumber)) 										
    testSet = []                                                  
    for i in range(int(SumNumber/5)):                             							#将数据集分为训练集和测试集                        
        randIndex = int(random.uniform(0, len(trainingSet)))                                                            #将选中的文档归为测试集
        testSet.append(trainingSet[randIndex])                           						#删除在数据集里选中的文档
        del(trainingSet[randIndex])                                         
    trainMat = []
    trainClass = []                                                     
    for docIndex in trainingSet:                                            
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))							#将所有训练集化成一个元素为列表的列表，其中每个列表为所有单词类数维，用0表示词在文档中未出现，用1表示词在文档中出现       
        trainClass.append(classList[docIndex])                           
    pVect, pClass= trainNB0(np.array(trainMat), np.array(trainClass), ClassNum)  
    errorCount = 0                                                          
    for docIndex in testSet:                                                
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])           
        if classifyNB(np.array(wordVector), pVect, pClass, ClassNum) != classList[docIndex]:    
            errorCount += 1                                                 
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))
    
    #对放在Test文件下的测试文件进行分类，给出所属类别
    TestList = []
    TestClass = []
    DIR_Test = 'Test'
    TestNum = (len([name for name in os.listdir(DIR_Test) if os.path.isfile(os.path.join(DIR_Test, name))]))
    print(TestNum)
    if TestNum > 0:
        for i in range(TestNum):
            wordList = textParse(open(DIR_Test + '/%d.txt' %i, 'r', encoding="utf8").read())
            TestList.append(wordList)
        for TestDoc in TestList:
            wordVector = setOfWords2Vec(vocabList, TestDoc)
            TestClass.append(classifyNB(np.array(wordVector), pVect, pClass, ClassNum))
        print(TestClass)

if __name__ == '__main__':
    spamTest()