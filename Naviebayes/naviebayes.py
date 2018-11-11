from numpy import *
from object_json import *
from copy import *
import pdb

class navieBayes(object):
    def __init__(self,vocabSet = None, classPriorP= None, conditionP = None,\
                 classPriorP_ber= None, conditionP_ber = None,negConditionP_ber = None,lapFactor = 1,\
                 classlabelList = None, **args):
        '''modelType is NB model type,'multinomial' or 'bernoulli', the default type is'multinomial'
           parameters:
           classPriorP is (1,m)log of class prior probability list, m is class number, multinomial
           conditionP is (m,n) log of condition probability list, n is the count of vocabulary,mutinomial
           classPriorP_ber, conditionP_ber, bernoulli
           negConditionP_ber is the negtive condition probability for bernoulli model, actually log(1-condition probability)
           
           vocabSet is the vocabulary set  
           lapFactor is the laplace adjust factor
           classlabelList is the class labels
        '''
        obj_list = inspect.stack()[1][-2]
        self.__name__ = obj_list[0].split('=')[0].strip()

        #self.modelType = modelType
        self.classPriorP = array(classPriorP)
        self.conditionP = array(conditionP)
        self.classPriorP_ber = array(classPriorP_ber)
        self.conditionP_ber = array(conditionP_ber)
        self.negConditionP_ber = array(negConditionP_ber)
        self.vocabSet = vocabSet
        if vocabSet:
            self.vocabsetLen = len(self.vocabSet)
        self.lapFactor = lapFactor
        self.classlabelList = classlabelList        

    def jsonDumpsTransfer(self):
        '''essential transformation to Python basic type in order to
        store as json. dumps as objectname.json if filename missed '''
        #pdb.set_trace()
        self.conditionP = self.conditionP.tolist()
        self.classPriorP = self.classPriorP.tolist()

        self.conditionP_ber = self.conditionP_ber.tolist()
        self.classPriorP_ber = self.classPriorP_ber.tolist()
        self.negConditionP_ber = self.negConditionP_ber.tolist()

    def jsonDumps(self, filename=None):
        '''dumps to json file'''
        self.jsonDumpsTransfer()
        if not filename:
            jsonfile = self.__name__+'.json'
        else: jsonfile = filename
        objectDumps2File(self, jsonfile)
        
    def jsonLoadTransfer(self):      
        '''essential transformation to object required type, such as numpy matrix
        call this function after newobject = objectLoadFromFile(jsonfile)'''
        #pdb.set_trace()
        self.conditionP = array(self.conditionP)
        self.classPriorP = array(self.classPriorP)
        
        self.conditionP_ber = array(self.conditionP_ber)
        self.classPriorP_ber = array(self.classPriorP_ber)
        self.negConditionP_ber = array(self.negConditionP_ber)

    def classlabelTrain(self, classLabel):
        #self.fileCount len(classLabel)
        self.classlabelList = list(set(classLabel))
        #self.classlabelCount = len(self.classlabelList)
        #self.classlabelInteger = range(self.classlabelCount)

    def createVocablist(self, dataSet):
        '''create vocabulary set within the dataSet'''
        vocabList = []
        for doc in dataSet:
            vocabList.extend(doc)
        self.vocabCount = len(vocabList)#total words number of the train data
        self.vocabSet = list(set(vocabList))
        self.vocabsetLen = len(self.vocabSet)
        #return list(set(vocabList))

    def setOfWords2Vec(self, inputWordsSet):
        '''Bernoulli model, return a vector to label the existence of a word in vocabList
        set of word model'''
        returnVec = [0]*self.vocabsetLen
        for word in inputWordsSet:
            if word in self.vocabSet:
                returnVec[self.vocabSet.index(word)] =1
            else: print 'the word %s doesnt exist in my Vocabulary!'% word
        return returnVec

    def bagOfWords2Vec(self, inputWordsSet):
        ''' multinomial model, return a vector to label the number of a word in vocabList
        set of word model'''
        returnVec = [0]*self.vocabsetLen
        for word in inputWordsSet:
            if word in self.vocabSet:
                returnVec[self.vocabSet.index(word)] +=1
            else: print 'the word %s doesnt exist in my Vocabulary!'% word
        return returnVec

    def trainNB_ber(self, dataSet, trainCategpry,lapFactor):
        trainMatrix = [ self.setOfWords2Vec(postDoc)\
                 for postDoc in dataSet]
        numTrainDocs = len(trainMatrix)
        numWords = self.vocabsetLen#length of Words set  
        
        classPriorP = []
        conditionP = []
        negConditionP = []
        #pdb.set_trace()
        for label in self.classlabelList:
            pTrainMatrix = [trainMatrix[i] for i in range(numTrainDocs)\
                     if trainCategpry[i]==label]#get the train matrix of each class
            pDemon = float(len(pTrainMatrix))+ 2.0 #get file count of each class
            pNum = array(pTrainMatrix).sum(axis=0)#sum in columun, counts of file that have a word(each col represent a word) 

            pcond = log(float(len(pTrainMatrix))/numTrainDocs)
            classPriorP.append(pcond) #prior probability
            
            pNum = pNum + (ones(numWords)*lapFactor)#Laplace adjust
            pCon = pNum/pDemon
            pVec = log(pCon)# log of condition probability
            conditionP.append(pVec)
            pVec = log(1-pCon)# log of condition probability
            negConditionP.append(pVec)

        self.classPriorP_ber = array(classPriorP)
        self.conditionP_ber = array(conditionP)
        self.negConditionP_ber = array(negConditionP)

    def trainNB_multi(self, dataSet, trainCategpry,lapFactor):
        trainMatrix = [ self.bagOfWords2Vec(postDoc)\
                 for postDoc in dataSet]
        numTrainDocs = len(trainMatrix)
        numWords = self.vocabsetLen#length of Words set  
        
        classPriorP = []
        conditionP = []
        #pdb.set_trace()
        for label in self.classlabelList:
            
            pTrainMatrix = [trainMatrix[i] for i in range(numTrainDocs)\
                     if trainCategpry[i]==label]#get the train matrix of each class
            pDemon = float (sum([sum(item) for item in pTrainMatrix]))+ self.vocabsetLen #get words count of each class
            pNum = array(pTrainMatrix).sum(axis=0)#sum in columun

            pcond = log(float(pNum.sum())/self.vocabCount)
            classPriorP.append(pcond) #prior probability
            #pdb.set_trace()
            pNum = pNum + (ones(numWords)*lapFactor)#Laplace adjust
            pVec = log(pNum/pDemon)#condition probability, ln(a*b) =ln(a)+ln(b)
            conditionP.append(pVec)

        self.classPriorP = array(classPriorP)
        self.conditionP = array(conditionP)
        
    def trainNB(self, dataSet, trainCategpry,modelType = 'multinomial',lapFactor = 1):
        self.lapFactor = lapFactor#update classify parameters
        
        self.classlabelTrain(trainCategpry)# create classlabel list
        self.createVocablist(dataSet)#create vocabulary set

        if modelType == 'multinomial':
            self.trainNB_multi(dataSet, trainCategpry, lapFactor)
        elif modelType == 'bernoulli':
            self.trainNB_ber(dataSet, trainCategpry, lapFactor)
        else:
            print 'erro model type, support multinomial or bernoulli'

    def classifyNB_multi(self, vec2Classify):
        vec = self.bagOfWords2Vec(vec2Classify)
        #pdb.set_trace()
        postP = [sum(vec*self.conditionP[i])+ self.classPriorP[i]  for i in range(len(self.classlabelList))]
        labelIndex = postP.index(max(postP))
        classLabel = self.classlabelList[labelIndex]
        return classLabel
        
    def classifyNB_ber(self, vec2Classify):
        vec = array(self.setOfWords2Vec(vec2Classify))
        negVec = 1-vec#negative side
        postP = [sum(vec*self.conditionP_ber[i])+sum(negVec*self.negConditionP_ber[i])+ self.classPriorP_ber[i]  for i in range(len(self.classlabelList))]
        labelIndex = postP.index(max(postP))
        classLabel = self.classlabelList[labelIndex]
        return classLabel
        
    def classifyNB(self, vec2Classify, modelType = 'multinomial'):
        if modelType == 'multinomial':
            classLabel = self.classifyNB_multi(vec2Classify)
            return classLabel
        elif modelType == 'bernoulli':
            classLabel = self.classifyNB_ber(vec2Classify)
            return classLabel
        else:
            print 'erro model type, support multinomial or bernoulli'
            #raise()
            return None
        

if __name__ == '__main__':
    
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = ['kindness','abusive','kindness','abusive','kindness','abusive']
    testNB = navieBayes()
    testNB.trainNB(postingList,classVec)
    testNB.trainNB(postingList,classVec, modelType = 'bernoulli')
    '''for (key, value) in testNB.__dict__.items():
        print key, ':\n',value'''
    testNB.jsonDumps()
    testNB.jsonLoadTransfer()

    predictClass_mul = []
    predictClass_ber = []
    for item in postingList:
        predictClass_mul.append( testNB.classifyNB(item, modelType = 'multinomial'))
        predictClass_ber.append(testNB.classifyNB(item, modelType = 'bernoulli')) 
    print 'predictClass_mul ', predictClass_mul
    print 'predictClass_ber ', predictClass_ber

    testEntry = ['love', 'my', 'dalmation']
    print 'the predict class of ',testEntry, ' is ', testNB.classifyNB(testEntry, modelType = 'multinomial'),\
          ',  multinomial model'
    testEntry = ['stupid', 'garbage']
    print 'the predict class of ',testEntry, ' is ', testNB.classifyNB(testEntry, modelType = 'multinomial'),\
          ',  multinomial model'

    testEntry = ['love', 'my', 'dalmation']
    print 'the predict class of ',testEntry, ' is ', testNB.classifyNB(testEntry, modelType = 'bernoulli'),\
          ',  bernoulli model'
    testEntry = ['stupid', 'garbage']
    print 'the predict class of ',testEntry, ' is ', testNB.classifyNB(testEntry, modelType = 'bernoulli'),\
          ',  bernoulli model'
    

    
