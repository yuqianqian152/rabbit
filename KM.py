import numpy
from math import log
from math import exp
import operator
import random
def textParse(bigstring):
    res=[]
    temp=""
    for x in bigstring:
        if ord(x)==32:
            if len(temp)!=0:
                res.append(temp)
            temp=""
        if 65<=ord(x)<=90:
            m=chr(ord(x)+32)
            temp+=m
        if 97<=ord(x)<=122:
            temp+=x
    if len(temp)!=0:
        res.append(temp)
    return res
def createVocabList(dataSet):
    #生成一个包含所有数据条的单词（且每个单词数量为1）的一个单词字典
    vocabSet = set([])
    for document in dataSet:    #取传入矩阵数据集的每行元素
        vocabSet = vocabSet | set(document) #做并集运算
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    #把文本数据（包含元素为单词的数据postingList）转化为在上个函数的字典中inputSet这个向量的每个元素出现的次数
    returnVec = [0]*len(vocabList)  #初始化返回矩阵
    for word in inputSet:   #遍历inputSet中的单词
        if word in vocabList:   #出现在字典中就让返回数组在字典中对应的这个单词出现的位置处赋值1
            returnVec[vocabList.index(word)]+=1
    return returnVec
def loadSet(count):
    trainMatrix=[]
    for i in range(1,count+1):
        temp=textParse(open('sample/'+str(i)+'.txt',encoding='UTF-8').read())
        trainMatrix.append(temp)
    vocabSet=createVocabList(trainMatrix)
    trainVec=[]
    for i in range(0,count):
        Vec=setOfWords2Vec(vocabSet,trainMatrix[i])
        trainVec.append(Vec)
    return trainVec
def loadDataSet(filename):  #导入文件
    dataMat = []
    fr = open(filename)     
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))#把坐标转化为浮点数
        dataMat.append(fltLine)
    return dataMat
def distEclud(vecA,vecB):   #计算两个向量之间的距离
    return numpy.sqrt(numpy.sum(numpy.power(vecA-vecB,2)))
def randCent(dataMat,k):#随机生成K个质心，虽然随机，但是每个随机的质心的每个维度都在数据集该维度的最大值与最小值之间
    dataSet=numpy.mat(dataMat)  #把输入的数据集转化为矩阵的形式
    n = numpy.shape(dataSet)[1] #把数据集的维度存入n
    centroids = numpy.mat(numpy.zeros((k,n)))   #初始化k个质心
    for j in range(0,n):    #遍历数据集的每一列，求出数据集这一列的取值范围
        minJ = min(dataSet[:,j])    #求出最小值
        rangeJ = float(max(dataSet[:,j])-minJ)  #把最大值减去最小值，求出这列（这个特征）的范围
        centroids[:,j] = minJ +rangeJ*numpy.random.rand(k,1)    #numpy.random.rand(k,1)生成一个k行1列的矩阵，其中的每个值都位于0，1之间；其与minJ(该列最小值)与rangeJ(该列的浮动范围)配合，就可以产生一列随机的，但是值介于该列最大最小值之间的质心们在这列的值
    return centroids
def train(centroids,trainVec,time=200):
    i=0
    centroid=[]
    for j in range(0,4):
        temp=[]
        for k in range(0,len(trainVec[0])):
            temp.append(centroids[j,k])
        centroid.append(temp)
    centroid=[[1.0,1.0],[10.0,4.0],[-10.0,4.0],[-10.0,-10.0]]
    while i!=time:
        res=dict()
        for h in range(0,len(trainVec)):
            min_=distEclud(numpy.mat(trainVec[h]),numpy.mat(centroid[0]))
            num=1
            for g in range(1,4):
                if min_>distEclud(numpy.mat(trainVec[h]),numpy.mat(centroid[g])):
                    min_=distEclud(numpy.mat(trainVec[h]),numpy.mat(centroid[g]))
                    num=g+1
            if num not in res.keys():
                res[num]=[]
            res[num].append(h+1)
        centroid=[]
        for s in range(0,4):
            if s+1 in res.keys():
                temp=numpy.mat(trainVec[res[s+1][0]-1])
                for q in range(1,len(res[s+1])):
                    temp=temp+numpy.mat(trainVec[res[s+1][q]-1])
                temp=temp/len(trainVec)
                cent=[]
                for y in range(0,len(trainVec[0])):
                    cent.append(temp[0,y])
                centroid.append(cent)
            else:
                centroid.append(trainVec[random.randint(0,len(trainVec)-1)])
        i+=1
        #return res,len(centroid),len(centroid[0])
    return res
    
