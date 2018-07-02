
# coding: utf-8

# In[1]:


import time
import sys
import string
import numpy as np
import lda
import gensim

from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import metrics  


# In[2]:


def loadClassData(filename):
    dataList  = []
    for line in open('../input/'+filename,'r').readlines():#读取分类序列
        dataList.append(int(line.strip()))
    return dataList

def loadTrainData(filename):
    dataList  = []
    for line in open('../input/'+filename,'r').readlines():#读取分类序列
        dataList.append(line.strip())
    return dataList


# In[3]:


trainCorpus = []
classLabel = []

classLabel = loadClassData('classLabel.txt')
trainCorpus = loadTrainData('trainLeft.txt')  
trainData, testData, trainLabel, testLabel = train_test_split(trainCorpus, classLabel, test_size = 0.2,random_state=0) 


# In[4]:


#计算F值分数  
def totalScore(pred,y_test):
    A = 0
    C = 0
    B = 0
    D = 0
    for i in range(len(pred)):
        if y_test[i] == 0:
            if pred[i] == 0:
                A += 1
            elif pred[i] == 1:
                B += 1
        elif y_test[i] == 1:
            if pred[i] == 0:
                C += 1
            elif pred[i] == 1:
                D +=1
    rb_pr = 1.0*D/(B+D)
    rb_re = 1.0*D/(C+D)
    rt_pr = 1.0*A/(A+C)
    rt_re = 1.0*A/(A+B)
    
    Frb = 0.65*rb_pr + 0.35*rb_re
    Frt = 0.65*rt_pr + 0.35*rt_re
    Ftotal = 0.7*Frb + 0.3*Frt
    print Ftotal


# In[46]:


def logisticReg(trainData,testData,trainLabel,testLabel):

    vectorizer = CountVectorizer(binary=True)#0.971387536109
#     vectorizer =TfidfVectorizer(binary=True) #0.931660369875
    fea_train = vectorizer.fit_transform(trainData)
    fea_test = vectorizer.transform(testData);  
    clf =  LogisticRegression()
    clf.fit(fea_train,np.array(trainLabel)) 
    pred= clf.predict(fea_test)
    totalScore(pred,testLabel)


# In[56]:


def withoutFeature(trainData,testData,trainLabel,testLabel):
    vectorizer = CountVectorizer(binary=True) #0.974682247615
#     vectorizer =TfidfVectorizer(binary=True) #0.963440017657
    fea_train = vectorizer.fit_transform(trainData)
    fea_test = vectorizer.transform(testData)
    print type(fea_test)
    print 'Size of fea_train:' + repr(fea_train.shape) 
    print 'Size of fea_test:' + repr(fea_test.shape) 
    print fea_train.nnz
    print fea_test.nnz
    clf = LinearSVC( C= 0.8)
    clf.fit(fea_train,np.array(trainLabel))  
    pred = clf.predict(fea_test) 
    totalScore(pred,testLabel)


# In[57]:


#navie bayes classifier
def nbClassifier(trainData,testData,trainLabel,testLabel):
#     vectorizer = CountVectorizer(binary=True) #0.906835307297
    vectorizer =TfidfVectorizer(binary=True) #0.921827974983
    fea_train = vectorizer.fit_transform(trainData)
    fea_test = vectorizer.transform(testData);  
    print 'Size of fea_train:' + repr(fea_train.shape) 
    print 'Size of fea_test:' + repr(fea_test.shape) 
    print fea_train.nnz
    print fea_test.nnz

    clf = MultinomialNB(alpha = 0.01)   
    clf.fit(fea_train,np.array(trainLabel))
    pred = clf.predict(fea_test)
    totalScore(pred,testLabel)


# In[67]:


def linearSVCClassifier(trainData,testData,trainLabel,testLabel):
    hv = HashingVectorizer(n_features =80000)
#     vectorizer = make_pipeline(hv,TfidfTransformer()) #0.958681502859
    vectorizer=hv #0.963931013641
    fea_train = vectorizer.fit_transform(trainData)    #return feature vector 'fea_train' [n_samples,n_features]  
    fea_test = vectorizer.transform(testData);  
    print 'Size of fea_train:' + repr(fea_train.shape) 
    print 'Size of fea_train:' + repr(fea_test.shape) 
    print fea_train.nnz
    print fea_test.nnz
    
    clf = LinearSVC( C= 0.8)
    clf.fit(fea_train,np.array(trainLabel))  
    pred = clf.predict(fea_test);  
    totalScore(pred,testLabel)


# In[73]:


def ldaClassifier(trainData,testData,trainLabel,testLabel):
    
#     文档-词频矩阵
    vectorizer = CountVectorizer(binary=True)
    fea_train = vectorizer.fit_transform(trainData)
    fea_test = vectorizer.transform(testData);  
    
#     model.topic_word_ 输出主题-词语矩阵
#     model.doc_topic_ 输出文档主题矩阵
    model = lda.LDA(n_topics=20,n_iter= 500,random_state=1)
    doc_topic_train=model.fit_transform(fea_train)  
#     doc_topic_train1=model.doc_topic_
    doc_topic_test = model.transform(fea_test)
    
#     print doc_topic_train.shape
#     print doc_topic_train1.shape
#     print doc_topic_test.shape
    clf = LinearSVC( C= 0.8)
    clf.fit(doc_topic_train,np.array(trainLabel)) 
    pred = clf.predict(doc_topic_test);  
    totalScore(pred,testLabel)


# In[75]:


def rfClassifier(trainData,testData,trainLabel,testLabel):
    hv = HashingVectorizer(n_features = 10000,non_negative=True)
    voctorizer = make_pipeline(hv,TfidfTransformer())  
    fea_train = voctorizer.fit_transform(trainData)    #return feature vector 'fea_train' [n_samples,n_features]  
    fea_test = voctorizer.transform(testData);  
    print 'Size of fea_train:' + repr(fea_train.shape) 
    print 'Size of fea_train:' + repr(fea_test.shape) 
    print fea_train.nnz
    print fea_test.nnz
    
    clf = RandomForestClassifier()
    clf.fit(fea_train,np.array(trainLabel))  
    pred = clf.predict(fea_test);  
    totalScore(pred,testLabel)


# In[47]:


# CountVectorizer + LogisticRegression
logisticReg(trainData,testData,trainLabel,testLabel)


# In[49]:


# CountVectorizer + LinearSVC
withoutFeature(trainData,testData,trainLabel,testLabel)


# In[51]:


# TfidfVectorizer + MultinomialNB
nbClassifier(trainData,testData,trainLabel,testLabel)


# In[68]:


# HashingVectorizer + (TfidfTransformer)+ LinearSVC
linearSVCClassifier(trainData,testData,trainLabel,testLabel)


# In[74]:


# LDA + LinearSVC
ldaClassifier(trainData,testData,trainLabel,testLabel)


# In[76]:


# HashingVectorizer + (TfidfTransformer)+ rf
rfClassifier(trainData,testData,trainLabel,testLabel)

