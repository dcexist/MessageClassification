
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import jieba
import jieba.analyse
from langconv import *
from zh_wiki import *
from alphachange import *


# In[2]:


# 读取停用词
stop_words=[line.strip().decode('utf-8')  for line in open('../input/stop_words.txt').readlines()]


# In[3]:


# 将类别保存为txt文件
def writeStr(str,filename):
    fout = open('../input/'+filename, 'a+') 
    fout.write(str+'\n')
    fout.close()


# In[4]:


def preProcess(uStr):
    ustring = uStr.replace(' ','')
#     去除非法字符，并以非法字符分割字符串,返回list
    ret=string2List(ustring.decode('utf-8'))
    msg = ''
    for key in ret:
#         繁体转简体
        key = Converter('zh-hans').convert(key)
        msg += key
    ustring =   msg.encode('utf-8')
    ustring = ustring.replace('x元','价钱')
    ustring = ustring.replace('x日','日期')
    ustring = ustring.replace('x折','打折')
    ustring = ustring.replace('www','网站')
    return ustring


# In[5]:


def cutWords(msg,stopWords):
    seg_list = jieba.cut(msg,cut_all=False)
    seg_list=list(seg_list)
    #key_list = jieba.analyse.extract_tags(msg,20) #get keywords 
    leftWords = [] 
    for i in seg_list:
        if (i not in stopWords):
            leftWords.append(i)        
    return leftWords


# In[6]:


def writeListWords(seg_list,filename):
    fout = open('../input/'+filename, 'a+') 
    wordList = list(seg_list)
    outStr = ' '
    for word in wordList:
        outStr += word
        outStr += ' '
    fout.write(outStr.encode('utf-8')+'\n')
    fout.close()    


# In[7]:


fr = open("../input/80w.txt",'r')
print type(fr)
arrayOfLines = fr.readlines()
print type(arrayOfLines)
print len(arrayOfLines)
i=0
for line in arrayOfLines:
    if (i==50000):
        break
    i=i+1
    if (i%2000==0):
        print i
    line = line.strip()
    line = line.split('\t')
#     print(line[0])
    writeStr(line[1],'classLabel.txt') 

#     去除非法字符、将繁体转化为简体以及其他字符串处理,返回字符串
    ustring = preProcess(line[2])
#     分词并去除停用词，返回数组
    leftWords = cutWords(ustring, stop_words)
#     保存处理后的字符串,以空格分隔
    writeListWords(leftWords,'trainLeft.txt')


# In[8]:


fr = open("../input/20w.txt")
arrayOfLines = fr.readlines()
len1=len(arrayOfLines)
i=0
for line in arrayOfLines:
    if (i==10000):
        break
    i=i+1
    if (i%1000==0):
        print i
    line = line.strip()
    line = line.split('\t')
    writeStr(line[0],'testMsgNum.txt') 
    if len(line) == 1:
        line.append('空')
    ustring = preProcess(line[1])
    leftWords = cutWords(ustring,stop_words)
    writeListWords(leftWords,'testLeft.txt')

