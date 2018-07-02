
# coding: utf-8

# In[1]:


import time
import sys
import string
import numpy as np
from scipy import sparse

from sklearn.cross_validation import train_test_split 
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# In[2]:


def loadClassData(filename):
    dataList  = []
    for line in open('../input/'+filename,'r').readlines():#读取分类序列
        dataList.append(int(line.strip()))
    return dataList

def loadTrainData(filename):
    dataList  = []
    for line in open('../input/'+filename,'r').readlines():
        dataList.append(line.strip())
    return dataList


# In[3]:


trainCorpus = []
classLabel = []

classLabel = loadClassData('classLabel.txt')
trainCorpus = loadTrainData('trainLeft.txt') 
trainData, testData, trainLabel, testLabel = train_test_split(trainCorpus, classLabel, test_size = 0.2) 


# In[4]:


method = ['IG']
# 类别字典
class_set = sorted(list(set(trainLabel)))
class_dict=dict(zip(class_set, range(len(class_set))))


# In[5]:


class_dict


# In[6]:


# 词语字典{term:1}
term_set_dict = {}
for doc_terms in trainData:
    for term in doc_terms.split():
        term_set_dict[term] = 1
term_set_list = sorted(term_set_dict.keys())       #term set 排序后，按照索引做出字典
term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
term_dict=term_set_dict


# In[7]:


print len(term_dict)
term_dict


# In[8]:


# 正负样本数
class_df_list = [0] * len(class_dict)
for doc_class in trainLabel:
    class_df_list[class_dict[doc_class]] += 1


# In[9]:


class_df_list


# In[10]:


# 建立每个单词term每个类别的个数，每个单词在矩阵中的顺序根据之前生成的词语字典位置来定，矩阵不含index，但实际上index即代表了不同的单词
term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float64)
for k in range(len(trainLabel)):
    class_index = class_dict[trainLabel[k]]
    doc_terms = trainData[k]
    for term in doc_terms.split():
        term_index = term_dict[term]
        term_class_df_mat[term_index][class_index] +=1


# In[11]:


print term_class_df_mat.shape
term_class_df_mat


# In[12]:


term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x : x[1])]


# In[13]:


print len(term_set)
term_set


# In[15]:


A = term_class_df_mat
print A.shape
A


# In[17]:


B = np.array([(sum(x) - x).tolist() for x in A])
print B.shape
B


# In[18]:


C = np.tile(class_df_list, (A.shape[0], 1)) - A
print C.shape
C


# In[20]:


N = sum(class_df_list)
N


# In[21]:


D = N - A - B - C
D


# In[22]:


term_df_array = np.sum(A, axis = 1)
term_df_array


# In[23]:


class_set_size = len(class_df_list)
class_set_size


# In[24]:


p_t = term_df_array / N
p_t


# In[25]:


p_not_t = 1 - p_t
p_not_t


# In[28]:


p_c_t_mat =  (A + 1) / (A + B + class_set_size)
print p_c_t_mat.shape
p_c_t_mat


# In[30]:


p_c_not_t_mat = (C+1) / (C + D + class_set_size)
print p_c_not_t_mat.shape
p_c_not_t_mat


# In[32]:


p_c_t = np.sum(p_c_t_mat  *  np.log(p_c_t_mat), axis =1)
print len(p_c_t)
p_c_t


# In[33]:


p_c_not_t = np.sum(p_c_not_t_mat *  np.log(p_c_not_t_mat), axis =1)
print len(p_c_not_t)
p_c_not_t


# In[35]:


term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
print term_score_array.shape
term_score_array 


# In[37]:


sorted_term_score_index = term_score_array.argsort()[: : -1]
print sorted_term_score_index .shape
sorted_term_score_index 


# In[39]:


term_set_fs = [term_set[index] for index in sorted_term_score_index]
print len(term_set_fs)
term_set_fs


# In[ ]:


# IG信息增益
A = term_class_df_mat
#将A的两列互换位置
B = np.array([(sum(x) - x).tolist() for x in A])
#将class_df_list重复65000多次
C = np.tile(class_df_list, (A.shape[0], 1)) - A
N = sum(class_df_list)
D = N - A - B - C
term_df_array = np.sum(A, axis = 1)
class_set_size = len(class_df_list)

p_t = term_df_array / N
p_not_t = 1 - p_t
p_c_t_mat =  (A + 1) / (A + B + class_set_size)
p_c_not_t_mat = (C+1) / (C + D + class_set_size)
p_c_t = np.sum(p_c_t_mat  *  np.log(p_c_t_mat), axis =1)
p_c_not_t = np.sum(p_c_not_t_mat *  np.log(p_c_not_t_mat), axis =1) 
term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
sorted_term_score_index = term_score_array.argsort()[: : -1]
term_set_fs = [term_set[index] for index in sorted_term_score_index]  


# In[ ]:


def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return  class_dict


# In[ ]:


def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms.split():
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())       #term set 排序后，按照索引做出字典
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict


# In[ ]:


#正负样本数
def stats_class_df(trainLabel, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


# In[ ]:


def stats_term_class_df(trainData, trainLabel, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float64)
    for k in range(len(trainLabel)):
        class_index = class_dict[trainLabel[k]]
        doc_terms = trainData[k]
        for term in doc_terms.split():
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] +=1
    return  term_class_df_mat


# In[ ]:


def feature_selection(doc_terms_list, doc_class_list,fs_method):
    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_terms_list) # 字典{'dict':序号}
    class_df_list = stats_class_df(doc_class_list, class_dict)#正负样本数
     #字典对应的词在不同类别下的词频(len(term_dict),len(class_dict))
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict)
    term_set = [term[0] for term in sorted(term_dict.items(), key = lambda x : x[1])]
    
    if fs_method == 'MI':
        print 'MI'
        term_set_fs = feature_selection_mi(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        print 'IG'
        term_set_fs = feature_selection_ig(class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'WLLR':
        print "WLLR"
        term_set_fs = feature_selection_wllr(class_df_list, term_set, term_class_df_mat)
   
    return term_set_fs


# In[ ]:


method = ['IG']
termSet = feature_selection(trainData, trainLabel,method)

