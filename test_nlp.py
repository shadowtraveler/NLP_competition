# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:41:19 2020

@author: steve
"""

import pandas as pd
from ckiptagger import  WS, POS#, NER ,construct_dictionary
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
import pickle
import csv
import numpy as np
import datetime


def my_time(endtime,starttime):
    print ("耗時:"+str((datetime.datetime.now() - endtime).seconds)+"s")
    print ("總耗時:"+str((datetime.datetime.now() - starttime).seconds)+"s")
    return datetime.datetime.now()

#做斷詞
"""
ws = WS("./data")
pos = POS("./data")
#ner = NER("./data")
Verb=[['VA'],['VB'],['VC'],['VCL'],['VD'],['VE'],['VF'],['VG'],['VH'],['VHC'],['VI'],['VJ'],['VK'],['VL']]
Noun=[['Na'],['Nc'],['Nd'],['Ne'],['Nf'],['Ng'],['Nh']]
Noun_Nb=[['Nb']]
D_adverd=[['Da'],['Dba'],['Dbb'],['Dbc'],['Dc'],['Dd'],['Df'],['Dg'],['Dh'],['Di'],['Dj'],['Dk']]
print("model loaded.")

df=pd.read_csv('./output_no_name_context_crawl.csv',encoding='utf-8')

print("csv file loaded.")

#word_extract=ws(df['content'])
num=0
error_num=0
nan_num=0
with open('output_no_name_context_crawl_WS.csv', 'w',encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['news_ID', 'content','name'])
    for i in range(len(df['content'])):
        if(df['content'][i]!=''):
            try:
                word_extract=ws(df['content'][i])
                pos_sentence=pos(word_extract)
                num+=1
                tmp=[]
                for j in range(len(word_extract)):
                    
                    #print(pos_sentence[j])
                    if(pos_sentence[j] in Verb):
                        #print(pos_sentence[j])
                        tmp.extend(word_extract[j])
                writer.writerow([df['news_ID'][i],tmp,df['name'][i]])
            except:
                error_num+=1
        else:
            nan_num+=1
    print("total num:"+str(num+error_num))#總共多少個
    print("total line:"+str(num))
    print("error line:"+str(error_num))
    print("nan_num:"+str(nan_num))
"""


#做dictionary
"""
dict_squares = {}
df=pd.read_csv('./output_has_name_context_crawl_WS.csv',encoding='utf-8')
for i in range(len(df['content'])):
    for j in range(len(df['content'][i])):
        if(df['content'][i][j]!=',' and df['content'][i][j]!='\''):
            
            if(dict_squares.get(df['content'][i][j])==None):
                dict_squares[df['content'][i][j]]=1
            else:
                dict_squares[df['content'][i][j]]+=1


# Save
np.save('first_test_dict.npy', dict_squares) 

# Load
read_dictionary = np.load('./first_test_dict.npy',allow_pickle=True).item()
print(len(read_dictionary))
"""

#計算threshold值
"""
dict_point={}
read_dictionary = np.load('./first_test_dict.npy',allow_pickle=True).item()
df=pd.read_csv('./output_has_name_context_crawl_WS.csv',encoding='utf-8')
total=len(df['content'])
for i in range(len(df['content'])):
    tmp=0
    tmp_key=0
    for j in range(len(df['content'][i])):
        if(df['content'][i][j]!=',' and df['content'][i][j]!='\''):
            if(df['content'][i][j]!=' ' and df['content'][i][j]!='[' and df['content'][i][j]!=']'):
                #if(read_dictionary[df['content'][i][j]]>50):
                tmp+=read_dictionary[df['content'][i][j]]
                #if(dict_squares.get(df['content'][i][j])==None):
                #    dict_squares[df['content'][i][j]]=1
                #else:
                #    dict_squares[df['content'][i][j]]+=1
                tmp_key+=1
    dict_point[i]=tmp/tmp_key
average=0
for i in range(total):
    average+=dict_point[i]
print(average/total)

np.save('first_key_threshold.npy', dict_point) 
"""

#測試正確率
"""
read_dictionary = np.load('./first_test_dict.npy',allow_pickle=True).item()
df=pd.read_csv('./output_no_name_context_crawl_WS.csv',encoding='utf-8')
total=len(df['content'])
num=0
error_num=0
for i in range(len(df['content'])):
    tmp=0
    tmp_key=0
    for j in range(len(df['content'][i])):
        if(df['content'][i][j]!=',' and df['content'][i][j]!='\''):
            if(df['content'][i][j]!=' ' and df['content'][i][j]!='[' and df['content'][i][j]!=']'):
                if(read_dictionary.get(df['content'][i][j])!=None):
                    tmp+=read_dictionary[df['content'][i][j]]
                    #if(dict_squares.get(df['content'][i][j])==None):
                    #    dict_squares[df['content'][i][j]]=1
                    #else:
                    #    dict_squares[df['content'][i][j]]+=1
                tmp_key+=1
    if(tmp_key!=0):
        if(tmp/tmp_key>300.0):
            print("error")
            error_num+=1
        else:
            num+=1
    else:
        num+=1
print("total num:"+str(num+error_num))#總共多少個
print("total line:"+str(num))
print("error line:"+str(error_num))
"""

#做KNN
"""
#有名字
tmp_dic={}
read_dictionary = np.load('./first_test_dict.npy',allow_pickle=True).item()
for key in read_dictionary:
    if(key!=' ' and key!='[' and key!=']'):
        tmp_dic[key]=0
df=pd.read_csv('./output_has_name_context_crawl_WS.csv',encoding='utf-8')
test=np.zeros([len(df['content']),len(tmp_dic)])
for i in range(len(df['content'])):
    for j in range(len(df['content'][i])):
        if(df['content'][i][j]!=',' and df['content'][i][j]!='\''):
            if(tmp_dic.get(df['content'][i][j])!=None):
            #    dict_squares[df['content'][i][j]]=1
            #else:
                tmp_dic[df['content'][i][j]]+=1
    j=0
    for key in tmp_dic:
        test[i][j]=tmp_dic[key]
        j+=1
        tmp_dic[key]=0
label=np.ones(len(df['content']))
print("有名字讀取完成")
#沒有名字
df_1=pd.read_csv('./output_no_name_context_crawl_WS.csv',encoding='utf-8')
test_1=np.zeros([len(df_1['content']),len(tmp_dic)])
for i in range(len(df_1['content'])):
    for j in range(len(df_1['content'][i])):
        if(df_1['content'][i][j]!=',' and df_1['content'][i][j]!='\''):
            if(tmp_dic.get(df_1['content'][i][j])!=None):
            #    dict_squares[df['content'][i][j]]=1
            #else:
                tmp_dic[df_1['content'][i][j]]+=1
    j=0
    for key in tmp_dic:
        test_1[i][j]=tmp_dic[key]
        j+=1
        tmp_dic[key]=0
#print(test)
label_1=np.zeros(len(df_1['content']))
print("沒名字讀取完成")
test_total = np.vstack((test,test_1))
label_total= np.hstack((label,label_1))
print("label data合併完成")
knn = KNeighborsClassifier()
train_data , test_data , train_label , test_label = train_test_split(test_total,label_total,test_size=0.1)
knn.fit(train_data,train_label)
print("模型建構完成,預測中")
with open('knn_first.pickle', 'wb') as f:
    pickle.dump(knn, f)
output_knn=knn.predict(test_data)
num=0
error_num=0
for i in range(len(test_label)):
    if(output_knn[i]==test_label[i]):
        num+=1
    else:
        error_num+=1
        print(test_label[i])
print("total num:"+str(num+error_num))#總共多少個
print("total line:"+str(num))
print("error line:"+str(error_num))
"""

#KNN_比例
"""
#有名字
starttime = datetime.datetime.now()
print("開始:目前0秒")
tmp_dic={}
read_dictionary = np.load('./first_test_dict.npy',allow_pickle=True).item()
for key in read_dictionary:
    if(key!=' ' and key!='[' and key!=']'):
        tmp_dic[key]=0
df=pd.read_csv('./output_has_name_context_crawl_WS.csv',encoding='utf-8')
test=np.zeros([len(df['content']),len(tmp_dic)])
for i in range(len(df['content'])):
    total=0
    for j in range(len(df['content'][i])):
        if(df['content'][i][j]!=',' and df['content'][i][j]!='\''):
            if(tmp_dic.get(df['content'][i][j])!=None):
            #    dict_squares[df['content'][i][j]]=1
            #else:
                tmp_dic[df['content'][i][j]]+=1
                total+=1
    j=0
    if(total==0):
        total=1
    for key in tmp_dic:
        test[i][j]=tmp_dic[key]/total
        j+=1
        tmp_dic[key]=0
label=np.ones(len(df['content']))
print("有名字讀取完成")
endtime=my_time(starttime,starttime)
#沒有名字
df_1=pd.read_csv('./output_no_name_context_crawl_WS.csv',encoding='utf-8')
test_1=np.zeros([len(df_1['content']),len(tmp_dic)])
for i in range(len(df_1['content'])):
    total=0
    for j in range(len(df_1['content'][i])):
        if(df_1['content'][i][j]!=',' and df_1['content'][i][j]!='\''):
            if(tmp_dic.get(df_1['content'][i][j])!=None):
            #    dict_squares[df['content'][i][j]]=1
            #else:
                tmp_dic[df_1['content'][i][j]]+=1
                total+=1
    j=0
    if(total==0):
        total=1
    for key in tmp_dic:
        test_1[i][j]=tmp_dic[key]/total
        j+=1
        tmp_dic[key]=0
#print(test)
label_1=np.zeros(len(df_1['content']))
print("沒名字讀取完成")
endtime=my_time(endtime,starttime)
test_total = np.vstack((test,test_1))
label_total= np.hstack((label,label_1))
print("label data合併完成")
endtime=my_time(endtime,starttime)
knn = KNeighborsClassifier()
train_data , test_data , train_label , test_label = train_test_split(test_total,label_total,test_size=0.1)
knn.fit(train_data,train_label)
print("模型建構完成,預測中")
endtime=my_time(endtime,starttime)
with open('knn_second_percent.pickle', 'wb') as f:
    pickle.dump(knn, f)
output_knn=knn.predict(test_data)
num=0
error_num=0
total_test_one=0
for i in range(len(test_label)):
    if(output_knn[i]==test_label[i]):
        num+=1
    else:
        error_num+=1
        print(test_label[i])
    if(test_label[i]==1):
        total_test_one+=1
print("total num:"+str(num+error_num))#總共多少個
print("total line:"+str(num))
print("error line:"+str(error_num))
print("total one in test data:"+str(total_test_one))
endtime=my_time(endtime,starttime)
"""
"""
for word, pos in zip(word_sentence, pos_sentence):
    if(pos=='Nb'):
        out.write(word)
        out.write(',')
"""

#print(pos_sentence)
#print(word_extract)
