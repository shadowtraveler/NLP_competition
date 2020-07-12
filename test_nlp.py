# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:41:19 2020

@author: steve
"""

import pandas as pd
from ckiptagger import  WS, POS#, NER ,construct_dictionary
import csv
import numpy as np

#做斷詞
"""
ws = WS("./data")
pos = POS("./data")
#ner = NER("./data")
Verb=[['VA'],['VB'],['VC'],['VD'],['VE'],['VF'],['VG'],['VH'],['VI'],['VJ'],['VK'],['VL']]
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
for word, pos in zip(word_sentence, pos_sentence):
    if(pos=='Nb'):
        out.write(word)
        out.write(',')
"""

#print(pos_sentence)
#print(word_extract)
