# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:38:26 2020

@author: steve
"""

import pandas as pd
import urllib.request as req
import bs4
import csv

#import urllib
"""
from ckiptagger import  WS, POS, NER ,construct_dictionary

print("Loading WS,POS model\n")
ws = WS("./data")
pos = POS("./data")
print("Model loaded.\n")
"""

#df=pd.read_csv('./output_has_name_context_crawl_WS.csv',encoding='utf-8')


#用於爬蟲抓取資料
"""
df=pd.read_csv('./output_no_name.csv',encoding='utf-8')
num=0
error_num=0
with open('output_no_name_context_crawl.csv', 'w',encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['news_ID', 'content','name'])
    for i in range(len(df)):
        url = df.iloc[i,1]
        #url = urllib.parse.quote(url)
        #url = "https://"+ url[10:]
        try:
            request = req.Request(url, headers = {
                "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36"
            })
            with req.urlopen(request) as response:
                data = response.read().decode("utf-8")
    
            soup = bs4.BeautifulSoup(data, "lxml")
            articleContent = soup.find_all('p')
    
            article = []
            for p in articleContent:
                article.append(p.text)
    
            articleAll = '\n'.join(article)
            writer.writerow([df['news_ID'][i],articleAll,df['name'][i]])
            num+=1
            #word_extract=ws(articleAll)
            #pos_sentence=pos(word_extract)
        except:
            error_num+=1
            print("article", df['news_ID'][i],"Error")
print("total num:"+str(num+error_num))#總共多少個
print("total line:"+str(num))
print("error line:"+str(error_num))
"""

#用於將所有有名字的資料拉出轉存在output_has_name.csv
"""
df=pd.read_csv('./tbrain_train_final_0610.csv',encoding='utf-8')
num=0
error_num=0

with open('output_no_name.csv', 'w',encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['news_ID', 'hyperlink', 'content','name'])
    for i in range(len(df)):
    #for i in range(140):
        if(df['name'][i]=='[]'):
            try:
                writer.writerow([df['news_ID'][i],df['hyperlink'][i],df['content'][i],df['name'][i]])
                #writer.writerow(['岳靈珊', 165, 57])
                print(df['news_ID'][i])
                num+=1
            except:
                print("Error"+str(df['news_ID'][i]))
                error_num+=1
print("total num:"+str(num+error_num))#總共多少個
print("total line:"+str(num))
print("error line:"+str(error_num))
#print(df.head(1))
"""

#用於找model位置
"""
import requests
print(requests.__path__)
"""