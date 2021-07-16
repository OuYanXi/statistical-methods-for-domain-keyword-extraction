# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:51:30 2020

@author: yms
"""
import os
import sys
import math
import pandas as pd
import numpy as np
import time
import json
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def df_tf_features(data, outfile_path, method): #'content_S','industry_id' in data.columns

    print('开始读取文件，合并同行业得到new_data')
    new_data = data.groupby(by='industry_id').apply(lambda x:' '.join(x['content_S']))
    new_data = pd.DataFrame(new_data,columns=['content_S'])
    new_data = new_data.reset_index()

    if method in ['all','unigram']:
        print('开始unigram')
        st = time.time()
        vector = CountVectorizer(min_df=1, ngram_range=(1,1))
        result = vector.fit_transform(new_data.content_S.values) # transform text to metrix
        df_unigram = pd.DataFrame(result.toarray(), columns=vector.get_feature_names(),                   index=new_data.industry_id.values)
        df_unigram = df_unigram.T.reset_index().rename(columns={'index':'words'})

        '''unigram排序并写入文本文件，每个行业写入一个文件'''
        print('开始unigram保存文件')
        tags_list = [tag for tag in df_unigram.columns if tag != 'words']
        for tag in tags_list:
            #print([x for x in tag])
            outfile_name = 'unigram_words_ind'+ str(tag) + '.txt'
            outtxt_path = os.path.join(outfile_path,outfile_name)  #保存文件路径和名字

            f = open(outtxt_path, 'w',encoding='utf-8-sig')
            sort_id = df_unigram[['words',tag]].sort_values(by=tag, ascending=False)
            sort_id = sort_id.reset_index(drop=True)
            sort_id = sort_id[sort_id[tag]>0] #去掉没有出现的单词
            #print(sort_id[0:10])
            for i in range(len(sort_id)):
        #     for i in range(30): #选择头部词语个数
                 f.write(sort_id.words[i] + "\t" + str(sort_id[tag][i]) + "\n")
            f.close()

        et = time.time()
        print("unigram运行时间：%.8s s" % (et-st))


    if method in ['all','tf_idf']:
        print('开始tf-idf')
        st = time.time()
        vector = TfidfVectorizer()
        result = vector.fit_transform(new_data.content_S.values)
        df_tf_idf = pd.DataFrame(result.toarray(),columns=vector.get_feature_names(),index=new_data.industry_id.values)
        df_tf_idf = df_tf_idf.T.reset_index().rename(columns={'index':'words'})

        print('开始tf-idf保存文件')
        '''tf-idf排序并写入文本文件，每个行业写入一个文件'''
        tags_list = [tag for tag in df_tf_idf.columns if tag != 'words']
        for tag in tags_list:
            #print(type(tag))
            outfile_name = 'tf_idf_words_ind'+ str(tag) + '.txt'
            outtxt_path = os.path.join(outfile_path,outfile_name)  #保存文件路径和名字

            f = open(outtxt_path, 'w',encoding='utf-8-sig')
            sort_id = df_tf_idf[['words',tag]].sort_values(by=tag, ascending=False)
            sort_id = sort_id.reset_index(drop=True)
            sort_id = sort_id[sort_id[tag]>0] #去掉没有出现的单词
            #print(sort_id[0:10])
            for i in range(len(sort_id)):
        #     for i in range(30): #选择头部词语个数
                 f.write(sort_id.words[i] + "\t" + str(sort_id[tag][i]) + "\n")
            f.close()

        et = time.time()
        print("tf-idf运行时间：%.8s s" % (et-st))
