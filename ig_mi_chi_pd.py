# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:54:21 2020

@author: yms
"""

import os
import sys
import math
import pandas as pd
import numpy as np
import time
import json
import shutil
import collections
import warnings
warnings.filterwarnings("ignore")

'''清空非空目录'''
def CleanDir( Dir ):
    if os.path.isdir( Dir ):
        paths = os.listdir( Dir )
        for path in paths:
            filePath = os.path.join( Dir, path )
            if os.path.isfile( filePath ):
                try:
                    os.remove( filePath )
                except os.error:
                    print( "remove %s error." %filePath )#引入logging
            elif os.path.isdir( filePath ):
#                 if filePath[-4:].lower() == ".svn".lower():
#                     continue
                shutil.rmtree(filePath,True)

'''创建新的保存目录'''
def add_new_path(path):
    if os.path.exists(path):
        CleanDir(path)
    else:
        os.makedirs(path)
    return path

'''判断词汇是否为纯数字'''
def not_number(s):
    try:
        float(s)
        return False
    except:
        return True

#%%

def cal_words(data):
    '''统计行业标签{词：标签1@标签2}'''
    worddf_dict={}
    tag_list=[]
    txt_list = list(data.values)
    for line in txt_list:
        category = line[1]
        tag_list.append(category)
        words_l = line[0].split(' ')
        for word in set(words_l):
            if word not in worddf_dict:
                worddf_dict[word] = category
            else:
                worddf_dict[word] += '@' + category
    return tag_list, worddf_dict

def cal_tags(worddf_dict):
    '''{ 词：{标签1：个数，标签2：个数} }'''
    word_dict={}
    for word, word_category in worddf_dict.items():
        word_category = word_category.split('@')
        cate_dict = dict(collections.Counter(word_category))
        word_dict[word] = cate_dict

    words_df = pd.DataFrame(word_dict).T
    words_df = words_df.fillna(0)
    words_df.index = pd.Series(words_df.index).apply(lambda x: str(x).lower())
    return words_df

#%%
def cal_mi_ig_chi(tag_list,words_df,tag,method,a=0.0001): #a = 0.0001 防止domain error
    words_df_ind = words_df[words_df[tag]!=0]
    words_list = list(words_df_ind.index)
    cal_list = ['N_word','N_nword','N_tag','N_ntag','N_word_tag','N_word_ntag','N_nword_tag','N_nword_ntag']
    N = len(tag_list)
    words_ind = pd.DataFrame(data=None,index=words_list,columns=cal_list)

    words_ind['N_word'] = words_df.sum(1)
    words_ind['N_nword'] = N - words_ind['N_word']

    N_tag = tag_list.count(tag)
    N_ntag = N - N_tag
    words_ind['N_tag'] = N_tag
    words_ind['N_ntag'] = N_ntag
    H_tag = - ((N_tag+a)/N)*math.log(((N_tag+a)/N)) - ((N_ntag+a)/N)*math.log(((N_ntag+a)/N))
    words_ind['H_tag'] = H_tag

    words_ind['N_word_tag'] = words_df_ind.loc[:,tag]
    words_ind['N_word_ntag'] = words_ind['N_word'] - words_ind['N_word_tag']

    words_ind['N_nword_tag'] = words_ind['N_tag'] - words_ind['N_word_tag']
    words_ind['N_nword_ntag'] = words_ind['N_ntag'] - words_ind['N_word_ntag']

    words_ind = words_ind.applymap(lambda x: x+a) #避免domain error

    words_ind['p_word_tag'] = words_ind['N_word_tag'] / N
    words_ind['p_word_ntag'] = words_ind['N_word_ntag'] / N
    words_ind['p_nword_tag'] = words_ind['N_nword_tag'] / N
    words_ind['p_nword_ntag'] = words_ind['N_nword_ntag'] / N
    words_ind['p_tag'] = words_ind['N_tag'] / N
    words_ind['p_ntag'] = words_ind['N_ntag'] / N
    words_ind['p_word'] = words_ind['N_word'] / N
    words_ind['p_nword'] = words_ind['N_nword'] / N

    '''mi'''
    if method in ['all','mi']:
        print('正在计算mi')
        # #以下为互信息的计算 p(x,y)*log( p(x,y) / p(x)*p(y) )的累加
        # words_ind['pmi_word_tag'] = words_ind.apply(lambda x: x['p_word_tag'] * math.log((x['p_word_tag'] / (x['p_word']*x['p_tag']))),axis=1)
        # words_ind['pmi_nword_tag'] = words_ind.apply(lambda x: x['p_nword_tag'] * math.log((x['p_nword_tag'] / (x['p_nword']*x['p_tag']))),axis=1)
        # words_ind['pmi_word_ntag'] = words_ind.apply(lambda x: x['p_word_ntag'] * math.log((x['p_word_ntag'] / (x['p_word']*x['p_ntag']))),axis=1)
        # words_ind['pmi_nword_ntag'] = words_ind.apply(lambda x: x['p_nword_ntag'] * math.log((x['p_nword_ntag'] / (x['p_nword']*x['p_ntag']))),axis=1)
        # words_ind['mi'] = words_ind['pmi_word_tag'] + words_ind['pmi_nword_tag'] + words_ind['pmi_word_ntag'] + words_ind['pmi_nword_ntag']
        words_ind['mi'] = words_ind.apply(lambda x: math.log((x['p_word_tag'] / (x['p_word']*x['p_tag']))),axis=1) #实质为计算点互信息 log( p(x,y) / p(x)*p(y) )

    '''ig,ig_rate'''
    if method in ['all','ig','ig_rate']:
        print('正在计算ig或ig_rate')
        words_ind['H_word_tag'] = words_ind.apply(lambda x: - (x['N_word_tag']/x['N_word'])*math.log((x['N_word_tag']/x['N_word'])) - (x['N_word_ntag']/x['N_word'])*math.log((x['N_word_ntag']/x['N_word'])),axis=1)
        words_ind['H_nword_tag'] = words_ind.apply(lambda x: - (x['N_nword_tag']/x['N_nword'])*math.log((x['N_nword_tag']/x['N_nword'])) - (x['N_nword_ntag']/x['N_nword'])*math.log((x['N_nword_ntag']/x['N_nword'])),axis=1)
        words_ind['ig'] = words_ind.apply(lambda x: x['H_tag'] - x['p_word']*x['H_word_tag'] - x['p_nword']*x['H_nword_tag'], axis=1)

        if method in ['all','ig_rate']:
            words_ind['H_word'] = words_ind.apply(lambda x: - x['p_word']*math.log(x['p_word']) - x['p_nword']*math.log(x['p_nword']), axis=1)
            words_ind['ig_rate'] = words_ind.apply(lambda x: x['ig'] / x['H_word'], axis=1)

    '''chi'''
    if method in ['all','chi']:
        print('正在计算chi')
        words_ind['numerator'] = (words_ind['N_word_tag'] + words_ind['N_word_ntag'] + words_ind['N_nword_tag'] + words_ind['N_nword_ntag']) *                      (words_ind['N_word_tag']*words_ind['N_nword_ntag'] -words_ind['N_word_ntag']*words_ind['N_nword_tag'])  # * (n11*n00 - n10*n01)
        words_ind['denominator'] = (words_ind['N_word_tag'] + words_ind['N_nword_tag']) * (words_ind['N_word_tag'] + words_ind['N_word_ntag']) *                          (words_ind['N_word_ntag'] + words_ind['N_nword_ntag']) * (words_ind['N_nword_tag'] + words_ind['N_nword_ntag']) + 1
        words_ind['chi'] = words_ind['numerator'] / words_ind['denominator']

    return words_ind

#%%
def save_txt(words_ind,outfile_path,tag,method):
    if method in ['all','mi']:
        words_ind = words_ind.sort_values(by=['mi','N_word_tag'],ascending=[False,False])
        words_ind[['mi']].to_csv(outfile_path + '/mi_words_ind%s.txt'%tag,sep='\t',header=None,encoding='utf_8_sig')
    if method in ['all','ig']:
        words_ind = words_ind.sort_values(by=['ig','N_word_tag'],ascending=[False,False])
        words_ind[['ig']].to_csv(outfile_path + '/ig_words_ind%s.txt'%tag,sep='\t',header=None,encoding='utf_8_sig')
    if method in ['all','ig_rate']:
        words_ind = words_ind.sort_values(by=['ig_rate','N_word_tag'],ascending=[False,False])
        words_ind[['ig_rate']].to_csv(outfile_path + '/ig_rate_words_ind%s.txt'%tag,sep='\t',header=None,encoding='utf_8_sig')
    if method in ['all','chi']:
        words_ind = words_ind.sort_values(by=['chi','N_word_tag'],ascending=[False,False])
        words_ind[['chi']].to_csv(outfile_path + '/chi_words_ind%s.txt'%tag,sep='\t',header=None,encoding='utf_8_sig')

def chi_mi_ig_features(data, outfile_path, method):  #'content_S','industry_id' in data.columns
    st = time.time()

    print('开始统计词标签。。。')
    tag_list, worddf_dict = cal_words(data)
    words_df = cal_tags(worddf_dict)
    print('开始chi,mi,ig,igrate。。。')
    for tag in set(tag_list):
        print('正在执行行业%s...'%tag)
        words_ind = cal_mi_ig_chi(tag_list,words_df,tag,method,a=0.0001)
        #words_ind.to_csv('words_%s_ind%s.csv'%(method,tag),encoding='utf_8_sig')
        save_txt(words_ind,outfile_path,tag,method)

    et = time.time()
    print('计算总耗时%.2fs'%(et-st))

#%%
if __name__ == '__main__':
    infile = 'example_split_text.txt'
    outfile_path = 'output_result'
    outfile_path = add_new_path(outfile_path)
    print('开始读取数据。。。')
    data = pd.read_table(infile,header=None,dtype=str,names=['content_S','industry_id'])
    data = data.dropna()
    data = data.drop_duplicates()
    chi_mi_ig_features(data, outfile_path, method='all')





