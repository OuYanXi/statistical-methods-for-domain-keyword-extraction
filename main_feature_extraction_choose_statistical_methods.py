# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 17:54:15 2020

@author: SQIT1506017
"""

import os
import sys
import math
import pandas as pd
import numpy as np
import time
import shutil
import json
import warnings
warnings.filterwarnings("ignore")

#print(os.path.dirname(os.path.abspath(__file__)))
print('当前执行路径：',os.getcwd())

#%%

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

def main(infile,outfile_path,txt_top=0,method='all'):
    outfile_path = add_new_path(outfile_path)
    print('开始读取数据')
    data = pd.read_table(infile,header=None,dtype=str,names=['content_S','industry_id'])
    data = data.dropna()
    data = data.drop_duplicates()
    #print(data)
    if txt_top:
        data = data[0:txt_top]

    '''进行unigram和tf-idf,并保存txt'''
    if method in ['all','unigram','tf_idf']:
        import unigram_tfidf as UT
        UT.df_tf_features(data,outfile_path,method) #raw_file不用合并同行业
    '''进行mi_et,并保存txt'''
    if method in ['all','mi','ig','ig_rate','chi']:
        import ig_mi_chi_pd as IMCA
        IMCA.chi_mi_ig_features(data, outfile_path, method)

#%%
if __name__ == '__main__':
    '''输入和输出文件路径'''
    infile = 'example_split_text.txt' 
    # 输入文件位置，’word1’+’ ’+’word2’+’\t’+’ind_id’，输入文本条数一般低于300万，最高不超过500万
    outfile_path = 'result'
    main(infile,outfile_path,txt_top=100,method='all') 
    # txt_top:读取输入文件头部行数，默认全部行
    # method:特征提取方法，可选择项：'all'(默认全部方法), 'unigram'(词频统计), 'tf_idf', 'chi'(卡方), 'ig'(信息增益), 'pmi'(点互信息), 'ig_rate'(信息增益率)



