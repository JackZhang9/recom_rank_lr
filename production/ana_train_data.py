#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/6/7 8:46
# @Author : CN-JackZhang
# @File: ana_train_data.py
'''1.特征选择和样本选择，2.离散特征处理，连续特征处理，3.LR模型训练'''
import pandas as pd
import numpy as np
import sys


pd.set_option('display.max_columns',15)
pd.set_option('display.width',200)


def get_input(input_train_file,input_test_file):
    '''
    特征选择和样本选择：选择需要的特征，样本去除空值
    :param input_train_file:
    :param input_test_file:
    :return:两个dataframe,train_dataframe和test_dataframe
    '''
    #要使用的cols
    use_cols = [i for i in range(15)]
    use_cols.remove(2)
    #定义数据类型
    d_types = {'age': np.int32,
              'workclass':np.object,
              'education':np.object,
              'education-num':np.int32,
              'marital-status':np.object,
              'occupation':np.object,
              'sex':np.object,
              'capital-gain':np.int32,
              'capital-loss':np.int32,
              'hours-per-week':np.int32,
              'native-country':np.object}
    #读取输入文件,得到训练数据和测试数据的dataframe，默认header=0,即第一行为表头，默认sep=',',即以','为分隔符，
    train_data_df = pd.read_csv(input_train_file,usecols=use_cols,dtype=d_types,na_values='?')
    test_data_df = pd.read_csv(input_test_file,usecols=use_cols,dtype=d_types,na_values='?')
    #对dataframe只要有空值就丢掉,默认axis=0,how='any',即删除行。有空值就删除，需要接受返回值
    train_data_df = train_data_df.dropna()
    test_data_df = test_data_df.dropna()
    return train_data_df,test_data_df


def label_trans(label):
    '''label转换成:'0'或'1'
    '''
    if label == '<=50K':
        return '0'
    else:
        return '1'


def process_label_feature(label,df_input):
    '''
    处理label
    :param label:'label'
    :param df_input:
    :return: none
    '''
    df_input.loc[:,label] = df_input.loc[:,label].apply(label_trans)


def dict_trans(origin_dict):
    '''
    得到每个key的位置/索引
    :param origin_dict:a dict,key str,value int
    :return: a dict,key str, value index
    '''
    output_dict = {}
    index = 0
    for zuhe in origin_dict.items():
        # print(zuhe)
        output_dict[zuhe[0]] = index
        index += 1
    return output_dict


def dis_feature_trans(dis_feature,feature_dict):
    '''
    离散特征转换：将离散特征转换成'0,1,0'
    :param dis_feature:
    :param feature_dict: pos dict
    :return:a list,as [1,0,1],a str,as '0,1,0'
    '''
    output_list = [0]*len(feature_dict)
    if dis_feature not in feature_dict:
        return ','.join([str(ele) for ele in output_list])
    else:
        index = feature_dict[dis_feature]
        output_list[index] = 1
    return ','.join([str(ele) for ele in output_list])


def process_dis_feature(dis_feature,train_data_df,test_data_df):
    '''
    处理离散特征：逐行处理
    :param dis_feature:'label'
    :param df_input:
    :return:none
    '''
    #看dis_feature列有多少不同的数值，统计个数，并以字典形式保存,如：{'Private': 22286, 'Self-emp-not-inc': 2499, 'Local-gov': 2067, 'State-gov': 1279, 'Self-emp-inc': 1074, 'Federal-gov': 943, 'Without-pay': 14}
    origin_dict = train_data_df.loc[:,dis_feature].value_counts().to_dict()
    # workclass的dict {'Private': 22286, 'Self-emp-not-inc': 2499, 'Local-gov': 2067, 'State-gov': 1279, 'Self-emp-inc': 1074, 'Federal-gov': 943, 'Without-pay': 14}
    feature_dict = dict_trans(origin_dict)
    train_data_df.loc[:,dis_feature] = train_data_df.loc[:,dis_feature].apply(dis_feature_trans,args=(feature_dict,))
    test_data_df.loc[:,dis_feature] = test_data_df.loc[:,dis_feature].apply(dis_feature_trans,args=(feature_dict,))
    return len(feature_dict)

def list_trans(origin_dict):
    '''

    :param origin_dict: a dict
    :return: a list:[0.1,0.2,0.3,0.4,0.5]
    '''
    output_list = [0]*5
    key_list = ['min','25%','50%','75%','max']
    for index in range(len(key_list)):
        fix_key = key_list[index]
        if fix_key not in origin_dict:
            print('error')
            sys.exit()
        else:
            output_list[index] = origin_dict[fix_key]
    return output_list


def con_to_feature(con_feature,feature_list):
    '''
    连续特征转换，对符合某一区间的元素变为1
    :param con_feature:每一行元素
    :param feature_list:list for feature trans
    :return: str:'1_0_1'
    '''
    feature_len = len(feature_list)-1
    result = [0]*feature_len
    for index in range(feature_len):
        if con_feature >= feature_list[index] and con_feature <= feature_list[index+1]:
            result[index] = 1
            return ','.join([str(ele) for ele in result])
    return ','.join([str(ele) for ele in result])


def process_con_feature(con_feature,train_data_df,test_data_df):
    '''
    处理连续特征
    :param con_feature:
    :param train_data_df:
    :param test_data_df:
    :return:none
    '''
    #连续特征处理要先统计分布
    origin_dict=train_data_df.loc[:,con_feature].describe().to_dict()
    feature_list = list_trans(origin_dict)
    train_data_df.loc[:,con_feature] = train_data_df.loc[:,con_feature].apply(con_to_feature,args=(feature_list,))
    test_data_df.loc[:,con_feature] = test_data_df.loc[:,con_feature].apply(con_to_feature,args=(feature_list,))
    return len(feature_list)-1

def output_file(df_in, out_file):
    '''
    将dataframe写入out_file
    :param df_in:
    :param out_file:
    :return:
    '''
    with open(out_file,'w+') as fp:
        for row_index in df_in.index:
            outline = ','.join([str(ele) for ele in df_in.loc[row_index].values])
            fp.write(outline+'\n')
    fp.close()







def ann_train_data(input_train_file, input_test_file,out_train_file,out_test_file,feature_num_file):
    '''

    :param input_train_file:
    :param input_test_file:
    :param out_train_file:
    :param out_test_file:
    :param feature_num_file:
    :return:
    '''
    #得到样本和特征
    train_data_df,test_data_df=get_input(input_train_file,input_test_file)
    #处理label
    process_label_feature('label',train_data_df)
    process_label_feature('label',test_data_df)
    feature_num_dict = {}
    #处理离散特征dis_feature，如：workclass
    dis_feature_num = 0
    dis_feature_list = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    for dis_feature in dis_feature_list:
        tmp_feature_num=process_dis_feature(dis_feature,train_data_df,test_data_df)
        dis_feature_num += tmp_feature_num
        feature_num_dict[dis_feature] = tmp_feature_num
    #处理连续特征con_feature,如：age
    con_feature_num = 0
    con_feature_list = ['age','education-num','capital-gain','capital-loss','hours-per-week']
    for con_feature in con_feature_list:
        tmp_feature_num=process_con_feature(con_feature,train_data_df,test_data_df)
        con_feature_num += tmp_feature_num
        feature_num_dict[con_feature] = tmp_feature_num
    #将dataframe写入输出文件out_file
    output_file(train_data_df,out_train_file)
    output_file(test_data_df,out_test_file)
    with open(feature_num_file,'w+') as fp:
        fp.write('feature_num='+str(dis_feature_num + con_feature_num))
    fp.close()


if __name__ == '__main__':
    # train_data_df,test_data_df = get_input('../data/train.txt','../data/test.txt')
    # print(train_data_df.head(10),test_data_df.head(10),len(train_data_df),len(test_data_df))
    # ann_train_data('../data/train.txt','../data/test.txt',None,None)
    # dict={'Private': 22286, 'Self-emp-not-inc': 2499, 'Local-gov': 2067, 'State-gov': 1279, 'Self-emp-inc': 1074,'Federal-gov': 943, 'Without-pay': 14}
    # print(dict_trans(dict))
    # process_dis_feature('workclass',train_data_df,test_data_df)
    ann_train_data('../data/train.txt','../data/test.txt','../data/train_file','../data/test_file','../data/feature_num')