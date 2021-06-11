#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/6/9 8:29
# @Author : CN-JackZhang
# @File: train.py
'''训练lr模型'''
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
import joblib
from utility import get_feature_num as gf

def train_lr_model(train_file,model_coef,model_file,feature_num_file):
    '''
    训练lr模型
    :param train_file:
    :param model_coef:w1,w2,...
    :param model_file:model pkl
    :param feature_num_file:记录特征的个数
    :return:none 实例化/写入文件中
    '''
    total_feature_num = gf.get_feature_num(feature_num_file)
    #使用np模块读入文件,得到np的ndarray数据类型,得到所有特征
    train_label = np.genfromtxt(train_file,dtype=np.int,delimiter=',',usecols=[-1])
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file,dtype=np.int,delimiter=',',usecols=feature_list)
    # lr_clf = LRCV(Cs=[1],cv=5,tol=0.0001,max_iter=500)
    # lr_clf.fit(train_feature,train_label)
    # scores = lr_clf.scores_.values()
    # for i in scores:
    #     scores = i
    # print('05',scores.mean(axis=0))
    # print('0Accuracy',scores.mean(),scores.std()*2)
    #实例化一个lr分类器
    lr_clf = LRCV(Cs=[1],cv=5,tol=0.0001,max_iter=500,scoring='roc_auc')
    lr_clf.fit(train_feature,train_label)
    # scores = lr_clf.scores_.values()
    # for i in scores:
    #     scores = i
    # print('05',scores.mean(axis=0))
    # print('0Accuracy',scores.mean(),scores.std()*2)
    coef = lr_clf.coef_[0]
    #保存lr模型参数
    with open(model_coef,'w+') as fp:
        fp.write(','.join([str(ele) for ele in coef]))
    fp.close()
    #保存训练好的lr模型
    joblib.dump(lr_clf,model_file)
    # print(','.join([str(ele) for ele in scores.mean(axis=0)]))


if __name__ == '__main__':
    train_lr_model('../data/train_file','../data/model_coef','../data/model_file','../data/feature_num')
