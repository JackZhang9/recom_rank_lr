#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/6/9 18:16
# @Author : CN-JackZhang
# @File: check.py
'''用lr模型去检查测试文件的表现
准确率:accuracy, 1 0 1 0 1
                0 1 0 0 1  准确率：accuracy=2/5
auc：模型预测样本的概率，
每个样本的概率
1 0.9
1 0.8
1 0.3
0 0.2
0 0.4
分母：正负样本对3*2=6
分子：2+2+1=5
auc=5/6=0.83,在点击率预估中，只要auc高于0.7，便可以上线应用，
'''
import numpy as np
import joblib
import math


def get_test_data(test_file):
    '''

    :param test_file:
    :return:
    '''
    total_feature = 118
    #获得label
    test_label = np.genfromtxt(test_file,dtype=np.float,delimiter=',',usecols=[-1])
    #获得所有feature
    feature_list = range(total_feature)
    test_feature = np.genfromtxt(test_file,dtype=np.float,delimiter=',',usecols=feature_list)
    return test_feature,test_label


def lr_model_predict(test_feature,lr_model):
    '''
    得到每个样本为1的概率，
    :param test_feature:
    :param lr_model:
    :return:
    '''
    #result_list记录所有预测为1的概率
    result_list = []
    #proba返回所有结果的概率，此时有0，1两个结果，就有两个概率，从左往右对应0，1的概率
    proba_list = lr_model.predict_proba(test_feature)
    #得到结果为1的概率
    for index in range(len(proba_list)):
        result_list.append(proba_list[index][1])
    return result_list

def sigmoid(x):
    '''跃迁函数sigmoid'''
    return 1/(1+math.exp(-x))


def lr_coef_predict(test_feature,model_coef):
    '''
    每一个特征样本与模型参数相乘，然后过一下阶跃函数
    :param test_feature:
    :param model_coef:
    :return:
    '''
    sigmoid_func = np.frompyfunc(sigmoid,1,1)
    return sigmoid_func(np.dot(test_feature,model_coef))


def get_auc(predict_list,test_label):
    '''
    得到auc，auc=(sum(pos_index)-pos_num(pos_num+1)/2) / pos_num*neg_num
    :param predict_list:
    :param test_label:
    :return:none
    '''
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label,predict_score))
    sorted_total_list = sorted(total_list,key = lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label,predict_score = zuhe
        if label == 0:
            neg_num+=1
        else:
            pos_num+=1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num)*(pos_num+1)/2) / (pos_num*neg_num)
    # print('auc:',auc_score)


def get_accuracy(predict_list,test_label):
    '''

    :param predict_list:模型预测打分列表
    :param test_label:
    :return: none
    '''
    right_num = 0
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= 0.5:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuracy_score = right_num/total_num
    # print('accuracy:',accuracy_score)



def run_check_core(test_feature,test_label,lr_model,score_func):
    '''
    计算AUC和accuracy
    :param test_feature:
    :param test_label:
    :param lr_model:
    :param score_func: 使用不同的model预测打分
    :return: none
    '''
    #给每个样本输出label为1的概率
    predict_list = score_func(test_feature,lr_model)
    get_auc(predict_list,test_label)
    get_accuracy(predict_list,test_label)


def run_check(test_file,model_coef_file,model_file):
    '''

    :param test_file:
    :param model_coef: w1,w2,...
    :param model_file: dump file
    :return:none
    '''
    test_feature,test_label = get_test_data(test_file)
    model_coef = np.genfromtxt(model_coef_file,dtype=np.float,delimiter=',')
    lr_model = joblib.load(model_file)
    #计算AUC和accuracy
    run_check_core(test_feature,test_label,lr_model,lr_model_predict)
    run_check_core(test_feature,test_label,model_coef,lr_coef_predict)


if __name__ == '__main__':
    run_check('../data/test_file','../data/model_coef','../data/model_file')

'''
测试效果不错
auc: 0.8973855633802816
accuracy: 0.8428286852589641
auc: 0.8973853968405024
accuracy: 0.7448207171314741
auc>0.7，即是可用的模型
'''