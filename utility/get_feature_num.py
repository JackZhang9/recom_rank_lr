#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/6/11 7:16
# @Author : CN-JackZhang
# @File: get_feature_num.py


def get_feature_num(feature_num_file):
    fp = open(feature_num_file)
    for line in fp:
        item = line.strip().split("=")
        if item[0] == "feature_num":
            return int(item[1])
    return 0