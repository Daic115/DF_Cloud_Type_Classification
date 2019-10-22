#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   re_clean_data.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/10/8 20:42   Daic       1.0        
'''
import os
import csv
import json
def get_cates(str):
    cat_list = []
    pre_idx = 0
    for x in range(len(str)-1):
        if str[x]==';':
            cat_list.append(int(str[pre_idx:x])-1)
            pre_idx = x+1
    cat_list.append(int(str[pre_idx:])-1)
    return cat_list

def get_main_cate(list,dict):
    final_cate = list[0]
    for li in list:
        if dict[str(li)]>dict[str(final_cate)]:
            final_cate = li

    return final_cate

def one_hot_list(cate):
    onehot = []
    for x in range(29):
        if x in cate:
            onehot.append(1)
        else:
            onehot.append(0)
    return onehot

### this  is  cheecking single label that is belong to the category with most count
# r = csv.reader(open('/media/disk2/daic/Cloud/Train_label.csv'))
# datas = [l for l in r]
# predata = json.load(open('/media/disk2/daic/Cloud/dataset_k1.json'))
# count = {}
# for x in range(29):
#     count[str(x)] = 0
# for tmp in predata:
#     count[str(tmp['cate'])]+=1
#
# for i in range(1,len(datas)):
#     da = datas[i]
#     if len(da[1])>2:
#         tmp_cates = get_cates(da[1])
#         tmp_cate = get_main_cate(tmp_cates,count)
#         img = da[0]
#         for tmp in predata:
#             if tmp['img']==img and tmp['cate']!=tmp_cate:
#                 print("!!!!!!!!!!!!!!!!!!")

### this is getting the mutilabel dataset;

if __name__ == '__main__':
    r = csv.reader(open('./Train_label.csv'))
    datas = [l for l in r]
    predata = json.load(open('./dataset_k1.json'))

    idx = {}
    for x in range(1,len(datas)):
        idx[datas[x][0]] = datas[x][1]

    for x in range(len(predata)):
        k = predata[x]['img']
        oneh = one_hot_list(get_cates(idx[k]))

        predata[x]['muticate'] = oneh

    json.dump(predata)






