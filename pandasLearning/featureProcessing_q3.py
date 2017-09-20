import numpy as np
import pandas as pd
from math import *
from sklearn import preprocessing

# def cut(tasks):
#     count = 4
#     tasks['任务gps经度'] = pd.factorize(pd.qcut(tasks['任务gps经度'], count))[0]
#     tasks['任务gps纬度'] = pd.factorize(pd.qcut(tasks['任务gps纬度'], count))[0]
    
#     attribute = 'vip_count_around_33934.34627910165'
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]
    
#     attribute = 'averaged_begining_time_33934.34627910165'
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]

#     attribute = 'averaged_credit_33934.34627910165'
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]

#     attribute = 'vip_count_around_16967.173139550825'
#     #tasks[attribute][tasks.where(tasks[attribute]==0)[0]] = tasks[attribute][tasks[attribute].nonzero()[0]].min() / 10
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]

#     attribute = 'averaged_begining_time_16967.173139550825'
#     #tasks[attribute][tasks.where(tasks[attribute]==0)[0]] = tasks[attribute][tasks[attribute].nonzero()[0]].min() / 10
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]

#     attribute = 'averaged_credit_16967.173139550825'
#     #tasks[attribute][tasks.where(tasks[attribute]==0)[0]] = tasks[attribute][tasks[attribute].nonzero()[0]].min() / 10
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]

#     attribute = '【24653.4159638米内】任务数'
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]

#     attribute = '【12326.7079819米内】任务数'    
#     #tasks[attribute][tasks.where(tasks[attribute]==0)[0]] = tasks[attribute][tasks[attribute].nonzero()[0]].min() / 10
#     tasks[attribute] = pd.factorize(pd.qcut(tasks[attribute], count))[0]
def standard(tasks, attri):
    scaler = preprocessing.StandardScaler()
    tasks[attri] = scaler.fit_transform(tasks[attri])

tasks = pd.read_csv('/home/laboratory/Desktop/math/result_q3.txt', header=0)
vips  = pd.read_csv('/home/laboratory/Desktop/math/vips.csv', header=0)

# tasks = tasks[(tasks['任务gps纬度'] % 0.000001 > 1e-8 )]
# tasks = tasks[(tasks['任务gps经度'] % 0.00001 > 1e-7 )]
# tasks = tasks[(tasks['任务标价'] >= 65) & (tasks['任务标价'] <= 75)]
# tasks = tasks[(tasks['任务标价'] >= 66.3) & (tasks['任务标价'] <= 69.3)]

# vips = vips[(vips['会员gps纬度'] % 0.00001 > 1e-7)]
# vips = vips[(vips['会员gps经度'] % 0.00001 > 1e-7)]


standard(tasks, '任务gps经度')
standard(tasks, '任务gps纬度')
standard(tasks, 'vip_count_around_33934.34627910165')
standard(tasks, 'averaged_credit_33934.34627910165')
standard(tasks, 'vip_count_around_16967.173139550825')
standard(tasks, 'averaged_credit_16967.173139550825')
standard(tasks, 'averaged_limit16967.173139550825')
standard(tasks, 'averaged_limit33934.34627910165')
standard(tasks, '位置_factorized')
standard(tasks, '【24653.4159638米内】任务数')
standard(tasks, '【12326.7079819米内】任务数')
standard(tasks, '任务到会员的最小距离')
standard(tasks, '任务到会员的平均值')
# standard(tasks, '质心和')

tasks = tasks[[
           '任务gps经度', 
           '任务gps纬度', 
           'vip_count_around_33934.34627910165', 
           'averaged_credit_33934.34627910165',
           'vip_count_around_16967.173139550825',
           'averaged_credit_16967.173139550825',
           '位置_factorized',
           '【24653.4159638米内】任务数',
           '【12326.7079819米内】任务数',
           'averaged_limit16967.173139550825',
           'averaged_limit33934.34627910165',
           '任务标价',
           '任务执行情况',
           '任务到会员的最小距离',
           '任务到会员的平均值',
           '质心和'
           ]]

tasks.to_csv("q3_linear.csv", index=False)
vips.to_csv('vips_linear.csv', index=False)
#tasks.to_csv("final_feature_tree.csv")