import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from setSeperateValue import *

# 任务号码,
# 任务gps经度,
# 任务gps纬度,
# 位置,
# 任务标价,
# vip_count_around_33934.34627910165,
# averaged_begining_time_33934.34627910165,
# averaged_credit_33934.34627910165,
# vip_count_around_16967.173139550825,
# averaged_begining_time_16967.173139550825,
# averaged_credit_16967.173139550825,
# 位置_factorized
# 【24653.4159638米内】任务数
# 【12326.7079819米内】任务数
# 聚类特征
tasks = pd.read_csv('final_feature_linear.csv', header=0)

# tasks = tasks.sort_values(['任务gps经度'])
# tasks = tasks.sort_values(['任务gps纬度'])
# #tasks = tasks[['任务gps经度', '任务gps纬度', '位置_factorized', '任务标价']]
# X_pre = tasks[['任务gps经度', # 0.01745984
#            '任务gps纬度', # 0.26555615
#         #    'vip_count_around_33934.34627910165', # 0.00606003
#         #    'averaged_begining_time_33934.34627910165', # -0.03381279
#         #    'averaged_credit_33934.34627910165', # -0.15653108
#            'vip_count_around_16967.173139550825', # -0.20088015
#         #    'averaged_begining_time_16967.173139550825',# -0.0730108
#         #    'averaged_credit_16967.173139550825',# -0.10147161
#            '位置_factorized',
#         #    '【24653.4159638米内】任务数', # 0.06107608
#         #    '【12326.7079819米内】任务数', # -0.29088841
#         #    'averaged_limit16967.173139550825',
#         #    'averaged_limit33934.34627910165',
#         #    '聚类特征',
#            '任务到会员的最小距离',
#            '任务到会员的平均值'
#         #    '任务执行情况'
#             ]]
# Y_pre = tasks[['任务标价']]

# count = len(tasks)
# generateCount = 50
# rows_list = []

# for i in range(0, count-1, 2):
#     startLong = min([tasks['任务gps经度'][i], tasks['任务gps经度'][i+1]])
#     endLong   = max([tasks['任务gps经度'][i], tasks['任务gps经度'][i+1]])

#     startlati = min([tasks['任务gps纬度'][i], tasks['任务gps纬度'][i+1]])
#     endlati   = max([tasks['任务gps纬度'][i], tasks['任务gps纬度'][i+1]])

#     startPrice = min([tasks['任务标价'][i], tasks['任务标价'][i+1]])
#     endPrice = min([tasks['任务标价'][i], tasks['任务标价'][i+1]])

#     for j in range(generateCount):
#         newLong = np.random.uniform(startLong, endLong)
#         newLati = np.random.uniform(startlati, endlati)
#         price   = np.random.uniform(startPrice, endPrice)
#         # longts.append(newLong)
#         # longts.append(newLati)
#         # location.append(X['位置_factorized'][0])
#         # X.set_value(len(X), newLong, newLati, X['位置_factorized'][i])
#         rows_list.append({'任务gps经度':newLong, '任务gps纬度':newLati, '位置_factorized': tasks['位置_factorized'][i],
#                             '任务标价':price})
# df = pd.DataFrame(rows_list)

# tasks = pd.concat([tasks, df])

Y = tasks['任务标价']
X = tasks[['任务gps经度',
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
           '任务到会员的最小距离',
           '任务到会员的平均值',
            ]]


print("DDDDDDDDDDDD")
#reg = linear_model.Ridge(alpha=0.1)
reg = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('linear', linear_model.Lasso(alpha=0.006, max_iter=2000))])

reg.fit(X, Y)
result = reg.predict(X)
# setSeperateValue(result)

r2 = r2_score(Y, result)
print("r2: ", r2)
print(reg.named_steps['linear'].coef_)
print(reg.named_steps['linear'].intercept_)
# print(reg.named_steps['linear'].intercept_)
# print(reg.coef_)
# print(reg.intercept_)

tasks = pd.read_csv('/home/laboratory/Desktop/math/q4_linear.csv', header=0)

# Y_q4 = tasks['任务标价']
X_q3= tasks[['任务gps经度',
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
            '任务到会员的最小距离',
            '任务到会员的平均值',
            ]]
result = reg.predict(X_q3)
coe = reg.named_steps['linear'].coef_
keyCount = 0
# for key in tasks:
#     for i in range(len(result)):
#         result[i] = result[i] + coe[keyCount] * tasks[key][i] * tasks['质心和'][i]
#     keyCount += 1
# for i in range(len(result)):
#     result[i] = result[i] + 0.043 * tasks['质心和'][i]
print(sum(result))
tasks = pd.read_csv('/home/laboratory/Desktop/math/q4_linear.csv', header=0)
tasks['任务标价'] = list(result)
tasks.to_csv('./ning/q4_pred2.csv', index=False)