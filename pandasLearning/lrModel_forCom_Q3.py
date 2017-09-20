import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import neural_network

def standard(tasks, attri):
    scaler = preprocessing.StandardScaler()
    tasks[attri] = scaler.fit_transform(tasks[attri])

tasks = pd.read_csv('final_feature_linear.csv', header=0)

standard(tasks, '任务标价')
Y = tasks['任务执行情况']
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
           '任务标价']]

reg = neural_network.MLPClassifier((80, 20), max_iter=600)

reg.fit(X, Y)
print(accuracy_score(Y, reg.predict(X)))
result = reg.predict(X)
print(sum([1 if i == 1 else 0 for i in result]))

#print(len(tasks[(tasks['任务执行情况'] == 1)]))
tasks = pd.read_csv('/home/laboratory/Desktop/math/ning/q3_pred.csv', header=0)
standard(tasks, '任务标价')
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
           '任务标价']]
# for i in range(len(X)):
#     X['任务标价'][i] = 40
result = reg.predict(X)
count = 0
for i in range(len(result)):
    if tasks['质心和'][i] > 0.0:
        if result[i] == 1:
            count += 6
    else:
        if result[i] == 1:
            count += 1


#print(sum([1 if i == 1 else 0 for i in result]))
print(count)
count = 0
# for i in range(len(result)):
#     if result[i] == 1:
#         count += tasks['cont'][i]
# print(count)
# # print(reg.coef_)
# # print(reg.intercept_)

# 489
# 539
