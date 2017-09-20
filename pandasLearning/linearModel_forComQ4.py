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
from sklearn import model_selection

tasks = pd.read_csv('final_feature_linear.csv', header=0)

print("old: ", sum(tasks['任务标价']))

tasks_complete = tasks[(tasks['任务执行情况']==1)]
tasks_not_complete = tasks[(tasks['任务执行情况']==0)]

# reg = Pipeline([('poly', PolynomialFeatures(degree=1)), ('linear', linear_model.Lasso(alpha=0.01, max_iter=2000))])

# completeCount = len(tasks_complete)
# foldSize = completeCount / 5
# index = [i for i in range(completeCount)]
# np.random.shuffle(index)

# tasks_complete['index'] = index

# folds = []

# for i in range(5):
#     test  = tasks_complete[(tasks_complete['index'] >= i * foldSize)&(tasks_complete['index'] < (i+1)*foldSize)]
#     train = tasks_complete[(tasks_complete['index'] < i * foldSize)|(tasks_complete['index'] >= (i+1)*foldSize)]

#     Y = train['任务标价']
#     X = train[['任务gps经度',
#             '任务gps纬度',
#             'vip_count_around_33934.34627910165',
#             'averaged_credit_33934.34627910165',
#             'vip_count_around_16967.173139550825',
#             'averaged_credit_16967.173139550825',
#             '位置_factorized',
#             '【24653.4159638米内】任务数',
#             '【12326.7079819米内】任务数',
#             'averaged_limit16967.173139550825',
#             'averaged_limit33934.34627910165',
#             '任务到会员的最小距离',
#             '任务到会员的平均值',
#                 ]]
#     Y_test = test['任务标价']
#     X_test = test[['任务gps经度',
#             '任务gps纬度',
#             'vip_count_around_33934.34627910165',
#             'averaged_credit_33934.34627910165',
#             'vip_count_around_16967.173139550825',
#             'averaged_credit_16967.173139550825',
#             '位置_factorized',
#             '【24653.4159638米内】任务数',
#             '【12326.7079819米内】任务数',
#             'averaged_limit16967.173139550825',
#             'averaged_limit33934.34627910165',
#             '任务到会员的最小距离',
#             '任务到会员的平均值',
#                 ]]
    
#     reg.fit(X, Y)
#     result = reg.predict(X_test)

#     r2 = r2_score(Y_test, result)
#     print("r2: ", r2)

    ## predict new sample
    # tasks_not_complete
    # print(reg.coef_)
    # print(reg.intercept_)

Y = tasks_complete['任务标价']
X = tasks_complete[['任务gps经度',
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
reg = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.Lasso(alpha=0.001, max_iter=2000))])
reg.fit(X, Y)
result = reg.predict(X)
r2 = r2_score(Y, result)
print("r2: ", r2)
print(reg.named_steps['linear'].coef_)
print(reg.named_steps['linear'].intercept_)

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

tasks.to_csv('./ning/q4_pred.csv', index=False)