import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn import neural_network

def standard(tasks, attri):
    scaler = preprocessing.StandardScaler()
    tasks[attri] = scaler.fit_transform(tasks[attri])

tasks = pd.read_csv('final_feature_linear.csv', header=0)

standard(tasks, '任务标价')
completeCount = len(tasks)
foldSize = completeCount / 5
index = [i for i in range(completeCount)]
np.random.shuffle(index)

tasks['index'] = index

reg = neural_network.MLPClassifier((80, 20), max_iter=600)

for i in range(5):
    test  = tasks[(tasks['index'] >= i * foldSize)&(tasks['index'] < (i+1)*foldSize)]
    train = tasks[(tasks['index'] < i * foldSize)|(tasks['index'] >= (i+1)*foldSize)]

    Y = train['任务执行情况']
    X = train[['任务gps经度',
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
           '任务标价'
                ]]
    Y_test = test['任务执行情况']
    X_test = test[['任务gps经度',
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
           '任务标价'
                ]]
    
    reg.fit(X, Y)
    result = reg.predict(X_test)

    acc = accuracy_score(Y_test, result)
    print("acc: ", acc)

    ## predict new sample
    # tasks_not_complete
    # print(reg.coef_)
    # print(reg.intercept_)

# Y = tasks['任务执行情况']
# X = tasks[['任务gps经度',
#            '任务gps纬度',
#            'vip_count_around_33934.34627910165',
#            'averaged_credit_33934.34627910165',
#            'vip_count_around_16967.173139550825',
#            'averaged_credit_16967.173139550825',
#            '位置_factorized',
#            '【24653.4159638米内】任务数',
#            '【12326.7079819米内】任务数',
#            'averaged_limit16967.173139550825',
#            'averaged_limit33934.34627910165',
#            '任务到会员的最小距离',
#            '任务到会员的平均值',
#            '任务标价']]

# reg.fit(X, Y)
# print(accuracy_score(Y, reg.predict(X)))
# result = reg.predict(X)
# print(sum([1 if i == 1 else 0 for i in result]))
