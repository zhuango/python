import numpy as np
import pandas as pd
from math import *

def getMinDistances(tasks, vips):
    distances = []
    for longitude, latitude in zip(tasks['任务gps经度'], tasks['任务gps纬度']):
        minDistance = 100000000000000
        for longVip, latiVip in zip(vips['会员gps经度'], vips['会员gps纬度']):
            # print(longitude, latitude, longVip, latiVip)
            dis = cal_dist(longitude, latitude, longVip, latiVip)
            if minDistance > dis:
                minDistance = dis
        distances.append(minDistance)
    return distances

def getMinute(timeStr):
    """6:30:00"""
    items = [int(elem) for elem in timeStr.split(":")]
    minutes = items[0] * 60 + items[1]
    return minutes

def cal_dist(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    求AB两点经纬度之间的距离
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def addVipAroundCount(tasks, vips, thres=0.324208925969):
    vipsAroundCounts = []
    averageBeginningTimes = []
    averageCredits = []
    averageLimit = []
    for longitude, latitude in zip(tasks['任务gps经度'], tasks['任务gps纬度']):
        count       = 0
        totalTime   = 0
        totalCredit = 0
        totalLimit  = 0
        for longVip, latiVip, timeStr, credit, limit in zip(vips['会员gps经度'], vips['会员gps纬度'], vips['预订任务开始时间'], vips['信誉值'], vips['预订任务限额']):
            # print(longitude, latitude, longVip, latiVip)
            dis = cal_dist(longitude, latitude, longVip, latiVip)
            time = getMinute(str(timeStr))
            if dis <= thres:
                count       += 1
                totalTime   += time
                totalCredit += credit
                totalLimit  += limit
        vipsAroundCounts.append(count)
        if count == 0:
            averageBeginningTimes.append(getMinute('8:00:00'))
            averageCredits.append(0.0)
            averageLimit.append(0.0)
        else:
            averageBeginningTimes.append(totalTime / count)
            averageCredits.append(totalCredit / count)
            averageLimit.append(totalLimit / count)
    tasks['vip_count_around_' + str(thres)] = vipsAroundCounts
    tasks['averaged_begining_time_' + str(thres)] = averageBeginningTimes
    tasks['averaged_credit_' + str(thres)] = averageCredits
    tasks['averaged_limit' + str(thres)] = averageLimit
    return vipsAroundCounts, 

def factorizeLocation(tasks):
    tasks['位置_factorized'] = pd.factorize(tasks['位置'])[0]

# tasks: 任务号码,任务gps经度,任务gps纬度,位置,任务标价
# vips:  会员编号,会员gps纬度,会员gps纬度,预订任务限额,预订任务开始时间,信誉值

tasks = pd.read_csv('tasks.csv', header=0)
print(tasks.info())

vips = pd.read_csv('vips.csv', header=0)
print(vips.info())

distances = getMinDistances(tasks, vips)
addVipAroundCount(tasks, vips, max(distances))
addVipAroundCount(tasks, vips, max(distances) / 2)
factorizeLocation(tasks)

tasks.to_excel('featured_tasks.xls')
tasks.to_csv('featured_tasks.csv')