dataPath = "/home/laboratory/Downloads/dutir_tianchi_mobile_recommend_train_user_train.csv"
import csv
import MySQLdb

conn=MySQLdb.connect(host='localhost',user='root',passwd='lab',db='machineLearning',port=3306)
cur=conn.cursor()

with open(dataPath, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        value1 = row[0]
        value2 = row[1]
        value3 = row[2]
        value4 = row[3]
        value5 = row[4]
        value6 = row[5]
        try:
            sql = "INSERT INTO `dutir_tianchi_mobile_recommend_train_user_train`(`user_id`, `item_id`, `behavior_type`, `user_geohash`, `item_category`, `time`) VALUES ('"+value1+"','"+value2+"','"+value3+"','"+value4+"','"+value5+"','"+value6+":00:00')"
            cur.execute(sql)
            conn.commit()
        except MySQLdb.Error,e:
            print "Mysql Error %d: %s" % (e.args[0], e.args[1])
            print(row)

cur.close()
conn.close()

#'119616442', '239814281', '3', '94gn6nd', '4410', '2014-11-26 20:00:00'
# INSERT INTO `dutir_tianchi_mobile_recommend_train_user_train` (`user_id`, `item_id`, `behavior_type`, `user_geohash`, `item_category`, `time`) VALUES ('119616442', '239814281', '3', '94gn6nd', '4410', '2014-11-26 20:00:00');