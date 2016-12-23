import MySQLdb
 
try:
    conn=MySQLdb.connect(host='localhost',user='root',passwd='lab',db='machineLearning',port=3306)
    cur=conn.cursor()
    cur.execute('select * from dutir_tianchi_mobile_recommend_train_user_train')
    cur.close()
    conn.close()
except MySQLdb.Error,e:
     print "Mysql Error %d: %s" % (e.args[0], e.args[1])