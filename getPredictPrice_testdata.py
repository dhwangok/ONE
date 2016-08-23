# -*- coding:utf8 -*-
# __author__ = 'lz'
import urllib, urllib2, json
import pymysql


#调用API获取数据
def getChe300PriceData(insertsql):
    model_id,regDate,mile,city_id,id = getDataFromDB(connection(),insertsql)
    # city_id,provin_id = getCityNumByName(city_name)
    url = 'http://api.che300.com/service/getUsedCarPrice?modelId={}&regDate={}&mile={}&zone={}&token=0ba8c35802795c89c0b2d29f284ea56c'.format(model_id,regDate,mile,city_id)
    req = urllib.urlopen(url)
    # print req
    content = json.loads(req.read())
    content = json.dumps(content)
    print content
    c2b = content['dealer_buy_price']
    c2c = content['individual_price']
    b2c = content['dealer_price']
    print c2b
    return c2b,c2c,b2c,id


#城市识别
def getCityNumByName(cityname):
    url = 'http://www.che300.com:8182/identifyCity?CityName={}'.format(cityname)
    req = urllib.urlopen(url)
    content = json.loads(req.read())
    city_id = content['city_id']
    provin_id = content['prov_id']
    return city_id,provin_id
#打开数据库连接
def connection(host='192.168.0.225',user='sa',password='sa@1234',db='test',charset='utf8'):
    conn = pymysql.connect(host=host,user = user,password=password,db=db,charset=charset)
    return conn

#从数据库中获取数据
def getDataFromDB(conn,findsql):
    cur = conn.cursor()
    cur.execute(findsql)
    result = cur.fetchall()
    id = result[0][0]
    city_name =result[0][2].encode('utf8')
    time = result[0][5]
    mile = str(round(result[0][3]/10000,0))[0:-2]

    model_id = result[0][8]
    city_id = result[0][9]


    regDate = str(time)[0:7]

    cur.close()
    conn.close()
    return model_id,regDate,mile,city_id,id


#插入数据库
def insertIntoDB(conn,c2b,c2c,b2c,i):
    sql = 'update che_test set c2b={},c2c={},b2c={} where ID={}'.format(c2b,c2c,b2c,i)
    print '第%d条记录已经生成...'%i
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()



def getId(sql):
    cur = connection().cursor()
    cur.execute(sql)
    result = cur.fetchall()
    idlist = []
    for i in result:
        id = i[0]
        idlist.append(id)
    return idlist


if __name__=='__main__':
    sql = 'SELECT * FROM che_test WHERE c2c IS NULL'
    idlist = getId(sql)
    for id in idlist:
        sql1 = 'SELECT * FROM che_test WHERE ID=%d'%id
        try:
            c2b,c2c,b2c,id = getChe300PriceData(sql1)
            # insertIntoDB(connection(),c2b,c2c,b2c,id)
        except (KeyError, ValueError):
            continue
