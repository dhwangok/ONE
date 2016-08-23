#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import pandas as pd
import pymysql
print 1

conn=pymysql.connect(host="192.168.0.225",user="majian",passwd="MjAn@9832#",db="majian",charset="utf8")
cursor = conn.cursor()
sql="select distinct series_id from da_car_series_i"
cursor.execute(sql)
result = cursor.fetchall()
# print result 
distinct_series_id=[]
for line in result:
    distinct_series_id.append(line[0])
# print distinct_series_id 

for j in distinct_series_id:
    print j
    sql= "select series_id,model_id,prov_id,reg_year,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,old_eval_price from da_car_series_i where series_id = %s"%j
    cursor.execute(sql)
    # alldata = cursor.fetchall()
    # for row in alldata:   
    #       print('%s\t%s' %row)
    result = cursor.fetchall()
    # print result
    # col=[]
    # for line in result:
    #     col.append(line[0])
    # print col  
    # print result[0]
    series_id,model_id,prov_id,reg_year,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,old_eval_price=[],[],[],[],[],[],[],[],[],[],[],[]

    for line in result:
        series_id.append(line[0])
        model_id.append(line[1])
        prov_id.append(line[2]) 
        reg_year.append(line[3])       
        city_id.append(line[4])
        reg_date.append(line[5])
        mile_age.append(line[6])
        post_time.append(line[7])
        liter.append(line[8])
        model_year.append(line[9])
        model_price.append(line[10])
        old_eval_price.append(line[-1])
        
    df = pd.DataFrame([series_id, model_id,prov_id,reg_year,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,old_eval_price]).T
    df = df.rename(columns={0:"series_id",1:"model_id" ,2:"prov_id",3:"reg_year",4:"city_id",5:"reg_date",6:"mile_age",7:"post_time",8:"liter",9:"model_year",10:"model_price",11:"old_eval_price"})
    # print df
    df.to_csv("C:/danhua/predict_price_series1/predict_price_series_1_%s.csv"%j,index=False)






# sql= "select model_id,prov_id,series_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,old_eval_price from da_car_series_i "
# for i in series_id:


# cursor.execute(sql)
# # alldata = cursor.fetchall()
# # for row in alldata:   
# #       print('%s\t%s' %row)
# result = cursor.fetchall()

# # print result
# # col=[]
# # for line in result:
# #     col.append(line[0])
# # print col  
# # print result[0]
# model_id,prov_id,series_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,old_eval_price=[],[],[],[],[],[],[],[],[],[]

# for line in result:
#     model_id.append(line[0])
#     prov_id.append(line[1])
#     city_id.append(line[2])
#     reg_date.append(line[3])
#     mile_age.append(line[4])
#     post_time.append(line[5])
#     liter.append(line[6])
#     model_year.append(line[7])
#     model_price.append(line[8])
#     old_eval_price.append(line[9])
#     series_id.append(line[10])
# df = pd.DataFrame([model_id,prov_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,old_eval_price ]).T
# df = df.rename(columns={0:"series_id",1:"model_id" ,2:"prov_id",3:"city_id",4:"reg_date",5:"mile_age",6:"post_time",7:"liter",8:"model_year",9:"model_price",10:"old_eval_price"})
# # print df
# df.to_csv("C:/danhua/predict_price_series/predict_price_series_1.csv",index=False)



# try:
#     conn = pymysql.connect(host='139.129.97.251',user='majian',passwd='MjAn@9832#',db='majian')
# except Exception, e:
#     print e
#     sys.exit()
# cursor = conn.cursor()
# sql= "select * from da_car_series_i"
# cursor.execute(sql)
# alldata = cursor.fetchall()
# print alldata['model_id']
