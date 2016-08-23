#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import pandas as pd
import pymysql
import csv


conn=pymysql.connect(host="192.168.0.225",user="majian",passwd="MjAn@9832#",db="majian",charset="utf8")
cursor = conn.cursor()
i=1
while i<2482:
    try:
        with open("C:/danhua/predict_price_series/predict_price_series_1_%s.csv"%i,'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                y_predict,model_id1,prov_id1,reg_year1=row[12],row[1],row[2],row[3]
                sql='update da_car_series_i set predict_price="%s" where model_id = "%s" and prov_id = "%s "and reg_year = "%s"'%(y_predict,model_id1,prov_id1,reg_year1)
                cursor.execute(sql)
                conn.commit()
        i+=1
    except (AttributeError,ValueError,IOError):
        i+=1

cursor.close()
conn.close()