#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import cPickle as pickle
from matplotlib.backends.backend_pdf import PdfPages
import pymysql
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#连接数据库取真实交易数据
def getRealdata(i):
    conn = pymysql.connect(host="139.129.97.251", user="spider",
                           passwd="spider@123$!", db="statistics", charset="utf8")
    cursor = conn.cursor()
    sql = "SELECT m.name,m.sid,s.model_id,m.gear_type,s.prov_id,SUBSTRING(s.reg_date,1,4) AS reg_year,s.city_id,s.reg_date,s.mile_age,DATE(NOW()), m.liter,m.year,m.price,s.dealer_buy_price,s.trade_price FROM statistics.`sta_real_data_eval_qa_daily` s  INNER JOIN myche.mt_model m ON s.model_id = m.id WHERE m.sid=%s and eval_date = DATE(NOW()) AND s.car_id NOT IN (SELECT car_id FROM statistics.`inf_real_invalid_car`)" % i
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()

    name, series_id_data, model_data, prov_data, city_data, reg_data,  mile_data, post_data, liter_data, model_year_data, model_price_data, price_data, reg_year_data, gear_type, dealer_buy_price = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for line in result:
        if line[3] != -1:
            gear = line[3] - 1
        else:
            gear = line[3]
        name.append(line[0])
        post_time = str(line[9])
        series_id_data.append(line[1])
        model_data.append(line[2])
        prov_data.append(line[4])
        reg_year_data.append(line[5])
        city_data.append(line[6])
        reg_data.append(line[7])
        mile_data.append(line[8])
        post_data.append(post_time.split(' ')[0])
        liter_data.append(line[10])
        model_year_data.append(line[11])
        model_price_data.append(line[12])
        dealer_buy_price.append(line[13])
        price_data.append(line[14])
        gear_type.append(gear)

    df2 = pd.DataFrame([name, series_id_data, model_data, gear_type, prov_data, reg_year_data, city_data, reg_data,
                        mile_data, post_data, liter_data, model_year_data, model_price_data, dealer_buy_price, price_data]).T
    df2 = df2.rename(columns={0: "name", 1: "series_id", 2: "model_id", 3: "gear_type", 4: "prov_id", 5: "reg_year", 6: "city_id", 7:  "reg_date", 8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price", 13: "dealer_buy_price", 14: "trade_price"})
    df2.to_csv("D:/data/carset_ex_0818_2/trade_price_series_1/trade_price_series_%s.csv" % i, index=False)


if __name__ == '__main__':
    f = open('D:/data/series.txt', 'r')
    series = []
    lines = f.readlines()
    for line in lines:
        s_id = line.strip()
        # if  int(s_id)>1130:
        series.append(s_id)

    for i in series:
        print i
        getRealdata(i)
