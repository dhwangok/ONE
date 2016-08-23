#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'
import pandas as pd
import numpy as np
import pymysql
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import datetime
import os
import sys
import time
import zipfile

def splitLine(dt):
    car_id = dt[0]
    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[6]
    car_status = dt[7]
    mile_age = dt[8]
    post_time = dt[9][:10]
    liter = dt[10]
    model_year = dt[11]
    model_price = dt[12]
    reg_year = dt[4][:4]

    price = dt[5]
    return car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price, reg_year



def model_name(i):
    conn = pymysql.connect(host="192.168.0.225", user="majian",
                           passwd="MjAn@9832#", db="myche", charset="utf8")
    cursor = conn.cursor()
    sql = "SELECT id,NAME,year,price FROM `mt_model` WHERE sid=%s " % i
    cursor.execute(sql)
    result = cursor.fetchall()
    # print result

    model_id, model_name, model_year, model_price = [], [], [], []

    for line in result:
        model_id.append(line[0])
        model_name.append(line[1])
        model_year.append(line[2])
        model_price.append(line[3])

        df = pd.DataFrame([model_id, model_name, model_year, model_price]).T
        df = df.rename(
            columns={0: "model_id", 1: "model_name", 2: "model_year", 3: "model_price"})
        # print df
        df.to_csv("C:/danhua/carset_ex_77/model_name.csv", index=False)


def gear(i):
    conn = pymysql.connect(host="192.168.0.225", user="majian",
                           passwd="MjAn@9832#", db="myche", charset="utf8")
    cursor = conn.cursor()
    sql = "SELECT id,gear_type FROM `mt_model` WHERE sid=%s" % i
    cursor.execute(sql)
    result = cursor.fetchall()
    # print result

    model_id, gear_type = [], []

    for line in result:
    	if line[1] !=-1: 
            gear = line[1] - 1
        else: 
            gear=line[1]   
        model_id.append(line[0])
        gear_type.append(gear)
        


        df = pd.DataFrame([model_id, gear_type]).T
        df = df.rename(columns={0: "model_id", 1: "gear_type"})
        # print df
        df.to_csv("D:/data/gear/id_gear_%s.csv"% i, index=False) 


def mergeDf(i):
    data = pd.read_csv("D:/data/carset_ex_0818_2/carset_ex_drop/carset_ex_%s_drop.csv"%i)  #"D:/data/carset_ex_drop/carset_ex_%s_drop.csv"% i
    if data.shape[0] !=0:

        id_gear = pd.read_csv("D:/data/gear/id_gear_%s.csv" %i) 

        train = pd.merge(data, id_gear, how='left', on='model_id')
        
        train.to_csv("D:/data/carset_ex_0818_2/carset_ex_gear/carset_ex_%s_gear.csv"%i,index=False)  #

        ##报价高于指导价是可能的，所以不需要剔除
        # train=pd.read_csv("D:/data/carset_ex_gear/carset_ex_%s_drop.csv"% i,parse_dates=['reg_date'])
        # train['reg_year'] = train['reg_date'].dt.year
        # df1 = train[(train['report_price'] >= train['model_price'])]
        # df1.to_csv("C:/danhua/carset_ex_77/false_data.csv",index=False)
        # df = train.drop(df1.index)
        # df.drop('reg_year', axis=1, inplace=True)
        # df.to_csv("D:/data/carset_ex_drFalse/carset_ex_%s_drFalse.csv"%i,
        #           index=False) 

        conn = pymysql.connect(host="192.168.0.225", user="majian",
                               passwd="MjAn@9832#", db="myche", charset="utf8")
        cursor = conn.cursor()
        sql = "SELECT m.sid AS series_Id, m.id AS model_id, c.prov_id, c.register_year AS reg_year, (SELECT MAX(city_id) FROM mt_city WHERE prov_id=c.prov_id) AS city_id,CONCAT(c.register_year, '-8-1') AS reg_date, 1.8 * (2016 - c.register_year) + 0.1 AS mile_age, eval_time AS post_time, m.liter, m.year AS model_year, m.price AS model_price, c.eval_price AS old_eval_price FROM myche.inf_calculate c INNER JOIN myche.mt_model m ON c.model_id = m.id WHERE m.sid = %s " % i
        cursor.execute(sql)
        result = cursor.fetchall()

        series_id_data, model_data, prov_data, city_data, reg_data,  mile_data, post_data, liter_data, model_year_data, model_price_data, price_data, reg_year_data = [
        ], [], [], [], [], [], [], [], [], [], [], []

        for line in result:
            post_time=str(line[7])
            series_id_data.append(line[0])
            model_data.append(line[1])
            prov_data.append(line[2])
            reg_year_data.append(line[3])
            city_data.append(line[4])
            reg_data.append(line[5])
            mile_data.append(line[6])
            post_data.append(post_time.split(' ')[0])
            liter_data.append(line[8])
            model_year_data.append(line[9])
            model_price_data.append(line[10])
            price_data.append(line[11])
            

        df2 = pd.DataFrame([series_id_data, model_data, prov_data,reg_year_data, city_data, reg_data, mile_data,post_data, liter_data, model_year_data, model_price_data,price_data]).T
        df2 = df2.rename(columns={0: "series_id", 1: "model_id", 2: "prov_id", 3: "reg_year", 4:"city_id", 5:  "reg_date", 6: "mile_age", 7: "post_time",8: "liter", 9:"model_year", 10:"model_price",  11: "old_eval_price"})
        df2.to_csv("D:/data/carset_ex_0818_2/predict_price_series/predict_price_series_%s.csv"%i, index=False)  #

        test_data=pd.read_csv("D:/data/predict_price_series/predict_price_series_%s.csv"%i,parse_dates=['reg_date'])
        test = pd.merge(test_data, id_gear, how='left', on='model_id')
        gear_type = test['gear_type']
        test.drop(labels=['gear_type'], axis=1, inplace=True)
        test.insert(2, 'gear_type', gear_type)
        test.to_csv("D:/data/carset_ex_0818_2/predict_price_series_gear/predict_price_series_%s_gear.csv"%i, index=False)   #

def getData(path,i):
    f = open(path) 
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
    ], [], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()

    c_s = ['273', 'ganji', 'hx2car', 'iautos', 'zg2sc', 'kuche', 'taoche', 'taotaocar',
           'xcar', 'xici', 'cn2che', 'baixing', 'chelaike', 'carxoo', '58', 'souche', 'sohu','ygche','jiajiahaoche']
    while line:
        dt = line.split('	')
        # dt = line.split(',')
        car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price, reg_year = splitLine(
            dt)
        if car_source not in c_s:
            car_id_data.append(car_id)
            model_data.append(model_id)
            prov_data.append(prov_id)
            city_data.append(city_id)
            reg_data.append(reg_date)
            car_source_data.append(car_source)
            car_status_data.append(car_status)
            mile_data.append(mile_age)
            post_data.append(post_time)  
            liter_data.append(liter)
            model_year_data.append(model_year)
            model_price_data.append(model_price)
            price_data.append(price)
            line = f.readline().strip()
        else:
            line = f.readline().strip()

    df = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data]).T
    df = df.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price"})
    df.to_csv("D:/data/carset_ex_0818_2/carset_ex_drop/carset_ex_%s_drop.csv"% i, index=False)  #

def getModel1(i):
    conn = pymysql.connect(host="192.168.0.225", user="majian",
                           passwd="MjAn@9832#", db="test_db_1", charset="utf8")
    cursor = conn.cursor()  #REPLACE(REPLACE(c.liter,'T',''),'L','')
    sql = "SELECT c.id,c.model_id,c.prov, c.city, c.reg_date,c.price,c.car_source,c.car_status,c.mile_age,c.post_time,m.liter,m.year,c.model_price,m.gear_type FROM `test_db_1`.inf_car c INNER JOIN myche.mt_model m ON c.model_id = m.id WHERE series_id=%s" % i
    cursor.execute(sql)
    result = cursor.fetchall()

    c_s = ['273', 'ganji', 'hx2car', 'iautos', 'zg2sc', 'kuche', 'taoche', 'taotaocar',
           'xcar', 'xici', 'cn2che', 'baixing', 'chelaike', 'carxoo', '58', 'souche', 'sohu','ygche','jiajiahaoche']
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data,model_year_data, model_price_data, price_data,gear_type = [
    ], [], [], [], [], [], [], [], [], [], [], [],[],[]
    for line in result:
        if line[6] not in c_s:
            post_time=str(line[9])
            gear=line[13]-1
            car_id_data.append(line[0])
            model_data.append(line[1])
            prov_data.append(line[2])
            city_data.append(line[3])
            reg_data.append(line[4])
            car_source_data.append(line[6])
            car_status_data.append(line[7])
            mile_data.append(line[8])
            post_data.append(post_time.split(' ')[0])
            liter_data.append(line[10])
            model_year_data.append(line[11])
            model_price_data.append(line[12])
            gear_type.append(gear)
            price_data.append(line[5])


    df = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data,gear_type]).T
    df = df.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price",13:"gear_type"})
    df.to_csv("C:/danhua/carset_ex_1/carset_ex_1_0810/160817/carset_ex_%s_original.csv"% i, index=False) 






if __name__ == '__main__':

    i =  230 
    while i < 231:  
        try:
            # getModel1(i)
            # i+=1
            path='D:/data/carset_ex_0818_2/carset_ex/carset_ex_%s.csv'%i    #"D:/data/carset_ex/carset_ex_%s.csv"% i   
            # getData(path,i)
            print i
            f=pd.read_csv(path)
            if f.shape[0] !=0:
                getData(path,i)
                # model_name(i)
                gear(i)
                mergeDf(i)
                i+=1
            else:	
                i += 1
        except(IOError):
            i+=1
