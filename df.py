#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import csv
import sys
import logging
import logging.handlers




def splitLine(dt):
    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source=dt[5]
    car_status=dt[6]
    mile_age = dt[7]
    post_time = dt[8][:10]
    liter = dt[9]
    model_year = dt[10]
    model_price = dt[11]

    price = dt[-1]
    return model_id, prov_id, city_id, reg_date,car_source,car_status, mile_age, post_time, liter, model_year, model_price, price


def splitLine1(dt):
    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    # car_source=dt[6]
    # car_status=dt[7]
    mile_age = dt[5]
    post_time = dt[6][:10]
    liter = dt[7]
    model_year = dt[8]
    model_price = dt[9]

    price = dt[10]
    return model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price


def getModelset(data):
    f = open(data)

    line = f.readline().strip()
    line = f.readline().strip()

    model_year_set=set()
    model_set = set()

    while line:
        dt = line.split(',')
        model_id, prov_id, city_id, reg_date,car_source,car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine(dt)
        model_year_set.add(model_year)
        line = f.readline().strip()
    print model_year_set

def getTestModelset(test_data):
    f = open(test_data)

    line = f.readline().strip()
    line = f.readline().strip()

    model_year_set=set()
    model_set = set()

    while line:
        dt = line.split(',')
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine1(dt)
        model_year_set.add(model_year)
        line = f.readline().strip()
    print model_year_set


def getDf(data):
    f = open(data)
    model_data, prov_data, city_data, reg_data,car_source_data,car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
        ], [], [], [], [], [], [], [], [], [],[],[]
    line = f.readline().strip()
    line = f.readline().strip()
    print line
    while line:
        # dt = line.split('	')
        dt= line.split(',')
        model_id, prov_id, city_id, reg_date,car_source,car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine(dt)
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


    df = pd.DataFrame([model_data, prov_data, city_data, reg_data,car_source_data,car_status_data, mile_data,
                   post_data, liter_data, model_year_data, model_price_data, price_data],column=None).T
    df = df.rename(columns={0: "model_id", 1: "prov_id", 2: "city_id", 3: "reg_date",4:"car_source",5:"car_status",
                        6: "mile_age", 7: "post_time", 8: "liter", 9: "model_year", 10: "model_price", 11: "price"})
    # print df


    # df1=df.query('model_id == ["2000","2001","2002" ,"2003","2004","2005","2006","2007"]')
    # df2=df.query('model_id == ["2008","2009" ,"2010"]')
    # df3=df.query('model_id == ["2011","2012" ,"2013","2014","2015","2016"]')

    # df1 = df[(df['model_year'] == '2000') | (df['model_year'] == '2001') | (df['model_year'] == '2002') | (df['model_year'] == '2003') | (
    #     df['model_year'] == '2004') | (df['model_year'] == '2005') | (df['model_year'] == '2006') | (df['model_year'] == '2007') | (df['model_year'] == '2008')]
    # df2 = df[(df['model_year'] == '2009') | (df['model_year'] == '2010')]
    # df3 = df[(df['model_year'] == '2011') | (df['model_year'] == '2012') | (df['model_year'] == '2013') | (
    #     df['model_year'] == '2014') | (df['model_year'] == '2015') | (df['model_year'] == '2016')]
    # df1 = df[(df['model_year'] == '2003') | (df['model_year'] == '2004') | (df['model_year'] == '2005')]
    # df2=df[ (df['model_year'] == '2006') | (df['model_year'] == '2007') | (df['model_year'] == '2008')]
    # df3 = df[(df['model_year'] == '2010') | (df['model_year'] == '2011')| (df['model_year'] == '2013')| (df['model_year'] == '2015')]
    # df4=df[df['model_year'].isin(['2003'])]

    df1=df[df['model_year'].isin(['2004','2005','2006'])]
    df2=df[df['model_year'].isin(['2013'])]
    # print df2

    # df4=df[(df['model_year']=='2014')|(df['model_year']=='2015')|(df['model_year']=='2016')]
    # df0.to_csv("C:/danhua/da_carset/df/da_car_series_3_1_1.csv",index=False)
    df1.to_csv("C:/danhua/code/df/exact_data_408_1.csv", index=False)
    df2.to_csv("C:/danhua/code/df/exact_data_408_2.csv", index=False)
    # df3.to_csv("C:/danhua/da_carset/df/da_car_series_80_3.csv", index=False)
    # df4.to_csv("C:/danhua/da_carset/df/da_car_series_3_4.csv",index=False)
def gettestDf(test_data):
    f = open(test_data)
    model_data, prov_data, city_data, reg_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
        ], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    print line
    while line:
        dt= line.split(',')
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine1(dt)
        model_data.append(model_id)
        prov_data.append(prov_id)
        city_data.append(city_id)
        reg_data.append(reg_date)
        mile_data.append(mile_age)
        post_data.append(post_time)
        liter_data.append(liter)
        model_year_data.append(model_year)
        model_price_data.append(model_price)
        price_data.append(price)
        line = f.readline().strip()


    df = pd.DataFrame([model_data, prov_data, city_data, reg_data, mile_data,
                   post_data, liter_data, model_year_data, model_price_data, price_data]).T
    df = df.rename(columns={0: "model_id", 1: "prov_id", 2: "city_id", 3: "reg_date",
                        4: "mile_age", 5: "post_time", 6: "liter", 7: "model_year", 8: "model_price", 9: "price"})
    df1=df[df['model_year'].isin(['2004','2005','2006'])]
    df2=df[df['model_year'].isin(['2013'])]

    # df0.to_csv("C:/danhua/da_carset/df/predict_price_series/predict_price_series_3_0.csv",index=False)
    df1.to_csv("C:/danhua/code/df/exact_test_data_408_1.csv",index=False)
    df2.to_csv("C:/danhua/code/df/exact_test_data_408_2.csv",index=False)
    # df3.to_csv("C:/danhua/da_carset/df/predict_price_series/predict_price_series_80_3.csv",index=False)
    # df4.to_csv("C:/danhua/da_carset/df/predict_price_series/predict_price_series_3_4.csv",index=False)

if __name__ =='__main__':
    data = "C:/danhua/code/analysis/exact_data_0.3/exact_data_408.csv"
    test_data = 'C:/danhua/code/analysis/exact_test_data_0.3/exact_test_data_408.csv'
    getModelset(data)
    getTestModelset(test_data)
    getDf(data)
    gettestDf(test_data)
