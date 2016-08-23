#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import time


def splitLine(dt):
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

    price = dt[5]
    return model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price


def firstDf(data1):
    f = open(data1)
    model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    while line:
        dt = line.split('	')
        # dt= line.split(',')
        model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
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

    df = pd.DataFrame([model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data, price_data]).T

    # df = df.rename(columns={0: "model_id", 1: "prov_id", 2: "city_id", 3: "reg_date", 4: "car_source", 5: "car_status",
    #                         6: "mile_age", 7: "post_time", 8: "liter", 9: "model_year", 10: "model_price", 11: "price"})
    # df.to_csv("C:/danhua/da_carset/da_car_series_1.csv")
    # df = pd.read_csv('C:/danhua/da_carset/df2/da_car_series_1.csv')
    return df


def getDf(data):
    f = open(data)
    model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
    ], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    while line:
        dt = line.split('	')
        # dt= line.split(',')
        model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
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

    df = pd.DataFrame([model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data, price_data]).T

    df = df.rename(columns={0: "model_id", 1: "prov_id", 2: "city_id", 3: "reg_date", 4: "car_source", 5: "car_status",
                            6: "mile_age", 7: "post_time", 8: "liter", 9: "model_year", 10: "model_price", 11: "price"})
    return df
    # df.to_csv('C:/danhua/da_carset/df1/da_car_brand_1.csv',index=False)

    # df=pd.concat([df1,df])


if __name__ == '__main__':

    start = time.clock()

    i = 2
    data1 = "C:/danhua/da_carset/da_car_series_1.csv"
    df=firstDf(data1)
    df = df.rename(columns={0: "model_id", 1: "prov_id", 2: "city_id", 3: "reg_date", 4: "car_source",
                            5: "car_status", 6: "mile_age", 7: "post_time", 8: "liter", 9: "model_year", 10: "model_price", 11: "price"})
    while i < 30:
        try:
            data = "C:/danhua/da_carset/da_car_series_%s.csv" % i
            df = pd.concat([df, getDf(data)])
            i += 1
        except (AttributeError, ValueError, IOError):
            i += 1
            print u"出错啦"
    df.to_csv('C:/danhua/da_carset/df1/da_car_brand_1.csv', index=False)
