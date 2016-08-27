#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import csv
import sys
import logging
import logging.handlers
import random
import datetime
import time


def splitLine1(dt):
    series_id = dt[0]
    model_id = dt[1]
    prov_id = dt[2]
    reg_year = dt[3]
    city_id = dt[4]
    reg_date = dt[5]
    mile_age = dt[6]
    post_time = dt[7][:10]
    liter = dt[8]
    model_year = dt[9]
    model_price = dt[10]

    price = dt[11]
    return series_id, model_id, prov_id, city_id, reg_date,  mile_age, post_time, liter, model_year, model_price, price, reg_year


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
    gear=dt[-1]

    price = dt[5]
    return car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price,gear


def getDf(data):
    f = open(data)
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data, reg_year_data = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    while line:
        dt = line.split('	')
        car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price, reg_year = splitLine(
            dt)
        if reg_year != '2008':
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
	        reg_year_data.append(reg_year)
        else:
        	line = f.readline().strip()

        line = f.readline().strip()

    df = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data, reg_year_data]).T
    df = df.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price", 13: "reg_year"})

    df0 = df.groupby([df['prov_id'], df['reg_year']])
    # print df0.size()
    df1 = df0.get_group(('12', '2009'))
    # ln1=int(len(df1)*0.5)
    df1 = df1[:1001]
    df2 = df0.get_group(('12', '2010'))
    df2 = df2[:1001]
    df3 = df0.get_group(('12', '2011'))
    df3 = df3[:1001]
    df4 = df0.get_group(('12', '2012'))
    df4 = df4[:1001]
    df5 = df0.get_group(('12', '2013'))
    df5 = df5[:1001]
    df6 = df0.get_group(('12', '2014'))
    df6 = df6[:1001]
    df7 = df0.get_group(('12', '2015'))
    df7 = df7[:1001]
    df8 = df0.get_group(('12', '2016'))

    f = open(data)
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data, reg_year_data = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    while line:
        dt = line.split('	')
        car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price, reg_year = splitLine(
            dt)
        if prov_id != '12':
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
	        reg_year_data.append(reg_year)
        else:
	    	line = f.readline().strip()

        line = f.readline().strip()

    df11 = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data, reg_year_data]).T
    df11 = df11.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price", 13: "reg_year"})

    df12 = pd.concat([df11, df1, df2, df3, df4, df5, df6, df7, df8])
    df12.to_csv("C:/danhua/carset_ex_1/carset_ex_1_12.csv", index=False)

    # df1 = df[df["prov_id"].isin(['12'])]
    # print len(df1)
    # df1.to_csv("C:/danhua/carset_ex_1/analysis/prov_id_12.csv", index=False)


def getformat(ori_data):
	f = open(ori_data)
	line = f.readline().strip()
	line = f.readline().strip()
	# instance=''
	instance = 'model_id' + ',' + 'prov_id' + ',' + 'city_id' + ',' + 'reg_date' + ',' + 'car_source' + ',' + 'car_status' + \
	    ',' + 'mile_age' + ',' + 'post_time ' + ',' + 'liter' + ',' + \
	        'model_year' + ',' + 'model_price' + ',' + 'price' + '\n'
	while line:
		dt = line.split('	')
		car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
		data = ''
		data += str(model_id) + ',' + str(prov_id) + ',' + str(city_id) + ',' + str(reg_date) + ',' + str(car_source) + ',' + str(car_status) + \
		            ',' + str(mile_age) + ',' + str(post_time) + ',' + str(liter) + ',' + \
		                      str(model_year) + ',' + str(model_price) + \
		                          ',' + str(price) + '\n'
		instance += data
		line = f.readline().strip()
	open('C:/danhua/weka/carset_1.csv', 'wb').write(instance)


def getTestdata(test_data):
	h = open(test_data)
	line = h.readline().strip()
	line = h.readline().strip()
	series_id_data, model_data, prov_data, city_data, reg_data,  mile_data, post_data, liter_data, model_year_data, model_price_data, price_data, reg_year_data = [
	], [], [], [], [], [], [], [], [], [], [], []
	while line:
		dt = line.split(',')
		series_id, model_id, prov_id, city_id, reg_date,  mile_age, post_time, liter, model_year, model_price, price, reg_year = splitLine1(
			dt)
		if reg_year != '2008':
			series_id_data.append(series_id)
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
			reg_year_data.append(reg_year)
		else:
			line = h.readline().strip()
		line = h.readline().strip()
	df1 = pd.DataFrame([series_id_data, model_data, prov_data,reg_year_data, city_data, reg_data, mile_data,post_data, liter_data, model_year_data, model_price_data,price_data]).T
	df1 = df1.rename(columns={0: "series_id", 1: "model_id", 2: "prov_id", 3: "reg_year", 4:"city_id", 5:  "reg_date", 6: "mile_age", 7: "post_time",8: "liter", 9:"model_year", 10:"model_price",  11: "old_eval_price"})

	df1.to_csv("C:/danhua/carset_ex_1/test_data_1.csv", index=False)


def getData(ori_data):
    f=open(ori_data)
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data,gear_type=[], [], [], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    c_s=['273','ganji','hx2car','iautos','zg2sc','kuche','taoche','taotaocar','xcar','xici','cn2che','baixing','chelaike','carxoo','58','souche','sohu','ygche','jiajiahaoche']  #加了'ygche','jiajiahaoche'
    while line:
        # print line
        # dt = line.split('	')
        dt = line.split(',')
        car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price,gear = splitLine(
            dt)
        if car_source not in c_s and len(car_id)<=9:
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
            gear_type.append(gear)
            line = f.readline().strip()
        else:
            line = f.readline().strip()

    df = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data]).T
    df = df.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price",13:"gear_type"})
    df.to_csv('D:/data/car_set_328/carset_ex_328_original.csv', index=False)
    # df1 = df[df["model_id"].isin(['28409'])]
    # print len(df1)
    # df1.to_csv("C:/danhua/carset_ex_1/analysis2/model_id_28409.csv", index=False)

    # df = pd.DataFrame([ model_data, prov_data, reg_data, price_data,  mile_data,
    #                    post_data, liter_data, model_year_data, model_price_data]).T
    # df = df.rename(columns={ 0: "model_id", 1: "prov_id",  2: "reg_date", 3: "report_price",4: "mile_age", 5: "post_time", 6: "liter", 7: "model_year", 8: "model_price"})
    # df1 = df[df["model_id"].isin(['1'])]
    # df1.to_csv("C:/danhua/prov/cluster_train_1_model_id1.csv", index=False)
    











if __name__ == '__main__':
    start = time.clock()

    data ="D:/data/carset_ex_0818_2/carset_ex_gear/carset_ex_328_gear.csv" #"C:/danhua/carset_ex_1/carset_ex_1_new/carset_ex_1_gear_drFalse2008.csv"  # "C:/danhua/carset_ex_77/carset_ex_77.csv"
    ori_data="C:/danhua/carset_ex_1/carset_ex_1_0810/carset_ex_1.csv" #"C:/danhua/carset/carset_1.csv"  #"C:/danhua/da_carset/da_car_series_3.csv"
    test_data="C:/danhua/predict_price_series1/predict_price_series_1_1.csv"
    # getDf(data)
    # getformat(ori_data)
    # getTestdata(test_data)
    getData(data)


    print 'done',time.clock()-start
