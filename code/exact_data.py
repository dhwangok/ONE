# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import time

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
    gear_type=dt[-1]

    price = dt[5]
    return car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price, reg_year,gear_type

def getDf(data):
    f = open(data)
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data, reg_year_data,gear = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], [],[]
    line = f.readline().strip()
    line = f.readline().strip()
    while line:
        dt = line.split(',')
        car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price, reg_year,gear_type = splitLine(
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
	        gear.append(gear_type)
        else:
        	line = f.readline().strip()

        line = f.readline().strip()

    df = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data, reg_year_data, gear]).T
    df = df.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price", 13: "reg_year",14:"gear_type"})
    return df

def exact_data(path):
	data = getDf(path)	
	df1=data[(data['reg_year']=='2016')]
	df2=data[(data['reg_year']=='2015')]
	df3=data[(data['reg_year']=='2014')]
	df4=data[(data['reg_year']=='2013')]
	df5=data[(data['reg_year']=='2012')]
	df6=data[(data['reg_year']=='2011')]
	df7=data[(data['reg_year']=='2010')]
	df8=data[(data['reg_year']=='2009')]

	sampler2 = np.random.permutation(len(df2))
	sampler21=sampler2[:2000]
	sampler22=sampler2[2000:]
	sampler3 = np.random.permutation(len(df3))
	sampler31=sampler3[:2000]
	sampler32=sampler3[2000:]
	sampler4 = np.random.permutation(len(df4))
	sampler41=sampler4[:2000]
	sampler42=sampler4[2000:]
	sampler5 = np.random.permutation(len(df5))
	sampler51=sampler5[:2000]
	sampler52=sampler5[2000:]
	sampler6 = np.random.permutation(len(df6))
	sampler61=sampler6[:2000]
	sampler62=sampler6[2000:]
	sampler7 = np.random.permutation(len(df7))
	sampler71=sampler7[:2000]
	sampler72=sampler7[2000:]
	sampler8 = np.random.permutation(len(df8))
	sampler81=sampler8[:2000]
	sampler82=sampler8[2000:]


	df12=df2.take(sampler21)
	df13=df3.take(sampler31)
	df14=df4.take(sampler41)
	df15=df5.take(sampler51)
	df16=df6.take(sampler61)
	df17=df7.take(sampler71)
	df18=df8.take(sampler81)
	# print df18

	df = pd.concat([df1, df12, df13, df14, df15, df16, df17, df18]) 
	df.drop('reg_year',axis=1,inplace=True)
	df.to_csv("C:/danhua/carset_ex_1/carset_ex_1_new/carset_ex_1_exact.csv", index=False)

	df22=df2.take(sampler22)
	df23=df3.take(sampler32)
	df24=df4.take(sampler42)
	df25=df5.take(sampler52)
	df26=df6.take(sampler62)
	df27=df7.take(sampler72)
	df28=df8.take(sampler82)

	df0 = pd.concat([df22, df23, df24, df25, df26, df27, df28]) 
	df0.drop('reg_year',axis=1,inplace=True)
	df0.to_csv("C:/danhua/carset_ex_1/carset_ex_1_new/carset_ex_1_exact_rest.csv", index=False)






if __name__=='__main__':
	start = time.clock()
	#'C:/danhua/carset_ex_1/train_data_11.csv'
	path="C:/danhua/carset_ex_1/carset_ex_1_new/carset_ex_1_gear.csv"
	exact_data(path)
	print time.clock()-start