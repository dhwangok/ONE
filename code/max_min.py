#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import csv
import sys
import logging
import logging.handlers

data="C:/danhua/da_carset/da_car_series_283.csv"
# path="C:/danhua/da_carset/df/da_car_series_2.csv"

def splitLine(dt):
	model_id=dt[1]
	prov_id=dt[2]
	city_id=dt[3]
	reg_date=dt[4]
	# car_source=dt[6]
	# car_status=dt[7]
	mile_age=dt[8]
	post_time=dt[9][:10]
	liter =dt[10]
	model_year=dt[11]
	model_price=dt[12]
	score=dt[-1]
	
	price =dt[5]
	return model_id,prov_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,price

sys.stdout = open('C:/danhua/analysise/log/my4.log', 'a+')
logging.basicConfig(level=logging.INFO,
					format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
					datefmt='%a, %d %b %Y %H:%M:%S',
					filename='my4.log',
					filemode='a+',stream=sys.stdout)


logging.info('This is info message')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


f=open(data)

line=f.readline().strip()
line=f.readline().strip() 

model_set=set()

while line:
	dt=line.split('	')
	model_id,prov_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,price=splitLine(dt)
	model_set.add(model_id)
	line=f.readline().strip() 
print model_set

for i in model_set:
	f=open(data)
	model_data,prov_data,city_data,reg_data,mile_data,post_data,liter_data,model_year_data,model_price_data,price_data=[],[],[],[],[],[],[],[],[],[]
	line=f.readline().strip()
	line=f.readline().strip() 
	while line:
		dt=line.split('	')
		model_id,prov_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,price=splitLine(dt)
		if i==model_id:	
			model_data.append(model_id)
		# prov_data.append(prov_id)
		# city_data.append(city_id)
		# reg_data.append(reg_date)
		# mile_data.append(mile_age)
		# post_data.append(post_time)
		# liter_data.append(liter)
			model_year_data.append(model_year)
		# model_price_data.append(model_price)
			price_data.append(price)
		line=f.readline().strip() 
	print i, len(model_data)
	df1=pd.DataFrame([model_data,price_data]).T
	df1= df1.rename(columns={0:"model_id",1:"price"})
	# print df1.describe()
	print df1.max(dtype=None)
	print df1.min()



# df = pd.DataFrame([model_data,prov_data,city_data,reg_data,mile_data,post_data,liter_data,model_year_data,model_price_data,price_data]).T
# df = df.rename(columns={0:"model_id",1:"prov_id",2:"city_id",3:"reg_date",4:"mile_age",5:"post_time",6:"liter",7:"model_year",8:"model_price",9:"price"})
# # print df
# df.to_csv(path,index=False)



# for i in model_set:
# 	di=data["model_id"=i]