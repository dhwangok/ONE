# -*- coding: UTF-8 -*-
# __author__ = 'wdh'
import time
import datetime
import numpy as np
import pandas as pd
import collections
import time
import sys
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def splitLine(dt):
	series_id = dt[0]
	model_id = dt[1]
	prov_id = dt[2]
	city_id = dt[4]
	reg_date = dt[5]
	mile_age = dt[6]
	post_time = dt[7]
	liter = dt[8]
	model_year = dt[9]
	model_price = dt[10]

	price = float(dt[-1])

	return series_id, model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price


def getOrder(path):
	f = open(path)
	line = f.readline().strip()
	line = f.readline().strip()
	one = set()
	while line:
		dt = line.split('>')
		# print dt[1]
		one.add(dt[1])
		one.add(dt[0])
		line = f.readline().strip()
	# print len(one)
	bdict = dict()
	for i in one:
		f = open(path)
		line = f.readline().strip()
		line = f.readline().strip()
		left_val = set()
		right_val = set()

		while line:
			dt = line.split('>')
			# print dt[0],dt[1]
			if i == dt[1]:
				left_val.add(dt[0])
			if i == dt[0]:
				right_val.add(dt[1])
			line = f.readline().strip()

		# print left_val,right_val

		bdict[i] = [left_val, right_val]
	# print bdict
	return bdict

def getName(train_path):
	f = open(train_path)
	model = set()
	name = ''
	# prov_id=set()
	# reg_year=set()
	line = f.readline().strip()
	line = f.readline().strip()

	while line:
		dt = line.split(',')
		name = dt[1] + ' ' + dt[2] + ' ' + dt[5][:4]
		model.add(name)
		# prov_id.add(dt[2])
		# reg_year.add((dt[4][:4]))
		line = f.readline().strip()
	# print model
	return model

def completeName(train_path, path):
	name = getName(train_path)
	print len(name)
	# bleft_name=set()
	# bright_name=set()
	cdict = dict()
	ddict = dict()
	for i in name:
		# print i
		o_name = i.split(' ')  # a里的model
		# print o_name[2]
		b = getOrder(path)
		for j in b.keys():  # b的key
			left = []
			right = []
			# print j
			if o_name[0] == j:
				left_o_name = b[j][0]
				right_o_name = b[j][1]
				# print j,left_o_name , right_o_name
				for m in left_o_name:
					left_name = m + ' ' + o_name[1] + ' ' + o_name[2]  # 把a里的prov和reg_year补全到b
					left.append(left_name)
				cdict[i] = left
					# print left_name
					# bleft_name.add(left_name)
				# print bleft_name
				for n in right_o_name:
					right_name = n + ' ' + o_name[1] + ' ' + o_name[2]
					# 	bright_name.add(right_name)
					# print right_name
					right.append(right_name)
				ddict[i] = right

	return cdict, ddict




def search(train_path, path):
	cdict, ddict = completeName(train_path, path)
	f = open(train_path)
	line = f.readline().strip()
	line = f.readline().strip()
	line_number = 1
	# line_number_set,orderNum_set,fulfill_set,no_ful_set=[],[],[],[]
	result = ''
	data = 'series_id' + ',' + 'model_id' + ',' + 'prov_id' + ',' + 'city_id' + ',' + 'reg_date' + ',' + \
	    'mile_age' + ',' + 'post_time' + ',' + 'liter' + ',' + \
	        'model_year' + ',' + 'model_price' + ',' + 'price' + '\n'
	drop_data = 'series_id' + ',' + 'model_id' + ',' + 'prov_id' + ',' + 'city_id' + ',' + 'reg_date' + ',' + \
	    'mile_age' + ',' + 'post_time' + ',' + 'liter' + ',' + \
	        'model_year' + ',' + 'model_price' + ',' + 'price' + '\n'
	while line:
		orderNum = 0
		l_orderNum = 0
		r_orderNum = 0
		l_fulfill = 0
		l_no_ful = 0
		r_fulfill = 0
		r_no_ful = 0
		data1 = ''
		data2 = ''
		# print 'line:',line_number
		# dt=line.split('	')
		dt = line.split(',')
		series_id1, model_id1, prov_id1, city_id1, reg_date1, mile_age1, post_time1, liter1, model_year1, model_price1, price1 = splitLine(
		    dt)


		name = dt[1] + ' ' + dt[2] + ' ' + dt[5][:4]
		price1 = float(dt[-1])
		# print cdict

		try:
			for bleft_name in cdict[name]:
				# print bleft_name
				g = open(train_path)
				line = g.readline().strip()
				line = g.readline().strip()
				while line:
					dt = line.split(',')
					series_id, model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine(
					    dt)

					a_name = dt[1] + ' ' + dt[2] + ' ' + dt[5][:4]

					if bleft_name == a_name:
						l_orderNum += 1
						price2 = float(dt[-1])

						if price2 > price1:
							l_fulfill += 1
							# data1+=str(car_id)+','+str(model_id)+','+str(prov_id)+','+str(city_id)+','+str(reg_date)+','+str(car_source)+','+str(car_status)+','+str(mile_age)+','+str(post_time)+','+str(liter)+','+str(model_year)+','+str(model_price)+','+str(price)+'\n'
						else:
							l_no_ful += 1

					line = g.readline().strip()
		except(KeyError, IOError):
			pass

		try:
			for bright_name in ddict[name]:
				h = open(train_path)
				line = h.readline().strip()
				line = h.readline().strip()
				while line:
					dt = line.split(',')
					series_id, model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine(
					    dt)

					a_name = dt[1] + ' ' + dt[2] + ' ' + dt[5][:4]
					if bright_name == a_name:
						r_orderNum += 1
						price2 = float(dt[-1])
						if price1 > price2:
							r_fulfill += 1
							# data1+=str(car_id)+','+str(model_id)+','+str(prov_id)+','+str(city_id)+','+str(reg_date)+','+str(car_source)+','+str(car_status)+','+str(mile_age)+','+str(post_time)+','+str(liter)+','+str(model_year)+','+str(model_price)+','+str(price)+'\n'
						else:
							r_no_ful += 1
					line = h.readline().strip()
		except(KeyError, IOError):
			pass

		orderNum = l_orderNum + r_orderNum
		fulfill = l_fulfill + r_fulfill
		no_ful = l_no_ful + r_no_ful

		if orderNum == 0:
			data1 += str(series_id1) + ',' + str(model_id1) + ',' + str(prov_id1) + ',' + str(city_id1) + ',' + str(reg_date1) + ',' + \
			             str(mile_age1) + ',' + str(post_time1) + ',' + str(liter1) + ',' + \
			                 str(model_year1) + ',' + str(model_price1) + \
			                     ',' + str(price1) + '\n'
		data += data1



		try:

			rate = 1. * fulfill / orderNum
			if rate >= 0.1:
				data1 += str(series_id1) + ',' + str(model_id1) + ',' + str(prov_id1) + ',' + str(city_id1) + ',' + str(reg_date1) + ','+\
				             ',' + str(mile_age1) + ',' + str(post_time1) + ',' + str(liter1) + ',' + \
				                       str(model_year1) + ',' + str(model_price1) + \
				                           ',' + str(price1) + '\n'
			else:
				data2 += str(series_id1) + ',' + str(model_id1) + ',' + str(prov_id1) + ',' + str(city_id1) + ',' + str(reg_date1) + ','  + \
				             ',' + str(mile_age1) + ',' + str(post_time1) + ',' + str(liter1) + ',' + \
				                       str(model_year1) + ',' + str(model_price1) + \
				                           ',' + str(price1) + '\n'
			data += data1
			drop_data += data2

		# print result
		except(ZeroDivisionError):
			pass






		result += str(line_number) + ',' + str(orderNum) + ',' + \
		              str(fulfill) + ',' + str(no_ful) + '\n'
		# print result




		line_number += 1

		line = f.readline().strip()

	# open('C:/danhua/code/analysis/train_order/train_order_%s_0.3.csv'%j,'wb').write(result)

	# open('C:/danhua/code/analysis/exact_data/exact_data_%s_0.3.csv'%j,'wb').write(data)

	# open('C:/danhua/code/analysis/drop_data/drop_data_%s_0.3.csv'%j,'wb').write(drop_data)
	open('C:/danhua/carset_ex_1/xgb/test_order/xgb_test_train_order_1_int.csv',
	     'wb').write(result)

	open('C:/danhua/carset_ex_1/xgb/test_order/xgb_test_exact_data_1_int.csv', 'wb').write(data)

	open('C:/danhua/carset_ex_1/xgb/test_order/xgb_test_drop_data_1_int.csv',
	     'wb').write(drop_data)



if __name__ == '__main__':

	# sys.stdout = open('C:/danhua/code/log/my3.log', 'a+')

	# logging.basicConfig(level=logging.INFO,
	# 					format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
	# 					datefmt='%a, %d %b %Y %H:%M:%S',
	# 					filename='my3.log',
	# 					filemode='a+',stream=sys.stdout)


	# logging.info('This is info message')

	# console = logging.StreamHandler()
	# console.setLevel(logging.INFO)
	# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	# console.setFormatter(formatter)
	# logging.getLogger('').addHandler(console)

	# j=50
	# while j<101:
	# 	try:
	# 		print j

			start = time.clock()
			path = 'C:/danhua/code/sid/mt_model_order_1.csv'
			# "C:/danhua/carset_ex_1/train_data_11.csv"
			train_path = 'C:/danhua/carset_ex_1/xgb/test/test_data_1_Predict_1_xgb_int.csv'
			search(train_path, path)
			print 'done', time.clock() - start
		# 	j+=1
		# except(AttributeError,ValueError,IOError,KeyError):
		# 	j+=1
