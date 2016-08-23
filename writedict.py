# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import time 
import datetime
import numpy as np
import pandas as pd
import pymysql
import collections
import sys
import time
import math
import re
import Levenshtein
import logging
import logging.handlers
import sys
reload(sys)
sys.setdefaultencoding('utf-8') 



def splitLine(dt):
	car_id=dt[0]
	series_id=dt[1]
	model_id=dt[2]
	prov_id=dt[3]
	city_id=dt[4]
	reg_date=dt[5]
	car_source=dt[6]
	car_status=dt[7]
	mile_age=dt[8]
	post_time=dt[9]
	liter =dt[10]
	model_year=dt[11]
	model_price=dt[12]
	
	price =dt[-2]

	reg_year=dt[-1]
	
	return car_id,series_id,model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_year,model_price,price,reg_year

def splitLine1(dt):

	model_id=dt[0]
	prov_id=dt[1]
	reg_year=dt[2]
	series_id=dt[3]
	city_id=dt[4]
	reg_date=dt[5]
	mile_age=dt[6]
	post_time=dt[7]
	liter =dt[8]
	model_year=dt[9]
	model_price=dt[10]
	
	price =dt[-1]
	
	return model_id,prov_id,reg_year,series_id,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,price



def getName(right_order_set,line,modelDict):
	

	all_inf=dict()
	
	for right_order_model_id in right_order_set:
		for model_id in right_order_model_id:
			dt1 = modelDict[str(model_id)]
			dt1 = dt1.decode("utf-8").split(",")
						
			name1 = dt1[1].split(' ')
			price = float(dt1[-2])
			year = dt1[-1]
			# print dt1
			
			# the name, removed the model_year
			n3 = ''
			value = ''
			for k in range(1, len(name1)):
				n3 += name1[k] + ' '			
			name_1 = n3[:-1]			
			value = name_1 +','+str(price)+','+str(year)			
			all_inf[model_id]=value
			
	dt2 = modelDict[str(line)]	
	dt2 = dt2.decode("utf-8").split(",")	

	name2 = dt2[1].split(' ')
	price2=float(dt2[-2])
	year2=dt2[-1]

	n4=''
	for i in range(1,len(name2)): # 去掉年款
		n4+=name2[i]+' '
	name_4=n4[:-1]
	
	# print name_3,name_4
	right_data_value=dict()
	for m in all_inf.keys():
		name_3=all_inf[m].split(',')[0]
		year1=all_inf[m].split(',')[-1]
		price1=float(all_inf[m].split(',')[1])
		# print price1


		if year1==year2 and name_3!=name_4: #年款相同,字段不同
			x=abs(price1-price2)*2/(price1+price2)
			if x<=0.3:
				isone=math.sqrt(1-(x/0.3)**2)
			else:
				isone=0
		# print name_3,name_4,isone
			# print result_1[1],result_2[1],isone
			# print abs(result_1[7]-result_2[7]),x
		elif year1!=year2 and name_3==name_4: #年款不同，去掉年款字段相同
			isone=1
		else:
			isone=0			
		right_data_value[m]=isone
	# print right_data_value
				
	return right_data_value
	
 
	

def getName2(left_order_set,line,modelDict):
	# f=open(path)
	# row=f.readline().strip()
	# row=f.readline().strip()

	# name1=''	
	
	# year1=''
	# name2=''
	
	# year2=''
	
	# print line
	# while row:
		
	# 	dt=row.split(',')
	# 	model_id,name,sid,price,year=dt[0],dt[1],dt[2],dt[3],dt[-1]

	# 	for left_order_value in left_order_set:
	# 		for i in left_order_value:
	# 			# print model_id,i
	# 			# print type(model_id),type(i),type(line)
	# 			if model_id==str(i):
	# 				name1=name.decode("utf-8").split(' ')
	# 				n3=''
	# 				value=''
	# 				for k in range(1,len(name1)):
	# 					n3+=name1[k]+' '
	# 				name_1=n3[:-1]
	# 				# print name_3
	# 				value=name_1+','+str(price)+','+str(year)
	# 				# all_value.append(value)
	# 				all_inf[i]=value
	# 	if model_id==str(line):
	# 		name2=name.decode("utf-8").split(' ')
	# 		# print name2
	# 		price2=float(price)
	# 		year2=year	

	# 	row=f.readline().strip()
	# print all_inf,'name2',name2 
	# print all_inf.keys()

	all_inf=dict()
	for left_order_model_id in left_order_set:
		for model_id in left_order_model_id:
			dt1 = modelDict[str(model_id)]
			dt1 = dt1.decode("utf-8").split(",")
						
			name1 = dt1[1].split(' ')
			price = float(dt1[-2])
			year = dt1[-1]
			# print dt1
			
			# the name, removed the model_year
			n3 = ''
			value = ''
			for k in range(1, len(name1)):
				n3 += name1[k] + ' '			
			name_1 = n3[:-1]			
			value = name_1 +','+str(price)+','+str(year)			
			all_inf[model_id]=value
			
	dt2 = modelDict[str(line)]	
	dt2 = dt2.decode("utf-8").split(",")	

	name2 = dt2[1].split(' ')
	price2=float(dt2[-2])
	year2=dt2[-1]

	n4=''
	for i in range(1,len(name2)): # 去掉年款
		n4+=name2[i]+' '
	name_4=n4[:-1]

	# p=re.compile( '\d u'款'')
	# name_5=p.group()
	# print name_5
	
	
	# print name_3,name_4
	left_data_value=dict()
	# isone_value=[]
	for m in all_inf.keys():
		name_3=all_inf[m].split(',')[0]
		year1=all_inf[m].split(',')[-1]
		price1=float(all_inf[m].split(',')[1])

		if year1==year2 and name_3!=name_4: #年款相同,字段不同
			x=abs(price1-price2)*2/(price1+price2)
			if x<=0.3:
				isone=math.sqrt(1-(x/0.3)**2)
			else:
				isone=0
			# print m,isone
			# print result_1[1],result_2[1],isone
			# print abs(result_1[7]-result_2[7]),x
		elif year1!=year2 and name_3==name_4: #年款不同，去掉年款字段相同
			isone=1
		else:
			isone=0	
		# print m,isone
		left_data_value[m]=isone
	# print left_data_value
				

	# print result_1[1],result_2[1],isone
	# print name_1,name_2
	return left_data_value

def getFulfillNum(line):
	orderNum=0
	l_fulfill=0
	l_no_ful=0
	r_fulfill=0
	r_no_ful=0
	conn=pymysql.connect(host="192.168.0.225",user="sa",passwd="sa@1234",db="majian",charset="utf8")

	cursor = conn.cursor()	
	right_order="select model_id from train_da_car where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=1)  and prov_id='%s' and reg_year='%s'" %(line[2],line[3],line[-1])
	cursor.execute(right_order)
	right_order=cursor.fetchall()		
		# print right_order
	right_order_set=set()
	for i in right_order:
		right_order_set.add(i)
		# print right_order_set


		# log("start left")
	cursor = conn.cursor()
	right_data="select * from train_da_car where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=1)  and prov_id='%s' and reg_year='%s'" %(line[2],line[3],line[-1])
	cursor.execute(right_data)
	right_data = cursor.fetchall()
	# print len(right_data)
	# log("start data")
	right_data_value=getName(right_order_set,line[2],modelDict)
	# print right_data_value
	# print right_data
	# log("end data")

	for right_line in right_data:
		# print right_line[2]
		isone=right_data_value[right_line[2]]
		# print isone
			# print type(isone)
			# print isone
		if right_line[-2]<line[-2]:
			r_fulfill+=float(isone)
		else:
			r_no_ful+=float(isone)
				
				
		# log("end left")

	cursor = conn.cursor()
	left_order="select model_id from train_da_car where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=0)  and prov_id='%s' and reg_year='%s'" %(line[2],line[3],line[-1])
	cursor.execute(left_order)
	left_order=cursor.fetchall()		
	# print left_order
	left_order_set=set()
	for i in left_order:
		left_order_set.add(i)
	# print left_order_set

	# log("start right")
	cursor = conn.cursor()
	left_data="select * from train_da_car where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=0) and prov_id='%s' and reg_year='%s'" %(line[2],line[3],line[-1])
	cursor.execute(left_data)
	left_data = cursor.fetchall()
	# log("ok")
	# print len(left_data)
	# log("start data")
	left_data_value=getName2(left_order_set,line[2],modelDict)
	# log("end data")

	for left_line in left_data:
		# print left_line[2]
		isone2=left_data_value[left_line[2]]
		# print isone2
		if left_line[-2]>line[-2]:
			l_fulfill+=float(isone2)
		else:
			l_no_ful+=float(isone2)

	# log("end right")

	fulfill=r_fulfill+l_fulfill
	no_ful=r_no_ful+l_no_ful
	orderNum=fulfill+no_ful
	return fulfill,no_ful,orderNum

def log(str11):
	print str(time.clock()) + " " + str11

def connectSQL(j,modelDict):
	log("start fetch data of series.")
	conn=pymysql.connect(host="192.168.0.225",user="sa",passwd="sa@1234",db="majian",charset="utf8")
	cursor = conn.cursor()
	sql="select * from train_da_car where series_id=%s"%j
	cursor.execute(sql)
	result = cursor.fetchall()	
	# print len(result)
	log("done fetch data of series.")
	# data_name='car_id'+','+'model_id'+','+'prov_id'+','+'city_id'+','+'reg_date'+','+'car_source'+','+'car_status'+','+'mile_age'+','+'post_time'+','+'liter'+','+'model_year'+','+'model_price'+','+'price'+'\n'
	# drop_data='car_id'+','+'model_id'+','+'prov_id'+','+'city_id'+','+'reg_date'+','+'car_source'+','+'car_status'+','+'mile_age'+','+'post_time'+','+'liter'+','+'model_year'+','+'model_price'+','+'price'+'\n'
	
	data=''
	drop_data=''
	all_result=''

	for line in result:
		# print line
		spe_result=''
		data1=''
		data2=''
		car_id1,series_id1,model_id1,prov_id1,city_id1,reg_date1,car_source1,car_status1,mile_age1,post_time1,liter1,model_year1,model_price1,price1,reg_year1=splitLine(line)

		cursor = conn.cursor()
		correlation="select price from train_da_car where model_id='%s' and prov_id='%s' and reg_year='%s'" %(line[2],line[3],line[-1])
		cursor.execute(correlation)
		correlation=cursor.fetchall()

		# log("begin not_cor")
		not_cor=0
		if len(correlation)>5:			
			for one_cor in correlation:
				for one in one_cor:
					# print line[-2],one
					if abs(line[-2]-one)>5:
						not_cor+=1
			# print not_cor,len(correlation)
			if float(not_cor)/(len(correlation)-1)>0.5:
				# data2+=str(car_id1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(car_source1)+','+str(car_status1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'
				fulfill=0
				no_ful=1
				orderNum=1
				rate=0
				# continue
			else:
				fulfill,no_ful,orderNum=getFulfillNum(line)
		else:
			fulfill,no_ful,orderNum=getFulfillNum(line)
		# print fulfill,no_ful,orderNum
		

		if orderNum==0:
			rate=0
			data1+=str(car_id1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(car_source1)+','+str(car_status1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'
			# print data1

		try:				
			rate=1.*fulfill/orderNum
			if rate>=0.25 :
				data1+=str(car_id1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(car_source1)+','+str(car_status1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'
			else:
				data2+=str(car_id1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(car_source1)+','+str(car_status1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'			
			# print result
		except(ZeroDivisionError):
			pass	
		
		spe_result+=str(car_id1)+','+str(orderNum) + ',' + str(fulfill )+ ',' + str(no_ful)+ ',' + str(rate)+'\n'
		data+=data1
	 	drop_data+=data2
		
		# print spe_result

			

		all_result+=spe_result
	# print all_result

	open('C:/danhua/code/analysis/fromSQL/0.25_1/train_order_0.25/train_order_%s_0.25.csv'%j,'wb').write(all_result)

	open('C:/danhua/code/analysis/fromSQL/0.25_1/exact_data_0.25/exact_data_%s_0.25.csv'%j,'wb').write(data)

	open('C:/danhua/code/analysis/fromSQL/0.25_1/drop_data_0.25/drop_data_%s_0.25.csv'%j,'wb').write(drop_data)


def testdata(j,modelDict):
	conn=pymysql.connect(host="192.168.0.225",user="majian",passwd="MjAn@9832#",db="majian",charset="utf8")
	cursor = conn.cursor()
	sql="select * from da_car_series_i where series_id=%s"%j
	cursor.execute(sql)
	result = cursor.fetchall()	
	# log("done fetch data of series.")
	# data_name='car_id'+','+'model_id'+','+'prov_id'+','+'city_id'+','+'reg_date'+','+'car_source'+','+'car_status'+','+'mile_age'+','+'post_time'+','+'liter'+','+'model_year'+','+'model_price'+','+'price'+'\n'
	# drop_data='car_id'+','+'model_id'+','+'prov_id'+','+'city_id'+','+'reg_date'+','+'car_source'+','+'car_status'+','+'mile_age'+','+'post_time'+','+'liter'+','+'model_year'+','+'model_price'+','+'price'+'\n'
	
	data=''
	drop_data=''
	

	for line in result:
		# print line
		leftNum=0
		rightNum=0
		orderNum=0
		l_orderNum=0
		r_orderNum=0
		l_fulfill=0
		l_no_ful=0
		r_fulfill=0
		r_no_ful=0
		num=0
		spe_result=''
		data1=''
		data2=''

		cursor = conn.cursor()
		right_order="select model_id from da_car_series_i where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=1)  and prov_id='%s' and reg_year='%s'" %(line[0],line[1],line[2])
		cursor.execute(right_order)
		right_order=cursor.fetchall()		
		# print right_order
		right_order_set=set()
		for i in right_order:
			right_order_set.add(i)
		# print right_order_set


		# log("start left")
		cursor = conn.cursor()
		right_data="select * from da_car_series_i where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=1)  and prov_id='%s' and reg_year='%s'" %(line[0],line[1],line[2])
		cursor.execute(right_data)
		right_data = cursor.fetchall()
		# print len(right_data)
		# log("start data")
		right_data_value=getName(right_order_set,line[0],modelDict)
		# print right_data_value
		# print right_data
		# log("end data")

		for right_line in right_data:
			# print right_line
			isone=right_data_value[right_line[0]]
			# print isone
				# print type(isone)
				# print isone
			if right_line[-1]<line[-1]:
				r_fulfill+=float(isone)
			else:
				r_no_ful+=float(isone)

		# log("end left")

		# log("start right")

		cursor = conn.cursor()
		left_order="select model_id from da_car_series_i where model_id in (select rightvalue from car_order where leftvalue = '%s' and relation=0)  and prov_id='%s' and reg_year='%s'" %(line[0],line[1],line[2])
		cursor.execute(left_order)
		left_order=cursor.fetchall()		
		# print left_order
		left_order_set=set()
		for i in left_order:
			left_order_set.add(i)
		# print left_order_set

		cursor = conn.cursor()
		left_data="select * from da_car_series_i where model_id in (select rightvalue from car_order where leftvalue = '%s'and relation='0') and prov_id='%s' and reg_year='%s'" %(line[0],line[1],line[2])
		cursor.execute(left_data)
		left_data = cursor.fetchall()
		# print len(left_data)
		# log("start data")
		left_data_value=getName2(left_order_set,line[0],modelDict)
		# log("end data")

		for left_line in left_data:
			# print left_line
			isone2=left_data_value[left_line[0]]
			# print isone2
			if left_line[-1]>line[-1]:
				l_fulfill+=float(isone2)
			else:
				l_no_ful+=float(isone2)
	

		# log("end right")

		
		fulfill=r_fulfill+l_fulfill
		no_ful=r_no_ful+l_no_ful
		orderNum=fulfill+no_ful
		# print orderNum

		model_id1,prov_id1,reg_year1,series_id1,city_id1,reg_date1,mile_age1,post_time1,liter1,model_year1,model_price1,price1=splitLine1(line)

		if orderNum==0:
			rate=0
			data1+=str(-1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(-1)+','+str(-1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'
		data+=data1
		# print data1



		try:				
			rate=1.*fulfill/orderNum
			num+=1
			if rate>=0.25 :
				data1+=str(-1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(-1)+','+str(-1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'
			else:
				data2+=str(-1)+','+str(model_id1)+','+str(prov_id1)+','+str(city_id1)+','+str(reg_date1)+','+str(-1)+','+str(-1)+','+str(mile_age1)+','+str(post_time1)+','+str(liter1)+','+str(model_year1)+','+str(model_price1)+','+str(price1)+'\n'
			
			data+=data1
 			drop_data+=data2
		# print result
		except(ZeroDivisionError):
			pass	
		

		# spe_result+=str(car_id1)+','+str(orderNum) + ',' + str(fulfill )+ ',' + str(no_ful)+ ',' + str(rate)+'\n'


			

		# all_result+=spe_result
	# print all_result

	# open('C:/danhua/code/analysis/fromSQL/train_order/train_order_%s_0.3.csv'%j,'wb').write(all_result) 

	open('C:/danhua/code/analysis/fromSQL/0.25_1/exact_testdata_0.25/exact_test_data_%s_0.25.csv'%j,'wb').write(data)

	open('C:/danhua/code/analysis/fromSQL/0.25_1/drop_testdata_0.25/drop_test_data_%s_0.25.csv'%j,'wb').write(drop_data)




def initDict(path):
	f=open(path)
	# remove first line
	row=f.readline().strip()

	all_data=dict()
	row=f.readline().strip()
	while row:
		dt = row.split(',')
		all_data[dt[0]]=row

		# read next line
		row = f.readline().strip()
	# print all_inf
	return all_data


if __name__=='__main__':

	path="C:/danhua/code/analysis/fromSQL/mt_model.csv"	
	modelDict=initDict(path)

	j=57
	while j<58:
		try: 
			start = time.clock()
			print j			
			print 'train'
			connectSQL(j,modelDict)
			print 'test'
			testdata(j,modelDict)
			print 'done',time.clock()-start
			j+=1
		except(AttributeError,ValueError,IOError,KeyError):
			j+=1

