# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import time 
import datetime
import numpy as np
import pandas as pd
import pymysql
import csv

def change(ori_path,path,j):
	f=open(ori_path)
	line=f.readline().strip()
	line=f.readline().strip()
	order=''	
	while line:
		content=''
		dt=line.split('>')
		# print dt[0],dt[1]
		leftvalue1,rightvalue1=dt[0],dt[1]
		leftvalue0,rightvalue0=dt[1],dt[0]
		relation1=1
		relation0=0
		content+=str(leftvalue1)+' '+str(relation1)+' '+str(rightvalue1)+'\n'
		content+=str(leftvalue0)+' '+str(relation0)+' '+str(rightvalue0)+'\n'
		order+=content
		line=f.readline().strip()
	open(path,'wb').write(order)

def updateSQL(path):
	conn=pymysql.connect(host="192.168.0.225",user="sa",passwd="sa@1234",db="myche",charset="utf8")
	cursor = conn.cursor()
	g=open(path)
	line=g.readline().strip()
	while line:
		dt=line.split(' ')
		# print dt[0],dt[1],dt[2]
		sql="insert into car_order (leftvalue,relation,rightvalue) values ('%s','%s','%s')" %(dt[0],dt[1],dt[2])
		cursor.execute(sql)
		conn.commit()
		line=g.readline().strip()
	cursor.close()
	conn.close()


if __name__=='__main__':
	j=1
	while j<2531:
		try:
			print j
			ori_path='C:/danhua/code/sid/mt_model_order_%s.csv'%j
			path='C:/danhua/code/order/mt_model_order_%s.csv'%j
			change(ori_path,path,j)
			updateSQL(path)
			j+=1
		except(IOError):
			j+=1