# -*- coding:utf8 -*-
# __author__ = 'lz'
import urllib, urllib2, json
import pymysql
import os


def getCsv(i,csv_path):
	url='http://192.168.0.225:8182/getCarSet?SeriesId=%s'%i

	print "downloading with urllib"
	urllib.urlretrieve(url, csv_path)   

	
	

if __name__=='__main__':

	i = 7
	while i<8:

		try:
			csv_path='C:/danhua/20160520/da_carset/da_car_series_%s.csv'%i
			getCsv(i,csv_path)
			i+=1
		except (AttributeError,ValueError,IOError):
			i+=1
  


