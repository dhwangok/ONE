#!/usr/bin/env python
# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import pandas as pd
import pymysql
import csv

with open("C:/danhua/predict_price_series/predict_price_series_1_1.csv",'rb') as f:
    reader = csv.reader(f)
    for row in reader:
    	y_predict,model_id,prov_id,reg_year=row[12],row[1],row[2],row[3]
    	print y_predict