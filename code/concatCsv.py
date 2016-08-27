#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import cPickle as pickle
from matplotlib.backends.backend_pdf import PdfPages
import pymysql
import os 

#把一个文件夹的csv合并到一个csv里面
# listfile=os.listdir('D:/data/carset_ex_0818_2/trade_price_series_discount_1/')
# # print listfile
# for i in listfile:
# 	# print i
# 	csv=pd.read_csv("D:/data/carset_ex_0818_2/trade_price_series_discount_1/"+i)
# 	csv.to_csv("D:/data/carset_ex_0818_2/new_one2.csv",index=False,mode="a+",header=None)

df=pd.read_csv("D:/data/carset_ex_0818_2/new_one2.csv")
df1=df['trade_price']
# print df1
df2=df['y_discount']
df['error']=df['y_discount']-df['trade_price']
df3=df['error']
# print df['error']

print "trade_price mean",df1.mean()
print "error mean",df3.mean()
print "error var",df3.var()



