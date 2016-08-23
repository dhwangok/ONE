# -*- coding: UTF-8 -*-   
# __author__ = 'lz'
import time 
import datetime
import numpy as np
import pandas as pd
import pymysql
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

data=pd.read_csv('C:/danhua/code/data1/mt_model_order_1_1.csv')
print len(data)
data=data.drop_duplicates()
print len(data)
data.to_csv('C:/danhua/code/data/mt_model_order_1.csv')