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


def getModelsort(feature_id):
    data = pd.read_csv("C:/danhua/carset_ex_1/SVR/model_name.csv")  #"C:/danhua/carset_ex_77/model_name.csv"
    result = data.sort_values(['model_price','model_year'], ascending=False)
    # print result
    id_sort = list(result['model_id'])
    # print id_sort
    number = id_sort.index(int(feature_id))
    # print feature_id, number
    return str(number)
    # return inttobinary(i,number)


def getProvsort(feature_id):
    data=pd.read_csv("C:/danhua/carset_ex_1/SVR/prov.csv")
    prov_sort=list(data['id'])
    # print prov_sort
    number=prov_sort.index(int(feature_id))
    # num=number+1000
    # print num,number
    # print feature_id,number
    # return str(num)
    return str(number)
    # return inttobinary(i,number)

def getCitysort(feature_id):  
    city=pd.read_csv("C:/danhua/carset_ex_1/SVR/city_new_prov.csv")
    result=city.sort_values(['new_prov','city_id'], ascending=True)
    city_sort=list(result['city_id'])
    # print city_sort
    number=city_sort.index(int(feature_id))
    # print feature_id,number
    return str(number)
    # return cityinttobinary(i,number)




def tobinary(i,feature_id,feature_set):
    feature_list = list(feature_set)
    feature_list.sort()
    number = feature_list.index(feature_id)
    # return str(number)
    return inttobinary(i,number)


def inttobinary(i,number):

    # originalbinary = oct(number).replace("0", "")  # bin()  "0b"
    # misslength = 2 - len(originalbinary)  # 6
    originalbinary = str(number)
    misslength = 2 - len(originalbinary)
    fillbinary = "0" * misslength + originalbinary
    
    # finalbinary = ''
    # for i in fillbinary:
    #     finalbinary += i + ','
    # print finalbinary
    finalbinary=[int(x) for x in list(fillbinary)]
    return fillbinary[i]

def citytobinary(i, feature_id, feature_set):
    feature_list = list(feature_set)
    feature_list.sort()
    number = feature_list.index(feature_id)
    # return str(number)
    return cityinttobinary(i,number)


def cityinttobinary(i,number):
    # originalbinary = oct(number).replace("0", "")  # bin() "0b"
    # misslength = 3 - len(originalbinary)  # 9
    originalbinary = str(number)
    misslength = 3 - len(originalbinary)
    fillbinary = "0" * misslength + originalbinary
    finalbinary=[int(x) for x in list(fillbinary)]
    return finalbinary[i]

def intersect(a,b):
    return list(set(a) & set(b))

def getFeatures(test):
    testName=list(test.columns)
    return testName.remove('trade_price')

def getDays(reg_date):
    d1 = datetime.datetime(2016, 8, 18)
    year, month, day = reg_date.split('-')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day

def getTDays(reg_date):
    d1 = datetime.datetime(2016, 8, 18)
    year, month, day = reg_date.split('/')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day

def getType( feature_id, feature_set):
    feature_list = list(feature_set)
    feature_list.sort()
    number = feature_list.index(feature_id)
    return str(number)


def getRegtype(reg_date_day):
    if reg_date_day<=0.5:
        reg_type=0.5
    elif reg_date_day>0.5 and reg_date_day<=1:
        reg_type=1
    elif reg_date_day>1 and reg_date_day<=1.5:
        reg_type=1.5
    elif reg_date_day>1.5 and reg_date_day<=2:
        reg_type=2
    elif reg_date_day>2 and reg_date_day<=2.5:
        reg_type=2.5
    elif reg_date_day>2.5 and reg_date_day<=3:
        reg_type=3
    elif reg_date_day>3 and reg_date_day<=3.5:
        reg_type=3.5
    elif reg_date_day>3.5 and reg_date_day<=4:
        reg_type=4
    elif reg_date_day>4 and reg_date_day<=4.5:
        reg_type=4.5
    elif reg_date_day>4.5 and reg_date_day<=5:
        reg_type=5
    elif reg_date_day>5 and reg_date_day<=5.5:
        reg_type=5.5
    elif reg_date_day>5.5 and reg_date_day<=6:
        reg_type=6
    elif reg_date_day>6 and reg_date_day<=6.5:
        reg_type=6.5
    elif reg_date_day>6.5 and reg_date_day<=7:
        reg_type=7
    elif reg_date_day>7 and reg_date_day<=7.5:
        reg_type=7.5
    elif reg_date_day>7.5 and reg_date_day<=8:
        reg_type=8
    elif reg_date_day>8 and reg_date_day<=8.5:
        reg_type=8.5
    elif reg_date_day>8.5 and reg_date_day<=9:
        reg_type=9
    elif reg_date_day>9 and reg_date_day<=9.5:
        reg_type=9.5 
    elif reg_date_day>9.5 and reg_date_day<=10:
        reg_type=10
    elif reg_date_day>10 and reg_date_day<=10.5:
        reg_type=10.5 
    elif reg_date_day>10.5 and reg_date_day<=11:
        reg_type=11 
    elif reg_date_day>11 and reg_date_day<=11.5:
        reg_type=11.5 
    elif reg_date_day>11.5 and reg_date_day<=12:
        reg_type=12
    elif reg_date_day>12 and reg_date_day<=12.5:
        reg_type=12.5 
    elif reg_date_day>12.5 and reg_date_day<=13:
        reg_type=13
    elif reg_date_day>13 and reg_date_day<=13.5:
        reg_type=13.5 
    elif reg_date_day>13.5 and reg_date_day<=14:
        reg_type=14
    elif reg_date_day>14 and reg_date_day<=14.5:
        reg_type=14.5 
    elif reg_date_day>14.5 and reg_date_day<=15:
        reg_type=15
    elif reg_date_day>15 and reg_date_day<=15.5:
        reg_type=15.5 
    elif reg_date_day>15.5 and reg_date_day<=16:
        reg_type=16
    elif reg_date_day>16 and reg_date_day<=16.5:
        reg_type=16.5 
    elif reg_date_day>16.5 and reg_date_day<=17:
        reg_type=17
    elif reg_date_day>17 and reg_date_day<=17.5:
        reg_type=17.5 
    elif reg_date_day>17.5 and reg_date_day<=18:
        reg_type=18

    return reg_type


def Preprocess(j):
    print 'Read train data...'
    train=pd.read_csv("D:/data/carset_ex_0818_2/carset_ex_gear/carset_ex_%s_gear.csv"%j)


    print 'Read test data...'
    # test=pd.read_csv("C:/danhua/carset_ex_1/carset_ex_1_new/predict_price_series/predict_price_series_1_gear_drop2008.csv")  #"D:/data/predict_price_series_gear/predict_price_series_2_gear.csv" 
    test=pd.read_csv("D:/data/carset_ex_0818_2/trade_price_series_1/trade_price_series_%s.csv"%j)

    test['reg_day']=test['reg_date'].apply(lambda x:getDays(x))  #getTDays(x)
    # max_reg_day=max(train['reg_day'].unique().max(),test['reg_day'].unique().max())

    train['post_day']=train['post_time'].apply(lambda x:getDays(x))
    test['post_day']=test['post_time'].apply(lambda x:getDays(x)) #getTDays(x[:9])
    # print train['post_day']
    max_post_day=max(train['post_day'].unique().max(),test['post_day'].unique().max())

        # dummies_modelId=pd.get_dummies(table['model_id'],prefix='model_id')
        # print dummies_modelId
        # for i in range(1,3):
        #     print i   
        # table['feature_'+str(1)]=table['model_id'].apply(lambda x: getModelsort(x))
        # for j in range(3,5):j-3,
    test['feature_'+str(12)]=test['prov_id'].apply(lambda x: getProvsort(x))
        # for k in range(5,8): k-5,
    test['feature_'+str(13)]=test['city_id'].apply(lambda x: getCitysort(x))

    test['model_year_type']=test['model_year'].apply(lambda x:getType(x,test['model_year'].unique()))


    test['reg_date_day']=test['reg_day'].apply(lambda x:1.0*x/365)  #max_reg_day
    test['post_date_day']=test['post_day'].apply(lambda x:1.0*x/max_post_day)  #365


        # table.loc[table["reg_date_day"]<=0.5,"reg_type"]=0.5
        # table.loc[(table["reg_date_day"]>0.5)&(table["reg_date_day"]<=1),"reg_type"]=1
        # print table["reg_type"]


    test['reg_type']=test['reg_date_day'].apply(lambda x: getRegtype(x))
        

        # print table['reg_day'],table['reg_date_day'],table['reg_type']

        
    test.drop(['model_id','prov_id','city_id','reg_date','post_time','reg_day','post_day','model_year','reg_date_day'],axis=1,inplace=True)
        # table=pd.concat([table,dummies_modelId],axis=1)
        # print table.columns
    # report_price1=test['report_price']
    # test.drop(labels=['report_price'], axis=1,inplace = True)
    # test.insert(14, 'report_price',report_price1)

    
    

    test.drop(['series_id','reg_year'],axis=1,inplace=True)  #'series_Id'
    # print test.columns[0:]
    # print train.head()
    # print test.head()

    features=['mile_age', 'model_year_type', 'feature_12', 'feature_13', 'liter', 'reg_type', 'model_price', 'post_date_day', 'gear_type']

    return test,features

def checkOneByOne(precision=10, X_test=None, y_test=None, model=None):

    errorNum = 0
    y_predict = model.predict(X_test)
    for i in range(0, len(y_predict)):
        # print y_test[i],y_predict[i]
        if abs(y_test[i] - y_predict[i]) > y_test[i] / 100 * precision:
            errorNum += 1
    print
    print 'errorNum:', errorNum,
    print 'allNum:', len(y_predict),
    print 'precision:', (1 - 1. * errorNum / len(y_predict)) * 100
    print

    return (1 - 1. * errorNum / len(y_predict)) * 100



def load_model(model):  
    fr = open(model,'rb')
    Clf= pickle.load(fr)
    fr.close()
    return Clf

def run_test(precision,test,features,target,j):
    print 'testing...'
    model=load_model('D:/data/carset_ex_0818_2/model/model_%s.pkl'%j)  
    x_test=test[features]
    y_test=test[target]
    x_test=np.array(x_test)

    erroNum=0
    y_predict=model.predict(x_test)
    instance = ''
    for i in range(0,len(y_predict)):        
        if abs(y_test[i]-y_predict[i])>y_test[i]/100*precision:
            erroNum+=1
            # print  y[i],y_predict[i]
        
    
    print 'errorNum',erroNum,
    print 'allNum',len(y_predict)
    print 'precision:',(1-1.*erroNum/len(y_predict))*100
    allNum=len(y_predict)
    precision1=(1-1.*erroNum/len(y_predict))*100
    # print y_predict


    #"C:/danhua/carset_ex_1/carset_ex_1_new/predict_price_series/predict_price_series_1_gear_drop2008.csv"  "D:/data/predict_price_series_gear/predict_price_series_2_gear.csv"
    df=pd.read_csv("D:/data/carset_ex_0818_2/trade_price_series_1/trade_price_series_%s.csv"%j)  #"C:/danhua/carset_ex_77/predict_price_series/predict_price_series_77.csv"
    df["y_predict"]=y_predict
    df.to_csv("D:/data/carset_ex_0818_2/trade_price_series_predict_1/trade_price_series_%s.csv"%j,index=False) #carset_ex_77
    return erroNum,allNum,precision1

def getResult(j):
    df=pd.read_csv("D:/data/carset_ex_0818_2/trade_price_series_predict_1/trade_price_series_%s.csv"%j)
    y_result=[]
    discount=getDiscount(j)

    for i in range(len(df)):
        dfline=df.loc[i]
        key=str(dfline['model_id'])+','+str(dfline['prov_id'])+','+str(dfline['reg_year'])
        # print key
        try:
            newgetDiscount=discount[key]
        except(KeyError):
            newgetDiscount=0.95

        # print newgetDiscount
        y_discount=dfline['y_predict']*newgetDiscount
        y_result.append(y_discount)

        # dfline['y_discount']=dfline['y_predict']*newgetDiscount
        # df.loc[i]=dfline
    df['y_discount']=y_result
    df.drop(['y_predict'],axis=1,inplace=True)
    
    df.to_csv('D:/data/carset_ex_0818_2/trade_price_series_discount_1/trade_price_series_%s_discount.csv'%j,index=False)

    
def getDiscount(j):
    conn = pymysql.connect(host="139.129.97.251", user="spider",
                           passwd="spider@123$!", db="statistics", charset="utf8")
    cursor = conn.cursor()    
    
    sql = "SELECT model_id,prov_id,register_year,dealer_buy_price,eval_price FROM myche.`inf_calculate` WHERE series_id=%s"%j
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()

    discount=dict()

    for line in result:
        key=str(line[0])+','+str(line[1])+','+str(line[2])
        discount[key]=1.0*line[3]/line[4]
    
    return discount


if __name__ == '__main__':
    f=open('D:/data/series.txt','r')
    series=[]
    lines = f.readlines()
    for line in lines:
        s_id=line.strip()
        if int(s_id)>230:  # s_id=='100'and 
            series.append(s_id)

    for j in series:
        try:
            print j            
            df=pd.read_csv("D:/data/carset_ex_0818_2/trade_price_series_1/trade_price_series_%s.csv"%j) 
            if len(df)>=1:
                test,features=Preprocess(j)
                run_test(10,test, features, 'trade_price',j)
                getResult(j)
        except(IOError):
            pass


