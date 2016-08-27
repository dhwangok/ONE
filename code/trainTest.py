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


#得到省份id在排好的prov_sort里的索引
def getProvsort(feature_id):
    data=pd.read_csv("C:/danhua/carset_ex_1/SVR/prov.csv")
    prov_sort=list(data['id'])
    number=prov_sort.index(int(feature_id))
    return str(number)

#得到城市id在排好的city_sort里的索引
def getCitysort(feature_id):  
    city=pd.read_csv("C:/danhua/carset_ex_1/SVR/city_new_prov.csv")
    result=city.sort_values(['new_prov','city_id'], ascending=True)
    city_sort=list(result['city_id'])
    number=city_sort.index(int(feature_id))
    return str(number)


#取两个集合的交集
def intersect(a,b):
    return list(set(a) & set(b))

def getFeatures(train,test):
    trainName=list(train.columns)
    testName=list(test.columns)
    output=intersect(trainName,testName)
    return output

#获得日期距现在多少天
def getDays(reg_date):
    d1 = datetime.datetime(2016, 8, 18)
    year, month, day = reg_date.split('-')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day

#获得日期距现在多少天
def getTDays(reg_date):
    d1 = datetime.datetime(2016, 8, 18)
    year, month, day = reg_date.split('/')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day

#对年款集合先排序，然后取年款id的对应索引
def getType(j, feature_id,feature_set):
    feature_list = list(feature_set)
    feature_list.sort()
    print feature_list
    number = feature_list.index(feature_id)
    return str(number)

#上牌时间，不满半年的记为0.5，不满1年的记为1，以此类推
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

#对数据进行预处理
def Preprocess(j):
    print 'Read train data...'
    train=pd.read_csv("D:/data/carset_ex_0818_2/carset_ex_gear/carset_ex_%s_gear.csv"%j)
    feature_set=train['model_year'].unique()


    print 'Read test data...'
    test=pd.read_csv("D:/data/carset_ex_0818_2/test/predict_price_series_%s.csv"%j)

    #得到训练集和测试集中上牌时间距离现在的天数
    train['reg_day']=train['reg_date'].apply(lambda x:getDays(x))    
    test['reg_day']=test['reg_date'].apply(lambda x:getDays(x))  #getTDays(x)  #有些年月日是以-来间隔，有些是以/间隔


    #训练集和测试集中发布时间改成距离现在最大值
    train['post_day']=train['post_time'].apply(lambda x:getDays(x))
    test['post_day']=test['post_time'].apply(lambda x:getDays(x)) #getTDays(x[:9])  #有些年月日是以-来间隔，有些是以/间隔，并且可能含有具体时间
    max_post_day=max(train['post_day'].unique().max(),test['post_day'].unique().max())

    for table in [train,test]:    
        # table['feature_'+str(1)]=table['model_id'].apply(lambda x: getModelsort(x))  #已经不需要model_id这个特征了
     
        #对省份做处理
        table['feature_'+str(12)]=table['prov_id'].apply(lambda x: getProvsort(x))   
        

        #对城市做处理
        table['feature_'+str(13)]=table['city_id'].apply(lambda x: getCitysort(x))   


        #对年款做处理
        table['model_year_type']=table['model_year'].apply(lambda x:getType(j,x,feature_set))  
        # print table['model_year_type']

    
        #对上牌时间做处理
        table['reg_date_day']=table['reg_day'].apply(lambda x:1.0*x/365)  #max_reg_day    
        table['reg_type']=table['reg_date_day'].apply(lambda x: getRegtype(x))  


        #对发布时间做处理
        table['post_date_day']=table['post_day'].apply(lambda x:1.0*x/max_post_day)  #365 


        
        #去掉不需要的列
        table.drop(['model_id','prov_id','city_id','reg_date','post_time','reg_day','post_day','model_year','reg_date_day'],axis=1,inplace=True)

    #去掉训练集里的'car_id','car_source','car_status'这些列
    train.drop(['car_id','car_source','car_status'],axis=1,inplace=True)

    
    #去掉测试集里的'series_id','reg_year'这些列
    test.drop(['series_id','reg_year'],axis=1,inplace=True)  #'series_Id'

    print train.head()
    print test.head()

    #取features为训练集和测试集的共有特征
    features=getFeatures(train,test)
    print features
    # train.to_csv()
    # test.to_csv()

    return train,test,features

#对于预测结果正确与错误的判断
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

#用训练集训练模型
def training(train,features,target,j):
    #取数据集里的90%用来训练，10%作为验证集
    X_train, X_vali = train_test_split(train, test_size=0.1, random_state=0)  
     
    y_train=X_train[target]
    y_vali=X_vali[target]
    feature_name=X_train[features].columns
    X_train=np.array(X_train[features])
    #显示出训练数据的数目
    print X_train.shape[0]
    X_vali=np.array(X_vali[features])
    y_train=np.array(y_train)
    y_vali=np.array(y_vali)

    print 'training...'


    est = GradientBoostingRegressor(n_estimators=900,
                                        learning_rate=0.005,
                                        max_depth=5,
                                        random_state=1,
                                        loss='huber').fit(X_train, y_train) 

    #存模型
    model_path = open('D:/data/carset_ex_0818_2/model/model_%s.pkl'%j, 'wb') #carset_ex_77
    pickle.dump(est,model_path)
    model_path.close

    vali_result=checkOneByOne(precision=10,X_test=X_vali,y_test=y_vali,model=est)


    # Plot feature importance
    feature_importance = est.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    # print sorted_idx
    # print feature_importance[sorted_idx]
    # print feature_name
    plt.yticks(pos, feature_name[sorted_idx])

    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    # plt.show()

    #存图片
    plt.savefig('D:/data/carset_ex_0818_2/figure/figure_%s.png'%j)



def load_model(model):  
    fr = open(model,'rb')
    Clf= pickle.load(fr)
    fr.close()
    return Clf


#跑测试集
def run_test(precision,test,features,target,j):
    print 'testing...'
    #导入前面生成好的模型
    model=load_model('D:/data/carset_ex_0818_2/model/model_%s.pkl'%j)
    x_test=test[features]
    y_test=test[target]
    x_test=np.array(x_test)

    erroNum=0
    #用模型预测结果
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

    #将预测结果存入csv
    df=pd.read_csv("D:/data/carset_ex_0818_2/trade_price_series_1/trade_price_series_%s.csv"%j)  #"C:/danhua/carset_ex_77/predict_price_series/predict_price_series_77.csv"
    df["y_predict"]=y_predict
    df.to_csv('D:/data/carset_ex_0818_2/trade_price_series_predict_1/predict_price_series_%s.csv'%j,index=False) #carset_ex_77
    return erroNum,allNum,precision1


#对于模型预测的结果乘以相应的比例值
def getResult(j):
    df=pd.read_csv('D:/data/carset_ex_0818_2/trade_price_series_predict_1/predict_price_series_%s.csv'%j)
    y_result=[]
    discount=getDiscount(j)

    for i in range(len(df)):
        dfline=df.loc[i]
        key=str(dfline['model_id'])+','+str(dfline['prov_id'])+','+str(dfline['reg_year'])
        try:
            newgetDiscount=discount[key]
        except(KeyError):
            newgetDiscount=0.95

        y_discount=dfline['y_predict']*newgetDiscount
        y_result.append(y_discount)

    df['y_discount']=y_result
    #去掉'y_predict'这一列
    df.drop(['y_predict'],axis=1,inplace=True)
    
    df.to_csv('D:/data/carset_ex_0818_2/trade_price_series_discount_1/predict_price_series_%s_discount.csv'%j,index=False)


#从数据库里得到一个车系的比例值，把结果写成字典
def getDiscount(j):
    conn = pymysql.connect(host="192.168.0.225", user="majian",
                           passwd="MjAn@9832#", db="majian", charset="utf8")
    cursor = conn.cursor()    
    sql = "SELECT * FROM majian.`inf_eval_discount` WHERE model_id IN (SELECT id FROM myche.`mt_model` WHERE sid=%s)"%j
    cursor.execute(sql)
    result = cursor.fetchall()
    conn.close()

    discount=dict()

    for line in result:
        key=str(line[0])+','+str(line[1])+','+str(line[2])
        discount[key]=line[3]
    
    return discount 



if __name__ == '__main__':
    #从series.txt获得车系
    f=open('D:/data/series.txt','r')
    series=[]
    lines = f.readlines()
    for line in lines:
        s_id=line.strip()
        if s_id=='824': #int(s_id)>1112 and
            series.append(s_id)

    for j in series:
        try:
            print j            
            df=pd.read_csv("D:/data/carset_ex_0818_2/carset_ex_gear/carset_ex_%s_gear.csv"%j) 
            if len(df)>=10:
                train,test,features=Preprocess(j)
                training(train,features,'report_price',j)
                run_test(10,test, features, 'old_eval_price',j)
                getResult(j)
        except(IOError):
            pass




    # j=77
    # while j<230:
    #     try:
    #         print j            
    #         df=pd.read_csv("D:/data/carset_ex_0818_2/carset_ex_gear/carset_ex_%s_gear.csv"%j) 
    #         if len(df)>=10:
    #             train,test,features=Preprocess(j)
    #             training(train,features,'report_price',j)
    #             run_test(10,test, features, 'old_eval_price',j)
    #             getResult(j)
    #             j+=1
    #         else:
    #             j+=1
    #     except(IOError):
    #         j+=1


