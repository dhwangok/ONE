# -*- coding: UTF-8 -*-
# __author__ = 'lz'

import numpy as np
import pandas as pd
import cPickle as pickle
import time
import datetime
# import pymysql


# class data():
#     def __init__(self):
#         pass
#
#
#     def read_data(self,path):
#         f = open(path)
#         data = ''
#         reg_date_new = ''
#         post_time_new = ''
#         instance = ''
#         line = f.readline().strip()
#         line = f.readline().strip()
#         while line:
#             dt = line.split(',')
#             model_id,prov_id,city_id,reg_date,report_price,mile_age,post_time,liter,model_year,model_price,che300_price = self.splitline(dt)
#             year,month,day = reg_date.split('-')
#             year1,month1,day1 = post_time.split('-')
#             reg_date_new=str(year)+'-'+str(month)+'-'+str(day)
#             post_time_new=str(year1)+'-'+str(month1)+'-'+str(day1)
#             instance+=model_id+'\t'
#             instance+=prov_id+'\t'
#             instance+=city_id+'\t'
#             instance+=reg_date_new+'\t'
#             instance+=report_price+'\t'
#             instance+=mile_age+'\t'
#             instance+=post_time_new+'\t'
#             instance+=liter+'\t'
#             instance+=model_year+'\t'
#             instance+=model_price+'\t'
#             instance+=che300_price+'\n'
#             line = f.readline().strip()
#         open('errorReport_self.csv','wb').write(instance)


# findindex的用法是返回featureid在feature_set中的索引位置，以数字返回
def findIndex(feature_id, feature_set):
    index = -1
    for _id in feature_set:
        index += 1
        if _id == feature_id:
            return index


def oneHotEncoding(instance, feature_set, feature_id):
    feature_index = findIndex(feature_id, feature_set)
    for i in range(len(feature_set)):
        if i == feature_index:
            instance += '1' + ','
        else:
            instance += '0' + ','
    return instance

# 从数据表中将model_id,car_source,prov_set,city_set去重

def getModelsort(feature_id):
    data = pd.read_csv("C:/danhua/carset_ex_1/SVR/model_name.csv")
    result = data.sort_values(['model_price','model_year'], ascending=False)
    # print result
    id_sort = list(result['model_id'])
    # print id_sort
    number = id_sort.index(int(feature_id))
    # print feature_id, number
    return str(number)
    # return inttobinary(number)

def getProvsort(feature_id):
    data=pd.read_csv("C:/danhua/carset_ex_1/SVR/prov.csv")
    prov_sort=list(data['id'])
    # print prov_sort
    number=prov_sort.index(int(feature_id))
    # print feature_id,number
    return str(number)
    # return inttobinary(number)

def getCitysort(feature_id):  
    city=pd.read_csv("C:/danhua/carset_ex_1/SVR/city_new_prov.csv")
    result=city.sort_values(['new_prov','city_id'], ascending=True)
    city_sort=list(result['city_id'])
    # print city_sort
    number=city_sort.index(int(feature_id))
    # print feature_id,number
    return str(number)
    # return cityinttobinary(number)


def tobinary(feature_set,feature_id):
    feature_list=list(feature_set)
    feature_list.sort()
    number= feature_list.index(feature_id)
    return inttobinary(number)


def inttobinary(number):
    # originalbinary=bin(number).replace("0b","")  #bin() "0b"
    # misslength=6-len(originalbinary)            #6
    originalbinary=str(number)
    misslength=2-len(originalbinary)
    fillbinary="0"*misslength+originalbinary
    finalbinary=''
    for i in fillbinary:
        finalbinary+=i +','
    # finalbinary=[int(x) for x in list(fillbinary)]
    return finalbinary

def citytobinary(feature_set, feature_id):
    feature_list = list(feature_set)
    feature_list.sort()
    number = feature_list.index(feature_id)
    return cityinttobinary(number)


def cityinttobinary(number):
    # originalbinary = bin(number).replace("0b", "")  #bin() "0b"
    # misslength = 9 - len(originalbinary)           #9
    originalbinary=str(number)
    misslength=3-len(originalbinary)
    fillbinary = "0" * misslength + originalbinary
    # print fillbinary
    finalbinary = ''
    for i in fillbinary:
        finalbinary += i + ','
    # print finalbinary
    # finalbinary=[int(x) for x in list(fillbinary)]
    return finalbinary


def getIdSet():
    model_set = set()
    # car_source_set=set()
    prov_set = set()
    city_set = set()
    model_year_set=set()
    # 读取model_id,car_source,prov_set,city_set
    f_model_id = open('C:/danhua/carset_ex_1/xgb/id_Set/model_id_set_1.csv')
    line_model_id = f_model_id.readline().strip()
    while line_model_id:
        d_model_id = line_model_id
        model_set.add(d_model_id)
        line_model_id = f_model_id.readline().strip()
    # 读取car_source
    # f_car_source_item = open('C:/danhua/carset_ex_1/SVR/id_Set/car_source_set.csv')
    # line_car_source = f_car_source_item.readline().strip()
    # while line_car_source:
    #     d_car_source = line_car_source
    #     car_source_set.add(d_car_source)
    #     line_car_source = f_car_source_item.readline().strip()
    #  #prov_set
    f_prov_id = open('C:/danhua/carset_ex_1/xgb/id_Set/prov_id_set_1.csv')
    line_prov_id = f_prov_id.readline().strip()
    while line_prov_id:
        d_prov_id = line_prov_id
        prov_set.add(d_prov_id)
        line_prov_id = f_prov_id.readline().strip()
     # 读取city_set
    f_city_id = open('C:/danhua/carset_ex_1/xgb/id_Set/city_id_set_1.csv')
    line_city_id = f_city_id.readline().strip()
    while line_city_id:
        d_city_id = line_city_id
        city_set.add(d_city_id)
        line_city_id = f_city_id.readline().strip()

    # f_modelyear_id = open('C:/danhua/carset_ex_1/SVR/id_Set/model_year_set_1.csv')
    # line_modelyear_id = f_modelyear_id.readline().strip()
    # while line_modelyear_id:
    #     d_modelyear_id = line_modelyear_id
    #     model_year_set.add(d_modelyear_id)
    #     line_modelyear_id = f_modelyear_id.readline().strip()

    return model_set, prov_set, city_set


# 更换csv格式文件的间隔符
def changeCsv(frompath):
    from_file = open(frompath)
    line = from_file.readline().strip()
    line = from_file.readline().strip()
    instance = ''
    feature = ''
    while line:
        dt = line.split('\t')
        model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, report_price = splitline1(
            dt)
        instance += model_id + '\t'
        instance += prov_id + '\t'
        instance += city_id + '\t'
        instance += reg_date + '\t'
        # instance+=car_source+','
        # instance+=car_status+','
        instance += mile_age + '\t'
        instance += post_time.strip() + '\t'
        instance += liter + '\t'
        instance += model_year + '\t'
        instance += model_price + '\t'

        instance += report_price + '\n'
        line = from_file.readline().strip()

    feature += instance
    open('da_car_series_1.csv', 'wb').write(feature)


# 处理csv格式问题
def dealCsv(df):
    # f = open(path)
    line = df.readline().strip()
    line = df.readline().strip()
    d1 = datetime.datetime(2016, 6, 28)
    feature = ''
    while line:
        instance = ''
        dt = line.split('\t')
        model_id, prov_id, city_id, reg_date,  mile_age, post_time, liter, model_year, model_price, report_price = splitline1(
            dt)
        year, month, day = reg_date.split('-')
        d2 = datetime.datetime(int(year), int(month), int(day))
        reg_date_day = (d1 - d2).days
        year, month, day = post_time.split('-')
        d2 = datetime.datetime(int(year), int(month), int(day))
        post_date_day = (d1 - d2).days
        instance += prov_id + ','
        instance += city_id + ','
        instance += str(reg_date_day) + ','
        instance += mile_age + ','
        instance += str(post_date_day) + ','
        instance += liter + ','
        instance += model_year + ','
        instance += model_price + ','
        instance += report_price + '\n'
        feature += instance
        line = df.readline().strip()
    open('C:/danhua/cheprice_series1.csv', 'wb').write(feature)


def splitLine1(dt):
    series_id=dt[0]
    model_id=dt[1]
    prov_id=dt[2]
    reg_year=dt[3]
    city_id=dt[4]
    reg_date=dt[5]  
    # car_source=dt[5]
    # car_status=dt[6]
    mile_age=dt[6]
    post_time=dt[7][:10] #取时间的日期，不取具体时刻
    liter =dt[8][:-1]   #有些数据有单位L
    model_year=dt[9]
    model_price=dt[10]
    
    price =dt[-1]
    # car_source,car_status,
    return series_id,model_id,prov_id,reg_year,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,price

def splitLine2(dt):
    car_id=dt[0]
    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[6]
    car_status = dt[7]
    mile_age = dt[8]
    post_time = dt[9][:10]  # 取时间的日期，不取具体时刻
    liter = dt[10]  # 有些数据有单位L
    model_year = dt[11]
    model_price = dt[12]

    price = dt[5]

    return car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price


def splitLine(dt):
    
    model_id = dt[1]
    gear_type=dt[2]
    prov_id = dt[3]
    city_id = dt[5]
    reg_date = dt[6]
    mile_age = dt[7]
    post_time = dt[8][:8]      #dt[8][:10]
    liter = dt[9]
    model_year = dt[10]
    model_price = dt[11]

    price = dt[12]

    return model_id, gear_type, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price

def priceInterval(path):
    f = open(path)
    line = f.readline().strip()
    line = f.readline().strip()
    instance ='model_id' + ',' + 'prov_id' + ',' + 'city_id' + ',' + 'reg_date'+ ',' + 'mile_age' + ',' + 'post_time '+ ',' + 'liter' + ',' + 'model_year' + ',' + 'model_price' + ',' + 'price' + ','+'left_price' + ',' + 'right_price '+ '\n'
    instance1 ='model_id' + ',' + 'prov_id' + ',' + 'city_id' + ',' + 'reg_date'+ ',' + 'mile_age' + ',' + 'post_time '+ ',' + 'liter' + ',' + 'model_year' + ',' + 'model_price' + ',' + 'price' + ','+'left_price' + ',' + 'right_price '+ '\n'
    instance3 ='model_id' + ',' + 'prov_id' + ',' + 'city_id' + ',' + 'reg_date'+ ',' + 'mile_age' + ',' + 'post_time '+ ',' + 'liter' + ',' + 'model_year' + ',' + 'model_price' + ',' + 'price' + '\n'
    d1 = datetime.datetime(2016, 6, 28)
    while line:
        dt = line.split(',')
        model_id, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
        year, month, day = reg_date.split('-')
        d2 = datetime.datetime(int(year), int(month), int(day))
        reg_date_day = (d1 - d2).days
        left_price = float(model_price) * \
            (0.9**(1. * reg_date_day / 365)) * 0.8
        right_price = min(float(model_price) * (0.9**(1. * reg_date_day / 365)) * 1.2,float(model_price))
        data1=''
        if float(price) < left_price or float(price) > right_price:
            data1+= str(model_id) + ',' + str(prov_id) + ',' + str(city_id) + ',' + str(reg_date) + ',' + str(mile_age) + ',' + str(post_time) + ',' + str(liter) + ',' + str(model_year) + ',' + str(model_price) + ',' + str(price) + ','+str(left_price) + ',' + str(right_price) + '\n'
            line = f.readline().strip()
            continue
        instance1+=data1
        data = ''
        data += str(model_id) + ',' + str(prov_id) + ',' + str(city_id) + ',' + str(reg_date) + ',' + str(mile_age) + ',' + str(post_time) + ',' + str(liter) + ',' + str(model_year) + ',' + str(model_price) + ',' + str(price) + ','+str(left_price) + ',' + str(right_price) + '\n'
        data3=''
        data3+=str(model_id) + ',' + str(prov_id) + ',' + str(city_id) + ',' + str(reg_date) + ',' + str(mile_age) + ',' + str(post_time) + ',' + str(liter) + ',' + str(model_year) + ',' + str(model_price) + ',' + str(price)+'\n'
        instance3+=data3
        instance += data
        line = f.readline().strip()
    # open('C:/danhua/carset/testPirceinterval.csv', 'wb').write(instance)
    # open('C:/danhua/carset/notin_testPirceinterval.csv', 'wb').write(instance1)
    open('C:/danhua/carset/carset_test.csv', 'wb').write(instance3)

def getDays(reg_date):
    d1 = datetime.datetime(2016, 8, 9)
    year, month, day = reg_date.split('-')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day

def getTDays(reg_date):
    d1 = datetime.datetime(2016, 8, 9)
    year, month, day = reg_date.split('/')
    d2 = datetime.datetime(int(year), int(month), int(day))
    reg_date_day = (d1 - d2).days
    return reg_date_day




def PreprocessData(featureFile, path):
    train_data=pd.read_csv("C:/danhua/carset_ex_1/carset_ex_1_0810/carset_ex_1_gear_drFalse2008.csv")
    test_data=pd.read_csv("C:/danhua/carset_ex_1/carset_ex_1_new/predict_price_series/predict_price_series_1_gear_drop2008.csv")
    now_time=d1 = datetime.datetime(2016, 8, 9)
    train_data['reg_day']=train_data['reg_date'].apply(lambda x:getDays(x))
    test_data['reg_day']=test_data['reg_date'].apply(lambda x:getTDays(x))
    max_reg_day=max(train_data['reg_day'].unique().max(),test_data['reg_day'].unique().max())

    train_data['post_day']=train_data['post_time'].apply(lambda x:getDays(x))
    test_data['post_day']=test_data['post_time'].apply(lambda x:getTDays(x[:9]))
    max_post_day=max(train_data['post_day'].unique().max(),test_data['post_day'].unique().max())


    f = open(path)
    line=f.readline().strip()
    line=f.readline().strip()
    data = []
    feature = ''
    d1 = datetime.datetime(2016, 8, 9)
    model_set, prov_set, city_set = getIdSet()

    f = open(path)

    line = f.readline().strip()
    line = f.readline().strip()
    k = 0
    while line:
        k += 1
        dt = line.split(',')
        model_id, gear_type, prov_id, city_id, reg_date, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
        # model_id,prov_id,  reg_date, mile_age, post_time, liter, model_year, model_price, price=splitline2(dt)

        if model_year=='2008':
            continue
        if model_year=='2009':
            model_year_type=0
        elif model_year=='2010':
            model_year_type=1
        elif model_year=='2011':
            model_year_type=2
        elif model_year=='2012':
            model_year_type=3
        elif model_year=='2013':
            model_year_type=4
        elif model_year=='2014':
            model_year_type=5
        elif model_year=='2015':
            model_year_type=6
        elif model_year=='2016':
            model_year_type=7

        year, month, day = reg_date.split('/')
        d2 = datetime.datetime(int(year), int(month), int(day))
        reg_date_day = (d1 - d2).days

        year, month, day = post_time.split('/')
        d2 = datetime.datetime(int(year), int(month), int(day))
        post_date_day = (d1 - d2).days
        # data.append(dt)
        # if float(price) < float(model_price) * (0.9**(1. * reg_date_day / 365)) * 0.8 or float(price) > min(float(model_price)*(0.9**(1.*reg_date_day/365))*1.2,float(model_price)):
        #     line = f.readline().strip()
        #     continue

        instance = ''
        # one hot encoing model_id
        # instance = oneHotEncoding(instance, model_set, model_id)
        # instance=oneHotEncoding(instance,car_source_set,car_source)
        # instance = oneHotEncoding(instance, prov_set, prov_id)
        # instance = oneHotEncoding(instance, city_set, city_id)
        # instance = tobinary(model_set, model_id) 
        # instance += tobinary(prov_set, prov_id) 

        # instance += citytobinary(city_set, city_id) 
        

        # try:
        #     instance += citytobinary(city_set, city_id)
        # except(ValueError):
        #     print city_id
        # print instance
        instance = getModelsort(model_id)+','
        instance += getProvsort(prov_id)+','
        instance += getCitysort(city_id)+','
        # instance +=tobinary(model_year_set, model_year)
        instance+=gear_type+','
        instance += str(model_year_type) + ','
        instance += model_price + ','
        instance += liter + ','
        # instance+=car_status+','
        # instance += str(reg_date_day) + ','
        # instance += str(post_date_day) + ','
        instance+=str(1.0*reg_date_day/max_reg_day)+','
        instance+=str(1.0*post_date_day/max_post_day)+','
        instance += mile_age + ','



        instance += price + '\n'


        if k == 1:
            print instance
            print len(instance.split(','))
        feature += instance

        line = f.readline().strip()

    open(featureFile, 'wb').write(feature)

def getTestdf(path):
    h = open(path)
    line = h.readline().strip()
    line = h.readline().strip()
    series_id_data, model_data, prov_id_data,reg_year_data, city_data, reg_data,  mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
    ], [], [], [], [], [], [], [], [], [], [], []
    while line:
        dt = line.split(',')
        series_id,model_id,prov_id,reg_year,city_id,reg_date,mile_age,post_time,liter,model_year,model_price,price = splitLine1(
            dt)
        without=['127','129','22018','29841','29842','29844','29846','29848']
        if prov_id != '45':
            series_id_data.append(series_id)
            model_data.append(model_id)
            prov_id_data.append(prov_id)
            city_data.append(city_id)
            reg_data.append(reg_date)
            mile_data.append(mile_age)
            post_data.append(post_time)
            liter_data.append(liter)
            model_year_data.append(model_year)
            model_price_data.append(model_price)
            price_data.append(price)
            reg_year_data.append(reg_year)
        else:
            line = h.readline().strip()
        line = h.readline().strip()
    df1 = pd.DataFrame([series_id_data, model_data,prov_id_data,reg_year_data, city_data, reg_data, mile_data,post_data, liter_data, model_year_data, model_price_data,price_data]).T
    df1 = df1.rename(columns={0: "series_id", 1: "model_id", 2: "prov_id", 3: "reg_year", 4:"city_id", 5:  "reg_date", 6: "mile_age", 7: "post_time",8: "liter", 9:"model_year", 10:"model_price",  11: "old_eval_price"})

    df1.to_csv("C:/danhua/carset_ex_1/xgb/carset_ex_1_exact_rest_noCity_id.csv", index=False)

    # f = open(path)
    # car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data= [
    # ], [], [], [], [], [], [], [], [], [], [], [], []
    # line = f.readline().strip()
    # line = f.readline().strip()
    # while line:
    #     dt = line.split(',')
    #     car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine2(
    #         dt)
    #     without=['301','45','248','308','118','378','85','374','174']
    #     if city_id not in without:
    #         car_id_data.append(car_id)
    #         model_data.append(model_id)
    #         prov_data.append(prov_id)
    #         city_data.append(city_id)
    #         reg_data.append(reg_date)
    #         car_source_data.append(car_source)
    #         car_status_data.append(car_status)
    #         mile_data.append(mile_age)
    #         post_data.append(post_time)
    #         liter_data.append(liter)
    #         model_year_data.append(model_year)
    #         model_price_data.append(model_price)
    #         price_data.append(price)
            
    #     else:
    #         line = f.readline().strip()

    #     line = f.readline().strip()

    # df11 = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
    #                    post_data, liter_data, model_year_data, model_price_data]).T
    # df11 = df11.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
    #                         8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price"})
    # df11.to_csv("C:/danhua/carset_ex_1/xgb/carset_ex_1_exact_noCity_id.csv", index=False)




if __name__ == '__main__':
    #"C:/danhua/carset_ex_1/SVR/test_data_1_noCity_id29_gear.csv"
    raw_data_path ="C:/danhua/carset_ex_1/carset_ex_1_new/predict_price_series/predict_price_series_1_gear_drop2008.csv" #"C:/danhua/carset_ex_1/test_data_1_noCity_id29.csv" #"C:/danhua/carset_ex_1/xgb/carset_ex_1_exact_rest_noCity_id.csv" #      
    featureFile = 'C:/danhua/carset_ex_1/xgb/fea/fea_test_1_xgb_int_gear_new.csv'
    PreprocessData(featureFile,raw_data_path)


    # getTestdf("C:/danhua/carset_ex_1/test_data_1_noCity_id29.csv")

    # getIdSet('da_car_series_1202.csv')
    # changeCsv('da_car_series_1202.csv')
    # dealCsv('../data/da_car_series_1.csv')
