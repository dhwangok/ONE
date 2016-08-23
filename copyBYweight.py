#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __author__ = 'wdh'

import numpy as np
import pandas as pd
import csv
import sys
import logging
import logging.handlers
import random
import datetime
import time


def splitLine(dt):
    car_id = dt[0]
    model_id = dt[1]
    prov_id = dt[2]
    city_id = dt[3]
    reg_date = dt[4]
    car_source = dt[6]
    car_status = dt[7]
    mile_age = dt[8]
    post_time = dt[9][:10]
    liter = dt[10]
    model_year = dt[11]
    model_price = dt[-1]

    price = dt[5]
    return car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price


def getDf(data,vali_path,train_path):
    f = open(data)
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
    ], [], [], [], [], [], [], [], [], [], [], [], []
    line = f.readline().strip()
    line = f.readline().strip()
    # print line
    while line:
        dt = line.split('\t')
        car_id, model_id, prov_id, city_id, reg_date, car_source, car_status, mile_age, post_time, liter, model_year, model_price, price = splitLine(
            dt)
        car_id_data.append(car_id)
        model_data.append(model_id)
        prov_data.append(prov_id)
        city_data.append(city_id)
        reg_data.append(reg_date)
        car_source_data.append(car_source)
        car_status_data.append(car_status)
        mile_data.append(mile_age)
        post_data.append(post_time)
        liter_data.append(liter)
        model_year_data.append(model_year)
        model_price_data.append(model_price)
        price_data.append(price)
        line = f.readline().strip()

    df = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data]).T
    df = df.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price"})
    # ln=int(len(df_all)*0.8)
    # df=df_all[:ln]
    # df_test=df_all[ln+1:]
    

    # indexs=[i for i in range(len(df_all))]
    # # print indexs
    # trainIndexs=random.sample(indexs,int(0.8*len(df_all)))
    # testIndexs=[]
    # for i in range(len(df_all)):
    #     if i not in trainIndexs:
    #         testIndexs.append(i)

    # for j in trainIndexs:
    #     df=df_all.loc[j]
    # print df

    # df_test.to_csv("C:/danhua/carset/copy_test_1.csv", index=False)


    df1 = df[df["car_source"].isin(
        ['58', 'dafengche', 'souche'])]  # eval_weight=0.8
    df2 = df[df['car_source'].isin(['273', '51auto', 'baixing', 'carxoo', 'che300_pro', 'cheyipai', 'cn2che', 'ganji',
                                    'hx2car', 'iautos', 'kuche', 'sohu', 'taoche', 'taotaocar', 'ttpai', 'ttpai_c2c', 'xcar', 'xici', 'youche', 'youxin', 'youxinpai', 'youyiche', 'zg2sc'])]  # eval_weight=1
    df3 = df[df['car_source'].isin(['carking', 'chemao', 'ygche'])]  # eval_weight=1.2
    df4 = df[df['car_source'].isin(['che101', 'che168', 'che300','chelaike','cheyitao','jiarenzheng','kx','soucheke'])]  # eval_weight=1.5
    df5 = df[df['car_source'].isin(['ganjihaoche', 'haoche51', 'jiajiahaoche','renrenche'])]  # eval_weight=1.8

    ln1=int(len(df1)*0.8)
    df11=df1[:ln1]
    df21=df1[ln1+1:]

    ln2=int(len(df2)*0.8)
    df12=df2[:ln2]
    df22=df2[ln2+1:]

    ln3=int(len(df3)*0.8)
    df13=df3[:ln3]
    df23=df3[ln3+1:]

    ln4=int(len(df4)*0.8)
    df14=df4[:ln4]
    df24=df4[ln4+1:]

    ln5=int(len(df5)*0.8)
    df15=df5[:ln5]
    df25=df5[ln5+1:]

    # df_vali=pd.concat([df21,df22,df23,df24,df25])
    # df_vali.to_csv("C:/danhua/carset/vali_carset_1.csv", index=False)

    # df_train=pd.concat([df11,df12,df13,df14,df15])
    # df_train.to_csv("C:/danhua/carset/train_carset_1.csv", index=False)

    df_1=df11
    df_1.to_csv("C:/danhua/carset/train_carset_1_1.csv", index=False)

    df_2=pd.concat([df12,df13])
    df_2.to_csv("C:/danhua/carset/train_carset_1_2.csv", index=False)

    df_3=pd.concat([df14,df15])
    df_3.to_csv("C:/danhua/carset/train_carset_1_3.csv", index=False)

    df_6=df21 
    df_6.to_csv("C:/danhua/carset/vali_carset_1_1.csv", index=False)

    df_7=pd.concat([df22,df23])
    df_7.to_csv("C:/danhua/carset/vali_carset_1_2.csv", index=False)

    df_8=pd.concat([df24,df25])
    df_8.to_csv("C:/danhua/carset/vali_carset_1_3.csv", index=False)

    df_11=pd.concat([df_1,df_6])
    df_11.to_csv("C:/danhua/carset/all_carset_1_1.csv", index=False)

    df_12=pd.concat([df_2,df_7])
    df_12.to_csv("C:/danhua/carset/all_carset_1_2.csv", index=False)

    df_13=pd.concat([df_3,df_8])
    df_13.to_csv("C:/danhua/carset/all_carset_1_3.csv", index=False)






    # df6=df11
    # # print df6
    # # df11=df1.append(df6)
    # df31=pd.concat([df11,df6,df6,df6,df6],ignore_index=True)
    
    # df7=df12
    # df32=pd.concat([df12,df7,df7,df7,df7,df7,df7,df7,df7,df7],ignore_index=True)

    # df8=df13
    # df33=pd.concat([df13,df8,df8,df8,df8,df8,df8,df8,df8,df8,df8,df8],ignore_index=True)

    # df9=df14
    # df34=pd.concat([df14,df9,df9,df9,df9,df9,df9,df9,df9,df9,df9,df9,df9,df9,df9],ignore_index=True)

    # df10=df15
    # df35=pd.concat([df15,df10,df10,df10,df10,df10,df10,df10,df10,df10,df10,df10,df10,df10,df10],ignore_index=True)

    # df30=pd.concat([df31,df32,df33,df34,df35],ignore_index=True)
    # df30.to_csv("C:/danhua/carset/copy_carset_1.csv", index=False)

def copyData(train_path):
    df=pd.read_csv(train_path)
    d1 = datetime.datetime(2016, 6, 28)
    car_id_data, model_data, prov_data, city_data, reg_data, car_source_data, car_status_data, mile_data, post_data, liter_data, model_year_data, model_price_data, price_data = [
    ], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(0,len(df)):  
        car_id=df.iloc[i][0]
        model_id=df.iloc[i][1]
        prov_id=df.iloc[i][2]
        city_id=df.iloc[i][3]
        reg_date=df.iloc[i][4]
        price=df.iloc[i][5]
        car_source=df.iloc[i][6]
        car_status=df.iloc[i][7] 
        mile_age=df.iloc[i][8]
        post_time=df.iloc[i][9]
        liter=df.iloc[i][10]
        model_year=df.iloc[i][11] 
        model_price=df.iloc[i][-1]

        car_id_data.append(car_id)
        model_data.append(model_id)
        prov_data.append(prov_id)
        city_data.append(city_id)
        reg_data.append(reg_date)
        car_source_data.append(car_source)
        car_status_data.append(car_status)
        mile_data.append(mile_age)
        post_data.append(post_time)
        liter_data.append(liter)
        model_year_data.append(model_year)
        model_price_data.append(model_price)
        price_data.append(price)

        year,month,day=reg_date.split('-')
        d2 = datetime.datetime(int(year),int(month),int(day))
        reg_date_day=(d1-d2).days
        # if float(price) < float(model_price)*(0.9**(1.*reg_date_day/365))*0.8 or float(price) > min(float(model_price)*(0.9**(1.*reg_date_day/365))*1.2,float(model_price)):
        #     continue

        one=['58', 'dafengche', 'souche']
        two=['273', '51auto', 'baixing', 'carxoo', 'che300_pro', 'cheyipai', 'cn2che', 'ganji','hx2car', 'iautos', 'kuche', 'sohu', 'taoche', 'taotaocar', 'ttpai', 'ttpai_c2c', 'xcar', 'xici', 'youche', 'youxin', 'youxinpai', 'youyiche', 'zg2sc']
        three=['carking', 'chemao', 'ygche']
        four=['che101', 'che168', 'che300','chelaike','cheyitao','jiarenzheng','kx','soucheke']
        five=['ganjihaoche', 'haoche51', 'jiajiahaoche','renrenche']
        if car_source in one:
            weight=8
        elif car_source in two:
            weight=10
        elif car_source in three:
            weight=12
        elif car_source in four:
            weight=15
        else:
            weight=18

        for j in range(0,weight):
            car_id_data.append(car_id)
            model_data.append(model_id)
            prov_data.append(prov_id)
            city_data.append(city_id)
            reg_data.append(reg_date)
            car_source_data.append(car_source)
            car_status_data.append(car_status)
            mile_data.append(mile_age)
            post_data.append(post_time)
            liter_data.append(liter)
            model_year_data.append(model_year)
            model_price_data.append(model_price)
            price_data.append(price)
    df1 = pd.DataFrame([car_id_data, model_data, prov_data, city_data, reg_data, price_data, car_source_data, car_status_data, mile_data,
                       post_data, liter_data, model_year_data, model_price_data]).T
    df1 = df1.rename(columns={0: "car_id", 1: "model_id", 2: "prov_id", 3: "city_id", 4: "reg_date", 5: "report_price", 6: "car_source", 7: "car_status",
                            8: "mile_age", 9: "post_time", 10: "liter", 11: "model_year", 12: "model_price"})
    print len(df1)

    df1.to_csv("C:/danhua/carset_ex_1/copy_train_1.csv", index=False)
    



        






if __name__ == '__main__':
    start = time.clock()

    data = "C:/danhua/carset_ex_1/train_data_11.csv"
    vali_path="C:/danhua/carset/vali_carset_1.csv"
    train_path="C:/danhua/carset/train_carset_1.csv"
    # getDf(data,vali_path,train_path)
    copyData(data)

    print 'done',time.clock()-start
