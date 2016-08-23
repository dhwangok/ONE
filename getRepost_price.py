# -*- coding: UTF-8 -*-
# __author__ = 'lz'


import cPickle as pickle
import datetime
import random
import sys
import time
import pymysql
import numpy as np
import pandas as pd
import requests
from lxml import etree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

reload(sys)
sys.setdefaultencoding( "utf-8" )

class dropToExact():
    def __init__(self):
        self.headers = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36'
        self.url = 'http://www.che300.com/buycar/'
        self.cookies = 'PHPSESSID=c2l9ufbd9sgsh7s7cofjiigdt2; Hm_lvt_f33b83d5b301d5a0c3e722bd9d89acc8=1459908269,1459909125,1460001753,1462255929; Hm_lpvt_f33b83d5b301d5a0c3e722bd9d89acc8=1462264578; Hm_lvt_f5ec9aea58f25882d61d31eb4b353550=1459908269,1459909125,1460001753,1462255930; Hm_lpvt_f5ec9aea58f25882d61d31eb4b353550=1462264578'



    def splitline(self,dt):
        car_id = dt[0]
        model_id = dt[1]
        prov_id = dt[2]
        city_id = dt[3]
        reg_date = dt[4]
        car_source =dt[5]
        car_status =dt[6]
        mile_age =dt[7]
        post_time =dt[8]
        liter = dt[9]
        model_price = dt[10]
        price =dt[11]
        return car_id,model_id,prov_id,city_id,reg_date,car_source,car_status,mile_age,post_time,liter,model_price,price



    def compareOnLine(self,url,car_source):
        # page = urllib.urlopen(url)
        # html = page.read()
        # # return html



        # req = urllib2.Request(url)
        # response = urllib2.urlopen(req)
        # html = response.read()


        html = requests.get(url, headers={
        'User-Agent': self.headers
    },cookies = {'Cookie':self.cookies}).content
        selector = etree.HTML(html)
        print selector

        repost_price = selector.xpath('//*[@id="price"]/@value')

        predict_price = selector.xpath('//*[@class="clearfix"]/li[12]/text()')

        comefrom = selector.xpath('//*[@class="dtir-4in clearfix"]/li[5]/a/@href')
        if len(str(comefrom[0]).split('='))<2:
            comefromnew = comefrom[0]
        else:
            comefromnew = str(comefrom[0]).split('=')[2]
        html58 = requests.get(comefromnew, headers={
        'User-Agent': self.headers
    },cookies = {'Cookie':self.cookies}).content
        selector58 = etree.HTML(html58)

        flag58 = selector58.xpath('//*[@id="bdshare_l_c"]/ul/li[3]/a/text()')
        midprice = float(repost_price[0])-float(predict_price[0])

        flag = False

        if (car_source=='58'):
            if flag58 is None:
                flag = False
            else:
                if midprice<float(10000):
                    flag = True
        else:
            if midprice<float(1):
                flag = True

        return flag



    def comPareDropData(self,frompath,topath):
        f = open(frompath)
        line = f.readline().strip()
        dropDataList= []
        featureToWrite = ''
        while line:
            dt = line.split(',')
            car_id = dt[0]
            car_source =dt[5]
            print car_id
            url = self.url+'x'+str(car_id)
            try:
                boolean = self.compareOnLine(url,car_source)
                print boolean
            except (IndexError,requests.exceptions):
                line= f.readline().strip()
                continue
            if boolean:
                featureToWrite = line+'\n'

            line= f.readline().strip()
        open(topath,'wb').write(featureToWrite)

if __name__=='__main__':
    ln=dropToExact()
    frompath='C:/danhua/code/analysis/fromSQL/0.25/exact_data_0.25/exact_data_1_0.25.csv'
    topath='C:/danhua/code/analysis/fromSQL/0.25/isright.csv'
    ln.comPareDropData(frompath,topath)
