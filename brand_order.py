# -*- coding: UTF-8 -*-   
# __author__ = 'wdh'
import time 
import datetime
import numpy as np
import pandas as pd
import pymysql

from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf-8') 

def connectSQL(sql):
    conn=pymysql.connect(host="192.168.0.225",user="sa",passwd="sa@1234",db="myche",charset="utf8")
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

def zero():
    return [0. for i in range(5)]

def generate_order(sql,j):
    result =connectSQL(sql)
    feature = ''
    k=0
    id_content=''
    brand_cont={}
    order_init=defaultdict(zero)
    for line in result:
        model_id=line[0]
        sid=line[3]
        bid=line[4]
        model_price=line[7]
        model_year=line[12]
        gear_type=line[16]
        # print type(model_price)

        sn=''
        short_name=line[1].decode("utf-8").split(' ')
        for i in range(1,len(short_name)): # 去掉年款
            sn+=short_name[i]+' '
        sn=sn[:-1]
        # print sn
        key=str(model_id)+','+sn  #车型，(去掉年款)中文字段
        # print key
        order_init[key][0]=int(bid)
        order_init[key][1]=int(sid)
        # try:
        #     order_init[key][2]=float(model_price)
        # except :
        #     print model_price   #这边就可以看到出错的是什么数据了
        #     order_init[key][2]=0.0 

        order_init[key][2]=float(model_price)
        order_init[key][3]=float(model_year)

        order_init[key][4]=float(gear_type)
        # try:
        #     order_init[key][4]=float(gear_type)
        # except:
        #     print gear_type   #这边你就可以看到出错的是什么数据了
        #     order_init[key][4]=0.0 

    
    order_init=sorted(order_init.items(),key=lambda x:(x[1][0],x[1][1],x[1][3],x[1][4],x[1][2]),reverse=False)
    for one_init in order_init:
        pass
        # print one_init[0].split(',')[0],one_init[1][0],one_init[1][1],one_init[1][3],one_init[1][4],one_init[1][2],one_init[0].split(',')[1]
    content='order\n'
    for one_init_a in order_init:
        one_init_a_model_id=one_init_a[0].split(',')[0]  #sid
        # print one_init_a[0],one_init_a[1]
        # print one_init_a[1][1],one_init_a[0].split(',')[0]
        for one_init_b in order_init:
            one_init_b_model_id=one_init_b[0].split(',')[0]  #sid
            if one_init_a[1][1] != one_init_b[1][1] or one_init_a[0].split(',')[0]==one_init_b[0].split(',')[0]: #sid不等 或者 id相等
                continue
            
            one_init_a_short_name=one_init_a[0].split(',')[1]  #取中文字段
            one_init_b_short_name=one_init_b[0].split(',')[1]
            one_init_a_model_year=one_init_a[1][3]   #取年款
            one_init_b_model_year=one_init_b[1][3]
            one_init_a_gear_type=one_init_a[1][4]    #取手自动
            one_init_b_gear_type=one_init_b[1][4]
            one_init_a_model_price=one_init_a[1][2]  #取指导价
            one_init_b_model_price=one_init_b[1][2]

            #同年款，比指导价
            if one_init_a_model_year==one_init_b_model_year :  
                if  one_init_a_model_price<one_init_b_model_price:
                    # print one_init_b_model_id,'>',one_init_a_model_id
                    # print one_init_b_model_year,one_init_b_short_name,'>',one_init_a_model_year,one_init_a_short_name
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    # content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'
                
                if  one_init_a_model_price>one_init_b_model_price:
                    # print one_init_a_model_id,'>',one_init_b_model_id
                    # print one_init_a_model_year,one_init_a_short_name,'>',one_init_b_model_year,one_init_b_short_name 
                    content+=str(one_init_a_model_id)+'>'+str(one_init_b_model_id)+'\n'
                    # content+=str(int(one_init_a_model_year))+' '+one_init_a_short_name+'>'+str(int(one_init_b_model_year))+' '+one_init_b_short_name+'\n' 

            #同车型（名字相同）或同指导价，比年款
            if one_init_a_short_name==one_init_b_short_name or one_init_a_model_price==one_init_b_model_price:
                if one_init_a_model_year<one_init_b_model_year:
                    # print one_init_b_model_id,'>',one_init_a_model_id
                    # print one_init_b_model_year,one_init_b_short_name,'>',one_init_a_model_year,one_init_a_short_name
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    # content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'
                               
                if one_init_a_model_year>one_init_b_model_year:
                    # print one_init_a_model_id,'>',one_init_b_model_id
                    # print one_init_a_model_year,one_init_a_short_name,'>',one_init_b_model_year,one_init_b_short_name 
                    content+=str(one_init_a_model_id)+'>'+str(one_init_b_model_id)+'\n'
                    # content+=str(int(one_init_a_model_year))+' '+one_init_a_short_name+'>'+str(int(one_init_b_model_year))+' '+one_init_b_short_name+'\n'

            #同车型（名字不同，同手自动，指导价差值/指导价较大的小于0.02），比年款
            if one_init_a_short_name!=one_init_b_short_name and one_init_a_gear_type==one_init_b_gear_type and abs(one_init_a_model_price-one_init_b_model_price)/max(one_init_a_model_price,one_init_b_model_price)<0.02:  
                if one_init_a_model_year<one_init_b_model_year:
                    # print one_init_b_model_id,'>',one_init_a_model_id
                    # print one_init_b_model_year,one_init_b_short_name,'>',one_init_a_model_year,one_init_a_short_name
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    # content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'
                if one_init_a_model_year>one_init_b_model_year:
                    # print one_init_a_model_id,'>',one_init_b_model_id
                    # print one_init_a_model_year,one_init_a_short_name,'>',one_init_b_model_year,one_init_b_short_name 
                    content+=str(one_init_a_model_id)+'>'+str(one_init_b_model_id)+'\n'
                    # content+=str(int(one_init_a_model_year))+' '+one_init_a_short_name+'>'+str(int(one_init_b_model_year))+' '+one_init_b_short_name+'\n'


            #同指导价，同年款，比自动手动
            if one_init_a_model_price==one_init_b_model_price and one_init_a_model_year==one_init_b_model_year: 
                if  one_init_a_gear_type>one_init_b_gear_type:
                    # print one_init_a_model_id,'>',one_init_b_model_id
                    # print one_init_a_model_year,one_init_a_short_name,'>',one_init_b_model_year,one_init_b_short_name 
                    content+=str(one_init_a_model_id)+'>'+str(one_init_b_model_id)+'\n'
                    # content+=str(int(one_init_a_model_year))+' '+one_init_a_short_name+'>'+str(int(one_init_b_model_year))+' '+one_init_b_short_name+'\n'
                if  one_init_b_gear_type>one_init_a_gear_type:
                    # print one_init_b_model_id,'>',one_init_a_model_id
                    # print one_init_b_model_year,one_init_b_short_name,'>',one_init_a_model_year,one_init_a_short_name 
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    # content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'

    # print content                
    open('C:/danhua/code/sid1/mt_model_order_1_%s.csv'%j,'wb').write(content)
    try:
        data=pd.read_csv('C:/danhua/code/sid1/mt_model_order_1_%s.csv'%j)
        data=data.drop_duplicates()
        data.to_csv('C:/danhua/code/sid/mt_model_order_%s.csv'%j,index=False)
    except(AttributeError,ValueError,IOError):
        pass



if __name__=='__main__':
    j=2530
    while j<2540:
        try: 
            print j
            sql="select * from mt_model where sid=%s and enabled=1"%j
            generate_order(sql,j)
            # data=pd.read_csv('C:/danhua/code/data1/mt_model_order_1_%s.csv'%j)
            # data=data.drop_duplicates()
            # data.to_csv('C:/danhua/code/data/mt_model_order_%s.csv'%j,index=False)
            j+=1
        except(IOError):
            j+=1


    # mt_model_to_csv('../data/mt_model.txt')
    # data_brand_path=connectSQL()
    