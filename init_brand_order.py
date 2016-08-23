# -*- coding: UTF-8 -*-   
# __author__ = 'lz'
import time 
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf-8') 


def findIndex(feature_id,feature_set):
    index=-1
    for _id in feature_set:
        index+=1
        if _id==feature_id:
            return index
def oneHotEncoding(instance,feature_set,feature_id):
    feature_index=findIndex(feature_id,feature_set)
    for i in range(len(feature_set)):
        if i==feature_index:
            instance+='1'+','
        else:
            instance+='0'+','
    return instance

def splitlinemtmodel(dt):
    # print len(dt)
    _id = dt[0]

    sid = dt[3]
    bid = dt[4]
    discharge_standard = dt[29]
    model_price = dt[7]
    price_level_in_series = dt[8] if dt[8]!='\N' else '1' # ?
    price_type = dt[9]############# not need
    enabled = dt[10]############# not need
    model_year = dt[12]
    liter = dt[13] if dt[13]!='\N' else '1' # ?
    liter_turbo = dt[14]
    engine_power = dt[15] if dt[15]!='\N' else '-1'# ?
    gear_type = dt[16] if dt[16]!='\N' else '1'# ?
    residual_value = dt[20]############# not need
    disable_eval_left_year=dt[23]############# not need
    disable_eval_right_year=dt[24]############# not need
    eval_group=dt[25] if dt[25]!='\N' else '0'
    price_group=dt[26]############# not need
    model_status=dt[27]
    market_date=dt[30]
    fuel_type = dt[31] if dt[31]!='\N' else '0'
    door_number = dt[36] if dt[36]!='\N' else '4'
    # seat_number = dt[37] if dt[37]!='\N'  else '5'
    drive_type = dt[41] if dt[41]!='\N' else '1'

    if '/'  in dt[37]:
        seat_number=dt[37].split('/')[0]
    elif '-'  in dt[37]:
        seat_number=dt[37].split('-')[0]
    elif dt[37]!='\N':
        seat_number=dt[37]
    else:
        seat_number='5'

    

    # seat_number = dt[37] if dt[37]!='5-8' else '5'

    if market_date!='\N':
        pass
    else:
        timeArray = time.strptime(str(int(model_year)-1), "%Y")
        market_date = time.strftime("%Y-%m-%d", timeArray)
    
    if discharge_standard!='\N':
        pass
    else:
        discharge_standard='国3' if int(model_year)<=2009 else '国4'

    return _id,sid,bid,discharge_standard,model_price,price_level_in_series,price_type,enabled,model_year,liter,liter_turbo,engine_power,gear_type,residual_value,disable_eval_left_year,disable_eval_right_year,eval_group, price_group,model_status,market_date,fuel_type,door_number,seat_number,drive_type

def mt_model_to_csv(path ):
 
    f = open(path)
    line = f.readline().strip()
    line = f.readline().strip()
    
    sid_set=set()
    bid_set=set()
    discharge_standard_set=set()
    city_set=set()

    while line:
        instance = ''
        dt = line.split('\t')
        # print dt[0]
        try:
            _id,sid,bid,discharge_standard,model_price,price_level_in_series,price_type,enabled,model_year,liter,liter_turbo,engine_power,gear_type,residual_value,disable_eval_left_year,disable_eval_right_year,eval_group, price_group,model_status,market_date,fuel_type,door_number,seat_number,drive_type = splitlinemtmodel(dt)
        except:
            line = f.readline().strip()
            print 'id===========---'
            continue           
        if int(enabled)!=1:
            line = f.readline().strip()
            continue
        sid_set.add(sid)
        bid_set.add(bid)
        discharge_standard_set.add(discharge_standard)
        line = f.readline().strip()

    print len(sid_set),len(bid_set),len(discharge_standard_set)
    # return

    f = open(path)
    line = f.readline().strip()
    line = f.readline().strip()
    d1 = datetime.datetime(2016, 3, 3)
    
    feature = ''
    while line:
        instance = ''
        dt = line.split('\t')
        try:
            _id,sid,bid,discharge_standard,model_price,price_level_in_series,price_type,enabled,model_year,liter,liter_turbo,engine_power,gear_type,residual_value,disable_eval_left_year,disable_eval_right_year,eval_group, price_group,model_status,market_date,fuel_type,door_number,seat_number,drive_type = splitlinemtmodel(dt)
        except:
            print 'id==========='
            line = f.readline().strip()
            
            continue
        if int(enabled)!=1:
            line = f.readline().strip()
            continue


        
        # instance += _id+','
        # instance += oneHotEncoding(instance,sid_set,sid)
        # instance += oneHotEncoding(instance,bid_set,bid)
        # instance += oneHotEncoding(instance,discharge_standard_set,discharge_standard)
        
        if bid !='1':
            line = f.readline().strip()
            continue
        instance += _id+','
        instance += sid+','
        instance += bid+','
        # instance += discharge_standard+','

        instance += model_price+','

        # instance += price_level_in_series+','
        # instance += price_type+','
        # instance += enabled+','
        instance += model_year+','
        instance += market_date+','
        year,month,day=market_date.split('-')
        d2 = datetime.datetime(int(year),int(month),int(day))
        reg_date_day=str((d1-d2).days)
        instance += reg_date_day+','
        # instance += liter+','
        # instance += liter_turbo+','
        # instance += engine_power+','
        # instance += gear_type+','
        # # instance += residual_value+','
        # # instance += disable_eval_left_year+','
        # # instance += disable_eval_right_year+','
        # instance += eval_group+','
        # # instance += price_group+','
        # instance += model_status+','
        
        
        # instance += fuel_type+','
        # instance += door_number+','
        # instance += seat_number+','
        # instance += drive_type+'\n'
        instance += dt[1]+','
        instance = instance[:-1]+'\n'
        feature+=instance

 
        print _id
        line = f.readline().strip()
    open('../data/mt_model_aodi.csv','wb').write(feature)
    return


    f = open(path)
    line = f.readline().strip()
    line = f.readline().strip()
    
    
    feature = ''
    k=0
    id_content=''
    while line:

        instance = ''
        dt = line.split('\t')
        _id,sid,bid,discharge_standard,model_price,price_level_in_series,price_type,enabled,model_year,liter,liter_turbo,engine_power,gear_type,residual_value,disable_eval_left_year,disable_eval_right_year,eval_group, price_group,model_status,market_date,fuel_type,door_number,seat_number,drive_type = splitlinemtmodel(dt)
        
        
        if int(enabled)!=1:
            line = f.readline().strip()
            continue
        k+=1
        id_content+=_id+','+dt[1]+'\n'
        if k>1000:
            pass
            # break


        # instance += _id+','
        # print market_date
        year,month,day=market_date.split('-')
        d2 = datetime.datetime(int(year),int(month),int(day))
        reg_date_day=str((d1-d2).days)


        # instance += _id+','
        # instance += sid+','
        # instance += bid+','
        # instance += discharge_standard+','

        instance += model_price+','
        # instance += price_level_in_series+','
        instance += price_type+','
        instance += enabled+','
        instance += model_year+','
        instance += liter+','
        instance += liter_turbo+','
        instance += engine_power+','
        instance += gear_type+','
        # instance += residual_value+','
        instance += disable_eval_left_year+','
        instance += disable_eval_right_year+','
        instance += eval_group+','
        instance += price_group+','
        instance += model_status+','
        instance += reg_date_day+','
        
        instance += fuel_type+','
        instance += door_number+','
        instance += seat_number+','
        instance += drive_type+','

        instance = oneHotEncoding(instance,sid_set,sid)
        instance = oneHotEncoding(instance,bid_set,bid)
        instance = oneHotEncoding(instance,discharge_standard_set,discharge_standard)

        instance =instance[:-1]+'\n'
        

        feature+=instance
        
        data=instance.strip().split(',')
        try:
            x=np.array(data)[:].astype(np.float)
        except Exception,e:
            print _id,data
            print e
            return
        if int(_id)%1000==0:
            print _id,len(x)

        line = f.readline().strip()
    # open('../data/fea_mt_model_sample.csv','wb').write(feature)
    # open('../data/id_mt_model_sample.csv','wb').write(id_content)

    open('../data/fea_mt_model_2.csv','wb').write(feature)
    open('../data/id_mt_model.csv','wb').write(id_content)
    print len(sid_set),len(bid_set),len(discharge_standard_set)
def zero():
    return [0. for i in range(5)]
def generate_order(data_brand_path):
    f = open(data_brand_path)
    line = f.readline().strip()
    feature = ''
    k=0
    id_content=''
    brand_cont={}
    order_init=defaultdict(zero)
    while line:

        instance = ''
        dt = line.split(',')
        model_id=dt[0]
        sid=dt[1]
        bid=dt[2]
        model_price=dt[3]
        model_year=dt[4]
        reg_date_day=dt[6]
        sn=''
        short_name=dt[7].split(' ')
        # print len(short_name)
        for i in range(1,len(short_name)):
            sn+=short_name[i]+' '
        # print 
        sn=sn[:-1]
        key=model_id+','+sn
        order_init[key][0]=int(bid)
        order_init[key][1]=int(sid)
        order_init[key][2]=float(model_price)
        order_init[key][3]=float(model_year)
        order_init[key][4]=float(reg_date_day)
        line = f.readline().strip()  
    order_init=sorted(order_init.items(),key=lambda x:(x[1][0],x[1][1],x[1][3],x[1][4],x[1][2]),reverse=False)
    for one_init in order_init:
        pass
        print one_init[0].split(',')[0],one_init[1][0],one_init[1][1],one_init[1][3],one_init[1][4],one_init[1][2],one_init[0].split(',')[1]
    content=''
    for one_init_a in order_init:
        model_year_reg_cnt=0
        model_short_name_cnt=0
        one_init_a_model_id=one_init_a[0].split(',')[0]
        # print one_init_a[0].split(',')[1]
        for one_init_b in order_init:
            one_init_b_model_id=one_init_b[0].split(',')[0]
            if one_init_a[1][1] != one_init_b[1][1] or one_init_a[0].split(',')[0]==one_init_b[0].split(',')[0]: #sid不等 或者 编号相等
                continue
            one_init_a_short_name=one_init_a[0].split(',')[1] #取中文字段
            one_init_b_short_name=one_init_b[0].split(',')[1]
            one_init_a_model_year=one_init_a[1][3]
            one_init_b_model_year=one_init_b[1][3]
            one_init_a_reg_date_day=one_init_a[1][4]
            one_init_b_reg_date_day=one_init_b[1][4]
            one_init_a_model_price=one_init_a[1][2]
            one_init_b_model_price=one_init_b[1][2]
            if one_init_a_model_year==one_init_b_model_year and one_init_a_reg_date_day==one_init_b_reg_date_day:
                model_year_reg_cnt+=1
                if  one_init_a_model_price<one_init_b_model_price:
                    print one_init_b_model_id,'>',one_init_a_model_id
                    print one_init_b_model_year,one_init_b_short_name,'>',one_init_a_model_year,one_init_a_short_name
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'
                
                if  one_init_a_model_price>one_init_b_model_price:
                    print one_init_a_model_id,'>',one_init_b_model_id
                    print one_init_a_model_year,one_init_a_short_name,'>',one_init_b_model_year,one_init_b_short_name 
                    content+=str(one_init_a_model_id)+'>'+str(one_init_b_model_id)+'\n'
                    content+=str(int(one_init_a_model_year))+' '+one_init_a_short_name+'>'+str(int(one_init_b_model_year))+' '+one_init_b_short_name+'\n'
                            
            if one_init_a_short_name==one_init_b_short_name:
                model_short_name_cnt+=1
                if one_init_a_model_year<one_init_b_model_year:
                    print one_init_b_model_id,'>',one_init_a_model_id
                    print one_init_b_model_year,one_init_b_short_name,'>',one_init_a_model_year,one_init_a_short_name
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'
                               
                if one_init_a_model_year>one_init_b_model_year:
                    print one_init_a_model_id,'>',one_init_b_model_id
                    print one_init_a_model_year,one_init_a_short_name,'>',one_init_b_model_year,one_init_b_short_name 
                    content+=str(one_init_b_model_id)+'>'+str(one_init_a_model_id)+'\n'
                    content+=str(int(one_init_b_model_year))+' '+one_init_b_short_name+'>'+str(int(one_init_a_model_year))+' '+one_init_a_short_name+'\n'
    # open('C:/danhua/code/data/mt_model_aodi_order.csv','wb').write(content)
        # if model_year_reg_cnt>0:
        #     for one_init_b in order_init:
        #         one_init_b_model_id=one_init_b[0].split(',')[0]
        #         if one_init_a[1][1] != one_init_b[1][1] or one_init_a[0].split(',')[0]==one_init_b[0].split(',')[0]:
        #             continue
        #         if one_init_a_model_year==one_init_b_model_year and one_init_a_reg_date_day==one_init_b_reg_date_day:
        #             if  one_init_a_model_price>one_init_b_model_price:
        #                 print one_init_b_model_id,'>',one_init_a_model_price
        #                 print one_init_b_model_year
                                 


if __name__=='__main__': 
    # mt_model_to_csv('../data/mt_model.txt')
    data_brand_path='C:/danhua/code/data/mt_model_aodi.csv'
    generate_order(data_brand_path)