#import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import datetime as dt
import json
import haversine as hs
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcol


def testfunction(num):
    return num+12

#########################
# IMPORT CLEANED DATA
#########################

def import_clean_data(file_name,file_type,polygon_name):
    if file_type == 'pickle':
        with open(file_name,'rb') as picklefile:
            data = pickle.load(picklefile)
    elif file_type =='csv':
        cols = ['unique_id','type','exp_id','DOW','ST','ET','DOY','track_id','time','speed','trv_dist','timestamp','geometry']
        data = pd.read_csv(file_name,encoding='iso-8859-1')
        try: 
            data['timestamp'] = data['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        except:
            data['timestamp'] = data['timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data['geometry'] = data['geometry'].apply(lambda x: Point([float(i) for i in x[7:-1].split()]))
    else:
        print('ERROR: File type not csv or pickle.')
        data = None
    return data

#########################
# LOAD POLYGON AND SENSOR DATA
#########################

def load_data(polygon_name,polygons,file_type='pickle'):
    
    # SELECT ROAD SEGMENT / POLYGON. All with bus route. Direction is driving direction.
    POLYGON = get_polygon(polygon_name,polygons)
    # IMPORT CLEAN DATA
    if file_type.lower()=='pickle':
        file_name = '../output/data_clean/prepared_data_%s.pkl'%(polygon_name) #change again
        file_type = 'pickle'
    else:
        file_name = '../output/data_clean/prepared_data_%s.csv'%(polygon_name) #change again
        file_type = 'csv'
    waypoints_w_dist_mode = import_clean_data(file_name,file_type,polygon_name)
    
    return waypoints_w_dist_mode, POLYGON


######################
# GET POLYGON-SPECIFIC DATA
######################

def get_polygon(polygon_name,polygons):
    #polygons = pd.read_csv('polygons.csv')
    POLYGON = polygons[polygons.name==polygon_name].iloc[0,:].to_dict()
    ps = ['p0_lon_frontleft','p0_lat_frontleft','p0_lon_frontright','p0_lat_frontright',
         'p0_lon_backright','p0_lat_backright','p0_lon_backleft','p0_lat_backleft']
    POLYGON['coords'] = Polygon([[POLYGON[ps[0]],POLYGON[ps[1]]],[POLYGON[ps[2]],POLYGON[ps[3]]],
                                 [POLYGON[ps[4]],POLYGON[ps[5]]],[POLYGON[ps[6]],POLYGON[ps[7]]]])
    [POLYGON.pop(k) for k in ps]
    return POLYGON

#########################
# FILTER SENSOR DATA BY MODE (and calc. vehicle length)
#########################

def get_mode_section(mode,carmodes,waypoints_w_dist_mode,lengths):
    print('Starting with %s.'%(mode))
    if mode=='all':
        mode_section = waypoints_w_dist_mode
        vehlength = 0
        ntot = len(waypoints_w_dist_mode.unique_id.unique())
        for m in set(waypoints_w_dist_mode['type']):
            n = len(waypoints_w_dist_mode[waypoints_w_dist_mode['type']==m].unique_id.unique())
            vehlength += (n/ntot)*lengths[m]
    elif mode in carmodes:
        mode_section = waypoints_w_dist_mode[waypoints_w_dist_mode['type']=='Car']
        vehlength = lengths['Car']
    else:
        mode_section = waypoints_w_dist_mode[waypoints_w_dist_mode['type']==mode]
        vehlength = lengths[mode]
    return mode_section,vehlength

######################
# CALCULATE VALUES - based on timestamp - PARALLEL VERSION
######################

# functions
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield pd.Timestamp(current, tz=None).to_pydatetime() #current
        current += delta

def process_exp_MS_Edie(ls):
    mode_section,mode,POLYGON,day,DOW,exp,step = ls
    
    # get data
    speeds,densities,flows,times = [],[],[],[]
    exp_section = mode_section[mode_section.exp_id==exp]
    # if mode ends with a number, implement penetration rate
    if mode[-1].isdigit():
        frac = int(mode[-4:])/1000
        keep = pd.DataFrame(exp_section.unique_id.unique()).sample(frac=frac).iloc[:,0].values
        exp_section = exp_section[exp_section.unique_id.isin(keep)]
    if not len(exp_section)>0:
        return 

    # make a start/end time
    start = exp_section.timestamp.min().replace(microsecond=0)
    end = exp_section.timestamp.max().replace(microsecond=0)+dt.timedelta(seconds=1)
    print('start:',start,' end:',end,' step:',step)
    
    for sec in datetime_range(start,end,step): # for now
        # 1. slice time interval
        section = exp_section[(exp_section.timestamp>=sec) & (exp_section.timestamp<sec+step)]
        if not (len(section)>0):
            speeds.append(np.nan); densities.append(0); flows.append(0); times.append(sec) 
            continue
        totdist = 0
        tottime = 0
        timeinterval = (section.timestamp.max() - section.timestamp.min()) # timedelta, maximum of step seconds, minimum=time of vehicles in the segment
        timespace = timeinterval.seconds/(60*60) * POLYGON['length'] # time (hours), distance (km)
        ##remove this, added by Alex:
        if not (timespace>0):
            continue
        # 2. group by uniqueid and avg the mean --> for bus/medVeh usually have ≤1 vehicle
        # distance
        dmax = section[['unique_id','trv_dist']].groupby(by=['unique_id']).max() 
        dmin = section[['unique_id','trv_dist']].groupby(by=['unique_id']).min() 
        ddiff = (dmax - dmin)
        totdist += sum(ddiff['trv_dist']) / 1000 # convert to km
        # time
        tmax = section[['unique_id','time']].groupby(by=['unique_id']).max() 
        tmin = section[['unique_id','time']].groupby(by=['unique_id']).min() 
        tdiff = (tmax - tmin) # maximum timesteps when leaving - minimum when entering for each vehicle
        tottime += sum(tdiff['time']) / (60*60) # convert to hr
        # 3. traffic variables
        flow = totdist / timespace
        density = tottime / timespace
        speed = flow / density
        # 4. save
        speeds.append(speed); densities.append(density); flows.append(flow); times.append(sec) 
    return [POLYGON['name'],mode,day,DOW,exp,speeds,densities,flows,times]

######################
# CALCULATE VALUES - based on timestamp - PARALLEL VERSION - with fixed 30 seconds interval for Time-Space, flow and density calculations
######################

# functions

def process_exp_MS_Edie_fixed_interval(ls):
    mode_section,mode,POLYGON,day,DOW,exp,step = ls
    
    # get data
    speeds,densities,flows,times = [],[],[],[]
    exp_section = mode_section[mode_section.exp_id==exp]
    # if mode ends with a number, implement penetration rate
    if mode[-1].isdigit():
        frac = int(mode[-4:])/1000
        keep = pd.DataFrame(exp_section.unique_id.unique()).sample(frac=frac).iloc[:,0].values
        exp_section = exp_section[exp_section.unique_id.isin(keep)]
    if not len(exp_section)>0:
        return 

    # make a start/end time
    start = exp_section.timestamp.min().replace(microsecond=0)
    end = exp_section.timestamp.max().replace(microsecond=0)+dt.timedelta(seconds=1)
    print('start:',start,' end:',end,' step:',step)
    
    for sec in datetime_range(start,end,step): # for now
        # 1. slice time interval
        section = exp_section[(exp_section.timestamp>=sec) & (exp_section.timestamp<sec+step)]
        if not (len(section)>0):
            speeds.append(np.nan); densities.append(0); flows.append(0); times.append(sec) 
            continue
        totdist = 0
        tottime = 0
        if sec < end-step:
            timeinterval = step# is exactly the length of separation, timedelta
        else:
            timeinterval = (section.timestamp.max() - section.timestamp.min())   
                
        timespace = timeinterval.seconds/(60*60) * POLYGON['length'] # time (hours), distance (km)
        ##remove this, added by Alex:
        if not (timespace>0):
            continue
        # 2. group by uniqueid and avg the mean --> for bus/medVeh usually have ≤1 vehicle
        # distance
        dmax = section[['unique_id','trv_dist']].groupby(by=['unique_id']).max() 
        dmin = section[['unique_id','trv_dist']].groupby(by=['unique_id']).min() 
        ddiff = (dmax - dmin)
        totdist += sum(ddiff['trv_dist']) / 1000 # convert to km
        # time
        tmax = section[['unique_id','time']].groupby(by=['unique_id']).max() 
        tmin = section[['unique_id','time']].groupby(by=['unique_id']).min() 
        tdiff = (tmax - tmin) # maximum timesteps when leaving - minimum when entering for each vehicle
        tottime += sum(tdiff['time']) / (60*60) # convert to hr
        # 3. traffic variables
        flow = totdist / timespace
        density = tottime / timespace
        speed = flow / density
        # 4. save
        speeds.append(speed); densities.append(density); flows.append(flow); times.append(sec) 
    return [POLYGON['name'],mode,day,DOW,exp,speeds,densities,flows,times]



######################
# IMPORT FLOW DATA
######################

def import_sensor_data(polygon_name,sensor,cars='No'):
    if sensor=='MS':
        if cars=='Yes':
            file_name = '../output/data_flow/flow_data_MS_%s_cars.json'%(polygon_name)
        else:
            file_name = '../output/data_flow/flow_data_MS_%s.json'%(polygon_name)
    elif sensor=='LD':
        file_name = '../output/data_flow/flow_data_singleLD_%s_30s.json'%(polygon_name)
    else:
        print('Please specify if LD or MS')
        return
    
    with open(file_name) as json_file:
        data = json.load(json_file)
    data = pd.DataFrame.from_records(data,columns=['polygon','mode','day','DOW','exp_id','speeds','densities','flows','times'])
    for i,row in data.iterrows():
        times_list = row.times
        new_list = []
        for v in times_list:
            try:
                date_format_str = '%Y-%m-%d %H:%M:%S.%f'
                v = dt.datetime.strptime(v, date_format_str)
            except:
                date_format_str = '%Y-%m-%d %H:%M:%S'
                v = dt.datetime.strptime(v, date_format_str)
            new_list.append(v)
        data.loc[i,'times'] = [[new_list]]
    return data

def import_sensor_data_with_parked(polygon_name,sensor,cars='No',resample_interval = '5s'):
    if sensor=='MS':
        if cars=='Yes':
            file_name = '../output/data_flow_with_parked_%s/flow_data_MS_%s_cars.json'%(resample_interval, polygon_name)
        else:
            file_name = '../output/data_flow_with_parked_%s/flow_data_MS_%s.json'%(resample_interval, polygon_name)
    elif sensor=='LD':
        file_name = '../output/data_flow_with_parked/flow_data_singleLD_%s_30s.json'%(polygon_name)
    else:
        print('Please specify if LD or MS')
        return
    
    with open(file_name) as json_file:
        data = json.load(json_file)
    data = pd.DataFrame.from_records(data,columns=['polygon','mode','day','DOW','exp_id','speeds','densities','flows','times'])
    for i,row in data.iterrows():
        times_list = row.times
        new_list = []
        for v in times_list:
            try:
                date_format_str = '%Y-%m-%d %H:%M:%S.%f'
                v = dt.datetime.strptime(v, date_format_str)
            except:
                date_format_str = '%Y-%m-%d %H:%M:%S'
                v = dt.datetime.strptime(v, date_format_str)
            new_list.append(v)
        data.loc[i,'times'] = [[new_list]]
    return data

######################
# RESAMPLE & MOVING AVERAGE
######################

def sensor_resample_window(save_nonresampled,modes,resample,window,POLYGON,plot='off',saveplots='off', filna=True):#remove filna
    save_resampled_MS_info = [] # save all
    for mode in modes:
        mode_section = save_nonresampled[save_nonresampled['mode']==mode]

        for DOW,day in [['Wed',[1,2,3,4,5]],['Mon',[6,7,9,10]],['Tue',[11,12,13,14,15]],['Thu',[17,18,19,20]]]:
            # must be restructured / formatted
            speeds, densities, flows, times, exp_ids = [],[],[],[],[]
            for exp in day: 
                exp_section = mode_section[mode_section['exp_id']==exp]
                if not (len(exp_section)>0):
                    continue
                speeds.extend([i for s in exp_section.speeds for i in s])
                #print(speeds)
                densities.extend([i for s in exp_section.densities for i in s])
                flows.extend([i for s in exp_section.flows for i in s])
                try:
                    times.extend([i for s in exp_section.times for i in s[0]][0])
                    #print(times)
                except:
                    times.extend([i for s in exp_section.times for i in s])
                exp_ids.extend([exp]*len(exp_section.speeds.values[0]))
            print(len(speeds),len(densities),len(flows),len(times))    
            if not (len(speeds)>0):
                continue
            df_day = pd.DataFrame(list(zip(speeds,flows,densities,exp_ids,times)),columns=['speeds','flows','densities','exp_ids','times'])
            df_day.index = pd.Series(times)
            #print(df_day)
            # resample: for every half minute, beginning with the starting minute of the intervall, resample the values together continuously. This means, that for every real life half minute, the values stick together and the values between the experiments (battery change) are created and set too zero (so they exist, for merging later on)
            df_day = df_day.resample(resample).mean() 
            # deal with NAN values
            if filna: #remove when removing filna
                df_day = df_day.fillna(0) 
            # moving average to get rid of traffic light effect
            idcs = (df_day.exp_ids!=0)
            cols = ['speeds','densities','flows']
            df_day.loc[idcs,cols] = df_day.loc[idcs,cols].rolling(window=window,min_periods=1).mean()
            # save
            for idx,row in df_day.iterrows():
                save_resampled_MS_info.append([POLYGON['name'],mode,day,DOW,row.exp_ids,row.speeds,
                                          row.densities,row.flows,idx])
  
    return pd.DataFrame(save_resampled_MS_info,columns=['polygon','mode','day','DOW','exp_id','speeds','densities','flows','times'])

######################
# GET CLOSTEST POINT ON LINE
######################

def p4(p1, p2, p3): # line from p1 to p2. Find closest point on line to third point p3.
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return x1+a*dx, y1+a*dy

def get_distance_from_segment_start(data,cs):
    distances = []
    for iter,row in data.iterrows():
        p = row.geometry
        p1 = Point(p4(cs[2],cs[3],[p.x,p.y])) # perpendicular distance to start line; cs=corners of Polygon, cs[2] = back right, cs[3] = back left
        perp_dist = hs.haversine([p.x,p.y],[p1.x,p1.y],unit=hs.Unit.METERS)
        start_centre = [(cs[2][0]+cs[3][0])/2,(cs[2][1]+cs[3][1])/2]
        control_dist = hs.haversine([p.x,p.y],start_centre,unit=hs.Unit.METERS) # dist to centre of start line
        dist_along_segment = min(control_dist,perp_dist)
        distances.append(dist_along_segment)
    return distances

######################
# LD parallel processing
#####################

def process_exp_LD(ls):
    mode_section,mode,POLYGON,day,DOW,exp,location,cs,vehlength,resample,resampleInSec,timetohour,window = ls

    # if mode ends with a number, implement penetration rate
    if mode[-1].isdigit():
        frac = int(mode[-4:])/1000
        keep = pd.DataFrame(mode_section.unique_id.unique()).sample(frac=frac).iloc[:,0].values
        mode_section = mode_section[mode_section.unique_id.isin(keep)]
    if not len(mode_section)>0:
        print('No data')
        return [POLYGON['name'],mode,day,DOW,exp,[],[],[],[]]

    # check POSITION along link. Do objs EVER cross LOOP DETECTOR? if not, remove them
    # check when unique_ids CROSS LOOP DETECTOR, save timestamp
    LD_times, LD_toccs = LD_get_all_data_from_exp(exp,mode_section,location,cs,vehlength)
    if not len(LD_times)>0:
        print('No data')
        return [POLYGON['name'],mode,day,DOW,exp,[],[],[],[]]
    
    # 1. gather all
    LD_info = pd.DataFrame(LD_toccs,columns=['t_occ'],index=LD_times)
    LD_info['count'] = LD_info.t_occ # for now
    # 2. RESAMPLE per minute then calculate flow (veh/hr): for every 'resample'-timesteps, aggregate the occupied time (from veh_length/velocity) summed up ('sum') and count the amount of vehicles during that time ('size')
    LD_info_ = LD_info.resample(resample).agg({'t_occ': 'sum','count':'size'})
    # this is per exp, so only drop last row (incomplete record (maybe only recorded for 2s not 30...))
    LD_info_ = LD_info_.iloc[:-1,:]
    LD_info_['occ'] = LD_info_.t_occ / resampleInSec # t_occupied/t_total
    max_occ = POLYGON['lanes']*1
    LD_info_['occ'].clip(upper=max_occ,inplace=True)
    LD_info_['density'] = LD_info_.occ / (vehlength/1000) # occ/(Lveh+Ldetector) --> occ/Lveh
    LD_info_['flow'] = LD_info_['count']*timetohour
    LD_info_['speed'] = LD_info_['flow'] / LD_info_['density']
    # 3. deal with NAN values
    LD_info_ = LD_info_.fillna(0) #LD_info_ = LD_info_.dropna(subset=['speed'])
    # 4. save
    speeds = list(LD_info_['speed'].values)
    densities = list(LD_info_['density'].values)
    flows = list(LD_info_['flow'].values)
    times = [pd.Timestamp(x, tz=None).to_pydatetime() for x in LD_info_.index.values] # npdatetime64 to pandasdatetime
    print(len(speeds),len(densities),len(flows),len(times))    
    return [POLYGON['name'],mode,day,DOW,exp,speeds,densities,flows,times]

######################
# SINGLE LD (interpolate nearest point, calc. t_occupied for LD)
# Only via distance: set the detector approx 30 m from the end of link, check whether trajectories cross:
# Minimum distance from start segment line < distance of detector to start line, Maximum distance accordingly higher
# Find point on detector: take the 2 nearest values (stopped vehicles handled extra) of the trajectory and interpolate linearly
######################

def LD_get_all_data_from_exp(exp,mode_section,location,cs,vehlength):

    LD_times, LD_toccs = [],[]
    data = mode_section[mode_section['exp_id']==exp]
    if not len(data)>0:
        print('No data found for exp.',exp)
        return [LD_times, LD_toccs]

    # check POSITION along link
    distances = get_distance_from_segment_start(data,cs)
    data['loc_on_seg'] = distances

    # do objs EVER cross LOOP DETECTOR?
    group_df = pd.DataFrame()
    group_df[['unique_id','loc_min']] = data.groupby('unique_id')['loc_on_seg'].min().reset_index()[['unique_id','loc_on_seg']]
    group_df['loc_max'] = data.groupby('unique_id')['loc_on_seg'].max().reset_index()['loc_on_seg']
    keep = group_df[(group_df.loc_min<location)&(group_df.loc_max>location)].unique_id.values
    data = data[data.unique_id.isin(keep)]
    
    # check when unique_ids CROSS LOOP DETECTOR
    # could also just check which geometry point is closest to LD point... but then can't check if they ever cross the LD
    for i in set(data.unique_id):
        id_section = data[data.unique_id==i]
        idx0,idx1 = abs(id_section.loc_on_seg-location).nsmallest(2).index # idx0 min, idx1 2nd min
        t0,x0,v0,t2,x2,v2,x1 = id_section.loc[idx0,'timestamp'],id_section.loc[idx0,'loc_on_seg'],id_section.loc[idx0,'speed'],id_section.loc[idx1,'timestamp'],id_section.loc[idx1,'loc_on_seg'],id_section.loc[idx1,'speed'],location
        # if car doesnt move, find other nearest spots, so don't have 0 velocity
        if x0==x2: 
            curr_mins_drop = id_section[id_section.loc_on_seg==x0].index.values[1:]
            id_section = id_section.drop(curr_mins_drop)
            idx0,idx1 = abs(id_section.loc_on_seg-location).nsmallest(2).index # idx0 min, idx1 2nd min
            t0,x0,v0,t2,x2,v2,x1 = id_section.loc[idx0,'timestamp'],id_section.loc[idx0,'loc_on_seg'],id_section.loc[idx0,'speed'],id_section.loc[idx1,'timestamp'],id_section.loc[idx1,'loc_on_seg'],id_section.loc[idx1,'speed'],location
        # if within 20cm of LD, just take closest one # approx 1/3 of cases
        if (abs(x0-x1)<0.2) or (abs(x0-x2)<0.2): 
            t1 = t0; v1 = v0
        else:
            s = (x2-x1)/(x2-x0)
            t1 = t2-s*(t2-t0) # as s = (t2-t1)/(t2-t0)
            v1 = v2-s*(v2-v0)
        LD_times.append( t1 )
        LD_toccs.append( vehlength*1/(v1/3.6) )  # tocc = Lveh * 1 / v
    return [LD_times, LD_toccs]

######################
# scale data back up using scalefactors
######################

def scaleup(data,polygon,scalefactors,targets):
    # prep factors
    factors = pd.DataFrame(scalefactors,index=['shrink','slope','meank','meanq']).T
    factors['polygon_MS_LD'] = factors.index
    factors['polygon'] = [a[:-3] for a in factors['polygon_MS_LD']]
    polygons = pd.DataFrame(polygon,columns=['polygon'])
    factorsMS = pd.merge(polygons, factors[factors.polygon_MS_LD.str.endswith('MS')], how="left", on=["polygon"])
    factorsLD = pd.merge(polygons, factors[factors.polygon_MS_LD.str.endswith('LD')], how="left", on=["polygon"])

    # scale up
    # MLR version - using pandas
    if len(data.shape)==1:
        if targets[-2:]=='LD':
            factors = factorsLD
        else:
            factors = factorsMS
        factors.index = data.index 
        if targets[0]=='q':
            data = data * factors.slope * factors.meanq # * factors.shrink 
        elif targets[0]=='k':
            data = data * factors.meank # * factors.shrink 
        else: 
            print('Arg4 must be *q* or *k*.')
    # NN version - using tensor/numpy
    else: 
        data = np.array(data)
        for it,t, in enumerate(targets):    
            if t[-2:]=='LD':
                factors = factorsLD
            else:
                factors = factorsMS
            if t[0]=='q':
                data[:,it] = data[:,it] * factors.slope * factors.meanq # * factors.shrink 
            elif t[0]=='k':
                data[:,it] = data[:,it] * factors.meank # * factors.shrink 
            else: 
                print('Arg4 must be *q* or *k*.')
    return data

def scaleup_with_events(data,polygon,scalefactors,targets,scale_stops_lanechanges):
    # prep factors
    factors = pd.DataFrame(scalefactors,index=['shrink','slope','meank','meanq']).T
    factors['polygon_MS_LD'] = factors.index
    factors['polygon'] = [a[:-3] for a in factors['polygon_MS_LD']]
    polygons = pd.DataFrame(polygon,columns=['polygon'])
    factorsMS = pd.merge(polygons, factors[factors.polygon_MS_LD.str.endswith('MS')], how="left", on=["polygon"])
    factorsLD = pd.merge(polygons, factors[factors.polygon_MS_LD.str.endswith('LD')], how="left", on=["polygon"])

    # scale up
    # MLR version - using pandas
    if len(data.shape)==1:
        if targets[-2:]=='LD':
            factors = factorsLD
        else:
            factors = factorsMS
        factors.index = data.index 
        if targets[0]=='q':
            data = data * factors.slope * factors.meanq # * factors.shrink 
        elif targets[0]=='k':
            data = data * factors.meank # * factors.shrink 
        else: 
            print('Arg4 must be *q* or *k*.')
    # NN version - using tensor/numpy
    else: 
        data = np.array(data)
        for it,t, in enumerate(targets):    
            if t[-2:]=='LD':
                factors = factorsLD
            elif t[-2:]=='MS':
                factors = factorsMS
            else:
                factors = scale_stops_lanechanges
            if t[0]=='q':
                data[:,it] = data[:,it] * factors.slope * factors.meanq # * factors.shrink 
            elif t[0]=='k':
                data[:,it] = data[:,it] * factors.meank # * factors.shrink 
            elif t[0]=='s':
                data[:,it] = data[:,it] * scale_stops_lanechanges[0][1] + scale_stops_lanechanges[0][0]
            elif t[0]=='l':
                data[:,it] = data[:,it] * scale_stops_lanechanges[1][1] + scale_stops_lanechanges[1][0]
            else: 
                print('Arg4 must be *q* or *k* or *stop_count**lane_changes*.')
    return data


################################################################################################
# plot line trajectories in x-t for mode 
################################################################################################

def plot_line_trajectories(df_all,mode='Bus',axes=['time','speed','trv_dist'],grid='off'):
    #axes_dict = dict{'speed':1,'distance':2,'time':0} 
    if mode=='all':
        mode_df = df_all
    else:
        mode_df = df_all[df_all['type']==mode]
    all_list = []
    for val in axes:
        all_list.append(mode_df.groupby('unique_id')[val].apply(list))
    fig, ax = plt.subplots(figsize=(12, 8))
    norm = plt.Normalize(0, 50)
    psize=1
    time_speed_trvdist = list(zip(all_list[0],all_list[1],all_list[2]))
    cmap = mcol.LinearSegmentedColormap.from_list("", ["red","yellow","green"])
    for x_y in time_speed_trvdist:
        points = np.array([x_y[0], [x-x_y[2][0] for x in x_y[2]]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=psize, cmap=cmap, norm=norm)
        lc.set_array(x_y[1])
        im = ax.add_collection(lc)
        ax.autoscale()
    if grid=='on':
        major_ticks_y = np.arange(0, df_all.trv_dist.max(), 50)
        minor_ticks_y = np.arange(0, df_all.trv_dist.max(), 10)
        major_ticks_x = np.arange(0, df_all.time.max(), 50)
        minor_ticks_x = np.arange(0, df_all.time.max(), 10)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        # And a corresponding grid
        ax.grid(which='both')
        
def make_shape_from_line_trajectories(df_trajectories,filename):
    gdf = df_trajectories
    def make_lines(point_list):
        if len(point_list)==1:
            return None #Point(point_list)
        else:
            return LineString(point_list)
    breakpoint()
    line_gdf = gdf.sort_values(by=['time']).groupby(['unique_id'])['geometry'].apply(lambda x:make_lines(x.tolist()))
    line_gdf = gpd.GeoDataFrame(line_gdf, geometry='geometry')
    line_gdf.crs="EPSG:4326"
    line_gdf.to_file(filename)


################################################################################################
# all combinations of sensor scenarios 
################################################################################################

sensor_scenarios = {
    # Scenario # Modes 
        # Input columns 
        # Output columns
    'Scen0_qk':# AllLD
        [['q_all_LD','k_all_LD'],
        ['q_all_MS','k_all_MS','q_Bus_MS','k_Bus_MS','q_Car_MS','k_Car_MS',
        'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS','q_Taxi_MS','k_Taxi_MS']],        
    'Scen1_qk':# Taxi AllLD Car5
        [[ 'q_Taxi_MS', 'k_Taxi_MS','q_all_LD', 'k_all_LD', 'q_Car0050_MS', 'k_Car0050_MS'],
        ['q_all_MS','k_all_MS','q_Bus_MS','k_Bus_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS']],
    'Scen2_qk': # Taxi Bus Car5
        [[ 'q_Taxi_MS', 'k_Taxi_MS','q_Bus_MS', 'k_Bus_MS', 'q_Car0050_MS', 'k_Car0050_MS'],
        ['q_all_LD','k_all_LD','q_all_MS','k_all_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS']],
    'Scen3_qk': # Bus AllLD Car5
        [[ 'q_Bus_MS', 'k_Bus_MS','q_all_LD', 'k_all_LD','q_Car0050_MS', 'k_Car0050_MS'],
        ['q_all_MS','k_all_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS','q_Taxi_MS','k_Taxi_MS']],
    'Scen4_qk':# Taxi AllLD Bus
        [[ 'q_Taxi_MS', 'k_Taxi_MS','q_all_LD', 'k_all_LD','q_Bus_MS','k_Bus_MS',],
        ['q_all_MS','k_all_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS']],
    'Scen5_qk':# Taxi Car5
        [[ 'q_Taxi_MS', 'k_Taxi_MS', 'q_Car0050_MS', 'k_Car0050_MS'],
        ['q_all_MS','k_all_MS','q_all_LD', 'k_all_LD','q_Bus_MS','k_Bus_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS']],
    'Scen6_qk':# Taxi Bus
        [[ 'q_Taxi_MS', 'k_Taxi_MS', 'q_Bus_MS', 'k_Bus_MS'],
        ['q_all_MS','k_all_MS','q_all_LD', 'k_all_LD','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS']],
    'Scen7_qk':# Taxi AllLD
        [[ 'q_Taxi_MS', 'k_Taxi_MS','q_all_LD', 'k_all_LD'],
        ['q_all_MS','k_all_MS','q_Bus_MS','k_Bus_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS']],
    'Scen8_qk':# Bus AllLD
        [[ 'q_Bus_MS', 'k_Bus_MS','q_all_LD', 'k_all_LD'],
        ['q_all_MS','k_all_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS','q_Taxi_MS', 'k_Taxi_MS']],
    'Scen9_qk': # Bus Car5
        [[ 'q_Bus_MS', 'k_Bus_MS','q_Car0050_MS','k_Car0050_MS'],
        ['q_all_MS','k_all_MS','q_all_LD', 'k_all_LD','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS','q_Taxi_MS', 'k_Taxi_MS']],
    'Scen10_qk': # AllLD Car5
        [['q_Car0050_MS','k_Car0050_MS','q_all_LD','k_all_LD'],
        ['q_all_MS','k_all_MS','q_Bus_MS','k_Bus_MS','q_Car_MS','k_Car_MS',
         'q_Medium Vehicle_MS','k_Medium Vehicle_MS','q_Motorcycle_MS','k_Motorcycle_MS','q_Taxi_MS','k_Taxi_MS']],
    'Scen11_qk': # AllLD Car5 Taxi Bus
        [['q_Car0050_MS','k_Car0050_MS','q_all_LD','k_all_LD','q_Bus_MS', 'k_Bus_MS','q_Taxi_MS', 'k_Taxi_MS'],
        ['q_all_MS','k_all_MS']],
}

