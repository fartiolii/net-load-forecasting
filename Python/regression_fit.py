import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import time
from scipy.stats import norm
import copy
import warnings

def add_autoregressive_features(df):
    """This function adds to the dataframe in input the following columns:
    - y0_96: containing the net-load values of 96 data points before (48h)
    - y0_336: containing the net-load values of 336 data points before (7d)
    - diff_96: containing the difference between the current net-load value and the one 48h before
    - diff_336: containing the difference between the current net-load value and the one 7d before
    """
    y_diff_96 = np.array(df['node'])[96::]-np.array(df['node'])[:-96:]
    y_diff_336 = np.array(df['node'])[336::]-np.array(df['node'])[:-336:]

    delay = 336
    vec = np.zeros(len(df),)*np.nan
    vec[delay:] = y_diff_336
    df['diff_336'] = vec
    vec = np.zeros(len(df),)*np.nan
    vec[delay:] = np.array(df['node'])[:-delay]
    df['y0_336'] = vec

    delay = 96
    vec = np.zeros(len(df),)*np.nan
    vec[delay:] = y_diff_96
    df['diff_96'] = vec
    vec = np.zeros(len(df),)*np.nan
    vec[delay:] = np.array(df['node'])[:-delay]
    df['y0_96'] = vec
    
    return df




def regression_feature_matrix(df):
    """ This function uses the features in the dataframe df and returns their evaluation matrix according to the linear regression model below."""
    t = df['t']
    dow = df['dow']
    doy = df['doy']
    clock_hour = df['clock_hour']
    hol = df['School_Hol']
    y0_336 = df['y0_336']
    y0_96 = df['y0_96']
    wind100 = df['WindSpd100_weighted.mean_cell']
    wind10 = df['WindSpd10_weighted.mean_cell']
    x2t = df['x2T_weighted.mean_p_max_point']
    x2t_sm = df['x2Tsm_point']
    ssrd = df['SSRD_mean_2_Cap']
    node_sm = df['node_sm']
    emb_wind = df['EMBEDDED_WIND_CAPACITY']
    temp = df['x2T_weighted.mean_p_max_point']
    prec = df['TP_weighted.mean_cell']
    
    
    # seasonality on yearly scale
    wy = 2*np.pi/365
    # seasons (summer or not)
    month = df['targetTime'].dt.month
    is_summer = (month > 5)*(month < 9)   
    # Weekend days
    is_saturday = dow == 'Sam'
    is_sunday = dow == 'Dim'
    is_weekend = 1*(dow=='Sam')+1*(dow=='Dim')
    # seasonality on weekly scale
    day_of_week = 1*(dow=='Lun')+2*(dow=='Mar')+3*(dow=='Mer')+4*(dow=='Jeu')+5*(dow=='Ven')+6*(dow=='Sam')+7*(dow=='Dim')
    ww = 2*np.pi/7
    # seasonality on daily scale
    wd = 2*np.pi/24
    daily_hours = 1*(clock_hour < 18)*(clock_hour > 6)*is_summer+\
                  1*(clock_hour < 17)*(clock_hour > 7)*(is_summer==0)#between 6am and 6pm in summer
    # public holidays
    is_hol = 1*(hol=='Christmas Holiday') + 2*(hol=='School Holiday')
    
    St = np.array([np.ones(doy.shape), t, 
                   
                   # Annual Trend
                   np.cos(wy*doy), np.sin(wy*doy), 
                   np.cos(2*wy*doy), np.sin(2*wy*doy), 
                   np.cos(12*wy*doy), np.sin(12*wy*doy), 
                   np.cos(52*wy*doy), np.sin(52*wy*doy), 
                   
                   # Weekly trend
                   np.cos(ww*day_of_week), np.sin(ww*day_of_week),
                   np.cos(2*ww*day_of_week), np.sin(2*ww*day_of_week),
                   
                   # Daily trend
                   clock_hour, clock_hour**2,
                   day_of_week*clock_hour, day_of_week*clock_hour**2,
                   
                   day_of_week*np.cos(wd*clock_hour), day_of_week*np.sin(wd*clock_hour),
                   day_of_week*np.cos(2*wd*clock_hour), day_of_week*np.sin(2*wd*clock_hour),
                   
                   # Daily trend for consumption
                   is_weekend, is_hol,
                   is_weekend*np.cos(wd*clock_hour), is_weekend*np.sin(wd*clock_hour),
                   is_weekend*np.cos(2*wd*clock_hour), is_weekend*np.sin(2*wd*clock_hour),
                   
                   # METEOROLOGICAL FEATURES
                   
                   # Solar production
                   ssrd, ssrd**2, 
                   ssrd*np.cos(wd*clock_hour)*daily_hours, ssrd*np.sin(wd*clock_hour)*daily_hours, 
                   ssrd*np.cos(2*wd*clock_hour)*daily_hours, ssrd*np.sin(2*wd*clock_hour)*daily_hours,
                   ssrd*np.cos(3*wd*clock_hour)*daily_hours, ssrd*np.sin(3*wd*clock_hour)*daily_hours,
                   
                   # Daily wind production
                   wind10, wind100, 
                   emb_wind*wind100, emb_wind*wind100**2, emb_wind*wind100**3,
                   
                   # Temperature
                   temp, temp**2, 
                   x2t_sm, x2t_sm**2, x2t_sm**3,
                   temp*np.cos(wd*clock_hour), temp*np.sin(wd*clock_hour),
                   temp*np.cos(2*wd*clock_hour), temp*np.sin(2*wd*clock_hour),
                   temp*np.cos(3*wd*clock_hour), temp*np.sin(3*wd*clock_hour), 
                   
                   # Precipitations
                   prec, prec**2, 
                   prec*np.cos(wd*clock_hour), prec*np.sin(wd*clock_hour),
                   prec*np.cos(2*wd*clock_hour), prec*np.sin(2*wd*clock_hour),
                   
                   # Available net-load past values (7d, 48h before and 2 week moving average)
                   y0_336, y0_96, node_sm                   
                  ]).T
    return St
