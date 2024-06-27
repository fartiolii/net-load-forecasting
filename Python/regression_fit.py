import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import time
from datetime import date, datetime, timezone, timedelta
import datetime
import pytz
import astral, astral.sun
from scipy.stats import norm
import copy
import warnings


def add_autoregressive_features(df):
    """This function adds to the dataframe in input the following columns:
    - y0_96: containing the net-load values of 96 data points before (48h)
    - y0_336: containing the net-load values of 336 data points before (7 days)
    - y0_672: containing the net-load values of 672 data points before (14 days)
    - diff_96: containing the difference between the current net-load value and the one 48h before
    - diff_336: containing the difference between the current net-load value and the one 7d before
    - diff_672: containing the difference between the current net-load value and the one 14d before
    """
    y_diff_96 = np.array(df['node'])[96::]-np.array(df['node'])[:-96:]
    y_diff_336 = np.array(df['node'])[336::]-np.array(df['node'])[:-336:]
    y_diff_672 = np.array(df['node'])[672::]-np.array(df['node'])[:-672:]
    
    delay = 672
    vec = np.zeros(len(df),)*np.nan
    vec[delay:] = y_diff_672
    df['diff_672'] = vec
    vec = np.zeros(len(df),)*np.nan
    vec[delay:] = np.array(df['node'])[:-delay]
    df['y0_672'] = vec
    
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
    """This function returns the model matrix for the LM-Point model. It uses the features in the input dataframe df"""
    t = df['t']
    dow = df['dow']
    doy = df['doy']
    dow_RpH = df['dow_RpH']
    clock_hour = df['clock_hour']
    hol = df['School_Hol']
    y0_336 = df['y0_336']
    y0_96 = df['y0_96']
    y0_672 = df['y0_672']
    wind100 = df['WindSpd100_weighted.mean_cell']
    wind10 = df['WindSpd10_weighted.mean_cell']
    x2t_sm = df['x2Tsm_point']
    ssrd = df['SSRD_mean_2_Cap']
    ss = df['SSRD_mean_2']
    node_sm = df['node_sm']
    emb_wind = df['EMBEDDED_WIND_CAPACITY']
    temp = df['x2T_weighted.mean_p_max_point']
    prec = df['TP_weighted.mean_cell']
    
    
    # Weekdays and weekends
    is_weekend = 1*(dow=='Sam')+1*(dow=='Dim')
    is_weekday = 1*(is_weekend == 0)
    is_mon_tue = 1*(dow=='Lun')+1*(dow=='Mar')
    is_mer_gio_fri = 1*(dow=='Mer')+1*(dow=='Jeu')+1*(dow=='Ven')
    
    # Yearly frequency
    wy = 2*np.pi/365  
    
    # Days of the week
    mon = 1*(dow=='Lun')
    tue = 1*(dow=='Mar')
    wed = 1*(dow=='Mer')
    thu = 1*(dow=='Jeu')
    fri = 1*(dow=='Ven')
    sat = 1*(dow=='Sam')
    sun = 1*(dow=='Dim')
    
    # Daily frequency
    wd = 2*np.pi/24
    
    # Public holidays
    is_hol = 1*(dow_RpH=='Christmas Holiday')+1*(dow_RpH=='School Holiday')+1*(dow_RpH=='Autumn Break')+1*(dow_RpH=='Summer Break')+\
             1*(dow_RpH=='Summer Break')+1*(dow_RpH=='Summer Holiday')+1*(dow_RpH=='Easter Holiday')+1*(dow_RpH=='February Half Term')
    is_hol[is_hol>1] = 1
    chris_hol = 1*(hol=='Christmas Holiday')
    school_hol = 1*(hol=='School Holiday')
    
    # Model matrix
    St = np.array([t, # Trend
                   
                   # Second order Fourier (annual)
                   np.cos(wy*doy), np.sin(wy*doy), 
                   np.cos(2*wy*doy), np.sin(2*wy*doy), 
                   
                   # First order Fourier (monthly)
                   np.cos(12*wy*doy), np.sin(12*wy*doy), 
                   
                   # Day of the week
                   mon, tue, wed, thu, fri, sat, sun, 
                   
                   # Clock time by day-type 
                   mon*clock_hour, tue*clock_hour, wed*clock_hour, thu*clock_hour, fri*clock_hour,
                   sat*clock_hour, sun*clock_hour,
                   
                   mon*clock_hour**2, tue*clock_hour**2, wed*clock_hour**2, thu*clock_hour**2, fri*clock_hour**2,
                   is_weekend*clock_hour**2,
                   
                   # Clock time by holiday-type
                   chris_hol*clock_hour, chris_hol*clock_hour**2, 
                   school_hol*clock_hour, school_hol*clock_hour**2,
                   
                   # Second order Fourier (daily) by weekday or weekend
                   is_weekday*np.cos(wd*clock_hour), is_weekday*np.sin(wd*clock_hour),
                   is_weekday*np.cos(2*wd*clock_hour), is_weekday*np.sin(2*wd*clock_hour),
                   is_weekend*np.cos(wd*clock_hour), is_weekend*np.sin(wd*clock_hour),
                   is_weekend*np.cos(2*wd*clock_hour), is_weekend*np.sin(2*wd*clock_hour),
                   
                    # Second order Fourier (daily) by holiday-type
                   is_hol*np.cos(wd*clock_hour), is_hol*np.sin(wd*clock_hour),
                   is_hol*np.cos(2*wd*clock_hour), is_hol*np.sin(2*wd*clock_hour),
                   
                
                   # Mean irradiance scaled by solar capacity
                   ssrd, ssrd**2,
                   
                   # Clock time by mean irradiance scaled by solar capacity
                   ssrd*clock_hour, (ssrd*clock_hour)**2, (ssrd*clock_hour)**3,
                   
                   # Population-weighted temperature by mean irradiance scaled by solar capacity
                   ssrd*temp, (ssrd*temp)**2, 
                   
                   # Second order Fourier (daily) by mean irradiance scaled by solar capacity 
                   ssrd*np.cos(wd*clock_hour), ssrd*np.sin(wd*clock_hour), 
                   ssrd*np.cos(2*wd*clock_hour), ssrd*np.sin(2*wd*clock_hour),
                   
                   # 10m and 100m wind speed
                   wind10, wind100, 
                   
                   # 100m wind speed by embedded wind capacity
                   emb_wind*wind100, emb_wind*wind100**2, emb_wind*wind100**3,
    
                   # Population-weighted temperature
                   temp, temp**2, temp**3,
                   
                   # Population-weighted temperature by clock-time
                   temp*clock_hour, (clock_hour*temp)**2, (clock_hour*temp)**3,
                   
                   # Second order Fourier (daily) by population-weighted temperature
                   temp*np.cos(wd*clock_hour), temp*np.sin(wd*clock_hour),
                   temp*np.cos(2*wd*clock_hour), temp*np.sin(2*wd*clock_hour),
                   temp*np.cos(3*wd*clock_hour), temp*np.sin(3*wd*clock_hour),
                   
                   # 48h rolling mean point temperature
                   x2t_sm, x2t_sm**2, x2t_sm**3,
                   
                   # Mean precipitation
                   prec, prec**2, 
                   
                   # Second order Fourier (daily) by mean precipitation
                   prec*np.cos(wd*clock_hour), prec*np.sin(wd*clock_hour),
                   prec*np.cos(2*wd*clock_hour), prec*np.sin(2*wd*clock_hour),
        
                   # Net-load value of 7 and 14 days before
                   y0_336, y0_672, 
                   
                   # Net-load value of 2 days before by day-type
                   is_mon_tue*y0_96, is_mer_gio_fri*y0_96, is_weekend*y0_96,
                   
                   # Two-week rolling average of net-load
                   node_sm                   
                  ]).T
    return St
