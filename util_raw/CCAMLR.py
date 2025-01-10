from echolab2.instruments import EK80, EK60
import os
import numpy as np
from echopy import read_calibration as readCAL
from scipy.interpolate import interp1d
import pandas as pd
from geopy.distance import distance
from scipy.signal import savgol_filter

from skimage.transform import  resize
from skimage.transform import  resize_local_mean
import shutil
from skimage.transform import  resize

from echolab2.instruments import EK80, EK60
import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob
import os
import h5py
# from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
# from skimage.transform import  resize

from echopy import transform as tf
from echopy import resample as rs
from echopy import mask_impulse as mIN
from echopy import mask_seabed as mSB
from echopy import get_background as gBN
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH
from echopy import mask_signal2noise as mSN

from pyproj import Geod
geod = Geod(ellps="WGS84")
from pathlib import Path


# from matplotlib.colors import ListedColormap
import re
import traceback
# from pyproj import Proj, transform
import zipfile

import smtplib
import ssl
# import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.text  import MIMEText

from threading import Timer

import xarray as xr
from scipy.interpolate import interp1d
from scipy import integrate



from functools import partial
import multiprocessing  
def find_files_with_suffix(folder_path, suffix):
    # 使用os模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)
    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files

# 读取数据
def read_raw(rawfile, config, transitspeed=3):
    """
    Read EK60 raw data.
    """
    # EK60或EK80文件读取
    try:     
        raw_obj = EK80.EK80()
        raw_obj.read_raw(rawfile)
        # print(raw_obj)
    except Exception as e:            
        # print(e)       
        try:     
            raw_obj = EK60.EK60()
            raw_obj.read_raw(rawfile)
            # print(raw_obj)
        except Exception as e:
            print(e)
    raw_freq= list(raw_obj.frequency_map.keys())
    # print(raw_freq) # [12000, 70000, 38000]
    raws = []
    for f in raw_freq:
        # print(raw_obj.frequency_map[f])
        
        if len( raw_obj.raw_data[raw_obj.frequency_map[f][0]][:]) >1:
            raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][:]  
            # svr = np.empty([ raw_data[0].n_samples , raw_obj.n_pings ])               
            # for k in range(len(raw_data)):
            #     svr[:,k]    =  raw_data[k].power
        else:
        
            raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]  
    
            if np.shape(raw_data)[0]>1:                     
                cal_obj = raw_data.get_calibration()
                
                try: 
                    cal_obj.gain=float(config['CALIBRATION']['gain']       )
                except:
                    pass
                try: 
                    cal_obj.sa_correction=float(config['CALIBRATION']['sa_correction'])
                except:
                    pass
                try: 
                    cal_obj.beam_width_alongship=float(config['CALIBRATION']['beam_width_alongship']       )
                except:
                    pass
                try: 
                    cal_obj.beam_width_athwartship=float(config['CALIBRATION']['beam_width_athwartship']       )
                except:
                    pass
                try: 
                    cal_obj.angle_offset_alongship=float(config['CALIBRATION']['angle_offset_alongship']       )
                except:
                    pass
                try: 
                    cal_obj.angle_offset_athwartship=float(config['CALIBRATION']['angle_offset_athwartship']       )
                except:
                    pass
                    
        # -------------------------------------------------------------------------
        # get raw data    
        Sv    = np.transpose(raw_data.get_Sv(calibration = cal_obj).data)
        theta = np.transpose(raw_data.angles_alongship_e)
        phi   = np.transpose(raw_data.angles_athwartship_e)
        t     = raw_data.get_Sv(calibration = cal_obj).ping_time
        r     = raw_data.get_Sv(calibration = cal_obj).range
        alpha = raw_data.absorption_coefficient[0]
                # sv_obj = raw_data.get_sv(calibration = cal_obj)
                # positions =pd.DataFrame(  raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1] )
                
                # svr = np.transpose( 10*np.log10( sv_obj.data ) )

            # print(sv_obj.range.max())
            # r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )

        # -------------------------------------------------------------------------
        # get nmea data
        transect,Tpos,LON,LAT,lon,lat,nm,km,kph,knt = nmea(raw_obj, t, preraw=None)
        km -= np.nanmin(km)
        nm -= np.nanmin(nm)

        # -------------------------------------------------------------------------
        # if stationary, turn transect number to negative & resume distances
        if (nm[-1]-nm[0]) / (np.float64(t[-1]-t[0])/(1000*60*60))<transitspeed:
            km -= np.nanmin(km)
            nm -= np.nanmin(nm)
        
        Tmot,PITCH,ROLL,HEAVE,pitch,roll,heave,pitchmax,rollmax,heavemax =motion(raw_obj, t, preraw=None)              
    
        


        # -------------------------------------------------------------------------
        # return RAW data  
        raw = {'rawfiles'  : [os.path.split(rawfile)[-1]],
            #    'continuous': continuous                  ,
                'transect'  : transect                    ,
                'Sv'        : Sv                          ,
                'theta'     : theta                       ,
                'phi'       : phi                         ,
                'alpha'     : alpha                       ,
                'r'         : r                           ,
                't'         : t                           ,       
                'lon'       : lon                         ,
                'lat'       : lat                         ,
                'nm'        : nm                          ,
                'km'        : km                          ,
                'kph'       : kph                         ,
                'knt'       : knt                         ,
                'pitch'     : pitch                       ,
                'roll'      : roll                        ,
                'heave'     : heave                       ,
                'pitchmax'  : pitchmax                    ,
                'rollmax'   : rollmax                     ,
                'heavemax'  : heavemax                    ,
                'Tpos'      : Tpos                        ,
                'LON'       : LON                         ,
                'LAT'       : LAT                         ,
                'Tmot'      : Tmot                        ,
                'PITCH'     : PITCH                       ,
                'ROLL'      : ROLL                        ,
                'HEAVE'     : HEAVE                       }
    
        raws.append(raw)
    # -------------------------------------------------------------------------
    # delete objects to free up memory RAM 
    del raw_obj
    return raws[0], raws[1], raws[2]

def nmea(ek60, t, preraw=None, maxspeed=25):
    """
    Reads NMEA time, longitude, and latitude, and use these variables to
    compute cumulated distance and speed.
    
    Args:
        ek60  (object): PyechoLab's object containing EK60 raw data
        t (datetime64): 1D array with ping time.
        maxspeed (int): Maximum speed allowed in knots. If above, there is 
                        probably an error in longitude and latitude positions
                        and data shouldn't be trusted. An error will be raised.
        
    Returns:
        datetime64: 1D array with NMEA time
        float     : 1D array with NMEA longitude
        float     : 1D array with NMEA latitude
        float     : 1D array with ping-interpolated longitude
        float     : 1D array with ping-interpolated latitude
        float     : 1D array with ping-interpolated distance (nautical miles)
        float     : 1D array with ping-interpolated distance (kilometers)
        float     : 1D array with ping interpolated speed (kilometers/hour)
        float     : 1D array with ping interpolated speed (knots) 
    """
    
    # get GPS datagrams
    GPS = ek60.nmea_data.get_datagrams(['GGA', 'GLL', 'RMC'],
                                      return_fields=['longitude','latitude'])
    
    # find NMEA datagrams with time, longitude, and latitude
    T, LON, LAT = None, None, None
    for k,v in GPS.items():
        if isinstance(v['time'], np.ndarray
                      ) & isinstance(v['longitude'], np.ndarray
                      ) & isinstance(v['latitude'], np.ndarray):
            
            if preraw is None:
                T          = v['time'     ]
                LON        = v['longitude']
                LAT        = v['latitude' ]
                transect   = 1
                continuous = False
            else:
                gpsrate   = np.float64(np.mean(np.diff(v['time'])))
                timelapse = np.float64(v['time'][0]-preraw['Tpos'][-1])
                if (timelapse<5*gpsrate)&(timelapse>0):
                    T          = np.r_[preraw['Tpos'][-7:] ,v['time'     ]]
                    LON        = np.r_[preraw['LON' ][-7:] ,v['longitude']]
                    LAT        = np.r_[preraw['LAT' ][-7:] ,v['latitude' ]]
                    transect   = abs(preraw['transect'])
                    continuous = True
                else:
                    print('time breach in preceding NMEA data')
                    T          = v['time'     ]
                    LON        = v['longitude']
                    LAT        = v['latitude' ]
                    transect   = abs(preraw['transect']) +1
                    continuous = False
            
            # filter LON/LAT to smooth out anomalies
            for i in range(3):        
                LAT = savgol_filter(LAT, 51, 3)
                LON = savgol_filter(LON, 51, 3)
                # TODO: so far, smoothing is applied whether or not the data
                #       needs to smoothed. Need to find a robust way for
                #       detecting noisy data, and apply the smoothing after
                #       warning the user.
            
            # calculate distance in kilometres and nautical miles
            KM = np.zeros(len(LON))*np.nan
            NM = np.zeros(len(LON))*np.nan
            for i in range(len(LON)-1):
                if np.isnan(LAT[i]) | np.isnan(LON[i]) | np.isnan(LAT[i+1]) | np.isnan(LON[i+1]):
                    KM[i+1] = np.nan
                    NM[i+1] = np.nan
                else:
                    KM[i+1] = distance((LAT[i], LON[i]), (LAT[i+1], LON[i+1])).km
                    NM[i+1] = distance((LAT[i], LON[i]), (LAT[i+1], LON[i+1])).nm
            
            # calculate speed in kilometers per hour and knots
            KPH = KM[1:]/(np.float64(np.diff(T))/1000)*3600
            KNT = NM[1:]/(np.float64(np.diff(T))/1000)*3600
            
            if max(KNT)>maxspeed:
                raise Exception(str('Incoherent maximum speed (>%s knts). NMEA'
                                    +' data shouldn\'t be trusted.')% maxspeed)
            
            # calculate cumulated distance in kilometres and nautical miles
            KM  = np.nancumsum(KM)
            NM  = np.nancumsum(NM)
            
            # convert time arrays to timestamp floats
            epoch = np.datetime64('1970-01-01T00:00:00')
            Tf = np.float64(T-epoch)
            tf = np.float64(t-epoch)
            
            # match array lengths (screwed up due to cumsum & diff operations)
            KM  = KM [1:]
            NM  = NM [1:]
            KPH = KPH[ :]
            KNT = KNT[ :] 
            LON = LON[1:]
            LAT = LAT[1:]
            T   = T  [1:]
            Tf  = Tf [1:]            
            
            # get time-position, time-distance, & time-speed interpolated functions
            fLON = interp1d(Tf, LON, bounds_error=False, fill_value='extrapolate')
            fLAT = interp1d(Tf, LAT, bounds_error=False, fill_value='extrapolate')
            fNM  = interp1d(Tf, NM , bounds_error=False, fill_value='extrapolate')
            fKM  = interp1d(Tf, KM , bounds_error=False, fill_value='extrapolate')
            fKPH = interp1d(Tf, KPH, bounds_error=False, fill_value='extrapolate')
            fKNT = interp1d(Tf, KNT, bounds_error=False, fill_value='extrapolate')
            
            # get interpolated position, distance and speed for pingtime
            lon = fLON(tf)
            lat = fLAT(tf)
            nm  = fNM (tf)
            km  = fKM (tf)
            kph = fKPH(tf)
            knt = fKNT(tf)       
            
            # Cumulated distance should continue the distance array from
            # the preceeding raw file, if there is continuity
            if continuous:
                
                # measure file gap distances
                kmgap = distance((preraw['lat'][-1], preraw['lon'][-1]),
                                 (lat[0]           , lon[0]           )).km
                nmgap = distance((preraw['lat'][-1], preraw['lon'][-1]),
                                 (lat[0]           , lon[0]           )).nm
                
                # reset current distances to zero and...
                km -= np.nanmin(km)
                nm -= np.nanmin(nm)
                
                # ... add gap & last distance value from preceeding file
                km = km + preraw['km'][-1] + kmgap
                nm = nm + preraw['nm'][-1] + nmgap
            
            
            # if there is no continuity with preceeding file
            else:
                
                # reset current distances to cero
                km -= np.nanmin(km)
                nm -= np.nanmin(nm)
    
            break
        
    if (T is None) | (LON is None) | (LAT is None):
        print('GPS data not found')
        
        T, LON, LAT= None, None, None
        lon = np.zeros_like(t)*np.nan
        lat = np.zeros_like(t)*np.nan
        nm  = np.zeros_like(t)*np.nan
        km  = np.zeros_like(t)*np.nan
        kph = np.zeros_like(t)*np.nan
        knt = np.zeros_like(t)*np.nan
        
    return transect, T, LON, LAT, lon, lat, nm, km, kph, knt

def motion(ek60, t, preraw=None):
    """
    Get motion data. Experimental. 
    """
    
    # get motion datagram
    shr= ek60.nmea_data.get_datagrams('SHR', return_fields=['pitch','roll','heave'])
    
    # return empty if motion data not found
    if any(v is None for v in shr['SHR'].values()):        
        # logger.warn('Motion data not found')    
        T        = None
        PITCH    = None
        ROLL     = None
        HEAVE    = None
        pitch    = np.ones(len(t))*np.nan
        roll     = np.ones(len(t))*np.nan
        heave    = np.ones(len(t))*np.nan
        pitchmax = np.ones(len(t))*np.nan
        rollmax  = np.ones(len(t))*np.nan
        heavemax = np.ones(len(t))*np.nan
        
        return T, PITCH, ROLL, HEAVE, pitch, roll, heave, pitchmax, rollmax, heavemax
    
    # extract values if motion data is found  
    else:
        
        # get only current values if there is no preceeding RAW data         
        if preraw is None:
            T     = shr['SHR']['time' ]
            PITCH = shr['SHR']['pitch']
            ROLL  = shr['SHR']['roll' ]
            HEAVE = shr['SHR']['heave']
        
        # concatenate preceeding values...
        else:
            nmeapingrate = np.float64(np.mean(np.diff(shr['SHR']['time' ])))
            timelapse    = np.float64(shr['SHR']['time' ][0]-preraw['Tmot'][-1])
            
            # ... if preceeding RAW data is consecutive
            if (timelapse<5*nmeapingrate)&(timelapse>0):
                T     = np.r_[preraw['Tmot' ][-14:] ,shr['SHR']['time' ]]
                PITCH = np.r_[preraw['PITCH'][-14:] ,shr['SHR']['pitch']]
                ROLL  = np.r_[preraw['ROLL' ][-14:] ,shr['SHR']['roll' ]]
                HEAVE = np.r_[preraw['HEAVE'][-14:] ,shr['SHR']['heave']]
            
            # , get current ones otherwise
            else:
                print('time breach in preceding MOTION data')
                T     = shr['SHR']['time' ]
                PITCH = shr['SHR']['pitch']
                ROLL  = shr['SHR']['roll' ]
                HEAVE = shr['SHR']['heave']
    
        # get maximum motion in a 10-seconds moving window (absolute values)
        PITCHdf  = pd.DataFrame(abs(PITCH))
        PITCHmax = np.array(PITCHdf.rolling(window=10,
                                            center=False).max()).flatten()
        ROLLdf  = pd.DataFrame(abs(ROLL))
        ROLLmax = np.array(ROLLdf.rolling(window=10,
                                          center=False).max()).flatten()
        HEAVEdf  = pd.DataFrame(abs(HEAVE))
        HEAVEmax = np.array(HEAVEdf.rolling(window=10,
                                            center=False).max()).flatten()
               
        # convert time arrays to timestamp floats
        epoch = np.datetime64('1970-01-01T00:00:00')
        Tf    = np.float64(T-epoch)
        tf    = np.float64(t-epoch)           
        
        # get time-motion interpolated functions
        fPITCH    = interp1d(Tf, PITCH   , bounds_error=False)
        fROLL     = interp1d(Tf, ROLL    , bounds_error=False)
        fHEAVE    = interp1d(Tf, HEAVE   , bounds_error=False)
        fPITCHmax = interp1d(Tf, PITCHmax, bounds_error=False)
        fROLLmax  = interp1d(Tf, ROLLmax , bounds_error=False)
        fHEAVEmax = interp1d(Tf, HEAVEmax, bounds_error=False)
        
        # get ping-interpolated motion
        pitch    = fPITCH   (tf)
        roll     = fROLL    (tf)
        heave    = fHEAVE   (tf)
        pitchmax = fPITCHmax(tf)
        rollmax  = fROLLmax (tf)
        heavemax = fHEAVEmax(tf)
        
        return T, PITCH, ROLL, HEAVE, pitch, roll, heave, pitchmax, rollmax, heavemax

def ccamlr(raw):
    jdx = [0, 0]
    #--------------------------------------------------------------------------       
    # Load variables
    rawfiles    = raw['rawfiles']
    transect    = raw['transect']
    alpha120    = raw['alpha'   ]
    r120        = raw['r'       ]    
    t120        = raw['t'       ]
    lon120      = raw['lon'     ]
    lat120      = raw['lat'     ]
    nm120       = raw['nm'      ]
    km120       = raw['km'      ]
    knt120      = raw['knt'     ]
    kph120      = raw['kph'     ]
    pitchmax120 = raw['pitchmax']
    rollmax120  = raw['rollmax' ]
    heavemax120 = raw['heavemax']
    Sv120       = raw['Sv'      ]
    theta120    = raw['theta'   ]
    phi120      = raw['phi'     ]
    #--------------------------------------------------------------------------       
    # Clean impulse noise      
    Sv120in, m120in_ = mIN.wang(Sv120, thr=(-70,-40), erode=[(3,3)],
                                dilate=[(7,7)], median=[(7,7)])
    #TODO: True is valid
    # -------------------------------------------------------------------------
    # estimate and correct background noise       
    p120           = np.arange(len(t120))                
    s120           = np.arange(len(r120))                
    bn120, m120bn_ = gBN.derobertis(Sv120, s120, p120, 5, 20, r120, alpha120)
    Sv120clean     = tf.log(tf.lin(Sv120in) - tf.lin(bn120))
    #TODO: True is valid
    # -------------------------------------------------------------------------
    # mask low signal-to-noise 
    m120sn             = mSN.derobertis(Sv120clean, bn120, thr=12)
    Sv120clean[m120sn] = -999
    
    # -------------------------------------------------------------------------
    # get mask for near-surface and deep data
    m120rg = mRG.outside(Sv120clean, r120, 19.9, 250)
    
    # -------------------------------------------------------------------------
    # get mask for seabed
    m120sb = mSB.ariza(Sv120, r120, r0=20, r1=1000, roff=0,
                       thr=-38, ec=1, ek=(3,3), dc=10, dk=(3,7))



    # -------------------------------------------------------------------------
    # get seabed line
    idx                = np.argmax(m120sb, axis=0)
    sbline             = r120[idx]
    sbline[idx==0]     = np.inf
    sbline             = sbline.reshape(1,-1)
    sbline[sbline>250] = np.nan
    # sbline[sbline>250] = -999
    
    # -------------------------------------------------------------------------
    # get mask for non-usable range    
    m120nu = mSN.fielding(bn120, -80)[0]
    
    # -------------------------------------------------------------------------
    # remove unwanted (near-surface & deep data, seabed & non-usable range)
    m120uw = m120rg|m120sb|m120nu
    Sv120clean[m120uw] = np.nan
    # Sv120clean[m120uw] = -999
    
    # -------------------------------------------------------------------------
    # get swarms mask
    k = np.ones((3, 3))/3**2
    Sv120cvv = tf.log(convolve2d(tf.lin(Sv120clean), k,'same',boundary='symm'))   
    p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
    s120           = np.arange(np.shape(Sv120cvv)[0]+1 )
    m120sh, m120sh_ = mSH.echoview(Sv120cvv, r120, km120*1000, thr=-70,
                                   mincan=(3,10), maxlink=(3,15), minsho=(3,15))
    
    # -------------------------------------------------------------------------
    # get Sv with only swarms
    Sv120sw                    = Sv120clean.copy()
    Sv120sw[~m120sh & ~m120uw] = -999
    
    # -------------------------------------------------------------------------
    # resample Sv from 20 to 250 m, and every 1nm     
    r120intervals                     = np.array([20, 250])
    nm120intervals                    = np.arange(jdx[1], nm120[-1],   1) 
    Sv120swr, r120r, nm120r, pc120swr = rs.twod(Sv120sw, r120, nm120,
                                                r120intervals, nm120intervals,
                                                log=True)
        
    # -------------------------------------------------------------------------
    # remove seabed from pc120swr calculation, only water column is considered
    m120sb_             = m120sb*1.0
    m120sb_[m120sb_==1] = np.nan
    # m120sb_[m120sb_==1] = -999
    pc120water          = rs.twod(m120sb_, r120, nm120,
                                  r120intervals, nm120intervals)[3]
    pc120swr            = pc120swr/pc120water * 100

    # -------------------------------------------------------------------------
    # resample seabed line every 1nm
    sbliner = rs.oned(sbline, nm120, nm120intervals, 1)[0]
    
    # -------------------------------------------------------------------------
    # get time resampled, interpolated from distance resampled
    epoch  = np.datetime64('1970-01-01T00:00:00')
    t120f  = np.float64(t120 - epoch)    
    f      = interp1d(nm120, t120f)
    t120rf = f(nm120r)
    t120r  = np.array(t120rf, dtype='timedelta64[ms]') + epoch
    
    t120intervalsf = f(nm120intervals)
    t120intervals  = np.array(t120intervalsf, dtype='timedelta64[ms]') + epoch
    
    # -------------------------------------------------------------------------
    # get latitude & longitude resampled, interpolated from time resampled
    f       = interp1d(t120f, lon120)
    lon120r = f(t120rf)
    f       = interp1d(t120f, lat120)
    lat120r = f(t120rf)
    
    # -------------------------------------------------------------------------
    # resample back to full resolution  
    Sv120swrf, m120swrf_   = rs.full(Sv120swr, r120intervals, nm120intervals, 
                                     r120, nm120)

    #TODO: True is valid
    # -------------------------------------------------------------------------
    # compute Sa and NASC from 20 to 250 m or down to the seabed depth
    # r_new = np.arange(20,250,10)  # 110
    r_resize = 10
    Sv120sw2 = Sv120sw.copy()
    # r_new= np.arange(0,r120.max(),r_resize)
    r_new = r120intervals
    Sv120sw2[Sv120sw2==-999] = np.nan
    Sv120swr2, r120r2, nm120r2, pc120swr2 = rs.twod(Sv120sw2, r120, nm120,
                                            r_new, nm120,
                                            log=True)
    Sa120swr2   = np.zeros_like(Sv120swr2)*np.nan
    NASC120swr2 = np.zeros_like(Sv120swr2)*np.nan
    for i in range(Sv120swr2.shape[0]):  
        for j in range(Sv120swr2.shape[1]):
            # print(i,j)
            if (np.isnan(sbline[0,j])) | (sbline[0,j]>250):
                Sa120swr2  [i,j] = tf.log(tf.lin(Sv120swr2[i,j])*10)
                NASC120swr2[i,j] = 4*np.pi*1852**2*tf.lin(Sv120swr2[i,j])*10
            else:
                Sa120swr2  [i,j] = tf.log(tf.lin(Sv120swr2[i,j])*(sbline[0,j]%10))
                NASC120swr2[i,j] = 4*np.pi*1852**2*tf.lin(Sv120swr2[i,j])*(sbline[0,j]%10)
    
    # -------------------------------------------------------------------------
    # compute Sa and NASC from 20 to 250 m or down to the seabed depth
    Sa120swr   = np.zeros_like(Sv120swr)*np.nan
    NASC120swr = np.zeros_like(Sv120swr)*np.nan
    for i in range(len(Sv120swr[0])):
        if (np.isnan(sbliner[0,i])) | (sbliner[0,i]>250):
            Sa120swr  [0,i] = tf.log(tf.lin(Sv120swr[0,i])*(250-20))
            NASC120swr[0,i] = 4*np.pi*1852**2*tf.lin(Sv120swr[0,i])*(250-20)
        else:
            Sa120swr  [0,i] = tf.log(tf.lin(Sv120swr[0,i])*(sbliner[0,i]-20))
            NASC120swr[0,i] = 4*np.pi*1852**2*tf.lin(Sv120swr[0,i])*(sbliner[0,i]-20)
    
    # -------------------------------------------------------------------------
    # return processed data outputs
    m120_ = m120in_ | m120bn_ | m120sh_ | m120swrf_
    #TODO: True is valid

    pro = {'rawfiles'       : rawfiles   , # list of rawfiles processed
           'transect'       : transect   , # transect number
           'r120'           : r120       , # range (m)
           't120'           : t120       , # time  (numpy.datetime64)
           'lon120'         : lon120     , # longitude (deg)
           'lat120'         : lat120     , # latitude (deg)
           'nm120'          : nm120      , # distance (nmi)
           'km120'          : km120      , # distance (km)
           'knt120'         : knt120     , # speed (knots)
           'kph120'         : kph120     , # speed (km h-1)
           'pitchmax120'    : pitchmax120, # max value in last pitching cycle (deg)
           'rollmax120'     : rollmax120 , # max value in last rolling cycle (deg)
           'heavemax120'    : heavemax120, # max value in last heave cycle (deg)
           'Sv120'          : Sv120      , # Sv (dB)
           'theta120'       : theta120   , # Athwart-ship angle (deg)
           'phi120'         : phi120     , # Alon-ship angle (deg)
           'bn120'          : bn120      , # Background noise (dB)
           'Sv120in'        : Sv120in    , # Sv without impulse noise (dB)
           'Sv120clean'     : Sv120clean , # Sv without background noise (dB)          
           'Sv120sw'        : Sv120sw    , # Sv with only swarms (dB)
           'nm120r'         : nm120r     , # Distance resampled (nmi)
           'r120intervals'  : r120intervals, # r resampling intervals
           'nm120intervals' : nm120intervals, # nmi resampling intervals
           't120intervals'  : t120intervals, # t resampling intervals
           'sbliner'        : sbliner    , # Seabed resampled (m)
           't120r'          : t120r      , # Time resampled (numpy.datetime64)
           'lon120r'        : lon120r    , # Longitude resampled (deg)
           'lat120r'        : lat120r    , # Latitude resampled (deg)
           'Sv120swr'       : Sv120swr   , # Sv with only swarms resampled (dB)
           'pc120swr'       : pc120swr   , # Valid samples used to compute Sv120swr (%)
           'Sa120swr'       : Sa120swr   , # Sa from swarms, resampled (m2 m-2)
           'NASC120swr'     : NASC120swr , # NASC from swarms, resampled (m2 nmi-2)
           'Sv120swrf'      : Sv120swrf  , # Sv with only swarms, resampled, full resolution (dB)         
           'm120_'          : m120_      ,  # Sv mask indicating valid processed data (where all filters could be applied)
           'm120sw'         : m120sh     ,
           'm120sb'         : m120sb     ,
           'nasc_swarm'     : NASC120swr2 ,
           'sbline'         : sbline[0],}
    return pro

def report(pro):
    df_nasc_file=pd.DataFrame([])
    df_nasc_file['distance_m'] = np.append(np.array([0]),geod.line_lengths(lons=pro['lon120'],lats=pro['lat120']) )
    df_nasc_file['lat']=pro['lat120']
    df_nasc_file['lon']=pro['lon120']
    df_nasc_file['bottomdepth_m']=pro['sbline'][0]
    df_nasc_file['nasc_swarm']=pro['nasc_swarm'][0]
    df_nasc_file.index=pro['t120']

    df_nasc_r_file=pd.DataFrame([])
    df_nasc_r_file['distance_m'] = np.append(np.array([0]),geod.line_lengths(lons=pro['lon120r'],lats=pro['lat120r']) )
    df_nasc_r_file['latr']=pro['lat120r']
    df_nasc_r_file['lonr']=pro['lon120r']
    df_nasc_r_file['bottomdepth_m']=pro['sbliner'][0]
    df_nasc_r_file['nasc_swarm']=pro['NASC120swr'][0]
    df_nasc_r_file.index=pro['t120r']
    return df_nasc_file, df_nasc_r_file

def makedata(pro, raw70, raw38):
    t120 = pro['t120']  # time  (numpy.datetime64)
    r120 = pro['r120']  # range (m)
    Sv120 = pro['Sv120']  # Sv (dB)
    Sv120_swarm = pro['Sv120sw']  # Sv with only swarms (dB)
    # Sv120swr = pro['Sv120sw']  # Sv with only swarms resampled (dB)
    # Sv120swrf = pro['Sv120swrf']  # Sv with only swarms, resampled, full resolution (dB)
    # t120r = pro['t120r']  # Time resampled
    t120intrvls = pro['t120intervals']  # t resampling intervals
    nm120 = pro['nm120']
    lat = pro['lat120']
    lon = pro['lon120']
    nm120r = pro['nm120r']  # Distance resampled (nmi)
    lon120r = pro['lon120r']  # Longitude resampled (deg)
    lat120r = pro['lat120r']  # Latitude resampled (deg)
    # sbline120r = pro['sbliner'][0, :]
    NASC120swr = pro['NASC120swr'][0, :]  # NASC from swarms, resampled (m2 nmi-2)
    # pc120swr = pro['pc120swr'][0, :]  # Valid samples used to compute Sv120swr (%)
    nm, rr = np.meshgrid(nm120, r120)
    # llat, llon = np.meshgrid(lat120r, lon120r)
    llat, llon = np.meshgrid(lat, lon)

    Sv_label = pro['m120sw']+pro['m120sb']*2
    Sv70, Sv38 = raw70['Sv'], raw38['Sv']

    image = [Sv120, Sv70, Sv38, rr, nm]
    return image, Sv_label, llat, llon, nm
    

def makeh5dataset(file_path, out_path, ini_file, day=21):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    txt_path = out_path + '\\make_dataset_report.txt'
    txt_file = open(txt_path, 'w')
    raw_lists = find_files_with_suffix(file_path, '.raw')
    config = configparser.ConfigParser()
    config.read(ini_file)
    for ii, rawname in enumerate(raw_lists):
        rawday = int(rawname[7:9])
        if rawday <= day:
            print(ii, rawname)
            print('Reading')
            rawfile = os.path.join(file_path, rawname)
            raw120, raw70, raw38 = read_raw(rawfile, config)
            print('ccamlring')
            try:
                pro120 = ccamlr(raw120)
            except:
                txt_file.write(rawname)
                continue
            print('making')
            image, label, llat, llon, nm = makedata(pro120, raw70, raw38)
            print('Saving')
            # 打包为h5文件
            f = h5py.File(out_path + '\\' + rawname.replace('.raw', '.h5'), 'w')
            f['image'] = image
            f['label'] = label
            f['lat'] = llat
            f['lon'] = llon
            f['nm'] = nm
            f.close()


if __name__ == '__main__':
    inputpath = r'D:\Mao\paper_mao\Mao_code\A_M\swarm_detect\exp4_forpaper\acoustic data'
    outputpath = r'D:\Mao\paper_mao\Mao_code\A_M\swarm_detect\exp4_forpaper\datah5_21'
    ini_file = r'F:\mao\paper_mao\Mao_code\A_M\exp4\show\seting_my.ini'
    makeh5dataset(inputpath, outputpath, ini_file, day=21)
    # rawfile = r'F:\mao\paper_mao\Mao_code\A_M\exp4\show\testdata\D20210521-T000640.raw'
    # folder_source = r'F:\mao\paper_mao\Mao_code\A_M\exp4\show\testdata'
    # globstr =  os.path.join( glob.escape(folder_source),'*.raw')  # 读取所有raw文件
    # new_df_files = pd.DataFrame([])
    # new_df_files['path'] = glob.glob( globstr )  # 获取文件路径
    # print('found '+str(len(new_df_files)) + ' raw files')   # 显示文件数
    # ini_file = r'F:\mao\paper_mao\Mao_code\A_M\exp4\show\seting_try.ini'
    # config = configparser.ConfigParser()
    # config.read(ini_file)
    # raw120, raw70, raw38 = read_raw(rawfile, config)
    # pro120 = ccamlr(raw120)
    # df_nasc_file, df_nasc_r_file = report(pro120)
    # print(df_nasc_file)
    # print('-------------------------------')
    # print(df_nasc_r_file)
