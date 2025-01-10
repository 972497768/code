import numpy as np
from packages.echolab2.instruments import EK60
import os, sys
from echopy.reading import read_calibration as readCAL
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from geopy.distance import distance
import pandas as pd
from packages.echolab2.processing import afsc_bot_detector
import matplotlib.ticker as ticker

def _adjust_xdata(x_data):
    """Shifts x axis data when displaying data with imshow.

    This method shifts the x axis data such that the values are centered
    on the sample pixel's x axis. When using the extents keyword with
    imshow, the pixels are no longer centered on the underlying grid.
    This method should be used to shift x axis values of data that is
    being plotted on top of an echogram.

    Args:
        x_data (array): A numpy array of x axis data to be shifted.

    Returns:
        An array, adj_x, of shifted data.
    """

    # Determine the number of pings.
    n_pings = x_data.shape[0]
    print('n_pings:', n_pings)

    # Determine the 1/2 median ping interval.
    ping_ints = np.ediff1d(x_data)
    ping_int = np.nanmedian(ping_ints) / 2.0
    print('ping_ints:', ping_ints.shape)
    print('ping_int:', ping_int)

    adj_x = np.empty(n_pings, dtype='float')
    mid = int((float(n_pings)/ 2.0) + 0.5)
    print('mid:', mid)
    adj_x[0:mid] = x_data[0:mid] + ((np.arange(
        mid, dtype='float')[::-1] / mid) * ping_int)
    adj_x[mid:] = x_data[mid:] - ((np.arange(
        (n_pings-mid), dtype='float') / mid) * ping_int)
    
    print('adj_x:', adj_x)
    print('len(adj_x):', len(adj_x))

    return adj_x


# def update(fig, threshold, data_attribute, Sv):
#     """Updates the plot."""

#     # This is a custom tick formatter for datetime64 values as floats.
#     # def format_datetime(x, pos=None):
#     #     try:
#     #         dt = x.astype('datetime64[ms]').astype('object')
#     #         tick_label = dt.strftime("%H:%M:%S")
#     #     except:
#     #         tick_label = ''

#     #     return tick_label

#     # Get the thresholds if we have been given one.
#     if data_attribute:
#         if not hasattr(Sv, data_attribute):
#             raise ValueError("The data_attribute : " + data_attribute +
#                                 "does not exist in the data_object provided.")
#         else:
#             data_attribute = getattr(Sv, data_attribute)
#             self_data_attribute = data_attribute
#     else:
#         self_data_attribute = Sv.data
#     if threshold:
#         threshold = threshold
#     else:
#         # Just use the min/max if we don't have thresholds.
#         threshold = [np.nanmin(self_data_attribute),
#                         np.nanmax(self_data_attribute)]

#     # Transform the data so it looks right with imshow.
#     echogram_data = np.flipud(np.rot90(self_data_attribute, 1))
#     # Determine the vertical axis attribute.
#     if hasattr(Sv, 'range'):
#         y_label_attribute = 'range'
#     elif hasattr(Sv, 'depth'):
#         y_label_attribute = 'depth'
#     else:
#         # It's neither range nor depth so we'll just plot sample number.
#         y_label_attribute = None

#     # Determine the vertical extent of the data and the y label.
#     if y_label_attribute is None:
#         # We don't have a valid axis.  Just use sample number.
#         yticks = np.arange(echogram_data.shape[0])
#         y_label = 'sample'
#     elif hasattr(Sv, y_label_attribute):
#         yticks = getattr(Sv, y_label_attribute)
#         y_label = y_label_attribute + ' (m)'
#     else:
#         # We don't have a valid axis.  Just use sample number.
#         yticks = np.arange(echogram_data.shape[0])
#         y_label = 'sample'

#     # The x ticks are the pings times as serial time.
#     xticks = Sv.ping_time.astype('float')

#     # Plot the sample data.
#     xes_image = fig.gca().imshow(
#         echogram_data, vmin=threshold[0], vmax=threshold[
#             1], aspect='auto', interpolation='none', extent=[
#             xticks[0], xticks[-1], yticks[-1], yticks[0]], origin='upper')

#     # Set our custom x-axis formatter.
#     # fig.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
#     #     format_datetime))

#     # Set the x axis label to the month-day-year of the first.
#     # datetime64 we have in the data. This will fail if there are no
#     # valid times so we'll not label the axis if that happens.
#     try:
#         x = fig.gca().get_xticks()[0]
#         dt = x.astype('datetime64[ms]').astype('object')

#         x_label = dt.strftime("%m-%d-%Y")
#     except:
#         x_label = ''
#     fig.gca().set_xlabel(x_label)

#     # Set the Y axis label.
#     fig.gca().set_ylabel(y_label)

#     #  TODO:  the grid should be optional and at some point we will have
#     #         a method to plot an integration grid which would disable the
#     #         mpl grid.

#     # Apply the grid.
#     fig.gca().grid(True, color='k')

#     # There seems to be a bug in matplotlib where extra white space is
#     # added to the figure around the echogram if other elements are
#     # plotted on the echogram even if their data ranges fall *within* the
#     # echogram.  This doesn't happen when we use ping number as the x axis,
#     # but we don't want to use ping number because everything we plot on
#     # the echogram would have to be mapped to ping number.  Weirdly, what
#     # solves the issue is calling set_xlim/set_ylim without arguments
#     # which should only return the current limits, but seems to fix this
#     # issue.
#     fig.gca().set_xlim()
#     fig.gca().set_ylim()



def _adjust_ydata( y_data, Sv, fig):
    """Shifts y axis data when displaying data with imshow.

    This method shifts the y axis data such that the values are centered
    on the sample pixel's y axis.  When using the extents keyword with
    imshow the pixels are no longer centered on the underlying grid.
    This method should be used to shift y axis values of data that is
    being plotted on top of an echogram.

    NOTE: I think this is correct, but I have not tested it fully.

    Args:
        y_data (array): A numpy array of y axis data to be shifted.

    Returns:
        An array, adj_y, of shifted data.
    """

    # Determine the 1/2 sample thickness.
    half_samp = Sv.sample_thickness / 2.0
    # print('half_samp:', half_samp)

    # Create the return array.
    adj_y = np.empty(y_data.shape[0], dtype='float')
    # print('adj_y:', adj_y)

    # Get the y axis limits and the limits midpoint.
    y_limits = fig.gca().set_ylim()
    mid = (y_limits[0] - y_limits[1]) / 2.0
    # print('mid:', mid)

    # Set all values at the midpoint without change.
    adj_y[y_data == mid] = y_data[y_data == mid]
    # print('adj_y:', adj_y.shape)

    # In the steps below, the signs are reversed from what you would think
    # they should be. This is (I am assuming) because were reversing the
    # Y axis when setting the extents to imshow.

    # Identify all values less than the midpoint and subtract 1/2 sample
    # thickness * 1 - % of midpoint.
    adj_idx = y_data < mid
    adj_y[adj_idx] = y_data[adj_idx] + \
            (1.0 - (y_data[adj_idx] / mid)) * half_samp
    # print('adj_y:', adj_y.shape)
    # Identify all values greater than the midpoint and add 1/2 sample
    # thickness * % of full range.
    adj_idx = y_data > mid
    adj_y[adj_idx] = y_data[adj_idx] - \
            (y_data[adj_idx] / y_limits[0]) * half_samp
    # print('adj_y:', adj_y.shape)

    return adj_y

def raw(rawfile, channel=120, transitspeed=3, calfile=None,
        soundspeed=None, absorption=None, preraw=None, search_min = 15, backstep = 35):
    """
    Read EK60 raw data.
    """

    # -------------------------------------------------------------------------
    # load rawfile
    ek60 = EK60.EK60()
    ek60.read_raw(rawfile)
    try:
        botfile = rawfile.replace('.raw', '.bot')
        ek60.read_raw(botfile)
        isbot = True
    except:
        isbot =False
        # print('no .bot data')

    # -------------------------------------------------------------------------
    # read frequency channel data
    # logger.info('Reading File '+rawfile.split('/')[-1]+'...')
    ch = None
    for i in ek60.channel_id_map:
        if str(channel) + ' kHz' in ek60.channel_id_map[i]:
            ch = i
            break
    if ch is None:
        sys.exit(str(channel) + ' kHz channel not found!')
    raw = ek60.get_raw_data(channel_number=ch)

    # -------------------------------------------------------------------------
    # apply 38 kHz calibration parameters
    if calfile is not None:
        params = readCAL.ices(calfile, channel)

    # -------------------------------------------------------------------------
    # correct data for speed of sound and absorption
    if 'params' not in locals():
        class params(object):
            pass
    if soundspeed is not None:
        params.sound_velocity = soundspeed
    if absorption is not None:
        params.absorption_coefficient = absorption
    # print(params)
    # -------------------------------------------------------------------------
    # get raw data
    # fig = figure()
    Sv = raw.get_Sv(calibration=params)
    if isbot:
        bottom = raw.get_bottom()
        bottom_depth_lists = bottom.data
    # update(fig=fig, threshold=[-70, -34], data_attribute=None, Sv=Sv)
        
        # xticks = bottom.ping_time.astype('float')
        # xticks = _adjust_xdata(xticks)
        
    # y1 = _adjust_ydata( y_data, Sv, fig)
    # Sv = np.transpose(raw.get_Sv(calibration=params).data)
    
    Sv = np.transpose(Sv.data)
    theta = np.transpose(raw.angles_alongship_e)
    phi = np.transpose(raw.angles_athwartship_e)
    t = raw.get_Sv(calibration=params).ping_time
    r = raw.get_Sv(calibration=params).range
    alpha = raw.absorption_coefficient[0]
    if isbot:
        bottom_depth_metric = np.zeros(Sv.shape)
        # for i, depth in enumerate(bottom_depth_lists):
        #     bottom[i] = depth
        for i, value_a in enumerate(bottom_depth_lists):
            # 计算 b 与当前 a 值的差值绝对值
            diff = np.abs(r - value_a)
            # 找到差值最小的索引
            min_index = np.argmin(diff)
            # 将 b 中该索引位置的值替换为 a 值
            bottom_depth_metric[min_index, i] = 1
    else:
        bottom_depth_metric = False
    # -------------------------------------------------------------------------
    # check continuity with preceeding RAW file
    if preraw is None:
        # logger.warn('no preceding RAW file')
        continuous = False
    else:
        pingrate  = np.float64(np.mean(np.diff(t)))
        timelapse = np.float64(t[0] - preraw['t'][-1])
        if timelapse <= 0:
            # logger.warn('no preceding RAW file')
            continuous = False
        else:
            if timelapse > 1.5 * pingrate:
                # logger.warn('time breach in preceding RAW file')
                continuous = False
            else:
                if preraw['r'].size != r.size:
                    # logger.warn('range discrepancy in preceding RAW file')
                    continuous = False
                else:
                    if (preraw['r'] != r).all():
                        # logger.warn('range discrepancy in preceding RAW file')
                        continuous = False
                    else:
                        continuous = True

    # -------------------------------------------------------------------------
    # get nmea data
    transect, Tpos, LON, LAT, lon, lat, nm, km, kph, knt = nmea(ek60, t, preraw=preraw)
    if preraw is not None:
        if transect != preraw['transect']:
            continuous = False

    # -------------------------------------------------------------------------
    # if not continuous, resume distances an start a new transect
    if not continuous:
        km -= np.nanmin(km)
        nm -= np.nanmin(nm)
        if preraw is not None:
            if transect == preraw['transect']:
                transect += 1

                # -------------------------------------------------------------------------
    # if stationary, turn transect number to negative & resume distances
    if (nm[-1] - nm[0]) / (np.float64(t[-1] - t[0]) / (1000 * 60 * 60)) < transitspeed:
        km -= np.nanmin(km)
        nm -= np.nanmin(nm)
        if preraw is None:
            transect = 0
        else:
            if preraw['transect'] > 0:
                transect = -preraw['transect']
            else:
                transect = preraw['transect']

    # -------------------------------------------------------------------------
    # if just went off station, go to next transect number & resume distances
    else:
        if preraw is not None:
            if (transect > 0) & (preraw['transect'] <= 0):
                transect += 1
                km -= np.nanmin(km)
                nm -= np.nanmin(nm)

    Tmot, PITCH, ROLL, HEAVE, pitch, roll, heave, pitchmax, rollmax, heavemax = motion(ek60, t, preraw=preraw)

    # -------------------------------------------------------------------------
    # delete objects to free up memory RAM
    del ek60, raw

    # -------------------------------------------------------------------------
    # return RAW data
    
    raw = {'rawfiles': [os.path.split(rawfile)[-1]],
        'continuous': continuous,
        'transect': transect,
        'Sv': Sv,
        'theta': theta,
        'phi': phi,
        'alpha': alpha,
        'r': r,
        't': t,
        'lon': lon,
        'lat': lat,
        'nm': nm,
        'km': km,
        'kph': kph,
        'knt': knt,
        'pitch': pitch,
        'roll': roll,
        'heave': heave,
        'pitchmax': pitchmax,
        'rollmax': rollmax,
        'heavemax': heavemax,
        'Tpos': Tpos,
        'LON': LON,
        'LAT': LAT,
        'Tmot': Tmot,
        'PITCH': PITCH,
        'ROLL': ROLL,
        'HEAVE': HEAVE,
        'bottom':bottom_depth_metric}
        


    return raw


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
                                       return_fields=['longitude', 'latitude'])

    # find NMEA datagrams with time, longitude, and latitude
    T, LON, LAT = None, None, None
    for k, v in GPS.items():
        if isinstance(v['time'], np.ndarray
                      ) & isinstance(v['longitude'], np.ndarray
                                     ) & isinstance(v['latitude'], np.ndarray):

            if preraw is None:
                T = v['time']
                LON = v['longitude']
                LAT = v['latitude']
                transect = 1
                continuous = False
            else:
                gpsrate = np.float64(np.mean(np.diff(v['time'])))
                timelapse = np.float64(v['time'][0] - preraw['Tpos'][-1])
                if (timelapse < 5 * gpsrate) & (timelapse > 0):
                    T = np.r_[preraw['Tpos'][-7:], v['time']]
                    LON = np.r_[preraw['LON'][-7:], v['longitude']]
                    LAT = np.r_[preraw['LAT'][-7:], v['latitude']]
                    transect = abs(preraw['transect'])
                    continuous = True
                else:
                    # logger.warn('time breach in preceding NMEA data')
                    T = v['time']
                    LON = v['longitude']
                    LAT = v['latitude']
                    transect = abs(preraw['transect']) + 1
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
            KM = np.zeros(len(LON)) * np.nan
            NM = np.zeros(len(LON)) * np.nan
            for i in range(len(LON) - 1):
                if np.isnan(LAT[i]) | np.isnan(LON[i]) | np.isnan(LAT[i + 1]) | np.isnan(LON[i + 1]):
                    KM[i + 1] = np.nan
                    NM[i + 1] = np.nan
                else:
                    KM[i + 1] = distance((LAT[i], LON[i]), (LAT[i + 1], LON[i + 1])).km
                    NM[i + 1] = distance((LAT[i], LON[i]), (LAT[i + 1], LON[i + 1])).nm

            # calculate speed in kilometers per hour and knots
            KPH = KM[1:] / (np.float64(np.diff(T)) / 1000) * 3600
            KNT = NM[1:] / (np.float64(np.diff(T)) / 1000) * 3600

            if max(KNT) > maxspeed:
                raise Exception(str('Incoherent maximum speed (>%s knts). NMEA'
                                    + ' data shouldn\'t be trusted.') % maxspeed)

            # calculate cumulated distance in kilometres and nautical miles
            KM = np.nancumsum(KM)
            NM = np.nancumsum(NM)

            # convert time arrays to timestamp floats
            epoch = np.datetime64('1970-01-01T00:00:00')
            Tf = np.float64(T - epoch)
            tf = np.float64(t - epoch)

            # match array lengths (screwed up due to cumsum & diff operations)
            KM = KM[1:]
            NM = NM[1:]
            KPH = KPH[:]
            KNT = KNT[:]
            LON = LON[1:]
            LAT = LAT[1:]
            T = T[1:]
            Tf = Tf[1:]

            # get time-position, time-distance, & time-speed interpolated functions
            fLON = interp1d(Tf, LON, bounds_error=False, fill_value='extrapolate')
            fLAT = interp1d(Tf, LAT, bounds_error=False, fill_value='extrapolate')
            fNM = interp1d(Tf, NM, bounds_error=False, fill_value='extrapolate')
            fKM = interp1d(Tf, KM, bounds_error=False, fill_value='extrapolate')
            fKPH = interp1d(Tf, KPH, bounds_error=False, fill_value='extrapolate')
            fKNT = interp1d(Tf, KNT, bounds_error=False, fill_value='extrapolate')

            # get interpolated position, distance and speed for pingtime
            lon = fLON(tf)
            lat = fLAT(tf)
            nm = fNM(tf)
            km = fKM(tf)
            kph = fKPH(tf)
            knt = fKNT(tf)

            # Cumulated distance should continue the distance array from
            # the preceeding raw file, if there is continuity
            if continuous:

                # measure file gap distances
                kmgap = distance((preraw['lat'][-1], preraw['lon'][-1]),
                                 (lat[0], lon[0])).km
                nmgap = distance((preraw['lat'][-1], preraw['lon'][-1]),
                                 (lat[0], lon[0])).nm

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
        # logger.warn('GPS data not found')

        T, LON, LAT = None, None, None
        lon = np.zeros_like(t) * np.nan
        lat = np.zeros_like(t) * np.nan
        nm = np.zeros_like(t) * np.nan
        km = np.zeros_like(t) * np.nan
        kph = np.zeros_like(t) * np.nan
        knt = np.zeros_like(t) * np.nan

    return transect, T, LON, LAT, lon, lat, nm, km, kph, knt


def motion(ek60, t, preraw=None):
    """
    Get motion data. Experimental.
    """

    # get motion datagram
    shr = ek60.nmea_data.get_datagrams('SHR', return_fields=['pitch', 'roll', 'heave'])

    # return empty if motion data not found
    if any(v is None for v in shr['SHR'].values()):
        # logger.warn('Motion data not found')
        T = None
        PITCH = None
        ROLL = None
        HEAVE = None
        pitch = np.ones(len(t)) * np.nan
        roll = np.ones(len(t)) * np.nan
        heave = np.ones(len(t)) * np.nan
        pitchmax = np.ones(len(t)) * np.nan
        rollmax = np.ones(len(t)) * np.nan
        heavemax = np.ones(len(t)) * np.nan

        return T, PITCH, ROLL, HEAVE, pitch, roll, heave, pitchmax, rollmax, heavemax

    # extract values if motion data is found
    else:

        # get only current values if there is no preceeding RAW data
        if preraw is None:
            T = shr['SHR']['time']
            PITCH = shr['SHR']['pitch']
            ROLL = shr['SHR']['roll']
            HEAVE = shr['SHR']['heave']

        # concatenate preceeding values...
        else:
            nmeapingrate = np.float64(np.mean(np.diff(shr['SHR']['time'])))
            timelapse = np.float64(shr['SHR']['time'][0] - preraw['Tmot'][-1])

            # ... if preceeding RAW data is consecutive
            if (timelapse < 5 * nmeapingrate) & (timelapse > 0):
                T = np.r_[preraw['Tmot'][-14:], shr['SHR']['time']]
                PITCH = np.r_[preraw['PITCH'][-14:], shr['SHR']['pitch']]
                ROLL = np.r_[preraw['ROLL'][-14:], shr['SHR']['roll']]
                HEAVE = np.r_[preraw['HEAVE'][-14:], shr['SHR']['heave']]

            # , get current ones otherwise
            else:
                # logger.warning('time breach in preceding MOTION data')
                T = shr['SHR']['time']
                PITCH = shr['SHR']['pitch']
                ROLL = shr['SHR']['roll']
                HEAVE = shr['SHR']['heave']

        # get maximum motion in a 10-seconds moving window (absolute values)
        PITCHdf = pd.DataFrame(abs(PITCH))
        PITCHmax = np.array(PITCHdf.rolling(window=10,
                                            center=False).max()).flatten()
        ROLLdf = pd.DataFrame(abs(ROLL))
        ROLLmax = np.array(ROLLdf.rolling(window=10,
                                          center=False).max()).flatten()
        HEAVEdf = pd.DataFrame(abs(HEAVE))
        HEAVEmax = np.array(HEAVEdf.rolling(window=10,
                                            center=False).max()).flatten()

        # convert time arrays to timestamp floats
        epoch = np.datetime64('1970-01-01T00:00:00')
        Tf = np.float64(T - epoch)
        tf = np.float64(t - epoch)

        # get time-motion interpolated functions
        fPITCH = interp1d(Tf, PITCH, bounds_error=False)
        fROLL = interp1d(Tf, ROLL, bounds_error=False)
        fHEAVE = interp1d(Tf, HEAVE, bounds_error=False)
        fPITCHmax = interp1d(Tf, PITCHmax, bounds_error=False)
        fROLLmax = interp1d(Tf, ROLLmax, bounds_error=False)
        fHEAVEmax = interp1d(Tf, HEAVEmax, bounds_error=False)

        # get ping-interpolated motion
        pitch = fPITCH(tf)
        roll = fROLL(tf)
        heave = fHEAVE(tf)
        pitchmax = fPITCHmax(tf)
        rollmax = fROLLmax(tf)
        heavemax = fHEAVEmax(tf)

        return T, PITCH, ROLL, HEAVE, pitch, roll, heave, pitchmax, rollmax, heavemax


def join(preraw, raw):
    """
    Join current and preceeding RAW data.
    """

    # -------------------------------------------------------------------------
    # join variables
    rawfiles = preraw['rawfiles'] + raw['rawfiles']
    Sv = np.c_[preraw['Sv'], raw['Sv']]
    theta = np.c_[preraw['theta'], raw['theta']]
    phi = np.c_[preraw['phi'], raw['phi']]
    t = np.r_[preraw['t'], raw['t']]
    lon = np.r_[preraw['lon'], raw['lon']]
    lat = np.r_[preraw['lat'], raw['lat']]
    nm = np.r_[preraw['nm'], raw['nm']]
    km = np.r_[preraw['km'], raw['km']]
    kph = np.r_[preraw['kph'], raw['kph']]
    knt = np.r_[preraw['knt'], raw['knt']]
    pitchmax = np.r_[preraw['pitchmax'], raw['pitchmax']]
    rollmax = np.r_[preraw['rollmax'], raw['rollmax']]
    heavemax = np.r_[preraw['heavemax'], raw['heavemax']]

    # -------------------------------------------------------------------------
    # load rest of variables
    transect = raw['transect']
    alpha = raw['alpha']
    r = raw['r']
    T = raw['Tpos']
    LON = raw['LON']
    LAT = raw['LAT']

    # -------------------------------------------------------------------------
    # return RAW data
    raw = {'rawfiles': rawfiles,
           'transect': transect,
           'Sv': Sv,
           'theta': theta,
           'phi': phi,
           'alpha': alpha,
           'r': r,
           't': t,
           'lon': lon,
           'lat': lat,
           'nm': nm,
           'km': km,
           'kph': kph,
           'knt': knt,
           'pitchmax': pitchmax,
           'rollmax': rollmax,
           'heavemax': heavemax,
           'T': T,
           'LON': LON,
           'LAT': LAT}

    return raw