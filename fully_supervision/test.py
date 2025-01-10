import argparse
# torch
import torch
import torch.nn as nn
# draw and show
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.use('TkAgg')
from PIL import Image
# time
import time
# system
import csv
import h5py
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# calculate
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import prettytable as pt
# echo
from geopy.distance import distance
from scipy.signal import savgol_filter, convolve2d
from echopy.utils import transform as tf
from echopy.processing import resample as rs
from echopy.processing import mask_impulse as mIN
from echopy.processing import mask_seabed as mSB
from echopy.processing import get_background as gBN
from echopy.processing import mask_range as mRG
from echopy.processing import mask_shoals as mSH
from echopy.processing import mask_signal2noise as mSN
from skimage.morphology import erosion
from skimage.morphology import dilation
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# from networks.net_factory import net_fact
from pyproj import Geod
geod = Geod(ellps="WGS84")
from packages.echolab2.instruments import EK60  #EK80, 
from networks.net_factory import net_factory
from Metric import SegmentationMetric, calc_confusionMatrix_results
from util_raw.read import *
from util_raw.cmap import *

##########################
# parameterization 
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=r'testdata', help='Paths for training and validating datasets')
parser.add_argument('--model_pth', type=str, default=r'model_and_results\pkl\120.pkl', help='Model Parameter Save File Path')
parser.add_argument('--cuda_set', type=int,  default=1, help='Selection of training equipment (single card)cuda id')
parser.add_argument('--stride', type=int,  default=250, help='slide step')
parser.add_argument('--size', type=int,  default=256, help='Cropping size')
parser.add_argument('--in_channels', type=int,  default=7, help='Number of input channels')
parser.add_argument('--num_classes', type=int,  default=3, help='Outputs the number of classes')
parser.add_argument('--is_savepng', type=bool,  default=True, help='Whether or not to save png')
parser.add_argument('--is_saveh5', type=bool,  default=True, help='Whether or not to save h5')
parser.add_argument('--is_savecsv', type=bool,  default=True, help='Whether or not to save csv')
parser.add_argument('--choosedepth', type=list,  default=[0, 300], help='[Minimum water depth, maximum water depth]')
args = parser.parse_args()


##########################
# Defining Functions
##########################

def find_files_with_suffix(folder_path, suffix):
    # Get the paths of all files in a folder using the os module
    all_files = os.listdir(folder_path)
    # Filter files ending with a specified suffix
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files


def ariza_new(Sv, r, r0=10, r1=1000, roff=0,
          thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7)):
   
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m). 
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (int): Sv threshold above which seabed might occur (dB).
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                  of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                  of the dilation kernel.
           
    Returns:
        bool: 2D array with seabed mask.
    """
    
     # raise errors if wrong arguments
    if r0>r1:
        raise Exception('Minimum range has to be shorter than maximum range')
    
    # return empty mask if searching range is outside the echosounder range
    if (r0>r[-1]) or (r1<r[0]):
        return np.zeros_like(Sv, dtype=bool)
    
    # get indexes for range offset and range limits
    r0   = np.nanargmin(abs(r - r0))
    r1   = np.nanargmin(abs(r - r1))
    roff = np.nanargmin(abs(r - roff))
    
    # set to -999 shallow and deep waters (prevents seabed detection)
    Sv_ = Sv.copy()
    Sv_[ 0:r0, :] = -999
    Sv_[r1:  , :] = -999
    
    # return empty mask if there is nothing above threshold
    if not (Sv_>thr).any():
        
        mask = np.zeros_like(Sv_, dtype=bool)
        return mask
    
    # search for seabed otherwise    
    else:
        
        # potential seabed will be everything above the threshold, the rest
        # will be set as -999
        seabed          = Sv_.copy()
        seabed[Sv_<thr] = -999
        
        # run erosion cycles to remove fake seabeds (e.g: spikes, small shoals)
        for i in range(ec):
            seabed = erosion(seabed, np.ones(ek))
        
        # run dilation cycles to fill seabed breaches   
        for i in range(dc):
            seabed = dilation(seabed, np.ones(dk))
        
        # mask as seabed everything greater than -999 
        mask = seabed>-999        
        
        # if seabed occur in a ping...
        # idx = np.argmax(mask, axis=0)
        # for j, i in enumerate(idx):
        #     if i != 0:
                
        #         # ...apply range offset & mask all the way down 
        #         i -= roff
        #         if i<0:
        #             i = 0
        #         mask[i:, j] = True 
                
    return mask

def resultprocess(raw, msw, path, raw_name):
    t = raw['t']
    nm = raw['nm']
    r = raw['r']
    Sv = raw['Sv']
    lon = raw['lon']
    lat = raw['lat']
    Sv_sw = Sv.copy()
    Sv_sw[msw==0] = np.nan
    jdx=[0, 0]
    r_intervals  = np.array([20, 250])
    nm_intervals = np.arange(jdx[1], nm[-1],   1)
    Sv_swr, r_r, nm_r, pc_swr = rs.twod(Sv_sw, r, nm,
                                                r_intervals, nm_intervals,
                                                log=True)
    # -------------------------------------------------------------------------
    # get time resampled, interpolated from distance resampled
    epoch = np.datetime64('1970-01-01T00:00:00')
    t_f = np.float64(t - epoch)
    f = interp1d(nm, t_f)
    t_rf = f(nm_r)
    t_r = np.array(t_rf, dtype='timedelta64[ms]') + epoch
    
    t_intervalsf = f(nm_intervals)
    t_intervals = np.array(t_intervalsf, dtype='timedelta64[ms]') + epoch

    # -------------------------------------------------------------------------
    # get latitude & longitude resampled, interpolated from time resampled
    f = interp1d(t_f, lon)
    lon_r = f(t_rf)
    f = interp1d(t_f, lat)
    lat_r = f(t_rf)

    Sa_swr   = np.zeros_like(Sv_swr)*np.nan
    NASC_swr = np.zeros_like(Sv_swr)*np.nan
    
    
    for i in range(len(Sv_swr[0])):
        # if (np.isnan(sbliner[0,i])) | (sbliner[0,i]>250):
        Sa_swr  [0,i] = tf.log(tf.lin(Sv_swr[0,i])*(250-20))
        NASC_swr[0,i] = 4*np.pi*1852**2*tf.lin(Sv_swr[0,i])*(250-20)
        # else:
        #     Sa120swr  [0,i] = tf.log(tf.lin(Sv120swr[0,i])*(sbliner[0,i]-20))
        #     NASC120swr[0,i] = 4*np.pi*1852**2*tf.lin(Sv120swr[0,i])*(sbliner[0,i]-20)

    
    results = {'Time'     : np.array(t_r      , dtype=str)         ,
            'Longitude': np.round(lon_r    , 5)                 ,
            'Latitude' : np.round(lat_r    , 5)                 ,
            'Miles'    : nm_r                                   ,
            # 'Seabed'   : np.round(sbline120r , 1)                 ,
            'NASC'     : np.round(NASC_swr[0] , 2)                 ,
            # '% samples': np.round(pc_swr   , 1)                 
            }
    # results = pd.DataFrame(results, columns= ['Time'     , 'Longitude',
    #                                           'Latitude' , 'Transect' ,
    #                                           'Miles'    , 'Seabed'   ,
    #                                           'NASC'     , '% samples'])

    # print(t_r.shape)
    # print(lon_r.shape)
    # print(lat_r.shape)
    # print(nm_r.shape)
    # print(NASC_swr.shape)
    results = pd.DataFrame(results, columns=['Time', 'Longitude',
                                             'Latitude',
                                             'Miles',
                                             'NASC'])
    if not os.path.exists(path):
        os.makedirs(path)
    logname = raw_name.replace('.raw', '.csv')
    # Write results in CSV log file
    with open(os.path.join(path,logname), 'a') as f:
        results.to_csv(os.path.join(path,logname), index=False, mode='a',
                       header=f.tell()==0)


def totestdataset(raw120, raw70, raw38,  choosedepth=[0, 300], ic=[]):
    is120, is70, is38 = False, False, False
    for  c in ic:
        if c=='120':
            Sv120 = raw120['Sv']
            is120 = True
        if c =='70':
            Sv70 = raw70['Sv']
            is70 = True
        if c == '38':
            Sv38 = raw38['Sv']
            is38 = True

        if is70==False:
            Sv70 = np.zeros_like(Sv120)
        if is38==False:
            Sv38 = np.zeros_like(Sv120)


    rawpile = raw120.copy()
    # try:
    raw = rawpile
    prepro=None
    jdx=[0, 0]
    if (isinstance(prepro, dict)) & (jdx[0] >= 0):
        raise Exception('Preceeding raw data needs appropiate j indexes')

    alpha120 = raw['alpha']
    r120 = raw['r']
    t120 = raw['t']
    lon120 = raw['lon']
    lat120 = raw['lat']
    nm120 = raw['nm']
    km120 = raw['km']
    bottom = raw['bottom']
    nm, rr = np.meshgrid(nm120, r120)
    lr = len(r120)
    llat = np.tile(lat120,(lr,1)) # Copy the extension n times in the row direction and 1 time in the column direction.
    llon = np.tile(lon120,(lr,1))
    r120 = np.array(r120)
    r_s = np.where(r120 >= choosedepth[0])[0][0]
    r_e = np.where(r120 <= choosedepth[1])[0][-1]

    time0 = time.time()
    # ------------------------------------------------------------------------
    # Interference Noise Removal
    Sv120in, m120in_ = mIN.wang(Sv120, thr=(-70, -40), erode=[(3, 3)],
                                dilate=[(7, 7)], median=[(7, 7)])
    # -------------------------------------------------------------------------
    # Background noise assessment
    p120 = np.arange(len(t120))
    s120 = np.arange(len(r120))
    bn120, m120bn_ = gBN.derobertis(Sv120, s120, p120, 5, 20, r120, alpha120)
    Sv120clean = tf.log(tf.lin(Sv120in) - tf.lin(bn120))

    # -------------------------------------------------------------------------
    # Removal of low signal noise
    m120sn = mSN.derobertis(Sv120clean, bn120, thr=12)
    Sv120clean[m120sn] = -999

    # # coastal surface
    m120rg = mRG.outside(Sv120clean, r120, 19.9, 300)
    
    # Masking the seabed
    m120sb = ariza_new(Sv120, r120, r0=20, r1=1000, roff=0,
                        thr=-38, ec=1, ek=(3, 3), dc=10, dk=(3, 7))
    if bottom is not False:
        BL = np.argwhere(bottom == 1) 
        msbL = np.argwhere(m120sb== True) 
        new_m120sb = np.zeros_like(bottom) 
        for position_BL in BL: 
            x = np.argwhere(m120sb[position_BL[0]:position_BL[0]+100,position_BL[1]] == True)
            new_m120sb[position_BL[0]+x[:], position_BL[1]] = True  
    else:
        new_m120sb = m120sb

    ## View Data      
    # new_m120sb[new_m120sb == 0] = np.nan
    # plt.pcolormesh(Sv120[r_s:r_e, :], vmin=-70, vmax=-34, cmap=cmaps().ek500)
    # plt.pcolormesh(new_m120sb[r_s:r_e, :])
    # plt.gca().invert_yaxis() 
    # -----------------------------------------------------
    # get to the seabed line
    # idx = np.argmax(m120sb, axis=0)
    # sbline = r120[idx]
    # sbline[idx == 0] = np.inf
    # sbline = sbline.reshape(1, -1)
    # sbline[sbline > 250] = np.nan
    # Masking of unavailable ranges
    m120nu = mSN.fielding(bn120, -80)[0]

    # Removal of unwanted data
    m120sb = np.where(new_m120sb == 0, False, True)
    m120uw = m120rg|m120sb|m120nu
    Sv120clean[m120uw] = np.nan

    # Get krill mask
    k = np.ones((3, 3)) / 3 ** 2
    Sv120cvv = tf.log(convolve2d(tf.lin(Sv120clean), k, 'same', boundary='symm'))
    m120sh, m120sh_ = mSH.echoview(Sv120cvv, r120, km120 * 1000, thr=-70,
                                    mincan=(3, 10), maxlink=(3, 15), minsho=(3, 15))
    # if seabed occur in a ping...
    idx = np.argmax(m120sb, axis=0)
    roff = np.nanargmin(abs(r120))
    for j, i in enumerate(idx):
        if i != 0:
            # # ...apply range offset & mask all the way down 
            i -= roff
            # if i<0:
            #     i = 0
            m120sh[i:, j] = False

    time1 = time.time()
    tr_time = time1 - time0

    Sv_label = Sv120clean.copy()
    Sv_label[m120sh == True] = 1
    Sv_label[new_m120sb == True] = 2
    Sv_label[(Sv_label != 1)& (Sv_label != 2)] = 0
    Sv_label = Sv_label[r_s:r_e, :]
    
    
    image = [Sv120[r_s:r_e, :], Sv70[r_s:r_e, :], Sv38[r_s:r_e, :], rr[r_s:r_e, :], nm[r_s:r_e, :], llat[r_s:r_e, :], llon[r_s:r_e, :]]
    image = np.array(image)
    Sv_label = np.array(Sv_label)
    return image, Sv_label, tr_time


def dataset_predict(args, data):
    cuda_set = args.cuda_set
    num_classes = args.num_classes
    size = args.size
    stride = args.stride
    data_path = args.data_path
    model_pth = args.model_pth 
    is_savepng = args.is_savepng
    is_saveh5 = args.is_saveh5
    is_savecsv = args.is_savecsv
    choosedepth = args.choosedepth
    path1 , _ = os.path.split(model_pth)
    path2 , _ = os.path.split(path1)
    path3 , _ = os.path.split(path2)
    path4 , _ = os.path.split(path3)
    path5 , model_type = os.path.split(path4)
    if data:
        data_type = data
    else:
        path6 , data_type = os.path.split(path5)
    # Training equipment
    print('cuda:{}'.format(cuda_set))
    device = torch.device('cuda:{}'.format(cuda_set))
    
    ModelPath, _ = os.path.split(model_pth)
    if choosedepth:
        SavePath = os.path.join(ModelPath, 'TestResult_{}_{}'.format(choosedepth[0], choosedepth[1]))
    else:
         SavePath = os.path.join(ModelPath, 'TestResult')
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
    PngSavePath = os.path.join(SavePath, 'PNG')
    if not os.path.exists(PngSavePath):
        os.makedirs(PngSavePath)
    H5SavePath = os.path.join(SavePath, 'H5')
    if not os.path.exists(H5SavePath):
        os.makedirs(H5SavePath)

    if data_type == '120':
        idx_ch = [0, 3]
        idepth = False
        in_channels = 1
    elif data_type == '70':
        idx_ch = [1, 3]
        idepth = False
        in_channels = 1
    elif data_type == '38':
        idx_ch = [2, 3]
        idepth = False
        in_channels = 1
    elif data_type == '120+70':
        idx_ch = [0,1, 3]
        idepth = False
        in_channels = 2
    elif data_type == '120+38':
        idx_ch = [0, 2, 3]
        idepth = False
        in_channels = 2
    elif data_type == '70+38':
        idx_ch = [1, 2, 3]
        idepth = False
        in_channels = 2
    elif data_type == '120+70+38':
        idx_ch = [0, 1, 2, 3]
        idepth = False
        in_channels = 3


    # Setting up the model
    if model_pth[-3:] == 'pkl':
        model = torch.load(model_pth) 
        model.to(device)
    elif model_pth[-3:] == 'pth':
        model = net_factory(net_type=model_type, in_chns=in_channels, class_num=num_classes)
        model.to(device)
        model.load_state_dict(torch.load(model_pth, map_location='cuda:{}'.format(cuda_set))['state_dict'])
        
    # data
    raw_lists = find_files_with_suffix(data_path, '.raw')
    criterion = nn.CrossEntropyLoss()

    testResult = np.zeros((5))

    totaltestSegMetric = SegmentationMetric(num_classes)
    totaltestSegMetric.reset()
    testSegMetric = SegmentationMetric(num_classes)
    

    all_results = []
    test_loss = 0.0
    total_test_loss = 0.0
    total_ipe = 0

    TotalTime = 0
    TotalTrTime = 0
    TotalDlTime = 0
    
    # ---------------------------------------------------------------
    # Deep learning models for outcome prediction
    model.eval()
    with torch.no_grad():
        for n_raw, raw_name in enumerate(raw_lists):
            testSegMetric.reset()
            testsince = time.time()
            day = raw_name[7:9]
            ipe = 0
            ic = []
            #-----------------------
            # Select data after 21 days
            if int(day) >= 0:
                # if count<=5:
                # retrieve data
                current_raw_path = os.path.join(data_path, raw_name)
                try:
                    raws120= raw(current_raw_path, channel=120)
                    ic.append('120')
                except:
                    print('no 120 channel')
                    raws120 = False
                try:
                    raws70= raw(current_raw_path, channel=70)
                    ic.append('70')
                except:
                    print('no 70 channel')
                    raws70 =  False

                try:
                    raws38= raw(current_raw_path, channel=38)
                    ic.append('38')
                except:
                    print('no 38 channel')
                    raws38 = False

                # Producing data sets
                big_image, big_label, TraditionalTime = totestdataset(raws120, raws70, raws38, choosedepth,ic)

                big_image, big_label = np.array(big_image), np.array(big_label)
                
                big_image = big_image[idx_ch, :, :]
                # Set depth range
                if choosedepth:
                    if idepth:
                        r = big_image[idepth, :, 0]
                        r_s = np.where(r >= choosedepth[0])[0][0]
                        r_e = np.where(r <= choosedepth[1])[0][-1]
                        big_image = big_image[:, r_s:r_e, :]
                        big_label = big_label[r_s:r_e, :]
                    else:
                        r = big_image[-1, :, 0]
                        r_s = np.where(r >= choosedepth[0])[0][0]
                        r_e = np.where(r <= choosedepth[1])[0][-1]
                        big_image = big_image[:-1, r_s:r_e, :]
                        big_label = big_label[r_s:r_e, :]


                height, width = big_label.shape
                print(big_image.shape)

                current_result = np.zeros((height, width)) 
                dl_time0 = time.time()
                # Splitting 2D data by step
                for i in range(0, int(height), stride):
                    for j in range(0, int(width), stride):
                        """This is cropped as a piece to be dropped into the network, and the network is of type tensor"""
                        # |(i     , j)     (i     , j+size)|
                        # |(i+size, j)     (i+size, j+size)|
                        left = j
                        right = j + size
                        top = i
                        bottom = i + size

                        if bottom > height:  
                            bottom = height
                            top = bottom - size
                        if right > width: 
                            right = width
                            left = right - size
                        ipe += 1
                        total_ipe += 1

                        patch_img = big_image[:, top:bottom, left:right] 
                        patch_img = np.array(patch_img)

                        patch_img = torch.from_numpy(patch_img).unsqueeze(0).nan_to_num(0.0).float()
                        patch_label = big_label[top:bottom, left:right]
                        patch_label = torch.from_numpy(patch_label).unsqueeze(0).unsqueeze(0).nan_to_num(0.0).float()

                        """Formal commencement of processing:"""
                        patch_img, patch_label = patch_img.to(device), patch_label.to(device)
                        y_pred = model(patch_img)# Drop the data into the network and compute the output
                        y_out = patch_label.view((-1, patch_label.size(2)*patch_label.size(3))).long()
                        y_pred_out = y_pred.view((-1, num_classes, patch_label.size(2)*patch_label.size(3)))
                        lossV = criterion(y_pred_out, y_out)
                        test_loss += lossV.item() 
                        y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
                        patch_label = patch_label.squeeze(1).detach().cpu().numpy()
                        testSegMetric.addBatch(y_pred, patch_label)
                        totaltestSegMetric.addBatch(y_pred, patch_label)
                        

                        # Get prediction results for the current data (cluster classification results, cluster masks)
                        current_result[top:bottom, left:right] = y_pred[0,:,:]  
                
                dl_time1 = time.time()
                # ----------Calculation of losses and evaluation of indicators-------------
                test_loss = round(test_loss / ipe,4)
                metrics = calc_confusionMatrix_results(testSegMetric)

                # ----------Terminal display time-------------
                print('\n')
                epochTime = time.time() - testsince
                # TraditionalTime = traditional_time1 - traditional_time0
                DlTime = dl_time1 - dl_time0
                TotalTime = TotalTime + epochTime
                TotalTrTime = TotalTrTime + TraditionalTime
                TotalDlTime = TotalDlTime + DlTime
                # ----------Terminal display indicator accuracy results-------------
                tb = pt.PrettyTable()
                tb.field_names = ["FileName","T/V", "Loss", "Accuracy(%)", "Precision(%)", "Recall(%)","F1(%)","IoU(%)"]
                rowheads = ["Test-Mean","Test-Background","Test-Swarm", "Test-Seabed"]
                
                Total_Metric = metrics
                Total_Metric = [[round(item*100 , 4) for item in sublist] for sublist in Total_Metric]
                for index, metric in enumerate(Total_Metric):
                    current_m = metric.copy()
                    if index == 0:
                        current_m.insert(0, test_loss)
                        current_m.insert(0,rowheads[index])
                        current_m.insert(0, "{}".format(raw_name))
                        tb.add_row(current_m)
                    else:
                        current_m.insert(0, " ")
                        current_m.insert(0,rowheads[index])
                        current_m.insert(0, " ")
                        tb.add_row(current_m)
                print(tb)
                # Converting a PrettyTable to a List
                rows = tb.get_csv_string().splitlines()
                
                # Open CSV file for writing
                if is_savecsv:
                    csv_save = os.path.join(SavePath, 'TextResult_{}.csv'.format(choosedepth))
                    if not os.path.exists(SavePath):
                        with open(csv_save, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for row in rows:
                                writer.writerow(row.split(','))
                    else:
                        with open(csv_save, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for row in rows:
                                writer.writerow(row.split(','))

                pngsave1 = os.path.join(PngSavePath, raw_name.replace('.raw', '_pred.png'))
                pngsave2 = os.path.join(PngSavePath, raw_name.replace('.raw', '_label.png'))
                pngsave3 = os.path.join(PngSavePath, raw_name.replace('.raw', '_120.png'))
                # pngsave4 = os.path.join(PngSavePath, raw_name.replace('.raw', '_compare.png'))
                pngsave5 = os.path.join(PngSavePath, raw_name.replace('.raw', '_70.png'))
                pngsave6 = os.path.join(PngSavePath, raw_name.replace('.raw', '_38.png'))
                h5save = os.path.join(H5SavePath, raw_name.replace('.raw', '_pred.h5'))
                
                if is_savepng:
                    custom_cmap = create_custom_cmap(['white', 'red', 'blue'])
                    plt.figure()
                    plt.pcolormesh(current_result,cmap=custom_cmap, vmin=0, vmax=2)
                    plt.axis('off')
                    plt.gca().invert_yaxis()
                    plt.savefig(pngsave1, bbox_inches='tight', pad_inches=0)

                    plt.figure()
                    plt.pcolormesh(big_label,cmap=custom_cmap, vmin=0, vmax=2)
                    plt.axis('off')
                    plt.gca().invert_yaxis()
                    plt.savefig(pngsave2, bbox_inches='tight', pad_inches=0)

                    plt.figure()
                    plt.pcolormesh(big_image[0, :, :], vmin=-80, vmax=-50, cmap=cmaps().ek500)
                    plt.axis('off')
                    plt.gca().invert_yaxis()
                    plt.savefig(pngsave3, bbox_inches='tight', pad_inches=0)
                    plt.close("all")

                if is_saveh5:
                    h5pred = h5py.File(h5save, "w")
                    h5pred['pred'] = current_result
                    h5pred['label'] = big_label
                    h5pred['image'] = big_image
                    h5pred['r_s_e'] = [r_s, r_e]
                    h5pred.close()

    
        print()
    # ----------Calculate all data loss and evaluation metrics for the test set-------------
    # everySegMetric.addBatch(current_result, big_label)
    total_test_loss = round(total_test_loss / total_ipe,4)
    total_metrics = calc_confusionMatrix_results(totaltestSegMetric)

    # ----------Terminal display indicator accuracy results-------------
    total_tb = pt.PrettyTable()
    total_tb.field_names = ["FileName","T/V", "Loss", "Accuracy(%)", "Precision(%)", "Recall(%)","F1(%)","IoU(%)"]
    total_rowheads = ["Test-Mean","Test-Background","Test-Swarm", "Test-Seabed"]
    
    total_Total_Metric = total_metrics
    total_Total_Metric = [[round(item*100 , 4) for item in sublist] for sublist in total_Total_Metric]
    for index, total_metric in enumerate(total_Total_Metric):
        total_current_m = total_metric.copy()
        if index == 0:
            total_current_m.insert(0, total_test_loss)
            total_current_m.insert(0,total_rowheads[index])
            total_current_m.insert(0, "Total_Metric")
            total_tb.add_row(total_current_m)
        else:
            total_current_m.insert(0, " ")
            total_current_m.insert(0,total_rowheads[index])
            total_current_m.insert(0, " ")
            total_tb.add_row(total_current_m)
    print(total_tb)
    # Converting a PrettyTable to a List
    total_rows = total_tb.get_csv_string().splitlines()
    if is_savecsv:
        # Open CSV file for writing
        total_csv_save = os.path.join(SavePath, 'total_TextResult_{}.csv'.format(choosedepth))
        if not os.path.exists(SavePath):
            with open(total_csv_save, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in total_rows:
                    writer.writerow(row.split(','))
        else:
            with open(total_csv_save, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in total_rows:
                    writer.writerow(row.split(','))

    return TotalTrTime, TotalDlTime, len(raw_lists)

datas = ['120'] # , '38', '120+70', '120+38', '70', '38', '120+70', '120+38'
for data in datas:
    ts = 1
    MeanTotalTrTime, MeanTotalDlTime =  0, 0
    for t in range(ts):
        CurrentTotalTrTime, CurrentTotalDlTime, l = dataset_predict(args, data)
        MeanTotalTrTime = MeanTotalTrTime + CurrentTotalTrTime
        MeanTotalDlTime =  MeanTotalDlTime + CurrentTotalDlTime
        print('Completed {}'.format(t+1))
    print_txt = 'Average time for traditional methods: {} minutes {} seconds, average time for deep learning methods: {} minutes {} seconds'.format(
                                                int((MeanTotalTrTime/ts/l % 3600) / 60),
                                                round(MeanTotalTrTime/ts/l % 60, 2),
                                                int((MeanTotalDlTime/ts/l % 3600) / 60),
                                                round(MeanTotalDlTime/ts/l % 60, 2)
                                                    )
    print(print_txt)
    TotalSavePath = r'model_and_results'
    TotalCsvSave = os.path.join(TotalSavePath, 'TimeResult.txt')
    if not os.path.exists(TotalCsvSave):
        with open(TotalCsvSave, 'w',  encoding='utf-8') as txtfile:
            txtfile.write(data+' '+print_txt+'\n')
    else:
        with open(TotalCsvSave, 'a', encoding='utf-8') as txtfile:
            txtfile.write(data+' '+print_txt+'\n')



