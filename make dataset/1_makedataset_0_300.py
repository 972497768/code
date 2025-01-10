import sys
import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from util_raw import read, process
import matplotlib.pyplot as plt
from util_raw.cmap import cmaps, ClassColor, create_custom_cmap
import h5py
import matplotlib.pyplot as plt
# import logging, logging.config
from scipy.signal import convolve2d
from scipy.interpolate import interp1d
from echopy.utils import transform as tf
from echopy.processing import resample as rs
from echopy.processing import mask_impulse as mIN
from echopy.processing import mask_seabed as mSB
from echopy.processing import get_background as gBN
from echopy.processing import mask_signal2noise as mSN
from echopy.processing import mask_range as mRG
from echopy.processing import mask_shoals as mSH
from skimage.morphology import erosion
from skimage.morphology import dilation
from echopy.processing import resample as rs


# ##########################
# ##    Defining Function ##
# ##########################
def find_files_with_suffix(folder_path, suffix):
    """Finding files in a specific format"""
    # Get the paths of all files in a folder using the os module
    all_files = os.listdir(folder_path)
    # Filter files ending with a specified suffix
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files

def ariza_new(Sv, r, r0=10, r1=1000, roff=0, thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7)):
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

# -----------------------------------------------------------
# ##########################
# ##   define a variable  ##
# ##########################    

# # read raw
root_path = r'acoustic data'  # (Modify!) raw data - location of the folder where the source data is stored
out_path = r'dataset_0_300' # (Modify!) File location for outputting h5 format
txt_path = out_path + '\\make_dataset_report.txt'  # (Modify!) File location for outputting h5 format
png_path = os.path.join(out_path,'PNG')
choosedepth = [0, 300] # (Modify!)

# Automatic creation of output folders
try:
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    # Documentation of make_dataset_report.txt
    txt_file = open(txt_path, 'w')  # Automatic creation of output folders
    if not os.path.isdir(png_path):
        os.mkdir(png_path)
    if not os.path.isdir(os.path.join(png_path, 'PNG_{}_{}'.format(choosedepth[0], choosedepth[1]))):
        os.mkdir(os.path.join(png_path, 'PNG_{}_{}'.format(choosedepth[0], choosedepth[1])))

except:
    print("Output folder path error")
    
try:
    raw_lists = find_files_with_suffix(root_path, '.raw') # Output folder path error

    print('A total of {} documents'.format(len(raw_lists)))
except:
    print("Source data folder path error!")



# -----------------------------------------------------------
# ##############################
# ##  Production of data sets ##
# ##############################
# rawname = raw_lists[1]
for ii, rawname in enumerate(raw_lists):
    if ii <= 1:  #757
        print(ii, rawname)
        print('Reading')
        rawfile = os.path.join(root_path, rawname)

        # -------------------------------
        # Initialisation settings
        prepro = None
        jdx = [0, 0]
        
        # -------------------------------
        # Reads three frequencies
        raw120 = read.raw(rawfile) # 120kHz
        Sv120 = raw120['Sv']
        raw70 = read.raw(rawfile, channel=70)
        Sv70 = raw70['Sv']
        raw38 = read.raw(rawfile, channel=38)
        Sv38 = raw38['Sv']

        rawpile = raw120.copy()

        #########################################
        #    Creating labels and entering data  #
        #########################################
        print('Processing')
        try:
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
            
            # Copy the extension n times in the row direction and 1 time in the column direction.
            llat = np.tile(lat120,(lr,1)) 
            llon = np.tile(lon120,(lr,1))
            r_s = np.where(r120 >= choosedepth[0])[0][0]
            r_e = np.where(r120 <= choosedepth[1])[0][-1]

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

            # # Offshore surface data mask
            m120rg = mRG.outside(Sv120clean, r120, 19.9, 300) 
            
            # Masking the seabed
            m120sb = ariza_new(Sv120, r120, r0=20, r1=1000, roff=0,
                                thr=-38, ec=1, ek=(3, 3), dc=10, dk=(3, 7))
            
            # Find the indexes of all elements of a equal to 1, find the bottom line in .bot
            BL = np.argwhere(bottom == 1) 

            # Thresholding methods for detecting seafloor
            msbL = np.argwhere(m120sb== True)

            # Create an array with the same shape as the bottom array to store the distances
            new_m120sb = np.zeros_like(bottom) 

            # Identified as submarine line up to 200 wide
            for position_BL in BL:  
                x = np.argwhere(m120sb[position_BL[0]:position_BL[0]+100,position_BL[1]] == True)
                new_m120sb[position_BL[0]+x[:], position_BL[1]] = True  

            ## View Data   
            # new_m120sb[new_m120sb == 0] = np.nan
            # plt.pcolormesh(Sv120[r_s:r_e, :], vmin=-70, vmax=-34, cmap=cmaps().ek500)
            # plt.pcolormesh(new_m120sb[r_s:r_e, :])
            # plt.gca().invert_yaxis() 
            # -----------------------------------------------------
            # Getting the seabed line
            # idx = np.argmax(m120sb, axis=0)
            # sbline = r120[idx]
            # sbline[idx == 0] = np.inf
            # sbline = sbline.reshape(1, -1)
            # sbline[sbline > 250] = np.nan
            
            # Masking of unavailable ranges
            m120nu = mSN.fielding(bn120, -80)[0]

            # Removal of unwanted data (sea surface, certain depth data, seabed, unavailable ranges)
            # m120uw = m120sb
            m120sb = np.where(new_m120sb == 0, False, True)
            m120uw = m120rg|m120sb|m120nu
            Sv120clean[m120uw] = np.nan

            # Get krill mask
            k = np.ones((3, 3)) / 3 ** 2
            Sv120cvv = tf.log(convolve2d(tf.lin(Sv120clean), k, 'same', boundary='symm'))
            m120sh, m120sh_ = mSH.echoview(Sv120cvv, r120, km120 * 1000, thr=-70,
                                            mincan=(3, 15), maxlink=(3, 15), minsho=(3, 15))
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

            # # Obtain backscatter values containing only krill
            # Sv120sw = Sv120clean.copy()
            # Sv120sw[~m120sh & ~m120uw] = -999

            Sv_label = Sv120clean.copy()
            Sv_label[m120sh == True] = 1
            Sv_label[new_m120sb == True] = 2
            Sv_label[(Sv_label != 1)& (Sv_label != 2)] = 0
            Sv_label = Sv_label[r_s:r_e, :]
            
            # How is the dataset going to be selected? : Selection is done when reading the data
            image = [Sv120[r_s:r_e, :], Sv70[r_s:r_e, :], Sv38[r_s:r_e, :], rr[r_s:r_e, :], nm[r_s:r_e, :], llat[r_s:r_e, :], llon[r_s:r_e, :]]
            image = np.array(image)
            Sv_label = np.array(Sv_label)

            # plt.pcolormesh(Sv_label, cmap=cmaps().ek500)
            # plt.axis('off')
            # plt.gca().invert_yaxis()

            ############################
            #  Save as h5 format data  #
            ############################    
            print('Saving')
            # Packaged as h5 file
            f = h5py.File(out_path + '\\' + rawname.replace('.raw', '.h5'), 'w')
            f['image'] = image
            f['label'] = Sv_label
            f.close()
            # print('Saving h5 ok')

            # # draw
            # custom_cmap = create_custom_cmap(['white', 'red', 'blue'])

            # plt.pcolormesh(image[0,:,:], vmin=-80, vmax=-50, cmap=cmaps().ek500)
            # plt.axis('off')
            # plt.gca().invert_yaxis()
            # plt.savefig(os.path.join(png_path, 'PNG_{}_{}'.format(choosedepth[0], choosedepth[1]), rawname.replace('.raw','.png')), bbox_inches='tight', pad_inches=0)
            # plt.close()
            # # print('Saving Sv ok')

            # plt.pcolormesh(Sv_label,cmap=custom_cmap, vmin=0, vmax=2)
            # plt.axis('off')
            # plt.gca().invert_yaxis()
            # plt.savefig(os.path.join(png_path, 'PNG_{}_{}'.format(choosedepth[0], choosedepth[1]), rawname.replace('.raw','_label.png')), bbox_inches='tight', pad_inches=0)
            # plt.close()
            # print('Saving label ok')

        except:
            print('error'+rawname)
            txt_file.write(str(ii)+rawname+'\n')
            txt_file.flush()
            
txt_file.close()
