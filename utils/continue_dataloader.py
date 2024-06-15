###############################################
# @Author  : Xu Si
# @Affiliation  : University of Science and Technolog of China
# @Email   : xusi@mail.ustc.edu.cn
# @Time    : 31/1/24
###############################################
import math
import torch
import sys
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch_geometric.data as gdata
from scipy.signal import hilbert
from torch.utils.data import Dataset, DataLoader
from geopy.distance import geodesic

gdata.InMemoryDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GNNDataset_ridge(gdata.Dataset):
    def __init__(self,stationfilepath,raw_data,left_index,right_index):
        gdata.Dataset.__init__(self)
        self.stationfilepath = stationfilepath
        self.raw_data = raw_data
        # input the list or array for each window data
        self.left_index = left_index
        self.right_index = right_index
        

    def __getitem__(self, index):
        #get index for each step continue data
        left_index = self.left_index[index]
        right_index = self.right_index[index]
        # get part data
        data = self.raw_data[:,:,left_index:right_index]

        # newdata = self.bandpass(data)
        newdata = self.Z_ScoreNormalization(data)
        newdata = torch.tensor(newdata,dtype = torch.float)
        

        station_loc = self.cal_distance(index)
        station_loc = torch.tensor(station_loc, dtype=torch.float)
        
        edge = self.cal_edge(data.shape[0]).T
        edge_index = torch.tensor(edge, dtype=torch.long)
        return gdata.Data(x = newdata,edge_index = edge_index,station_loc = station_loc) # 

    def __len__(self):
        
        return len(self.left_index)

    
    
    def cal_edge(self,pos_init):
        pos = np.zeros([pos_init**2,2])
        k = 0
        for i in range (pos_init**2):
            pos[i,0] = int(i/pos_init)
            pos[i,1] = int(i%pos_init)
        return pos
    
    def Z_ScoreNormalization(self,x):
        newdata = np.ones_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                newdata[i,j,:] = (x[i,j,:] - x[i,j,:].mean()) / x[i,j,:].std()
        return newdata
    
    
    
    def bandpass(self,x):
        newdata = np.ones_like(x)
        for i in range(16):
            for j in range(3):
                temp = x[i,j,:] - x[i,j,:].mean()
                temp = detrend(temp)
                newdata[i,j,:] = bandpass(temp,1,15,100,corners=10, zerophase=True)
        return newdata
    
    def cal_distance(self,index):
        
        station_pandas = pd.read_csv(self.stationfilepath, sep='|')
        station_pandas = station_pandas.drop([0])
        station_pandas.columns = ['Network', 'Station', 'Latitude','Longitude', 'Elevation', 'Sitename','StartTime', 'EndTime']
        station_pandas['dis'] = 0.0

        station_loc = np.array(station_pandas.iloc[:,2:5])
        Latitude_MAX = station_loc[:,0].max()
        Latitude_MIN = station_loc[:,0].min()
        Longtitude_MAX = station_loc[:,1].max()
        Longtitude_MIN = station_loc[:,1].min()
        Elevation_MAX = station_loc[:,2].max()
        station_loc[:,0] = (station_loc[:,0]- Latitude_MIN)/(Latitude_MAX - Latitude_MIN)
        station_loc[:,1] = (station_loc[:,1]- Longtitude_MIN)/(Longtitude_MAX - Longtitude_MIN)
        station_loc[:,2] = (station_loc[:,2])/Elevation_MAX 

        return station_loc

    
    
    
    
    
    
    
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)
from scipy.signal import sosfilt
from scipy.signal import zpk2sos,detrend

def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)