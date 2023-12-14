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


class MyGNNDataset_continue(gdata.Dataset):
    def __init__(self, edgefilepath, raw_data, left_index, right_index):
        gdata.Dataset.__init__(self)
        self.edgefilepath = edgefilepath
        self.raw_data = raw_data
        # input the list or array for each window data
        self.left_index = left_index
        self.right_index = right_index

    def __getitem__(self, index):
        #get index for each step continue data
        left_index = self.left_index[index]
        right_index = self.right_index[index]
        # get part data
        data = self.raw_data[:, :, left_index:right_index]

        # newdata = self.bandpass(data)
        newdata = self.Z_ScoreNormalization(data)
        newdata = torch.tensor(newdata, dtype=torch.float)

        station_loc = self.cal_distance(index)
        station_loc = torch.tensor(station_loc, dtype=torch.float)

        edge = self.cal_edge(data.shape[0]).T
        edge_index = torch.tensor(edge, dtype=torch.long)
        return gdata.Data(x=newdata, edge_index=edge_index, station_loc=station_loc)

    def __len__(self):

        return len(self.left_index)

    def cal_edge(self, pos_init):
        pos = np.zeros([pos_init**2, 2])
        k = 0
        for i in range(pos_init**2):
            pos[i, 0] = int(i/pos_init)
            pos[i, 1] = int(i % pos_init)
        return pos

    def Z_ScoreNormalization(self, x):
        newdata = np.ones_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                newdata[i, j, :] = (
                    x[i, j, :] - x[i, j, :].mean()) / x[i, j, :].std()
        return newdata

    def bandpass(self, x):
        newdata = np.ones_like(x)
        for i in range(16):
            for j in range(3):
                temp = x[i, j, :] - x[i, j, :].mean()
                temp = detrend(temp)
                newdata[i, j, :] = bandpass(
                    temp, 1, 15, 100, corners=10, zerophase=True)
        return newdata

    def cal_distance(self, index):

        station_pandas = pd.read_csv(
            "/home/xsi/Graph/Phase/GraphPhaseNet/revised8_locpick_ali/utils/gmap-stations.txt", sep='|')
        station_pandas = station_pandas.drop([0])
        station_pandas.columns = ['Network', 'Station', 'Latitude',
                                  'Longitude', 'Elevation', 'Sitename', 'StartTime', 'EndTime']
        station_pandas['dis'] = 0

        station_loc = np.array(station_pandas.iloc[:, 2:5])
        Latitude_MAX = station_loc[:, 0].max()
        Latitude_MIN = station_loc[:, 0].min()
        Longtitude_MAX = station_loc[:, 1].max()
        Longtitude_MIN = station_loc[:, 1].min()
        Elevation_MAX = station_loc[:, 2].max()
        station_loc[:, 0] = (station_loc[:, 0] - Latitude_MIN) / \
            (Latitude_MAX - Latitude_MIN)
        station_loc[:, 1] = (station_loc[:, 1] - Longtitude_MIN) / \
            (Longtitude_MAX - Longtitude_MIN)
        station_loc[:, 2] = (station_loc[:, 2])/Elevation_MAX

        return station_loc
