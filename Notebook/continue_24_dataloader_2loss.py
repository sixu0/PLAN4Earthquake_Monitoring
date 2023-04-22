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
    
def readdata(datafile,labelfile,total_filename,filenumber):
    data_index = 0
    temp_data = np.zeros([600000,3,3001])
    temp_label = np.zeros([600000,2,3001])
    for i in range(filenumber):
        sys.stdout.write('\r计算速度:{:.2%}'.format(i/filenumber))
        sys.stdout.flush()
#         file = '/home/ysi/Seismic_location/phasepick/rawdata/data/ridgecrest/select_npzdata_new/' + total_filename[i]
        datafilename = datafile + total_filename[i]
        labelfilename = labelfile + total_filename[i]
    
        data = np.load(datafilename)
        label = np.load(labelfilename)
        if (data.shape[0] == label.shape[0]):
            x = 16
            temp_data[data_index:data_index+x,:,:] = data[:,:,0:3001]
            temp_label[data_index:data_index+x,:,:] = label[:,:,0:3001]
            data_index += x
    print('\r计算速度:{:.2%}'.format(1))
    train_data = temp_data[0:data_index]
    train_label = temp_label[0:data_index]
    return train_data,train_label
 
    
class MyGNNDataset_continue(gdata.Dataset):
    def __init__(self,edgefilepath,raw_data,left_index,right_index):
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
        data = self.raw_data[:,:,left_index:right_index]
                        
        for i in range(3):
            data[:,i,:] = self.Z_ScoreNormalization(data[:,i,:])
        data = torch.tensor(data,dtype = torch.float)
        

        station_loc = self.cal_distance(index)
        station_loc = torch.tensor(station_loc, dtype=torch.float)
        
        edge = self.cal_edge(data.shape[0]).T
        edge_index = torch.tensor(edge, dtype=torch.long)
        return gdata.Data(x = data,edge_index = edge_index,station_loc = station_loc) # 

    def __len__(self):
        
        return len(self.left_index)

    
    
    def cal_edge(self,pos_init):
        # pos = np.zeros([pos_init*(pos_init-1),2])
        # k = 0
        # for i in range (pos_init**2):
        #     if int(i/pos_init) != int(i%pos_init):
        #         pos[k,0] = int(i/pos_init)
        #         pos[k,1] = int(i%pos_init)
                # k += 1
        pos = np.zeros([pos_init**2,2])
        k = 0
        for i in range (pos_init**2):
            pos[i,0] = int(i/pos_init)
            pos[i,1] = int(i%pos_init)
        return pos
    
    def Z_ScoreNormalization(self,x):
        for i in range(x.shape[0]):
            mu = np.average(x[i,:])
            sigma = np.std(x[i,:])
            if sigma == 0:
                sigma = 1e5
            x[i,:] = (x[i,:] - mu) / sigma
        return x
    
    def cal_distance(self,index):

        station_pandas = pd.read_csv("./gmap-stations.txt", sep='|')
        station_pandas = station_pandas.drop([0])
        station_pandas.columns = ['Network', 'Station', 'Latitude','Longitude', 'Elevation', 'Sitename','StartTime', 'EndTime']
        station_pandas['dis'] = 0

        station_loc = np.array(station_pandas.iloc[:,2:5])
        Latitude_MAX = station_loc[:,0].max()
        Latitude_MIN = station_loc[:,0].min()
        Longtitude_MAX = station_loc[:,1].max()
        Longtitude_MIN = station_loc[:,1].min()
        Elevation_MAX = station_loc[:,2].max()
        station_loc[:,0] = (station_loc[:,0]- Latitude_MIN)/(Latitude_MAX - Latitude_MIN)
        station_loc[:,1] = (station_loc[:,1]- Longtitude_MIN)/(Longtitude_MAX - Longtitude_MIN)
        station_loc[:,2] = (station_loc[:,2])/Elevation_MAX 
        # phasename = '../revised5_stead/dtw_multistaion_phase/' + str(self.filename[index][0:8]) +'.txt'
        
    #     try:
    # # 不能确定正确执行的代码
    #         f = pd.read_csv(phasename, sep='\\s+',header=None)
    #     except:
    #         return np.zeros(17),station_loc
    #     f = pd.read_csv(phasename, sep='\\s+',header=None)
    #     f = f.drop([1])
    #     source_lat = f[4].values
    #     source_lon = f[5].values
    #     depth = f[6].values
    #     for i in range(len(station_pandas)):
    #         station_pandas.iloc[i,8] = geodesic((station_pandas.iloc[i,2], station_pandas.iloc[i,3]), (source_lat, source_lon)).km
    #     temp = np.concatenate([np.array(station_pandas['dis']),depth])
        return station_loc
