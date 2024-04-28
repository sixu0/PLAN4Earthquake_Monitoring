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


    
class PLAN_Dataset_train(gdata.Dataset):
    def __init__(self,filename,edgefilepath,inputpath,ifrandom = False):
        gdata.Dataset.__init__(self)
        self.filename = filename
        self.edgefilepath = edgefilepath
        self.inputpath = inputpath
        self.ifrandom = ifrandom

    def __getitem__(self, index):
        datafile = self.inputpath + 'data/' + self.filename[index]
        data = np.load(datafile)[:,:,0:3072]
                        
        for i in range(3):
            if (self.ifrandom == True):
                data[:,i,:] = self.Z_ScoreNormalization_and_noise(data[:,i,:])
            elif (self.ifrandom == False):
                data[:,i,:] = self.Z_ScoreNormalization(data[:,i,:])
        data = torch.tensor(data,dtype = torch.float)
        labelfile = self.inputpath + 'label/' + self.filename[index]
        
        splitlabel = np.abs(np.load(labelfile)[:,:,0:3072])
        splitlabel = torch.tensor(splitlabel,dtype = torch.float)
        d_time = torch.argmax(splitlabel,dim =-1)
        mask = torch.clamp(d_time,0,1)
        train_mask = torch.zeros([16,2])
        for i in range(16):            
            if ((mask[i,0] == 1) and (mask[i,1] == 1)):
                train_mask[i,0] = 1
                train_mask[i,1] = 1
        
        data_mask = train_mask[:,0]
        data_mask = data_mask == 1
        
        splitlabel = self.gauss_filer(splitlabel,40,10)[:,:,0:3072]/10

        st_dis,station_loc = self.cal_distance(index)
        st_dis = torch.tensor(st_dis, dtype=torch.float).reshape(1,17)/100
        station_loc = torch.tensor(station_loc, dtype=torch.float)
        
        data = data[data_mask]
        splitlabel = splitlabel[data_mask]
        train_mask = train_mask[data_mask]
        mask = mask[data_mask]
        st_dis_new = st_dis[:,0:16].squeeze()[data_mask]
        st_dep_new = st_dis[:,16]
        station_loc = station_loc[data_mask]
        d_time = d_time[data_mask]
        edge = self.cal_edge(data.shape[0]).T
        edge_index = torch.tensor(edge, dtype=torch.long)
        return gdata.Data(x = data,edge_index = edge_index, y = splitlabel, mask = mask, train_mask = train_mask,data_mask = data_mask,st_dis = st_dis_new,st_dep = st_dep_new,station_loc = station_loc, d_time = d_time,
                          filename = self.filename[index]) # 

    def __len__(self):
        
        return len(self.filename)
    
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
    
    def Z_ScoreNormalization_and_noise(self,x):
        n1 = x.shape[-1]
        lx = np.linspace(-np.pi,np.pi,n1)
        for i in range(x.shape[0]):
            mu = np.average(x[i,:])
            sigma = np.std(x[i,:])
            if sigma == 0:
                sigma = 1e5
            x[i,:] = (x[i,:] - mu) / sigma
            nx = np.random.rand(n1)
            nx = 2*(nx-0.5)
            nx = np.gradient(nx)
            rx = np.sqrt(np.mean(x[i,:]**2))
            rn = np.sqrt(np.mean(nx**2))
            nr = 0.1*np.random.randint(7)+0.1
            x[i,:] = (nr*rx/rn)*nx+x[i,:]
        return x
        
    
    def gauss_filer(self,label,point,sigma):
#         label = label.unsqueeze(-2)
        gpu_signal = label.type(torch.FloatTensor)
        gauss_weight = np.arange(point)
        gauss_weight = np.exp(-(gauss_weight-gauss_weight.mean())**2/(2*sigma**2))/np.power(2*np.pi,0.5)/sigma
        gauss_weight = gauss_weight[np.newaxis,np.newaxis, :]/gauss_weight.max()
        weight = torch.tensor(gauss_weight).type(torch.FloatTensor)
#         weight = weight.repeat(1,2,1)
        split_label1 = gpu_signal[:,0,:]
        split_label2 = gpu_signal[:,1,:]
        split_label1 = F.conv1d(split_label1.unsqueeze(1),weight,padding = int(point/2)).squeeze()
        split_label2 = F.conv1d(split_label2.unsqueeze(1),weight,padding = int(point/2)).squeeze()
        result = torch.stack((split_label1,split_label2),dim = 1)
#         result = F.conv1d(gpu_signal,weight,padding = int(point/2))
        return result.squeeze()

    def cal_distance(self,index):

        station_pandas = pd.read_csv("./data/gmap-stations.txt", sep='|')
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
        phasename = './data/train_sample/location/' + \
            str(self.filename[index][0:8]) + '.txt'
        
        try:
    # 不能确定正确执行的代码
            f = pd.read_csv(phasename, sep='\\s+',header=None)
        except:
            return np.zeros(17),station_loc
        f = pd.read_csv(phasename, sep='\\s+',header=None)
        f = f.drop([1])
        source_lat = f[4].values
        source_lon = f[5].values
        depth = f[6].values
        for i in range(len(station_pandas)):
            station_pandas.iloc[i,8] = geodesic((station_pandas.iloc[i,2], station_pandas.iloc[i,3]), (source_lat, source_lon)).km
        temp = np.concatenate([np.array(station_pandas['dis']),depth])
        return temp,station_loc

    