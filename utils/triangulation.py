###############################################
# @Author  : Xu Si and Jun Zhu
# @Affiliation  : University of Science and Technolog of China
# @Email   : xusi@mail.ustc.edu.cn
# @Time    : 20/10/24
###############################################
import numpy as np
import math
import os
import glob
import pandas as pd
# stadf = pd.read_csv('./station_gcarc2center.csv')
# staloc = stadf[['lat','lon']].to_numpy()
# sourceloc = np.array([[31.3014,116.3739]])
from obspy.geodetics.base import locations2degrees as loc2deg
from obspy.geodetics.base import degrees2kilometers as deg2km
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

def simulate(staloc,sourceloc,pertubation=1e-2):
#	return np.array([deg2km(loc2deg(x[0],x[1],sourcelat,sourcelon)) for x in exmp])
    p = np.expand_dims((-.5+np.random.uniform(size=staloc.shape[0]))*2*pertubation,
            axis=1)
    x1,y1,z1 = trans_xy(staloc)
    x2,y2,z2 = trans_xy(sourceloc)
    diff = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5 + p
    # diff = staloc-sourceloc+p
    return diff

def trans_xy(loc):
    x,y,z = np.zeros(len(loc)),np.zeros(len(loc)),np.zeros(len(loc))
    R = 6371009  
    for i in range(len(loc)):
        x[i] = R * math.cos(loc[i,0]*(math.pi / 180)) * math.cos((math.pi / 180) * loc[i,1])
        y[i] = R * math.cos(loc[i,0]*(math.pi / 180)) * math.sin((math.pi / 180) * loc[i,1])
        z[i] = R * math.sin(loc[i,0]*(math.pi / 180))
    return x,y,z

class Triangulate(nn.Module):
    def __init__(self,staloc,dists,dtype=torch.float32, initial_evloc=None):
        super().__init__()
        self.register_buffer('staloc', torch.tensor(staloc, dtype=dtype))
        self.register_buffer('dists', torch.tensor(dists, dtype=dtype))
        self.evloc = nn.Embedding(1,2)
                
        # Set a initial weight to make optimizer more fast and accurate
        if initial_evloc is not None:
            with torch.no_grad():
                # 使用 torch.tensor() 
                initial_evloc_tensor = torch.tensor(initial_evloc, dtype=dtype).view(1, 2)
                self.evloc.weight = nn.Parameter(initial_evloc_tensor)
        
    # def cal_dist(self,staloc,evloc,eps=1e-10):
    #     diff = staloc-evloc
    #     return torch.sqrt(torch.sum(diff**2,axis=-1)+eps)
    def cal_dist(self,staloc,evloc,eps=1e-10):
        x1,y1,z1 = self.trans_xyz(staloc)
        x2,y2,z2 = self.trans_xyz(evloc)
        distance_euclidean = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) +eps
        return distance_euclidean
    
    def trans_xyz(self,loc):
        R = 6371009  # 地球半径，单位为米
        x = R * torch.cos(loc[:,0]*(math.pi / 180)) * torch.cos((math.pi / 180) * loc[:,1])
        y = R * torch.cos(loc[:,0]*(math.pi / 180)) * torch.sin((math.pi / 180) * loc[:,1])
        z = R * torch.sin(loc[:,0]*(math.pi / 180))
        return x,y,z
        
    def forward(self,index=torch.tensor([0])):
        dist = self.cal_dist(self.staloc,self.evloc(index))
        loss = torch.sum((dist-self.dists)**2)
        return {'loss':loss}