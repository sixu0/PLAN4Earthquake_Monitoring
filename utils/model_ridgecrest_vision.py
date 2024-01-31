import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn 
from torch_geometric.utils import unbatch
from torch_geometric.nn import global_mean_pool as gap
    
    
def save_model(path,modelname,model,optimizer,epoch):
    path = os.path.join(path,modelname)
    state = {'net': model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch':epoch}
    torch.save(state, path)
    

    
    
####################################################
#######################conv#########################
####################################################
####################################################
#######################conv#########################
####################################################


    
class Conv_downsample(nn.Module):
    def __init__(self,in_ch,out_ch,ifpool=False,poolstride = 2):
        super(Conv_downsample, self).__init__()  
        self.ifpool = ifpool
        layers = [
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_ch,out_channels=out_ch,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        ]
        self.multiconv=nn.Sequential(*layers) 
        self.downsample=nn.Sequential(
            nn.MaxPool1d(5, stride=poolstride,padding = 2)
        )
    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        # x = x.unsqueeze(0)
        out=self.multiconv(x)
        if self.ifpool == True:
            poolout = self.downsample(out)
            return out,poolout
        return out
    

class Conv_upsample(nn.Module):
    def __init__(self,in_ch,out_ch,poolstride = 2):
        super(Conv_upsample, self).__init__()  
        layers = [
            nn.ConvTranspose1d(in_channels=in_ch,out_channels=out_ch,kernel_size=7,stride=poolstride,padding=3,output_padding=(poolstride-1)),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.2),
        ]        
        self.upsample=nn.Sequential(*layers)
  
    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        # x = x.unsqueeze(0)
        out=self.upsample(x)
        
        return out 

######################################################################################################
######################################################################################################
###############################################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
############################################################################################################################################################################################################

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class location_head(torch.nn.Module):
    def __init__(self,in_ch,hid_ch,out_ch,in_feature,head = 1,drop=0):
        super(location_head, self).__init__()
        
        self.gd1 = Conv_downsample(in_ch,hid_ch,ifpool = True,poolstride = 2)
        self.gd2 = Conv_downsample(hid_ch,out_ch,ifpool = False,poolstride = 2)
        
        # self.mlp1 = Mlp(int(16*out_ch*in_feature/2),1024,512)
        # self.mlp2 = Mlp(512,256,64)
        # self.mlp3 = Mlp(64,17,17)
        
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = int(out_ch*in_feature/2), 
                   out_channels = 48,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        
        self.Graph_agg2 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 48, 
                   out_channels = 12,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        
        self.Graph_agg3 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 12, 
                   out_channels = 1,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        
        self.Graph_agg_depth = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = 12, 
                   out_channels = 1,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        
        self.staion_mlp = Mlp(3,48,96)
        self.total_mlp = Mlp(192,96,96)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x,station_loc, batch,edge_index):
        _,x = self.gd1(x)
        x = self.gd2(x)
        
        
        # x = torch.stack(unbatch(x, batch))
        x = torch.flatten(x,1,-1)
        station_loc = self.staion_mlp(station_loc)
        
        x = torch.concat([x,station_loc],dim = -1)
        x = self.total_mlp(x)
        
        x_temp = self.Graph_agg1(x, edge_index)
        x = self.Graph_agg2(x_temp, edge_index)
        x_offset = self.Graph_agg3(x, edge_index)
        x_depth = self.Graph_agg_depth(x, edge_index)
        
        # x_offset = torch.stack(unbatch(x_offset, batch))
        x_depth = gap(x_depth, batch)
        # x = self.mlp1(x)
        # x = self.mlp2(x)
        # x = self.mlp3(x)
        x_offset = self.sigmoid(x_offset)
        x_depth = self.sigmoid(x_depth)        
        # x = torch.concat([x_depth,x_offset],dim = 1)  
        return x_offset,x_depth,x_temp  
        

    

class cal_dtime(torch.nn.Module):
    def __init__(self,in_ch,hid_ch,out_ch,head = 1,drop=0):
        super(cal_dtime, self).__init__()
        
        self.Graph_aggp_1 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = in_ch, 
                   out_channels = hid_ch,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        self.Graph_aggp_2 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = hid_ch, 
                   out_channels = out_ch,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        
        self.Graph_aggs_1 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = in_ch, 
                   out_channels = hid_ch,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        self.Graph_aggs_2 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = hid_ch, 
                   out_channels = out_ch,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        
        
        self.Tanh = nn.Tanh()
        
        
    def forward(self, x_offset,x_depth,batch,edge_index):
        x_offset = x_offset.reshape(-1)
        x_depth = x_depth.reshape(-1)
        batch_count = torch.bincount(batch)
        x_depth = x_depth.repeat_interleave(batch_count,dim = 0)
        x = torch.stack([x_offset,x_depth],dim = -1)
        
        x_p = self.Graph_aggp_1(x, edge_index)
        x_p = self.Graph_aggp_2(x_p, edge_index).squeeze()
        x_s = self.Graph_aggs_1(x, edge_index)
        x_s = self.Graph_aggs_2(x_s, edge_index).squeeze()
        
        x_p = self.Tanh(x_p)
        x_s = self.Tanh(x_s)
        
        return x_p,x_s
        
    
    
##########################################################################################################################################
##########################################################################################################################################
    

class Main_GCNN(torch.nn.Module):
    def __init__(self,act_layer='Trans'):
        super(Main_GCNN, self).__init__()
        
        self.gd1 = Conv_downsample(3,16,ifpool = True,poolstride=4)#3072 768
        self.gd2 = Conv_downsample(16,32,ifpool = True,poolstride=4)#768 192
        self.gd3 = Conv_downsample(32,64,ifpool = True,poolstride=4)#192 48
        self.gd4 = Conv_downsample(64,128,ifpool = True,poolstride=2)#48 24
        
        self.aux_head = location_head(128,32,8,in_feature = 24)
        self.time_head = cal_dtime(2,4,1)
        
        
        # self.mlp_dtp = Mlp(17,16,16,act_layer = nn.Tanh)
        # self.mlp_dts = Mlp(17,16,16,act_layer = nn.Tanh)
        # self.Tanh = nn.Tanh()
        
        
        self.gc1 = Conv_downsample(128,128,ifpool = False)# 24 24
        self.gu1 = Conv_upsample(128,128,poolstride=2) #24 48

        self.gc2 = Conv_downsample(256,128,ifpool = False)#48 48
        self.gu2 = Conv_upsample(128,64,poolstride=4) #48 192
        
        self.gc3 = Conv_downsample(128,64,ifpool = False)#192 192
        self.gu3 = Conv_upsample(64,32,poolstride=4) #192 768
        
        self.gc4 = Conv_downsample(64,32,ifpool = False) #768 768
        self.gu4 = Conv_upsample(32,16,poolstride=4) #768 3072     
        self.gc5 = GNN_layer_Trans(32,2, head = 1 ,drop = 0.1)#3072 3072    

        
        self.o1_p=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(1),
        )
        self.o2_p=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=7, stride=1, padding=3),
        )     
        
        self.o1_s=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(1),
        )
        self.o2_s=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=7, stride=1, padding=3),
        )       
        
    def forward(self, x,station_loc,batch,edge_index):
        # with torch.no_grad():
        x3072, out = self.gd1(x)
        x768, out = self.gd2(out)
        x192, out = self.gd3(out)
        x48, out = self.gd4(out)
        aux_out_offset,aux_out_depth,x_temp = self.aux_head(out, station_loc,batch, edge_index)
        dtp,dts = self.time_head(aux_out_offset,aux_out_depth,batch, edge_index)
                
        out=self.gc1(out)
        out=self.gu1(out)
        out = torch.cat((x48,out),dim=1) # cat  2*128*48--256*48

        out=self.gc2(out)
        out=self.gu2(out)
        out = torch.cat((x192,out),dim=1)

        out=self.gc3(out)
        out=self.gu3(out)
        out = torch.cat((x768,out),dim=1)
        # print('out before gc4',out.shape)
        out=self.gc4(out)
        # print('out before gu4',out.shape)
        out=self.gu4(out)
        # print('out before cat',out.shape)
        out = torch.cat((x3072,out),dim=1)
        # print('out before gc5',out.shape)
        out=self.gc5(out,edge_index,dtp,dts)
        # print('out before o1',out.shape)
        out_p=self.o1_p(out[:,0,:].unsqueeze(1))
        out_p=self.o2_p(out_p)
        out_s=self.o1_s(out[:,1,:].unsqueeze(1))
        out_s=self.o2_s(out_s)
        # out = torch.stack([out_p.squeeze(),out_s.squeeze()],dim = 1)      
        
        return out_p,out_s,aux_out_offset,aux_out_depth,dtp,dts,x_temp



######################################################################################################
######################################################################################################
######################################################################################################
###################################################################################################################################################################################
######################################################################################################
######################################################################################################


##########################################################################################################################################
##########################################################################################################################################





class GNN_layer_GAT(torch.nn.Module):
    def __init__(self,in_ch,out_ch,head,drop,feature = 3072,factor = 1):
        super(GNN_layer_GAT, self).__init__()
        self.factor = factor
        self.Graph_Conv = Conv_downsample(in_ch,out_ch,ifpool = False)
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (gnn.GATConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,add_self_loops = True), 'x, edge_index -> x'),
        ])
        self.Graph_agg2 = gnn.Sequential('x, edge_index', [
            (gnn.GATConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,add_self_loops = True), 'x, edge_index -> x'),
        ])
            
    def forward(self,x,edge_index,dtp,dts):
        '''
        :param x: input
        :param out: cat with the GraphupsampleLayer
        '''
        dtp = dtp.view(-1)*3072
        dts = dts.view(-1)*3072
        value = self.Graph_Conv(x)
        temp_value = value.clone()
        for i in range(temp_value.shape[0]):
            temp_value[i,0,:] = torch.roll(value[i,0,:], int(dtp[i]))
            temp_value[i,1,:] = torch.roll(value[i,1,:], int(dts[i]))
        out = temp_value.clone()
        out[:,0,:] = self.Graph_agg1(temp_value[:,0,:], edge_index)
        out[:,1,:] = self.Graph_agg2(temp_value[:,1,:], edge_index)
        out1 = out.clone()
        for i in range(out.shape[0]):
            out1[i,0,:] = torch.roll(out[i,0,:], - int(dtp[i]))
            out1[i,1,:] = torch.roll(out[i,1,:], - int(dts[i]))
        # out1 = out1/15 + value * self.factor
        return out1,out,temp_value
    
##########################################################################################################################################
##########################################################################################################################################

class GNN_layer_SAGE(torch.nn.Module):
    def __init__(self,in_ch,out_ch,head,drop,feature = 3072,factor = 1):
        super(GNN_layer_SAGE, self).__init__()
        self.factor = factor
        self.Graph_Conv = Conv_downsample(in_ch,out_ch,ifpool = False)
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (gnn.SAGEConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        self.Graph_agg2 = gnn.Sequential('x, edge_index', [
            (gnn.SAGEConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
            
    def forward(self,x,edge_index,dtp,dts):
        '''
        :param x: input
        :param out: cat with the GraphupsampleLayer
        '''
        dtp = dtp.view(-1)*3072
        dts = dts.view(-1)*3072
        value = self.Graph_Conv(x)
        temp_value = value.clone()
        for i in range(temp_value.shape[0]):
            temp_value[i,0,:] = torch.roll(value[i,0,:], int(dtp[i]))
            temp_value[i,1,:] = torch.roll(value[i,1,:], int(dts[i]))
        out = temp_value.clone()
        out[:,0,:] = self.Graph_agg1(temp_value[:,0,:], edge_index)
        out[:,1,:] = self.Graph_agg2(temp_value[:,1,:], edge_index)
        out1 = out.clone()
        for i in range(out.shape[0]):
            out1[i,0,:] = torch.roll(out[i,0,:], - int(dtp[i]))
            out1[i,1,:] = torch.roll(out[i,1,:], - int(dts[i]))
        # out1 = out1/15 + value * self.factor
        return out1
    
##########################################################################################################################################
##########################################################################################################################################




class GNN_layer_GCN(torch.nn.Module):
    def __init__(self,in_ch,out_ch,head,drop,feature = 3072,factor = 1):
        super(GNN_layer_GCN, self).__init__()
        self.factor = factor
        self.Graph_Conv = Conv_downsample(in_ch,out_ch,ifpool = False)
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (gnn.GCNConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,add_self_loops = True), 'x, edge_index -> x'),
        ])
        self.Graph_agg2 = gnn.Sequential('x, edge_index', [
            (gnn.GCNConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,add_self_loops = True), 'x, edge_index -> x'),
        ])
            
    def forward(self,x,edge_index,dtp,dts):
        '''
        :param x: input
        :param out: cat with the GraphupsampleLayer
        '''
        dtp = dtp.view(-1)*3072
        dts = dts.view(-1)*3072
        value = self.Graph_Conv(x)
        temp_value = value.clone()
        for i in range(temp_value.shape[0]):
            temp_value[i,0,:] = torch.roll(value[i,0,:], int(dtp[i]))
            temp_value[i,1,:] = torch.roll(value[i,1,:], int(dts[i]))
        out = temp_value.clone()
        out[:,0,:] = self.Graph_agg1(temp_value[:,0,:], edge_index)
        out[:,1,:] = self.Graph_agg2(temp_value[:,1,:], edge_index)
        out1 = out.clone()
        for i in range(out.shape[0]):
            out1[i,0,:] = torch.roll(out[i,0,:], - int(dtp[i]))
            out1[i,1,:] = torch.roll(out[i,1,:], - int(dts[i]))
        # out1 = out1/15 + value * self.factor
        return out1
    
##########################################################################################################################################
##########################################################################################################################################


class GNN_layer_Trans(torch.nn.Module):
    def __init__(self,in_ch,out_ch,head,drop,feature = 3072,factor = 1):
        super(GNN_layer_Trans, self).__init__()
        self.factor = factor
        self.Graph_Conv = Conv_downsample(in_ch,out_ch,ifpool = False)
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
        self.Graph_agg2 = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(in_channels = feature, 
                   out_channels = feature,        
                   heads = head,dropout = drop,root_weight = True), 'x, edge_index -> x'),
        ])
            
    def forward(self,x,edge_index,dtp,dts):
        '''
        :param x: input
        :param out: cat with the GraphupsampleLayer
        '''
        dtp = dtp.view(-1)*3072
        dts = dts.view(-1)*3072
        value = self.Graph_Conv(x)
        temp_value = value.clone()
        for i in range(temp_value.shape[0]):
            temp_value[i,0,:] = torch.roll(value[i,0,:], int(dtp[i]))
            temp_value[i,1,:] = torch.roll(value[i,1,:], int(dts[i]))
        out = temp_value.clone()
        out[:,0,:] = self.Graph_agg1(temp_value[:,0,:], edge_index)
        out[:,1,:] = self.Graph_agg2(temp_value[:,1,:], edge_index)
        out1 = out.clone()
        for i in range(out.shape[0]):
            out1[i,0,:] = torch.roll(out[i,0,:], - int(dtp[i]))
            out1[i,1,:] = torch.roll(out[i,1,:], - int(dts[i]))
        # out1 = out1/15 + value * self.factor
        return out1
    
    
    
##########################################################################################################################################
##########################################################################################################################################  


class GNN_layer_SIM(torch.nn.Module):
    def __init__(self,in_ch,out_ch,head,drop,feature = 3072,factor = 1):
        super(GNN_layer_SIM, self).__init__()
        self.factor = factor
        self.Graph_Conv = Conv_downsample(in_ch,out_ch,ifpool = False)
        self.Graph_agg1 = gnn.Sequential('x, edge_index', [
            (SIMConv(in_channels = feature, 
                   out_channels = feature,
                   start_feature = 990,
                   end_feature = 1010), 'x, edge_index -> x'),
        ])
        self.Graph_agg2 = gnn.Sequential('x, edge_index', [
            (SIMConv(in_channels = feature, 
                    out_channels = feature,
                    start_feature = 1490,
                    end_feature = 1510), 'x, edge_index -> x'),
        ])
            
        
    def forward(self,x,edge_index,dtp,dts):
        '''
        :param x: input
        :param out: cat with the GraphupsampleLayer
        '''
        dtp = dtp.view(-1)*3072
        dts = dts.view(-1)*3072
        value = self.Graph_Conv(x)
        temp_value = value.clone()
        for i in range(temp_value.shape[0]):
            temp_value[i,0,:] = torch.roll(value[i,0,:], int(dtp[i]))
            temp_value[i,1,:] = torch.roll(value[i,1,:], int(dts[i]))
        out = temp_value.clone()
        out[:,0,:] = self.Graph_agg1(temp_value[:,0,:], edge_index)
        out[:,1,:] = self.Graph_agg2(temp_value[:,1,:], edge_index)
        out1 = out.clone()
        for i in range(out.shape[0]):
            out1[i,0,:] = torch.roll(out[i,0,:], - int(dtp[i]))
            out1[i,1,:] = torch.roll(out[i,1,:], - int(dts[i]))
        # out1 = out1/15 + value * self.factor
        return out1
    
##########################################################################################################################################
##########################################################################################################################################

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, Parameter
from torch_sparse import SparseTensor, matmul , set_diag

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class SIMConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        start_feature: int,
        end_feature: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        project: bool = False,
        bias: bool = True,
        add_self_loops: bool = False,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_feature = start_feature
        self.end_feature = end_feature
        self.normalize = normalize
        self.project = project
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        # self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        # if self.root_weight:
        #     self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # if self.project:
        #     self.lin.reset_parameters()
        self.aggr_module.reset_parameters()
        # self.lin_l.reset_parameters()
        # if self.root_weight:
            # self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])
        
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")
        
        
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        # out = self.lin_l(out)

        x_r = x[1]
        out = out + x_r
        
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor, ptr: OptTensor, 
                size_i: Optional[int]) -> Tensor:
        
        alpha = torch.exp(-(x_i[:,self.start_feature:self.end_feature] - x_j[:,self.start_feature:self.end_feature]).abs().sum(dim = -1))
        
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha.unsqueeze(-1) * x_j


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
    