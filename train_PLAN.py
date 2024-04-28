import os
import os.path as osp
import sys
# if use gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.nn as gnn 
import torch_geometric.data as gdata
import torch_geometric.loader as loader

from torch.optim import lr_scheduler
from torch_geometric.utils import unbatch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

# Fixing random state for reproducibility

from utils.model_ridgecrest_vision import *
from utils.train_dataloader import *
from utils.utils import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)


def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 16, type=int, help="batch size")
    parser.add_argument("--layer_name", default = 'Trans', help="Gnn_layer_name")
    parser.add_argument('--epochs', type=int, default=2001, help='Number of epochs to train.')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--p_std_time', type=float, default=1000, help='p_std_time.')
    parser.add_argument('--s_std_time', type=float, default=1500, help='s_std_time.')
    parser.add_argument('--save_model_interval', type=int, default=100, help='save_model_interval.')
    parser.add_argument("--board_path", default='./loss_path/loss_PLAN/', help="board_path")
    parser.add_argument("--save_model_path",default='./model_path/model_weight_PLAN', help="Save_model_path")
    parser.add_argument("--save_model_name", default = 'model_', help="Save_model_name")
    parser.add_argument("--cuda",default='cpu', help="if use gpu")

    args = parser.parse_args(args=[])

    return args



class val_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y,mask):
        B ,C, D = x.size()
        mask = mask.unsqueeze(-1)
        mask_p = mask[:,0,:]
        mask_s = mask[:,1,:]
        
        ep = (x[:,0,:] - y[:,0,:]).squeeze()
        es = (x[:,1,:] - y[:,1,:]).squeeze()
        ploss = torch.sqrt(torch.sum(torch.pow(ep, 2) *  mask_p)  / (mask_p.sum()  *D))
        sloss = torch.sqrt(torch.sum(torch.pow(es, 2) *  mask_s)  / (mask_s.sum()  *D))
        
        return ploss, sloss


class dtime_loss(nn.Module):
    def __init__(self,ptime,stime):
        super().__init__()
        self.ptime = ptime / 3072
        self.stime = stime / 3072

    def forward(self,d_time, dtp, dts ,mask):
        mask_p = mask[:,0]
        mask_s = mask[:,1]
        
        ep = (dtp.reshape(-1) + d_time[:,0]/3072 - self.ptime)
        es = (dts.reshape(-1) + d_time[:,1]/3072 - self.stime)
        # ploss = torch.sqrt(torch.sum(torch.pow(ep, 2) *  mask_p) / mask_p.sum())
        # sloss = torch.sqrt(torch.sum(torch.pow(es, 2) *  mask_s) / mask_s.sum())
        ploss = torch.sqrt(torch.sum(torch.pow(ep, 2) *  mask_p))
        sloss = torch.sqrt(torch.sum(torch.pow(es, 2) *  mask_s))        
        return ploss, sloss

    
def main(args):
    
    print(args.board_path)
    print(args.layer_name)
    #load data
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    edgefilepath = './data/edge_index.npy'
    filename = os.listdir('./data/train_sample/data')
    print(filename)
    inputpath = './data/train_sample/'
    
    writer = SummaryWriter(args.board_path)
    # load model
    device = torch.device(args.cuda)
    model = Main_GCNN(args.layer_name).to(device)
    torch.cuda.empty_cache()

    load_model_name = './model/model_PLAN_Ridge_continue.pt'
    model = load_model(load_model_name, model)
    
    lr_list = []
    print('learning rate = {}'.format(args.LR))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma = 0.9)
    epoch_total = args.epochs
    loss_total = np.zeros(epoch_total)
    loss_total_test = np.zeros(epoch_total)
    loc_loss_total = np.zeros(epoch_total)
    loc_loss_total_test = np.zeros(epoch_total)
    depth_loss_total = np.zeros(epoch_total)
    depth_loss_total_test = np.zeros(epoch_total)
    p_loss_total = np.zeros(epoch_total)
    s_loss_total = np.zeros(epoch_total)
    p_loss_total_test = np.zeros(epoch_total)
    s_loss_total_test = np.zeros(epoch_total)
    p_dt_total = np.zeros(epoch_total)
    s_dt_total = np.zeros(epoch_total)
    p_dt_total_test = np.zeros(epoch_total)
    s_dt_total_test = np.zeros(epoch_total)
    
    
    
    print('Finished data and model praparing , Begin Training!')
    start = time.time()      
    criterion = val_loss()
    criterion_depth = nn.MSELoss(reduction='sum').cuda()
    criterion_loc = nn.MSELoss(reduction='sum').cuda()
    criterion_dtime = dtime_loss(args.p_std_time,args.s_std_time)
    print('Finished data praparing')
    trainfilename = filename[0:2]
    testfilename = filename[2:4]
    for epoch in range(epoch_total):
        
        model.train()
        loss_all = 0
        acc_p_train_10 = 0
        acc_s_train_10 = 0
        acc_p_train_20 = 0
        acc_s_train_20 = 0
        acc_p_train_30 = 0
        acc_s_train_30 = 0
        acc_p_train_40 = 0
        acc_s_train_40 = 0
        acc_p_train_50 = 0
        acc_s_train_50 = 0
        acc_p_test_10 = 0
        acc_s_test_10 = 0
        acc_p_test_20 = 0
        acc_s_test_20 = 0
        acc_p_test_30 = 0
        acc_s_test_30 = 0
        acc_p_test_40 = 0
        acc_s_test_40 = 0
        acc_p_test_50 = 0
        acc_s_test_50 = 0
        
        
        
        
        
        # trainfilename_part = random.sample(trainfilename, 64*32)
        # testfilename = filename[60000:60000+32]
        # trainfilename_part = random.sample(trainfilename, 32)        
        # trainfilename_part = filename[0:32]                   
        # testfilename = filename[50:82]
        # inputdata_dataset = MyGNNDataset_npy_noise(trainfilename,edgefilepath,inputpath)

        inputdata_dataset = PLAN_Dataset_train(trainfilename, edgefilepath, inputpath)
        test_dataset = PLAN_Dataset_train(testfilename,edgefilepath,inputpath)
        train_loader = gdata.DataLoader(inputdata_dataset, shuffle = True, batch_size=args.batch_size,num_workers = 0)
        test_loader = gdata.DataLoader(
            test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=0)

        print(inputdata_dataset)
        print(test_dataset)
        
        trainbt = len(train_loader)
        testbt = len(test_loader)
        
        loc_epoch = 0
        loc_test_epoch = 0
        dep_epoch = 0
        dep_test_epoch = 0
        ploss_epoch = 0
        sloss_epoch = 0
        ploss_test_epoch = 0
        sloss_test_epoch = 0
        pdt_epoch = 0
        sdt_epoch = 0
        pdt_test_epoch = 0
        sdt_test_epoch = 0
        
        torch.cuda.empty_cache()
        
        
        
        for mydata in train_loader:
            
            mydata = mydata.to(device)
            optimizer.zero_grad()
            outputgraph_p, outputgraph_s, pred_loc, pred_depth, dtp, dts,_ = model(
                mydata.x, mydata.station_loc, mydata.batch, mydata.edge_index)
            outputgraph = torch.stack([outputgraph_p.squeeze(),outputgraph_s.squeeze()],dim = 1)      
            ploss,sloss = criterion(outputgraph, mydata.y,mydata.train_mask)
            dtp_loss,dts_loss = criterion_dtime(mydata.d_time,dtp,dts,mydata.train_mask)
            loc_loss = criterion_loc(pred_loc.squeeze(),mydata.st_dis)
            depth_loss = criterion_depth(pred_depth.squeeze(),mydata.st_dep)
            
            total_loss = ploss + sloss + loc_loss+ depth_loss + dtp_loss + dts_loss
            total_loss.backward()
            loss_all += total_loss.item()
            optimizer.step()
            
            accp,accs = acc(mydata.y,outputgraph,10)
            acc_p_train_10 += accp
            acc_s_train_10 += accs
            accp,accs = acc(mydata.y,outputgraph,20)
            acc_p_train_20 += accp
            acc_s_train_20 += accs
            accp,accs = acc(mydata.y,outputgraph,30)
            acc_p_train_30 += accp
            acc_s_train_30 += accs
            accp,accs = acc(mydata.y,outputgraph,40)
            acc_p_train_40 += accp
            acc_s_train_40 += accs
            accp,accs = acc(mydata.y,outputgraph,50)
            acc_p_train_50 += accp
            acc_s_train_50 += accs

            ploss_epoch += ploss.item()
            sloss_epoch += sloss.item()
            pdt_epoch += dtp_loss.item()
            sdt_epoch += dts_loss.item()
            loc_epoch += loc_loss.item()
            dep_epoch += depth_loss
            # torch.cuda.empty_cache()
            
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        end = time.time()
        torch.cuda.empty_cache()

        model.eval()
        loss_all_test = 0

        for data in test_loader:
            with torch.no_grad():
                data = data.to(device)
                optimizer.zero_grad()
                outputgraph_p, outputgraph_s, pred_loc, pred_depth, dtp, dts,_ = model(
                    data.x, data.station_loc, data.batch, data.edge_index)
                outputgraph = torch.stack([outputgraph_p.squeeze(),outputgraph_s.squeeze()],dim = 1)   
                ploss_test,sloss_test = criterion(outputgraph, data.y,data.train_mask)
                dtp_loss_test,dts_loss_test = criterion_dtime(data.d_time,dtp,dts,data.train_mask)
                loc_loss_test = criterion_loc(pred_loc.squeeze(),data.st_dis)
                if pred_depth.squeeze().shape == data.st_dep.shape:
                    depth_loss_test = criterion_depth(pred_depth.squeeze(),data.st_dep)
                else:
                    depth_loss_test = 0
                    
                total_loss_test = ploss_test + sloss_test + loc_loss_test + depth_loss_test + dtp_loss_test + dts_loss_test
                
                loss_all_test += total_loss_test.item()
                accp,accs = acc(data.y,outputgraph,10)
                acc_p_test_10 += accp
                acc_s_test_10 += accs
                accp,accs = acc(data.y,outputgraph,20)
                acc_p_test_20 += accp
                acc_s_test_20 += accs
                accp,accs = acc(data.y,outputgraph,30)
                acc_p_test_30 += accp
                acc_s_test_30 += accs
                accp,accs = acc(data.y,outputgraph,40)
                acc_p_test_40 += accp
                acc_s_test_40 += accs
                accp,accs = acc(data.y,outputgraph,50)
                acc_p_test_50 += accp
                acc_s_test_50 += accs
                ploss_test_epoch += ploss_test.item()
                sloss_test_epoch += sloss_test.item()
                pdt_test_epoch += dtp_loss_test.item()
                sdt_test_epoch += dts_loss_test.item()
                loc_test_epoch += loc_loss_test.item()
                dep_test_epoch += depth_loss_test
                
            # torch.cuda.empty_cache()
            

        scheduler.step()
        loss_total[epoch] = loss_all/trainbt
        loss_total_test[epoch] =  loss_all_test/testbt
        loc_loss_total[epoch] = loc_epoch/trainbt
        loc_loss_total_test[epoch] = loc_test_epoch/testbt
        depth_loss_total[epoch] = dep_epoch/trainbt
        depth_loss_total_test[epoch] = dep_test_epoch/testbt
        p_loss_total[epoch] =        ploss_epoch/trainbt
        s_loss_total[epoch] =        sloss_epoch/trainbt
        p_loss_total_test[epoch] =   ploss_test_epoch / testbt
        s_loss_total_test[epoch] =   sloss_test_epoch / testbt
        p_dt_total[epoch] = pdt_epoch/trainbt
        s_dt_total[epoch] = sdt_epoch/trainbt
        p_dt_total_test[epoch] = pdt_test_epoch / testbt
        s_dt_total_test[epoch] = sdt_test_epoch / testbt
        
        
        acc_p_train_10 = acc_p_train_10 / trainbt
        acc_s_train_10 = acc_s_train_10 / trainbt
        acc_p_train_20 = acc_p_train_20 / trainbt
        acc_s_train_20 = acc_s_train_20 / trainbt
        acc_p_train_30 = acc_p_train_30 / trainbt
        acc_s_train_30 = acc_s_train_30 / trainbt
        acc_p_train_40 = acc_p_train_40 / trainbt
        acc_s_train_40 = acc_s_train_40 / trainbt
        acc_p_train_50 = acc_p_train_50 / trainbt
        acc_s_train_50 = acc_s_train_50 / trainbt
        acc_p_test_10 = acc_p_test_10 / testbt
        acc_s_test_10 = acc_s_test_10 / testbt
        acc_p_test_20 = acc_p_test_20 / testbt
        acc_s_test_20 = acc_s_test_20 / testbt
        acc_p_test_30 = acc_p_test_30 / testbt
        acc_s_test_30 = acc_s_test_30 / testbt
        acc_p_test_40 = acc_p_test_40 / testbt
        acc_s_test_40 = acc_s_test_40 / testbt
        acc_p_test_50 = acc_p_test_50 / testbt
        acc_s_test_50 = acc_s_test_50 / testbt
        
        torch.cuda.empty_cache()
        
        
        writer.add_scalars('pick_loss',{
                                   'p_loss': p_loss_total[epoch],
                                   's_loss': s_loss_total[epoch],
                                   'p_loss_test': p_loss_total_test[epoch],
                                   's_loss_test': s_loss_total_test[epoch]}, epoch)
        
        writer.add_scalars('loss',{'train': loss_total[epoch],
                           'test': loss_total_test[epoch]}, epoch)
    
    
        writer.add_scalars('dt_loss',{
                           'pdt_loss': p_dt_total[epoch],
                           'sdt_loss': s_dt_total[epoch],
                           'pdt_loss_test': p_dt_total_test[epoch],
                           'sdt_loss_test': s_dt_total_test[epoch]}, epoch)

        
        writer.add_scalars('locloss',{'train': loc_loss_total[epoch],
                                   'test': loc_loss_total_test[epoch]}, epoch)    
        
        writer.add_scalars('depthloss',{'train': depth_loss_total[epoch],
                                   'test': depth_loss_total_test[epoch]}, epoch)      
        
        
        writer.add_scalars('acc_p_10',{'train': acc_p_train_10,
                                  'test': acc_p_test_10
                                  # ,'val': acc_p_val
                                   }, epoch)
            
        writer.add_scalars('acc_s_10',{'train': acc_s_train_10,
                                  'test': acc_s_test_10
                                    # ,'val' : acc_s_val
                                   }, epoch)
        
        
        writer.add_scalars('acc_p_20',{'train': acc_p_train_20,
                                  'test': acc_p_test_20
                                  # ,'val': acc_p_val
                                   }, epoch)
            
        writer.add_scalars('acc_s_20',{'train': acc_s_train_20,
                                  'test': acc_s_test_20
                                    # ,'val' : acc_s_val
                                   }, epoch)
        
        
        writer.add_scalars('acc_p_30',{'train': acc_p_train_30,
                                  'test': acc_p_test_30
                                  # ,'val': acc_p_val
                                   }, epoch)
            
        writer.add_scalars('acc_s_30',{'train': acc_s_train_30,
                                  'test': acc_s_test_30
                                    # ,'val' : acc_s_val
                                   }, epoch)
        
        
        writer.add_scalars('acc_p_40',{'train': acc_p_train_40,
                                  'test': acc_p_test_40
                                  # ,'val': acc_p_val
                                   }, epoch)
            
        writer.add_scalars('acc_s_40',{'train': acc_s_train_40,
                                  'test': acc_s_test_40
                                    # ,'val' : acc_s_val
                                   }, epoch)
        
        
        writer.add_scalars('acc_p_50',{'train': acc_p_train_50,
                                  'test': acc_p_test_50
                                  # ,'val': acc_p_val
                                   }, epoch)
            
        writer.add_scalars('acc_s_50',{'train': acc_s_train_50,
                                  'test': acc_s_test_50
                                    # ,'val' : acc_s_val
                                   }, epoch)
        
        if epoch%1==0:
            print('Epoch: {:04d}, Acc_p_train: {:.5f} , Acc_p_test: {:.5f},Acc_s_train: {:.5f} , Acc_s_test: {:.5f}, Time:{:.6f}, lr:{:.10f}'.
                  format(epoch, acc_p_train_20, acc_p_test_20, acc_s_train_20, acc_s_test_20, end-start, lr_list[epoch]))
        if epoch%(args.save_model_interval)==0:
            save_model_name = args.save_model_name + str(epoch) + '.pt'
            save_model(args.save_model_path,save_model_name,model,optimizer,epoch)
    print('Finished Training!')    
    print(args.save_model_name)
    writer.close()
    

def acc(label,predictions,threshold):
    # predictions = torch.cat((predictions[0],predictions[1]),dim=1)
    label = label.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    diff_s = np.zeros(len(label))
    diff_p = np.zeros(len(label))
    for i in range(len(label)):
        if np.argmax(label[i,0,:]) != 0:
            diff_p[i] = np.argmax(label[i,0,:]) - np.argmax(predictions[i,0,:])
        else:
            diff_p[i] = 10000
        if np.argmax(label[i,1,:]) != 0:
            diff_s[i] =  np.argmax(label[i,1,:]) - np.argmax(predictions[i,1,:])
        else:
            diff_s[i] = 10000
            
    abs_p = np.abs(diff_p)
    abs_s = np.abs(diff_s)
    
    accs = sum(abs_s < threshold)/(len(label)-sum(abs_s == 10000))
    accp = sum(abs_p < threshold)/(len(label)-sum(abs_p == 10000))
    snumber = (len(label)-sum(abs_s == 10000))
    pnumber = (len(label)-sum(abs_p == 10000))
    return accp,accs    
    

                        
                        
                        

if __name__ == '__main__':
    args = read_args()
    main(args)

    