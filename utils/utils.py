###############################################
# @Author  : Xu Si
# @Affiliation  : University of Science and Technolog of China
# @Email   : xusi@mail.ustc.edu.cn
# @Time    : 31/1/24
###############################################
# Function description
# setup_seed: fix seed
# load_model: load model weights
# load_continous_data: load continuous .sac data from Ridgecrest region (other region need some revision)
# Region_renorm_info: get renorm info of specific region
# get_repred_mask: use threshold to select the specific station corresponding to an event
# cal_edge: calculate the edge_index of differen nodes
# distance_to_time: convert time_start to datetime object
# construct_dataloader: construct dataloader of continuous waveform
# pred: the workflow how to process continous waveform using PLAN
# merge_time_series: remove the duplicate detections
# calculate_catalog_overlap: compare different catalogs using cross validation (Proposed by Prof. Zhu)
###############################################
import numpy as np
import pandas as pd
import torch_geometric.loader as loader

from tqdm import tqdm
from obspy import read
from geopy.distance import geodesic

import sys   #导入sys模块
sys.path.append("..")
from utils.detect_peaks import *
from utils.triangulation import *
from utils.continue_dataloader import *



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model(model_name,model, cpu_avaliable = True):
    if cpu_avaliable == True:
        param = torch.load(model_name,map_location='cpu')['net']
    elif cpu_avaliable == False:
        param = torch.load(model_name)['net']
    model.load_state_dict(param)
    return model


def load_continous_data(station_file_path,data_file,data_length = 3600):
    station_pandas = pd.read_csv(station_file_path, sep='|')
    station_pandas = station_pandas.drop([0])
    station_name = station_pandas.iloc[:,1].tolist()
    data = np.zeros([len(station_name),3,data_length*100+1])
    # the sort of channel in data is ZEN , little different with other phase picking method
    for index,name in tqdm(enumerate(station_name)):
        dataname1 = data_file + str(name) + '.HNZ.sac'
        dataname2 = data_file + str(name) + '.HNE.sac'
        dataname3 = data_file + str(name) + '.HNN.sac'

        tempdata1 = read(dataname1)
        tr1 = tempdata1[0][0:data_length*100+1]
        tempdata2 = read(dataname2)    
        tr2 = tempdata2[0][0:data_length*100+1]
        tempdata3 = read(dataname3)    
        tr3 = tempdata3[0][0:data_length*100+1]
        
        data[index,0,:] = tr1
        data[index,1,:] = tr2
        data[index,2,:] = tr3
    
    return data

def Region_renorm_info(station_file_path):
    station_pandas = pd.read_csv(station_file_path, sep='|')
    station_pandas = station_pandas.drop([0])
    station_pandas.columns = ['Network', 'Station', 'Latitude','Longitude', 'Elevation', 'Sitename','StartTime', 'EndTime']
    station_pandas['dis'] = 0

    station_loc = np.array(station_pandas.iloc[:,2:5])
    Latitude_MAX = station_loc[:,0].max()
    Latitude_MIN = station_loc[:,0].min()
    Longtitude_MAX = station_loc[:,1].max()
    Longtitude_MIN = station_loc[:,1].min()
    Elevation_MAX = station_loc[:,2].max()
    return Latitude_MAX,Latitude_MIN,Longtitude_MAX,Longtitude_MIN,Elevation_MAX

def get_repred_mask(ppred,spred,threshold = 0.4):
    mask = np.zeros(len(ppred))
    for i in range(len(ppred)):
        temp_pick_p = detect_peaks(ppred[i,0],mph = threshold, show=False)
        temp_pick_s = detect_peaks(spred[i,0],mph = threshold, show=False)
        # print(len(temp_pick_p[0]),len(temp_pick_s[0]))
        if (len(temp_pick_p[0]>0) or len(temp_pick_s[0]>0)):
            mask[i] = 1
        
    return mask

    
def cal_edge(pos_init):
    pos = np.zeros([pos_init**2,2])
    k = 0
    for i in range (pos_init**2):
        pos[i,0] = int(i/pos_init)
        pos[i,1] = int(i%pos_init)
    edge_index = torch.tensor(pos, dtype=torch.long)
    return edge_index

from datetime import datetime, timedelta

def distance_to_time(sample, time_start):
    # convert time_start to datetime object
    start_time = datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S.%f')
    # calculate time delta based on sample and time interval
    delta = timedelta(seconds=(sample/100))
                      
    # add time delta to start time
    end_time = start_time + delta

    # convert end time to string
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S.%f')

    return end_time_str

def construct_dataloader(data,station_file_path,start_time = 30000,end_time = 70000,interval = 500,batchsize = 1,num_workers = 1):
    # start_time means how long after 17:30:00. Since the data is 100Hz sample, 30000 means 5 minute (5*60*100)
    # end_time means how long after 17:30:00.
    # interval means shift window when deal with continous data. (Here is 5 seconds)
    
    
    left_index = np.arange(start_time,end_time,interval)
    right_index = left_index + 3072
    # not useful here
    stationfilepath = station_file_path

    inputdata_dataset = GNNDataset_ridge(stationfilepath,data,left_index,right_index)
    test_loader = loader.DataLoader(inputdata_dataset, shuffle = False, batch_size=batchsize,num_workers = num_workers)
    return test_loader,left_index



def pred(model,data_loader,station_file_path,device,batch_start_time,P_stack_value = 2.4,S_stack_value = 1.2, P_value = 0.24, S_value = 0.12, time_sample = 200, station_num = 4):
    Latitude_MAX,Latitude_MIN,Longtitude_MAX,Longtitude_MIN,Elevation_MAX = Region_renorm_info(station_file_path)
    
    m = -1
    earthquake_time_list_p = []
    earthquake_time_list_s = []
    earthquake_time_list_total = []

    earthquake_loc_list = []

    model.eval()

    for mydata in tqdm(data_loader):
    
        m += 1
        mydata = mydata.to(device)
        # First Time pred using all stations
        repred_mask = np.array([True, True,  True,  True,  True,  True,  True,  True, True, True,  True,  True,  True,  True, True, True])
        # Corresponding to Workflow (1) in Method Section of PLAN paper
        out_p,out_s,pred_loc,pred_depth,dtp,dts,_ = model(mydata.x,mydata.station_loc, mydata.batch,mydata.edge_index)
        # Corresponding to Workflow (2.1) Shift in Method Section of PLAN paper
        temp_x_shift = mydata.x[repred_mask].detach()
        temp_p_shift = out_p.clone().detach() # old pick
        temp_s_shift = out_s.clone().detach() # old pick
        #
        for i in range(repred_mask.sum()):
            temp_x_shift[i,0,:] = torch.roll(mydata.x[i,0,:], int(dts[i]*3072+1000))
            temp_x_shift[i,1,:] = torch.roll(mydata.x[i,1,:], int(dts[i]*3072+1000))
            temp_x_shift[i,2,:] = torch.roll(mydata.x[i,2,:], int(dts[i]*3072+1000))    
            temp_p_shift[i,:] = torch.roll(out_p[i,:], int(dtp[i]*3072+1000))
            temp_s_shift[i,:] = torch.roll(out_s[i,:], int(dts[i]*3072+1000))
        
        # Corresponding to Workflow (2.2) Stack in Method Section of PLAN paper
        
        out_s_sum = temp_s_shift.sum(dim=0).squeeze()
        out_p_sum = temp_p_shift.sum(dim=0).squeeze()
        # Corresponding to Workflow (2.3) Threshold Selection in Method Section of PLAN paper
        if out_s_sum.max() >S_stack_value or out_p_sum.max()>P_stack_value:
        # 1900
            temp_s_time = out_s_sum.argmax() # get argmax position
            temp_p_time = out_p_sum.argmax() # get argmax position
            train_mask = torch.zeros(mydata.x.shape[0])

            for i in range(mydata.x.shape[0]):  
                # Corresponding to Workflow (3) Stations Selection in Method Section of PLAN paper
                if out_p[i,:].max() > P_value and out_s[i,:].max() > S_value and (temp_p_shift[i,:].argmax() < temp_p_time+time_sample) and (temp_p_shift[i,:].argmax() > temp_p_time-time_sample) and (temp_s_shift[i,:].argmax() < temp_s_time+time_sample) and (temp_s_shift[i,:].argmax() > temp_s_time-time_sample):
                    train_mask[i] = 1

            train_mask = train_mask == 1
            edge_index_part = cal_edge(train_mask.sum())

            if train_mask.sum()<4:
                continue
            # repred and get final picks, offset and depth.
            out_p,out_s,pred_loc,pred_depth,dtp,dts,_ = model(mydata.x[train_mask],mydata.station_loc[train_mask], mydata.batch[train_mask],edge_index_part.T)

            list_pick_p = []
            list_pick_s = []
            temp_pick_sp = np.zeros([len(out_p),2])
            for i in range(len(out_p)):
                temp_pick_p = detect_peaks(out_p.cpu().detach().numpy()[i,0],mph=P_value, show=False)
                temp_pick_s = detect_peaks(out_s.cpu().detach().numpy()[i,0],mph=S_value, show=False)
                list_pick_s.append(temp_pick_s)
                list_pick_p.append(temp_pick_p)
                if ((len(temp_pick_p[0])>0) & (len(temp_pick_s[0])>0)):
                    temp_pick_sp[i,0] = temp_pick_p[0][temp_pick_p[1].argmax()] 
                    temp_pick_sp[i,1] = temp_pick_s[0][temp_pick_s[1].argmax()]

            # Corresponding to Workflow (4) Catalog Generation in Method Section of PLAN paper
            earthquake_stime = (temp_pick_sp[:,1] - pred_loc.squeeze().cpu().detach().numpy()*100/3.4*100).mean()
            earthquake_ptime = (temp_pick_sp[:,0] - pred_loc.squeeze().cpu().detach().numpy()*100/6*100).mean()        

            earthquake_time = (earthquake_stime + earthquake_ptime)/2
            earthquake_time = distance_to_time(batch_start_time[m] + earthquake_time, time_start = '2019-07-04 17:30:00.000')
            earthquake_ptime = distance_to_time(batch_start_time[m] + earthquake_ptime, time_start = '2019-07-04 17:30:00.000')
            earthquake_stime = distance_to_time(batch_start_time[m] + earthquake_stime, time_start = '2019-07-04 17:30:00.000')
            # Get event time
            earthquake_time_list_p.append(earthquake_ptime)
            earthquake_time_list_s.append(earthquake_stime)
            earthquake_time_list_total.append(earthquake_time)

            # Get event location
            
            mydata_renorm = mydata.station_loc[train_mask].clone().detach()

            mydata_renorm[:,0] = mydata_renorm[:,0] * (Latitude_MAX - Latitude_MIN) + Latitude_MIN
            mydata_renorm[:,1] = mydata_renorm[:,1] * (Longtitude_MAX - Longtitude_MIN) + Longtitude_MIN
            mydata_renorm[:,2] = mydata_renorm[:,2] * Elevation_MAX

            st_dis = pred_loc.clone().detach()*100

            clip_station = mydata_renorm.cpu().clone().detach()
            clip_offset = st_dis.cpu().clone().detach()
            # Only use near station for Triangulation
            clip_station = clip_station[(clip_offset<60).squeeze()]
            clip_offset = clip_offset[clip_offset<60]        

            Tri = Triangulate(clip_station,clip_offset*1000)
            opt = optim.LBFGS(params=Tri.parameters(), max_iter=2000,
                    line_search_fn='strong_wolfe')
            def invert():
                opt.zero_grad()
                loss = Tri()['loss']
                loss.backward()
                return loss
            opt.step(invert)
            evloc = Tri.evloc.weight.clone().detach()
            tmp = torch.cat([evloc,pred_depth.clone().detach().cpu()*100],axis=1).squeeze().numpy()
            earthquake_loc_list.append(tmp.tolist())
    return earthquake_time_list_p,earthquake_time_list_s,earthquake_time_list_total,earthquake_loc_list
    
    

    

############Process Catalog###################################
from datetime import datetime, timedelta

def merge_time_series(time_series, coordinates, time_type = '%Y-%m-%d %H:%M:%S.%f',time_threshold = 2):
    merged_time_series = []
    merged_coordinates = []
    current_time = datetime.strptime(time_series[0], time_type)
    current_coordinates = np.array(coordinates[0], dtype=float)
    count = 1

    for i in range(1, len(time_series)):
        timestamp = datetime.strptime(time_series[i], time_type)
        time_diff = (timestamp - current_time).total_seconds()

        if time_diff <= time_threshold:
            current_time += timedelta(seconds=time_diff)
            current_coordinates += np.array(coordinates[i], dtype=float)
            count += 1
        else:
            if count == 1:  # 处理没有重复事件的情况
                merged_time_series.append(current_time.strftime(time_type))
                merged_coordinates.append(current_coordinates)
            else:
                previous_time = datetime.strptime(time_series[i-2], time_type)
                # current_time = datetime.strptime(time_series[i-1], time_type)
                merged_time_series.append(current_time.strftime(time_type))
                merged_coordinates.append(current_coordinates / count)
            current_time = timestamp
            current_coordinates = np.array(coordinates[i], dtype=float)
            count = 1

    # Add the last merged entry
    merged_time_series.append(current_time.strftime(time_type))
    merged_coordinates.append(current_coordinates / count)

    return merged_time_series, np.array(merged_coordinates)


# # Example of merge_time_series
# time_series = [
#     '2019-07-04 17:35:04.988137',
#     '2019-07-04 17:35:05.871854',
#     '2019-07-04 17:35:58.692799',
#     '2019-07-08 23:58:24.372852',
#     '2019-07-08 23:58:38.239662',
#     '2019-07-09 00:00:02.190733'
# ]

# coordinates = [
#     [1.0, 2.0, 3.0],
#     [2.0, 3.0, 4.0],
#     [3.0, 4.0, 5.0],
#     [4.0, 5.0, 6.0],
#     [5.0, 6.0, 7.0],
#     [6.0, 7.0, 8.0]
# ]

# merged_time_series, merged_coordinates = merge_time_series(time_series, coordinates)

# for t, c in zip(merged_time_series, merged_coordinates):
#     print(f'Time: {t}, Coordinates: {c}')
    
    
    
def calculate_catalog_overlap(true_labels, predicted_labels, time_threshold=3,true_type = '%Y/%m/%d,%H:%M:%S.%f',pred_type = '%Y-%m-%d %H:%M:%S.%f'):
    true_labels = [datetime.strptime(timestamp, true_type) for timestamp in true_labels]
    predicted_labels = [datetime.strptime(timestamp, pred_type) for timestamp in predicted_labels]
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    time_difference_list = []
    for true_label in tqdm(true_labels):
        matched = False
        for predicted_label in predicted_labels:
            time_difference = abs((true_label - predicted_label).total_seconds())
            if time_difference <= time_threshold:
                time_difference_list.append(time_difference)
                true_positives += 1
                matched = True
                predicted_labels.remove(predicted_label)
                break
        if not matched:
            false_negatives += 1
    
    false_positives = len(predicted_labels)
    # print(true_positives,false_positives)
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score
