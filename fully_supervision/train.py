# pytorch
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
# draw
import matplotlib
matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# calculate
import prettytable as pt
import numpy as np
import random
from scipy.interpolate import make_interp_spline
# time
import time
import datetime
# system operation
import shutil
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from networks.net_factory import net_factory
from dataloader import MyDataset
from Metric import SegmentationMetric, calc_confusionMatrix_results

##########################
# parameterization 
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=r'crop_selectdataset_0_300_select_256_180', help='Paths for training and validating datasets')
parser.add_argument('--SavePath', type=str,  default=r'', help='Save Path')
parser.add_argument('--data_type', type=str, default='120', 
    help='Input Data Type:120,70,38,120+70,120+38,70+38,120+70+38')
parser.add_argument('--exp', type=str, default='exp', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--model', type=str, default="unet", help='Model name:unet'
                    )
parser.add_argument('--epoch', type=int, default=500, help='training round。maximum epoch number tp train')
parser.add_argument('--cuda_set', type=int,  default=1, help='Selection of training equipment (single card):cuda id')
parser.add_argument('--resume', type=str,  default=False, help='Whether or not to breakpoint training.:resume model path')
parser.add_argument('--num_classes', type=int,  default=3, help='number of classes')
parser.add_argument('--seed', type=float,  default=1307, help='random seed')
parser.add_argument('--step_save', type=int,  default=10, help='Interval of rounds for saving data')
parser.add_argument('--losstype', type=str,  default='CrossEntry', help='CrossEntry')
args = parser.parse_args()


##########################
# Defining Functions 
##########################
def find_files_with_suffix(folder_path, suffix):
    all_files = os.listdir(folder_path)
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False 
    # torch.backends.cudnn.enabled = False
    
def train_net(args):
    """Functions for performing model training"""
    data_path = args.data_path
    base_lr = args.base_lr
    batch_size = args.batch_size
    epoch = args.epoch
    cuda_set = args.cuda_set
    data_path = args.data_path
    SavePath = args.SavePath
    data_type = args.data_type
    num_classes = args.num_classes
    model_type = args.model
    resume = args.resume
    step_save = args.step_save
    losstype = args.losstype

    # Training equipment
    print('cuda:{}'.format(cuda_set))
    device = torch.device('cuda:{}'.format(cuda_set))
    
    # Input Data Mode
    if data_type == '120':
        in_channels = 1
    elif data_type == '70':
        in_channels = 1
    elif data_type == '38':
        in_channels = 1
    elif data_type == '120+70':
        in_channels = 2
    elif data_type == '120+38':
        in_channels = 2
    elif data_type == '70+38':
        in_channels = 2
    elif data_type == '120+70+38':
        in_channels = 3


    # Creating Folders and Files
    if not os.path.exists(os.path.join(SavePath, 'model_and_results')):
        os.makedirs(os.path.join(SavePath, 'model_and_results'))
    if not os.path.exists(os.path.join(SavePath,  'model_and_results',  data_type)):
        os.makedirs(os.path.join(SavePath,  'model_and_results', data_type))    
    if not os.path.exists(os.path.join(SavePath, 'model_and_results', data_type, model_type)):
        os.makedirs(os.path.join(SavePath, 'model_and_results', data_type, model_type))
    if not os.path.exists(os.path.join(SavePath,  'model_and_results', data_type,model_type, args.exp)):
        os.makedirs(os.path.join(SavePath, 'model_and_results',  data_type,model_type, args.exp))
    if not os.path.exists(os.path.join(SavePath,  'model_and_results', data_type,model_type, args.exp, str(base_lr))):
        os.makedirs(os.path.join(SavePath, 'model_and_results',  data_type,model_type, args.exp, str(base_lr)))
    modelSavePath = os.path.join(SavePath,  'model_and_results', data_type, model_type, args.exp,  str(base_lr), losstype)
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)
    log_path = os.path.join(modelSavePath , 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path) 
    writer = SummaryWriter(log_path)  # visualization
    
    # Reading data for training and validation
    trainset = MyDataset(data_path, 'train',data_type)
    validset = MyDataset(data_path, 'val', data_type)
    print('Toatal train silice is:{}, val slices is:{}'.format(len(trainset), len(validset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)
    
    # Setting up the model
    model = net_factory(net_type=model_type, in_chns=in_channels, class_num=num_classes)
    model.to(device)

    # Optimizers and Loss Functions
    if losstype == 'CrossEntry':
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),
                lr=base_lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)

    # initialization
    trainSegMetric = SegmentationMetric(num_classes)
    validSegMetric = SegmentationMetric(num_classes)
    trainMetrics = np.array([])
    validMetrics = np.array([])
    trainLossList = []
    validLossList = []
    timeElapsed = 0
    best_mF1 = 0

    # Is it a continuation of previous training
    if resume:
        try:
            checkpoint = torch.load(os.path.join(modelSavePath, 'current_model.pth'))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            init_epoch = checkpoint['epoch'] + 1
            print("===>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
            if init_epoch >= epoch:
                print('please input right epoch')
            
        except:
            print('The current folder does not have a pth file')
        
    else:
        init_epoch = 1

    # Training and validation
    for epoch_num in range(init_epoch, 1+epoch):
        # initialization
        train_loss = 0.0
        valid_loss = 0.0
        since = time.time()
        trainSegMetric.reset()
        validSegMetric.reset()

        
        #  --------Start network training-------------
        model.train()
        for i_batch, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            try:
                y_pred = model(x)
            except:
                print(i_batch, x.shape, y.shape)
            y_out = y.view((-1, y.size(2)*y.size(3))).long()
            y_pred_out = y_pred.view((-1, num_classes, y.size(2)*y.size(3)))
            lossT = criterion(y_pred_out, y_out)
            lossT.backward()
            optimizer.step()
            
            train_loss += lossT.item() * x.size(0)
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
                y = y.squeeze(1).detach().cpu().numpy()
                trainSegMetric.addBatch(y_pred, y)
                
        #  --------------------
        model.eval()
        with torch.no_grad():
            for j_batch,  (x1, y1) in enumerate(validloader):
                x1, y1 = x1.to(device), y1.to(device)
                y1_pred = model(x1)
                y1_out = y1.view((-1, y1.size(2)*y1.size(3))).long()
                y1_pred_out = y1_pred.view((-1, num_classes, y1.size(2)*y1.size(3)))
                lossV = criterion(y1_pred_out, y1_out)
                valid_loss += lossV.item() * x1.size(0)  # valid_loss += lossV.item() * x1.size(0)
                y1_pred = torch.argmax(y1_pred, dim=1).detach().cpu().numpy()
                y1 = y1.squeeze(1).detach().cpu().numpy()
                validSegMetric.addBatch(y1_pred, y1)
                

        # ----------Calculation of losses and evaluation of indicators-------------
        # Calculation of losses
        train_loss = round(train_loss / len(trainset),4)
        valid_loss = round(valid_loss / len(validset), 4)
        # Validation indicators for each category obtained by counting confusion matrices
        metrics = calc_confusionMatrix_results(trainSegMetric)
        metrics1 = calc_confusionMatrix_results(validSegMetric)
        # Losses by round
        trainLossList.append(train_loss)
        validLossList.append(valid_loss)
        # One-dimensionalization of multi-category matrices
        metrics_flatten = np.array(metrics).flatten()
        metrics1_flatten = np.array(metrics1).flatten()
        # Validation results by round
        trainMetrics = np.vstack((trainMetrics, metrics_flatten)) if len(trainMetrics) > 0 else metrics_flatten #numpy.vstack 垂直行数组堆叠
        validMetrics = np.vstack((validMetrics, metrics1_flatten)) if len(validMetrics) > 0 else metrics1_flatten
    
       
        # ----------Terminal display indicator accuracy results-------------
        print('\n')
        tb = pt.PrettyTable()
        tb.field_names = ["Epoch","T/V", "Loss", "Accuracy(%) ", "Precision(%)", "  Recall(%) ","    F1(%)  ","    IoU(%)  "]
        rowheads = ["Train-Mean","Train-Background","Train-Swarm", "Train-Seabed", "Valid-Mean", "Valid-Background", "Valid-Swarm", "Valid-Seabed"]
        
        Total_Metric = metrics + metrics1
        Total_Metric = [[round(item * 100, 2) for item in sublist] for sublist in Total_Metric]
        for index, metric in enumerate(Total_Metric):
            current_m = metric.copy()
            m_index = len(Total_Metric)/2
            if index == m_index:
                current_m.insert(0, valid_loss)
                current_m.insert(0,rowheads[index])
                current_m.insert(0, " ")
                tb.add_row([" ", " "," ","----","----", "----", "----", "----"])
                tb.add_row(current_m)
            elif index == 0:
                current_m.insert(0, train_loss)
                current_m.insert(0,rowheads[index])
                current_m.insert(0, "{}/{}".format(epoch_num, epoch))
                tb.add_row(current_m)
            else:
                current_m.insert(0, " ")
                current_m.insert(0,rowheads[index])
                current_m.insert(0, " ")
                tb.add_row(current_m)

        if (epoch_num) % 1 == 0:
            print(tb)

        # ----------Tensorboard shows loss and class average results-------------
        writer.add_scalar('info/train_loss', train_loss, epoch_num)
        writer.add_scalar('info/valid_loss', valid_loss, epoch_num)
        writer.add_scalar('info/train_maccuracy', metrics[0][0], epoch_num)
        writer.add_scalar('info/train_mprecision', metrics[0][1], epoch_num)
        writer.add_scalar('info/train_mrecall', metrics[0][2], epoch_num)
        writer.add_scalar('info/train_mf1', metrics[0][3], epoch_num)
        writer.add_scalar('info/train_miou', metrics[0][4], epoch_num)

        writer.add_scalar('info/valid_maccuracy', metrics1[0][0], epoch_num)
        writer.add_scalar('info/valid_mprecision', metrics1[0][1], epoch_num)
        writer.add_scalar('info/valid_mrecall', metrics1[0][2], epoch_num)
        writer.add_scalar('info/valid_mf1', metrics1[0][3], epoch_num)
        writer.add_scalar('info/valid_miou', metrics1[0][4], epoch_num)

        writer.add_scalar('info/valid_accuracy_swarm', metrics1[2][0], epoch_num)
        writer.add_scalar('info/valid_precision_swarm', metrics1[2][1], epoch_num)
        writer.add_scalar('info/valid_recall_swarm', metrics1[2][2], epoch_num)
        writer.add_scalar('info/valid_f1_swarm', metrics1[2][3], epoch_num)
        writer.add_scalar('info/valid_iou_swarm', metrics1[2][4], epoch_num)       

        writer.add_scalar('info/valid_accuracy_water', metrics1[1][0], epoch_num)
        writer.add_scalar('info/valid_precision_water', metrics1[1][1], epoch_num)
        writer.add_scalar('info/valid_recall_water', metrics1[1][2], epoch_num)
        writer.add_scalar('info/valid_f1_water', metrics1[1][3], epoch_num)
        writer.add_scalar('info/valid_iou_water', metrics1[1][4], epoch_num)  

        writer.add_scalar('info/valid_accuracy_seabed', metrics1[2][0], epoch_num)
        writer.add_scalar('info/valid_precision_seabed', metrics1[2][1], epoch_num)
        writer.add_scalar('info/valid_recall_seabed', metrics1[2][2], epoch_num)
        writer.add_scalar('info/valid_f1_seabed', metrics1[2][3], epoch_num)
        writer.add_scalar('info/valid_iou_seabed', metrics1[2][4], epoch_num)  
        
        
        # ----------The terminal displays the time and remaining time-------------
        epochTime = time.time() - since
        timeElapsed += epochTime
        if resume:
            averageTime = timeElapsed / (epoch_num - init_epoch + 1)
        else:
            averageTime = timeElapsed / (epoch_num )  
        estimatedTime = averageTime * (args.epoch - (epoch_num))  
        endTime = datetime.datetime.now() + datetime.timedelta(seconds=estimatedTime)
        print('Average {:.0f} m {:.0f} s/epoch, needs {} days {} hours {} minutes {} seconds, expected to end {} month {} day {}:{:0>2}'.format(averageTime // 60,
                                                                                    averageTime % 60,
                                                                                    int(estimatedTime / (3600 * 24)),
                                                                                    int((estimatedTime % (3600 * 24)) / 3600),
                                                                                    int((estimatedTime % 3600) / 60),
                                                                                    int(estimatedTime % 60),
                                                                                    endTime.month,
                                                                                    endTime.day,
                                                                                    endTime.hour,
                                                                                    endTime.minute,
                                                                                    ))
        state = {'epoch':epoch_num,
                'state_dict': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'loss':valid_loss,
                'best_f1':best_mF1,
            }
        
        # ----------Save model and training files-------------
        if (epoch_num) % step_save == 0:
            torch.save(state, os.path.join(modelSavePath, 'current_model.pth'))
            torch.save(model, os.path.join(modelSavePath, 'current_model.pkl'))
            # Set the save paths for the loss csv file, the validation result csv file, and the training result plot
            fn = 'epoch{}_{:.4f}__{:.4f}_{:.4f}__{:.4f}__{:.4f}_{:.4f}__{:.4f}__{:.4f}'.format(
                args.epoch, train_loss, valid_loss, metrics[0][0], metrics[0][-2], metrics[0][-1], metrics1[0][0], metrics1[0][-2], metrics1[0][-1])
            print(fn)
            fp_csv = os.path.join(modelSavePath,  'result.csv')
            fp_csv_loss = os.path.join(modelSavePath, 'result_loss.csv')
            # Save training and validation metrics
            heads = "T_A_M,T_P_M,T_R_M,T_F_M,T_I_M,"+\
                    "T_A_0,T_P_0,T_R_0,T_F_0,T_I_0,"+\
                    "T_A_1,T_P_1,T_R_1,T_F_1,T_I_1,"+\
                    "T_A_2,T_P_2,T_R_2,T_F_2,T_I_2,"+\
                    "V_A_M,V_P_M,V_R_M,V_F_M,V_I_M,"+\
                    "V_A_0,V_P_0,V_R_0,V_F_0,V_I_0,"+\
                    "V_A_1,V_P_1,V_R_1,V_F_1,V_I_1,"+\
                    "V_A_1,V_P_2,V_R_2,V_F_2,V_I_2"
            trainProcessList = np.hstack((trainMetrics, validMetrics))
            np.savetxt(fp_csv, trainProcessList, delimiter=",",header=heads)
            # Preserving Training and Verification Losses
            loss_header = "train_loss,valid_loss"
            trainProcessLoss = np.vstack((trainLossList, validLossList))
            np.savetxt(fp_csv_loss, trainProcessLoss, delimiter=",", header=loss_header)
            
        if metrics1[0][3] > best_mF1:
            best_mF1 = metrics1[0][3]
            torch.save(state, os.path.join(modelSavePath, 'best_f1_model.pth'))
            torch.save(model, os.path.join(modelSavePath, 'best_f1_model.pkl'))
            print('update bestValidResult!')

        print()
    # ----------Drawing and saving the final file-------------
    # Set the save paths for the loss csv file, the validation result csv file, and the training result plot
    fp = os.path.join(modelSavePath, fn + '.png')
    fn = 'epoch{}_{:.4f}__{:.4f}_{:.4f}__{:.4f}__{:.4f}_{:.4f}__{:.4f}__{:.4f}'.format(
                args.epoch, train_loss, valid_loss, metrics[0][0], metrics[0][-2], metrics[0][-1], metrics1[0][0], metrics1[0][-2], metrics1[0][-1])
    
    # draw
    legendList = ['accuracy', 'Recall', 'Precison', 'f1Sorce', 'mIoU']
    epochList = [x for x in range(init_epoch, epoch+1)]
    plt.figure(figsize=(20, 8))
    plt.subplot(141)  # Training and validating loss maps
    plt.plot(epochList, trainLossList)
    plt.plot(epochList, validLossList)
    plt.grid()
    plt.title('loss')
    plt.legend(['train', 'valid'])

    plt.subplot(142)  # Chart of 5 training accuracy indicators for each round
    for i in range(5):
        plt.plot(epochList, trainMetrics[:, i])
    plt.grid()
    plt.title('train')
    plt.legend(legendList)
    plt.subplot(143)  # Chart of 5 validation accuracy metrics for each round
    if args.epoch < 200:
        for i in range(5):
            plt.plot(epochList, validMetrics[:, i])
    else:
        smooth_num = len(epochList) // 4 if len(epochList) // 4 >= 5 else 5
        # smooth_num = 50
        x_smooth = np.linspace(min(epochList), max(epochList), smooth_num)
        y_smooth = make_interp_spline(epochList, validMetrics[:, :5])(x_smooth)
        # plt.plot(x, y, '*')
        plt.plot(x_smooth, y_smooth)
    plt.grid()
    plt.title('valid')
    plt.legend(legendList[:5])

    plt.subplot(144)  # Plot of f1 and miou accuracy metrics for each round
    plt.plot(epochList, validMetrics[:, 3:5])  # f1、iou
    plt.grid()
    plt.title('valid')
    plt.legend(legendList[-2:])
    plt.savefig(fp)
    plt.close("all")

##########################
# Main
##########################
set_seed(args.seed)
train_net(args)
