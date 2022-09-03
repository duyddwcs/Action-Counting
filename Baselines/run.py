import os
import torch
import random
import datetime
import warnings
import copy
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from builder import build_model, build_dataset
from torch.utils.tensorboard import SummaryWriter
from config import get_default_cfg, merge_from_file, show_cfg, save_cfg

# Filter warning
warnings.simplefilter("ignore", UserWarning)

# Experiment config
exp_name = '[Recofit][TransRAC]'
exp_setting_path = './Config/' + exp_name +'.yaml'
log_root = './Log'
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")
exp_name = '[' + experiment_date + ']' + exp_name
log_save_dir = os.path.join(log_root, exp_name)
if not os.path.exists(log_save_dir):
    os.makedirs(log_save_dir)
config_save_dir = os.path.join(log_save_dir, 'Setting.yaml')

# Load experiment config
cfg = get_default_cfg()
cfg = merge_from_file(cfg, exp_setting_path)

# Show experiment config
show_cfg(cfg)
save_cfg(cfg, config_save_dir)

# Fix the seed
seed = cfg.Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Experiment Result
Result = {}
TBwriter = SummaryWriter(log_save_dir)

# Base Setting
print('GPU: ', cfg.GPUID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUID)
LR = cfg.Train.LR
weight_decay = cfg.Train.WeightDecay
Epoch = cfg.Train.Epoch

train_set, val_set = build_dataset(cfg)
train_loader = DataLoader(train_set, batch_size = 1)
val_loader = DataLoader(val_set, batch_size = 1)

model = build_model(cfg)
model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay = weight_decay)
loss = 0

# print('Training samples:', len(train_loader))
print(exp_name)
print('Large window size: 30')
for epoch in range(Epoch):
    #print('####################################')
    Result[epoch + 1] = {}
    Result[epoch + 1]['train'] = {}
    Result[epoch + 1]['val'] = {}
    model.train()
    print('#################################################################')
    print('Epoch:', epoch)
    print('Current Learning Rate:', optimizer.param_groups[0]['lr'])
    coung_regloss_list = []
    density_regloss_list = []
    error_list = []
    RM_error_list = []
    for idx, data in enumerate(train_loader):

        query_t, query_count, query_density, exemplar_t, exemplar_count, exemplar_density = data
        #print(query_count)
        #print(exemplar_count)
        # print(query_t[0].shape)
        pred_count = model(query_t)
        query_count = torch.tensor(query_count).cuda()
        error = np.absolute((pred_count - query_count).detach().cpu().numpy())

        # MAE
        error_list.append(error)
        # RMSE
        RM_error_list.append(error ** 2)
        # Calculate loss
        loss = loss + (pred_count - query_count) ** 2
        coung_regloss_list.append(error ** 2)
        # loss = loss + ((pred_density - density) ** 2).sum()

        if (idx + 1) % 16 == 0 or (idx + 1) == len(train_loader):
            sample_num = (idx + 1) % 16
            if sample_num == 0:
                sample_num = 16
            optimizer.zero_grad()
            loss /= sample_num
            loss.backward()
            optimizer.step()
            loss = 0

    train_mae = sum(error_list) / len(error_list)
    train_rmse = np.sqrt(sum(RM_error_list) / len(RM_error_list))
    if len(coung_regloss_list) != 0:
        train_count_loss = sum(coung_regloss_list) / len(coung_regloss_list)
    else:
        train_count_loss = 0
    if len(density_regloss_list) != 0:
        train_density_loss = sum(density_regloss_list) / len(density_regloss_list)
    else:
        train_density_loss = 0
    print('Train MAE: ', train_mae)
    print('Train RMSE: ', train_rmse)
    print('Train count Loss: ', train_count_loss)
    print('Train density Loss: ', train_density_loss)
    TBwriter.add_scalar('train/MAE', train_mae, epoch + 1)
    TBwriter.add_scalar('train/RMSE', train_rmse, epoch + 1)
    TBwriter.add_scalar('train/Count loss', train_count_loss, epoch + 1)
    TBwriter.add_scalar('train/Density loss', train_density_loss, epoch + 1)
    Result[epoch + 1]['train']['MAE'] = float(train_mae[0])
    Result[epoch + 1]['train']['RMSE'] = float(train_rmse[0])

    #Evaluate
    model.eval()
    print('Evaluation: ')
    val_error_list = []
    val_RM_error_list = []
    for idx, data in enumerate(val_loader):

        query_t, query_count, query_density, exemplar_t, exemplar_count, exemplar_density = data
        #print(query_count)
        #print(exemplar_count)
        pred_count = model(query_t)
        query_count = torch.tensor(query_count).cuda()
        error = np.absolute((pred_count - query_count).detach().cpu().numpy())

        val_error_list.append(error)
        val_RM_error_list.append(error ** 2)
    val_mae = sum(val_error_list) / len(val_error_list)
    val_rmse = np.sqrt(sum(val_RM_error_list) / len(val_RM_error_list))
    print('Val MAE: ', val_mae)
    print('Val RMSE: ', val_rmse)
    TBwriter.add_scalar('val/MAE', val_mae, epoch + 1)
    TBwriter.add_scalar('val/RMSE', val_rmse, epoch + 1)
    Result[epoch + 1]['val']['MAE'] = float(val_mae[0])
    Result[epoch + 1]['val']['RMSE'] = float(val_rmse[0])

model_path = os.path.join(log_save_dir, 'count_model.pth')
torch.save(model.state_dict(), model_path)

print('Test time adaptation:')
val_error_list = []
val_RM_error_list = []
model.eval()
for idx, data in enumerate(val_loader):

    query_t, query_count, query_density, exemplar_t, exemplar_count, exemplar_density = data

    ada_model = copy.deepcopy(model)
    ada_model.train()
    ada_optimizer = torch.optim.AdamW(ada_model.parameters(), lr = 1e-7)
    exemplar_count = torch.tensor(exemplar_count).cuda()

    for i in range(10):
        pred_count = ada_model(exemplar_t)
        ada_loss = (pred_count - exemplar_count) ** 2
        # print(ada_loss)
        ada_optimizer.zero_grad()
        ada_loss.backward()
        ada_optimizer.step()

    pred_query_count = ada_model(query_t)
    query_count = torch.tensor(query_count).cuda()
    error = np.absolute((pred_query_count - query_count).detach().cpu().numpy())
    val_error_list.append(error)
    val_RM_error_list.append(error ** 2)

val_mae = sum(val_error_list) / len(val_error_list)
val_rmse = np.sqrt(sum(val_RM_error_list) / len(val_RM_error_list))
print('Adaptation Val MAE: ', val_mae)
print('Adaptation Val RMSE: ', val_rmse)
Result['Final adapt MAE'] = float(val_mae[0])
Result['Adaptation Val RMSE'] = float(val_rmse[0])
TBwriter.add_scalar('val/Adaptation Val MAE', val_mae, 1)
TBwriter.add_scalar('val/Adaptation Val RMSE', val_rmse, 1)

result_conf = OmegaConf.create(Result)
Result_save_path = os.path.join(log_save_dir, 'Result.yaml')
save_cfg(result_conf, Result_save_path)

#Result_save_path = os.path.join(log_save_dir, 'Result.pth')
#torch.save(Result, Result_save_path)
TBwriter.close()