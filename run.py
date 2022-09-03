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

exp_name = '[Cara][Transformer][SimiReg][Density][NoAdaptation][8.30.1AM]'
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
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
loss = 0

# print('Training samples:', len(train_loader))
print(exp_name)
print('Large window size: 30')
for epoch in range(Epoch):
    #print('####################################')
    Result[epoch + 1] = {}
    Result[epoch + 1]['train'] = {}
    Result[epoch + 1]['val'] = {}
    Result[epoch + 1]['LR'] = {}
    model.train()
    print('#################################################################')
    print('Epoch:', epoch)
    print('Current Learning Rate:', optimizer.param_groups[0]['lr'])
    coung_regloss_list = []
    density_regloss_list = []
    error_list = []
    RM_error_list = []
    for idx, data in enumerate(train_loader):

        if cfg.Dataset.Name == 'CaraDataset':
            if cfg.Dataset.Density_map:
                query_t, query_count, density, exemplar_t, exemplar_count, exemplar_density = data
                query_t = query_t.cuda()
                density = density.cuda()
            else:
                query_t, query_count, exemplar_t, exemplar_count = data
                query_t = query_t.cuda()

        elif cfg.Dataset.Name == 'MultiCaraDataset':
            if cfg.Dataset.Density_map:
                query_t, query_count, density = data
                density = density.cuda()
            else:
                query_t, query_count = data

        pred_density = model(query_t)
        # loss and update

        count = pred_density.sum()

        query_count = torch.tensor(query_count).cuda()
        error = np.absolute((count - query_count).detach().cpu().numpy())
        # print(error)
        # MAE
        error_list.append(error)
        # RMSE
        RM_error_list.append(error ** 2)



        if cfg.Train.Loss == 'Density':
            loss = loss + ((pred_density - density) ** 2).sum()
            density_reg_loss = ((pred_density - density) ** 2).sum()
            density_regloss_list.append(density_reg_loss.detach().cpu().numpy())
        elif cfg.Train.Loss == 'Combine':
            loss = loss + (count - query_count) ** 2 + 20 * (((pred_density - density) ** 2).sum())
            density_reg_loss = ((pred_density - density) ** 2).sum()
            density_regloss_list.append(density_reg_loss.detach().cpu().numpy())
            coung_regloss_list.append(error ** 2)
        else:
            loss = loss + (count - query_count) ** 2
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
    Result[epoch + 1]['train']['MAE'] = train_mae
    Result[epoch + 1]['train']['RMSE'] = train_rmse
    Result[epoch + 1]['train']['Total Loss'] = train_count_loss + train_density_loss

    #Evaluate
    model.eval()
    print('Evaluation: ')
    val_error_list = []
    val_RM_error_list = []
    for idx, data in enumerate(val_loader):

        if cfg.Dataset.Name == 'CaraDataset':
            if cfg.Dataset.Density_map:
                query_t, query_count, _, exemplar_t, exemplar_count, exemplar_density = data
                query_t = query_t.cuda()
                exemplar_t = exemplar_t.cuda()
                exemplar_density = exemplar_density.cuda()
            else:
                query_t, query_count = data
                query_t = query_t.cuda()

        elif cfg.Dataset.Name == 'MultiCaraDataset':
            if cfg.Dataset.Density_map:
                query_t, query_count, _ = data
            else:
                query_t, query_count = data


        if cfg.Val.Adaptation:
            ada_model = copy.deepcopy(model)
            ada_optimizer = torch.optim.AdamW(ada_model.parameters(), lr=1e-6)

            for i in range(10):
                pred_exemplar_density = ada_model(exemplar_t)
                ada_loss = ((pred_exemplar_density - exemplar_density) ** 2).sum()
                ada_optimizer.zero_grad()
                ada_loss.backward()
                ada_optimizer.step()
            pred_density = ada_model(query_t)
        else:
            pred_density = model(query_t)
        count = pred_density.sum()
        query_count = torch.tensor(query_count).cuda()
        error = np.absolute((count - query_count).detach().cpu().numpy())
        val_error_list.append(error)
        val_RM_error_list.append(error ** 2)
    val_mae = sum(val_error_list) / len(val_error_list)
    val_rmse = np.sqrt(sum(val_RM_error_list) / len(val_RM_error_list))
    print('Val MAE: ', val_mae)
    print('Val RMSE: ', val_rmse)
    TBwriter.add_scalar('val/MAE', val_mae, epoch + 1)
    TBwriter.add_scalar('val/RMSE', val_rmse, epoch + 1)
    Result[epoch + 1]['val']['MAE'] = train_mae
    Result[epoch + 1]['val']['RMSE'] = train_rmse
    scheduler.step()
model_path = os.path.join(log_save_dir, 'count_model.pth')
torch.save(model.state_dict(), model_path)

print('Test time adaptation:')

val_error_list = []
val_RM_error_list = []
model.eval()
for idx, data in enumerate(val_loader):
    if cfg.Dataset.Name == 'CaraDataset':
        if cfg.Dataset.Density_map:
            query_t, query_count, _, exemplar_t, exemplar_count, exemplar_density = data
            query_t = query_t.cuda()
            exemplar_t = exemplar_t.cuda()
            exemplar_density = exemplar_density.cuda()
        else:
            query_t, query_count = data
            query_t = query_t.cuda()

    elif cfg.Dataset.Name == 'MultiCaraDataset':
        if cfg.Dataset.Density_map:
            query_t, query_count, _ = data
        else:
            query_t, query_count = data


    ada_model = copy.deepcopy(model)
    ada_optimizer = torch.optim.AdamW(ada_model.regressionhead.parameters(), lr = 1e-7)
    exemplar_count = torch.tensor(exemplar_count).cuda()
    for i in range(100):
        pred_exemplar_density = ada_model(exemplar_t)
        count = pred_exemplar_density.sum()
        ada_loss = (count - exemplar_count) ** 2
        # print(ada_loss)
        # + 20 * (((pred_exemplar_density - density) ** 2).sum())
        ada_optimizer.zero_grad()
        ada_loss.backward()
        ada_optimizer.step()

    pred_density = ada_model(query_t)
    count = pred_density.sum()
    query_count = torch.tensor(query_count).cuda()
    error = np.absolute((count - query_count).detach().cpu().numpy())
    val_error_list.append(error)
    val_RM_error_list.append(error ** 2)


val_mae = sum(val_error_list) / len(val_error_list)
val_rmse = np.sqrt(sum(val_RM_error_list) / len(val_RM_error_list))
print('Val MAE: ', val_mae)
print('Val RMSE: ', val_rmse)

    # scheduler.step()
#result_conf = OmegaConf.create(Result)
#Result_save_path = os.path.join(log_save_dir, 'Result.yaml')
#save_cfg(result_conf, Result_save_path)
#torch.save(Result, Result_save_path)