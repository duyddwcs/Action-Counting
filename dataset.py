import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from utils import GenerateDensityMap

class OriCaraDataset(Dataset):

    def __init__(self, train_val_split_path = './Train_val_split.json', data_split = 'Train', root_dir = None, window_size = 5):
        assert data_split in ['Train', 'Val']

        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())
        self.window_size = window_size
        self.root_dir = root_dir

    def __getitem__(self, item):
        idx, data_sample = self.data_list[item]
        exemplar_dir = data_sample['ExemplarPath']
        exemplar_count = data_sample['ExemplarCount']
        query_dir = data_sample['QueryPath']
        query_count = data_sample['QueryCount']
        query_count = int(query_count)
        exemplar_count = int(exemplar_count)

        # Get raw signal data
        if self.root_dir is None:
            self.root_dir = './'
        query_dir = os.path.join(self.root_dir, query_dir)
        exemplar_dir = os.path.join(self.root_dir, exemplar_dir)

        query = pd.read_csv(query_dir, header=None)
        query = query.to_numpy()
        query = query[:, 1:]

        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()
        exemplar = exemplar[:, 1:]

        return query, query_count

    def __len__(self):
        return len(self.data_list)

class CaraDataset(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.Dataset.Train_val_split_path
        self.root_dir = cfg.Dataset.Root_dir
        self.window_size = cfg.Dataset.Window_size
        self.step_size = cfg.Dataset.Step_size
        self.gen_density_map = cfg.Dataset.Density_map
        self.density_map_length = cfg.Dataset.Density_map_length
        self.max_len = cfg.Dataset.Max_len
        assert data_split in ['Train', 'Val']


        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())

    def __getitem__(self, item):

        idx, data_sample = self.data_list[item]
        exemplar_dir = data_sample['ExemplarPath']
        exemplar_count = data_sample['ExemplarCount']
        query_dir = data_sample['QueryPath']
        query_count = data_sample['QueryCount']
        query_count = int(query_count)
        exemplar_count = int(exemplar_count)

        # Get raw query signal data
        if self.root_dir is None:
            self.root_dir = './'
        query_dir = os.path.join(self.root_dir, query_dir)
        exemplar_dir = os.path.join(self.root_dir, exemplar_dir)

        # Load query feature
        query = pd.read_csv(query_dir, header=None)
        query = query.to_numpy()
        query = query[:, 1:]

        if self.gen_density_map:
            # Generate the density map
            query_density_map = GenerateDensityMap(query, query_count)
            query_density_map = torch.FloatTensor(query_density_map)
            query_density_map = query_density_map.unsqueeze(0).unsqueeze(0)
            query_density_map = F.interpolate(query_density_map, size=(self.density_map_length), mode='linear')
            # Interpolate to 128
            query_density_map = query_density_map * query_count / query_density_map.sum()
            query_density_map = query_density_map.squeeze()
        else:
            query_density_map = None

        # Load exemplar feature
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()
        exemplar = exemplar[:, 1:]

        # Generate the exemplar density map
        if self.gen_density_map:
            # Generate the density map
            exemplar_density_map = GenerateDensityMap(exemplar, exemplar_count)
            exemplar_density_map = torch.FloatTensor(exemplar_density_map)
            exemplar_density_map = exemplar_density_map.unsqueeze(0).unsqueeze(0)
            exemplar_density_map = F.interpolate(exemplar_density_map, size=(self.density_map_length), mode='linear')
            # Interpolate to 128
            exemplar_density_map = exemplar_density_map * exemplar_count / exemplar_density_map.sum()
            exemplar_density_map = exemplar_density_map.squeeze()
        else:
            exemplar_density_map = None

        # Resample the query signal
        seq_len = query.shape[0]
        query_t_f = None
        query_t = torch.FloatTensor(query)  # L x C
        start_index = 0
        while (start_index + self.window_size) < seq_len:
            window_feat = query_t[start_index: start_index + self.window_size, :]
            # 10 x 6
            window_feat = window_feat.permute(1, 0)
            # print(window_feat.shape)
            window_feat = window_feat.reshape(1, -1)
            if query_t_f is None:
                query_t_f = window_feat
            else:
                query_t_f = torch.cat((query_t_f, window_feat), 0)

            if (start_index + self.window_size) < seq_len:
                start_index += self.step_size
            else:
                final_window_feat = query_t[start_index: seq_len, :]
                final_window_feat = final_window_feat.reshape(1, -1)
                query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                break

        if query_t_f.shape[0] > self.max_len:
            query_t_f = query_t_f[:self.max_len, :]


        # Resample the exemplar signal
        exemplar_seq_len = exemplar.shape[0]
        exemplar_t_f = None
        exemplar_t = torch.FloatTensor(exemplar)  # L x C
        start_index = 0
        while (start_index + self.window_size) < exemplar_seq_len:
            window_feat = exemplar_t[start_index: start_index + self.window_size, :]
            # 10 x 6
            window_feat = window_feat.permute(1, 0)
            # print(window_feat.shape)
            window_feat = window_feat.reshape(1, -1)
            if exemplar_t_f is None:
                exemplar_t_f = window_feat
            else:
                exemplar_t_f = torch.cat((exemplar_t_f, window_feat), 0)

            if (start_index + self.window_size) < exemplar_seq_len:
                start_index += self.step_size
            else:
                final_window_feat = exemplar_t[start_index: exemplar_seq_len, :]
                final_window_feat = final_window_feat.reshape(1, -1)
                exemplar_t_f = torch.cat((query_t_f, final_window_feat), 0)
                break

        if exemplar_t_f.shape[0] > self.max_len:
            exemplar_t_f = exemplar_t_f[:self.max_len, :]

        if query_density_map is not None:
            return query_t_f, query_count, query_density_map, exemplar_t_f, exemplar_count, exemplar_density_map

        else:
            return query_t_f, query_count, exemplar_t_f, exemplar_count

    def __len__(self):
        return len(self.data_list)

# Augmentation doesn't work

'''class AugmentCaraDataset(Dataset):

    def __init__(self, train_val_split_path = './Train_val_split.json', data_split = 'Train', root_dir = None, window_size = 10):
        assert data_split in ['Train', 'Val']

        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())
        self.window_size = window_size
        self.root_dir = root_dir
        self.aug_data_list = []


        # Augment query with exemplar
        for item in range(len(self.data_list)):
            idx, data_sample = self.data_list[item]
            exemplar_dir = data_sample['ExemplarPath']
            exemplar_count = data_sample['ExemplarCount']
            query_dir = data_sample['QueryPath']
            query_count = data_sample['QueryCount']
            query_count = int(query_count)
            exemplar_count = int(exemplar_count)
            print(query_dir, query_count)
            # Get raw signal data
            if self.root_dir is None:
                self.root_dir = './'
            query_dir = os.path.join(self.root_dir, query_dir)
            exemplar_dir = os.path.join(self.root_dir, exemplar_dir)

            query = pd.read_csv(query_dir, header=None)
            query = query.to_numpy()
            query = query[:, 1:]

            exemplar = pd.read_csv(exemplar_dir, header=None)
            exemplar = exemplar.to_numpy()
            exemplar = exemplar[:, 1:]

            # Interplot the query
            seq_len = query.shape[0]
            query_t = torch.FloatTensor(query)
            query_t = query_t.permute(1, 0).unsqueeze(0)  # B x C x L
            new_window_len = np.ceil(seq_len / self.window_size).astype(np.int32)
            new_seq_len = new_window_len * self.window_size
            query_t = F.interpolate(query_t, size=(new_seq_len), mode='linear')
            query_t = query_t.squeeze().permute(1, 0)

            # Divide the query into window
            query_t = query_t.reshape(new_window_len, -1)
            self.aug_data_list.append([query_t, query_count])

            # Augmentation
            emp_len = exemplar.shape[0]

            # Shorter than the max length
            if (new_seq_len + emp_len) <= 3000 * self.window_size:
                exemplar_t = torch.FloatTensor(exemplar)
                exemplar_t = exemplar_t.permute(1, 0).unsqueeze(0)  # B x C x L
                new_exemplar_window_len = np.ceil(emp_len / self.window_size).astype(np.int32)
                new_exemplar_seq_len = new_exemplar_window_len * self.window_size
                exemplar_t = F.interpolate(exemplar_t, size=(new_exemplar_seq_len), mode='linear')
                exemplar_t = exemplar_t.squeeze().permute(1, 0)

                # Divide the query into window
                exemplar_t = exemplar_t.reshape(new_exemplar_window_len, -1)
                #print(query_t.shape)
                #print(exemplar_t.shape)
                #print('########################')
                p = np.random.rand()
                aug_count = query_count + exemplar_count
                #print(query_count)
                #print(exemplar_count)
                #print('###################')
                #print(aug_count)
                if p >= 0.5:
                    aug_t = torch.cat((query_t, exemplar_t), 0)
                else:
                    aug_t = torch.cat((exemplar_t, query_t), 0)
                self.aug_data_list.append([aug_t, aug_count])


    def __getitem__(self, item):
        query_t, query_count = self.aug_data_list[item]

        return query_t, query_count

    def __len__(self):
        return len(self.aug_data_list)'''

class AugmentCaraDataset(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.Dataset.Train_val_split_path
        self.root_dir = cfg.Dataset.Root_dir
        self.window_size = cfg.Dataset.Window_size
        self.step_size = cfg.Dataset.Step_size
        self.gen_density_map = cfg.Dataset.Density_map
        self.density_map_length = cfg.Dataset.Density_map_length
        self.max_len = cfg.Dataset.Max_len
        assert data_split in ['Train', 'Val']
        self.aug_data_list = []

        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())

        for item in range(len(self.data_list)):
            print(item)
            idx, data_sample = self.data_list[item]
            exemplar_dir = data_sample['ExemplarPath']
            exemplar_count = data_sample['ExemplarCount']
            query_dir = data_sample['QueryPath']
            query_count = data_sample['QueryCount']
            query_count = int(query_count)
            exemplar_count = int(exemplar_count)

            # Query numpy
            query = pd.read_csv(query_dir, header=None)
            query = query.to_numpy()
            query = query[:, 1:]

            # Add query into the aug data list
            if self.gen_density_map:
                # Generate the density map
                density_map = GenerateDensityMap(query, query_count)
                density_map = torch.FloatTensor(density_map)
                density_map = density_map.unsqueeze(0).unsqueeze(0)
                density_map = F.interpolate(density_map, size=(self.density_map_length), mode='linear')
                # Interpolate to 128
                density_map = density_map * query_count / density_map.sum()
                query_density_map = density_map.squeeze()
            else:
                query_density_map = None

            # Resample the query
            seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            self.aug_data_list.append([query_t_f, query_count, query_density_map])

            # Exemplar
            exemplar = pd.read_csv(exemplar_dir, header=None)
            exemplar = exemplar.to_numpy()
            exemplar = exemplar[:, 1:]

            # Augment the query + exemplar
            emp_len = exemplar.shape[0]

            # Shorter than the max length
            if (seq_len + emp_len) <= 30000:

                p = np.random.rand()
                if p >= 0.5:
                    augfeat = np.concatenate((query, exemplar), 0)
                else:
                    augfeat = np.concatenate((exemplar, query), 0)

                aug_count = query_count + exemplar_count

                if self.gen_density_map:
                    # Generate the density map
                    aug_density_map = GenerateDensityMap(augfeat, aug_count)
                    aug_density_map = torch.FloatTensor(aug_density_map)
                    aug_density_map = aug_density_map.unsqueeze(0).unsqueeze(0)
                    aug_density_map = F.interpolate(aug_density_map, size=(self.density_map_length), mode='linear')
                    # Interpolate to 128
                    aug_density_map = aug_density_map * aug_count / aug_density_map.sum()
                    aug_density_map = aug_density_map.squeeze()
                else:
                    aug_density_map = None

                # Resample the augmented data
                aug_seq_len = augfeat.shape[0]
                # print('aug_len', aug_seq_len)
                augfeat_t_f = None
                augfeat_t = torch.FloatTensor(augfeat)  # L x C
                start_index = 0
                while (start_index + self.window_size) < aug_seq_len:
                    window_feat = augfeat_t[start_index: start_index + self.window_size, :]
                    # 10 x 6
                    window_feat = window_feat.permute(1, 0)
                    # print(window_feat.shape)
                    window_feat = window_feat.reshape(1, -1)
                    if augfeat_t_f is None:
                        augfeat_t_f = window_feat
                    else:
                        augfeat_t_f = torch.cat((augfeat_t_f, window_feat), 0)

                    if (start_index + self.window_size) < aug_seq_len:
                        start_index += self.step_size
                    else:
                        final_window_feat = augfeat_t[start_index: aug_seq_len, :]
                        final_window_feat = final_window_feat.reshape(1, -1)
                        augfeat_t_f = torch.cat((augfeat_t_f, final_window_feat), 0)
                        break

                if augfeat_t_f.shape[0] > self.max_len:
                    augfeat_t_f = augfeat_t_f[:self.max_len, :]

                self.aug_data_list.append([augfeat_t_f, aug_count, aug_density_map])

    def __getitem__(self, item):
        feat, count, density_map = self.aug_data_list[item]

        return feat, count, density_map

    def __len__(self):
        return len(self.aug_data_list)

class SlideCaraDataset(Dataset):
    '''
    Will generate a padding mask for same length
    (length / window size, feature * window size) Avoid too many token
    '''
    def __init__(self, cfg, data_split = 'Train', root_dir = None):
        assert data_split in ['Train', 'Val']

        train_val_split_path = cfg.Dataset.Train_val_split_path
        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())
        self.window_size = cfg.Dataset.Windowsize
        self.root_dir = root_dir
        self.step_size = cfg.Dataset.Step_size
        self.max_len = cfg.Dataset.Max_len


    def __getitem__(self, item):
        idx, data_sample = self.data_list[item]
        exemplar_dir = data_sample['ExemplarPath']
        exemplar_count = data_sample['ExemplarCount']
        query_dir = data_sample['QueryPath']
        query_count = data_sample['QueryCount']
        query_count = int(query_count)
        exemplar_count = int(exemplar_count)

        # Get raw query signal data
        if self.root_dir is None:
            self.root_dir = './'
        query_dir = os.path.join(self.root_dir, query_dir)
        exemplar_dir = os.path.join(self.root_dir, exemplar_dir)

        query = pd.read_csv(query_dir, header=None)
        query = query.to_numpy()
        query = query[:, 1:]

        # Get raw exemplar signal data
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()
        exemplar = exemplar[:, 1:]

        # Resample the raw query signal into token
        seq_len = query.shape[0]
        query_t_f = None
        query_t = torch.FloatTensor(query) # L x C
        start_index = 0
        while (start_index + self.window_size) < seq_len:
            window_feat = query_t[start_index : start_index + self.window_size, :]
            # 10 x 6
            window_feat = window_feat.permute(1, 0)
            # print(window_feat.shape)
            window_feat = window_feat.reshape(1, -1)
            if query_t_f is None:
                query_t_f = window_feat
            else:
                query_t_f = torch.cat((query_t_f, window_feat), 0)

            if (start_index + self.window_size) < seq_len:
                start_index += self.step_size
            else:
                final_window_feat = query_t[start_index : seq_len, :]
                final_window_feat = final_window_feat.reshape(1, -1)
                query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                break

        if query_t_f.shape[0] > self.max_len:
            query_t_f = query_t_f[:self.max_len, :]

        # Exemplar feature
        exemplar_t = torch.FloatTensor(exemplar)

        return query_t_f, query_count

    def __len__(self):
        return len(self.data_list)

class DensitySlideCaraDataset(Dataset):


    def __init__(self, cfg, data_split = 'Train', root_dir = None):
        assert data_split in ['Train', 'Val']

        train_val_split_path = cfg.Dataset.Train_val_split_path
        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())
        self.window_size = 10
        self.root_dir = root_dir
        self.step_size = cfg.Dataset.Step_size
        self.max_len = cfg.Dataset.Max_len


    def __getitem__(self, item):
        idx, data_sample = self.data_list[item]
        exemplar_dir = data_sample['ExemplarPath']
        exemplar_count = data_sample['ExemplarCount']
        query_dir = data_sample['QueryPath']
        query_count = data_sample['QueryCount']
        query_count = int(query_count)
        exemplar_count = int(exemplar_count)

        # Get raw signal data
        if self.root_dir is None:
            self.root_dir = './'
        query_dir = os.path.join(self.root_dir, query_dir)
        exemplar_dir = os.path.join(self.root_dir, exemplar_dir)
        query = pd.read_csv(query_dir, header=None)
        query = query.to_numpy()
        query = query[:, 1:]
        density_map = GenerateDensityMap(query, query_count)
        density_map = torch.FloatTensor(density_map)
        density_map = density_map.unsqueeze(0).unsqueeze(0)
        density_map = F.interpolate(density_map, size=(128), mode='linear')
        # Interpolate to 128
        density_map = density_map * query_count / density_map.sum()
        density_map = density_map.squeeze()

        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()
        exemplar = exemplar[:, 1:]

        seq_len = query.shape[0]
        #print(seq_len)
        query_t_f = None
        query_t = torch.FloatTensor(query) # L x C
        start_index = 0
        while (start_index + self.window_size) < seq_len:
            window_feat = query_t[start_index : start_index + self.window_size, :]
            # 10 x 6
            window_feat = window_feat.permute(1, 0)
            # print(window_feat.shape)
            window_feat = window_feat.reshape(1, -1)
            if query_t_f is None:
                query_t_f = window_feat
            else:
                query_t_f = torch.cat((query_t_f, window_feat), 0)

            # If it will exceed later
            if (start_index + self.window_size) < seq_len:
                start_index += self.step_size
            else:
                final_window_feat = query_t[start_index : seq_len, :]
                final_window_feat = final_window_feat.reshape(1, -1)
                query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                break

        if query_t_f.shape[0] > self.max_len:
            query_t_f = query_t_f[:self.max_len, :]
        #print(query_t_f.shape)
        exemplar_t = torch.FloatTensor(exemplar)
        #print(query_count)
        return query_t_f, query_count, density_map, query

    def __len__(self):
        return len(self.data_list)


'''class MultiCaraDataset(Dataset):

    def __init__(self, train_val_split_path = './Train_val_split.json', data_split = 'Train', root_dir = None, window_size_list = [5, 10, 15]):
        assert data_split in ['Train', 'Val']

        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())
        self.window_size_list = window_size_list
        self.root_dir = root_dir

    def __getitem__(self, item):
        idx, data_sample = self.data_list[item]
        exemplar_dir = data_sample['ExemplarPath']
        exemplar_count = data_sample['ExemplarCount']
        query_dir = data_sample['QueryPath']
        query_count = data_sample['QueryCount']
        query_count = int(query_count)
        exemplar_count = int(exemplar_count)
        query_list = []

        for window_size in self.window_size_list:

            # Get raw signal data
            if self.root_dir is None:
                self.root_dir = './'
            query_dir = os.path.join(self.root_dir, query_dir)
            exemplar_dir = os.path.join(self.root_dir, exemplar_dir)

            query = pd.read_csv(query_dir, header=None)
            query = query.to_numpy()
            query = query[:, 1:]

            exemplar = pd.read_csv(exemplar_dir, header=None)
            exemplar = exemplar.to_numpy()
            exemplar = exemplar[:, 1:]

            # Interplot the query
            seq_len = query.shape[0]
            query_t = torch.FloatTensor(query)
            query_t = query_t.permute(1, 0).unsqueeze(0) # B x C x L
            new_window_len = np.ceil(seq_len / window_size).astype(np.int32)
            new_seq_len = new_window_len * window_size
            query_t = F.interpolate(query_t, size=(new_seq_len), mode='linear')
            query_t = query_t.squeeze().permute(1, 0)

            # Divide the query into window
            query_t = query_t.reshape(new_window_len, -1)
            query_list.append(query_t)
            # 24/8 Since now I don't need the exemplar I didn't process the exemplar in the samew way.
            # Now simply convert the exemplar to a tensor
            exemplar_t = torch.FloatTensor(exemplar)
            #print(query_count)
        assert len(query_list) == len(self.window_size_list)
        return query_list, query_count

    def __len__(self):
        return len(self.data_list)'''

class MultiCaraDataset(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.MultiCaraDataset.Train_val_split_path
        self.root_dir = cfg.MultiCaraDataset.Root_dir
        self.window_size_list = cfg.MultiCaraDataset.Window_size_list
        self.step_size_list = cfg.MultiCaraDataset.Step_size_list
        self.gen_density_map = cfg.Dataset.Density_map
        self.density_map_length = cfg.Dataset.Density_map_length
        self.max_len = cfg.MultiCaraDataset.Max_len
        assert data_split in ['Train', 'Val']


        with open(train_val_split_path) as json_file:
            train_val_split = json.load(json_file)

        self.data_dict = train_val_split[data_split]
        self.data_list = list(self.data_dict.items())

    def __getitem__(self, item):

        idx, data_sample = self.data_list[item]
        exemplar_dir = data_sample['ExemplarPath']
        exemplar_count = data_sample['ExemplarCount']
        query_dir = data_sample['QueryPath']
        query_count = data_sample['QueryCount']
        query_count = int(query_count)
        exemplar_count = int(exemplar_count)

        # Get raw query signal data
        if self.root_dir is None:
            self.root_dir = './'
        query_dir = os.path.join(self.root_dir, query_dir)
        exemplar_dir = os.path.join(self.root_dir, exemplar_dir)

        query = pd.read_csv(query_dir, header=None)
        query = query.to_numpy()
        query = query[:, 1:]

        # Get raw exemplar signal data
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()
        exemplar = exemplar[:, 1:]

        if self.gen_density_map:
            # Generate the density map
            density_map = GenerateDensityMap(query, query_count)
            density_map = torch.FloatTensor(density_map)
            density_map = density_map.unsqueeze(0).unsqueeze(0)
            density_map = F.interpolate(density_map, size=(self.density_map_length), mode='linear')
            # Interpolate to 128
            density_map = density_map * query_count / density_map.sum()
            density_map = density_map.squeeze()
        else:
            density_map = None

        multi_query_list = []

        # Multi scale
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the query
            seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            multi_query_list.append(query_t_f)

        assert len(multi_query_list) == len(self.step_size_list)
        exemplar_t = torch.FloatTensor(exemplar)
        if density_map is not None:
            return multi_query_list, query_count, density_map

        else:
            return multi_query_list, query_count

    def __len__(self):
        return len(self.data_list)

#d = MultiCaraDataset()
#q, e = d.__getitem__(2)
#for qe in q:
#    print(qe.shape)
# Debug dataset
'''
d = CaraDataset()
print(d.__len__())
q, e = d.__getitem__(2)
print(q.shape)
print(e.shape)
'''

#d = AugmentCaraDataset()
#q, qc, e, ec = d.__getitem__(6)
#sample_num = d.__len__()
#length = []
#for s in range(sample_num):
#    q, e = d.__getitem__(s)
#    length.append(q.shape[0])
#print(max(length))
#print(sum(length) / len(length))


#max_length : 28050
#Avg_length : 5647
# 512 average size 512 -> 128 -> 1 to regress the count number.
# 2805
# 5650 -> 565


# Max length:3000, feat_dim = 60
# The preprocess -> interpoltan to a window sized sequence -> divide into windows -> each window is a token -> transformer -> regression
# Regression head with padding.


#Maybe write a json file to organize the dataset.
# Query Path
# Query Count
# Exemplar Path
# Exemplar Count
# Category