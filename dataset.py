import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from utils import GenerateDensityMap

# CaraDataset
# For baseline density map is not needed
class CaraDataset(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.CARA.Train_val_split_path
        self.root_dir = cfg.CARA.Root_dir
        self.window_size = cfg.CARA.Window_size
        self.step_size = cfg.CARA.Step_size
        self.gen_density_map = cfg.CARA.Density_map
        self.density_map_length = cfg.CARA.Density_map_length
        self.max_len = cfg.CARA.Max_len
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

# Multi scale for TransRAC
class MultiCaraDataset(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.CARA.Train_val_split_path
        self.root_dir = cfg.CARA.Root_dir
        self.window_size_list = cfg.CARA.Window_size
        self.step_size_list = cfg.CARA.Step_size
        self.gen_density_map = cfg.CARA.Density_map
        self.density_map_length = cfg.CARA.Density_map_length
        self.max_len = cfg.CARA.Max_len
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

        # Generate the query density map
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

        # Generate the exemplar density map
        if self.gen_density_map:
            # Generate the density map
            exemplar_density_map = GenerateDensityMap(exemplar, exemplar_count)
            exemplar_density_map = torch.FloatTensor(exemplar_density_map)
            exemplar_density_map = exemplar_density_map.unsqueeze(0).unsqueeze(0)
            exemplar_density_map = F.interpolate(exemplar_density_map, size=(self.density_map_length),
                                                 mode='linear')
            # Interpolate to 128
            exemplar_density_map = exemplar_density_map * exemplar_count / exemplar_density_map.sum()
            exemplar_density_map = exemplar_density_map.squeeze()
        else:
            exemplar_density_map = None


        multi_query_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the query
            query_seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < query_seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < query_seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: query_seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            multi_query_list.append(query_t_f)

        assert len(multi_query_list) == len(self.step_size_list)

        multi_exemplar_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the exemplar
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
                    query_t_f = torch.cat((exemplar_t_f, final_window_feat), 0)
                    break

            if exemplar_t_f.shape[0] > self.max_len:
                exemplar_t_f = exemplar_t_f[:self.max_len, :]

            multi_exemplar_list.append(exemplar_t_f)

        assert len(multi_exemplar_list) == len(self.step_size_list)
        if query_density_map is not None:
            return multi_query_list, query_count, query_density_map, multi_exemplar_list, exemplar_count, exemplar_density_map
        else:
            return multi_query_list, query_count, torch.rand(128), multi_exemplar_list, exemplar_count, torch.rand(128)

    def __len__(self):
        return len(self.data_list)


# Multi scale for TransRAC
class MultiMMfit(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.MMFit.Train_val_split_path
        self.root_dir = cfg.MMFit.Root_dir
        self.window_size_list = cfg.MMFit.Window_size
        self.step_size_list = cfg.MMFit.Step_size
        self.gen_density_map = cfg.MMFit.Density_map
        self.density_map_length = cfg.MMFit.Density_map_length
        self.max_len = cfg.MMFit.Max_len
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

        # Get raw exemplar signal data
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()

        # Generate the query density map
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

        # Generate the exemplar density map
        if self.gen_density_map:
            # Generate the density map
            exemplar_density_map = GenerateDensityMap(exemplar, exemplar_count)
            exemplar_density_map = torch.FloatTensor(exemplar_density_map)
            exemplar_density_map = exemplar_density_map.unsqueeze(0).unsqueeze(0)
            exemplar_density_map = F.interpolate(exemplar_density_map, size=(self.density_map_length),
                                                 mode='linear')
            # Interpolate to 128
            exemplar_density_map = exemplar_density_map * exemplar_count / exemplar_density_map.sum()
            exemplar_density_map = exemplar_density_map.squeeze()
        else:
            exemplar_density_map = None


        multi_query_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the query
            query_seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < query_seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < query_seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: query_seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            multi_query_list.append(query_t_f)

        assert len(multi_query_list) == len(self.step_size_list)

        multi_exemplar_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the exemplar
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
                    query_t_f = torch.cat((exemplar_t_f, final_window_feat), 0)
                    break

            if exemplar_t_f.shape[0] > self.max_len:
                exemplar_t_f = exemplar_t_f[:self.max_len, :]

            multi_exemplar_list.append(exemplar_t_f)

        assert len(multi_exemplar_list) == len(self.step_size_list)
        if query_density_map is not None:
            return multi_query_list, query_count, query_density_map, multi_exemplar_list, exemplar_count, exemplar_density_map
        else:
            return multi_query_list, query_count, torch.rand(128), multi_exemplar_list, exemplar_count, torch.rand(128)

    def __len__(self):
        return len(self.data_list)

# Multi scale for TransRAC
class MultiRecofit(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.Recofit.Train_val_split_path
        self.root_dir = cfg.Recofit.Root_dir
        self.window_size_list = cfg.Recofit.Window_size
        self.step_size_list = cfg.Recofit.Step_size
        self.gen_density_map = cfg.Recofit.Density_map
        self.density_map_length = cfg.Recofit.Density_map_length
        self.max_len = cfg.Recofit.Max_len
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
        # drop the first row first col
        query = query[1:, 1:]

        # Get raw exemplar signal data
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()
        # drop the first row first col
        exemplar = exemplar[1:, 1:]

        # Generate the query density map
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

        # Generate the exemplar density map
        if self.gen_density_map:
            # Generate the density map
            exemplar_density_map = GenerateDensityMap(exemplar, exemplar_count)
            exemplar_density_map = torch.FloatTensor(exemplar_density_map)
            exemplar_density_map = exemplar_density_map.unsqueeze(0).unsqueeze(0)
            exemplar_density_map = F.interpolate(exemplar_density_map, size=(self.density_map_length),
                                                 mode='linear')
            # Interpolate to 128
            exemplar_density_map = exemplar_density_map * exemplar_count / exemplar_density_map.sum()
            exemplar_density_map = exemplar_density_map.squeeze()
        else:
            exemplar_density_map = None


        multi_query_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the query
            query_seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < query_seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < query_seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: query_seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            multi_query_list.append(query_t_f)

        assert len(multi_query_list) == len(self.step_size_list)

        multi_exemplar_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the exemplar
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
                    query_t_f = torch.cat((exemplar_t_f, final_window_feat), 0)
                    break

            if exemplar_t_f.shape[0] > self.max_len:
                exemplar_t_f = exemplar_t_f[:self.max_len, :]

            multi_exemplar_list.append(exemplar_t_f)

        assert len(multi_exemplar_list) == len(self.step_size_list)
        if query_density_map is not None:
            return multi_query_list, query_count, query_density_map, multi_exemplar_list, exemplar_count, exemplar_density_map
        else:
            return multi_query_list, query_count, torch.rand(128), multi_exemplar_list, exemplar_count, torch.rand(128)

    def __len__(self):
        return len(self.data_list)

# Multi scale for TransRAC
class MultiCrossfit(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.Crossfit.Train_val_split_path
        self.root_dir = cfg.Crossfit.Root_dir
        self.window_size_list = cfg.Crossfit.Window_size
        self.step_size_list = cfg.Crossfit.Step_size
        self.gen_density_map = cfg.Crossfit.Density_map
        self.density_map_length = cfg.Crossfit.Density_map_length
        self.max_len = cfg.Crossfit.Max_len
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

        # Get raw exemplar signal data
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()

        # Generate the query density map
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

        # Generate the exemplar density map
        if self.gen_density_map:
            # Generate the density map
            exemplar_density_map = GenerateDensityMap(exemplar, exemplar_count)
            exemplar_density_map = torch.FloatTensor(exemplar_density_map)
            exemplar_density_map = exemplar_density_map.unsqueeze(0).unsqueeze(0)
            exemplar_density_map = F.interpolate(exemplar_density_map, size=(self.density_map_length),
                                                 mode='linear')
            # Interpolate to 128
            exemplar_density_map = exemplar_density_map * exemplar_count / exemplar_density_map.sum()
            exemplar_density_map = exemplar_density_map.squeeze()
        else:
            exemplar_density_map = None


        multi_query_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the query
            query_seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < query_seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < query_seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: query_seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            multi_query_list.append(query_t_f)

        assert len(multi_query_list) == len(self.step_size_list)

        multi_exemplar_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the exemplar
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
                    query_t_f = torch.cat((exemplar_t_f, final_window_feat), 0)
                    break

            if exemplar_t_f.shape[0] > self.max_len:
                exemplar_t_f = exemplar_t_f[:self.max_len, :]

            multi_exemplar_list.append(exemplar_t_f)

        assert len(multi_exemplar_list) == len(self.step_size_list)
        if query_density_map is not None:
            return multi_query_list, query_count, query_density_map, multi_exemplar_list, exemplar_count, exemplar_density_map
        else:
            return multi_query_list, query_count, torch.rand(128), multi_exemplar_list, exemplar_count, torch.rand(128)

    def __len__(self):
        return len(self.data_list)

# Multi scale for TransRAC
class MultiWeakcounter(Dataset):

    def __init__(self, cfg,  data_split = 'Train'):

        train_val_split_path = cfg.Weakcounter.Train_val_split_path
        self.root_dir = cfg.Weakcounter.Root_dir
        self.window_size_list = cfg.Weakcounter.Window_size
        self.step_size_list = cfg.Weakcounter.Step_size
        self.gen_density_map = cfg.Weakcounter.Density_map
        self.density_map_length = cfg.Weakcounter.Density_map_length
        self.max_len = cfg.Weakcounter.Max_len
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
        # print(query_dir)
        query = query.to_numpy()
        # print('qqqqqqq', query.shape)

        # Get raw exemplar signal data
        exemplar = pd.read_csv(exemplar_dir, header=None)
        exemplar = exemplar.to_numpy()

        # Generate the query density map
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

        # Generate the exemplar density map
        if self.gen_density_map:
            # Generate the density map
            exemplar_density_map = GenerateDensityMap(exemplar, exemplar_count)
            exemplar_density_map = torch.FloatTensor(exemplar_density_map)
            exemplar_density_map = exemplar_density_map.unsqueeze(0).unsqueeze(0)
            exemplar_density_map = F.interpolate(exemplar_density_map, size=(self.density_map_length),
                                                 mode='linear')
            # Interpolate to 128
            exemplar_density_map = exemplar_density_map * exemplar_count / exemplar_density_map.sum()
            exemplar_density_map = exemplar_density_map.squeeze()
        else:
            exemplar_density_map = None


        multi_query_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the query
            query_seq_len = query.shape[0]
            query_t_f = None
            query_t = torch.FloatTensor(query)  # L x C
            start_index = 0
            while (start_index + self.window_size) < query_seq_len:
                window_feat = query_t[start_index: start_index + self.window_size, :]
                # 10 x 6
                window_feat = window_feat.permute(1, 0)
                # print(window_feat.shape)
                window_feat = window_feat.reshape(1, -1)
                if query_t_f is None:
                    query_t_f = window_feat
                else:
                    query_t_f = torch.cat((query_t_f, window_feat), 0)

                if (start_index + self.window_size) < query_seq_len:
                    start_index += self.step_size
                else:
                    final_window_feat = query_t[start_index: query_seq_len, :]
                    final_window_feat = final_window_feat.reshape(1, -1)
                    query_t_f = torch.cat((query_t_f, final_window_feat), 0)
                    break

            if query_t_f.shape[0] > self.max_len:
                query_t_f = query_t_f[:self.max_len, :]

            multi_query_list.append(query_t_f)

        assert len(multi_query_list) == len(self.step_size_list)

        multi_exemplar_list = []
        # Multi scale query
        for index in range(len(self.step_size_list)):
            self.window_size = self.window_size_list[index]
            self.step_size = self.step_size_list[index]

            # Resample the exemplar
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
                    query_t_f = torch.cat((exemplar_t_f, final_window_feat), 0)
                    break

            if exemplar_t_f.shape[0] > self.max_len:
                exemplar_t_f = exemplar_t_f[:self.max_len, :]

            multi_exemplar_list.append(exemplar_t_f)

        assert len(multi_exemplar_list) == len(self.step_size_list)
        if query_density_map is not None:
            return multi_query_list, query_count, query_density_map, multi_exemplar_list, exemplar_count, exemplar_density_map
        else:
            return multi_query_list, query_count, torch.rand(128), multi_exemplar_list, exemplar_count, torch.rand(128)

    def __len__(self):
        return len(self.data_list)