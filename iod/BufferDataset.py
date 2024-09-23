import torch
from torch.utils.data import Dataset, DataLoader

class BufferDataset(Dataset):
    def __init__(self, data, len):
        self.data = data
        self.len = len

    def __getitem__(self, index):
        keys = ['obs' , 'next_obs', 'sub_goal', 's_0', 'options', 'next_options', 'dones', 'actions']  
        epoch_data = {}
        for i in range(len(keys)):
            key = keys[i]
            if key in ['obs', 'next_obs']:
                key_ = key + '_pixel'
                epoch_data[key] = torch.tensor(self.data[key_][index], dtype=torch.float32)
            elif key in ['s_0', 'sub_goal']:
                relative_index = self.data[key][index][0]
                epoch_data[key] = torch.tensor(self.data['obs_pixel'][index + relative_index], dtype=torch.float32)
            else:
                epoch_data[key] = torch.tensor(self.data[key][index], dtype=torch.float32)   
                
        return epoch_data

    def __len__(self):
        return self.len










