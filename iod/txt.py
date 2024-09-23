import torch
from torch.utils.data import Dataset, DataLoader
# import multiprocessing
# multiprocessing.set_start_method('spawn')

# 自定义数据集
class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 示例数据
data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
labels = torch.tensor([0, 1, 0, 1])

# 创建数据集和数据加载器
dataset = SimpleDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, multiprocessing_context='fork')

# 使用 DataLoader
for batch_data, batch_labels in dataloader:
    print(f"Data: {batch_data}, Labels: {batch_labels}")