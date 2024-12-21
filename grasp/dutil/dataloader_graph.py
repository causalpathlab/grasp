import torch
from torch_geometric.data import InMemoryDataset, Data

class GraphDataset(InMemoryDataset):

    def __init__(self, x, x_label, x_be_edges, x_ge_edges, x_zc, x_batch_labels, transform=None):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.x_be_data = Data(x=torch.FloatTensor(x), edge_index=torch.LongTensor(x_be_edges).T, y=torch.LongTensor(x_label))
        self.x_ge_data = Data(x=torch.LongTensor(x_ge_edges).T)
        self.x_zc_data = Data(x=torch.FloatTensor(x_zc))
        self.x_batch_data = Data(x=torch.LongTensor(x_batch_labels))

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return self.x_be_data, self.x_ge_data, self.x_zc_data,self.x_batch_data
    
    