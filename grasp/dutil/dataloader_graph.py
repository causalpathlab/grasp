import torch
from torch_geometric.data import InMemoryDataset, Data
import logging
logger = logging.getLogger(__name__)


class GraphDataset(InMemoryDataset):

    def __init__(self, x, x_label, x_zc,edges, edges_sec, batch_labels, transform=None):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.x_data = Data(x=torch.FloatTensor(x), edge_index=torch.LongTensor(edges), y=torch.LongTensor(x_label))
        self.x_g = Data(x=torch.LongTensor(edges_sec))
        self.x_zc = Data(x=torch.FloatTensor(x_zc))
        self.batch_labels = Data(x=torch.LongTensor(batch_labels))
        
        self.num_samples = len(x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x_data, self.x_zc, self.x_g, self.batch_labels