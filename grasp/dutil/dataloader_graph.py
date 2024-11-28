import torch
from torch_geometric.data import InMemoryDataset, Data
import logging
logger = logging.getLogger(__name__)


class GraphDataset(InMemoryDataset):

    def __init__(self, x, x_label, x_zc,edges,batch_labels, transform=None):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.x_data = Data(x=torch.FloatTensor(x), edge_index=torch.LongTensor(edges).T, y=torch.LongTensor(x_label))
        self.x_zc = Data(x=torch.FloatTensor(x_zc))
        self.batch_labels = Data(x=torch.LongTensor(batch_labels))

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.x, self.x_zc,self.batch_labels