# Import modules
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, InMemoryDataset

# Class to create graph dataset.
class od_flow_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    # No raw files to be processed.
    @property
    def raw_file_names(self):
        return []

    # Read all files in the "processed" folder including key-word "graph". 
    @property
    def processed_file_names(self):
        lst_graphs = os.listdir(self.processed_dir)
        lst_graphs_fil = [i for i in lst_graphs if "graph" in i]        
        return lst_graphs_fil

    # Number of all files.
    def len(self):
        return len(self.processed_file_names)

    # How to return graph data: Read it and return it.
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'graph_{:04d}.pt'.format(idx+1)))
        return data
    
# Class to create in memory graph data.
class od_flow_graphs_inMemory(InMemoryDataset):
    
    def __init__(self, root, lst_path_graphs:str, transform=None):        
        self.data_list = list()
        self.lst_path_graphs_tmp = lst_path_graphs
        # Init of parenet class should be done after declairing needed lists.
        super().__init__(root, transform) 
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.data_list = [torch.load(path) for path in self.lst_path_graphs_tmp]        
        torch.save(self.collate(self.data_list), self.processed_paths[0])
        
# Next classes...