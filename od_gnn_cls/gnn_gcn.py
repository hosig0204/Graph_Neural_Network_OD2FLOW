# Import modules.
import torch
import torch.nn.functional as nnF
from torch_geometric.nn import GCNConv, Linear

# Class for GCN test model. Class to observe how GCN layer is working.
class gnn_GCN_CONV_test(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, 
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim_x, in_dim_hid)] + 
            [GCNConv(in_dim_hid, in_dim_hid) for i in range(in_num_layers - 2)] +
            [GCNConv(in_dim_hid, in_dim_hid)]
        )

        # Graph Linear layer as final one.
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        conv_1 = self.convs[0]
        out = conv_1(node_x, edge_idx)
        
        return out
    
# Class for ordinary GCN model.
class gnn_GCN_CONV_LIN(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, 
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim_x, in_dim_hid)] + 
            [GCNConv(in_dim_hid, in_dim_hid) for i in range(in_num_layers - 2)] +
            [GCNConv(in_dim_hid, in_dim_hid)]
        )

        # Graph Linear layer as final one.
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        if self.lv_norm and self.lv_dropout:            # Use normalisation & drop-out.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                if self.training:
                    node_x_trans = nnF.dropout(node_x_trans, p= self.rat_dropout)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        elif self.lv_norm and not(self.lv_dropout):     # Use normalisation.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        else:                                           # Just basics...
            for conv in self.convs[:-1]:
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans
            node_x = self.convs[-1](node_x, edge_idx)
            out = self.out_linear(node_x)
        # Return output.
        return out
    
# Class for GCN model with 2 LIN layer.
class gnn_GCN_CONV_LIN2(torch.nn.Module):
    
    def __init__(self,  in_dim_x: int, in_dim_y: int, in_dim_hid: int, 
                        in_num_layers: int = 2, in_lc_norm: bool = False, in_lc_dropout: bool = False,
                        in_rat_dropout: float = 0.0, 
        ) -> None:
        
        super().__init__()
        # Here, node level layers should be defined.
        # Number of GCN layers & Normarlization layers can be adjusted.

        # Graph Convolution layers.
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_dim_x, in_dim_hid)] + 
            [GCNConv(in_dim_hid, in_dim_hid) for i in range(in_num_layers - 2)] +
            [GCNConv(in_dim_hid, in_dim_hid)]
        )

        # Graph Linear layer as final one.
        self.inter_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_hid)
        self.out_linear = Linear(in_channels= in_dim_hid, out_channels= in_dim_y)
        
        # Node feature normalization layers. 
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features= in_dim_hid) for i in range(in_num_layers - 1)]
        )
        
        # Store internal attributes.
        self.lv_norm = in_lc_norm
        self.lv_dropout = in_lc_dropout        
        self.rat_dropout = in_rat_dropout
        
    def reset_parameters(self):
        
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.bns:
            bn.reset_parameters()
            
        self.out_linear.reset_parameters()
    
    def forward(self, loaded_data):
        # Load node feature and edge index information.
        node_x = loaded_data.x.type(torch.float)
        edge_idx = loaded_data.edge_index
        # Stack layers.
        if self.lv_norm and self.lv_dropout:            # Use normalisation & drop-out.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                if self.training:
                    node_x_trans = nnF.dropout(node_x_trans, p= self.rat_dropout)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            node_x = self.inter_linear(node_x)
            out = self.out_linear(node_x)
        elif self.lv_norm and not(self.lv_dropout):     # Use normalisation.
            for conv, bn in zip(self.convs[:-1], self.bns):
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = bn(node_x_trans)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans            
            node_x = self.convs[-1](node_x, edge_idx)
            node_x = self.inter_linear(node_x)
            out = self.out_linear(node_x)
        else:                                           # Just basics...
            for conv in self.convs[:-1]:
                node_x_trans = conv(node_x, edge_idx)
                node_x_trans = nnF.relu(node_x_trans)
                node_x = node_x_trans
            node_x = self.convs[-1](node_x, edge_idx)
            node_x = self.inter_linear(node_x)
            out = self.out_linear(node_x)
        # Return output.
        return out

# Next models...