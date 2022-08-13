# Import modules.
import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

# Function to convert adjcency matrix into adjcency list.
def conv_adj_mat (in_str_file_adjMat: str, in_str_file_edge_idx: str):
    # Import raw adjcency matrix csv file.
    df_adj = pd.read_csv(in_str_file_adjMat, index_col= "idx")
    arr_adj = df_adj.values
    # Check if adjcency matrix is symmetric.
    if not np.array_equal(arr_adj, arr_adj.T):
        print("Adjcency matrix is not symmetric. Check again!")
        return
    else:
        # If symmetric, store non-zero indices pair. 
        lst_edge_index = []
        for idx_row in range(df_adj.shape[0]):
            for idx_col in range(df_adj.shape[1]):
                if df_adj.iat[idx_row, idx_col] == 0:
                    continue
                else:
                    lst_edge_index.append([int(idx_row), int(idx_col)])
    # Make edge_index array and store it.
    arr_adj_2cols = np.array(lst_edge_index)
    np.save(in_str_file_edge_idx, arr_adj_2cols)
    # Notice to user.
    print("TASK DONE, CHECK OUTPUT DIR!!")
    return

# Function to make graphs.
def make_graphs (
    in_dir_od:str, in_dir_flow:str, in_file_edge_idx:str,
    in_dir_graphs:str, 
):
    # Import file list for od matrix.
    lst_df_od = os.listdir(in_dir_od)
    lst_df_od_fil = [i for i in lst_df_od if "od_sample" in i]
    
    # Import file list for flow matrix.
    lst_df_flow = os.listdir(in_dir_flow)
    lst_df_flow_fil = [i for i in lst_df_flow if "edgeInfo" in i]
    
    # Import edge_index array file.
    arr_edge_idx = np.load(in_file_edge_idx)
    tensor_edge_idx = torch.tensor(arr_edge_idx, dtype=torch.long)
    
    # LOOP_1: Each od & flow sample
    nrOdFlow = len(lst_df_od_fil)
    for idx in range(nrOdFlow):
        idx += 1
        str_idx_tmp1 = "{:04d}".format(idx)
        str_file_od_tmp1 = list(filter(lambda x: str_idx_tmp1 in x, lst_df_od_fil))[0]
        str_file_flow_tmp1 = list(filter(lambda x: str_idx_tmp1 in x, lst_df_flow_fil))[0]
        str_path_od_tmp1 = in_dir_od + "/" + str_file_od_tmp1
        str_path_flow_tmp1 = in_dir_flow + "/" + str_file_flow_tmp1
        df_od_tmp1 = pd.read_csv(str_path_od_tmp1, index_col= 0)
        df_flow_tmp1 = pd.read_csv(str_path_flow_tmp1, index_col= 0)
        
        # LOOP_2: Each node (edge in the real map...)
        # for idx_node in range(df_od_tmp1.shape[0]):
        #     if idx_node == 0:
        #         arr_x_tmp2 = df_od_tmp1.iloc[idx_node, :].values
        #         arr_y_tmp2 = np.array(
        #             [df_flow_tmp1.iat[idx_node, 1], df_flow_tmp1.iat[idx_node, 2]]
        #         )
        #     else:
        #         arr_x_tmp2 = np.vstack((arr_x_tmp2, df_od_tmp1.iloc[idx_node, :].values))
        #         arr_y_cur_tmp2 = np.array(
        #             [df_flow_tmp1.iat[idx_node, 1], df_flow_tmp1.iat[idx_node, 2]]
        #         )
        #         arr_y_tmp2 = np.vstack((arr_y_tmp2, arr_y_cur_tmp2))
        
        # Define ndarrays.
        arr_x_tmp1 = df_od_tmp1.values
        arr_y_tmp1 = df_flow_tmp1.values        
        # Define Tensors.
        tensor_x_tmp1 = torch.tensor(arr_x_tmp1, dtype=torch.int)
        tensor_y_tmp1 = torch.tensor(arr_y_tmp1, dtype=torch.float)
        # Define Graph and store it.
        graph_tmp1 = Data(
            x= tensor_x_tmp1, y= tensor_y_tmp1,
            edge_index= tensor_edge_idx.t().contiguous()
        )
        str_path_graph_tmp1 = in_dir_graphs + "/" + "graph_{:04d}.pt".format(idx)
        torch.save(graph_tmp1, str_path_graph_tmp1)
        # Notice to user about status.
        print("Process Done {}/{}...".format(idx, nrOdFlow+1))
    
    # Final notive to user.
    print("TASK DONE. CHECK OUTPUT DIR !!!")

# Function to make graphs from historical data.    
def make_history_graphs (
    in_dir_od:str, in_dir_flow:str, in_file_edge_idx:str,
    in_dir_graphs:str, 
):
    # Import file list for od matrix.
    lst_df_od = os.listdir(in_dir_od)
    lst_df_od_fil = [i for i in lst_df_od if "_od_" in i]
    
    # Import edge_index array file.
    arr_edge_idx = np.load(in_file_edge_idx)
    tensor_edge_idx = torch.tensor(arr_edge_idx, dtype=torch.long)
    
    # LOOP_1: Each od & flow sample
    nrOdFlow = len(lst_df_od_fil)
    idx = 0
    for str_file_od in lst_df_od_fil:
        idx += 1
        str_file_flow_tmp1 = str_file_od.replace("_od_", "_flow_")
        str_path_od_tmp1 = osp.join(in_dir_od, str_file_od)
        str_path_flow_tmp1 = osp.join(in_dir_flow, str_file_flow_tmp1)
        df_od_tmp1 = pd.read_csv(str_path_od_tmp1, index_col= 0)
        df_flow_tmp1 = pd.read_csv(str_path_flow_tmp1, index_col= 0)
        # Define ndarrays.
        arr_x_tmp1 = df_od_tmp1.values
        arr_y_tmp1 = df_flow_tmp1.values        
        # Define Tensors.
        tensor_x_tmp1 = torch.tensor(arr_x_tmp1, dtype=torch.float)
        tensor_y_tmp1 = torch.tensor(arr_y_tmp1, dtype=torch.float)
        # Define Graph and store it.
        graph_tmp1 = Data(
            x= tensor_x_tmp1, y= tensor_y_tmp1,
            edge_index= tensor_edge_idx.t().contiguous()
        )
        str_path_graph_tmp1 = in_dir_graphs + "/" + "graph_{:04d}.pt".format(idx)
        torch.save(graph_tmp1, str_path_graph_tmp1)
        # Notice to user about status.
        print("Process Done {}/{}...".format(idx, nrOdFlow))
    
    # Final notive to user.
    print("TASK DONE. CHECK OUTPUT DIR !!!")