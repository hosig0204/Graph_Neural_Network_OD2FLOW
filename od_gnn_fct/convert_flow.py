# Import modules.
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

# DEFNITION OF EDGES
DIC_EDGE = {
    "alpha" : ["alpha_e", "alpha_w"], "beta" : ["beta_e", "beta_w"],
    "1" : ["1_e", "1_w"], "2" : ["2_e", "2_w"], "3" : ["3_e", "3_w"],
    "4" : ["4_e", "4_w"], "5" : ["5_e", "5_w"], "6" : ["6_e", "6_w"],
    "a" : ["a_n", "a_s"], "b" : ["b_n", "b_s"], "c" : ["c_n", "c_s"],
    "d" : ["d_n", "d_s"], "e" : ["e_n", "e_s"], "f" : ["f_n", "f_s"],
}

# DEFNITION OF EDGE SEQUENCE
EDGE_SEQ = [
    "alpha", "a", "b", "c", "d", "e", "f",
    "1", "2", "3", "4", "5", "6", "beta",
]

# Helper function: Return upper level edge name.
def searchEdge (in_id_edge: str, in_dic_edge: dict):
    for key, val in in_dic_edge.items():
        if in_id_edge in val:
            return key
        else:
            continue
    return 'NaN'

# Function to convert edge flow data.
def read_edge_flow (in_str_dir_flow: str, in_str_dir_out: str):
    
    # List up edge flow data file in xml format.
    lst_edge_flow = os.listdir(in_str_dir_flow)
    lst_edge_flow_fil = [i for i in lst_edge_flow if "edgeInfo" in i]
    total_len = len(lst_edge_flow_fil)
    total_idx = 0
    
    # LOOP_1: Each edge flow data file (.xml)
    for edge_flow_xml in lst_edge_flow_fil:
        total_idx += 1
        file_path_tmp1 = in_str_dir_flow + "/" + edge_flow_xml
        tree_tmp1 = ET.parse(file_path_tmp1)
        root_tmp1 = tree_tmp1.getroot()
        nrEdges_tmp1 = len(root_tmp1[0].findall("edge"))
        lst_df_single_tmp1 = []
        
        # LOOP_2: Eage edge info in the file.
        for edge_idx in range(nrEdges_tmp1):
            dic_tmp2= {}
            dic_tmp2["edge_id"] = root_tmp1[0][edge_idx].get("id", "NaN")
            edge_density_tmp2 = float(root_tmp1[0][edge_idx].get("density", 'NaN'))
            edge_speed_tmp2 = float(root_tmp1[0][edge_idx].get("speed", 'NaN'))
            edge_flow_tmp2 = edge_density_tmp2 * edge_speed_tmp2 * 3.6
            dic_tmp2["edge_flow"] = edge_flow_tmp2
            df_single_tmp2 = pd.DataFrame(
                [list(dic_tmp2.values())],
                columns= list(dic_tmp2.keys())                
            )
            lst_df_single_tmp1.append(df_single_tmp2)
        
        # Concatnate edge flows per each file.
        df_concat_rows_tmp1 = pd.concat(
            lst_df_single_tmp1,
            ignore_index= True, sort= False
        )
        # Add upper edge "id" column.
        df_concat_rows_tmp1["id"] = df_concat_rows_tmp1["edge_id"].apply(
            lambda x: searchEdge(x, DIC_EDGE)
        )
        # Create final output dataframe.
        df_out_tmp1 = pd.DataFrame(np.full((len(EDGE_SEQ),3),np.nan),
            columns= ["id", "flow_0", "flow_1"]
        )
        # LOOP_3: Insert proper infomation into output dataframe.
        # Predefined edge sequence will be applied here.        
        for idx_out_row in range(len(EDGE_SEQ)):
            # edge id from sequence.
            id_tmp3 = EDGE_SEQ[idx_out_row]
            # child edge ids from dictionary.
            edge_id_0_tmp3 = DIC_EDGE[id_tmp3][0]
            edge_id_1_tmp3 = DIC_EDGE[id_tmp3][1]
            # flow value from chied edge.
            flow_0_tmp3 = df_concat_rows_tmp1[
                df_concat_rows_tmp1["edge_id"] == edge_id_0_tmp3
            ]["edge_flow"].values
            flow_1_tmp3 = df_concat_rows_tmp1[
                df_concat_rows_tmp1["edge_id"] == edge_id_1_tmp3
            ]["edge_flow"].values
            # Insert values into output df.
            df_out_tmp1.iat[idx_out_row, 0] = id_tmp3
            df_out_tmp1.iat[idx_out_row, 1] = float(flow_0_tmp3)
            df_out_tmp1.iat[idx_out_row, 2] = float(flow_1_tmp3)
        
        # Create output file name and store it.
        out_fName_tmp1 = edge_flow_xml.replace(".xml", ".csv")
        out_path_tmp1 = in_str_dir_out + "/" + out_fName_tmp1
        df_out_tmp1.to_csv(out_path_tmp1, index= False)
        
        # Notice to user.
        print("Processing...{}/{}".format(total_idx, total_len))
    
    # Notice to user.
    print("TASK DONE, CHECK OUTPUT DIR!!")
    
# Next fuunctions...