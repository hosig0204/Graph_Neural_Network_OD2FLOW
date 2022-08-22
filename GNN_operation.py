# Import necessary modules.
import torch
import time
import pandas as pd
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError as MAPE
from od_gnn_cls.gnn_dataset import *
from od_gnn_cls.gnn_gat import *
import plotly.graph_objects as go
from od_gnn_fct.user_utill import fileListCreator

# |******************************************************************|
# |    Preparation                                                   |
# |******************************************************************|

# Set key value.
now = time.localtime()
key = "{:02d}{:02d}_{:02d}_{:02d}".format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

# Set up plotting templete.
tempelete_01_white = dict(
    layout = go.Layout(
        # Layout properties
        title_font_size= 14,
        title_x= 0.1,
        font_size= 11,
        font_color= "#000000",
        font_family= "Times New Roman",
        margin_b = 65,
        margin_l = 60,
        margin_r = 30,
        margin_t = 50,
        plot_bgcolor= "#ffffff",
        # X axis properties
        xaxis_color= "#000000",
        xaxis_linecolor= "#000000",
        xaxis_ticks= "inside",        
        xaxis_tickfont_color= "#000000",
        xaxis_tickfont_family= "Times New Roman",
        xaxis_mirror= True,
        xaxis_showline= True,
        xaxis_showgrid= False,
        # Y axis properties
        yaxis_color= "#000000",
        yaxis_linecolor= "#000000",
        yaxis_ticks= "inside",
        yaxis_tickfont_color= "#000000",
        yaxis_tickfont_family= "Times New Roman",
        yaxis_mirror= True,
        yaxis_showline= True,
        yaxis_showgrid= False,
    )
)

# Device check.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# |******************************************************************|
# |    Read Graphs & Load Dataset                                    |
# |******************************************************************|

# Import dataset.

lv_inMemory_dataset = True      # Check if to use in memory dataset.
if lv_inMemory_dataset:
    # In momory dataset.
    lst_path_graphs = fileListCreator('data_pyg/graph_history') # List of graph file paths.
    dataset_od_flow = od_flow_graphs_inMemory(
        root= "dataset_history_pyg_inMemory",
        lst_path_graphs= lst_path_graphs
    )
else:
    # For larger files.
    str_dir_dataset_root = "./dataset_history_pyg"              # Just include root directory.
    dataset_od_flow = od_flow_graphs(str_dir_dataset_root)      # Import graph dataset.
    
data_sample = dataset_od_flow[0]                            # Sample data to extract dimension info.
int_dim_node_features = int(data_sample.num_node_features)  # Node feature dimension.
int_dim_node_out = int(data_sample.y.shape[1])              # Node output value dimension.
int_num_nodes = int(data_sample.num_nodes)                  # Number of nodes.

#  Define size of datasets
int_size_dataset = len(dataset_od_flow)
float_rat_train = 0.8     # Sum of ratios should be 1.
float_rat_test = 0.2 
int_size_train = int(int_size_dataset*float_rat_train)
int_size_test = int(int_size_dataset*float_rat_test)

# Split original dataset.
dataset_train, dataset_test = random_split(dataset_od_flow, [int_size_train, int_size_test])

# Print size information.
print("Graph sets have been split.")
print("Total Graphs: {}".format(int_size_dataset))
print("   Train Graphs: {}".format(int_size_train))
print("   Test Graphs: {}".format(int_size_test))

# Let's have batched (surely from PyG, not basic Pytorch)
int_size_batch = 32 # Some number as 2^x... (e.g. 32,64 ..)
loaded_train = DataLoader(dataset_train, batch_size= int_size_batch, shuffle= True)
loaded_test = DataLoader(dataset_test, batch_size= int_size_batch, shuffle= True)
# Print out process.
print("Data has been loaded. Batch Size: {}".format(int_size_batch))

# |******************************************************************|
# |    Define Train & Test Loops                                     |
# |******************************************************************|

# Loop for training, Parameters should be reset before training.
def train_loop(dataloader, model, loss_fn, optimizer):
    
    model.train() # Activate drop-out layers.
    
    loss = 0
    nr_used_data = 0
    size = len(dataloader.dataset)
    
    for batch, data in enumerate(dataloader):
        # Prediction from forward calculation.
        # Loss term calculation.
        data.to(device)
        pred = model(data)
        loss = loss_fn(pred, data.y)
        # Back-propagation and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Reporting.
        nr_used_data += int(len(data.x) / int_num_nodes)
        if batch % 10 == 0 : # For each 10 batchs.
            loss_val = loss.item()
            
            print("Loss: {loss:>.5f}  [{current:>5d}/{size:>5d}]".format(loss=loss_val, current= nr_used_data, size= size))

# Loop for test.
@torch.no_grad()    # Context-manager that disabled gradient calculation.
def test_loop(dataloader, model, loss_fn):
    
    model.eval() # Deactivate drop-out layer.
    
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    clc_mape = MAPE().to(device)
    test_loss, correct = 0, 0
    
    for data in dataloader:
        data.to(device)
        pred = model(data)
        test_loss += loss_fn(pred, data.y).item()
        correct += clc_mape(pred, data.y)
        # Below is for classification !!
        # correct += (pred.argmax(1) == y).type(torch.float).sum().item() 

    test_loss /= num_batches
    correct /= num_batches
    
    print(f"Test Error: \n MAPE: {(correct*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return [correct*100, test_loss]

# |******************************************************************|
# |    Define GNN Model                                              |
# |******************************************************************|

# Model importing with relevant arguments.
model_GATv2_3CONV = gnn_GATv2_CONV_LIN(
    in_dim_x= int_dim_node_features, in_dim_y= int_dim_node_out,
    in_dim_hid= int_dim_node_features, in_num_layers= 3, 
    in_lc_norm= False, in_lc_dropout= False
).to(device)

# Print-out model spec.
# Actual layer structure is not same as printed results!
print(model_GATv2_3CONV)

# |******************************************************************|
# |    Train GNN Model                                               |
# |******************************************************************|

# Set timer.
start_time = time.time()
# Set Loss function.
loss_fn = torch.nn.MSELoss()
# Set Optimizer and Learning Step.
optimizer = torch.optim.Adam(model_GATv2_3CONV.parameters(), lr=0.001)
# Set Learning Step Scheduler.
lst_milestones= list(range(100, 1000, 50))
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones= lst_milestones, gamma= 0.85, verbose= True
)

# In case you have pre-trained parameters.
lv_use_checkPoint = False

if lv_use_checkPoint:
    # Load learnt parameters.
    str_path_cp = "YOUR_CHECKPOINT_HERE" # NOTE: YOUR CHECKPOINT HERE!
    cp = torch.load(str_path_cp)    
    model_GATv2_3CONV.load_state_dict(cp["model_state_dict"])
    optimizer.load_state_dict(cp["optimizer_state_dict"])
    scheduler.load_state_dict(cp["scheduler_state_dict"])
    start_epoch = cp["epoch"]    
    print(f"Training will start from the last checkpoint. Start epoch: {start_epoch+1}")
else:
    # Otherwise, reset all parameters.
    model_GATv2_3CONV.reset_parameters()
    start_epoch = 0

# Set number of maximum epochs.
target_epoch = 1000
tot_epochs = start_epoch + target_epoch

# Define lists for reporting.
lst_mape = []
lst_loss = []
lst_time = []

# Training LOOP.
for t in range(start_epoch+1, tot_epochs+1):
    
    # Update learning step.
    # scheduler.step()
    
    print(f"Epoch {t}\n-------------------------------")
    train_loop(loaded_train, model_GATv2_3CONV, loss_fn, optimizer)
    mape, loss = test_loop(loaded_test, model_GATv2_3CONV, loss_fn)
    time_epoch = time.time() - start_time
    
    lst_time.append(int(time_epoch))
    lst_mape.append(float(mape))
    lst_loss.append(float(loss))
    
    # Break training loop when MAPE is lower than threshold.
    if mape <= 2.5:
        break
        
print("Done!")

# |******************************************************************|
# |    REPORT & STORE Learnt Parameters                              |
# |******************************************************************|

# Store learnt parameters when training is finished.
torch.save(model_GATv2_3CONV.state_dict(), f"./learnt_parameters/{key}__GNN_params.pth")

# Make reporting dataframe and store it.
len_hist_learn = len(lst_loss)
dic_hist_learn = {
    "Iteration" : range(1, len_hist_learn + 1),
    "Time": lst_time,
    "MSE_Loss" : lst_loss,
    "MAPE" : lst_mape
}
df_hist_learn = pd.DataFrame(dic_hist_learn)
df_hist_learn.to_csv(f"./reporting/{key}__GNN_report.csv")

print("Model parameters are saved. Reporting DataFrame is saved.")

# Store check point.
torch.save(
    {
        "epoch": t,
        "model_state_dict": model_GATv2_3CONV.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,        
    }, f"./checkpoints/{key}_checkpoint.tar"
)

print("Check point has been created.")

# |******************************************************************|
# |    Populate Figures & Store them                                 |
# |******************************************************************|

fig_iter = go.Figure()

fig_iter.add_trace(
    go.Scatter(
        x= df_hist_learn["Iteration"],
        y= df_hist_learn["MSE_Loss"],
        line_color = "#000000",
    )
)

fig_iter.update_layout(
    title= "GATv2",
    xaxis_title= "Number Iteration",
    yaxis_title= "MSE_Loss [NrVeh/hr]^2",
    width= 500,
    height= 350,
    template= tempelete_01_white,
)

fig_iter.update_xaxes(
    range= [0, df_hist_learn["Iteration"].max() + 5]
)

fig_iter.update_yaxes(
    range= [0,df_hist_learn["MSE_Loss"].max()]
)

fig_iter.write_html(f"./figures/{key}__GNN_Iter_MSE.html")

fig_iter_02 = go.Figure()

fig_iter_02.add_trace(
    go.Scatter(
        x= df_hist_learn["Iteration"],
        y= df_hist_learn["MAPE"],
        line_color = "#000000",
    )
)

fig_iter_02.update_layout(
    title= "GATv2",
    xaxis_title= "Number Iteration",
    yaxis_title= "MAPE [%]",
    width= 500,
    height= 350,
    template= tempelete_01_white,
)

fig_iter_02.update_xaxes(
    range= [0, df_hist_learn["Iteration"].max() + 5]
)

fig_iter_02.update_yaxes(
    range= [0,df_hist_learn["MAPE"].max()]
)

fig_iter_02.write_html(f"./figures/{key}__GNN_Iter_MAPE.html")

print("Relevant figures are saved.")