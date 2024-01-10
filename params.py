from model.util import setup_seed
import torch

num_in = 576
num_class = 11


# setup_seed(50)

train_path = './data/trainSet.mat'
test_path = './data/testSet.mat'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

C = 4
dt = 1e-3
structure = [576, 1024, 11]
w_init_std = 0.1
w_init_mean = 0.
epoch_num = 100
lr = 6e-4
fc_drop = 0.5
delta = 0
C_sim = 0.97
TCA_slice = 4

load_if = False
save_dir = './checkpoints'
log_dir = './log'
prefix = 'TD_SOM_'
pre_train = False
model_path = "./checkpoints/STCA-max-96.20.t7"

print(device, dtype)
