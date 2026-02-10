# config for count-by-twos experiment

# data
dataset = 'count_by_twos'
data_dir = 'data/basic'
out_dir = 'out'

# model - small model for this simple task
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# training
batch_size = 64
block_size = 64  # context length
max_iters = 200  # start with 200 for quick test, then change to 2000

# learning rate
learning_rate = 1e-3
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-4

# evaluation
eval_interval = 100
eval_iters = 20
log_interval = 10

# system
device = 'cpu' # or 'cpu' if no GPU
compile = False  # set to True if you have PyTorch 2.0+

# checkpoint
always_save_checkpoint = True
