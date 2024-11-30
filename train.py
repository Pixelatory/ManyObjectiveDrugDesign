import os

import torch.utils.data
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import TrainDataset, PadToMaxLenCollater, ContrastiveTrainDataset, FragNetDataset, FragNetCollater
from model import ReLSO, ContrastiveReLSO, FragNet

torch.set_float32_matmul_precision("medium")

# CHANGE THESE
task_cols = ['logp', 'sas', 'qed']
dataset_file = 'allmolgen_198max_SMILES_SELFIES_tokenlen.csv'
vocab_file = 'selfies_vocab.txt'
seq_col = 'smiles'  # sequence col from dataset file
selfies = True
max_seq_len = 200

contrastive_learning = False
model_name = 'relso'  # either 'relso' or 'fragnet'

batch_size = 64
random_state = 42  # used for dataset splitting and smiles enumeration
device = ("cuda" if torch.cuda.is_available() else "cpu")
logger_name = model_name

if selfies:
    logger_name += "_selfies"
else:
    logger_name += "_smiles"

# Model parameters
recon_weight = 1
aux_weights = [0.25, 0.25, 0.25]
zrep_l2_weight = 0.1

latent_dim = 256
d_model = 256
n_transformer_layers = 6
n_heads = 8
lr = 2e-5
dim_feedforward = 1024
dropout = 0.2
padding_vocab_idx = 0  # padding token is first

# Contrastive learning parameters
contrast_weight = 1
temperature = 0.1
n_views = 2

if contrastive_learning:
    logger_name += "_contrastive"
    batch_size //= n_views

# END OF CHANGE THESE

valid_model_names = ['relso', 'fragnet']

if model_name not in valid_model_names:
    raise Exception(f'invalid model name ')
assert len(aux_weights) == len(task_cols)

logger = TensorBoardLogger("runs", name=logger_name)

if model_name == 'fragnet':
    dataset = FragNetDataset(dataset_file, seq_col, task_cols, vocab_file, selfies, n_views, random_state, max_seq_len)
else:
    if contrastive_learning:
        dataset = ContrastiveTrainDataset(dataset_file, seq_col, task_cols, vocab_file, selfies, n_views, random_state,
                                          max_seq_len)
    else:
        dataset = TrainDataset(dataset_file, seq_col, task_cols, vocab_file, selfies)

if model_name == 'relso':
    if contrastive_learning:
        model = ContrastiveReLSO(lr, max_seq_len, len(task_cols), dataset.vocab_size(), latent_dim,
                                 n_transformer_layers,
                                 d_model, n_heads, recon_weight, aux_weights, zrep_l2_weight, contrast_weight,
                                 temperature, n_views,
                                 dim_feedforward, dropout, padding_vocab_idx)
    else:
        model = ReLSO(lr, max_seq_len, len(task_cols), dataset.vocab_size(), latent_dim, n_transformer_layers, d_model,
                      n_heads,
                      recon_weight, aux_weights, zrep_l2_weight, dim_feedforward, dropout, padding_vocab_idx)
elif model_name == 'fragnet':
    if contrastive_learning:
        model = FragNet(lr, max_seq_len, len(task_cols), dataset.vocab_size(), n_transformer_layers, d_model,
                        n_heads, recon_weight, aux_weights, zrep_l2_weight, contrast_weight, temperature, n_views,
                        dim_feedforward, dropout, padding_vocab_idx)
    else:
        raise Exception("Fragnet has no non contrastive-learning implementation, turn it on.")

train_size = int(0.7 * len(dataset))  # 70-10-20 train/val/test split
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
generator = torch.Generator().manual_seed(random_state)  # split the same way even for different executions!
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size],
                                                                         generator=generator)

if model_name == 'relso':
    collater = PadToMaxLenCollater(masked_pretrain=False, max_seq_len=max_seq_len,
                                   device=device, contrastive=contrastive_learning)
else:
    collater = FragNetCollater(max_seq_len=max_seq_len, device=device)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collater,
                              num_workers=os.cpu_count(),
                              pin_memory=True)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collater,
                              num_workers=os.cpu_count(),
                              pin_memory=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collater,
                             num_workers=os.cpu_count(),
                             pin_memory=True)

print(f"CPU count (cores): {os.cpu_count()}")
early_stopping_callback = EarlyStopping(
    monitor="valid_loss",
    min_delta=0.001,
    patience=4,
    verbose=True,
    mode="min",
    check_finite=True
)

checkpoint_callback = ModelCheckpoint(
    dirpath=logger.log_dir,
    monitor='valid_loss',
    save_last=True,
    save_top_k=2,
    mode='min',
    save_weights_only=False,
    save_on_train_epoch_end=False,
    verbose=True,
    every_n_train_steps=100,
)

trainer = pl.Trainer(
    max_epochs=-1,
    devices=1,
    accelerator="gpu",
    gradient_clip_val=1,
    logger=logger,
    #fast_dev_run=True,
    val_check_interval=0.2,
    callbacks=[early_stopping_callback, checkpoint_callback],
)

trainer.fit(model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader)

trainer.test(model, dataloaders=test_dataloader)
