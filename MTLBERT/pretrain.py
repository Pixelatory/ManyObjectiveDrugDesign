import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Pretrain_Collater, load_encoders, OldSmiles_Bert_Dataset
from device import device
from metrics import AverageMeter
from model import BertModel


def main(random_state):
    parser = argparse.ArgumentParser()
    parser.add_argument('--Smiles_head', nargs='+', default=["smiles"], type=str)
    args = parser.parse_args()

    small = {'name': 'small', 'num_layers': 4, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights'}
    medium = {'name': 'medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
    large = {'name': 'large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights'}

    arch = medium  # small 3 4 128   medium: 6 6  256     large:  12 8 516
    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    dff = d_model * 4

    model_save_name = 'allmolgen_selfies'  # change for different file name when saving model
    frag_vocab_path = '../selfies_vocab.txt'  # change me for different fragment vocabs

    model_str2num, smiles_str2num, frag_str2num = load_encoders(False, smiles_vocab_path="../selfies_vocab.txt")
    vocab_size = len(model_str2num) + len(smiles_str2num)

    model = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                      maximum_positional_encoding=300)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), 1e-4, betas=(0.9, 0.98))
    loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    def train_step(x, y, weights):
        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.detach().item(), x.shape[0])
        train_acc.update(
            ((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item() / weights.cpu().sum().item(),
            weights.cpu().sum().item())

    def test_step(x, y, weights):
        model.eval()
        with torch.no_grad():
            predictions = model(x)
            loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum() / weights.sum()

            test_loss.update(loss.detach().item(), x.shape[0])
            test_acc.update(
                ((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item() / weights.cpu().sum().item(),
                weights.cpu().sum().item())

    log_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    with open(f'logs/{log_time}_{model_save_name}-pretrain-log.txt', 'a') as f:
        f.write("max testing hasn't improved count: 2\n")
        f.write("perform test epoch every 5000 training batches\n")
        f.write(f"smiles vocab path: {frag_vocab_path}\n")
        f.write(f"random state: {random_state}\n")

    full_dataset = OldSmiles_Bert_Dataset('../allmolgen_198max_SMILES_SELFIES_tokenlen.csv',
                                          Smiles_head=args.Smiles_head, fragmentation=False,
                                          model_str2num=model_str2num, smiles_str2num=smiles_str2num,
                                          frag_str2num=frag_str2num, selfies=True)

    train_size = int(0.8 * len(full_dataset))  # 80-20 train/test split
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(
        random_state)  # split the same way even for different phases and executions!
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=Pretrain_Collater())
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=Pretrain_Collater())

    early_stop_val = float('inf')
    has_not_improved_count = 0
    for epoch in range(50):
        start = time.time()

        test_losses = []
        test_accs = []

        for (batch, (x, y, weights)) in enumerate(tqdm(train_dataloader)):
            train_step(x, y, weights)

            if batch % 500 == 0:
                print('Epoch {} Batch {} training Loss {:.4f}'.format(
                    epoch + 1, batch, train_loss.avg))
                print('training Accuracy: {:.4f}'.format(train_acc.avg))

            if batch % 5000 == 0:
                for x, y, weights in tqdm(test_dataloader):
                    test_step(x, y, weights)
                print('Test loss: {:.4f}'.format(test_loss.avg))
                print('Test Accuracy: {:.4f}'.format(test_acc.avg))
                test_losses.append(test_loss.avg)
                test_accs.append(test_acc.avg)
                if (test_loss.avg < early_stop_val + 0.001):
                    early_stop_val = test_loss.avg
                    has_not_improved_count = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_encoder_state_dict": model.encoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "random_state": random_state,
                        "epoch": epoch,
                        "name": model_save_name,
                        "arch": arch,
                        "log_time": log_time,
                    }, f'weights/{log_time}_{arch["path"]}_{model_save_name}_best.pt')
                else:
                    has_not_improved_count += 1

                print(f"has not improved in the past f{has_not_improved_count} test epochs")
                test_acc.reset()
                test_loss.reset()
                train_acc.reset()
                train_loss.reset()

            if has_not_improved_count >= 2:
                break
        print('Epoch {} is Done!'.format(epoch))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, train_loss.avg))
        print('training Accuracy: {:.4f}'.format(train_acc.avg))
        print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.avg))
        print('test Accuracy: {:.4f}'.format(test_acc.avg))

        with open(f'logs/{log_time}_{model_save_name}_pretrain_log.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}\n")
            f.write("Test Losses\n")
            for tmp in test_losses:
                f.write(str(tmp) + "\n")

            f.write("Test Accuracies\n")
            for tmp in test_accs:
                f.write(str(tmp) + "\n")
        if has_not_improved_count >= 2:
            break


if __name__ == "__main__":
    random_states = [42]
    for random_state in random_states:
        main(random_state)
