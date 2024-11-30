from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Prediction_Dataset, Pretrain_Collater, Finetune_Collater, kfolds, load_encoders
from sklearn.metrics import r2_score, roc_auc_score
from metrics import AverageMeter, Records_R2, Records_AUC, Records_RMSE, Records_Acc

from model import PredictionModel, BertModel
import argparse
from device import device

parser = argparse.ArgumentParser()
parser.add_argument('--smiles-head', nargs='+', default=['SMILES'], type=str)
parser.add_argument('--clf-heads', nargs='+',
                    default=['pampa_ncats', 'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins',
                             'cyp2c19_veith', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp1a2_veith', 'cyp2c9_veith',
                             'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels',
                             'cyp3a4_substrate_carbonmangels', 'AMES', 'DILI', 'Skin_Reaction', 'Carcinogens_Lagunin',
                             'ClinTox', 'hERG'], type=str)
parser.add_argument('--reg-heads', nargs='+', default=['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb',
                                                       'hydrationfreeenergy_freesolv', 'ppbr_az', 'vdss_lombardo',
                                                       'half_life_obach', 'clearance_hepatocyte_az',
                                                       'clearance_microsome_az', 'LD50_Zhu'], type=list)
args = parser.parse_args()

def main(seed):
    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'small_weights'}
    medium = {'name': 'Medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights'}

    arch = medium  ## small 3 4 128   medium: 6 6  256     large:  12 8 516

    k = 5  # number of folds
    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']

    dff = d_model * 4

    model_save_name = 'smiles_finetune'  # change for different file name when saving model

    model_str2num, smiles_str2num, frag_str2num = load_encoders(False, smiles_vocab_path="../allmolgen198_vocab.txt")
    vocab_size = len(model_str2num) + len(smiles_str2num)

    np.random.seed(seed=seed)

    dfs = []
    columns = set()
    for reg_head in args.reg_heads:
        df = pd.read_csv('data/reg/{}.csv'.format(reg_head))
        # NORMALIZING THE REGRESSION DATA
        if reg_head == 'solubility_aqsoldb':
            print('solubility_aqsoldb', df[reg_head].mean(), df[reg_head].std())
        elif reg_head == 'LD50_Zhu':
            print('LD50_Zhu', df[reg_head].mean(), df[reg_head].std())
        original = df[reg_head].copy()
        t = df[reg_head].mean()
        tt = df[reg_head].std()
        df[reg_head] = (df[reg_head] - df[reg_head].mean()) / (df[reg_head].std())
        if reg_head == 'solubility_aqsoldb' or reg_head == 'LD50_Zhu':
            print(original, ((df[reg_head] * tt) + t))
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        dfs.append(df)
        columns.update(df.columns.to_list())
    for clf_head in args.clf_heads:
        df = pd.read_csv('data/clf/{}.csv'.format(clf_head))
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        dfs.append(df)
        columns.update(df.columns.to_list())

    all_folds = kfolds(dfs, k)
    model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                            dropout_rate=0.1, reg_nums=len(args.reg_heads), clf_nums=len(args.clf_heads), maximum_positional_encoding=300)

    # change path inside torch.load
    model.encoder.load_state_dict(torch.load('weights/07-07-2023-12-53-50_medium_weights_allmolgen_smiles_best.pt')["model_encoder_state_dict"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-4, betas=(0.9, 0.98))

    train_loss = AverageMeter()
    test_loss = AverageMeter()

    train_aucs = Records_AUC()
    test_aucs = Records_AUC()
    test_accs = Records_Acc()

    train_r2 = Records_R2()
    test_r2 = Records_R2()
    test_rmse = Records_RMSE()

    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_func2 = torch.nn.MSELoss(reduction='none')

    def train_step(x, properties):
        model.train()
        clf_true = properties['clf']
        reg_true = properties['reg']
        properties_pred = model(x)

        clf_pred = properties_pred['clf']
        reg_pred = properties_pred['reg']

        loss = 0

        if len(args.clf_heads) > 0:
            loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                    clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

        if len(args.reg_heads) > 0:
            loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / (
                    (reg_true != -1000).float().sum() + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(args.clf_heads) > 0:
            train_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
        if len(args.reg_heads) > 0:
            train_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
        train_loss.update(loss.detach().cpu().item(), x.shape[0])

    def test_step(x, properties):
        model.eval()
        with torch.no_grad():
            clf_true = properties['clf']
            reg_true = properties['reg']
            properties_pred = model(x)

            clf_pred = properties_pred['clf']
            reg_pred = properties_pred['reg']

            loss = 0

            if len(args.clf_heads) > 0:
                loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                        clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

            if len(args.reg_heads) > 0:
                loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / (
                        (reg_true != -1000).sum() + 1e-6)

            if len(args.clf_heads) > 0:
                test_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
                test_accs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
            if len(args.reg_heads) > 0:
                test_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
                test_rmse.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
            test_loss.update(loss.detach().cpu().item(), x.shape[0])

    log_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    for i in range(k):
        train_data = None
        test_data = None
        for fold_list in all_folds:
            train_folds = pd.concat(fold_list[:i] + fold_list[i+1:]).reset_index(drop=True)
            test_folds = fold_list[i].reset_index(drop=True)
            if train_data is None:
                train_data = train_folds
            else:
                train_data = pd.concat((train_data, train_folds), ignore_index=True).reset_index(drop=True)

            if test_data is None:
                test_data = test_folds
            else:
                test_data = pd.concat((test_data, test_folds), ignore_index=True).reset_index(drop=True)

        train_dataset = Prediction_Dataset(train_data, smiles_head=args.smiles_head,
                                           reg_heads=args.reg_heads, clf_heads=args.clf_heads, fragmentation=False,
                                           model_str2num=model_str2num, smiles_str2num=smiles_str2num, frag_str2num=frag_str2num)
        test_dataset = Prediction_Dataset(test_data, smiles_head=args.smiles_head,
                                          reg_heads=args.reg_heads, clf_heads=args.clf_heads, fragmentation=False,
                                          model_str2num=model_str2num, smiles_str2num=smiles_str2num, frag_str2num=frag_str2num)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                      collate_fn=Finetune_Collater(args))
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=Finetune_Collater(args))

        early_stop_val = float('inf')
        has_not_improved_count = 0
        testing_aucs = [[] for _ in range(len(args.clf_heads))]
        testing_accs = [[] for _ in range(len(args.clf_heads))]
        testing_r2s = [[] for _ in range(len(args.reg_heads))]
        testing_rmses = [[] for _ in range(len(args.reg_heads))]
        test_losses = []
        test_aucs.reset()
        test_accs.reset()
        test_rmse.reset()
        test_r2.reset()
        test_loss.reset()
        for epoch in range(200):
            for x, properties in tqdm(train_dataloader):
                train_step(x, properties)

            print('epoch: ', epoch, 'train loss: {:.4f}'.format(train_loss.avg))
            if len(args.clf_heads) > 0:
                clf_results = train_aucs.results()
                for num, clf_head in enumerate(args.clf_heads):
                    print('train auc {}: {:.4f}'.format(clf_head, clf_results[num]))
            if len(args.reg_heads) > 0:
                reg_results = train_r2.results()
                for num, reg_head in enumerate(args.reg_heads):
                    print('train r2 {}: {:.4f}'.format(reg_head, reg_results[num]))
            train_aucs.reset()
            train_r2.reset()
            train_loss.reset()

            for x, properties in test_dataloader:
                test_step(x, properties)
            print('epoch: ', epoch, 'test loss: {:.4f}'.format(test_loss.avg))
            if len(args.clf_heads) > 0:
                clf_results = test_aucs.results()
                for num, clf_head in enumerate(args.clf_heads):
                    print('test auc {}: {:.4f}'.format(clf_head, clf_results[num]))
                    testing_aucs[num].append(clf_results[num])
                clf_results = test_accs.results()
                for num, clf_head in enumerate(args.clf_heads):
                    print('test accs {}: {:.4f}'.format(clf_head, clf_results[num]))
                    testing_accs[num].append(clf_results[num])
            if len(args.reg_heads) > 0:
                reg_results = test_r2.results()
                for num, reg_head in enumerate(args.reg_heads):
                    print('test r2 {}: {:.4f}'.format(reg_head, reg_results[num]))
                    testing_r2s[num].append(reg_results[num])
                reg_results = test_rmse.results()
                for num, reg_head in enumerate(args.reg_heads):
                    print('test rmse {}: {:.4f}'.format(reg_head, reg_results[num]))
                    testing_rmses[num].append(reg_results[num])
            test_losses.append(test_loss.avg)

            if (test_loss.avg < early_stop_val):
                print(f"improved by {early_stop_val - test_loss.avg}")
                early_stop_val = test_loss.avg
                has_not_improved_count = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "random_state": seed,
                    "epoch": epoch,
                    "name": model_save_name,
                    "arch": arch,
                    "log_time": log_time,
                }, f'weights/{log_time}_{model_save_name}_finetuned_fold{i}_best.pt')
            else:
                has_not_improved_count += 1
                print(f"has not improved in the past f{has_not_improved_count} test epochs")

            test_aucs.reset()
            test_accs.reset()
            test_r2.reset()
            test_rmse.reset()
            test_loss.reset()

            if has_not_improved_count >= 2:
                print("Threshold of testing non-improvement reached.")
                break

        with open(f'logs/{log_time}-{model_save_name}-log.txt', 'a') as f:
            f.write(f'############## FOLD {i} ##############\n')
            for j in range(len(args.clf_heads)):
                f.write(args.clf_heads[j] + ' (AUC)\n')
                for val in testing_aucs[j]:
                    f.write(str(val) + '\n')
                f.write(args.clf_heads[j] + ' (Acc)\n')
                for val in testing_accs[j]:
                    f.write(str(val) + '\n')

            for j in range(len(args.reg_heads)):
                f.write(args.reg_heads[j] + ' (R^2)\n')
                for val in testing_r2s[j]:
                    f.write(str(val) + '\n')
                f.write(args.reg_heads[j] + ' (RMSE)\n')
                for val in testing_rmses[j]:
                    f.write(str(val) + '\n')
            f.write('test_epoch_losses\n')
            for val in test_losses:
                f.write(str(val) + '\n')


if __name__ == '__main__':
    main(42)
