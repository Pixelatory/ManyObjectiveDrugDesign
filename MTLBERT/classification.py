from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Prediction_Dataset, Pretrain_Collater, Finetune_Collater, smiles_str2num
from sklearn.metrics import r2_score, roc_auc_score
from metrics import AverageMeter, Records_R2, Records_AUC, Records_RMSE, Records_Acc

import os
from model import PredictionModel, BertModel
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--smiles-head', nargs='+', default=['SMILES'], type=str)
parser.add_argument('--clf-heads', nargs='+', default=['PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma', 'BBB_Martins', 'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'AMES', 'DILI', 'skin_reaction', 'Carcinogens_Lagunin', 'ClinTox', 'hERG'], type=str)
parser.add_argument('--reg-heads', nargs='+', default=['Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'HydrationFreeEnergy_FreeSolv', 'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'LD50_Zhu'], type=list)
args = parser.parse_args()


# 'Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub'
# 'caco2', 'logD','logS','tox','PPB'

def main(seed):
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['BBB', 'FDAMDD',  'Pgp_sub']

    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'small_weights'}
    medium = {'name': 'Medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights'}

    arch = medium  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']

    dff = d_model * 4
    vocab_size = len(smiles_str2num)
    testings = [[]] * (len(args.clf_heads) + len(args.reg_heads))

    np.random.seed(seed=seed)

    dfs = []
    columns = set()
    for reg_head in args.reg_heads:
        df = pd.read_csv('data/reg/{}.csv'.format(reg_head))
        # NORMALIZING THE REGRESSION DATA
        df[reg_head] = (df[reg_head] - df[reg_head].mean()) / (df[reg_head].std())
        dfs.append(df)
        columns.update(df.columns.to_list())
    for clf_head in args.clf_heads:
        df = pd.read_csv('data/clf/{}.csv'.format(clf_head))
        dfs.append(df)
        columns.update(df.columns.to_list())

    train_temps = []
    test_temps = []
    valid_temps = []

    for df in dfs:
        temp = pd.DataFrame(index=range(len(df)))
        for column in df.columns:
            temp[column] = df[column]
        temp = temp.sample(frac=1).reset_index(drop=True)
        train_temp = temp[:int(0.8 * len(temp))]
        train_temps.append(train_temp)

        test_temp = temp[int(0.8 * len(temp)):int(0.9 * len(temp))]
        test_temps.append(test_temp)

        valid_temp = temp[int(0.9 * len(temp)):]
        valid_temps.append(valid_temp)

    train_df = pd.concat(train_temps, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_temps, axis=0).reset_index(drop=True)
    valid_df = pd.concat(valid_temps, axis=0).reset_index(drop=True)

    train_dataset = Prediction_Dataset(train_df, smiles_head=args.smiles_head,
                                       reg_heads=args.reg_heads, clf_heads=args.clf_heads)
    test_dataset = Prediction_Dataset(test_df, smiles_head=args.smiles_head,
                                      reg_heads=args.reg_heads, clf_heads=args.clf_heads)
    valid_dataset = Prediction_Dataset(valid_df, smiles_head=args.smiles_head,
                                       reg_heads=args.reg_heads, clf_heads=args.clf_heads)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=Finetune_Collater(args))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=Finetune_Collater(args))
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=Finetune_Collater(args))

    # x, property = next(iter(train_dataset))
    model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                            dropout_rate=0.1, reg_nums=len(args.reg_heads), clf_nums=len(args.clf_heads))




    model.load_state_dict(torch.load('weights/medium_weights_finetuned_weights_6.pt'))
    model = model.to(device)



    #model.encoder.load_state_dict(torch.load('weights/medium_weights_bert_encoder_weightsmedium_final.pt'))
    #model = model.to(device)
    print(device)
    # if pretraining:
    #     model.encoder.load_state_dict(torch.load())
    #     print('load_wieghts')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-4, betas=(0.9, 0.98))
    # lm = lambda x:x/10*(5e-5) if x<10 else (5e-5)*10/x
    # lms = LambdaLR(optimizer,[lm])

    train_loss = AverageMeter()
    test_loss = AverageMeter()
    valid_loss = AverageMeter()

    train_aucs = Records_AUC()
    test_aucs = Records_AUC()
    test_accs = Records_Acc()
    valid_aucs = Records_AUC()

    train_r2 = Records_R2()
    test_r2 = Records_R2()
    test_rmse = Records_RMSE()
    valid_r2 = Records_R2()


    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_func2 = torch.nn.MSELoss(reduction='none')

    stopping_monitor = 0

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

    def valid_step(x, properties):
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
                valid_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
            if len(args.reg_heads) > 0:
                valid_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
            valid_loss.update(loss.detach().cpu().item(), x.shape[0])



    for x, properties in test_dataloader:
        test_step(x, properties)
    print('test loss: {:.4f}'.format(test_loss.avg))
    if len(args.clf_heads) > 0:
        clf_results = test_aucs.results()
        for num, clf_head in enumerate(args.clf_heads):
            print('test auc {}: {:.4f}'.format(clf_head, clf_results[num]))
        clf_results = test_accs.results()
        for num, clf_head in enumerate(args.clf_heads):
            print('test acc {}: {:.4f}'.format(clf_head, clf_results[num]))
    if len(args.reg_heads) > 0:
        reg_results = test_r2.results()
        for num, reg_head in enumerate(args.reg_heads):
            print('test r2 {}: {:.4f}'.format(reg_head, reg_results[num]))
        reg_results = test_rmse.results()
        for num, reg_head in enumerate(args.reg_heads):
            print('test rmse {}: {:.4f}'.format(reg_head, reg_results[num]))

    exit(1)


    early_stop_val = float('inf')
    has_not_improved_count = 0
    testings = [[] for _ in range(len(args.clf_heads) + len(args.reg_heads))]
    test_losses = []
    index = 0
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

        for x, properties in valid_dataloader:
            valid_step(x, properties)
        print('epoch: ', epoch, 'valid loss: {:.4f}'.format(valid_loss.avg))
        if len(args.clf_heads) > 0:
            clf_results = valid_aucs.results()
            for num, clf_head in enumerate(args.clf_heads):
                print('valid auc {}: {:.4f}'.format(clf_head, clf_results[num]))
        if len(args.reg_heads) > 0:
            reg_results = valid_r2.results()
            for num, reg_head in enumerate(args.reg_heads):
                print('valid r2 {}: {:.4f}'.format(reg_head, reg_results[num]))

        valid_aucs.reset()
        valid_r2.reset()
        valid_loss.reset()

        for x, properties in test_dataloader:
            test_step(x, properties)
        print('epoch: ', epoch, 'test loss: {:.4f}'.format(test_loss.avg))
        if len(args.clf_heads) > 0:
            clf_results = test_aucs.results()
            for num, clf_head in enumerate(args.clf_heads):
                print('test auc {}: {:.4f}'.format(clf_head, clf_results[num]))
                testings[index].append(clf_results[num])
                index += 1
        if len(args.reg_heads) > 0:
            reg_results = test_r2.results()
            for num, reg_head in enumerate(args.reg_heads):
                print('test r2 {}: {:.4f}'.format(reg_head, reg_results[num]))
                testings[index].append(reg_results[num])
                index += 1
        test_losses.append(test_loss.avg)
        index = 0

        if (test_loss.avg < early_stop_val + 0.001):
            early_stop_val = test_loss.avg
            has_not_improved_count = 0
        else:
            has_not_improved_count += 1

        print(f"has not improved in f{has_not_improved_count} test epochs")

        test_aucs.reset()
        test_r2.reset()
        test_loss.reset()

        if has_not_improved_count >= 20:
            print("Threshold of non testing improvement reached.")
            break
        else:
            torch.save(model.state_dict(), f'weights/{arch["path"]}_finetuned_weights_{epoch + 1}.pt')
            torch.save(model.encoder.state_dict(), f'weights/{arch["path"]}_encoder_finetuned_weights_{epoch + 1}.pt')

    torch.save(model.state_dict(), f'weights/{arch["path"]}_finetuned_weights_final.pt')
    torch.save(model.encoder.state_dict(), f'weights/{arch["path"]}_encoder_finetuned_weights_final.pt')
    with open(f'{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-log.txt', 'w') as f:
        for i in range(len(testings)):
            if i < len(args.clf_heads):
                f.write(args.clf_heads[i] + ' (AUC)\n')
            else:
                f.write(args.reg_heads[i - len(args.clf_heads)] + ' (R^2)\n')
            for val in testings[i]:
                f.write(str(val) + '\n')
        f.write('test_epoch_losses\n')
        for val in test_losses:
            f.write(str(val) + '\n')

if __name__ == '__main__':
    main(7)
