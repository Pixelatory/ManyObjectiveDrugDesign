import math
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from build_vocab import selfies_encode_molecule
from device import device

from DeepFMPO.python3.Modules.build_vocab import combine_vocabs, load_vocab_from_file, get_hybrid_encoders, \
    hybrid_encode_molecule, smiles_encode_molecule

smiles_regex_pattern = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


def load_encoders(fragmentation, smiles_vocab_path='DeepFMPO/python3/Modules/allmolgen_vocab.txt',
                  smiles_frag_vocab_path='DeepFMPO/python3/Modules/allmolgen_frag_smiles_vocab.txt',
                  frag_vocab_path='DeepFMPO/python3/Modules/1000freq_vocab.txt'):
    if fragmentation:
        smiles_vocab = combine_vocabs(
            load_vocab_from_file(smiles_vocab_path) + load_vocab_from_file(smiles_frag_vocab_path))
    else:
        smiles_vocab = load_vocab_from_file(smiles_vocab_path)

    if fragmentation:
        frag_vocab = load_vocab_from_file(frag_vocab_path)
    else:
        frag_vocab = None

    # add 100 task tokens
    additional_model_tokens = []
    for i in range(100):
        additional_model_tokens.append(f'<p{i}>')

    # if fragmentation is false, just don't use frag_str2num and you're good to go
    return get_hybrid_encoders(smiles_vocab, frag_vocab, additional_model_tokens)


def randomize_smile(sml):
    m = Chem.MolFromSmiles(sml)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles


def canonical_smile(sml):
    m = Chem.MolFromSmiles(sml)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles


class Smiles_Bert_Dataset(Dataset):
    def __init__(self, path, Smiles_head, model_str2num, smiles_str2num, frag_str2num, fragmentation):
        if path.endswith('txt'):
            self.df = pd.read_csv(path, sep='\t')
        elif path.endswith('pkl'):
            with open(path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.df = pd.read_csv(path)

        self.fragmentation = fragmentation
        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num
        self.frag_str2num = frag_str2num

        if not path.endswith('pkl'):
            self.data = self.df[Smiles_head].to_numpy().reshape(-1).tolist()
            self.data = [self._char_to_idx(entry) for entry in tqdm(self.data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smiles = self.data[item]
        x, y, weights = self.numerical_smiles(smiles)
        return x, y, weights

    def numerical_smiles(self, smiles):
        # nums_list = self._char_to_idx(smiles)
        nums_list = smiles  # smiles is really the encoding
        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = self.model_str2num['<UNK>']
            elif rand < 0.9:
                if self.fragmentation:
                    nums_list[i] = int(
                        np.random.randint(len(self.model_str2num), len(self.smiles_str2num) + len(self.frag_str2num)))
                else:
                    nums_list[i] = int(np.random.randint(len(self.model_str2num), len(self.smiles_str2num)))
        x = np.array(nums_list).astype('int64')
        weights = weight.astype('float32')
        return x, y, weights

    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding

    def pickle_data_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.data, f)


class OldSmiles_Bert_Dataset(Dataset):
    def __init__(self, path, Smiles_head, model_str2num, smiles_str2num, frag_str2num, fragmentation, selfies):
        if path.endswith('txt'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)

        self.fragmentation = fragmentation
        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num  # even if selfies is used, I just keep this called smiles_str2num
        self.frag_str2num = frag_str2num
        self.selfies = selfies

        assert selfies != fragmentation

        self.data = self.df[Smiles_head].to_numpy().reshape(-1).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smiles = self.data[item]
        x, y, weights = self.numerical_smiles(smiles)
        return x, y, weights

    def numerical_smiles(self, smiles):
        nums_list = self._char_to_idx(smiles)
        # nums_list = smiles  # smiles is really the encoding
        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = self.model_str2num['<MASK>']
            elif rand < 0.9:
                if self.fragmentation:
                    nums_list[i] = int(
                        np.random.randint(len(self.model_str2num), len(self.smiles_str2num) + len(self.frag_str2num)))
                else:
                    nums_list[i] = int(np.random.randint(len(self.model_str2num), len(self.smiles_str2num)))
        x = np.array(nums_list).astype('int64')
        weights = weight.astype('float32')
        return x, y, weights

    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            if self.selfies:
                encoding = selfies_encode_molecule(seq, self.model_str2num, self.smiles_str2num)
            else:
                encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding

    def pickle_data_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.data, f)


class Prediction_Dataset(object):
    def __init__(self, df, fragmentation, model_str2num, smiles_str2num, frag_str2num, smiles_head='SMILES',
                 reg_heads=None, clf_heads=None):
        if clf_heads is None:
            clf_heads = []
        if reg_heads is None:
            reg_heads = []

        self.df = df
        self.reg_heads = reg_heads
        self.clf_heads = clf_heads

        self.smiles = self.df[smiles_head].to_numpy().reshape(-1).tolist()

        self.reg = np.array(self.df[reg_heads].fillna(-1000)).astype('float32')
        self.clf = np.array(self.df[clf_heads].fillna(-1000)).astype('int32')
        self.fragmentation = fragmentation
        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num
        self.frag_str2num = frag_str2num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        smiles = self.smiles[item]

        properties = [None, None]
        if len(self.clf_heads) > 0:
            clf = self.clf[item]
            properties[0] = clf

        if len(self.reg_heads) > 0:
            reg = self.reg[item]
            properties[1] = reg

        nums_list = self._char_to_idx(seq=smiles)
        if len(self.reg_heads) + len(self.clf_heads) > 0:
            ps = [f'<p{i}>' for i in range(len(self.reg_heads) + len(self.clf_heads))]
            nums_list = [self.model_str2num[p] for p in ps] + nums_list
        x = np.array(nums_list).astype('int32')
        return x, properties

    def numerical_smiles(self, smiles):
        smiles = self._char_to_idx(seq=smiles)
        x = np.array(smiles).astype('int64')
        return x

    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding


class NEWPrediction_Dataset(object):
    def __init__(self, df, fragmentation, model_str2num, smiles_str2num, frag_str2num, smiles_head='SMILES',
                 reg_heads=None, clf_heads=None):
        if clf_heads is None:
            clf_heads = []
        if reg_heads is None:
            reg_heads = []

        self.df = df
        self.reg_heads = reg_heads
        self.clf_heads = clf_heads

        self.smiles = self.df[smiles_head].to_numpy().reshape(-1).tolist()

        self.reg = np.array(self.df[reg_heads].fillna(-1000)).astype('float32')
        self.clf = np.array(self.df[clf_heads].fillna(-1000)).astype('int32')
        self.fragmentation = fragmentation
        self.model_str2num = model_str2num
        self.smiles_str2num = smiles_str2num
        self.frag_str2num = frag_str2num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        smiles = self.smiles[item]

        properties = [None, None]
        if len(self.clf_heads) > 0:
            clf = self.clf[item]
            properties[0] = clf

        if len(self.reg_heads) > 0:
            reg = self.reg[item]
            properties[1] = reg

        nums_list = self._char_to_idx(seq=smiles)
        if len(self.reg_heads) + len(self.clf_heads) > 0:
            ps = [f'<p{i}>' for i in range(len(self.reg_heads) + len(self.clf_heads))]
            nums_list = [self.model_str2num[p] for p in ps] + nums_list
        x = np.array(nums_list).astype('int32')
        return x, properties

    def numerical_smiles(self, smiles):
        smiles = self._char_to_idx(seq=smiles)
        x = np.array(smiles).astype('int64')
        return x

    def _char_to_idx(self, seq):
        if self.fragmentation:
            encoding = hybrid_encode_molecule(seq, self.model_str2num, self.smiles_str2num, self.frag_str2num)
        else:
            encoding = smiles_encode_molecule(seq, self.model_str2num, self.smiles_str2num)

        return [self.model_str2num['<GLOBAL>']] + encoding


class Pretrain_Collater():
    def __init__(self):
        super(Pretrain_Collater, self).__init__()

    def __call__(self, data):
        xs, ys, weights = zip(*data)

        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long().to(device)
        weights = pad_sequence([torch.from_numpy(np.array(weight)) for weight in weights], batch_first=True).float().to(
            device)

        return xs, ys, weights


class Finetune_Collater():
    def __init__(self, args):
        super(Finetune_Collater, self).__init__()
        self.clf_heads = args.clf_heads
        self.reg_heads = args.reg_heads

    def __call__(self, data):
        xs, properties_list = zip(*data)
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        properties_dict = {'clf': None, 'reg': None}

        if len(self.clf_heads) > 0:
            properties_dict['clf'] = torch.from_numpy(
                np.concatenate([p[0].reshape(1, -1) for p in properties_list], 0).astype('int32')).to(device)

        if len(self.reg_heads) > 0:
            properties_dict['reg'] = torch.from_numpy(
                np.concatenate([p[1].reshape(1, -1) for p in properties_list], 0).astype('float32')).to(device)

        return xs, properties_dict


def kfolds(datas, k):
    # Generates k similarly sized folds (may not be equally sized)
    # Note: doesn't shuffle, so do the shuffle before calling this
    result = []
    for data in datas:
        fold_size = math.ceil(len(data) / k)
        tmp_folds = []
        for i in range(k):
            tmp_folds.append(data[fold_size * i: fold_size * (i + 1)].reset_index(drop=True))
        result.append(tmp_folds)
    return result