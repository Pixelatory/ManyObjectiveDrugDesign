import itertools
from typing import List
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from selfies import EncoderError
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import build_vocab


def randomize_smiles(smiles, generator):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    generator.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=False)


class TrainDataset(Dataset):
    def __init__(self, path: str, seq_col: str, task_cols: List[str], vocab_file: str, selfies: bool, additional_tokens=None):
        if path.endswith('txt'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)

        vocab = build_vocab.load_vocab_from_file(vocab_file)
        model_str2num, vocab_str2num = build_vocab.get_single_encoder(vocab, additional_tokens)

        self.model_str2num = model_str2num
        self.vocab_str2num = vocab_str2num
        self.selfies = selfies

        for tc in task_cols:
            self.df[tc] = (self.df[tc] - self.df[tc].min()) / (self.df[tc].max() - self.df[tc].min())

        self.seq_data = self.df[seq_col].to_numpy().reshape(-1).tolist()
        self.task_datas = [self.df[col].to_numpy().reshape(-1).tolist() for col in task_cols]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        seq = self.seq_data[idx]
        x, y = self.encode_seq(seq)
        task_targets = [task_data[idx] for task_data in self.task_datas]
        return x, y, task_targets

    def vocab_size(self):
        return len(self.model_str2num) + len(self.vocab_str2num)

    def encode_seq(self, seq):
        x = self._char_to_idx(seq, prepend=['<GLOBAL>'])
        y = list(x)
        return x, y

    def _char_to_idx(self, seq, prepend=None, append=None):
        if append is None:
            append = []
        if prepend is None:
            prepend = []
        if not self.selfies:
            encoding = build_vocab.smiles_encode_molecule(seq, self.model_str2num, self.vocab_str2num)
        else:
            encoding = build_vocab.selfies_encode_molecule(seq, self.model_str2num, self.vocab_str2num)
        return [self.model_str2num[e] for e in prepend] + encoding + [self.model_str2num[e] for e in append]

    def _modify_encoding(self, encoding, prepend=None, append=None):
        if append is None:
            append = []
        if prepend is None:
            prepend = []
        return [self.model_str2num[e] for e in prepend] + encoding + [self.model_str2num[e] for e in append]


class ContrastiveTrainDataset(TrainDataset):
    """
    Used for contrastive learning as SMILES are enumerated twice.
    """

    def __init__(self, path: str, seq_col: str, task_cols: List[str], vocab_file: str, selfies: bool, n_views: int,
                 random_seed: int, max_seq_len: int, additional_tokens=None):
        super().__init__(path, seq_col, task_cols, vocab_file, selfies, additional_tokens)
        self.n_views = n_views
        self.generator = np.random.default_rng(seed=random_seed)
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        seq = self.seq_data[idx]
        xs = []
        ys = []
        for _ in range(self.n_views):
            while True:
                try:
                    rand_seq = randomize_smiles(seq, self.generator)
                    x = self._char_to_idx(rand_seq, prepend=['<GLOBAL>'])
                    if len(x) < self.max_seq_len:
                        y = list(x)
                        break
                except EncoderError:
                    print("error occurred randomizing " + str(seq))
                    pass
            xs.append(x)
            ys.append(y)

        task_targets = []
        for _ in range(self.n_views):
            task_targets.append([task_data[idx] for task_data in self.task_datas])
        return xs, ys, task_targets


class FragNetDataset(ContrastiveTrainDataset):
    def __init__(self, path: str, seq_col: str, task_cols: List[str], vocab_file: str, selfies: bool, n_views: int, random_seed: int, max_seq_len: int):
        super().__init__(path, seq_col, task_cols, vocab_file, selfies, n_views, random_seed, max_seq_len, additional_tokens=['<START>', '<END>'])

    def __getitem__(self, idx):
        seq = self.seq_data[idx]
        enc_inputs = []
        dec_inputs = []
        dec_targets = []
        for i in range(self.n_views):
            while True:
                try:
                    rand_seq = randomize_smiles(seq, self.generator)
                    encoding = self._char_to_idx(rand_seq)
                    if len(encoding) < (self.max_seq_len - 1):
                        enc_input = list(self._modify_encoding(encoding))
                        dec_input = list(self._modify_encoding(encoding, prepend=['<START>']))
                        dec_target = list(self._modify_encoding(encoding, append=['<END>']))
                        break
                except:
                    print("error occurred randomizing " + str(seq))
                    pass
            enc_inputs.append(enc_input)
            dec_inputs.append(dec_input)
            dec_targets.append(dec_target)

        task_targets = []
        for i in range(self.n_views):
            task_targets.append([task_data[idx] for task_data in self.task_datas])
        return enc_inputs, dec_inputs, dec_targets, task_targets


def pad_seqs(xs, max_seq_len):
    xs = [torch.tensor(x) for x in xs]
    xs = [F.pad(x, pad=(0, max_seq_len - len(x)), value=0).unsqueeze(dim=0) for x in xs]
    return torch.cat(xs)


class FragNetCollater:
    def __init__(self, max_seq_len: int, device):
        self.max_seq_len = max_seq_len
        self.device = device
        self.pad_idx = 0

    def __call__(self, data):
        enc_inputs, dec_inputs, dec_targets, task_datas = zip(*data)
        enc_inputs = pad_seqs(list(itertools.chain(*enc_inputs)), max_seq_len=self.max_seq_len)
        dec_inputs = pad_seqs(list(itertools.chain(*dec_inputs)), max_seq_len=self.max_seq_len)
        dec_targets = pad_seqs(list(itertools.chain(*dec_targets)), max_seq_len=self.max_seq_len)

        enc_pad_mask = (enc_inputs == self.pad_idx).bool()
        dec_pad_mask = (dec_inputs == self.pad_idx).bool()

        task_datas = torch.tensor(list(itertools.chain(*task_datas)))

        return (enc_inputs, enc_pad_mask), (dec_inputs, dec_pad_mask), dec_targets, task_datas


class PadToMaxLenCollater:
    def __init__(self, masked_pretrain: bool, max_seq_len: int, device, contrastive: bool):
        super(PadToMaxLenCollater, self).__init__()
        self.masked_pretrain = masked_pretrain
        self.max_seq_len = max_seq_len
        self.device = device
        self.contrastive = contrastive

    def __call__(self, data):
        if self.masked_pretrain:
            xs, ys, weights = zip(*data)
            xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long()
            ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long()
            weights = pad_sequence([torch.from_numpy(np.array(weight)) for weight in weights],
                                   batch_first=True).float()

            return xs, ys, weights
        else:
            xs, ys, task_datas = zip(*data)

            if self.contrastive:
                xs = list(itertools.chain(*xs))
                ys = list(itertools.chain(*ys))
                task_datas = list(itertools.chain(*task_datas))

            xs = pad_seqs(xs, max_seq_len=self.max_seq_len)
            ys = pad_seqs(ys, max_seq_len=self.max_seq_len)

            task_datas = [torch.from_numpy(np.array(task_data)).unsqueeze(dim=0).float() for task_data in task_datas]
            task_datas = torch.cat(task_datas)

            return xs, ys, task_datas
