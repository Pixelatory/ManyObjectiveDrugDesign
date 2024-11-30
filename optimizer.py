import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger, Chem
from torch import nn
from tqdm import tqdm
import selfies as sf

from MTLBERT.model import PredictionModel
from QuickVinaTwoGPU.docking import calculateDockingScore
import build_vocab
from model import ContrastiveReLSO
import torch.nn.functional as F
import sys

sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))

from SA_Score.sascorer import calculateScore as calculateSAScore


# CHANGE THESE
pop_size = 2000  # Can be changed, but should match what's set on PlatEMO
latent_model_path = 'runs/relso_selfies_contrastive/version_0/last.ckpt'
latent_vocab_file = 'selfies_vocab.txt'
admet_vocab_file = 'allmolgen198_vocab.txt'
optimizer_boundary_path = 'optimizer_boundary_relsocontrast.pt'
max_seq_len = 200
device = 'cpu'  # may need to change device in device.py too
selfies = True

admet_model_path = 'MTL-BERT_model.pt'

# There are some more variables below that can be changed.
# END OF CHANGE THESE


def get_admet_encoder(vocab, additional_model_tokens=None):
    """
        Additional tokens can be added for model_str2num. This may be useful for
        adding additional task-specific tokens like in MTLBERT.
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<GLOBAL>': 3,
        '<SEP>': 4,  # if fragment cannot be directly encoded, separate fragment's SMILES encoding
    }

    vocab_str2num = {}
    for i, j in enumerate(vocab):
        vocab_str2num[j] = len(vocab_str2num) + i

    for token in additional_model_tokens:
        model_str2num[token] = len(model_str2num) + len(vocab_str2num)

    return model_str2num, vocab_str2num


class ReLSOOptimizerModel:
    def __init__(self, latent_model_path, admet_model_path, max_seq_len, latent_vocab_file, admet_vocab_file, selfies, device):
        # encoder/decoders
        latent_vocab = build_vocab.load_vocab_from_file(latent_vocab_file)
        admet_vocab = build_vocab.load_vocab_from_file(admet_vocab_file)
        self.latent_model_str2num, self.latent_vocab_str2num = build_vocab.get_single_encoder(latent_vocab)
        self.latent_model_num2str, self.latent_vocab_num2str = build_vocab.get_decoders(self.latent_model_str2num,
                                                                            self.latent_vocab_str2num)

        additional_model_tokens = []
        for i in range(100):
            additional_model_tokens.append(f'<p{i}>')

        self.admet_model_str2num, self.admet_vocab_str2num = get_admet_encoder(admet_vocab, additional_model_tokens)
        self.max_seq_len = max_seq_len

        # loading ADMET prediction model
        self.admet_model = PredictionModel(num_layers=8, d_model=256, dff=256 * 4, num_heads=8,
                                           vocab_size=(len(self.admet_model_str2num) + len(self.admet_vocab_str2num)),
                                           dropout_rate=0.1, reg_nums=10, clf_nums=19, maximum_positional_encoding=300)
        self.admet_model.load_state_dict(torch.load(admet_model_path, map_location=torch.device('cpu'))["model_state_dict"])
        self.admet_model.eval()
        self.admet_model = self.admet_model.to('cpu')

        self.latent_model = ContrastiveReLSO.load_from_checkpoint(latent_model_path)

        self.latent_model.eval()
        self.latent_model.to(device)
        self.selfies = selfies

        self.softmax = nn.Softmax(dim=2)

        self.device = device

    def encode_seq(self, seq, latent_model=True):
        if latent_model:
            model_str2num = self.latent_model_str2num
            vocab_str2num = self.latent_vocab_str2num
        else:
            model_str2num = self.admet_model_str2num
            vocab_str2num = self.admet_vocab_str2num

        if self.selfies:
            encoding = build_vocab.selfies_encode_molecule(seq, model_str2num, vocab_str2num)
        else:
            encoding = build_vocab.smiles_encode_molecule(seq, model_str2num, vocab_str2num)

        encoding = [model_str2num['<GLOBAL>']] + encoding
        encoding = torch.from_numpy(np.array(encoding)).to(device)
        encoding = F.pad(encoding, pad=(0, self.max_seq_len - len(encoding)), value=0)
        return encoding

    def seq_to_emb(self, seq):
        encoding = self.encode_seq(seq)
        encoding = encoding.unsqueeze(0)
        with torch.no_grad():
            z_rep, _ = self.latent_model.encode(encoding)
        return z_rep

    def emb_to_seq(self, z_rep):
        with torch.no_grad():
            z_rep = z_rep.to(self.latent_model.device)
            out = self.latent_model.decode(z_rep).permute(0, 2, 1)
            out = self.softmax(out)
            out = torch.argmax(out, dim=2).tolist()
            if self.selfies:
                return [sf.decoder(build_vocab.selfies_decode_molecule(num_arr, self.latent_model_num2str, self.latent_vocab_num2str)) for
                        num_arr in
                        out]
            else:
                return [build_vocab.smiles_decode_molecule(num_arr, self.latent_model_num2str, self.latent_vocab_num2str) for num_arr in
                        out]

    def properties(self, seqs):
        encodings = torch.stack([self.encode_seq(seq) for seq in seqs]).to(self.device)
        with torch.no_grad():
            return self.admet_model(encodings)


def get_optimizer():
    return ReLSOOptimizerModel(latent_model_path, admet_model_path, max_seq_len, latent_vocab_file,
                               admet_vocab_file, selfies, device)


def read_matrix_file(file_path):
    with open(file_path, 'r') as f:
        numbers = []
        for line in f.readlines():
            numbers.append([])
            for number in line.split(','):
                try:
                    number = float(number)
                except ValueError:
                    print(number)
                    raise Exception
                numbers[-1].append(number)
    return numbers


def write_matrix_file(tmp_file_path, file_path, content):
    with open(tmp_file_path, 'w') as f:
        if type(content[0]) is list:
            for arr in content:
                arr = [str(n) for n in arr]
                f.write(','.join(arr) + '\n')
        elif type(content[0]) is float:
            arr = [str(n) for n in content]
            f.write(','.join(arr) + '\n')
        else:
            raise Exception(f"Not sure how to write {content} as matrix file")
    os.rename(tmp_file_path, file_path)


def convert_obj(objs, objs_used, obj_optim_type):
    """
        Convert objectives that are maximization to minimization by multiplying by -1.
        PlatEMO optimization algorithms assume minimization, so we comply by doing so.
    """
    max_objs = np.argwhere(obj_optim_type[objs_used] == 'max').squeeze()
    objs[:, max_objs] *= -1
    return objs


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    # all_clf_heads and all_reg_heads from the trained MTL-BERT in the exact ordering as it was trained
    all_clf_heads = ['pampa_ncats', 'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins',
                     'cyp2c19_veith', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp1a2_veith', 'cyp2c9_veith',
                     'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels',
                     'cyp3a4_substrate_carbonmangels', 'AMES', 'DILI', 'Skin_Reaction', 'Carcinogens_Lagunin',
                     'ClinTox', 'hERG']
    all_reg_heads = ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb',
                     'hydrationfreeenergy_freesolv', 'ppbr_az', 'vdss_lombardo',
                     'half_life_obach', 'clearance_hepatocyte_az',
                     'clearance_microsome_az', 'LD50_Zhu']
    clf_obj_type = np.array(['max', 'max', 'max', 'max', 'max', 'unk', 'unk', 'unk', 'unk', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min'])
    reg_obj_type = np.array(['max', 'max', 'max', 'unk', 'min', 'unk', 'unk', 'unk', 'unk', 'min'])
    all_additional_objectives = ['Binding_Affinity', 'SA_Score']
    optimizer = ReLSOOptimizerModel(latent_model_path, admet_model_path, max_seq_len, latent_vocab_file,
                                    admet_vocab_file, selfies, device)

    ###################################################################
    # VARIABLES BELOW HERE ARE FOR THE MOST PART CHANGEABLE AS NEEDED #
    ###################################################################

    # Paths used for communication with PlatEMO.
    py_EMB_path = Path("PlatEMO 4.2/py_EMB.txt")
    py_EMB_tmp_path = Path("PlatEMO 4.2/py_EMB_tmp.txt")
    py_OBJ_path = Path("PlatEMO 4.2/py_OBJ.txt")
    py_OBJ_tmp_path = Path("PlatEMO 4.2/py_OBJ_tmp.txt")
    py_upper_bound_path = Path("PlatEMO 4.2/py_UPPER.txt")
    py_upper_bound_tmp_path = Path("PlatEMO 4.2/py_UPPER_tmp.txt")
    py_lower_bound_path = Path("PlatEMO 4.2/py_LOWER.txt")
    py_lower_bound_tmp_path = Path("PlatEMO 4.2/py_LOWER_tmp.txt")
    matlab_repair_emb_path = Path("PlatEMO 4.2/matlab_REPAIR_EMB.txt")
    py_M_path = Path("PlatEMO 4.2/py_M.txt")
    py_M_tmp_path = Path("PlatEMO 4.2/py_M_tmp.txt")
    py_N_path = Path("PlatEMO 4.2/py_N.txt")
    py_N_tmp_path = Path("PlatEMO 4.2/py_N_tmp.txt")
    py_init_pop_path = Path("PlatEMO 4.2/py_init_pop.txt")
    py_init_pop_tmp_path = Path("PlatEMO 4.2/py_init_pop_tmp.txt")
    py_shutdown_path = Path("PlatEMO 4.2/py_SHUTDOWN.txt")
    py_new_run_path = Path("PlatEMO 4.2/py_NEW_RUN.txt")

    # Docking directory paths
    lig_dir = '/home/nick/Desktop/ManyObjectiveDrugDesign/QuickVinaTwoGPU/ligand_files'
    out_dir = '/home/nick/Desktop/ManyObjectiveDrugDesign/QuickVinaTwoGPU/output'
    log_dir = '/home/nick/Desktop/ManyObjectiveDrugDesign/QuickVinaTwoGPU/log'
    conf_dir = '/home/nick/Desktop/ManyObjectiveDrugDesign/QuickVinaTwoGPU/config'
    vina_cwd = '/home/nick/Desktop/ManyObjectiveDrugDesign/QuickVinaTwoGPU'
    protein_file = "/home/nick/Desktop/ManyObjectiveDrugDesign/QuickVinaTwoGPU/proteins/LPA1-7yu4.pdbqt"

    reg_objectives = ['LD50_Zhu', 'solubility_aqsoldb']
    clf_objectives = ['bioavailability_ma', 'ClinTox']
    additional_objectives = ['Binding_Affinity', 'SA_Score']

    print(reg_objectives, clf_objectives)
    # Convert regression objectives to tensor index from ADMET prediction model & task verification
    for i in range(len(reg_objectives)):
        for j in range(len(all_reg_heads)):
            if all_reg_heads[j] == reg_objectives[i]:
                reg_objectives[i] = j
                break
        else:
            raise Exception(f"Unknown regression objective given: {reg_objectives[i]}")

    # Convert classification objectives to tensor index from ADMET prediction model & task verification
    for i in range(len(clf_objectives)):
        for j in range(len(all_clf_heads)):
            if all_clf_heads[j] == clf_objectives[i]:
                clf_objectives[i] = j
                break
        else:
            raise Exception(f"Unknown classification objective given: {clf_objectives[i]}")

    print(reg_objectives, clf_objectives)

    for obj in additional_objectives:
        if obj not in all_additional_objectives:
            raise Exception(f"Unknown additional objective given: {obj}")

    def sa_score(smi):
        try:
            return calculateSAScore(Chem.MolFromSmiles(smi))
        except:
            return 10.0

    upper_bound, lower_bound = torch.load(optimizer_boundary_path, map_location=torch.device('cpu'))  # calculated from generate_optimizer_boundary.py

    df = pd.read_csv('allmolgen_198max_SMILES_SELFIES_tokenlen.csv')['smiles']  # sampling from this for initial population

    pbar = tqdm()
    print(f"Total objectives selected: {len(reg_objectives)} reg, {len(clf_objectives)} clf, and {len(additional_objectives)} additional")
    invalid = 0
    total = 0

    while True:
        if py_shutdown_path.exists():
            os.remove(py_shutdown_path)
            if total > 0:
                print(f"Invalid solutions {invalid / total * 100}")
            print("Found python shutdown file.")
            exit(1)
        elif py_new_run_path.exists():
            """
            New run file found, so write all necessary information using files so MATLAB can read it.
            """
            if total > 0:
                print(f"Invalid solutions {invalid / total * 100}")
            print(read_matrix_file(py_new_run_path))
            random_state = int(read_matrix_file(py_new_run_path)[0][0])
            print(random_state)
            init_pop = df.sample(pop_size, replace=True, random_state=random_state).tolist()
            init_pop = [optimizer.seq_to_emb(seq).squeeze().tolist() for seq in init_pop]
            write_matrix_file(py_init_pop_tmp_path, py_init_pop_path, init_pop)

            write_matrix_file(py_upper_bound_tmp_path, py_upper_bound_path, [upper_bound.tolist()])
            write_matrix_file(py_lower_bound_tmp_path, py_lower_bound_path, [lower_bound.tolist()])
            write_matrix_file(py_M_tmp_path, py_M_path, [[len(reg_objectives) + len(clf_objectives) + len(additional_objectives)]])  # M is num of objectives
            write_matrix_file(py_N_tmp_path, py_N_path, [[pop_size]])  # N is pop size
            os.remove(py_new_run_path)
        elif matlab_repair_emb_path.exists() and \
                not py_EMB_path.exists() and \
                not py_OBJ_path.exists():
            """
            First, repair the sequences by taking latent space, converting to sequence,
            then using encoder on sequences to get latent space result.

            Objectives get calculated here.
            
            Inspired from work done in CDDD & MSO by Winter et al.
            """

            tb = time.time()
            py_EMB = torch.tensor(read_matrix_file(matlab_repair_emb_path))
            py_EMB = optimizer.emb_to_seq(py_EMB)
            py_EMB = torch.stack([optimizer.seq_to_emb(seq) for seq in py_EMB]).squeeze()
            t1 = time.time() - tb

            desc = f"py_EMB: {t1}"

            # Calculate properties using appropriate models
            tb = time.time()
            seqs = optimizer.emb_to_seq(py_EMB)
            res = optimizer.properties(seqs)
            t2 = time.time() - tb

            desc += f", Models: {t2}"

            for seq in seqs:
                total += 1
                if Chem.MolFromSmiles(seq) is None:
                    invalid += 1

            # Collect objectives and prepare for writing back into file for matlab to read
            tb = time.time()
            clf_res = res['clf'][:, clf_objectives]
            reg_res = res['reg'][:, reg_objectives]
            print(reg_res, clf_res)
            clf_res = convert_obj(clf_res, clf_objectives, clf_obj_type)
            reg_res = convert_obj(reg_res, reg_objectives, reg_obj_type)
            print(reg_res, clf_res)
            obj = torch.cat([clf_res, reg_res], dim=1)

            if 'SA_Score' in additional_objectives:
                sa_obj = torch.tensor([sa_score(smi) for smi in seqs]).unsqueeze(dim=1)
                obj = torch.cat([obj, sa_obj], dim=1)

            if 'Binding_Affinity' in additional_objectives:
                print("Docking")
                dock_obj = torch.tensor([calculateDockingScore(smi, protein_file, lig_dir, out_dir, log_dir, conf_dir, vina_cwd) for smi in tqdm(seqs)]).unsqueeze(dim=1)
                print(dock_obj)
                obj = torch.cat([obj, dock_obj], dim=1)

            obj = obj.tolist()
            write_matrix_file(py_OBJ_tmp_path, py_OBJ_path, obj)
            write_matrix_file(py_EMB_tmp_path, py_EMB_path, py_EMB.tolist())
            os.remove(matlab_repair_emb_path)

            # print(f"pyOBJ {time.time() * 1000}")  # used for debugging deadlock w/ MatLab
            t3 = time.time() - tb
            desc += f", py_OBJ: {t3}"

            pbar.update(1)
            pbar.set_description(desc)
