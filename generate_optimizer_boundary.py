import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from build_vocab import load_vocab_from_file, get_single_encoder, selfies_encode_molecule, smiles_encode_molecule, \
    get_decoders, smiles_decode_molecule
from model import ReLSO, ContrastiveReLSO, FragNet

# CHANGE THESE
model = ReLSO
latent_size = 512
model_ckpt_path = 'runs/relso_selfies/version_0/last.ckpt'
vocab_file = 'selfies_vocab.txt'
selfies = True
max_seq_len = 200
device = ("cuda" if torch.cuda.is_available() else "cpu")


# END OF CHANGE THESE

class InferenceModel:
    def __init__(self, model_class, model_ckpt_path, max_seq_len, vocab_file, selfies, device):
        vocab = load_vocab_from_file(vocab_file)
        self.model_str2num, self.vocab_str2num = get_single_encoder(vocab)
        self.selfies = selfies
        self.device = device
        if model_ckpt_path is not None:
            self.model = model_class.load_from_checkpoint(model_ckpt_path)
            self.model.eval()
            self.model.to(self.device)
        self.softmax = nn.Softmax(dim=1)
        self.model_num2str, self.vocab_num2str = get_decoders(self.model_str2num, self.vocab_str2num)
        self.max_seq_len = max_seq_len

    def encode_seq(self, seqs):
        encodings = []
        if type(self.model) is FragNet:
            prepend = []
        else:
            prepend = [self.model_str2num['<GLOBAL>']]

        for seq in seqs:
            if self.selfies:
                encodings.append(prepend + selfies_encode_molecule(seq, self.model_str2num,
                                                                   self.vocab_str2num))
            else:
                encodings.append(prepend + smiles_encode_molecule(seq, self.model_str2num,
                                                                  self.vocab_str2num))
            encodings[-1] = F.pad(torch.tensor(encodings[-1]), pad=(0, self.max_seq_len - len(encodings[-1])), value=0)
        encodings = torch.stack(encodings, dim=0).to(self.device)
        return encodings

    def seq_to_emb(self, seqs):
        encodings = self.encode_seq(seqs)
        with torch.no_grad():
            if type(self.model) is FragNet:
                enc_pad_mask = (encodings == 0).bool()
                z_reps, _ = self.model.encode(encodings, src_key_padding_mask=enc_pad_mask)
            else:
                z_reps, _ = self.model.encode(encodings)
        return z_reps

    def emb_to_seq(self, z_rep):
        with torch.no_grad():
            out = self.model.decode(z_rep).squeeze(0).permute(1, 0)
            out = self.softmax(out)
            out = torch.argmax(out, dim=1).tolist()
            return smiles_decode_molecule(out, self.model_num2str, self.vocab_num2str)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    df = pd.read_csv('allmolgen_198max_SMILES_SELFIES_tokenlen.csv', chunksize=1024)
    inference_model = InferenceModel(model, model_ckpt_path, max_seq_len, vocab_file, selfies, device)

    max_tensor = torch.full(size=(latent_size,), fill_value=-float('inf')).to(device)
    min_tensor = torch.full(size=(latent_size,), fill_value=float('inf')).to(device)
    for chunk in tqdm(df):
        z_rep = inference_model.seq_to_emb(chunk["smiles"]).squeeze()
        max_z_rep = torch.max(z_rep, dim=0).values
        max_tensor = torch.maximum(max_tensor, max_z_rep)

        min_z_rep = torch.min(z_rep, dim=0).values
        min_tensor = torch.minimum(min_tensor, min_z_rep)

    torch.save([max_tensor, min_tensor], 'optimizer_boundary_relso.pt')
