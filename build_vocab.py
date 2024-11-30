import re
import pandas as pd
import selfies as sf
from rdkit import Chem
from selfies import EncoderError
from tqdm import tqdm

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


def build_smiles_vocab(file_name, smiles_col, chunksize=None):
    df = pd.read_csv(file_name, chunksize=chunksize)

    if chunksize is None:
        smiles_vocab = []
        for smi in tqdm(df[smiles_col]):
            splits = re.findall(SMI_REGEX_PATTERN, smi)
            for token in splits:
                if token not in smiles_vocab:
                    smiles_vocab.append(token)
    else:
        smiles_vocab = []
        for chunk in tqdm(df):
            for smi in chunk[smiles_col]:
                splits = re.findall(SMI_REGEX_PATTERN, smi)
                for token in splits:
                    if token not in smiles_vocab:
                        smiles_vocab.append(token)
    return smiles_vocab


def build_selfies_vocab(file_name, smiles_col):
    df = pd.read_csv(file_name)

    selfies_vocab = []
    for smi in tqdm(df[smiles_col]):
        try:
            selfie = sf.encoder(smi)
            for split in sf.split_selfies(selfie):
                if split not in selfies_vocab:
                    selfies_vocab.append(split)
        except EncoderError:
            pass
    return selfies_vocab


def unused_vocab(file_name, smiles_col, vocab):
    df = pd.read_csv(file_name)
    unused = set()
    for smi in df[smiles_col]:
        splits = re.findall(SMI_REGEX_PATTERN, smi)
        for token in splits:
            if token not in vocab:
                unused.add(token)
    return unused


def save_vocab_to_file(vocab, file_name):
    with open(file_name, 'w') as f:
        for word in vocab:
            f.write(word + '\n')


def load_vocab_from_file(file_name):
    with open(file_name, 'r') as f:
        return [line.strip() for line in f.readlines()]


def frags_above_freq_thresh(frag_freq, threshold):
    frag_vocab = []
    for key, value in frag_freq.items():
        if value >= threshold:
            frag_vocab.append(key)
    return frag_vocab


def get_single_encoder(vocab, additional_model_tokens=None):
    """
        Additional tokens can be added for model_str2num (not at end). This may be useful for
        adding additional task-specific tokens like in MTLBERT.
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<GLOBAL>': 3,
    }

    for token in additional_model_tokens:
        model_str2num[token] = len(model_str2num)

    vocab_str2num = {}
    for i, j in enumerate(vocab):
        vocab_str2num[j] = len(model_str2num) + i

    return model_str2num, vocab_str2num


def get_hybrid_encoders(smiles_vocab, additional_model_tokens=None):
    """
        Additional model encodings are added to end of vocab encodings.
    """
    if additional_model_tokens is None:
        additional_model_tokens = []

    model_str2num = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<GLOBAL>': 3,
    }

    smiles_str2num = {}
    for i, j in enumerate(smiles_vocab):
        smiles_str2num[j] = len(model_str2num) + i

    for token in additional_model_tokens:
        model_str2num[token] = len(model_str2num) + len(smiles_vocab)

    return model_str2num, smiles_str2num


def get_decoders(*args):
    return [{i: j for j, i in entry.items()} for entry in args]


def smiles_encode_molecule(smi, model_str2num, smiles_str2num):
    """
        Encodes molecule using SMILES. If atom not in vocab, use <UNK>
    """
    return [smiles_str2num.get(char, model_str2num['<UNK>']) for char in re.findall(SMI_REGEX_PATTERN, smi)]


def selfies_encode_molecule(seq, model_str2num, vocab_str2num):
    """
        Encodes molecule using SELFIES. If token not in vocab, use <UNK>
    """
    try:
        tokens = sf.split_selfies(sf.encoder(seq))
        return [vocab_str2num.get(char, model_str2num['<UNK>']) for char in tokens]
    except EncoderError:
        try:
            seq = Chem.MolToSmiles(Chem.MolFromSmiles(seq))
            tokens = sf.split_selfies(sf.encoder(seq))
            return [vocab_str2num.get(char, model_str2num['<UNK>']) for char in tokens]
        except:
            return [model_str2num['<UNK>']]


def smiles_decode_molecule(encoded_smi, model_num2str, smiles_num2str, list_output=False):
    """
        Decodes SMILES-encoded molecule.
    """
    res = []
    for num in encoded_smi:
        if num in model_num2str:
            if list_output:
                res.append(model_num2str[num])
        elif num in smiles_num2str:
            res.append(smiles_num2str[num])
        else:
            raise Exception(f"Unexpected encoding value: {num}, not found in model_num2str or smiles_num2str")

    if not list_output:
        return ''.join(res)
    else:
        return res


def selfies_decode_molecule(encoded_sf, model_num2str, vocab_num2str, list_output=False):
    """
        Decodes SELFIES-encoded molecule.
    """
    res = []
    for num in encoded_sf:
        if num in model_num2str:
            if list_output:
                res.append(model_num2str[num])
        elif num in vocab_num2str:
            res.append(vocab_num2str[num])
        else:
            raise Exception(f"Unexpected encoding value: {num}, not found in model_num2str or smiles_num2str")

    if not list_output:
        return ''.join(res)
    else:
        return res


def combine_vocabs(*args):
    """
    Combine multiple vocabularies, but without any duplicates.

    Note: originally, I used a python set to remove duplicates, however
    since a set is unordered, the combined vocabularies would be in a
    different order every time they're combined. Now, when combining
    the vocabularies they will always be combined in the same order.
    """
    vocab = []
    for arg in args:
        for val in arg:
            if val not in vocab:
                vocab.append(val)
    return vocab


def convert_encoding_to_tokens(encoded_smi, model_num2str, smiles_num2str):
    res = []
    for num in encoded_smi:
        num = int(num)
        # print(num)
        # print(model_num2str)
        if num in model_num2str:
            res.append(model_num2str[num])
        elif num in smiles_num2str:
            res.append(smiles_num2str[num])
        else:
            raise Exception(f"Unknown encoding value: {num}")
    return res
