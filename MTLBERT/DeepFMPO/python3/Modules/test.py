import pickle

import pandas as pd
from rdkit import Chem
from mol_utils import split_molecule, get_fragments, join_fragments, get_join_list, should_use
from build_encoding import get_encodings, encode_list, encode_molecule
from global_parameters import MAX_FRAGMENTS
import seaborn as sns
import matplotlib.pyplot as plt

"""
df = pd.read_csv('C:\\Users\\nicka\\Desktop\\MTLBERT\\chembl_100maxseq.csv')
mols = []
fragments = []
for entry in df['smiles']:
    try:
        mols.append(Chem.MolFromSmiles(entry))
    except:
        continue

allmolgen_frags = get_fragments(mols)

with open('allmolgen_frags.pkl', 'wb') as f:
    pickle.dump(allmolgen_frags, f)
"""

"""
with open('allmolgen_frags.pkl', 'rb') as f:
    allmolgen_frags = pickle.load(f)

df = pd.read_csv('C:\\Users\\nicka\\Desktop\\MTLBERT\\chembl_100maxseq.csv')

frag_freq = {}
for key in allmolgen_frags[0].keys():
    frag_freq[key] = 0
frag_keys = list(frag_freq.keys())
for entry in df['smiles']:
    try:
        frags = [Chem.MolToSmiles(entry) for entry in split_molecule(Chem.MolFromSmiles(entry))]
        for frag in frags:
            if frag in frag_keys:
                frag_freq[frag] += 1
    except:
        continue
print(frag_freq)

with open('allmolgen_frag_freq.pkl', 'wb') as f:
    pickle.dump(frag_freq, f)
"""

with open('allmolgen_frag_freq.pkl', 'rb') as f:
    frag_freq = pickle.load(f)

res = {key: val for key, val in sorted(frag_freq.items(), key=lambda ele: ele[1])}

df = pd.DataFrame()
df['fragments'] = list(res.keys())
df['frequency'] = list(res.values())
df['lengths'] = [len(entry) for entry in list(res.keys())]

print(len(list(res.values())))
print(len(df[df['frequency'] >= 1000]) / len(df['frequency']))
#sns.displot(data=df[df['frequency'] > 50000], x='frequency', kind='kde')
sns.displot(data=df, x='frequency', kind="ecdf", stat='proportion', complementary=True, log_scale=True)
plt.xlim(left=1)
plt.axvline(x=1000)
plt.text(1000.1, 0, '0.7%', rotation=90)
plt.xlabel("Fragment Frequency")
plt.show()

exit(1)
fragments = df[df['frequency'] >= 1000]['fragments'].tolist()

smiles_regex_pattern = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

smiles_str2num = {}
with open('smiles_vocab.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for i, j in enumerate(lines):
        smiles_str2num[j] = i

len_vocab = len(smiles_str2num)

for i in range(len(fragments)):
    smiles_str2num[fragments[i]] = len_vocab + i

print(len(fragments))
print(smiles_str2num)
print(len(smiles_str2num))
exit(1)

smiles_num2str = {i: j for j, i in smiles_str2num.items()}

def hybrid_encode_molecule(mol):
    fragments = split_molecule(mol)

    encoded = []
    for i in range(len(fragments)):
        if fragments[i] in smiles_str2num:
            encoded.append(smiles_str2num[fragments[i]])



"""
encodings, decodings = get_encodings(fragments)

print(encodings)
for k, v in decodings.items():
    print(k, Chem.MolToSmiles(v))


def one_hot_to_int(one_hot):
    return int(one_hot, 2)



def int_to_one_hot(integer):
    return bin(integer)


def list_one_hot_to_int(one_hot_list):
    return [one_hot_to_int(entry) for entry in one_hot_list]


def list_int_to_one_hot(int_list):
    one_hot_list = [int_to_one_hot(entry)[2:] for entry in int_list]
    max_len_str = max(map(len, one_hot_list))
    one_hot_list = ['0' * abs((len(entry) - max_len_str)) + entry for entry in one_hot_list]
    return one_hot_list


tmp = list(decodings.keys())
tmp2 = list_int_to_one_hot(list_one_hot_to_int(list(decodings.keys())))
"""
