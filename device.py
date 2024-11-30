import torch.cuda

device = ('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'