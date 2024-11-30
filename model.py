import lightning.pytorch as pl
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import nn, Tensor, optim
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder, Embedding
import torch.nn.functional as F


def NTXentLoss(features, temperature, n_views, device):
    labels = torch.tensor([i // n_views for i in range(features.shape[0])])  # updated this to label correctly
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def calc_accuracy(pred: Tensor, target: Tensor, threshold: float = 0.5):
    # for reconstruction
    if pred.dim() == 3 and target.dim() == 2:
        pred = pred.argmax(1)
        total = pred.size(0) * pred.size(1)
    else:
        # aux module classification
        pred = nn.Sigmoid()(pred)
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0
        total = len(pred)
    return 100 * (pred == target).float().sum() / total


def relabel(loss_dict, label):
    loss_dict = {label + str(key): val for key, val in loss_dict.items()}
    return loss_dict


class BaseRegressor(nn.Module):
    def __init__(self, latent_dim):
        super(BaseRegressor, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn_reg = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.nonlin = nn.ReLU()

    def forward(self, z):
        h = self.nonlin(self.bn_reg(self.fc1(z)))
        y_hat = self.fc2(h)
        return y_hat


class Block(pl.LightningModule):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.

          from:
          https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch
        """
        super(Block, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = x

        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class ReLSO(pl.LightningModule):
    def __init__(self, lr, max_seq_len, num_tasks, vocab_size, latent_dim, num_layers, d_model, nhead, recon_weight,
                 aux_weights, zrep_l2_weight,
                 dim_feedforward=1024, dropout=0.1, padding_vocab_idx=0):
        super().__init__()
        self.save_hyperparameters()
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True)
        self.padding_vocab_idx = padding_vocab_idx
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding1D(self.d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_dim = latent_dim
        self.recon_weight = recon_weight
        self.aux_weights = aux_weights
        self.zrep_l2_weight = zrep_l2_weight
        self.lr = lr

        self.glob_attn_module = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        self.bottleneck_module = nn.Sequential(
            nn.Linear(d_model, latent_dim)
        )

        z_to_hidden_layers = [
            nn.Linear(self.latent_dim, self.max_seq_len * (self.d_model // 2)),
            Block(self.d_model // 2, self.d_model),
            nn.Conv1d(self.d_model, self.vocab_size, kernel_size=3, padding=1),
        ]

        self.aux_modules = [BaseRegressor(self.latent_dim) for _ in range(num_tasks)]
        self.aux_modules = nn.ModuleList(self.aux_modules)

        self.dec_conv_module = nn.ModuleList(z_to_hidden_layers)

    def forward(self, batch):
        z_reps, _ = self.encode(batch)
        task_preds = [aux_module(z_reps) for aux_module in self.aux_modules]
        return self.decode(z_reps), task_preds, z_reps

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(self.device)

    def encode(self, batch):
        embed = self.embedding(batch)
        embed = embed + self.pos_encoder(embed)
        mask = self.generate_square_subsequent_mask(self.max_seq_len)
        enc = self.encoder(embed, mask=mask)

        glob_attn = self.glob_attn_module(enc)
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), enc).squeeze()

        if len(embed) == 1:
            z_rep = z_rep.unsqueeze(0)

        return self.bottleneck_module(z_rep), embed

    def decode(self, z_rep):
        h_rep = z_rep  # B x 1 X L
        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.d_model // 2, self.max_seq_len)
            h_rep = layer(h_rep)
        return h_rep

    def training_step(self, batch, batch_idx):
        x, y, task_targets = batch
        pred, task_preds, z_reps = self(x)
        train_loss, train_loss_logs = self.loss_function([pred, task_preds, z_reps], [y, task_targets])

        train_loss_logs = relabel(train_loss_logs, f"train_")
        self.log_dict(train_loss_logs, on_step=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y, task_targets = batch
        pred, task_preds, z_reps = self(x)
        _, valid_loss_logs = self.loss_function([pred, task_preds, z_reps], [y, task_targets], valid_step=True)

        valid_loss_logs = relabel(valid_loss_logs, f"valid_")
        self.log_dict(valid_loss_logs, on_epoch=True)

        return valid_loss_logs

    def test_step(self, batch, batch_idx):
        x, y, task_targets = batch
        pred, task_preds, z_reps = self(x)
        _, test_loss_logs = self.loss_function([pred, task_preds, z_reps], [y, task_targets], valid_step=True)

        test_loss_logs = relabel(test_loss_logs, f"test_")
        self.log_dict(test_loss_logs, on_epoch=True)

        return test_loss_logs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_function(self, predictions, targets, valid_step=False):
        # unpack everything
        # x is sequence, y is auxiliary
        x_hat, y_hat, z_reps = predictions
        x_true, y_true = targets

        # lower weight of padding token in loss
        ce_loss_weights = torch.ones(self.vocab_size, device=self.device)
        ce_loss_weights[self.padding_vocab_idx] = 0.8

        # reconstruction loss
        ae_loss = F.cross_entropy(x_hat, x_true, weight=ce_loss_weights)
        if not valid_step:
            ae_loss *= self.recon_weight

        with torch.no_grad():
            recon_accuracy = calc_accuracy(x_hat, x_true)

        # property prediction loss
        aux_losses = []
        for i in range(len(y_hat)):
            aux_loss = F.mse_loss(y_hat[i].flatten(), y_true[:, i])
            if not valid_step:
                aux_loss *= self.aux_weights[i]
            aux_losses.append(aux_loss)

        # RAE L_z loss
        zrep_l2_loss = 0.5 * torch.norm(z_reps, 2, dim=1) ** 2
        zrep_l2_loss = zrep_l2_loss.mean()

        if not valid_step:
            zrep_l2_loss *= self.zrep_l2_weight

        total_loss = ae_loss + sum(aux_losses) + zrep_l2_loss

        mloss_dict = {
            "recon_loss": ae_loss,
            "zrep_l2_loss": zrep_l2_loss,
            "loss": total_loss,
        }

        for i in range(len(aux_losses)):
            mloss_dict[f"aux_loss_{i}"] = aux_losses[i]

        mloss_dict['recon_accuracy'] = recon_accuracy

        return total_loss, mloss_dict


class ContrastiveReLSO(ReLSO):
    def __init__(self, lr, max_seq_len, num_tasks, vocab_size, latent_dim, num_layers, d_model, nhead, recon_weight,
                 aux_weights, zrep_l2_weight, contrast_weight, temperature, n_views,
                 dim_feedforward=1024, dropout=0.1, padding_vocab_idx=0):
        super().__init__(lr, max_seq_len, num_tasks, vocab_size, latent_dim, num_layers, d_model, nhead, recon_weight,
                         aux_weights, zrep_l2_weight, dim_feedforward, dropout, padding_vocab_idx)
        self.contrast_weight = contrast_weight
        self.temperature = temperature
        self.n_views = n_views

    def loss_function(self, predictions, targets, valid_step=False):
        # unpack everything
        # x is sequence, y is auxiliary
        x_hat, y_hat, z_reps = predictions
        x_true, y_true = targets

        # lower weight of padding token in loss
        ce_loss_weights = torch.ones(self.vocab_size, device=self.device)
        ce_loss_weights[self.padding_vocab_idx] = 0.8

        # reconstruction loss
        ae_loss = F.cross_entropy(x_hat, x_true, weight=ce_loss_weights)
        if not valid_step:
            ae_loss *= self.recon_weight

        with torch.no_grad():
            recon_accuracy = calc_accuracy(x_hat, x_true)

        # property prediction loss
        aux_losses = []
        for i in range(len(y_hat)):
            aux_loss = F.mse_loss(y_hat[i].flatten(), y_true[:, i])
            if not valid_step:
                aux_loss *= self.aux_weights[i]
            aux_losses.append(aux_loss)

        # contrastive loss
        logits, labels = NTXentLoss(z_reps, self.temperature, self.n_views, self.device)
        c_loss = F.cross_entropy(logits, labels)

        if not valid_step:
            c_loss *= self.contrast_weight

        # RAE L_z loss
        zrep_l2_loss = 0.5 * torch.norm(z_reps, 2, dim=1) ** 2
        zrep_l2_loss = zrep_l2_loss.mean()

        if not valid_step:
            zrep_l2_loss *= self.zrep_l2_weight

        total_loss = ae_loss + sum(aux_losses) + zrep_l2_loss + c_loss

        mloss_dict = {
            "recon_loss": ae_loss,
            "zrep_l2_loss": zrep_l2_loss,
            "contrast_loss": c_loss,
            "loss": total_loss,
        }

        for i in range(len(aux_losses)):
            mloss_dict[f"aux_loss_{i}"] = aux_losses[i]

        mloss_dict['recon_accuracy'] = recon_accuracy

        return total_loss, mloss_dict


class FragNet(pl.LightningModule):
    def __init__(self, lr, max_seq_len, num_tasks, vocab_size, num_layers, d_model, nhead, recon_weight,
                 aux_weights, zrep_l2_weight, contrast_weight, temperature, n_views,
                 dim_feedforward=1024, dropout=0.1, padding_vocab_idx=0):
        super().__init__()
        self.save_hyperparameters()
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True)
        decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True)
        self.padding_vocab_idx = padding_vocab_idx
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding1D(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.recon_weight = recon_weight
        self.aux_weights = aux_weights
        self.zrep_l2_weight = zrep_l2_weight
        self.lr = lr

        self.contrast_weight = contrast_weight
        self.temperature = temperature
        self.n_views = n_views

        self.aux_modules = [BaseRegressor(self.d_model) for _ in range(num_tasks)]
        self.aux_modules = nn.ModuleList(self.aux_modules)

        self.projection_head = nn.Sequential(
            nn.Linear(self.max_seq_len, self.max_seq_len // 2),
            nn.ReLU(),
            nn.Linear(self.max_seq_len // 2, self.max_seq_len // 4),
            nn.ReLU(),
            nn.Linear(self.max_seq_len // 4, self.max_seq_len // 8),
            nn.ReLU(),
            nn.Linear(self.max_seq_len // 8, 1),
        )

        self.unprojection_head = nn.Sequential(
            nn.Linear(1, self.max_seq_len // 8),
            nn.ReLU(),
            nn.Linear(self.max_seq_len // 8, self.max_seq_len // 4),
            nn.ReLU(),
            nn.Linear(self.max_seq_len // 4, self.max_seq_len // 2),
            nn.ReLU(),
            nn.Linear(self.max_seq_len // 2, self.max_seq_len),
        )

        self.decode_linear = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(self.device)

    def forward(self, batch):
        (enc_inputs, enc_pad_mask), (dec_inputs, dec_pad_mask) = batch
        z_reps, _ = self.encode(enc_inputs, src_key_padding_mask=enc_pad_mask)

        task_preds = [aux_module(z_reps) for aux_module in self.aux_modules]
        return self.decode(z_reps, dec_inputs, dec_pad_mask, enc_pad_mask), task_preds, z_reps

    def encode(self, enc_inputs, src_key_padding_mask):
        embedded_input = self.embedding(enc_inputs)
        embedded_input = embedded_input + self.pos_encoder(embedded_input)
        hidden = self.encoder(embedded_input,
                              src_key_padding_mask=src_key_padding_mask)  # B x S x E (hidden state after Transformer encoder)
        hidden = hidden.permute(0, -1, -2)  # B x E x S
        z_rep = self.projection_head(hidden).squeeze()  # B x E (latent space)

        if len(embedded_input) == 1:
            z_rep = z_rep.unsqueeze(0)

        return z_rep, hidden

    def decode(self, z_rep, dec_input, tgt_key_padding_mask, memory_key_padding_mask, use_softmax=False):
        embed_output = self.embedding(dec_input)
        embed_output = embed_output + self.pos_encoder(embed_output)
        tmp = z_rep.unsqueeze(-1)  # B x E x 1
        hidden = self.unprojection_head(tmp)  # B x E x S
        hidden = hidden.permute(0, -1, -2)  # B x S x E

        tgt_mask = self.generate_square_subsequent_mask(self.max_seq_len)  # S x S
        decode = self.decoder(embed_output, hidden,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)  # B x S x E
        decode = self.decode_linear(decode)  # B x S x Vocab

        if use_softmax:
            out = nn.Softmax(dim=-1)(decode)
        else:
            out = decode
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        (enc_inputs, enc_pad_mask), (dec_inputs, dec_pad_mask), dec_targets, task_targets = batch
        seq_preds, task_preds, z_reps = self([(enc_inputs, enc_pad_mask), (dec_inputs, dec_pad_mask)])
        train_loss, train_loss_logs = self.loss_function([seq_preds, task_preds, z_reps], [dec_targets, task_targets])

        train_loss_logs = relabel(train_loss_logs, f"train_")
        self.log_dict(train_loss_logs, on_step=True)

        return train_loss

    def loss_function(self, predictions, targets, valid_step=False):
        # unpack everything
        # x is sequence, y is auxiliary
        x_hat, y_hat, z_reps = predictions
        x_true, y_true = targets

        # reconstruction loss (ignores loss on padded output values)
        x_hat = x_hat.permute(0, -1, -2)
        ae_loss = F.cross_entropy(x_hat, x_true)
        #ae_pad_mask = (x_true != self.padding_vocab_idx)
        #ae_loss = ae_loss * ae_pad_mask
        #ae_loss = ae_loss.sum() / ae_pad_mask.sum()

        if not valid_step:
            ae_loss *= self.recon_weight

        with torch.no_grad():
            recon_accuracy = calc_accuracy(x_hat, x_true)

        # property prediction loss
        aux_losses = []
        for i in range(len(y_hat)):
            aux_loss = F.mse_loss(y_hat[i].flatten(), y_true[:, i])
            if not valid_step:
                aux_loss *= self.aux_weights[i]
            aux_losses.append(aux_loss)

        # contrastive loss
        logits, labels = NTXentLoss(z_reps, self.temperature, self.n_views, self.device)
        c_loss = F.cross_entropy(logits, labels)

        if not valid_step:
            c_loss *= self.contrast_weight

        # RAE L_z loss
        zrep_l2_loss = 0.5 * torch.norm(z_reps, 2, dim=1) ** 2
        zrep_l2_loss = zrep_l2_loss.mean()

        if not valid_step:
            zrep_l2_loss *= self.zrep_l2_weight

        total_loss = ae_loss + sum(aux_losses) + zrep_l2_loss + c_loss

        mloss_dict = {
            "recon_loss": ae_loss,
            "zrep_l2_loss": zrep_l2_loss,
            "contrast_loss": c_loss,
            "loss": total_loss,
        }

        for i in range(len(aux_losses)):
            mloss_dict[f"aux_loss_{i}"] = aux_losses[i]

        mloss_dict['recon_accuracy'] = recon_accuracy

        return total_loss, mloss_dict

    def test_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, dec_targets, task_targets = batch
        seq_preds, task_preds, z_reps = self([enc_inputs, dec_inputs])
        _, test_loss_logs = self.loss_function([seq_preds, task_preds, z_reps], [dec_targets, task_targets],
                                               valid_step=True)

        test_loss_logs = relabel(test_loss_logs, f"test_")
        self.log_dict(test_loss_logs, on_epoch=True)

        return test_loss_logs

    def validation_step(self, batch, batch_idx):
        enc_inputs, dec_inputs, dec_targets, task_targets = batch
        seq_preds, task_preds, z_reps = self([enc_inputs, dec_inputs])
        _, valid_loss_logs = self.loss_function([seq_preds, task_preds, z_reps], [dec_targets, task_targets],
                                                valid_step=True)

        valid_loss_logs = relabel(valid_loss_logs, f"valid_")
        self.log_dict(valid_loss_logs, on_epoch=True)

        return valid_loss_logs
