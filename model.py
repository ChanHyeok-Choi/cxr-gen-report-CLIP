import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Transformer

logger = logging.getLogger(__name__)


class ImageEncoderReportDecoderConfig:

    def __init__(self, vocab_size, block_size, n_embd):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.embd_pdrop = 0.1
        self.n_lbl_class = 4
        self.n_lbls = 14
        self.tgt_drop_p = 0.3

class ImageEncoderReportDecoder(nn.Module):

    def __init__(self, config, img_enc, img_enc_out_shape, rgb = True):
        super().__init__()

        self.cnf = config
        self.rgb = rgb
        
        self.label_emb = nn.Embedding(config.n_lbl_class, config.n_embd)
        self.pos_emb_labels = nn.Parameter(torch.zeros(1, config.n_lbls, config.n_embd))

        self.img_enc = img_enc
        self.img_enc_linear = nn.Linear(img_enc_out_shape[1], config.n_embd, bias=False)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.tgt_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.tgt_drop = nn.Dropout(config.tgt_drop_p)

        self.normalize = nn.LayerNorm((config.block_size, config.n_embd))
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=8, dim_feedforward=512, \
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) #2

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.n_embd, nhead=8, dim_feedforward=512, \
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)


        self.dcd_proj = nn.Linear(config.n_embd, config.vocab_size)
        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.register_buffer("mask", torch.triu(torch.ones(config.block_size, config.block_size)*float('-inf'),diagonal=1))
        self.mask = self.mask < 0
        #print(self.mask)


        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets, len_mask, labels):

        reps = self.representation(x, labels)
        logit, loss, prediction = self.decode(reps, targets, len_mask)

        return logit, loss, prediction

    def configure_optimizers(self, train_config):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def get_block_size(self):
        return self.block_size

    def representation(self, x, labels):

        if labels != None:
            label_emb = self.label_emb(labels)
            label_emb = label_emb + self.pos_emb_labels


        if self.rgb:
            x = torch.cat((x.unsqueeze(1),x.unsqueeze(1),x.unsqueeze(1)), dim=1)

        with torch.no_grad():
            x = self.img_enc(x)

        if len(x.shape) < 3:
            x = x.unsqueeze(2)
        elif len(x.shape) > 3:
            x = x.squeeze(1)

        x = self.img_enc_linear(x)
        b, t, e = x.size()
        x = x + self.pos_emb[:, :t,:]
        x = self.drop(x)
        
        x = torch.cat((x, label_emb), dim=1)

        reps = self.transformer_encoder(x)
        return reps

    def decode(self, reps, targets, len_mask):

        tgt_emb = self.tgt_emb(targets)
        bt, tt, et = tgt_emb.size()
        tgt_emb = tgt_emb + self.pos_emb[:, :tt, :]
        tgt_emb = self.tgt_drop(tgt_emb)

        if len_mask != None:
            len_mask = len_mask > 0
            logits_mask = torch.where(len_mask == True, 0, 1).unsqueeze(2).repeat(1,1,2321)

            logits = self.transformer_decoder(tgt_emb[:,:-1,:], reps, tgt_mask=self.mask[:-1,:-1], tgt_key_padding_mask=len_mask[:,:-1])
        else:
            logits = self.transformer_decoder(tgt_emb, reps)

        logits = self.dcd_proj(logits)

        loss = None
        if len_mask is not None:
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets[:,1:].reshape(-1))
            #print("loss shape:",loss.shape)

            #if len_mask is not None:
            #    numzeros = torch.sum((len_mask == 0))
            #    loss = loss.masked_fill(len_mask.view(-1) == 1, 0).sum()
            #    loss = loss/numzeros

        prediction = torch.argmax(logits,dim=-1)
        return logits, loss, prediction

