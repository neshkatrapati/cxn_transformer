import math
import torch
import torch.nn as nn
from ..model import PositionalEncoding, generate_square_subsequent_mask

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None):
        src = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        out = self.transformer.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc_out(out)

    def encode(self, src, src_mask=None, src_padding_mask=None):
        src = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src, mask=src_mask,
                                        src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None,
               memory_padding_mask=None):
        tgt = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
