import torch
import sentencepiece as spm

from dataset import make_loader
from transformer import TransformerModel


sp = spm.SentencePieceProcessor(
    model_file="sql_sp.model"
)

train_dl = make_loader(
    "train_pairs.jsonl",
    sp,
    train=True
)

src, tgt = next(iter(train_dl))

model = TransformerModel(
    vocab_size=sp.get_piece_size()
)

logits = model(
    src,
    tgt[:, :-1]
)

print("src:", tuple(src.shape))
print("tgt:", tuple(tgt[:, :-1].shape))
print("logits:", tuple(logits.shape))
print("parameters:", sum(p.numel() for p in model.parameters()))