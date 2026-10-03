
import torch
import sentencepiece as spm

from transformer import TransformerModel
from tokenizer import PAD_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor(model_file="sql_sp.model")

checkpoint = torch.load(
    "best_model.pt",
    map_location=device
)

model = TransformerModel(
    vocab_size=sp.get_piece_size(),
    d_model=256,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=1024,
    dropout=0.1,
    pad_id=PAD_ID
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

print("Tokenizer vocabulary:", sp.get_piece_size())
print("Model loaded successfully")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint validation loss:", checkpoint.get("val_loss"))
print("Device:", device)