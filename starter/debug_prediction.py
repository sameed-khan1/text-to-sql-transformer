import torch
import sentencepiece as spm

from transformer import TransformerModel
from tokenizer import BOS_ID, EOS_ID, PAD_ID


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

sp = spm.SentencePieceProcessor(
    model_file="sql_sp.model"
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
).to(DEVICE)

checkpoint = torch.load(
    "best_model.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

question = "what school or club team did terrence ross play for"

source = (
    question
    + " <sep> "
    + "<c0> player "
    + "<c1> no. "
    + "<c2> nationality "
    + "<c3> position "
    + "<c4> years in toronto "
    + "<c5> school/club team"
).lower()

src_ids = sp.encode(source) + [EOS_ID]

src = torch.tensor(
    [src_ids],
    dtype=torch.long,
    device=DEVICE
)

select_id = sp.piece_to_id("▁select")
space_id = sp.piece_to_id("▁")

target = torch.tensor(
    [[BOS_ID, select_id, space_id]],
    dtype=torch.long,
    device=DEVICE
)

with torch.no_grad():
    logits = model(src, target)
    probabilities = torch.softmax(
        logits[:, -1, :],
        dim=-1
    )

    values, indices = torch.topk(
        probabilities,
        20,
        dim=-1
    )

print("Top predictions after select:")

for value, index in zip(values[0], indices[0]):
    token = sp.id_to_piece(index.item())
    print(
        f"{token:25} {value.item():.6f}"
    )