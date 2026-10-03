import streamlit as st
import torch
import sentencepiece as spm
import json
from transformer import TransformerModel
from tokenizer import BOS_ID, EOS_ID, PAD_ID
from sql_converter import convert_sql


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.set_page_config(
    page_title="Text-to-SQL Transformer",
    page_icon="🔎",
    layout="centered"
)

st.title("Text-to-SQL Transformer")
st.write("Convert natural language questions into SQL queries.")

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


def generate_sql(question, header, beam_size=4, max_len=64):

    columns = " ".join(
        f"<c{i}> {name}"
        for i, name in enumerate(header)
    )

    source = (
        question.strip()
        + " <sep> "
        + columns
    ).lower()

    src_ids = sp.encode(source) + [EOS_ID]

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE
    )

    beams = [
        ([BOS_ID], 0.0)
    ]

    with torch.no_grad():

        for _ in range(max_len - 1):

            candidates = []

            for tokens, score in beams:

                if tokens[-1] == EOS_ID:
                    candidates.append(
                        (tokens, score)
                    )
                    continue

                tgt = torch.tensor(
                    [tokens],
                    dtype=torch.long,
                    device=DEVICE
                )

                logits = model(
                    src,
                    tgt
                )

                log_probs = torch.log_softmax(
                    logits[:, -1, :],
                    dim=-1
                )

                values, indices = torch.topk(
                    log_probs,
                    beam_size,
                    dim=-1
                )

                for value, index in zip(
                    values[0],
                    indices[0]
                ):
                    candidates.append(
                        (
                            tokens + [index.item()],
                            score + value.item()
                        )
                    )

            candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            beams = candidates[:beam_size]

            if all(
                tokens[-1] == EOS_ID
                for tokens, _ in beams
            ):
                break

    generated = sp.decode(
        beams[0][0]
    )

    return generated


question = st.text_input(
    "Enter your question",
    "what is terrence ross' nationality"
)

header_text = st.text_area(
    "Enter table columns separated by commas",
    "Player, No., Nationality, Position, Years in Toronto, School/Club Team"
)

if st.button("Generate SQL"):

    header = [
        column.strip()
        for column in header_text.split(",")
    ]

    generated = generate_sql(
        question,
        header,
        beam_size=4
    )

    sql = convert_sql(
        generated,
        header
    )

    st.subheader("Generated SQL")

    st.code(
        sql if sql else generated,
        language="sql"
    )

    st.subheader("Model Output")

    st.code(
        generated,
        language="text"
    )

st.caption(
    f"Running on: {DEVICE}"
)
st.divider()

st.header("Training Results")

with open("results.json", encoding="utf-8") as f:
    results = json.load(f)

st.subheader("Training and Validation Loss")

loss_data = {
    "Epoch": list(range(1, results["epochs"] + 1)),
    "Training Loss": results["train_losses"],
    "Validation Loss": results["validation_losses"]
}

st.line_chart(
    loss_data,
    x="Epoch",
    y=["Training Loss", "Validation Loss"]
)

st.subheader("Final Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Execution Accuracy",
        f"{results['execution_accuracy'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Logical Form Accuracy",
        f"{results['logical_form_accuracy'] * 100:.2f}%"
    )

st.subheader("Model Configuration")

st.write({
    "Epochs": results["epochs"],
    "Batch Size": results["batch_size"],
    "Embedding Dimension": results["d_model"],
    "Attention Heads": results["num_heads"],
    "Encoder Layers": results["encoder_layers"],
    "Decoder Layers": results["decoder_layers"],
    "Feed Forward Dimension": results["feed_forward"],
    "Dropout": results["dropout"],
    "Label Smoothing": results["label_smoothing"],
    "Optimizer": results["optimizer"],
    "Beam Size": 4
})