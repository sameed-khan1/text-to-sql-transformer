import sentencepiece as spm

sp = spm.SentencePieceProcessor(
    model_file="sql_sp.model"
)

source = (
    "what school did patrick o'bryant play for "
    "<sep> columns "
    "<c0> player "
    "<c1> no. "
    "<c2> nationality "
    "<c3> position "
    "<c4> years in toronto "
    "<c5> school/club team"
).lower()

tokens = sp.encode(source, out_type=str)

for i, token in enumerate(tokens):
    print(i, repr(token))