import matplotlib.pyplot as plt
import torch

from embeddings import PositionalEncoding


d_model = 256
max_len = 100

pe = PositionalEncoding(
    d_model=d_model,
    max_len=max_len,
    dropout=0.0
)

data = pe.pe[0].cpu().numpy()

plt.figure(figsize=(12, 6))
plt.imshow(data, aspect="auto")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.title("Positional Encoding Heatmap")
plt.colorbar()
plt.tight_layout()
plt.show()