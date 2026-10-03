import math
import torch
import torch.nn as nn

from embeddings import PositionalEncoding


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    mask: bool tensor, True = attend, False = ignore (filled with -inf).
    Returns (output, attention_weights).
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        # (B, h, T, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask: True = keep, False = fill with -inf
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        # Guard against all-masked rows (all -inf → NaN after softmax)
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    h parallel attention heads, projected and concatenated.
    Returns (output, attention_weights averaged over heads).
    """

    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, L, d_model) -> (B, h, L, d_k)"""
        B, L, _ = x.size()
        return (
            x.view(B, L, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

    def _combine_heads(self, x):
        """(B, h, L, d_k) -> (B, L, d_model)"""
        B, _, L, _ = x.size()
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, L, self.d_model)
        )

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.q_linear(query))   # (B, h, T, d_k)
        k = self._split_heads(self.k_linear(key))     # (B, h, S, d_k)
        v = self._split_heads(self.v_linear(value))   # (B, h, S, d_k)

        output, attn = self.attention(q, k, v, mask)  # (B,h,T,d_k), (B,h,T,S)

        output = self._combine_heads(output)           # (B, T, d_model)
        output = self.out_linear(output)
        output = self.dropout(output)

        return output, attn


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """FFN(x) = max(0, xW1+b1)W2+b2"""

    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_padding_mask(ids, pad_id=0):
    """
    Returns (B, 1, 1, S) bool tensor.
    True  = real token (attend), False = padding (ignore).
    """
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length, device):
    """
    Returns (1, 1, L, L) lower-triangular bool tensor.
    True  = allowed to attend, False = future (ignore).
    """
    mask = torch.tril(
        torch.ones(length, length, device=device, dtype=torch.bool)
    )
    return mask.unsqueeze(0).unsqueeze(1)


def make_decoder_mask(tgt, pad_id=0):
    """Combined causal + padding mask for decoder self-attention."""
    pad_mask = make_padding_mask(tgt, pad_id)                         # (B,1,1,T)
    caus_mask = make_causal_mask(tgt.size(1), tgt.device)            # (1,1,T,T)
    return pad_mask & caus_mask                                        # (B,1,T,T)


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Self-attn → Add&Norm → FFN → Add&Norm  (post-norm)"""

    def __init__(self, d_model=256, num_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.drop1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.drop2(ff_out))
        return x


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Masked self-attn → Add&Norm → Cross-attn → Add&Norm → FFN → Add&Norm"""

    def __init__(self, d_model=256, num_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        # Masked self-attention
        self_out, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(self_out))

        # Cross-attention over encoder output
        cross_out, cross_attn = self.cross_attention(x, memory, memory, src_mask)
        x = self.norm2(x + self.drop2(cross_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.drop3(ff_out))

        return x, cross_attn   # return cross-attn weights for visualization


# ---------------------------------------------------------------------------
# Full Transformer Model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """
    Encoder-Decoder Transformer (Vaswani et al. 2017).
    Weight sharing: embedding = decoder embedding = output projection.
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=1024,
        dropout=0.1,
        max_len=512,
        pad_id=0,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.d_model = d_model

        # Shared embedding (enc + dec + output projection)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

        self.position = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection — weight SHARED with embedding (tied weights)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight   # weight tying

    # ------------------------------------------------------------------
    # Encoding (can be cached during beam search)
    # ------------------------------------------------------------------

    def encode(self, src):
        """Encode source tokens → memory (B, S, d_model), src_mask."""
        src_mask = make_padding_mask(src, self.pad_id)            # (B,1,1,S)
        x = self.position(self.embedding(src) * self.scale)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    # ------------------------------------------------------------------
    # Decoding one step or full sequence
    # ------------------------------------------------------------------

    def decode(self, tgt, memory, src_mask, return_cross_attn=False):
        """
        Decode target ids given pre-computed encoder memory.
        Returns logits (B, T, vocab_size).
        Optionally returns cross-attention from the last decoder layer.
        """
        tgt_mask = make_decoder_mask(tgt, self.pad_id)            # (B,1,T,T)
        x = self.position(self.embedding(tgt) * self.scale)

        last_cross_attn = None
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, memory, tgt_mask, src_mask)
            last_cross_attn = cross_attn                           # keep last

        logits = self.output(x)

        if return_cross_attn:
            return logits, last_cross_attn
        return logits

    # ------------------------------------------------------------------
    # Standard forward pass (used during training)
    # ------------------------------------------------------------------

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)