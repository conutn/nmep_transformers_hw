from typing import Optional

import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, FeedForwardNN
from seq2seq.data.fr_en import tokenizer


class PositionalEncoding(nn.Module):
    """
    The PositionalEncoding layer will take in an input tensor
    of shape (B, T, C) and will output a tensor of the same
    shape, but with positional encodings added to the input.

    We provide you with the full implementation for this
    homework.

    Based on:
        https://web.archive.org/web/20230315052215/https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        """
        Initialize the PositionalEncoding layer.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape (B, T, C)
        """
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float,
    ):
        """
        Each encoder layer will take in an embedding of
        shape (B, T, C) and will output an encoded representation
        of the same shape.

        The encoder layer will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        self.WQ = nn.Linear(self.embedding_dim, self.num_heads * self.qk_length)
        self.WK = nn.Linear(self.embedding_dim, self.num_heads * self.qk_length)
        self.WV = nn.Linear(self.embedding_dim, self.num_heads * self.value_length)

        self.ln1 = nn.LayerNorm(self.embedding_dim)
        self.ln2 = nn.LayerNorm(self.embedding_dim)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.att = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.ffn = FeedForwardNN(self.embedding_dim, self.ffn_hidden_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The forward pass of the EncoderLayer.
        """
        y = self.ln1(x)
        Q = self.WQ(y)
        K = self.WK(y)
        V = self.WV(y)
        x = x + self.drop1(self.att(Q, K, V, mask))

        x = x + self.drop2(self.ffn(self.ln2(x)))

        return x;

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float,
    ):
        """
        Remember that the encoder will take in a sequence
        of tokens and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Encoder layers,
        each of which will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer. For this, we
        need to specify the number of layers and the number of
        heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers from some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer encoder.

        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos = PositionalEncoding(self.embedding_dim, max_len=max_length)
        self.layers = nn.ModuleList([
            EncoderLayer(self.num_heads, self.embedding_dim, self.ffn_hidden_dim, self.qk_length, self.value_length, dropout)
            for i in range(num_layers)])

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The forward pass of the Encoder.
        """
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
