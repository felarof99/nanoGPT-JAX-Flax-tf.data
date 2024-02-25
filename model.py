from typing import List, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand
import flax.linen as nn
from flax.training import train_state  # Useful dataclass to keep train state
import optax
import tensorflow as tf
import pdb
import functools

class FeedForward(nn.Module):
  output_size: int

  def setup(self):
    # attention paper uses 4 times token_info_size when doing linear transformation.
    # and then projects it back to token_info_size in linear transformation layer.
    self.ffwd = nn.Dense(features=4 * self.output_size)

    # projection layer, which goes back into residual pathway.
    self.projection = nn.Dense(self.output_size)

  def __call__(self, x, training: bool):
    x = nn.relu(self.ffwd(x))
    x = self.projection(x)
    return x


class Head(nn.Module):
  token_info_size: int # head_size; how much (emb dim) info each token emits for keys, queries, values.
  T: int # block size; number of tokens in a block

  def setup(self):
    # key, query will take vector of size C.
    # i.e., channels containing info of token and will output token_info_size
    self.key_layer = nn.Dense(self.token_info_size, use_bias=False)
    self.query_layer = nn.Dense(self.token_info_size, use_bias=False)
    self.value_layer = nn.Dense(self.token_info_size, use_bias=False)

    self.dropout = nn.Dropout(rate=0.2)


  def __call__(self, block_of_tokens_with_info_channels: jnp.array, training: bool):
    """Accepts a block of tokens with info channels, like (8, 65)."""

    # TODO(ntnsonti): Double check; but tril should not be learnable according cGPT.
    tril = jnp.tril(jnp.ones(shape=(self.T, self.T)))

    # input: (T, info channels)
    # output: (T, token_info_size)
    keys = self.key_layer(block_of_tokens_with_info_channels)
    queries = self.query_layer(block_of_tokens_with_info_channels)
    values = self.value_layer(block_of_tokens_with_info_channels)

    # chanel info size
    C = int(block_of_tokens_with_info_channels.shape[-1])
    # print("[ntn99] channel_info_size: ", C)

    # compute attention score.
    wei = jnp.dot(queries, keys.T) * C**0.5 # (T, token_info_size) * (token_info_size, T) == (T, T)
    wei = jnp.where(tril==0, -jnp.inf, wei)
    wei = nn.softmax(wei, axis=-1)

    attention_values = jnp.dot(wei, values) # (T, T) * (T, token_info_size))

    attention_values = self.dropout(attention_values, deterministic=not training)

    return attention_values # (T, token_info_size)


class MultiHeadAttention(nn.Module):
  num_heads: int
  final_token_info_size: int # After concatenating from all heads, how much info (values -- emb size) you have on each token.
  T: int

  def setup(self):
    self.token_info_size_per_head = int(self.final_token_info_size/self.num_heads)
    self.heads = [
        Head(token_info_size=self.token_info_size_per_head, T=self.T) for _ in range(self.num_heads)
    ]

    self.projection = nn.Dense(features=self.final_token_info_size)

    self.dropout = nn.Dropout(rate=0.2)

  def __call__(self, block_of_tokens_with_info_channels: jnp.array, training: bool):
    out_from_each_head = jnp.array([h(block_of_tokens_with_info_channels, training) for h in self.heads])

    # You just run multiple attention heads in parallel and concatenate
    # their output along channel dimension, i.e., dim==-1
    out_from_all_heads = jnp.concatenate(out_from_each_head, axis=-1)
    # print("[ntn99] out_from_all_heads concatenated shape: ", out_from_all_heads.shape)

    projection =  self.projection(out_from_all_heads)

    return self.dropout(projection, deterministic=not training)


class Block(nn.Module):
  num_heads: int
  final_token_info_size: int
  T: int

  def setup(self):
    # communication.
    self.self_attention_heads = MultiHeadAttention(num_heads=self.num_heads,
                                                   final_token_info_size=self.final_token_info_size,
                                                   T=self.T)

    # computation.
    self.computation_layer = FeedForward(output_size=self.final_token_info_size)

    self.ln1 = nn.LayerNorm()
    self.ln2 = nn.LayerNorm()

    self.dropout = nn.Dropout(rate=0.2)

  def __call__(self, x, training: bool):
    x = x + self.self_attention_heads(self.ln1(x), training)

    x = x + self.computation_layer(self.ln2(x), training)

    x = self.dropout(x, deterministic=not training)
    return x


class LanguageModel(nn.Module):
  """Reads one char and predicits the next char."""
  vocab_size: int # number of vocabulary (number of rows of embedding table)
  n_embed: int # embedding dim after lookup
  T: int # block size, i.e., number of tokens attention block is looking at once

  def setup(self):
    # number of channels you want to use for store info for each token.
    self.C = self.vocab_size

    self.token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed)

    self.pos_embedding_table = nn.Embed(num_embeddings=self.T, features=self.n_embed)

    # Since, there are 4 heads, each head only needs to output token_info of size 8.
    # Concantenate token_info from all 4 heards, gives us 32
    self.num_blocks = 4
    self.blocks = [
        Block(num_heads=4, final_token_info_size=self.n_embed, T=self.T) for _ in range(self.num_blocks)
    ]
    self.ln = nn.LayerNorm()
    self.lang_model_head = nn.Dense(features=self.C)

  def __call__(self, block_of_tokens: jnp.array, training: bool):
    """Accepts a block of tokens, like [0, 1, 2, 3, 4, 5, 6, 7]."""

    # generate em for each token. output: (T, n_embed)
    token_embs = self.token_embedding_table(block_of_tokens)

    # generate position embs for each token.
    num_pos = self.T
    positions = jnp.arange(0, num_pos)
    pos_embs = self.pos_embedding_table(positions)

    # generate actual input to attention, x, which is sum of token_embs + pos_embs
    x = token_embs + pos_embs

    # feed x into self-attention head.
    for i in range(self.num_blocks):
      x = self.blocks[i](x, training)

    x = self.ln(x)

    # generate logits for each token. output: (T, channels for info -- C)
    token_logits = self.lang_model_head(x)

    return token_logits
