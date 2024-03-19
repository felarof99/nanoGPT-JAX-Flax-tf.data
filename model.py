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

class SingleHeadAttention(nn.Module):
  head_size: int
  T: int  # num_tokens
  dropout_rate: float = 0.2

  @nn.compact 
  def __call__(self, tokens: jnp.array, training: bool):
    """Tokens, each with some channel dim. Ex: [[0.1, 0.2], [0.3, 0.4]]"""
    # tokens shape: (T, channels)
    # attention output shape: (T, head_size)
    # Use separate single dense layers for calculating keys, query, values
    keys = nn.Dense(self.head_size, use_bias=False)(tokens)
    queries = nn.Dense(self.head_size, use_bias=False)(tokens)
    values = nn.Dense(self.head_size, use_bias=False)(tokens)

    # Use causal attention mask (only attend to tokens in the past).
    mask = jnp.tril(jnp.ones(shape=(self.T, self.T))) # jnp.ones(shape=(self.T, self.T))

    # compute attention score.
    wei = jnp.dot(queries, keys.T) / jnp.sqrt(self.head_size)
    wei = jnp.where(mask==0, -jnp.inf, wei)
    wei = nn.softmax(wei, axis=-1)

    attention_values = jnp.dot(wei, values) # (T, T) * (T, head_size))
    attention_values = nn.Dropout(rate=self.dropout_rate)(attention_values, deterministic=not training)
    return attention_values # (T, head_size)


class MultiHeadAttention(nn.Module):
  num_heads: int
  head_size: int # head_size * num_heads is the final embedding dimension you get, after concatenating from all heads
  T: int
  dropout_rate: float = 0.2

  def setup(self):
    self.heads = [
        SingleHeadAttention(head_size=self.head_size, T=self.T) for _ in range(self.num_heads)
    ]
    self.projection = nn.Dense(features=self.num_heads * self.head_size)
    self.dropout = nn.Dropout(rate=self.dropout_rate)

  def __call__(self, tokens: jnp.array, training: bool):
    output_from_each_head = []
    for h in self.heads:
      head_output = h(tokens, training)
      output_from_each_head.append(head_output)

    # Run multiple attention heads in parallel and concatenate
    # their output along channel dimension, i.e., dim==-1
    out_from_all_heads = jnp.concatenate(output_from_each_head, axis=-1)

    projection =  self.projection(out_from_all_heads)
    return self.dropout(projection, deterministic=not training)

class MultiHeadAttentionBatch(nn.Module):
  num_heads: int
  head_size: int # head_size * num_heads is the final embedding dimension you get, after concatenating from all heads
  T: int
  dropout_rate: float = 0.2

  def setup(self):
    self.multi_head_attn_single_example = MultiHeadAttention(
      num_heads=self.num_heads,
      head_size=self.head_size, 
      T=self.T, 
      dropout_rate=self.dropout_rate)

    self.multi_head_attn_batch = jax.vmap(self.multi_head_attn_single_example, 
                        in_axes=(0, None),  # tokens, training
                        out_axes=(0))

  def __call__(self, tokens: jnp.array, training: bool):
    return self.multi_head_attn_batch(tokens, training)

class FeedForward(nn.Module):
  output_size: int

  def setup(self):
    # Attention paper uses 4 times token_info_size when doing linear transformation
    # and then projects it back to token_info_size in linear transformation layer.
    self.ffwd = nn.Dense(features=4 * self.output_size)
    self.projection = nn.Dense(self.output_size)

  def __call__(self, x, training: bool):
    x = nn.relu(self.ffwd(x))
    x = self.projection(x)
    return x

class TransformerEncoderBlock(nn.Module):
  num_heads: int
  # output_size = head_size * num_heads, is the final embedding dimension you get after concatenating from all heads.
  output_size: int
  T: int # num_tokens

  def setup(self):
    # communication.
    self.head_size = self.output_size // self.num_heads  # each SingleAttentionHead will produce head_size worth of info for key, value, querie. You concatenate all of them to get the final output_size.
    self.self_attention_heads = MultiHeadAttention(num_heads=self.num_heads,
                                                   head_size = self.head_size,
                                                   T=self.T)

    # computation.
    self.computation_layer = FeedForward(output_size=self.output_size)

    self.ln1 = nn.LayerNorm()
    self.ln2 = nn.LayerNorm()

    self.dropout = nn.Dropout(rate=0.2)

  def __call__(self, x, training: bool):
    # transformer encoder forward pass
    x = x + self.self_attention_heads(self.ln1(x), training)

    x = x + self.computation_layer(self.ln2(x), training)

    x = self.dropout(x, deterministic=not training)
    return x

class LanguageModel(nn.Module):
  """Reads one char and predicits the next char."""
  vocab_size: int # number of vocabulary (number of rows of embedding table)
  n_embed: int # embedding dim after lookup
  T: int # block size, i.e., number of tokens attention block is looking at once
  num_heads: int
  num_layers: int

  def setup(self):
    # number of channels you want to use for store info for each token.
    self.C = self.vocab_size

    self.token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed)

    self.pos_embedding_table = nn.Embed(num_embeddings=self.T, features=self.n_embed)

    # Since, there are 4 heads, each head only needs to output token_info of size 8.
    # Concantenate token_info from all 4 heards, gives us 32
    self.blocks = [
        TransformerEncoderBlock(num_heads=self.num_heads,
                                output_size=self.n_embed,
                                T=self.T) for _ in range(self.num_layers)
    ]
    self.ln = nn.LayerNorm()
    self.lang_model_head = nn.Dense(features=self.C)

  def __call__(self, block_of_tokens: jnp.array, training: bool):
    """Accepts a block of tokens, like [0, 1, 2, 3, 4, 5, 6, 7]."""
    # generate emb for each token. output: (T, n_embed)
    token_embs = self.token_embedding_table(block_of_tokens)

    # generate position embs for each token.
    num_pos = block_of_tokens.shape[0]
    positions = jnp.arange(0, num_pos)
    pos_embs = self.pos_embedding_table(positions)

    # generate actual input to attention, x, which is sum of token_embs + pos_embs
    x = token_embs + pos_embs

    # feed x into self-attention head.
    ## language model, forward pass, block_of_tokens
    for i in range(self.num_layers):
      x = self.blocks[i](x, training)

    x = self.ln(x)

    # generate logits for each token. output: (T, channels for info -- C)
    token_logits = self.lang_model_head(x)

    return token_logits
