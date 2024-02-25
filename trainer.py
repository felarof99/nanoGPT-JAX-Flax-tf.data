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

from dataset import Dataset
from model import LanguageModel

class TrainState(train_state.TrainState):
  key: jax.random.KeyArray

from dataclasses import dataclass

@dataclass
class Config:
    BATCH_SIZE: int
    BLOCK_SIZE: int
    T: int 


class Main:
    def __init__(self):
        config = Config(BATCH_SIZE=8, BLOCK_SIZE=16, T=16)
        self.BATCH_SIZE = config.BATCH_SIZE
        self.BLOCK_SIZE = config.BLOCK_SIZE
        self.T = config.T 
        
        self.random_key = jax.random.PRNGKey(99)
        self.random_key, self.dropout_key = jax.random.split(self.random_key)

        self.model = LanguageModel(vocab_size=65, n_embed=32, T=self.BLOCK_SIZE)

        sample_block_of_tokens = jnp.ones(shape=(self.T), dtype=jnp.int32)
        output, params = self.model.init_with_output(jrand.PRNGKey(99), sample_block_of_tokens, training=False)
        self.params = params["params"]

        self.model_apply_batch = jax.vmap(self.model_apply, in_axes=(None, 0), out_axes=(0))

        opt = optax.adam(learning_rate=0.0001)
        self.state = TrainState.create(apply_fn=self.model_apply_batch, params=self.params, tx=opt, key=self.random_key)
        
        self.dataset = Dataset(batch_size=self.BATCH_SIZE, block_size=self.BLOCK_SIZE)

    def model_apply(self, params, inputs):
        return self.model.apply({"params": params}, inputs, False, rngs={'dropout': self.dropout_key})

    def forward_pass(self, params, state, batch):
        inputs, targets = batch
        logits = state.apply_fn(params, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss = loss.mean()
        return loss

    def train_step(self, state, batch):
        grad_fn = jax.value_and_grad(self.forward_pass, argnums=(0))
        loss, grads = grad_fn(state.params, state, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    def train_and_evaluate(self):
        for epoch in range(1):
            batch = self.dataset.get_batch()

            # random_key, random_subkey = jax.random.split(random_key)
            # dropout_key = jax.random.fold_in(key=random_key, data=state.step)

            state, loss = self.train_step(state, batch)
            print("loss", loss, "epoch", epoch) if epoch%100==0 else None