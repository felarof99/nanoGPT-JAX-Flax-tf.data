from typing import List, Dict, Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand
import flax
import flax.linen as nn
from flax.training import train_state  # Useful dataclass to keep train state
import optax
import tensorflow as tf
import pdb
import functools

def println(*args):
  for arg in args:
    print(arg)

import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
jax.devices()

import dataset as data
import model

class TrainState(train_state.TrainState):
    key: jax.random.KeyArray


@dataclasses.dataclass
class Config:
    vocab_size: int = 66
    batch_size: int = 512
    block_size: int = 64
    n_embed: int = 256
    num_heads: int = 8
    num_layers: int = 6

config = Config()
DEVICE_COUNT = 8

def forward_pass(params, state, batch, training, rng):
  inputs, targets = batch
  logits = state.apply_fn({"params": params}, inputs, training, rngs={"dropout": rng})

  # logits: T, C
  # targets: T
  # logits predict each position
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
  loss = loss.mean()
  return loss
       
def backward_pass(state, batch, training, rng):
  grad_fn = jax.value_and_grad(forward_pass, argnums=(0))
  loss, grads = grad_fn(state.params, state, batch, training, rng)

  state = state.apply_gradients(grads=grads)
  return state, loss
 
def backward_pass_pmap(state, batch, training, rng):
  grad_fn = jax.value_and_grad(forward_pass, argnums=(0))
  loss, grads = grad_fn(state.params, state, batch, training, rng)

  loss = jax.lax.pmean(loss, axis_name="devices")
  grads = jax.lax.pmean(grads, axis_name="devices")

  state = state.apply_gradients(grads=grads)
  return state, loss

def train_step(state, batch, training, rng):
  state, loss = backward_pass(state, batch, training, rng)
  return state, loss

train_step_pmap = jax.pmap(
    jax.jit(train_step), in_axes=(0, 0, None, 0), out_axes=(0), axis_name="devices")

def get_batch_pmap(dataset):
  inputs, targets = data.get_batch(dataset)
  inputs = inputs.reshape((jax.device_count(), -1, inputs.shape[-1]))
  targets = targets.reshape((jax.device_count(), -1, targets.shape[-1]))
  return inputs, targets

model = model.LanguageModelBatch(vocab_size=config.vocab_size,
                      n_embed=config.n_embed,
                      num_tokens=config.block_size,
                      num_heads=config.num_heads,
                      num_layers=config.num_layers)

train_dataset = data.create_train_dataset()
inputs, targets = data.get_batch(train_dataset)
inputsp, targetsp = get_batch_pmap(train_dataset)

output, params = model.init_with_output(jax.random.PRNGKey(99), inputs, training=False)
params = params["params"]

opt = optax.adam(learning_rate=0.0001)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

states = jax.device_put_replicated(state, jax.local_devices())

states, loss = train_step_pmap(states, get_batch_pmap(train_dataset), False, jax.random.split(jax.random.PRNGKey(9),
                                                                                              num=DEVICE_COUNT))
rngs = jax.random.split(jax.random.PRNGKey(9), num=DEVICE_COUNT)
for step in range(5000):
  rngs = jax.random.split(rngs[0], num=DEVICE_COUNT)
  train_batch = get_batch_pmap(train_dataset)
  states, loss = train_step_pmap(states, train_batch, False, rngs)

  print("loss", loss[0], "step", step) if step%100==0 else None

# if __name__ == "__main__":
#     run_train_step()