from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from dataclasses import dataclass

from dataset import Dataset
from model import LanguageModel

class TrainState(train_state.TrainState):
    key: jax.random.KeyArray

@dataclass
class Config:
    BATCH_SIZE: int = 8
    BLOCK_SIZE: int = 16
    T: int = 16

config = Config()


random_key = jax.random.PRNGKey(99)
random_key, dropout_key = jax.random.split(random_key)

# Initialize model
model = LanguageModel(vocab_size=65, n_embed=32, T=config.BLOCK_SIZE)
sample_block_of_tokens = jnp.ones(shape=(config.T,), dtype=jnp.int32)
output, params = model.init_with_output(jax.random.PRNGKey(99), sample_block_of_tokens, training=False)
params = params["params"]

def model_apply(params, inputs):
    return model.apply({"params": params}, inputs, False, rngs={'dropout': dropout_key})

# Vectorize model apply function
model_apply_batch = jax.vmap(model_apply, in_axes=(None, 0), out_axes=(0))

# Define forward pass
def forward_pass(params, state, batch: Tuple[jnp.ndarray, jnp.ndarray]):
    inputs, targets = batch
    logits = state.apply_fn(params, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = loss.mean()
    return loss

# Define training step
def train_step(state, batch):
    grad_fn = jax.value_and_grad(forward_pass, argnums=(0))
    loss, grads = grad_fn(state.params, state, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Initialize optimizer and training state
opt = optax.adam(learning_rate=0.0001)
state = TrainState.create(apply_fn=model_apply_batch, params=params, tx=opt, key=random_key)
data = Dataset(batch_size=config.BATCH_SIZE, block_size=config.BLOCK_SIZE)

# Function to run a training step
def run_train_step():
    batch = data.get_batch()
    state, loss = train_step(state, batch)
    print(f"Loss: {loss}")

def run_train_and_evaluate():
    for epoch in range(1):
        loss = run_train_step()
        print("loss", loss, "epoch", epoch) if epoch % 100 == 0 else None
        
if __name__ == "__main__":
    run_train_step()