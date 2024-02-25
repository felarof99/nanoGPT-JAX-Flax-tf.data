from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import jax
import jax.numpy as jnp
import tensorflow as tf

# Below would result in a minibatch size of 32.
BATCH_SIZE = 8 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 16 # what is the maximum context length for predictions?

def println(*args):
  for arg in args:
    print(arg)

@dataclass
class Dataset:
    batch_size: int = BATCH_SIZE
    block_size: int = BLOCK_SIZE

    def __post_init__(self):
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # Create chars vocubulary using all the unique characters in the text.
        chars = sorted(list(set(text)))
        self.VOCAB_SIZE = len(chars)

        # Create mapping from characters to integers.
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Create reverse mapping from integers to characters.
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Create encode, decode function.
        def _encode(s: str, stoi: Mapping[str, int]) -> List[int]:
            return [stoi[c] for c in s]

        def _decode(tokens: List[int], itos: Mapping[int, str]) -> str:
            return ''.join([itos[i] for i in tokens])

        # Let's now split up the data into train and validation sets.
        data = jnp.array(_encode(text, self.stoi), dtype=jnp.int64)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.train_dataset = self._create_dataset(self.train_data)
        self.val_dataset = self._create_dataset(self.val_data)

    def _create_dataset(self, data):
        dataset = (tf.data.Dataset.from_tensor_slices(data)
                   .batch(self.block_size + 1)
                   .map(lambda input: (input[:self.block_size], input[1:self.block_size + 1]),
                        num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(self.batch_size)
                   .repeat()
                   .as_numpy_iterator())
        return dataset

    def create_train_dataset(self):
        train_dataset = self._create_dataset(self.train_data)
        return train_dataset

    def create_val_dataset(self):
        val_dataset = self._create_dataset(self.val_data)
        return val_dataset

    def get_batch(self, training: bool = True):
        if not training:
            val_batch = next(self.val_dataset)
            return jnp.array(val_batch)

        train_batch = next(self.train_dataset)
        return jnp.array(train_batch)
