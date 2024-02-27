
# nanoGPT-JAX-Flax-tf.data
nanGPT implemented in JAX!

nanoGPT is a minimalistic implementation of a GPT-like decoder-only transformer model, inspired by [Andrej Karpathy's](https://github.com/karpathy/nanoGPT) implementation. This version is built using the [JAX](https://github.com/google/jax) library and the [Flax](https://github.com/google/flax) neural network library.

## Key difference from Karpathy's nanoGPT:
- **JAX for Functional Elegance**: This repo highlights the beauty of JAX's functional programming paradigm, composable transformations, and the power of `jax.vmap` and `jax.pmap` to streamline transformer code. 
- **Simplified Attention Mechanism**: The core transformer attention logic is implemented for a single block of tokens, i.e., a single row from a batch. This approach simplifies the math and the code, eliminating the need to handle the batch dimension with complex manipulations.
- **Dataset Pipeline with TensorFlow**: The dataset pipeline is implemented using `tf.data`, providing efficient data loading and preprocessing.
- **Composable Transformations**: Leveraging JAX's composable transformations and the power of `jax.vmap`, the implementation achieves both simplicity and efficiency.

![nanoGPT](assets/nanogpt.jpg)


## Acknowledgements

Thank you to Karpathy for impleemnting the first version of nanoGPT!
