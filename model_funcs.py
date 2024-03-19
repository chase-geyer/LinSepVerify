import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import tensorflow as tf                # TensorFlow for datasets
import tensorflow_datasets as tfds     # TensorFlow Datasets

import pickle as pkl                   # For saving data
import matplotlib.pyplot as plt        # For plotting


## DATA LOADING
def get_dataset(dataset_name):
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder(dataset_name)
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 127.5 - 1. # Normalize to [-1, 1]
  test_ds['image'] = jnp.float32(test_ds['image']) / 127.5 - 1. # Normalize to [-1, 1]
  return train_ds, test_ds

def get_labels(train_ds, test_ds):
    """Extract labels from train and test datasets."""
    train_labels = train_ds['label'] 
    test_labels = test_ds['label']
    return train_labels, test_labels

def process_mnist_data():
    """Process MNIST data into format for training."""
    train_images, test_images = get_dataset('mnist')
    train_labels, test_labels = get_labels(train_images, test_images)
    imgs = [img.reshape(28, 28).tolist() for img in test_images['image'][0:150]]
    print(test_labels)
    labels = test_labels[0:150]
    pkl.dump(imgs, open("imgs/MNIST:images-for-verification", "wb" ))
    pkl.dump(labels, open("imgs/MNIST:labels-for-verification", "wb" ))
    return train_images, test_images, train_labels, test_labels



## ACTIVATION FUNCTIONS
# Defining the dorefa activation function using a similar mathematical framework as defined in the larq docs

def dorefa_activation(self, x, k_bits = 2):
    n = 2**k_bits - 1
    
    # First, apply the sigmoid activation function to squash the values between 0 and 1
    x = nn.sigmoid(x)

    # Then, quantize the values into a number of discrete intervals
    x = jnp.round(x * n) / n

    return x

def binary_step( x):
    return jnp.where(x >= 0, 1, 0)

## LOSS FUNC FOR MODEL (L1 LOSS)
def l1_loss_func(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return jnp.mean(jnp.abs(logits - labels_onehot))

## MODEL ACCURACY
def compute_metrics(*, logits, labels):
  loss = l1_loss_func(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def train_epoch(model, state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(model, state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state

