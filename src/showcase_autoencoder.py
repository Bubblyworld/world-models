import argparse
import pickle

desc = """
Plots comparisons of an autoencoder's reconstructed images versus the
originals, and some metadata like training/test loss per epoch.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("dataset", type=str)
parser.add_argument("model",type=str)
parser.add_argument("--latent-dim", type=int, default=1024)
parser.add_argument("--normalise-dataset", default=False, action="store_true")
args = parser.parse_args()

# Import these after checking for argparse errors to save time.
from nets.autoencoder import AutoEncoder
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = tf.data.experimental.load(args.dataset)
dataset = dataset.shuffle(1024) # buffer size for shuffling
if args.normalise_dataset:
    dataset = dataset.map(lambda x: x / 255.0) # rescale to [0, 1]
dataset_shape = dataset.element_spec.shape

# Compile the autoencoder model.
autoencoder = AutoEncoder(dataset_shape, [32, 16], args.latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.build([None] + dataset_shape)
autoencoder.load_weights(args.model)

# Compare original images to autoencoder's output.
sample_imgs = tf.stack(list(dataset.take(10)))
encoded_imgs = autoencoder.encoder(sample_imgs)
decoded_imgs = autoencoder.decoder(encoded_imgs)

n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(sample_imgs[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
