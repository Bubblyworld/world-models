import argparse

desc = """
Trains an autoencoder on a given dataset and outputs some comparisons of the
raw dataset to the autoencoder's reconstructions of the dataset.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("input_dataset", type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("--normalise-dataset", default=False, action="store_true")
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

# Import these after checking for argparse errors to save time.
from nets.autoencoder import AutoEncoder
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import tensorflow as tf

# Debug information for tensorflow:
if args.debug:
    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU devices recognised by tensorflow: {gpus}")

# Load and potentially normalise the dataset.
dataset = tf.data.experimental.load(args.input_dataset)
dataset = dataset.shuffle(1024) # buffer size for shuffling
if args.normalise_dataset:
    dataset = dataset.map(lambda x: x / 255.0) # rescale to [0, 1]

batched_dataset = dataset.map(lambda x: (x, x)) # input is output
batched_dataset = batched_dataset.batch(32)
dataset_shape = dataset.element_spec.shape

# Compile the autoencoder model.
autoencoder = AutoEncoder(dataset_shape, [32, 16], 1024)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.build([None] + dataset_shape)

print("Summary of the compiled model:")
print(autoencoder.summary(), "\n")

print("Starting the training process...")
history = autoencoder.fit(
    batched_dataset,
    epochs=1000,
    validation_data=batched_dataset # TODO: separate training set
)
