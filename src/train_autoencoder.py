import argparse
import pickle

desc = """
Trains an autoencoder on a given dataset and outputs some comparisons of the
raw dataset to the autoencoder's reconstructions of the dataset.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("train_dataset", type=str)
parser.add_argument("test_dataset", type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--latent-dim", type=int, default=1024)
parser.add_argument("--normalise-dataset", default=False, action="store_true")
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

# Import these after checking for argparse errors to save time.
from nets.autoencoder import AutoEncoder
from tensorflow.keras import losses
import tensorflow as tf

# Debug information to ensure we're actually using a GPU:
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU devices recognised by tensorflow: {gpus}")
if args.debug:
    tf.debugging.set_log_device_placement(True)

# Load and potentially normalise the datasets.
def preprocess(dataset_path):
    dataset = tf.data.experimental.load(dataset_path)
    dataset = dataset.shuffle(1024) # buffer size for shuffling
    if args.normalise_dataset:
        dataset = dataset.map(lambda x: x / 255.0) # rescale to [0, 1]
    
    batched_dataset = dataset.map(lambda x: (x, x)) # input is output
    batched_dataset = batched_dataset.batch(args.batch_size)
    return dataset, batched_dataset

train_ds, train_b_ds = preprocess(args.train_dataset)
test_ds, test_b_ds = preprocess(args.test_dataset)
dataset_shape = train_ds.element_spec.shape

# Compile the autoencoder model.
autoencoder = AutoEncoder(dataset_shape, [32, 16], args.latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.build([None] + dataset_shape)

print("Summary of the compiled model:")
print(autoencoder.summary(), "\n")

# Train the model.
print("Training the model:")
history = autoencoder.fit(
    train_b_ds,
    epochs=args.epochs,
    validation_data=test_b_ds
)

# Write artifacts to disk.
print("Writing artifacts to disk.")
autoencoder.save_weights(f"{args.output_dir}/autoencoder.tf")
with open(f"{args.output_dir}/history.pkl", "wb") as writer:
    writer.write(pickle.dumps(history.history))
