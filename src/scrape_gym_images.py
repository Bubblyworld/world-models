import argparse

desc = """
Scrapes images from a gym environment played at random and saves them as a
tensorflow dataset. Can be used to pretrain an autoencoder world model.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("output_path", type=str)
parser.add_argument("--gym-env", type=str, default="MsPacman-v4")
parser.add_argument("--rollouts", type=int, default=1024)
parser.add_argument("--probability", type=float, default=0.01)
parser.add_argument("--test-split", type=float, default=0.1)
args = parser.parse_args()

# Import these after checking for argparse errors to save time.
import tensorflow as tf
import numpy as np
import random
import gym

env = gym.make(args.gym_env)

observations = []
for i in range(args.rollouts):
    if i > 0 and i % 128 == 0:
        print(f"...running rollout {i}")

    env.reset()
    while True:
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        if random.random() < args.probability:
            observations.append(obs)
        if done:
            break

random.shuffle(observations)
num_test = int(len(observations) * args.test_split)
test_data = np.array(observations[:num_test], dtype=np.float32)
train_data = np.array(observations[num_test:], dtype=np.float32)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)

print(f"Observation shape: {observations[0].shape}")
print(f"Total number of observations: {len(observations)}")
print(f"... saving training dataset ({len(train_data)}) to: {args.output_path}_train")
tf.data.experimental.save(train_dataset, f"{args.output_path}_train")
print(f"... saving test dataset ({len(test_data)}) to: {args.output_path}_test")
tf.data.experimental.save(test_dataset, f"{args.output_path}_test")
