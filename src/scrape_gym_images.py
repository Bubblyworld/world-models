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

print(f"Observation shape: {observations[0].shape}")
print(f"Total number of observations: {len(observations)}")
print(f"... saving dataset to: {args.output_path}")

observations = np.array(observations, dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices(observations)
tf.data.experimental.save(dataset, args.output_path)
