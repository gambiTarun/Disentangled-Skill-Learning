import gym
import numpy as np
import imageio

#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.getcwd()+'/relay-policy-learning/adept_envs')

import adept_envs
import gym

from gym.envs.registration import register

register(
    id='kitchen_relax-v1',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxV1',
    max_episode_steps=280,
)

env = gym.make('kitchen_relax-v1')

frames = []
obs = env.reset()
img = env.render()

for _ in range(200):
    # obs = env.render(mode='rgb_array', width=width, height=height)
    frames.append(env.render())
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    # img = model.env.render(mode="rgb_array")
#     # env.render()
imageio.mimsave(f"metaworld_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)