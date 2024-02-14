import gym
import numpy as np
import imageio
import random
from stable_baselines3 import PPO

import dmc2gym
from dm_control import manipulation

env = dmc2gym.make(domain_name='quadruped', task_name='walk', seed=1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

frames = []
obs = env.reset()
img = env.render(mode="rgb_array", height=480, width=640, camera_id=2)

for _ in range(200):
    frames.append(env.render(mode="rgb_array", height=480, width=640, camera_id=2))
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

imageio.mimsave(f"dmc_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)
