import gym
import numpy as np
import imageio
import mujoco_py
	
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.env_checker import check_env

class AntWrapper(gym.Wrapper):
    def __init__(self, 
    env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

env = AntWrapper(gym.make("Ant-v3"))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# print(check_env(env))

model.save("mujoco_test")
del model
model = PPO.load("mujoco_test")

# viewer = mujoco_py.mjviewer.MjViewer(env)
# assert viewer is not None
# for key, value in DEFAULT_CAMERA_CONFIG.items():
#     if isinstance(value, np.ndarray):
#         getattr(viewer.cam, key)[:] = value
#     else:
#         setattr(viewer.cam, key, value)

frames = []
obs = env.reset()
img = env.render(mode="rgb_array")

for _ in range(200):
    frames.append(env.render(mode="rgb_array"))
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

imageio.mimsave(f"mujoco_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)