import gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from custEnvsCombined import HeirarchyEnvAnt

baseEnv = gym.make("Ant-v3")
env = HeirarchyEnvAnt(baseEnv,numModels=40)

# check_env(env)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

frames = []
obs = baseEnv.reset()
img = baseEnv.render(mode="rgb_array")

for _ in range(200): 
    # obs = env.render(mode='rgb_array', width=width, height=height)
    modelNo, _states = model.predict(obs)
    for _ in range(1):
        action, __states = env.model[modelNo].predict(obs)
        obs, rewards, dones, info = baseEnv.step(action)
        frames.append(baseEnv.render(mode="rgb_array"))
    # img = model.env.render(mode="rgb_array")
#     # env.render()
imageio.mimsave(f"ant_heirarchy_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)