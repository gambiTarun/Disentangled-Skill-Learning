import metaworld
import imageio
import random
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class MetaworldWrapper(gym.Wrapper):
    obs_size = 31

    def __init__(self, 
    env, 
    k_obs,
    ob_len=31,
    act_sanction=1,
    lambda_act=0.5
    ):
        super().__init__(env)
        self.env._k_obs = k_obs
        self.env._act = act_sanction
        self.env._lambda = lambda_act
        self.env.prev_act = np.zeros(4)
        obs_size = ob_len

    def step(self, action):
        prev_observation = self.get_obs()
        observation, reward, done, info = self.env.step(action)
        new_observation = self.get_obs()

        new_reward = np.linalg.norm(new_observation[self.env._k_obs==1] - prev_observation[self.env._k_obs==1], ord=2) - \
                    np.linalg.norm(new_observation[self.env._k_obs==0] - prev_observation[self.env._k_obs==0], ord=2) + \
                        self.env._lambda*(np.linalg.norm(action - self.env.prev_act, ord=2))*self.env._act

        envDone = done or getattr(self.env, 'curr_path_length', 0) > self.env.max_path_length
        self.env.prev_act = action.copy()
        return observation, reward, envDone, info

    def get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        observation = np.concatenate((position, velocity)).ravel()
        return observation

# Metaworld envs:
# ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 
# 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 
# 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 
# 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 
# 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 
# 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 
# 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 
# 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = MetaworldWrapper(ml1.train_classes['pick-place-v2'](),np.ones(31),ob_len=31,act_sanction=1)
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

# print(metaworld.ML1.ENV_NAMES)
# print(check_env(env))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# model.save("metaworld_test")
# del model
# model = PPO.load("metaworld_test")

frames = []
obs = env.reset()
img = env.render(mode="rgb_array")

for _ in range(200):
    frames.append(env.render(mode="rgb_array"))
    action, _state = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

imageio.mimsave(f"metaworld_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)