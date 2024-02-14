from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0Custom
import torch as th
from s3dg import S3D
class KitchenEnvSlidingReward(Env):
    def __init__(self):
        super(KitchenEnvSlidingReward,self)
        self.env = KitchenMicrowaveHingeSlideHingeV0()
        self.observation_space = self.env.observation_space
        self.action = self.env.action_space
        self.past_observations = []
        self.window_length = 32
        self.net = S3D('s3d_dict.npy', 512)
        # Load the model weights
        self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        text_output = self.net.text_module(["robot opening door"])
        self.target_embedding = text_output['text_embedding']

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_kitchen(self, frames):
        frames = np.array(frames)
        # frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        self.past_observations = self.past_observations[-self.window_length:]
        frames = self.preprocess_kitchen(self.past_observations)

        video = th.from_numpy(frames)
        # print(frames.shape)
        video_output = self.net(video.float())

        video_embedding = video_output['video_embedding']
        similarity_matrix = th.matmul(target_embedding, video_embedding.t())

        reward = similarity_matrix.detach().numpy()[0][0]
        return obs, reward, done, info

    def reset(self):
        self.past_observations = []
        return self.env.reset()


env = KitchenEnvSlidingReward()
print("reset", env.reset()