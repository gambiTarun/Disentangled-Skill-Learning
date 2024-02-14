import numpy as np
import imageio
import os
import random
import argparse
import glob

os.environ['MUJOCO_GL'] = 'egl'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from custEnvsDMC import DMCWrapper
from stable_baselines3.common.monitor import Monitor

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--total-time-steps', type=int, default=10000)
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--statemap', type=str, default="randomWindow")
    args = parser.parse_args()
    return args

def make_env_dmc(envID, rank, obs_map, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DMCWrapper(envID,obs_map,reward="custom",seed=seed)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def GenSkills(envID,timesteps,num_frames,obsSize=0,winSize=0,numSkills=0,stateMap="randomWindow"):

    gif_loc = "{}/{}/gifs_t_{}/".format(stateMap,envID,timesteps)
    model_loc = "{}/{}/models_t_{}/".format(stateMap,envID,timesteps)
    log_loc = "{}/{}/logs_t_{}".format(stateMap,envID,timesteps)

    for d in [gif_loc,model_loc,log_loc]:
        os.makedirs(os.getcwd()+f"/{d}", exist_ok=True)

    # clearing out the model folder (for status check)
    for f in glob.glob(model_loc+"/*"):
        os.remove(f)

    dmc_envID = envID[4:]

    if stateMap=="slidingWindow":
        numSkills = obsSize-winSize

    for i in range(numSkills):
        random.seed(i)
        k_obs = np.zeros(obsSize, dtype=np.intp)
        if stateMap=="randomWindow":
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs[winPos:winPos+winSize]=1
        elif stateMap=="slidingWindow":
            k_obs[i:i+winSize]=1

        envs = SubprocVecEnv([make_env_dmc(dmc_envID, j, k_obs, f"{log_loc}/skill_{i}") for j in range(8)])

        model = PPO("MlpPolicy", envs, verbose=1)
        model.learn(total_timesteps=train_timesteps)
        model.save(model_loc+f"skill_{i}")

        frames = []
        env = DMCWrapper(dmc_envID,k_obs,reward="custom")
        obs = env.reset()
        img = env.render(mode="rgb_array")
        
        for _ in range(num_frames):
            frames.append(env.render(mode="rgb_array"))
            action, _states = model.predict(obs)
            obs, _rewards, _dones, _info = env.step(action)

        imageio.mimsave(gif_loc+f"skill_{i}.gif", [np.array(img) for i, img in enumerate(frames)], fps=20)


if __name__ == "__main__":

    args = get_args()
    train_timesteps = args.total_time_steps
    frames = args.frames
    
    # envID, obsSize, winSize, numSkills
    env_list = [("dmc_stack_2_bricks_features",68,10,20), ("dmc_place_cradle_features",58,10,18), ("dmc_reassemble_3_bricks_fixed_order_features",81,10,30)]

    for envID, obsSize, winSize, numSkills in env_list:

        GenSkills(envID,train_timesteps,frames,obsSize=obsSize,winSize=winSize,numSkills=numSkills,stateMap=args.statemap)
        