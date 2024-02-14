import numpy as np
import argparse
import os
import imageio
import glob
from dm_control import manipulation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from custEnvsDMC import HeirarchyEnvDMC, DMCWrapper

from matplotlib import animation
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--numframes', type=int, default=200)
    parser.add_argument('--skipframes', type=int, default=10)
    parser.add_argument('--T', type=int, default=100000)
    parser.add_argument('--statemap', type=str, default="slidingWindow")
    args = parser.parse_args()
    return args

def make_henvs_dmc(rank, log_dir, dmc_envID=None, num_skills=None, env_step=None, state_map=None, baseEnv_t=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        baseEnv = DMCWrapper(dmc_envID, seed=seed + rank)
        # baseEnv = manipulation.load(dmc_envID)
        # env = HeirarchyEnvDMC(baseEnv,num_models=num_skills,base_envID=dmc_envID,env_step=env_step,state_map=state_map,baseEnv_t=baseEnv_t)
        baseEnv = Monitor(baseEnv, os.path.join(log_dir, str(rank)))
        baseEnv.seed(seed+rank)
        return baseEnv
    # set_global_seeds(seed)
    return _init


def save_frames_as_gif(frames, path='./', filename=f'gifs/re3_multitask.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


if __name__=="__main__":

    args = get_args() 
    num_frames = args.numframes
    skip_frames = args.skipframes
    timesteps = args.T
    stateMap = args.statemap
    
    for dmc_envID, num_skills in [("place_cradle_features",18),("reassemble_3_bricks_fixed_order_features",30), ("stack_2_bricks_features",20)]:

        gif_loc = "heirarchyModels/dmc_{}/{}/gifs_baseline_t_{}/".format(dmc_envID,stateMap,timesteps)
        model_loc = "heirarchyModels/dmc_{}/{}/logs_baseline_t_{}/".format(dmc_envID,stateMap,timesteps)
        log_loc = "heirarchyModels/dmc_{}/{}/logs_baseline_t_{}/".format(dmc_envID,stateMap,timesteps)

        for d in [gif_loc,model_loc,log_loc]:
            os.makedirs(os.getcwd()+f"/{d}", exist_ok=True)

        # clearing out the gif folder 
        for f in glob.glob(gif_loc+"/*"):
            os.remove(f)

        henvs = SubprocVecEnv([make_henvs_dmc(i, log_loc, dmc_envID=dmc_envID) for i in range(8)])
        # baseEnv = DMCWrapper(dmc_envID)
        # env = HeirarchyEnvDMC(baseEnv,numModels=10,baseEnvID=dmc_envID,envStep=skip_frames)

        model = PPO("MlpPolicy", henvs, verbose=1)
        model.learn(total_timesteps=timesteps)
        model.save(model_loc+f"{dmc_envID}")

        baseEnv = DMCWrapper(dmc_envID)
        # henv = HeirarchyEnvDMC(baseEnv,num_models=num_skills,base_envID=dmc_envID,env_step=skip_frames,state_map=stateMap,baseEnv_t=baseEnv_t)

        frames = []
        obs = baseEnv.reset()

        for _ in range(num_frames):
            frames.append(baseEnv.render(mode="rgb_array"))
            action, _states = model.predict(obs)
            obs, reward, done, info = baseEnv.step(action)

        print(len(frames))
        print(frames[0].shape)
        # save_frames_as_gif(frames, filename=gif_loc+f"{dmc_envID}.gif")
        imageio.mimsave(gif_loc+f"{dmc_envID}.gif", [np.array(img) for i, img in enumerate(frames) if i%2==0], fps=20)
