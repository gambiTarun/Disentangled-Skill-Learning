
import gym
import numpy as np
import imageio
import os
import random
import metaworld
	
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from custEnvsCombined import AntWrapper, CheetahWrapper, WalkerWrapper, HumanoidWrapper, HopperWrapper, MetaworldWrapper, DMCWrapper
from stable_baselines3.common.monitor import Monitor


def make_env_meta(env_id, rank, obs_map, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        ml1 = metaworld.ML1(env_id) 
        env = MetaworldWrapper(ml1.train_classes[env_id](),obs_map,act_sanction=1)
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def HandpickedObs(envID,timesteps,sanc=1,obsSize=0):

    if not os.path.exists(os.getcwd()+f"/handpicked/{envID}"):
        os.makedirs(os.getcwd()+f"/handpicked/{envID}")

    if envID=="ant":

        body = ["torso","front_left_leg","front_right_leg","back_leg","right_back_leg"]
        for part in body:
            k_obs = np.zeros(obsSize, dtype=np.intp)
            
            if part=="torso":
                k_obs[0:7]=1
                k_obs[15:21]=1
            elif part=="front_left_leg":
                k_obs[7:9]=1
                k_obs[21:23]=1
            elif part=="front_right_leg":
                k_obs[9:11]=1
                k_obs[23:25]=1
            elif part=="back_leg":
                k_obs[11:13]=1
                k_obs[25:27]=1
            elif part=="right_back_leg":
                k_obs[13:15]=1
                k_obs[27:29]=1
            
            filename = f"{envID}_{part}"

            env = AntWrapper(gym.make("Ant-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(f"handpicked/{envID}/{filename}")

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(f"handpicked/{envID}/{filename}.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

    elif envID=="cheetah":

        body = ["root","fleg","bleg"]
        for part in body:
            k_obs = np.zeros(obsSize, dtype=np.intp)
            # k_act = np.zeros(6, dtype=np.intp)

            if part=="root":
                k_obs[0:3]=1
                k_obs[9:12]=1
            elif part=="fleg":
                k_obs[3:6]=1
                k_obs[12:15]=1
                # k_act[0:3]=1
            elif part=="bleg":
                k_obs[6:9]=1
                k_obs[15:18]=1
                # k_act[3:6]=1
            
            filename = f"{envID}_{part}"

            env = CheetahWrapper(gym.make("HalfCheetah-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(f"handpicked/{envID}/{filename}")

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(f"handpicked/{envID}/{filename}.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)
        
    elif envID=="walker":
        
        body = ["torso","leg","leftleg"]
        for part in body:
            k_obs = np.zeros(obsSize, dtype=np.intp)
            # k_act = np.zeros(6, dtype=np.intp)

            if part=="torso":
                k_obs[0:3]=1
                k_obs[9:12]=1
            elif part=="leg":
                k_obs[3:6]=1
                k_obs[12:15]=1
                # k_act[0:3]=1
            elif part=="leftleg":
                k_obs[6:9]=1
                k_obs[15:18]=1
                # k_act[3:6]=1
            
            filename = f"{envID}_{part}"

            env = WalkerWrapper(gym.make("Walker2d-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(f"handpicked/{envID}/{filename}")

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(f"handpicked/{envID}/{filename}.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

    elif envID=="humanoid":

        body = ["torso","abdomen","leftleg","rightleg","leftarm","rightarm"]
        for part in body:
            k_obs = np.zeros(obsSize, dtype=np.intp)
            # k_act = np.zeros(17, dtype=np.intp)

            if part=="torso":
                k_obs[0:7]=1
                k_obs[24:30]=1
            elif part=="abdomen":
                k_obs[7:10]=1
                k_obs[30:33]=1
                # k_act[0:3]=1
            elif part=="leftleg":
                k_obs[14:18]=1
                k_obs[37:41]=1
                # k_act[7:11]=1
            elif part=="rightleg":
                k_obs[10:14]=1
                k_obs[33:37]=1
                # k_act[3:7]=1
            elif part=="leftarm":
                k_obs[21:24]=1
                k_obs[44:47]=1
                # k_act[14:17]=1
            elif part=="rightarm":
                k_obs[18:21]=1
                k_obs[41:44]=1
                # k_act[11:14]=1
            
            filename = f"{envID}_{part}"

            env = HumanoidWrapper(gym.make("Humanoid-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(f"handpicked/{envID}/{filename}")

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(f"handpicked/{envID}/{filename}.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

    elif envID=="hopper":

        body = ["root","leg"]
        for part in body:
            k_obs = np.zeros(obsSize, dtype=np.intp)
            # k_act = np.zeros(3, dtype=np.intp)

            if part=="root":
                k_obs[0:3]=1
                k_obs[6:9]=1
            elif part=="leg":
                k_obs[3:6]=1
                k_obs[9:12]=1
                # k_act[0:3]=1
            
            filename = f"{envID}_{part}"
            
            env = HopperWrapper(gym.make("Hopper-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(f"handpicked/{envID}/{filename}")

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(f"handpicked/{envID}/{filename}.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)



def RandomWindow(envID,timesteps,sanc=1,obsSize=0,winSize=0,numSkills=0):
    
    if not os.path.exists(os.getcwd()+f"/randomWindow/{envID}"):
        os.makedirs(os.getcwd()+f"/randomWindow/{envID}")

    if envID=="ant":
        for i in range(numSkills):
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs = np.zeros(obsSize, dtype=np.intp)
            k_obs[winPos*winSize:(winPos+1)*winSize]=1

            fileloc = f"randomWindow/{envID}/{envID}_{i}"

            env = AntWrapper(gym.make("Ant-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(fileloc)

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
            
            imageio.mimsave(fileloc+".gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)            
            
    elif envID=="cheetah":
        for i in range(numSkills):
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs = np.zeros(obsSize, dtype=np.intp)
            k_obs[winPos*winSize:(winPos+1)*winSize]=1
            
            fileloc = f"randomWindow/{envID}/{envID}_{i}"

            env = CheetahWrapper(gym.make("HalfCheetah-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(fileloc)

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(fileloc+".gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

    elif envID=="walker":
        for i in range(numSkills):
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs = np.zeros(obsSize, dtype=np.intp)
            k_obs[winPos*winSize:(winPos+1)*winSize]=1
            
            fileloc = f"randomWindow/{envID}/{envID}_{i}"

            env = WalkerWrapper(gym.make("Walker2d-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(fileloc)

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(fileloc+".gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)
        
    elif envID=="humanoid":
        for i in range(numSkills):
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs = np.zeros(obsSize, dtype=np.intp)
            k_obs[winPos*winSize:(winPos+1)*winSize]=1

            fileloc = f"randomWindow/{envID}/{envID}_{i}"

            env = HumanoidWrapper(gym.make("Humanoid-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(fileloc)

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(fileloc+".gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)
        
    elif envID=="hopper":
        for i in range(numSkills):
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs = np.zeros(obsSize, dtype=np.intp)
            k_obs[winPos*winSize:(winPos+1)*winSize]=1

            fileloc = f"randomWindow/{envID}/{envID}_{i}"
            
            env = HopperWrapper(gym.make("Hopper-v3"),k_obs,act_sanction=sanc)
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(fileloc)

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(fileloc+".gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

    elif envID[:9]=="metaworld":
        metaworld_envID = ""

        if "hammer" in envID:
            metaworld_envID = "hammer-v2"
        elif "pick_place" in envID:
            metaworld_envID = "pick-place-v2"

        for i in range(numSkills):
            winPos = random.randint(0,obsSize-winSize-1)
            k_obs = np.zeros(obsSize, dtype=np.intp)
            k_obs[winPos*winSize:(winPos+1)*winSize]=1
            
            fileloc = f"randomWindow/{envID}/{envID}_{i}"

            env = SubprocVecEnv([make_env_meta(metaworld_envID, i, k_obs) for i in range(8)])
    
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=train_timesteps)
            model.save(fileloc)

            frames = []
            obs = env.reset()
            img = env.render(mode="rgb_array")
            
            for _ in range(200):
                frames.append(env.render(mode="rgb_array"))
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

            imageio.mimsave(fileloc+".gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

        

if __name__ == "__main__":
    
    env = None
    train_timesteps = 10000
    sanc = 1
    log_dir = './results'

    # envID, obsSize, winSize, numSkills
    env_list = [("ant",111,10,40), ("cheetah",18,6,8), ("walker",18,6,8), ("humanoid",378,37,100), ("hopper",12,3,5),
                ("metaworld_hammer",33,8,10), ("metaworld_pick_place",31,8,10)]

    for envID, obsSize, winSize, numSkills in env_list:

        # HandpickedObs(envID,train_timesteps,sanc=sanc,obsSize=obsSize)
        RandomWindow(envID,train_timesteps,sanc=sanc,obsSize=obsSize,winSize=winSize,numSkills=numSkills)
        break
        