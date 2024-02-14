import gym
import numpy as np
import imageio
import random
import metaworld
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from custEnvsCombined import HeirarchyEnvMeta, MetaworldWrapper

k = 10
for metaworldEnvID in [["pick-place-v2","pick_place"], ["hammer-v2","hammer"]]:

    ml1 = metaworld.ML1(metaworldEnvID[0]) # Construct the benchmark, sampling tasks
    baseEnv = ml1.train_classes[metaworldEnvID[0]]()
    task = random.choice(ml1.train_tasks)
    baseEnv.set_task(task)  # Set task

    env = HeirarchyEnvMeta(baseEnv,numModels=10,baseEnvID=metaworldEnvID[1],envStep=k)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    frames = []
    obs = baseEnv.reset()
    img = baseEnv.render(offscreen=True)

    for _ in range(200): 
        modelNo, _states = model.predict(obs)
        for _ in range(k):
            action, __states = env.model[modelNo].predict(obs)
            if getattr(baseEnv, 'curr_path_length', 0) > baseEnv.max_path_length:
                break
            obs, rewards, dones, info = baseEnv.step(action)
            frames.append(baseEnv.render(offscreen=True))
    imageio.mimsave(f"{metaworldEnvID[1]}_heirarchy_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)