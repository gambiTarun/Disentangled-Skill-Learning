import numpy as np
import random
from gym import Env, spaces, Wrapper
from gym.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO

glob_lambda=0.5
random.seed(42)

class HeirarchyEnvMeta(Env):
    def __init__(self,env,numModels=0,baseEnvID="",envStep=10):
        super(HeirarchyEnvMeta,self)
        self.baseEnv = env
        self.baseEnv = TimeLimit(self.baseEnv, max_episode_steps=self.baseEnv.max_path_length)
        self.envStep = envStep
        
        self.action_space = spaces.Discrete(numModels)
        self.observation_space = self.baseEnv.observation_space
        self.prev_actions = []

        self.model = [None]*numModels
        for i in range(numModels):        
            self.model[i] = PPO.load(f"randomWindow/metaworld_{baseEnvID}/metaworld_{baseEnvID}_{i}")


    def get_obs(self):
        # position = env.sim.data.qpos.flat.copy()
        # velocity = env.sim.data.qvel.flat.copy()

        # observation = np.concatenate((position, velocity)).ravel()
        # return observation
        # do frame stacking
        pos_goal = self.baseEnv._get_pos_goal()
        if self.baseEnv._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self.baseEnv._get_curr_obs_combined_no_goal()
        # do frame stacking
        if self.baseEnv.isV2:
            obs = np.hstack((curr_obs, self.baseEnv._prev_obs, pos_goal))
        else:
            obs = np.hstack((curr_obs, pos_goal))
        self.baseEnv._prev_obs = curr_obs
        return obs

    def step(self, action):
		
        self.prev_actions.append(action)
        baseEnv_obs = self.get_obs()
        # self.do_simulation(action)

        # select the model used to predict next action
        # use the model to predict the action for base_env
        reward = 0
        for _ in range(self.envStep):
            baseEnv_action, _states = self.model[action].predict(baseEnv_obs)
            baseEnv_obs, re, done, info = self.baseEnv.step(baseEnv_action)
            reward += re
            # envDone = done or getattr(self.baseEnv, 'curr_path_length', 0) > self.baseEnv.max_path_length
            if done:
                break

        # done = self.done
        # reward = np.linalg.norm(observation[start_dim:start_dim+num_of_dim]-prev_observation[start_dim:start_dim+num_of_dim])

        return baseEnv_obs, reward, done, info

    def reset(self):
        return self.baseEnv.reset()

class HeirarchyEnvAnt(Env):
    def __init__(self,env,numModels=0):
        super(HeirarchyEnvAnt,self)
        self.baseEnv = env

        self.action_space = spaces.Discrete(numModels)
        self.observation_space = self.baseEnv.observation_space
        self.prev_actions = []

        self.model = [None]*numModels
        for i in range(numModels):        
            self.model[i] = PPO.load(f"randomWindow/ant/ant_{i}")


    def get_obs(self, env):
        position = env.sim.data.qpos.flat.copy()[2:]
        velocity = env.sim.data.qvel.flat.copy()
        contact_force = env.contact_forces.flat.copy()

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def step(self, action):
		
        self.prev_actions.append(action)
        baseEnv_obs = self.get_obs(self.baseEnv)
        # self.do_simulation(action)

        # select the model used to predict next action
        # use the model to predict the action for base_env
        reward = 0
        for _ in range(10):
            baseEnv_action, _states = self.model[action].predict(baseEnv_obs)
            baseEnv_obs, re, done, info = self.baseEnv.step(baseEnv_action)
            reward += re

        # done = self.done
        # reward = np.linalg.norm(observation[start_dim:start_dim+num_of_dim]-prev_observation[start_dim:start_dim+num_of_dim])

        return baseEnv_obs, reward, done, info

    def reset(self):
        return self.baseEnv.reset()
        

class AntWrapper(Wrapper):
    def __init__(self, 
    env, 
    k_obs,
    act_sanction=1,
    lambda_act=glob_lambda
    ):
        super().__init__(env)
        self.env._k_obs = k_obs
        self.env._act = act_sanction
        self.env._lambda = lambda_act
        self.env.prev_act = np.zeros(8)

    def step(self, action):
        prev_observation = self.get_obs()
        observation, reward, done, info = self.env.step(action)
        new_observation = self.get_obs()

        new_reward = np.linalg.norm(new_observation[self.env._k_obs==1] - prev_observation[self.env._k_obs==1], ord=2) - \
                    np.linalg.norm(new_observation[self.env._k_obs==0] - prev_observation[self.env._k_obs==0], ord=2) + \
                        self.env._lambda*(np.linalg.norm(action - self.env.prev_act, ord=2))*self.env._act

        self.env.prev_act = action.copy()
        return observation, new_reward, done, info

    def get_obs(self):
        position = self.env.sim.data.qpos.flat.copy()[2:]
        velocity = self.env.sim.data.qvel.flat.copy()
        contact_force = self.env.contact_forces.flat.copy()

        observations = np.concatenate((position, velocity, contact_force))

        return observations
    
class CheetahWrapper(Wrapper):
    def __init__(self, 
    env, 
    k_obs,
    act_sanction=1,
    lambda_act=glob_lambda
    ):
        super().__init__(env)
        self.env._k_obs = k_obs
        self.env._act = act_sanction
        self.env._lambda = lambda_act
        self.env.prev_act = np.zeros(6)

    def step(self, action):
        prev_observation = self.get_obs()
        observation, reward, done, info = self.env.step(action)
        new_observation = self.get_obs()

        new_reward = np.linalg.norm(new_observation[self.env._k_obs==1] - prev_observation[self.env._k_obs==1], ord=2) - \
                    np.linalg.norm(new_observation[self.env._k_obs==0] - prev_observation[self.env._k_obs==0], ord=2) + \
                        self.env._lambda*(np.linalg.norm(action - self.env.prev_act, ord=2))*self.env._act

        self.env.prev_act = action.copy()
        return observation, new_reward, done, info

    def get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        observation = np.concatenate((position, velocity)).ravel()
        return observation


class WalkerWrapper(Wrapper):
    def __init__(self, 
    env, 
    k_obs,
    act_sanction=1,
    lambda_act=glob_lambda
    ):
        super().__init__(env)
        self.env._k_obs = k_obs
        self.env._act = act_sanction
        self.env._lambda = lambda_act
        self.env.prev_act = np.zeros(6)

    def step(self, action):
        prev_observation = self.get_obs()
        observation, reward, done, info = self.env.step(action)
        new_observation = self.get_obs()

        new_reward = np.linalg.norm(new_observation[self.env._k_obs==1] - prev_observation[self.env._k_obs==1], ord=2) - \
                    np.linalg.norm(new_observation[self.env._k_obs==0] - prev_observation[self.env._k_obs==0], ord=2) + \
                        self.env._lambda*(np.linalg.norm(action - self.env.prev_act, ord=2))*self.env._act

        self.env.prev_act = action.copy()
        return observation, new_reward, done, info

    def get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        observation = np.concatenate((position, velocity)).ravel()
        return observation

class HumanoidWrapper(Wrapper):
    def __init__(self, 
    env, 
    k_obs,
    act_sanction=1,
    lambda_act=glob_lambda
    ):
        super().__init__(env)
        self.env._k_obs = k_obs
        self.env._act = act_sanction
        self.env._lambda = lambda_act
        self.env.prev_act = np.zeros(17)

    def step(self, action):
        prev_observation = self.get_obs()
        observation, reward, done, info = self.env.step(action)
        new_observation = self.get_obs()

        new_reward = np.linalg.norm(new_observation[self.env._k_obs==1] - prev_observation[self.env._k_obs==1], ord=2) - \
                    np.linalg.norm(new_observation[self.env._k_obs==0] - prev_observation[self.env._k_obs==0], ord=2) + \
                        self.env._lambda*(np.linalg.norm(action - self.env.prev_act, ord=2))*self.env._act

        self.env.prev_act = action.copy()
        return observation, new_reward, done, info

    def get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()

        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

class HopperWrapper(Wrapper):
    def __init__(self, 
    env, 
    k_obs,
    act_sanction=1,
    lambda_act=glob_lambda
    ):
        super().__init__(env)
        self.env._k_obs = k_obs
        self.env._act = act_sanction
        self.env._lambda = lambda_act
        self.env.prev_act = np.zeros(3)

    def step(self, action):
        prev_observation = self.get_obs()
        observation, reward, done, info = self.env.step(action)
        new_observation = self.get_obs()

        new_reward = np.linalg.norm(new_observation[self.env._k_obs==1] - prev_observation[self.env._k_obs==1], ord=2) - \
                    np.linalg.norm(new_observation[self.env._k_obs==0] - prev_observation[self.env._k_obs==0], ord=2) + \
                        self.env._lambda*(np.linalg.norm(action - self.env.prev_act, ord=2))*self.env._act

        self.env.prev_act = action.copy()
        return observation, new_reward, done, info

    def get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(self.sim.data.qvel.flat.copy(), -10, 10)

        observation = np.concatenate((position, velocity)).ravel()
        return observation

class MetaworldWrapper(Wrapper):
    
        def __init__(self, 
        env, 
        k_obs,
        act_sanction=1,
        lambda_act=glob_lambda
        ):
            super().__init__(env)
            self.env._k_obs = k_obs
            self.env._act = act_sanction
            self.env._lambda = lambda_act
            self.env.prev_act = np.zeros(4)

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

