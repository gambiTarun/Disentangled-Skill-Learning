import random
import numpy as np
from gym import Env, spaces
from stable_baselines3 import PPO

from dm_env import specs
# Manipulation
from dm_control import manipulation

glob_lambda=0.5
random.seed(42)

class HeirarchyEnvDMC(Env):
    def __init__(self,env,num_models=0,base_envID="",env_step=10,state_map="randomWindow",baseEnv_t=10000):
        super(HeirarchyEnvDMC,self)
        self.baseEnv = env
        self.envStep = env_step
        
        self.action_space = spaces.Discrete(num_models)
        self.observation_space = self.baseEnv.observation_space
        self.prev_actions = []

        self.model = [None]*num_models
        for i in range(num_models):        
            self.model[i] = PPO.load("{}/dmc_{}/models_t_{}/skill_{}".format(state_map,base_envID,baseEnv_t,i))

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

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

class DMCWrapper(Env):
    def __init__(
        self,
        environment_name,
        k_obs=0,
        reward="default",
        act_sanction=1,
        lambda_act=glob_lambda,
        seed=1,
        from_pixels=False,
        height=400,
        width=400,
        camera_id=0,
        channels_first=False
    ):
    
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._channels_first = channels_first
        self.rewardFunc = reward

        # create task
        self._env = manipulation.load(environment_name, seed=seed)
        self.prev_time_step = self._env.reset()
        

        # true and normalized action spaces
        self._true_action_space = self._spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        self._act_sanc = act_sanction
        self._lambda = lambda_act
        self.prev_act = np.zeros(self._true_action_space.shape)

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = self._spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )
            
        self._k_obs = k_obs

        # self._state_space = _spec_to_box(
        #     self._env.observation_spec().values(),
        #     np.float64
        # )
        
        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = self._flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    # @property
    # def state_space(self):
    #     return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    # @property
    # def reward_range(self):
    #     return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        extra = {'internal_state': self._env.physics.get_state().copy()}
        
        prev_observation = self._get_obs(self.prev_time_step)
        time_step = self._env.step(action)
        new_observation = self._get_obs(time_step)

        reward = 0
        if self.rewardFunc == "default":
            reward += time_step.reward or 0
        elif self.rewardFunc == "custom":
            reward += np.linalg.norm(new_observation[self._k_obs==1] - prev_observation[self._k_obs==1], ord=2) - \
                np.linalg.norm(new_observation[self._k_obs==0] - prev_observation[self._k_obs==0], ord=2) + \
                    self._lambda*(np.linalg.norm(action - self.prev_act, ord=2))*self._act_sanc

        self.prev_time_step = time_step
        done = time_step.last()
            
        obs = self._get_obs(time_step)
        self.current_state = self._flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = self._flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

    def _spec_to_box(self, spec, dtype):
        def extract_min_max(s):
            assert s.dtype == np.float64 or s.dtype == np.float32
            dim = int(np.prod(s.shape))
            if type(s) == specs.Array:
                bound = np.inf * np.ones(dim, dtype=np.float32)
                return -bound, bound
            elif type(s) == specs.BoundedArray:
                zeros = np.zeros(dim, dtype=np.float32)
                return s.minimum + zeros, s.maximum + zeros

        mins, maxs = [], []
        for s in spec:
            mn, mx = extract_min_max(s)
            mins.append(mn)
            maxs.append(mx)
        low = np.concatenate(mins, axis=0).astype(dtype)
        high = np.concatenate(maxs, axis=0).astype(dtype)
        assert low.shape == high.shape
        return spaces.Box(low, high, dtype=dtype)


    def _flatten_obs(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0)
