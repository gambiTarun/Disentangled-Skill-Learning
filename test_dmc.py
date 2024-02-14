from gym import spaces, Env
import numpy as np
import imageio
import random
from stable_baselines3 import PPO


# dmc
from dm_env import specs

# Manipulation
from dm_control import manipulation,suite



class DMCWrapper(Env):
    def __init__(
        self,
        environment_name,
        seed=1,
        from_pixels=False,
        height=400,
        width=400,
        camera_id=0,
        frame_skip=1,
        channels_first=False
    ):
    
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        self._env = manipulation.load(environment_name, seed=seed)

        # true and normalized action spaces
        self._true_action_space = self._spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

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

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
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


# stack_2_bricks_features
# stack_2_bricks_vision
# stack_2_bricks_moveable_base_features
# stack_2_bricks_moveable_base_vision
# stack_3_bricks_features
# stack_3_bricks_vision
# stack_3_bricks_random_order_features
# stack_2_of_3_bricks_random_order_features
# stack_2_of_3_bricks_random_order_vision
# reassemble_3_bricks_fixed_order_features
# reassemble_3_bricks_fixed_order_vision
# reassemble_5_bricks_random_order_features
# reassemble_5_bricks_random_order_vision
# lift_brick_features
# lift_brick_vision
# lift_large_box_features
# lift_large_box_vision
# place_brick_features
# place_brick_vision
# place_cradle_features
# place_cradle_vision
# reach_duplo_features
# reach_duplo_vision
# reach_site_features
# reach_site_vision
# print('\n'.join(manipulation.ALL))

env = manipulation.load('stack_2_of_3_bricks_random_order_vision', task_kwargs=dict(random=32), environment_kwargs=dict(flat_observation=True), visualize_reward=False)
# env = DMCWrapper('stack_2_of_3_bricks_random_order_features', seed=42)
env = suite.load('quadruped', 'walk', task_kwargs=dict(random=32), environment_kwargs=dict(flat_observation=True), visualize_reward=False)
obs_spec = env.observation_spec()
print(f"\nDEB: {obs_spec}\n")
action_spec = env.action_spec()

# def sample_random_action():
#   return env.random_state.uniform(
#       low=action_spec.minimum,
#       high=action_spec.maximum,
#   ).astype(action_spec.dtype, copy=False)

# Step the environment through a full episode using random actions and record
# the camera observations.

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

frames = []
obs = env.reset()

for _ in range(200):
    frames.append(env.render(mode="rgb_array", height=480, width=640))
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

imageio.mimsave(f"dmc_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)

# frames.append(timestep.observation['front_close'])

# while not timestep.last():
#     action, _states = model.predict(timestep)
#     obs = env.step(action)
#     frames.append(timestep.observation['front_close'])

# imageio.mimsave(f"dmc_test.gif", [np.array(img) for i, img in enumerate(np.concatenate(frames, axis=0))], fps=10)