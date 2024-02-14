import numpy as np
import robosuite as suite
import imageio 

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

frames = []
obs = env.reset()
img = env.render()

for _ in range(200):
    # obs = env.render(mode='rgb_array', width=width, height=height)
    frames.append(env.render())
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, rewards, dones, info = env.step(action)
    # img = model.env.render(mode="rgb_array")

imageio.mimsave(f"robosuite_test.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=10)