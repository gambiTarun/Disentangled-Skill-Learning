# Disentangled-Skill-Learning

This GitHub project provides a comprehensive testing suite for evaluating reinforcement learning (RL) algorithms across various environments and tasks. It includes scripts for interacting with environments from Robosuite, Mujoco, Metaworld, Franka Kitchen, and DeepMind Control Suite (DMC) through gym-like interfaces. The suite is designed to test the flexibility and adaptability of RL algorithms by exposing them to different dynamics, observation spaces, and challenges.

### Features

- **Diverse Environments**: Includes wrappers and test scripts for environments from Robosuite, Mujoco, Metaworld, Franka Kitchen, and DeepMind Control Suite.
- **Benchmarking Tools**: Scripts for running benchmarks and generating reward plots to evaluate the performance of RL models across tasks.
- **Custom Environment Enhancements**: Implements custom wrappers for environments, such as sliding window rewards and task hierarchies, to test advanced RL capabilities.

### Setup

1. **Dependencies**: Ensure you have Python 3.6+ installed. Install all required packages using `pip install -r requirements.txt` which should include `gym`, `numpy`, `imageio`, `stable_baselines3`, and environment-specific packages like `robosuite`, `mujoco_py`, and `metaworld`.

2. **Environment Wrappers**: Custom wrappers are provided for specific tasks to enhance the original environments with features like sliding window rewards or hierarchical task management.

### Usage

To run the test scripts, navigate to the script's directory and execute it using Python. For example:

```bash
python testMujocoEnvs.py
```

This will initiate the environment, run a predefined number of episodes with random or model-predicted actions, and save a GIF of the episode frames for visualization.

### Contributing

Contributions are welcome! If you'd like to add more environments, improve existing wrappers, or suggest new features, please open an issue or submit a pull request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
