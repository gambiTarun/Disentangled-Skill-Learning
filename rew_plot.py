import numpy as np
import csv
import seaborn as sns
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
data1 = {"rewards":[], "Episodes x 10":[], "type":[]}
num_episodes = 5000
seed_start = 0
seed_end = 8
explorer = 'revd'
window = 20
# listOfMasses = [0.05,0.1,0.15,0.2,0.25,0.3, 0.35, 0.4, 0.45, 0.5]
# masses = [0.1,0.2,0.3, 0.4, 0.5]
# sizes = [0.015,0.065]
# a = list(chain.from_iterable([[(masses[i],sizes[j]) for i in range(len(masses))] for j in range(len(sizes))]))
# for item in a:
# 	games_df = pd.read_csv(f'/lab/ssontakk/CausalWorld/tutorials/stable_baselines/stacking3Gen/{item[0]}_{item[1]}.monitor.csv', skiprows=2, header = None)
# 	# sumRew = []
# 	for j in range(4000):
# 		data1["rewards"].append(games_df[0].iloc[j])
# 		data1["t"].append(j)
# 		data1["type"].append("Generalist")
# # 	# rewards.append(sumRew)
# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/revd/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"revd_joint")
# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/rise/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"rise_joint")
# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/ride/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"ride_joint")
# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/rnd/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"rnd_joint")
# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/re3/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"re3_joint")

# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/saliency/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"saliency_joint")

# games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/{explorer}/turtlebot_nav/ppo/joint/0.monitor.csv', skiprows=1, header = 0)
# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/{explorer}/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		# data1["rewards_roll"].append(games_df[0].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"{explorer}_joint")

# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/{explorer}/turtlebot_nav/ppo/joint/with_saliency/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"{explorer}_joint_with_saliency")
for i in range(seed_start,seed_end):
	games_df = pd.read_csv(f'/lab/ssontakk/tgambhir/heirarchyModels/dmc_place_cradle_features/slidingWindow/logs_ht_1000000_bt_100000/{i}.monitor.csv', skiprows=1, header = 0)
	games_df['r'] = games_df['r'].rolling(window=window).mean()
	# sumRew = []
	for j in range(num_episodes):
		print(i,j)
		data1["rewards"].append(games_df['r'].iloc[j])
		data1["Episodes x 10"].append(j)
		data1["type"].append(f"ours")

# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/re3/multitask_pretrain/turtlebot_nav/ppo/task_rew_True_curiosity_True_saliency_True/finetuned_turtlebot_nav_Pomaria_1_int/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"Saliency")

# for i in range(seed_start,seed_end):
# 	games_df = pd.read_csv(f'/lab/ssontakk/rl-exploration-baselines/examples/logs/tr_0_collision/turtlebot_nav/ppo/joint/{i}.monitor.csv', skiprows=1, header = 0)
# 	games_df['r'] = games_df['r'].rolling(window=window).mean()
# 	# sumRew = []
# 	for j in range(num_episodes):
# 		print(i,j)
# 		data1["rewards"].append(games_df['r'].iloc[j])
# 		data1["Episodes x 10"].append(j)
# 		data1["type"].append(f"tabula rasa no collision penalty")


sns.set_theme(style="darkgrid")
sns_plot = sns.lineplot(data=data1,x="Episodes x 10",y="rewards",hue="type")
# plt.legend(loc='lower right', borderaxespad=0)
# # plt.ylim(-5, 10)

# # sns_plot.set(xscale="log")
# # sns_plot = sns.lineplot(data=data,x="t",y="rewards")

sns_plot.figure.savefig(f"ours.png")

# sns_plot.figure.savefig("PPO-Ihlen_0_int-TR.png")
# print(games_df)