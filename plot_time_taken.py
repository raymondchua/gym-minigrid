#!/usr/bin/env python3
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

q_avg_return = []
BF_avg_return = []

q_first = []
BF_first = []

q_good_policy = []
BF_good_policy = []

q_learning_filename = '/Users/raymondchua/Documents/gym-minigrid/storage/MiniGrid-ClassicGridWorldS7-v0_Q-learning_seed0_21-01-13-22-26-55/log.csv'
benna_fusi_model_filename = '/Users/raymondchua/Documents/gym-minigrid/storage/MiniGrid-ClassicGridWorldS7-v0_Benna-Fusi_model_Q-learning_seed0_21-01-13-08-54-40/log.csv'



with open(benna_fusi_model_filename) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	count = 0 
	for row in readCSV:
		count +=1
		if count == 1:
			continue
		else:
			# print(row[-1])
			BF_avg_return.append(float(row[-3]))
		
		if (count-1) % 700 == 0: 
			BF_first.append(float(row[-2]))
			BF_good_policy.append(float(row[-1]))
			

with open(q_learning_filename) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	count = 0 
	for row in readCSV:
		count +=1
		if count == 1:
			continue
		else:
			# print(row[-1])
			q_avg_return.append(float(row[-3]))

		if (count-1) % 700 == 0: 
			q_first.append(float(row[-2]))
			q_good_policy.append(float(row[-1]))
			print(q_good_policy[-1])

# sns.lineplot(data=avg_return)
# plt.ylim(0, 20)
# plt.show()

# style
# plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = sns.crayon_palette(colors=['Red',  'Green', 'Denim', 'Black',  'Brown','Tropical Rain Forest', 'Red', 'Neon Carrot'])

fig, ax = plt.subplots()


dataset = pd.DataFrame({'Benna-Fusi Model': BF_first, 'Q-learning':q_first})
# print(len(BF_avg_return))
# ax = sns.lineplot(data=avg_return)
# ax.plt.show()

# f, ax = plt.subplots(1, 1, figsize=(12,7))
sns.lineplot(data=dataset['Benna-Fusi Model'], color=palette[0], ax=ax, label='Benna-Fusi Model')
sns.lineplot(data=dataset['Q-learning'], color=palette[1], ax=ax, label='Q-learning')
# ax.set_ylabel('Running Avg Returns')
ax.set_ylabel('Time to first reward (Num of timesteps)')
# ax.set_ylabel('Time to good policy (Num of timesteps)')
ax.set_xlabel('Num of epochs')
ax.legend()
plt.show()


