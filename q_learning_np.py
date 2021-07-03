#!/usr/bin/env python3

import time
import argparse
import numpy as np
import math

from gridworld_np import GridWorldEnv

import time
import datetime
import utils
import sys

num_epochs = 12
discount = 0.9

EPS_START = 1.0
EPS_END = 0.05

MIN_STEPS_TRESHOLD = 13
MIN_EPISODES_TRESHOLD = 20


def eps_greedy_action(Q_values, state, rng, num_actions, eps_final):
	rand_val = rng.uniform()
	# eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
	eps_threshold = eps_final #as per Christos set up, using constant eps

	if rand_val > eps_threshold: 
		Q_values_squeezed = np.squeeze(Q_values[state,:])
		return eps_threshold, np.random.choice(np.where(Q_values_squeezed == Q_values_squeezed.max())[0])


	else:
		return eps_threshold, rng.integers(low=0, high=num_actions, size=1)[0]

def getStateID(obs, grid_size):
	state_space = np.zeros((grid_size, grid_size))
	x = obs['x'] 	
	y = obs['y']
	state_space[x,y] = 1
	state_space_vec = state_space.reshape(-1)
	return np.nonzero(state_space_vec)

def map_actions(action, env):
	actions = [
	env.actions.left,
	env.actions.right,
	env.actions.up,
	env.actions.down,
	]

	return actions[action]


def compute_td_error(state, next_state, action, reward, Q_values):
	max_nextQ = np.max(np.squeeze(Q_values[next_state,:]), axis=0)
	td_error = reward + (discount * max_nextQ) - Q_values[state,action]
	return td_error

def main():

	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--seed",
		type=int,
		help="random seed to generate the environment with",
		default=0
	)

	parser.add_argument(
		"--grid_size",
		type=int,
		help="size of the grid on each side",
		default=10
	)

	parser.add_argument(
		"--max_steps",
		type=int,
		help="max steps per episode",
		default=20000
	)

	parser.add_argument(
		"--eps",
		type=float,
		help="epsilon-greedy value",
		default=0.05
	)

	parser.add_argument(
		"--num_episodes",
		type=int,
		help="number of episodes per epoch",
		default=10000
	)

	parser.add_argument(
		"--lr_Q",
		type=float,
		help="learning rate for Q values",
		default=0.1
	)

	parser.add_argument(
		"--lambda_factor",
		type=float,
		help="lambda factor for eligibility trace",
		default=0.9
	)

	parser.add_argument(
		"--save_np_files",
		action='store_true',
		help="Use when we want to save the numpy files for Q values",
	)

	parser.add_argument(
		"--save_dir_Q",
		type=str,
		help="save directory for Q numpy files",
		default='./'
	)

	parser.add_argument(
		"--save_dir",
		type=str,
		help="save directory for SF numpy files",
		default='./'
	)

	parser.add_argument(
		"--algo_name",
		type=str,
		help="Name for algorithm",
		required=True
	)



	args = parser.parse_args()

	algo = args.algo_name

	#create train dir
	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	default_model_name = f"{algo}_seed{args.seed}_{date}"


	model_name = default_model_name
	model_dir = utils.get_model_dir(model_name, args.save_dir)


	# Load loggers and Tensorboard writer

	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)

	# Log command and all script arguments

	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))

	# Set seed for all randomness sources
	utils.seed(args.seed)

	eps_final = args.eps
	num_episodes = args.num_episodes
	lr = args.lr_Q
	lam_factor = args.lambda_factor
	max_steps = args.max_steps
	grid_size = args.grid_size


	env = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	txt_logger.info("Environments loaded\n")

	status = {"num_steps": 0, "update": 0, "num_episodes":0}
	txt_logger.info("Training status loaded\n")

	Q_values = np.zeros((grid_size*grid_size, len(env.actions)))
	rng =  np.random.default_rng(args.seed)

	start_time = time.time()
	totalReturns = []
	returnPerEpisode = [] 
	stepsPerEpisode = []
	steps_done = 0
	episode_count = 0

	save_dir_Q = args.save_dir_Q

	env1 = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	env2 = GridWorldEnv(size=grid_size, goal_pos=(grid_size-1,grid_size-1))

	steps_to_first_reward = np.zeros((num_epochs))
	steps_to_good_policy = np.zeros((num_epochs))

	for epoch in range(num_epochs):

		if epoch%2==0:
			env = env1
		else:
			env = env2

		count = 0

		returnPerEpisode = [] 
		stepsPerEpisode = []

		good_policy_count = 0

		for episode in range(num_episodes):
			
			state = env.reset()
			state = getStateID(state, grid_size)
			eps_reward = 0

			etrace = np.zeros((grid_size*grid_size, len(env.actions))) 

			for time_steps in range(max_steps):
				
				eps, action = eps_greedy_action(Q_values, state, rng, len(env.actions), eps_final)
				steps_done += 1

				next_state, reward, done, info = env.step(map_actions(action, env))
				next_state = getStateID(next_state, grid_size)
				
				eps_reward += reward

				#update eligibility trace
				etrace = np.multiply(etrace,discount * lam_factor)
				etrace[state,action] = 1

				td_error = compute_td_error(state, next_state, action, reward, Q_values)
				td_error_etrace = np.multiply(etrace, td_error)

				Q_values = Q_values + (lr * td_error_etrace)

				state = next_state
				count += 1

				if done:
					returnPerEpisode.append(eps_reward)
					Q_values = np.clip(Q_values,  a_min=0.0, a_max=1.0)
					stepsPerEpisode.append(time_steps)
					break

			#logging stats
			duration = int(time.time() - start_time)

			totalReturn_val = np.sum(np.array(returnPerEpisode))
			moving_avg_returns = np.mean(np.array(returnPerEpisode[-MIN_EPISODES_TRESHOLD:]))
			moving_avg_steps = np.mean(np.array(stepsPerEpisode[-MIN_EPISODES_TRESHOLD:]))

			if moving_avg_steps <= MIN_STEPS_TRESHOLD and steps_to_good_policy[epoch] == 0 and (len(stepsPerEpisode) >= MIN_EPISODES_TRESHOLD):
				steps_to_good_policy[epoch] = count


			header = ["epoch", "steps", "cur episode steps", "episode", "duration"]
			data = [epoch, steps_done, stepsPerEpisode[-1], episode_count, duration]

			header += ["eps", "cur episode return", "returns", "avg returns", "avg steps", "steps to good policy"]
			data += [eps, returnPerEpisode[-1], totalReturn_val, moving_avg_returns, moving_avg_steps, steps_to_good_policy[epoch]]

			if episode_count % 50 == 0: 
				txt_logger.info(
						"Epoch {} | S {} | Epi Steps {} | Episode {} | D {} | EPS {:.3f} | R {:.3f} | Total R {:.3f} | Avg R {:.3f} | Avg S {} | Good Policy {}"
						.format(*data))

			if episode_count == 0:
				csv_logger.writerow(header)
			csv_logger.writerow(data)
			csv_file.flush()

			# if episode_count % 10 == 0 and args.save_np_files: 
			# 	filename = save_dir_Q + 'Q_etraceReplace_'+str(episode_count)+'.npy'
			# 	np.save(filename, Q_values)
				
			episode_count += 1





if __name__ == "__main__":
	main()

# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)