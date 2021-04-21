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

grid_size =  10
num_epochs = 12
discount = 0.9

max_steps = 20000

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


def key_handler(event):
	print('pressed', event.key)

	if event.key == 'escape':
		window.close()
		return

	if event.key == 'backspace':
		reset()
		return

	if event.key == 'left':
		step(env.actions.left)
		return
	if event.key == 'right':
		step(env.actions.right)
		return
	if event.key == 'up':
		step(env.actions.up)
		return

	# Spacebar
	if event.key == ' ':
		step(env.actions.toggle)
		return
	if event.key == 'pageup':
		step(env.actions.pickup)
		return
	if event.key == 'pagedown':
		step(env.actions.drop)
		return

	if event.key == 'enter':
		step(env.actions.done)
		return

def redraw(img):
	if not args.agent_view:
		img = env.render('rgb_array', tile_size=args.tile_size)

	window.show_img(img)

def reset():
	if args.seed != -1:
		env.seed(args.seed)

	obs = env.reset()

	# if hasattr(env, 'mission'):
	# 	print('Mission: %s' % env.mission)
	# 	window.set_caption(env.mission)

	# redraw(obs)
	return getStateID(obs)

def getStateID(obs):
	state_space = np.zeros((grid_size, grid_size))
	x = obs['x'] 	
	y = obs['y']
	state_space[x,y] = 1
	state_space_vec = state_space.reshape(-1)
	return np.nonzero(state_space_vec)

def step(action):
	obs, reward, done, info = env.step(action)
	obs = getStateID(obs)
	
	return obs, reward, done, info

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


	env = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	txt_logger.info("Environments loaded\n")

	status = {"num_steps": 0, "update": 0, "num_episodes":0}
	txt_logger.info("Training status loaded\n")

	rng =  np.random.default_rng(args.seed)

	start_time = time.time()
	totalReturns = []
	returnPerEpisode = [] 
	stepsPerEpisode = []
	steps_done = 0
	epside_count = 0

	save_dir_Q = args.save_dir_Q

	env1 = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	env2 = GridWorldEnv(size=grid_size, goal_pos=(grid_size-1,grid_size-1))

	steps_to_first_reward = np.zeros((num_epochs))
	steps_to_good_policy = np.zeros((num_epochs))

	Q_u1 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u2 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u3 = np.zeros((grid_size*grid_size, len(env.actions)))

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
			state = getStateID(state)
			eps_reward = 0

			for time_steps in range(max_steps):
				
				eps, action = eps_greedy_action(Q_u1, state, rng, len(env.actions), eps_final)
				steps_done += 1

				next_state, reward, done, info = env.step(map_actions(action, env))
				next_state = getStateID(next_state)
				
				eps_reward += reward

				td_error = compute_td_error(state, next_state, action, reward, Q_u1)

				Q_update_u1 = Q_u1[state,action] + ((lr/C_1) * (td_error + g_1_2 * (Q_u2[state, action] - Q_u1[state,action])))
				Q_u1[state, action] = Q_update_u1

				#update Q_u2
				Q_update_u2 = Q_u2[state,action] + ((lr/C_2) * (g_1_2 * (Q_u1[state, action] - Q_u2[state,action]) + \
								g_2_3*(Q_u3[state, action] - Q_u2[state, action])))
				Q_u2[state,action] =  Q_update_u2

				#update Q_u3
				Q_update_u3 = Q_u3[state,action] + ((lr/C_3) * (g_2_3 * (Q_u2[state,action] - Q_u3[state, action])))
				Q_u3[state,action] = Q_update_u3

				state = next_state
				count += 1

				if done:
					returnPerEpisode.append(eps_reward)
					Q_u1 = np.clip(Q_u1,  a_min=0.0, a_max=1.0)
					Q_u2 = np.clip(Q_u2,  a_min=0.0, a_max=1.0)
					Q_u3 = np.clip(Q_u3,  a_min=0.0, a_max=1.0)
					stepsPerEpisode.append(time_steps)
					break

			#logging stats
			duration = int(time.time() - start_time)

			totalReturn_val = np.sum(np.array(returnPerEpisode))
			moving_avg_returns = np.mean(np.array(returnPerEpisode[-100:]))
			moving_avg_steps = np.mean(np.array(stepsPerEpisode[-20:]))

			header = ["epoch", "steps", "episode", "duration"]
			data = [epoch, steps_done, epside_count, duration]

			header += ["eps", "cur episode return", "returns", "avg returns", "avg steps", "steps to good policy"]
			data += [eps, returnPerEpisode[-1], totalReturn_val, moving_avg_returns, moving_avg_steps, steps_to_good_policy[epoch]]

			if epside_count % 200 == 0: 
				txt_logger.info(
						"Epoch {} | S {} | Episode {} | D {} | EPS {:.3f} | R {:.3f} | Total R {:.3f} | Avg R {:.3f} | Avg S {} | Good Policy {}"
						.format(*data))

			if epside_count == 0:
				csv_logger.writerow(header)
			csv_logger.writerow(data)
			csv_file.flush()

			if epside_count % 10 == 0 and args.save_np_files: 
				filename_u1 = save_dir_Q + 'Q_u1_'+ str(epside_count)+'.npy'
				filename_u2 = save_dir_Q + 'Q_u2_'+ str(epside_count)+'.npy'
				filename_u3 = save_dir_Q + 'Q_u3_'+ str(epside_count)+'.npy'
				np.save(filename_u1, Q_u1)
				np.save(filename_u2, Q_u2)
				np.save(filename_u3, Q_u3)
				
			epside_count += 1

			if moving_avg_steps <= MIN_STEPS_TRESHOLD and steps_to_good_policy[epoch] == 0 and (len(stepsPerEpisode) >= MIN_EPISODES_TRESHOLD):
				steps_to_good_policy[epoch] = count



if __name__ == "__main__":
	main()

# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)