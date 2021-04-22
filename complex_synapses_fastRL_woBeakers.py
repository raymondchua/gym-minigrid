#!/usr/bin/env python3

import time
import argparse
import numpy as np
import math
import os

import datetime
import utils
import sys

from gridworld_np import GridWorldEnv


grid_size = 10

max_steps = 20000

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

# def policyLEFT():
# 	return 0

# def policyRandomAction(rng, num_actions):
# 	return ng.integers(low=0, high=num_actions, size=1)[0]

# def policy75pLEFT(rng, num_actions):
# 	rand_val = rng.uniform()
# 	threshold = 0.75

# 	if rand_val > threshold:
# 		return rng.integers(low=0, high=num_actions, size=1)[0]

# 	else: 
# 		return 0

# def policy75pRIGHT(rng, num_actions):
# 	rand_val = rng.uniform()
# 	threshold = 0.75

# 	if rand_val > threshold:
# 		return rng.integers(low=0, high=num_actions, size=1)[0]

# 	else: 
# 		return 1


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
		step(env.actions.forward)
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

def compute_SR_td_error(state, next_state, SR_values):
	ind_func = np.zeros((grid_size*grid_size))
	ind_func[next_state] = 1 
	M_st = SR_values[state,:]
	M_st_next = SR_values[next_state,:]

	sr_error = ind_func + (discount * M_st_next) - M_st
	return sr_error

def compute_SF_td_error(state, action, next_state, next_action, SF_values, w):
	ind_func = np.zeros((grid_size*grid_size))
	ind_func[next_state] = 1 
	SF_st = SF_values[state,action, :]
	SF_next = SF_values[next_state,next_action,:]

	sf_error = ind_func + (discount * SF_next) - SF_st
	return sf_error


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument(
			"--seed",
			type=int,
			help="random seed to generate the environment with",
			default=0
	)

	parser.add_argument(
		"--lr_Q",
		type=float,
		help="learning rate for Q values",
		default=0.1
	)

	parser.add_argument(
		"--lr_SF",
		type=float,
		help="learning rate for successor features",
		default=0.1
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
		"--num_epochs",
		type=int,
		help="number of epochs",
		default=12
	)

	parser.add_argument(
		"--init_tube",
		type=float,
		help="value for g_1_2",
		default=0.00001
	)

	parser.add_argument(
		"--do_not_learn_Q",
		action='store_true',
		help="Use when not to learn Q function or not",
	)

	parser.add_argument(
		"--save_np_files",
		action='store_true',
		help="Use when we want to save the numpy files of SF and/or Q values",
	)

	parser.add_argument(
		"--save_SR",
		action='store_true',
		help="Use when we want to save the numpy files of SR",
	)

	parser.add_argument(
		"--lambda_factor",
		type=float,
		help="lambda factor for eligibility trace",
		default=0.9
	)

	parser.add_argument(
		"--etrace",
		action='store_true',
		help="Use replacing eligibility trace",
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

	eps_final = args.eps
	num_episodes = args.num_episodes

	sf_lr = args.lr_SF
	q_lr = args.lr_Q
	do_not_learn_Q = args.do_not_learn_Q
	use_etrace = args.etrace

	lam_factor = args.lambda_factor
	num_epochs = args.num_epochs

	# Load loggers and Tensorboard writer

	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)

	# Log command and all script arguments

	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))

	# Set seed for all randomness sources
	utils.seed(args.seed)

	SF_path = utils.get_SF_dir(model_dir)
	SR_path = utils.get_SR_dir(model_dir)
	Q_path = utils.get_Q_dir(model_dir)

	env = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	txt_logger.info("Environments loaded\n")

	env1 = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	env2 = GridWorldEnv(size=grid_size, goal_pos=(grid_size-1,grid_size-1))

	rng =  np.random.default_rng(args.seed)

	status = {"num_steps": 0, "update": 0, "num_episodes":0}
	txt_logger.info("Training status loaded\n")

	Q_u1 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u2 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u3 = np.zeros((grid_size*grid_size, len(env.actions)))

	SR_u1 = np.zeros((grid_size*grid_size, grid_size*grid_size))
	SR_u2 = np.zeros((grid_size*grid_size, grid_size*grid_size))
	SR_u3 = np.zeros((grid_size*grid_size, grid_size*grid_size))

	SF_u1 = np.zeros((grid_size*grid_size, len(env.actions), grid_size*grid_size))
	SF_u2 = np.zeros((grid_size*grid_size, len(env.actions), grid_size*grid_size))
	SF_u3 = np.zeros((grid_size*grid_size, len(env.actions), grid_size*grid_size))

	g_1_2 = args.init_tube #original was 0.00001
	g_2_3 = g_1_2 / 2

	C_1 = 1
	C_2 = 2**1
	C_3 = 2**2
	  
	rng =  np.random.default_rng(args.seed)

	start_time = time.time()
	totalReturns = []
	returnPerEpisode = [] 
	stepsPerEpisode = []
	steps_done = 0
	episode_count = 0

	steps_to_first_reward = np.zeros((num_epochs))
	steps_to_good_policy = np.zeros((num_epochs))

	cumulative_reward = 0

	w_1 = np.zeros((grid_size*grid_size))
	w_2 = np.zeros((grid_size*grid_size))
	
	w_1[getStateID({'x':0,'y':0})] = 1
	w_2[getStateID({'x':grid_size-1, 'y': grid_size-1})] = 1

	episode_saved_counter = 0

	action_left_counter = 0
	action_right_counter = 0
	action_up_counter = 0
	action_down_counter = 0

	action_left_episodes = []
	action_right_episodes = []
	action_up_episodes = []
	action_down_episodes = []

	start_positions = []

	for epoch in range(num_epochs):

		count = 0

		returnPerEpisode = [] 
		stepsPerEpisode = []

		good_policy_count = 0

		if epoch%2==0:
			env = env1
			w = w_1
		else:
			env = env2
			w = w_2

		for episode in range(num_episodes):
			
			state = env.reset()

			start_positions.append(state)


			state = getStateID(state)
			eps_reward = 0

			action_left_counter = 0
			action_right_counter = 0
			action_up_counter = 0
			action_down_counter = 0

			eps, action = eps_greedy_action(Q_u1, state, rng, len(env.actions), eps_final)

			etrace = np.zeros((grid_size*grid_size, len(env.actions))) 

			for time_steps in range(max_steps):
				
				steps_done += 1

				if action == 0:
					action_left_counter += 1

				elif action == 1:
					action_right_counter += 1

				elif action == 2:
					action_up_counter += 1

				elif action == 3:
					action_down_counter += 1  

				next_state, reward, done, info = env.step(map_actions(action, env))
				next_state = getStateID(next_state)

				eps, next_action = eps_greedy_action(Q_u1, next_state, rng, len(env.actions), eps_final)
				
				eps_reward += reward

				#update SR_u1 using SR-td error
				sf_error = compute_SF_td_error(state, action, next_state, next_action, SF_u1, w)
				SF_u1[state,action,:] +=  (sf_lr * sf_error)

				Q_u1_update = np.squeeze(np.dot(SF_u1[state,action,:], w))

				SR_u1[state, next_state] += 1

				if do_not_learn_Q:
					Q_u1[state,action] = Q_u1_update
					

				else:
					Q_u1[state,action] = ((1-q_lr) * Q_u1[state,action]) + (q_lr * Q_u1_update)
					


				if use_etrace:
					etrace = np.multiply(etrace,discount * lam_factor)
					etrace[state,action] = 1
					Q_u1 = np.multiply(etrace, Q_u1)


				state = next_state
				action = next_action

				count += 1

				if done:
					returnPerEpisode.append(eps_reward)
					Q_u1 = np.clip(Q_u1, a_min=0, a_max=1.0)
					stepsPerEpisode.append(time_steps)
					action_left_episodes.append(action_left_counter)
					action_right_episodes.append(action_right_counter)
					action_up_episodes.append(action_up_counter)
					action_down_episodes.append(action_down_counter)
					break


			#logging stats
			duration = int(time.time() - start_time)

			totalReturn_val = np.sum(np.array(returnPerEpisode))
			moving_avg_returns = np.mean(np.array(returnPerEpisode[-10:]))
			moving_avg_steps = np.mean(np.array(stepsPerEpisode[-MIN_EPISODES_TRESHOLD:]))



			header = ["epoch", "steps", "episode", "duration"]
			data = [epoch, steps_done, episode_count, duration]

			header += ["eps", "cur episode return", "returns", "avg returns", "avg steps", "steps to good policy"]
			data += [eps, returnPerEpisode[-1], totalReturn_val, moving_avg_returns, moving_avg_steps, steps_to_good_policy[epoch]]

			header += ["Left", "Right", "Up", "Down", "Start P"]
			data += [np.sum(np.array(action_left_episodes[episode_count-200:episode_count])), np.sum(np.array(action_right_episodes[episode_count-200:episode_count])), np.sum(np.array(action_up_episodes[episode_count-200:episode_count])), np.sum(np.array(action_down_episodes[episode_count-200:episode_count])), start_positions[episode_count]]

			if episode_count % 200 == 0: 
				txt_logger.info(
						"Epoch {} | S {} | Episode {} | D {} | EPS {:.3f} | R {:.3f} | Total R {:.3f} | Avg R {:.3f} | Avg S {} | Good Policy {} | Left {} | Right {} | Up {} | Down {} | Start P {}" 
						.format(*data))


			if episode_count == 0:
				csv_logger.writerow(header)
			csv_logger.writerow(data)
			csv_file.flush()

			if episode_count %5 == 0: 
				file_index = str(episode_saved_counter)
				file_index_pad = file_index.zfill(7)

				if args.save_np_files: 
					filename_SF_u1 = 'fastRL_SF_u1_'+file_index_pad+'.npy'
					filename_Q_u1 = 'fastRL_Q_u1_'+file_index_pad+'.npy'
					
					filepath_SF_u1 = os.path.join(SF_path, filename_SF_u1)
					utils.create_folders_if_necessary(filepath_SF_u1)

					filepath_Q_u1 = os.path.join(Q_path, filename_Q_u1)
					utils.create_folders_if_necessary(filepath_Q_u1)

					np.save(filepath_SF_u1, SF_u1)
					np.save(filepath_Q_u1, Q_u1)

				if args.save_SR:
					filename_SR_u1 = 'fastRL_SR_u1_'+file_index_pad+'.npy'
					filepath_SR_u1 = os.path.join(SR_path, filename_SR_u1)
					utils.create_folders_if_necessary(filepath_SR_u1)
					np.save(filepath_SR_u1, SR_u1)

				episode_saved_counter+= 1
				# np.save(filename_u2, Q_u2)
				# np.save(filename_u3, Q_u3)

			episode_count += 1

			if moving_avg_steps <= MIN_STEPS_TRESHOLD and steps_to_good_policy[epoch] == 0 and (len(stepsPerEpisode) >= MIN_EPISODES_TRESHOLD):
				steps_to_good_policy[epoch] = count


if __name__ == "__main__":
	main()


# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)