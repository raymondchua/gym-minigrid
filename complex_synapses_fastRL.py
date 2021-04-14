#!/usr/bin/env python3

import time
import argparse
import numpy as np
import math

import time
import datetime
import utils
import sys

from gridworld_np import GridWorldEnv


grid_size = 10

max_steps = 20000
num_epochs = 12
discount = 0.9
lam_factor = 0.9


EPS_START = 1.0
EPS_END = 0.05


MIN_STEPS_TRESHOLD = 13


def eps_greedy_action(Q_values, state, rng, num_actions, eps_final):
	rand_val = rng.uniform()
	# eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
	eps_threshold = eps_final #as per Christos set up, using constant eps

	if rand_val > eps_threshold: 
		return eps_threshold, np.argmax(np.squeeze(Q_values[state,:]), axis=0)


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

# def compute_SF_td_error(state, action, next_state, SF_values, w):
# 	ind_func = np.zeros((grid_size*grid_size))
# 	ind_func[next_state] = 1 
# 	SF_st = SF_values[state,action, :]
# 	next_action = SF_values 
# 	SF_next = SR_values[next_state,:]

# 	sr_error = ind_func + (discount * M_st_next) - M_st
# 	return sr_error

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
		"--eps",
		type=float,
		help="epsilon-greedy value",
		default=0.05
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
		"--num_episodes",
		type=int,
		help="number of episodes per epoch",
		default=10000
	)

	parser.add_argument(
		"--init_tube",
		type=float,
		help="value for g_1_2",
		default=0.00001
	)

	args = parser.parse_args()

	algo = 'Benna-Fusi_fastRL'

	#create train dir
	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	default_model_name = f"{algo}_seed{args.seed}_{date}"

	model_name = default_model_name
	model_dir = utils.get_model_dir(model_name)

	eps_final = args.eps
	num_episodes = args.num_episodes

	# Load loggers and Tensorboard writer

	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)

	# Log command and all script arguments

	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))

	# Set seed for all randomness sources
	utils.seed(args.seed)

	env = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	txt_logger.info("Environments loaded\n")

	env1 = GridWorldEnv(size=grid_size, goal_pos=(0,0))
	env2 = GridWorldEnv(size=grid_size, goal_pos=(grid_size-1,grid_size-1))

	rng =  np.random.default_rng(args.seed)

	sf_lr = args.lr_SF
	q_lr = args.lr_Q

	status = {"num_steps": 0, "update": 0, "num_episodes":0}
	txt_logger.info("Training status loaded\n")

	Q_u1 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u2 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u3 = np.zeros((grid_size*grid_size, len(env.actions)))

	# SR_u1 = np.zeros((grid_size*grid_size, grid_size*grid_size))
	# SR_u2 = np.zeros((grid_size*grid_size, grid_size*grid_size))
	# SR_u3 = np.zeros((grid_size*grid_size, grid_size*grid_size))

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
	epside_count = 0

	steps_to_first_reward = np.zeros((num_epochs))
	steps_to_good_policy = np.zeros((num_epochs))

	w_1 = np.zeros((grid_size*grid_size))
	w_2 = np.zeros((grid_size*grid_size))
	
	w_1[getStateID({'x':0,'y':0})] = 1
	w_2[getStateID({'x':grid_size-1, 'y': grid_size-1})] = 1

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
			
			state = env.reset_center()
			state = getStateID(state)
			eps_reward = 0

			eps, action = eps_greedy_action(Q_u1, state, rng, len(env.actions), eps_final)

			for time_steps in range(max_steps):
				

				steps_done += 1

				# next_state, reward, done, info = env.step(map_actions(action, env))
				# next_state = getStateID(next_state)

				# eps, next_action = eps_greedy_action(Q_u1, next_state, rng, len(env.actions), eps_final)

				# feature_vec = np.zeros((grid_size*grid_size))
				# feature_vec[next_state] = 1
				
				# eps_reward += reward

				# #update SR_u1 using SR-td error
				# sf_error = compute_SF_td_error(state, next_state, SR_u1)
				# SR_update_u1 = SR_u1[state,:] + ((lr/C_1) * (sf_error + g_1_2 * (SR_u2[state,:] - SR_u1[state,:])))
				# # SR_u1 = jax.ops.index_update(SR_u1, jax.ops.index[state, :], SR_update_u1)
				# SR_u1[state,:] = SR_update_u1


				# #update SR_u2 using SR-td error
				# SR_update_u2 = SR_u2[state,:] + ((lr/C_2) * (g_1_2 * (SR_u1[state, :] - SR_u2[state,:]) + \
				# 				g_2_3*(SR_u3[state, :] - SR_u2[state, :])))
				# # SR_u2 = jax.ops.index_update(SR_u2, jax.ops.index[state, :], SR_update_u2)
				# SR_u2[state,:] = SR_update_u2


				# #update SR_u3 using SR-td error
				# SR_update_u3 = SR_u3[state,:] + ((lr/C_3) * (g_2_3 * (SR_u2[state, :] - SR_u3[state,:])))
				# # SR_u3 = jax.ops.index_update(SR_u3, jax.ops.index[state, :], SR_update_u3)
				# SR_u3[state,:] = SR_update_u3

				# state = next_state

				# Q_u1_update = np.
				# Q_u1 = (1-q_lr) * Q_u1 + (q_lr * )


				next_state, reward, done, info = env.step(map_actions(action, env))
				next_state = getStateID(next_state)

				eps, next_action = eps_greedy_action(Q_u1, next_state, rng, len(env.actions), eps_final)

				feature_vec = np.zeros((grid_size*grid_size))
				feature_vec[next_state] = 1
				
				eps_reward += reward

				# #update SR_u1 using SR-td error
				# sf_error = compute_SF_td_error(state, action, next_state, next_action, SF_u1, w)
				# SF_u1[state,action,:] = (1-sf_lr) * SF_u1[state,action,:] + (sf_lr * sf_error)

				# #update SF_u1 using SR-td error
				sf_error = compute_SF_td_error(state, action, next_state, next_action, SF_u1, w)
				SF_u1[state,:] = SF_u1[state,:] + ((sf_lr/C_1) * (sf_error + g_1_2 * (SF_u2[state,:] - SF_u1[state,:])))


				# #update SF_u2 using SR-td error
				SF_u2[state,:] = SF_u2[state,:] + ((sf_lr/C_2) * (g_1_2 * (SF_u1[state, :] - SF_u2[state,:]) + \
								g_2_3*(SF_u3[state, :] - SF_u2[state, :])))


				#update SF_u3 using SR-td error
				SF_u3[state,:] = SF_u3[state,:] + ((sf_lr/C_3) * (g_2_3 * (SF_u2[state, :] - SF_u3[state,:])))

				Q_u1_update = np.squeeze(np.dot(SF_u1[state,action,:], w))
				Q_u1[state,action] = ((1-q_lr) * Q_u1[state,action]) + (q_lr * Q_u1_update)
				
				state = next_state
				action = next_action

				count += 1

				if done:
					returnPerEpisode.append(eps_reward)
					Q_u1 = np.clip(Q_u1, a_min=0, a_max=1.0)
					Q_u2 = np.clip(Q_u2, a_min=0, a_max=1.0)
					Q_u3 = np.clip(Q_u3, a_min=0, a_max=1.0)
					stepsPerEpisode.append(time_steps)
					break



			#logging stats
			duration = int(time.time() - start_time)

			totalReturn_val = np.sum(np.array(returnPerEpisode))
			moving_avg_returns = np.mean(np.array(returnPerEpisode[-10:]))
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

			# if epside_count %10 == 0: 
			# 	filename_u1 = 'SR_u1_'+str(epside_count)+'.npy'
			# 	filename_u2 = 'SR_u2_'+str(epside_count)+'.npy'
			# 	filename_u3 = 'SR_u3_'+str(epside_count)+'.npy'
			# 	np.save(filename_u1, SR_u1)
			# 	np.save(filename_u2, SR_u2)
			# 	np.save(filename_u3, SR_u3)

			epside_count += 1

			if moving_avg_steps <= MIN_STEPS_TRESHOLD and steps_to_good_policy[epoch] == 0 and (len(stepsPerEpisode) >= 20):
				steps_to_good_policy[epoch] = count

if __name__ == "__main__":
	main()


# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)