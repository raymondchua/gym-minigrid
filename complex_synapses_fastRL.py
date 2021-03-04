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
lr = 0.1
lam_factor = 0.9


EPS_START = 1.0
EPS_END = 0.05
# EPS_DECAY = max_steps*num_episodes
EPS_DECAY = max_steps

MIN_STEPS_TRESHOLD = 13


def eps_greedy_action(Q_values, state, rng, num_actions, eps_final):
	rand_val = rng.uniform()
	# eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
	eps_threshold = eps_final #as per Christos set up, using constant eps

	if rand_val > eps_threshold: 
		return eps_threshold, np.argmax(np.squeeze(Q_values[state,:]), axis=0)


	else:
		return eps_threshold, rng.integers(low=0, high=num_actions, size=1)[0]

def policyLEFT():
	global steps_done
	steps_done += 1
	return 0

def policyRandomAction(subkey, num_actions):
	global steps_done
	steps_done += 1
	return random.randint(subkey, (1,), 0, num_actions)[0]

def policy75pLEFT(subkey, num_actions):
	global steps_done
	steps_done += 1
	rand_val = random.uniform(subkey)
	threshold = 0.75

	if rand_val > threshold:
		return random.randint(subkey, (1,), 0, num_actions)[0]

	else: 
		return 0

def policy75pRIGHT(subkey, num_actions):
	global steps_done
	steps_done += 1
	rand_val = random.uniform(subkey)
	threshold = 0.75

	if rand_val > threshold:
		return random.randint(subkey, (1,), 0, num_actions)[0]

	else: 
		return 1


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
	state_space = jnp.zeros((grid_size, grid_size))
	x = obs['x']-1 	#env coordinates start from 1
	y = obs['y']-1	#env coordinates start from 1
	state_space = jax.ops.index_update(state_space, (x, y), 1)
	state_space_vec = state_space.reshape(-1)
	return jnp.nonzero(state_space_vec)

def step(action):
	obs, reward, done, info = env.step(action)
	obs = getStateID(obs)
	
	return obs, reward, done, info

def map_actions(action):
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

env = GridWorldEnv(size=grid_size)

status = {"num_steps": 0, "update": 0, "num_episodes":0}
txt_logger.info("Training status loaded\n")

Q_u1 = jnp.zeros((grid_size*grid_size, len(env.actions)))
Q_u2 = jnp.zeros((grid_size*grid_size, len(env.actions)))
Q_u3 = jnp.zeros((grid_size*grid_size, len(env.actions)))

SR_u1 = np.zeros((grid_size*grid_size, grid_size*grid_size))
SR_u2 = np.zeros((grid_size*grid_size, grid_size*grid_size))
SR_u3 = np.zeros((grid_size*grid_size, grid_size*grid_size))

# env1 = GridWorldEnv(size=grid_size, goal_pos=(0,0))
# env2 = GridWorldEnv(size=grid_size, goal_pos=(grid_size-1,grid_size-1))


g_1_2 = args.init_tube #original was 0.00001
g_2_3 = g_1_2 / 2

C_1 = 1
C_2 = 2**1
C_3 = 2**2
  
rng =  np.random.default_rng(args.seed)

start_time = time.time()
totalReturns = []
returnPerEpisode = [] 
steps_done = 0
epside_count = 0


for epoch in range(num_epochs):

	

	for episode in range(num_episodes):
		
		state = env.reset_center()
		state = getStateID(state)
		eps_reward = 0

		for time_steps in range(max_steps):
			
			if epoch%2==0:
				action = policy75pLEFT(subkey, len(env.actions))
			
			else:
				action = policy75pRIGHT(subkey, len(env.actions))

			next_state, reward, done, info = env.step(map_actions(action))
			next_state = getStateID(next_state)
			
			eps_reward += reward

			#update SR_u1 using SR-td error
			sr_error = compute_SR_td_error(state, next_state, SR_u1)
			SR_update_u1 = SR_u1[state,:] + ((lr/C_1) * (sr_error + g_1_2 * (SR_u2[state,:] - SR_u1[state,:])))
			SR_u1 = jax.ops.index_update(SR_u1, jax.ops.index[state, :], SR_update_u1)


			#update SR_u2 using SR-td error
			SR_update_u2 = SR_u2[state,:] + ((lr/C_2) * (g_1_2 * (SR_u1[state, :] - SR_u2[state,:]) + \
							g_2_3*(SR_u3[state, :] - SR_u2[state, :])))
			SR_u2 = jax.ops.index_update(SR_u2, jax.ops.index[state, :], SR_update_u2)


			#update SR_u3 using SR-td error
			SR_update_u3 = SR_u3[state,:] + ((lr/C_3) * (g_2_3 * (SR_u2[state, :] - SR_u3[state,:])))
			SR_u3 = jax.ops.index_update(SR_u3, jax.ops.index[state, :], SR_update_u3)

			state = next_state

			if done:
				returnPerEpisode.append(eps_reward)
				break


		#logging stats
		duration = int(time.time() - start_time)

		totalReturn_val = jnp.sum(jnp.array(returnPerEpisode))
		moving_avg_returns = jnp.mean(jnp.array(returnPerEpisode[-10:]))

		header = ["epoch", "steps", "episode", "duration"]
		data = [epoch, steps_done, epside_count, duration]

		# header += ["eps", "cur episode return", "returns", "avg returns"]
		# data += [eps, returnPerEpisode[epside_count], totalReturn_val, moving_avg_returns]

		header += ["cur episode return", "returns", "avg returns"]
		data += [returnPerEpisode[epside_count], totalReturn_val, moving_avg_returns]

		txt_logger.info(
				"Epoch {} | S {} | Episode {} | D {} | R {:.3f} | Total R {:.3f} | Avg R {:.3f} "
				.format(*data))


		if epside_count == 0:
			csv_logger.writerow(header)
		csv_logger.writerow(data)
		csv_file.flush()

		if epside_count %10 == 0: 
			filename_u1 = 'SR_u1_'+str(epside_count)+'.npy'
			filename_u2 = 'SR_u2_'+str(epside_count)+'.npy'
			filename_u3 = 'SR_u3_'+str(epside_count)+'.npy'
			jnp.save(filename_u1, SR_u1)
			jnp.save(filename_u2, SR_u2)
			jnp.save(filename_u3, SR_u3)

		epside_count += 1




# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)