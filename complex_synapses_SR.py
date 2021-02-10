#!/usr/bin/env python3

import time
import argparse
import numpy as np
import math

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

import time
import datetime
import utils
import sys

import jax
import jax.numpy as jnp
from jax import random



# head_dir = 4
# grid_size = 5
# eps_final = 0.3
# max_steps = 20	#was 1000 
# num_episodes = 50 #was 500
# num_epochs = 6
# discount = 0.9
# lr = 0.1

grid_size = 5
eps_final = 0.3
max_steps = 1000
num_episodes = 500
num_epochs = 10
discount = 0.9
lr = 0.1


EPS_START = 1.0
EPS_END = 0.05
# EPS_DECAY = max_steps*num_episodes
EPS_DECAY = max_steps


def eps_greedy_action(Q_values, state, subkey, num_actions):
	global steps_done
	rand_val = random.uniform(subkey)
	# eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
	eps_threshold = eps_final #as per Christos set up, using constant eps
	steps_done += 1

	if rand_val > eps_threshold: 
		return eps_threshold, jnp.argmax(jnp.squeeze(Q_values[state,:]), axis=0)

	else:
		return eps_threshold, random.randint(subkey, (1,), 0, num_actions)[0]

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

	# if hasattr(env, 'mission'):
	# 	print('Mission: %s' % env.mission)
	# 	window.set_caption(env.mission)

	# redraw(obs)
	return getStateID(obs)

# def getStateID(obs):
# 	state_space = jnp.zeros((grid_size, grid_size, head_dir))
# 	hd = obs['head_direction']
# 	x = obs['x']-1 	#env coordinates start from 1
# 	y = obs['y']-1	#env coordinates start from 1
# 	state_space = jax.ops.index_update(state_space, (x, y, hd), 1)
# 	state_space_vec = state_space.reshape(-1)
# 	return jnp.nonzero(state_space_vec)

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
	max_nextQ = jnp.max(jnp.squeeze(Q_values[next_state,:]), axis=0)
	td_error = reward + (discount * max_nextQ) - Q_values[state,action]
	return td_error

def compute_SR_td_error(state, next_state, SR_values):
	# max_nextQ = jnp.max(jnp.squeeze(Q_values[next_state,:]), axis=0)
	# td_error = reward + (discount * max_nextQ) - Q_values[state,action]
	# return td_error
	ind_func = jnp.zeros((grid_size*grid_size))
	ind_func = jax.ops.index_update(ind_func, next_state, 1)	# ind_func[next_state] = 1 
	M_st = SR_values[state,:]
	M_st1 = SR_values[next_state,:]

	sr_error = ind_func + (discount * M_st1) - M_st
	return sr_error




parser = argparse.ArgumentParser()
parser.add_argument(
	"--env",
	help="gym environment to load",
	default='MiniGrid-ClassicGridWorldS7-v0'
)
parser.add_argument(
	"--seed",
	type=int,
	help="random seed to generate the environment with",
	default=0
)
parser.add_argument(
	"--tile_size",
	type=int,
	help="size at which to render tiles",
	default=32
)
parser.add_argument(
	'--agent_view',
	default=False,
	help="draw the agent sees (partially observable view)",
	action='store_true'
)

parser.add_argument(
	"--init_tube",
	type=float,
	help="value for g_1_2",
	default=0.00001
)

args = parser.parse_args()

algo = 'Benna-Fusi_model_Q-learning'

#create train dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{algo}_seed{args.seed}_{date}"

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

env = gym.make(args.env)
txt_logger.info("Environments loaded\n")

env = gym.make('MiniGrid-ClassicGridWorldS7WOGoal-v0')

status = {"num_steps": 0, "update": 0, "num_episodes":0}
txt_logger.info("Training status loaded\n")

# Q_u1 = jnp.zeros((grid_size*grid_size, len(env.actions)))
# Q_u2 = jnp.zeros((grid_size*grid_size, len(env.actions)))
# Q_u3 = jnp.zeros((grid_size*grid_size, len(env.actions)))

SR_u1 = jnp.zeros((grid_size*grid_size, grid_size*grid_size))
SR_u2 = jnp.zeros((grid_size*grid_size, grid_size*grid_size))
SR_u3 = jnp.zeros((grid_size*grid_size, grid_size*grid_size))


g_1_2 = args.init_tube #original was 0.00001
g_2_3 = g_1_2 / 2

C_1 = 1
C_2 = 2**1
C_3 = 2**2
  
key = random.PRNGKey(args.seed)

start_time = time.time()
totalReturns = []
returnPerEpisode = [] 
steps_done = 0
epside_count = 0


for epoch in range(num_epochs):

	

	for episode in range(num_episodes):
		
		state = env.reset()
		state = getStateID(state)
		eps_reward = 0

		for time_steps in range(max_steps):
			key, subkey = random.split(key)
			
			if epoch%2==0:
				action = policy75pLEFT(subkey, len(env.actions))
			
			else:
				action = policy75pRIGHT(subkey, len(env.actions))

			next_state, reward, done, info = env.step(map_actions(action))
			next_state = getStateID(next_state)
			
			eps_reward += reward

			#update Q_u1 using td error
			# td_error = compute_td_error(state, next_state, action, reward, Q_u1)
			# Q_update_u1 = Q_u1[state,action] + ((lr/C_1) * (td_error + g_1_2 * (Q_u2[state, action] - Q_u1[state,action])))
			# Q_u1 = jax.ops.index_update(Q_u1, (state, action), Q_update_u1)

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


			#update Q_u2
			# Q_update_u2 = Q_u2[state,action] + ((lr/C_2) * (g_1_2 * (Q_u1[state, action] - Q_u2[state,action]) + \
			# 				g_2_3*(Q_u3[state, action] - Q_u2[state, action])))
			# Q_u2 = jax.ops.index_update(Q_u2, (state, action), Q_update_u2)

			#update Q_u3
			# Q_update_u3 = Q_u3[state,action] + ((lr/C_3) * (g_2_3 * (Q_u3[state, action] - Q_u2[state,action])))
			# Q_u3 = jax.ops.index_update(Q_u3, (state, action), Q_update_u3)


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