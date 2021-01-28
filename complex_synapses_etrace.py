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



head_dir = 4
grid_size = 10
eps_final = 0.3
max_steps = 1000
num_episodes = 20000
num_epochs = 8
discount = 0.9
lr = 0.1
lam_factor = 0.9


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

# if args.agent_view:
# 	env = RGBImgPartialObsWrapper(env)
# 	env = ImgObsWrapper(env)

# window = Window('gym_minigrid - ' + args.env)
# reset()

# env1 = gym.make('MiniGrid-ClassicGridWorldS7-v0')
# env2 = gym.make('MiniGrid-ClassicGridWorldS7BLG-v0')

# env1 = gym.make('MiniGrid-ClassicGridWorldS9-v0')
# env2 = gym.make('MiniGrid-ClassicGridWorldS9BLG-v0')

env1 = gym.make('MiniGrid-ClassicGridWorldS12-v0')
env2 = gym.make('MiniGrid-ClassicGridWorldS12BLG-v0')

status = {"num_steps": 0, "update": 0, "num_episodes":0}
txt_logger.info("Training status loaded\n")

Q_u1 = jnp.zeros((grid_size*grid_size, len(env.actions)))
Q_u2 = jnp.zeros((grid_size*grid_size, len(env.actions)))
Q_u3 = jnp.zeros((grid_size*grid_size, len(env.actions)))


g_1_2 = 0.01 #original was 0.00001, second version was 0.001, third version was 0.001
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

steps_to_first_reward = jnp.zeros((num_epochs))
steps_to_good_policy = jnp.zeros((num_epochs))

cumulative_reward = 0



for epoch in range(num_epochs):

	if epoch%2==0:
		env = env1
	else:
		env = env2

	count = 0

	returnPerEpisode = [] 

	good_policy_count = 0

	for episode in range(num_episodes):
		
		state = env.reset()
		state = getStateID(state)
		eps_reward = 0

		etrace = jnp.zeros((grid_size*grid_size, len(env.actions))) 

		for time_steps in range(max_steps):
			key, subkey = random.split(key)
			eps, action = eps_greedy_action(Q_u1, state, subkey, len(env.actions))
			# next_state, reward, done, info = step(map_actions(action))

			next_state, reward, done, info = env.step(map_actions(action))
			next_state = getStateID(next_state)
			
			eps_reward += reward

			#update eligibility trace
			etrace = jnp.multiply(etrace,discount * lam_factor)
			etrace = jax.ops.index_update(etrace, (state, action), etrace[state, action]+1)

			#update Q_u1 using td error
			td_error = compute_td_error(state, next_state, action, reward, Q_u1)
			td_error_etrace = jnp.multiply(etrace, td_error)
			# Q_update_u1 = Q_u1[state,action] + ((lr/C_1) * (td_error + g_1_2 * (Q_u2[state, action] - Q_u1[state,action])))
			Q_u1 = Q_u1 + ((lr/C_1) * (td_error + jnp.multiply((Q_u2 - Q_u1), g_1_2)))
			# Q_u1 = jax.ops.index_update(Q_u1, (state, action), Q_update_u1)


			#update Q_u2
			# Q_update_u2 = Q_u2[state,action] + ((lr/C_2) * (g_1_2 * (Q_u1[state, action] - Q_u2[state,action]) + \
			# 				g_2_3*(Q_u3[state, action] - Q_u2[state, action])))
			# Q_u2 = jax.ops.index_update(Q_u2, (state, action), Q_update_u2)
			Q_u2 = Q_u2 + ((lr/C_2) * (g_1_2 * (Q_u1 - Q_u2) + \
							g_2_3*(Q_u3 - Q_u2)))

			#update Q_u3
			# Q_update_u3 = Q_u3[state,action] + ((lr/C_3) * (g_2_3 * (Q_u2[state,action] - Q_u3[state, action])))
			# Q_u3 = jax.ops.index_update(Q_u3, (state, action), Q_update_u3)
			Q_u3 = Q_u3 + ((lr/C_3) * (g_2_3 * (Q_u2 - Q_u3)))

			state = next_state

			count += 1

			if done:
				returnPerEpisode.append(eps_reward)
				Q_u1 = jnp.clip(Q_u1,  a_max=1.0)
				Q_u2 = jnp.clip(Q_u2,  a_max=1.0)
				Q_u3 = jnp.clip(Q_u3,  a_max=1.0)
				break

		cumulative_reward += returnPerEpisode[-1]


		#logging stats
		duration = int(time.time() - start_time)

		totalReturn_val = jnp.sum(jnp.array(returnPerEpisode))
		moving_avg_returns = jnp.mean(jnp.array(returnPerEpisode[-10:]))
 
		if moving_avg_returns >= 1.0 and steps_to_good_policy[epoch]==0.0:
			steps_to_good_policy = jax.ops.index_update(steps_to_good_policy, (epoch), count)
			good_policy_count += 1 

		if returnPerEpisode[-1] >= 1.0 and steps_to_first_reward[epoch]==0.0:
			steps_to_first_reward = jax.ops.index_update(steps_to_first_reward, (epoch), count)

		header = ["epoch", "steps", "episode", "duration"]
		data = [epoch, steps_done, epside_count, duration]

		header += ["eps", "cur episode return", "returns", "avg returns", "steps first R", "steps good policy", "cum R"]
		data += [eps, returnPerEpisode[-1], totalReturn_val, moving_avg_returns, steps_to_first_reward[epoch], steps_to_good_policy[epoch], cumulative_reward]

		if epside_count % 1000 == 0: 
			txt_logger.info(
					"Epoch {} | S {} | Episode {} | D {} | EPS {:.3f} | R {:.3f} | Total R {:.3f} | Avg R {:.3f} | Steps 1st R {}| Steps good P {} | Cum R {}"
					.format(*data))


		if epside_count == 0:
			csv_logger.writerow(header)
		csv_logger.writerow(data)
		csv_file.flush()

		if epside_count %10 == 0: 
			filename_u1 = 'Q_etrace_u1_'+str(epside_count)+'.npy'
			filename_u2 = 'Q_etrace_u2_'+str(epside_count)+'.npy'
			filename_u3 = 'Q_etrace_u3_'+str(epside_count)+'.npy'
			jnp.save(filename_u1, Q_u1)
			jnp.save(filename_u2, Q_u2)
			jnp.save(filename_u3, Q_u3)

		epside_count += 1

		if good_policy_count == 20:
			break




# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)