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

# import jax
# import jax.numpy as jnp
# from jax import random



head_dir = 4
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
MAX_GOOD_POLICY_COUNT = 100


def eps_greedy_action(Q_values, state, rng, num_actions, eps_final):
	rand_val = rng.uniform()
	# eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
	eps_threshold = eps_final #as per Christos set up, using constant eps

	if rand_val > eps_threshold: 
		return eps_threshold, np.argmax(np.squeeze(Q_values[state,:]), axis=0)


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

# def getStateID(obs):
# 	state_space = jnp.zeros((grid_size, grid_size))
# 	x = obs['x']-1 	#env coordinates start from 1
# 	y = obs['y']-1	#env coordinates start from 1
# 	state_space = jax.ops.index_update(state_space, (x, y), 1)
# 	state_space_vec = state_space.reshape(-1)
# 	return jnp.nonzero(state_space_vec)

def getStateID(obs):
	state_space = np.zeros((grid_size, grid_size))
	x = obs['x']-1 	#env coordinates start from 1
	y = obs['y']-1	#env coordinates start from 1
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


# def compute_td_error(state, next_state, action, reward, Q_values):
# 	max_nextQ = jnp.max(jnp.squeeze(Q_values[next_state,:]), axis=0)
# 	td_error = reward + (discount * max_nextQ) - Q_values[state,action]
# 	return td_error

def compute_td_error(state, next_state, action, reward, Q_values):
	max_nextQ = np.max(np.squeeze(Q_values[next_state,:]), axis=0)
	td_error = reward + (discount * max_nextQ) - Q_values[state,action]
	return td_error

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--env1",
		type=str,
		help="gym environment to load",
		default='MiniGrid-ClassicGridWorldS7-v0'
	)

	parser.add_argument(
		"--env2",
		type=str,
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
		"--eps",
		type=float,
		help="epsilon-greedy value",
		default=0.3
	)

	parser.add_argument(
		"--num_episodes",
		type=int,
		help="number of episodes per epoch",
		default=20000
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
	default_model_name = f"{args.env1}_{algo}_seed{args.seed}_{date}"

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

	env = gym.make(args.env1)
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

	# env1 = gym.make('MiniGrid-ClassicGridWorldS12-v0')
	# env2 = gym.make('MiniGrid-ClassicGridWorldS12BLG-v0')

	# env1 = gym.make('MiniGrid-ClassicGridWorldS12max20k-v0')
	# env2 = gym.make('MiniGrid-ClassicGridWorldS12max20kBLG-v0')

	env1 = gym.make(args.env1)
	env2 = gym.make(args.env2)

	status = {"num_steps": 0, "update": 0, "num_episodes":0}
	txt_logger.info("Training status loaded\n")

	Q_u1 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u2 = np.zeros((grid_size*grid_size, len(env.actions)))
	Q_u3 = np.zeros((grid_size*grid_size, len(env.actions)))


	g_1_2 = args.init_tube
	g_2_3 = g_1_2 / 2

	C_1 = 1
	C_2 = 2**1
	C_3 = 2**2
	  
	# key = random.PRNGKey(args.seed)
	rng =  np.random.default_rng(args.seed)

	start_time = time.time()
	totalReturns = []
	returnPerEpisode = [] 
	steps_done = 0
	epside_count = 0

	steps_to_first_reward = np.zeros((num_epochs))
	steps_to_good_policy = np.zeros((num_epochs))

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

			etrace = np.zeros((grid_size*grid_size, len(env.actions))) 

			for time_steps in range(max_steps):

				eps, action = eps_greedy_action(Q_u1, state, rng, len(env.actions), eps_final)
				steps_done += 1
				# next_state, reward, done, info = step(map_actions(action))

				next_state, reward, done, info = env.step(map_actions(action, env))
				next_state = getStateID(next_state)
				
				eps_reward += reward

				#update eligibility trace
				etrace = np.multiply(etrace,discount * lam_factor)
				etrace[state, action] = 1 

				#update Q_u1 using td error
				td_error = compute_td_error(state, next_state, action, reward, Q_u1)
				td_error_etrace = np.multiply(etrace, td_error)
				# Q_update_u1 = Q_u1[state,action] + ((lr/C_1) * (td_error + g_1_2 * (Q_u2[state, action] - Q_u1[state,action])))
				Q_u1 = Q_u1 + ((lr/C_1) * (td_error_etrace + np.multiply((Q_u2 - Q_u1), g_1_2)))
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
					Q_u1 = np.clip(Q_u1, a_min=0, a_max=1.0)
					Q_u2 = np.clip(Q_u2, a_min=0, a_max=1.0)
					Q_u3 = np.clip(Q_u3, a_min=0, a_max=1.0)
					break

			cumulative_reward += returnPerEpisode[-1]


			#logging stats
			duration = int(time.time() - start_time)

			totalReturn_val = np.sum(np.array(returnPerEpisode))
			moving_avg_returns = np.mean(np.array(returnPerEpisode[-10:]))
	 
			if moving_avg_returns >= 1.0 and steps_to_good_policy[epoch]==0.0 and len(returnPerEpisode) >= MAX_GOOD_POLICY_COUNT:
				steps_to_good_policy[epoch] = count
			
			elif moving_avg_returns >= 1.0 and steps_to_good_policy[epoch]>0:
				good_policy_count += 1

			if returnPerEpisode[-1] >= 1.0 and steps_to_first_reward[epoch]==0.0:
				steps_to_first_reward[epoch] = count

			header = ["epoch", "steps", "episode", "duration"]
			data = [epoch, steps_done, epside_count, duration]

			header += ["eps", "cur episode return", "returns", "avg returns", "steps first R", "steps good policy", "cum R"]
			data += [eps, returnPerEpisode[-1], totalReturn_val, moving_avg_returns, steps_to_first_reward[epoch], steps_to_good_policy[epoch], cumulative_reward]

			if epside_count % 200 == 0: 
				txt_logger.info(
						"Epoch {} | S {} | Episode {} | D {} | EPS {:.3f} | R {:.3f} | Total R {:.3f} | Avg R {:.3f} | Steps 1st R {}| Steps good P {} | Cum R {}"
						.format(*data))


			if epside_count == 0:
				csv_logger.writerow(header)
			csv_logger.writerow(data)
			csv_file.flush()

			if epside_count %5 == 0: 
				filename_u1 = 'Q_etraceReplace_u1_'+str(epside_count)+'.npy'
				filename_u2 = 'Q_etraceReplace_u2_'+str(epside_count)+'.npy'
				filename_u3 = 'Q_etraceReplace_u3_'+str(epside_count)+'.npy'
				np.save(filename_u1, Q_u1)
				np.save(filename_u2, Q_u2)
				np.save(filename_u3, Q_u3)

			epside_count += 1

			if good_policy_count == MAX_GOOD_POLICY_COUNT:
				break
		

if __name__ == "__main__":
	main()

# window.reg_key_handler(key_handler)




# Blocking event loop
# window.show(block=True)