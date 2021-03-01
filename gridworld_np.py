#!/usr/bin/env python3

import numpy as np

from enum import IntEnum

class GridWorldEnv(object):

	# Enumeration of possible actions
	class Actions(IntEnum):
		# Move left, right, upward or downward
		left = 0
		right = 1
		up = 2
		down = 3

	def __init__(self, size, max_steps=20000, agent_pos=None, goal_pos=None):
		self.size = size
		self.max_steps = max_steps
		self.agent_init_pos = agent_pos
		self.goal_pos = goal_pos

		self.agent_pos = agent_pos

		if self.agent_pos == None:
			self.random_drop_agent()

		self.step_count = 0 

		# Action enumeration for this environment
		self.actions = self.Actions



	def random_drop_agent(self):
		while True:
			agent_pos_x = np.random.randint(low=0, high=self.size)
			agent_pos_y = np.random.randint(low=0, high=self.size)

			self.agent_pos = {'x':agent_pos_x, 'y':agent_pos_y}

			if self.agent_pos != self.goal_pos:
				break


	def reset(self):
		self.step_count = 0 
		self.random_drop_agent()
		return self.agent_pos


	def step(self, action):
		reward = 0
		done = False
		self.step_count += 1

		# Move left
		if action == self.actions.left:

			left_cell = {'x': self.agent_pos['x'] - 1, 'y':self.agent_pos['y']}

			#boundary-check
			if left_cell['x'] >= 0:
				self.agent_pos = left_cell

			if self.agent_pos['x'] == self.goal_pos[0] and self.agent_pos['y'] == self.goal_pos[1]:
				done = True
				reward = 1

		# Move right
		elif action == self.actions.right:
			
			right_cell = {'x': self.agent_pos['x'] + 1, 'y': self.agent_pos['y']}

			#boundary-check
			if right_cell['x'] < self.size:
				self.agent_pos = right_cell

			if self.agent_pos['x'] == self.goal_pos[0] and self.agent_pos['y'] == self.goal_pos[1]:
				done = True
				reward = 1


		# Move upwards
		elif action == self.actions.up:

			up_cell = {'x': self.agent_pos['x'], 'y':self.agent_pos['y'] + 1}

			#boundary-check
			if up_cell['y'] < self.size:
				self.agent_pos = up_cell

			if self.agent_pos['x'] == self.goal_pos[0] and self.agent_pos['y'] == self.goal_pos[1]:
				done = True
				reward = 1

		# Move downwards
		elif action == self.actions.down:

			down_cell = {'x':self.agent_pos['x'], 'y':self.agent_pos['y'] - 1}

			#boundary-down
			if down_cell['y'] >= 0:
				self.agent_pos = down_cell

			if self.agent_pos['x'] == self.goal_pos[0] and self.agent_pos['y'] == self.goal_pos[1]:
				done = True
				reward = 1

		if self.step_count >= self.max_steps:
			done = True

		obs = self.agent_pos

		return obs, reward, done, {}
 





