#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
from os import walk


x = list(range(0,3))
y = list(range(0,3))
# print(x)
# print(y)


'''
Action indices:
0: env.actions.left,
1: env.actions.right,
2: env.actions.up,
3: env.actions.down,
'''

grid_size = 10
center = {'x':5, 'y':5}
center_left = {'x':5, 'y':4}
center_right = {'x':5, 'y':6}
center_up = {'x':4, 'y':5}
center_down = {'x':6, 'y':5}

def getStateID(obs):
	state_space = np.zeros((grid_size, grid_size))
	x = obs['x'] 
	y = obs['y']
	state_space[x,y] = 1
	state_space_vec = state_space.reshape(-1)
	return np.nonzero(state_space_vec)

X, Y= np.meshgrid(np.arange(0, 3), np.arange(0, 3))
# X, Y = np.meshgrid(x, y)
folder_dir = '/Users/raymondchua/Documents/gym-minigrid/fastRL_Q_npy/lrSF_v1/'

# print(filenames)

for i in range(0,24000):
	# file = folder_dir+'Q_u3_'+str(i)+'.npy'
	# temp = jnp.load(file)
	# temp = np.max(temp, axis=1)
	# temp = temp.reshape((5,5))

	# if i > 480:
	# 	break

	fig = plt.figure(figsize=(15.0, 4.8))
	fig.tight_layout()

	id_center = getStateID(center)
	id_left = getStateID(center_left)
	id_right = getStateID(center_right)
	id_up = getStateID(center_up)
	id_down = getStateID(center_down)

	template = np.zeros((3,3))


	file_index = str(i)
	file_index_pad = file_index.zfill(7)
	file_u1 = folder_dir+'fastRL_Q_u1_'+ file_index_pad+'.npy'
	temp_u1 = np.load(file_u1)
	# print(temp_u1.shape)
	# occ_center_u1 = temp_u1[id_center,:]
	# occ_center_u1 = np.squeeze(occ_center_u1)



	#plot Q_left
	Q_u1_act_left = temp_u1[id_center,0]
	Q_u1_act_right = temp_u1[id_center,1]
	Q_u1_act_up = temp_u1[id_center,2]
	Q_u1_act_down = temp_u1[id_center,3]
	# print(Q_u1_act_left.shape)
	# Q_u1_left_reshape = Q_u1_act_left.reshape((10,10))
	template[1,0] = Q_u1_act_left
	template[1,2] = Q_u1_act_right
	template[2,1] = Q_u1_act_up
	template[0,1] = Q_u1_act_down

	# ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax = fig.add_subplot(1, 1, 1)
	
	# ax.plot_surface(X, Y, template, rstride=1, cstride=1,
	#                 cmap='viridis', edgecolor='none')

	cf1 = ax.pcolormesh(X, Y, template, shading='auto', vmin=0, vmax=1, edgecolors='k', linewidths=2)
	# ax.pcolor(Z, edgecolors='k', linewidths=4)

	fig.colorbar(cf1, ax=ax)


	ax.set_title('Q_values of center cell')
	# ax.view_init(elev=23, azim=-26)
	# ax.set(zlim=(-2, 2))
	# ax.set_xticks([0,3])
	# ax.set_yticks([0,3])
	# ax.set_zticks([-50,0,50])
	# ax.set_xticks([], [])
	# ax.set_yticks([], [])
	plt.axis('off')
	
	fig.suptitle('Step: ' + str(i), fontsize=16)


	# plt.show()

	# break
	file_index = str(i)
	file_index_pad = file_index.zfill(7)
	plot_filename = folder_dir+'/png/'+'lrSF_v2_'+file_index_pad
	plt.savefig(plot_filename)
	plt.close()
	# if i!=4990: 
	# 	plt.close()

	if i %1000 == 0:
		print('Current: ', str(i))

# plt.show()
print('done!')



# fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot_surface(X, Y, Z, 50, cmap='binary')
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')

# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, temp, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('Q-learning')
# ax.set(zlim=(0, 3))
# ax.view_init(elev=20, azim=-26)

# plt.show()
