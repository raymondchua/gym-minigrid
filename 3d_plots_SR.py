#!/usr/bin/env python3
import csv
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from os import walk


x = list(range(0,5))
y = list(range(0,5))
# print(x)
# print(y)

gride_size = 5


X, Y = np.meshgrid(x, y)
folder_dir = '/Users/raymondchua/Documents/gym-minigrid/BF_model_SR/'

# print(filenames)

for i in range(0,5000,10):
	# file = folder_dir+'Q_u3_'+str(i)+'.npy'
	# temp = jnp.load(file)
	# temp = np.max(temp, axis=1)
	# temp = temp.reshape((5,5))

	# if i > 480:
	# 	break

	fig = plt.figure(figsize=(15.0, 4.8))
	fig.tight_layout()

	# ax = plt.axes(projection='3d')
	# ax.plot_surface(X, Y, Z, 50, cmap='binary')
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')

	#plot u_1
	file_u1 = folder_dir+'SR_u1_'+str(i)+'.npy'
	temp_u1 = jnp.load(file_u1)
	print('shape: ', temp_u1.shape)
	temp_u1 = np.max(temp_u1, axis=1)
	print('max: ', temp_u1)
	print('shape: ', temp_u1.shape)
	temp_u1 = temp_u1.reshape((5,5))

	ax = fig.add_subplot(1, 3, 1, projection='3d')
	ax.plot_surface(X, Y, temp_u1, rstride=1, cstride=1,
	                cmap='viridis', edgecolor='none')
	ax.set_title('u1')
	ax.view_init(elev=50, azim=-26)
	ax.set(zlim=(0, 1.2))
	ax.set_xticks([0,gride_size-1])
	ax.set_yticks([0,gride_size-1])
	ax.set_zticks([0,0.5,1])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Occupancy')
	
	#plot u_2
	file_u2 = folder_dir+'SR_u2_'+str(i)+'.npy'
	temp_u2 = jnp.load(file_u2)
	temp_u2 = np.max(temp_u2, axis=1)
	temp_u2 = temp_u2.reshape((5,5))

	ax = fig.add_subplot(1, 3, 2, projection='3d')
	ax.plot_surface(X, Y, temp_u2, rstride=1, cstride=1,
	                cmap='viridis', edgecolor='none')
	ax.set_title('u2')
	ax.view_init(elev=50, azim=-26)
	ax.set(zlim=(0, 1.2))
	ax.set_xticks([0,gride_size-1])
	ax.set_yticks([0,gride_size-1])
	ax.set_zticks([0,0.5,1])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Occupancy')

	#plot u_3
	file = folder_dir+'SR_u3_'+str(i)+'.npy'
	temp_u3 = jnp.load(file)
	temp_u3 = np.max(temp_u3, axis=1)



	temp_u3 = temp_u3.reshape((5,5))

	ax = fig.add_subplot(1, 3, 3, projection='3d')
	ax.plot_surface(X, Y, temp_u3, rstride=1, cstride=1,
	                cmap='viridis', edgecolor='none')
	ax.set_title('u3')
	ax.view_init(elev=50, azim=-26)
	ax.set(zlim=(0, 1.2))
	ax.set_xticks([0,gride_size-1])
	ax.set_yticks([0,gride_size-1])
	ax.set_zticks([0,0.5,1])

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Occupancy')


	# plt.show()

	break
	# plot_filename = folder_dir+'SR_3d_v1_'+str(i)
	# plt.savefig(plot_filename)
	# plt.close()

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
