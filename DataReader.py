import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
	"""Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

	### YOUR CODE HERE
	import pickle
	y_train = []
	x_train = []
	for batch in range(1, 6):
		with open(data_dir + '/data_batch_'+str(batch), 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		y_train.extend(dict[b'labels'])
		for x in dict[b'data']:
			x_train.append(x)

	with open(data_dir + '/test_batch', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		y_test = np.array(dict[b'labels'])
		y_test = y_test.astype('int32')
		x_test = dict[b'data']
		x_test = x_test.astype('float32')

	x_train = np.array(x_train)
	x_train = x_train.astype('float32')
	y_train = np.array(y_train)
	y_train = y_train.astype('int32')
	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape)
	
	### END CODE HERE

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid

