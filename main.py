import numpy as np
import tensorflow as tf 
import argparse
from gym_ma.envs.gridworld_env import MAEnv

def run(args):
	config = tf.ConfigProto()
	with tf.Session(config=config) as sess:
		env = MAEnv()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	print(args)
	run(args)
	pass