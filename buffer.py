import numpy as np

class Buffer(object):
	# The buffer in COMA is not Replay Buffer in off-policy algorithms
	# The buffer is small and will reset in few epsiodes
	# The volume is BS * T, BS for batch size and T for the maximum step of one episode
	# BS = n * n_ep, n for agents' quantity and n_ep for episodes per training cycle
	def __init__(self, obs_size, T, batch_size=30, n_agent=2):
		self.max_step = T
		self.batch_size = batch_size
		self.n_agent = n_agent
		self.n_episode = batch_size / n_agent
		self.obs_size = obs_size
		self.memory_size = [self.n_episode, self.n_agent, self.max_step]

		self.reset()

	def reset(self):
		self.obs1 = np.empty([self.memory_size]+self.obs_size, dtype=np.float16)
		self.obs2 = np.empty([self.memory_size]+self.obs_size, dtype=np.float16)
		self.actions = np.empty(self.memory_size, dtype=np.int16)
		self.rewards = np.empty(self.memory_size, dtype=np.float16)
		self.terminal = np.empty(self.memory_size, dtype=np.bool_)

		self.mask = np.zeros(self.memory_size, dtype=np.bool_)

	def insert(self, ep, agent, step, obs1, action, obs2, r, t):
		self.obs1[ep, agent, step] = obs1
		self.obs2[ep, agent, step] = obs2
		self.actions[ep, agent, step] = action
		self.rewards[ep, agent, step] = r 
		self.terminal[ep, agent, step] = t 
		self.mask[ep, agent, step] = True

	def retrieve_all(self):
		reduced_obs1 = []
		reduced_obs2 = []
		reduced_actions = []
		reduced_rewards = []
		reduced_terminals = []