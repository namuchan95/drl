import gym, tensorflow as tf, numpy as np
import argparse
from collections import deque

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--env", nargs=1, default="CartPole-v1",
					help="enter the environment name")
argument_parser.add_argument("--n_layers", nargs=None, type=int, default=2,
					help="number of hidden layers in the mlp")
argument_parser.add_argument("--hidden_units", nargs='?', default=[100, 100],
					help="number of units in hidden layers shape should be consistent with n_layers")
argument_parser.add_argument("--iterations", nargs=None, type=int, default=100,
					help="max number of iterations")
argument_parser.add_argument("--max_episode_len", nargs=None, type=int, default=200,
					help="maximum episode length")
argument_parser.add_argument("--buffer_size", nargs=None, type=int, default=1000,
					help="experience buffer size")


class Agent(object):
	def __init__(self, env, n_layers, hidden_units):
		self.env = gym.make(env)
		self.build_nn(self.env.observation_space.shape, n_layers, hidden_units, self.env.action_space.shape)
		self.experience_replay = deque()
		self.exploration_decay = 1e-3
		self.exploration = 0.5
		self.gamma = 0.1
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.batch_size = 64
		self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
		pass

	def sample_action(self, observation):
		if np.random.rand(0,1) < self.exploration :
			return self.env.sample()
		else :
			q_values = self.sess.run(self.predict_q, feed_dict={self.obs: observation}) 
			return np.argmax(q_values)

	def train(self):
		minibatch_experience = np.random.choice(self.experience_replay, self.batch_size)
		next_q = self.sess.run(self.predict_q, feed_dict={self.obs: minibatch_experience[:,2]})
		max_q = np.max(next_q, axis=1)
		labels = minibatch_experience[:,1] + self.gamma*max_q
		self.sess.run(self.optimizer.minimize(self.loss), 
						feed_dict={self.obs : minibatch_experience[:,0],
									self.labels : labels})

	def build_nn(self, input_shape, n_layers, hidden_units, output_shape, activation=tf.nn.tanh):
		assert len(hidden_units) == n_layers, "n_layers doesn't match with the shape of hidden_units"
		with tf.variable_scope('q_network'):
			self.obs = tf.placeholder(shape=[None, input_shape[0]], dtype=tf.float32)
			self.labels = tf.placeholder(shape=[None, 2], dtype=tf.float32)
			hidden_act = [self.obs]
			for i in range(n_layers):
				hidden_act += [tf.layers.Dense(hidden_units[i], activation=activation)(hidden_act[i])]
			self.predict_q = tf.layers.Dense(2, activation=None)(hidden_act[n_layers]) #final layer should be linear for convergence 
			self.loss = tf.losses.mean_squared_error(self.predict_q, self.labels)

def main():
	cmd_args = argument_parser.parse_args()
	print (cmd_args)
	agent = Agent(cmd_args.env, cmd_args.n_layers, cmd_args.hidden_units)
	#print (obs)
	for _ in range(cmd_args.iterations):
		obs = agent.env.reset()
		e = 0
		while e < cmd_args.max_episode_len:	
			action = agent.sample_action(np.array([obs]))
			next_obs, reward, done, _ = agent.env.step(action)
			if len(agent.experience_replay) > cmd_args.buffer_size : agent.experience_replay.popleft()
			agent.experience_replay.append([obs, reward, next_obs])
			if done : 
				print('Episode ended in {} steps'.format(e))
				break
			obs = next_obs
			e += 1

if __name__ == "__main__":
	main()

