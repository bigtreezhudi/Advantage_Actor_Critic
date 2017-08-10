import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random
from collections import deque
import baselines.common.tf_util as U


GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10
BATCH_SIZE = 10
UPDATE_SEQ = 1000
TRAIN_SEQ = 4

class Advantage_Actor_Critic():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.action_dim = env.action_space.n

        self.critic_func_input, self.critic_func = self.create_network(1, False, "critic_func")
        self.actor_func_input, self.actor_func = self.create_network(self.action_dim, True, "actor_func")

        self.critic_y_input, self.critic_optimizer = self.create_training_critic_method()
        self.I, self.delta, self.actor_optimizer = self.create_training_actor_method()

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_network(self, output_num, is_actor, scope):
        with tf.variable_scope(scope, reuse=False):
            state_input = tf.placeholder('float', [None, 4])
            out = layers.fully_connected(state_input, num_outputs=32, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=100, activation_fn=tf.nn.relu)
            output_value = layers.fully_connected(out, num_outputs=output_num, activation_fn=None)
            if is_actor:
                output_value =layers.softmax(output_value)

            return state_input, output_value

    def create_training_critic_method(self):
        y_input = tf.placeholder("float", [None, 1])
        cost = tf.reduce_mean(tf.square(y_input - self.critic_func))
        optimizer = tf.train.AdamOptimizer(1e-08).minimize(cost)
        return y_input, optimizer

    def create_training_actor_method(self):
        I = tf.placeholder("float", [None, 1])
        delta = tf.placeholder("float", [None, 1])
        self.state_actor = tf.placeholder("float", [None, self.action_dim])
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_func")
        actor = tf.multiply(self.state_actor, self.action_input)
        coefficient = tf.multiply(-1.0, tf.divide(tf.multiply(I, delta), actor))
        log_actor = tf.multiply(coefficient, self.actor_func)
        gredients = tf.gradients(log_actor, actor_vars)
        optimizer = tf.train.AdamOptimizer(1e-08).apply_gradients(zip(gredients, actor_vars))
        return I, delta, optimizer


    def perceive(self, state, action, reward, next_state, done, I, step_num):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done, I))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) >= REPLAY_SIZE and step_num % TRAIN_SEQ == 0:
            self.train_network()


    def train_network(self):
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        I_batch = []
        for state, action, reward, next_state, _, I in minibatch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            I_batch.append(np.array([I]))
        y_batch = []
        next_state_critic_batch = self.critic_func.eval(feed_dict={self.critic_func_input: next_state_batch})
        state_critic_batch = self.critic_func.eval(feed_dict={self.critic_func_input: state_batch})
        state_actor_batch = self.actor_func.eval(feed_dict={self.actor_func_input: state_batch})
        # state_actor_batch = np.reshape(state_actor_batch, (BATCH_SIZE, self.action_dim))
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(np.array([reward_batch[i]]))
            else:
                y_batch.append(np.array([reward_batch[i]]) + GAMMA * next_state_critic_batch[i])
        delta_batch = y_batch - state_critic_batch

        self.critic_optimizer.run(feed_dict={
            self.critic_y_input: y_batch,
            self.critic_func_input: state_batch
        })

        self.actor_optimizer.run(feed_dict={
            self.actor_func_input: state_batch,
            self.I: I_batch,
            self.delta: delta_batch,
            self.action_input: action_batch,
            self.state_actor: state_actor_batch
        })

    def select_action(self, state):
        action_dis = self.actor_func.eval(feed_dict={
            self.actor_func_input: [state]
        })[0]
        action = np.random.choice(self.action_dim, 1, p=action_dis)
        return action[0]

    def action(self, state):
        return np.argmax(self.q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
