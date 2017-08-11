import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from collections import deque
import numpy as np
from Advantage_Actor_Critic import Advantage_Actor_Critic
import pickle
import time

GAMMA = 0.99
EPISODE = 350
TEST = 100

def main():
    env = gym.make("CartPole-v0")
    agent = Advantage_Actor_Critic(env)
    episodes_rewards = []
    avg_rewards = []
    skip_rewards = []
    step_num = 0
    for episode in range(EPISODE):
        goal = 0
        I = 1.0
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            I = GAMMA * I
            # env.render()
            agent.perceive(state, action, reward, next_state, done, I, step_num)
            goal += reward
            step_num += 1
            state = next_state
            if done:
                if len(episodes_rewards) == 100:
                    episodes_rewards.pop(0)
                episodes_rewards.append(goal)
                break

        print("Episode: ", episode, " Last 100 episode average reward: ", np.average(episodes_rewards), " Toal step number: ", step_num)
        avg_rewards.append(np.average(episodes_rewards))
        if episode % 100 == 0:
            out_file = open("avg_rewards.pkl",'wb')
            pickle.dump(avg_rewards, out_file)
            out_file.close()
            agent.saver.save(agent.session, 'saved_networks/' + 'network' + '-dqn', global_step=episode)

    env.close()

def play():
    env = gym.make("CartPole-v0")
    agent = Advantage_Actor_Critic(env)
    for episode in range(TEST):
        goal = 0
        step_num = 0
        state = env.reset()
        while True:
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            step_num += 1
            env.render()
            goal += reward
            state = next_state
            if done:
                print("Episode: ", episode, " Total reward: ", goal)
                break

if __name__ == '__main__':
    main()
    play()
