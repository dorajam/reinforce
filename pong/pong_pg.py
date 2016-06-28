# Dora Jambor
# 06.27.2016

'''
Policy gradient for reinforcement learning
http://karpathy.github.io/2016/05/31/rl/
Numerical toy example + game of pong
'''

import numpy as np
import gym
import random

# hyperparameters
roll_outs = 100   # update after 100 games
episodes = 200    # 100 rollouts per episode
epsilon = 0.2     # take random action 20% of the time!

def preprocess(input_data):
    input_data  = input_data.ravel()
    return input_data

def feedforward(activation):
    output = np.dot(model['params1']['W1'], activation)
    logP = sigmoid(output)
    return logP


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# toy example
input_data = [
    [0.0,0],
    [0.1,0],
    [0.2,0],
    [0.3,0],
    [0.4,0],
    [0.5,1],
    [0.6,1],
    [0.7,1],
    [0.8,1],
    [0.9,1]
    ]

games = input_data
res = []
action_space = np.asarray([0,1])
num_actions = len(action_space)
num_inputs = len(input_data)

model = {}
model['params1'] = {'W1': np.random.rand(num_actions, num_inputs), 'b1': np.random.rand(num_actions, 1)}
for _ in xrange(episodes):
    for game, action in games:
        # feedforward
        logP = feedforward(game)
        index = random.randrange(len(action_space))
        sampled_action = action_space[index]

        res.append((logP, sampled_action))
        if sampled_action == action:
            reward = 1.0
        else:
            reward = -1.0
        gradient = np.zeros(action_space.shape)
        gradient[index] = reward
    # backprop
    





# MAIN
#################################
# env = gym.make("Pong-v0")
# observation = env.reset()
# for i in range(2000):
#     preprocess(observation)
#     # print observation.shape

#     env.render()
# take some data 210x160x3 -> RGB



