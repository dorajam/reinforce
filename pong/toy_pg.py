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
episodes = 20000000    # 100 rollouts per episode
eta = 1.5

input_data = [
    [0.1,0],
    [0.2,0],
    [0.3,0],
    [0.4,0],
    [0.4,0],
    [0.41,0],
    [0.5,1],
    [0.56,1],
    [0.6,1],
    [0.7,1],
    [0.81,1],
    [0.86,1],
    [0.98,1]
    ]

# helper funcs
def feedforward(activation):
    output = np.dot(model['params1']['W1'], activation) + model['params1']['b1']
    logP = sigmoid(output)
    return output, logP

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s*(1-s) + 1e-5

def loss(predicted, taken):
    return taken - predicted

games = input_data
action_space = np.asarray([[0],[1]])
num_actions = len(action_space)
num_inputs = 1
counter, correct = 0.0, 0.0

model = {}
model['params1'] = {'W1': np.random.rand(num_actions, num_inputs), 'b1': np.random.rand(num_actions, 1)}

for episode in xrange(episodes):
    np.random.shuffle(games)
    for game, action in games:
        counter+=1

        # feedforward
        preactivation, p = feedforward(game)

        # explore vs exploit
        flip = np.random.rand()
        probs = p/np.sum(p)
        cum_probs = np.cumsum(probs)
        action_taken = np.searchsorted(cum_probs, flip)

        if action_taken == action:
            reward = 1.0
            correct += 1
        else:
            reward = -1.0
        # no loss function needed -> the loss w.r.t. your output is your "hard-coded" rewards i.e. your gradients
        gradient = np.zeros(action_space.shape)
        gradient[action_taken] = reward

        # uncomment this if you wanna use the actual loss function -> change sign of the negative reward
        # label = np.zeros((2,1))
        # label[action] = action
        # gradient = loss(p, label) * gradient

        # BACKPROP
        # through sigmoid -> sigmoid(1-sigmoid) of the preactivation!
        sp = sigmoid_prime(preactivation)
        # gradient = loss(logP, action)      # for normal supervised learning
        delta = gradient * sp

        # print delta

        db = delta
        dw = delta * game
        # print 'delta', delta, 'sp', sp

        # GRADIENT ASCENT
        model['params1']['b1'] += eta * db
        model['params1']['W1'] += eta * dw

    if episode%2000 == 0:
        accuracy = round(correct/counter * 100,2)
        print 'Accuracy: {0} / {1}'.format(accuracy, 100)


