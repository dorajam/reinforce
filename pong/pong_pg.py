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
episodes = 20000000    # 100 rollouts per episode
epsilon = 0.2     # exploration parameter
eta = 0.4

def preprocess(input_data):
    input_data  = input_data.ravel()
    return input_data

def feedforward(activation):
    output = np.dot(model['params1']['W1'], activation) + model['params1']['b1']
    logP = sigmoid(output)
    return output, logP


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s*(1-s) + 1e-5

def loss(predicted, desired):
    return desired - predicted


# toy example
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

games = input_data
action_space = np.asarray([[0],[1]])
num_actions = len(action_space)
num_inputs = 1
counter, correct = 0.0, 0.0

model = {}
model['params1'] = {'W1': np.random.rand(num_actions, num_inputs), 'b1': np.random.rand(num_actions, 1)}
# model['params1'] = {'W1': np.asarray([[-10.0], [90.0]]), 'b1': np.asarray([[30.0], [-15.0]])}

for episode in xrange(episodes):
    np.random.shuffle(games)
    for game, action in games:
        counter+=1

        if episode > 10:
            pass
            # import ipdb;ipdb.set_trace()
        # feedforward
        preactivation, p = feedforward(game)

        # explore vs exploit
        flip = np.random.rand()
        # index = 0
        probs = p/np.sum(p)
        cum_probs = np.cumsum(probs)
        action_taken = np.searchsorted(cum_probs, flip)
        # if flip * np.sum(p) > p[index]:
        #     index = 1
        # action_taken = index

        # if flip < epsilon:
        #     # print 'Exploring....'
        #     index = random.randrange(len(action_space))
        #     action_taken = action_space[index]
        # else:
        #     # print 'Exploiting....'
        #     index = np.argmax(logP)
        #     action_taken = action_space[index]

        if action_taken == action:
            reward = 1.0
            correct += 1
        else:
            reward = -1.0
        # no loss function needed -> the loss w.r.t. your output is your "hard-coded" rewards i.e. your gradients
        gradient = np.zeros(action_space.shape)
        gradient[action_taken] = reward
        
        # BACKPROP
        # through sigmoid -> sigmoid(1-sigmoid) of the preactivation!
        sp = sigmoid_prime(preactivation)
        # gradient = loss(logP, action)
        # print gradient
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





# MAIN
#################################
# env = gym.make("Pong-v0")
# observation = env.reset()
# for i in range(2000):
#     preprocess(observation)
#     # print observation.shape

#     env.render()
# take some data 210x160x3 -> RGB



