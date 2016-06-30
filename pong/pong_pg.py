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
batch_size = 10
episodes = 200000    # 100 rollouts per episode
hidden = 200
epsilon = 0.2     # exploration parameter
gamma = 0.8
eta = 0.4

def preprocess(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def feedforward(activation):
    activation = activation.ravel()
    preactivation = np.dot(model['W1'], activation)
    preactivation[preactivation < 0] = 0    # ReLU nonlinearity
    final_activation = np.dot(model['W2'].transpose(), preactivation)
    output = sigmoid(final_activation)
    return preactivation, output

def backprop(preactivation, gradient):
    # skipping the backprop for the sigmoid func!!!! --> fix?!
    # sp = sigmoid_prime(final_output)
    # delta = gradient * sp
    
    dw1 = np.dot(preactivation.T, gradient)
    hidden_gradients = np.dot(model['W2'], gradient)
    hidden_gradients[preactivation == 0] = 0
    dw2 = np.dot(inp.T, hidden_gradients)
    return {'W1':dW1, 'W2':dW2}


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros(r.shape)
  R = 0
  for e in reversed(xrange(0, r.size)):
    if r[e] != 0: 0 = R # reset the sum, since this was a game boundary (pong specific!)
    R = R * gamma + r[t]
    discounted_r[t] = R
  return discounted_r

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s*(1-s) + 1e-5

def softmax(x):
    norm = np.exp(x)
    return norm / np.sum(norm)

def loss(predicted, desired):
    return desired - predicted

# MAIN
#################################
env = gym.make("Pong-v0")

# up or down
num_actions = 2
num_inputs = 80*80
episode, correct = 0, 0.0

model = {}
model['W1'] = np.random.rand(hidden, num_inputs)
model['W2'] = np.random.rand(hidden, num_actions)

prev_frame = None
inputs, preacts, rewards, losses = [],[],[],[]

frame = env.reset()
while True:
    env.render()
    # preprocessing
    # import ipdb;ipdb.set_trace()
    curr_frame = preprocess(frame)
    curr_frame = curr_frame - prev_frame if prev_frame is not None else np.zeros(num_inputs)
    prev_frame = curr_frame

    # feedforward
    preactivation, output = feedforward(curr_frame)
    preacts.append(preactivation)
    inputs.append(curr_frame)

    # explore vs exploit
    flip = np.random.rand()
    normalized = output/np.sum(output)
    cum_probs = np.cumsum(normalized)
    y = np.searchsorted(cum_probs, flip)
    losses.append(y - output)

    action = 2 if y == 1 else 3
    frame, reward, done, info = env.step(action)
    rewards.append(reward)

    if done:
        episode += 1
        frame = env.reset()

        np_preacts = vstack(preacts)
        np_inputs = vstack(inputs)
        np_rewards= vstack(rewards)
        np_losses = vstack(losses)
        preacts, inputs, rewards, losses = [],[],[],[]

        # normalize your rewards! 
        discounted_r = discount_rewards(np_rewards)
        discounter_r -= np.mean(discounter_r)
        discounter_r /= np.std(discounter_r)

        gradient = discounted_r
        dw = backprop(gradient)
        # remember dw

        # udate

            
    # # BACKPROP
    # # through sigmoid -> sigmoid(1-sigmoid) of the preactivation!
    # sp = sigmoid_prime(preactivation)
    # # gradient = loss(logP, action)
    # # print gradient
    # delta = gradient * sp

    # # print delta

    # db = delta
    # dw = delta * game
    # # print 'delta', delta, 'sp', sp
    
    # # GRADIENT ASCENT
    # model['params1']['b1'] += eta * db
    # model['params1']['W1'] += eta * dw

    # if episode%batch_size == 0:
    #     accuracy = round(correct/counter * 100,2)
    #     print 'Accuracy: {0} / {1}'.format(accuracy, 100)








