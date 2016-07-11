# Dora Jambor
# 06.27.2016

'''
Policy gradient for reinforcement learning
http://karpathy.github.io/2016/05/31/rl/
Game of pong
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
    # import ipdb;ipdb.set_trace()
    preactivation = np.dot(model['W1'], activation)
    preactivation[preactivation < 0] = 0    # ReLU nonlinearity
    final_activation = np.dot(model['W2'], preactivation)
    output = sigmoid(final_activation)
    return preactivation, output

def backprop(inp, preactivation, gradient):
    # skipping the backprop for the sigmoid func!!!! --> fix?!
    # sp = sigmoid_prime(final_output)
    # delta = gradient * sp
    dW2 = np.dot(gradient.T, preactivation).ravel()
    hidden_gradients = np.dot(model['W2'], gradient)
    hidden_gradients[preactivation == 0] = 0
    dw1 = np.dot(inp.T, hidden_gradients)
    return {'W1':dW1, 'W2':dW2}

def discount_rewards(r):
  """ take 1D float array of rewards from the last 21 games and discounts them"""
  discounted_r = np.zeros(r.shape)
  R = 0
  for e in reversed(xrange(0, r.size)):
    if r[e] != 0: R=0 # reset the sum, since this was a game boundary (pong specific!)
    R = R * gamma + r[e]
    discounted_r[e] = R
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
num_actions = 1
num_inputs = 80*80
episode, correct = 0, 0.0

model = {}
model['W1'] = np.random.rand(hidden, num_inputs)
model['W2'] = np.random.rand(num_actions, hidden)
grad_buffer = { k : np.zeros(w.shape) for k, w in model.iteritems() }

prev_frame = None
inputs, preacts, rewards, losses = [],[],[],[]

frame = env.reset()
while True:
    env.render()
    # preprocessing
    curr_frame = preprocess(frame)
    curr_frame = curr_frame - prev_frame if prev_frame is not None else np.zeros(num_inputs)
    prev_frame = curr_frame

    # feedforward
    print curr_frame
    preactivation, output = feedforward(curr_frame)
    preacts.append(preactivation)
    inputs.append(curr_frame)

    # explore vs exploit
    flip = np.random.uniform()
    print flip, output
    action = 2 if flip < output else 3
    y = 1 if action == 2 else 0
    losses.append(action - output)

    frame, reward, done, info = env.step(action)
    rewards.append(reward)

    if done:
        episode += 1
        frame = env.reset()

        np_preacts = np.vstack(preacts)
        np_inputs = np.vstack(inputs)
        np_rewards= np.vstack(rewards)
        np_losses = np.vstack(losses)
        preacts, inputs, rewards, losses = [],[],[],[]

        # normalize your rewards! 
        discounted_r = discount_rewards(np_rewards)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

        gradient = discounted_r
        print gradient.shape, np_preacts.shape
        updates = backprop(np_inputs, np_preacts, gradient)
        for k in model: grad_buffer[k] += updates[k] # accumulate grad over batch

        if episode % batch_size == 0:
            for k,v in model.iteritems():
                grad = grad_buffer[k]
                model[k] += eta * grad / batch_size
                grad_buffer = np.zeros(v.shape)








