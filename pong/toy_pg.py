# Dora Jambor
# 06.27.2016

'''
Policy gradients for reinforcement learning
http://karpathy.github.io/2016/05/31/rl/
Numerical toy example: approximating the round function
'''

import numpy as np
import random
import cPickle as pickle

# hyperparameters
eta = 1.5
batch_size = 10
episodes = 200000

# data point + correct action
input_data = [
    [0.1,0],
    [0.2,0],
    [0.3,0],
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

test_data = [
    [0.04,0],
    [0.12,0],
    [0.25,0],
    [0.35,0],
    [0.48,0],
    [0.42,0],
    [0.56,1],
    [0.62,1],
    [0.76,1],
    [0.88,1],
    [0.81,1],
    [0.99,1]
    ]

num_inputs = 1
action_space = np.asarray([[0],[1]])
num_actions = len(action_space)

# parameters to train
model = {}
model['W1'] = np.random.rand(num_actions, num_inputs)
model['b1'] = np.random.rand(num_actions, 1)

# helper funcs
def feedforward(data, model = model):
    ''' Function to be learnt -> takes some data point and outputs some class scores'''
    activation = np.dot(model['W1'], data) + model['b1']
    output = sigmoid(activation)
    return activation, output

def backprop(input_neurons, preactivation, gradient):
    '''Backpropagates the gradient/error through network and calculates some delta to update the parameters by'''
    sp = sigmoid_prime(preactivation)
    # gradient = loss(logP, action)      # for normal supervised learning
    delta = gradient * sp    # delta is the error
    db = delta
    dw = delta * input_neurons
    return dw, db

def update(dw, db, batch_size):
    '''Updates weights and biases, the parameters that will define our feedforward function'''
    model['b1'] += eta * db / batch_size
    model['W1'] += eta * dw / batch_size

def loss(desired, estimated):
    '''Calculates the Mean Squared Error -> See how far off the estimated output is from where we want to be'''
    return 0.5 * np.sum(desired - estimated)**2

def loss_prime(desired, estimated):
    ''' Calculates derivative of the Mean Squared Error'''
    return estimated - desired

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s*(1-s)

def sample(observed_action):
    '''Samples some action based on their probability distribution of the action space'''
    flip = np.random.rand()
    probs = observed_action/np.sum(observed_action)
    cum_probs = np.cumsum(probs)
    return np.searchsorted(cum_probs, flip)

def round_num(num, model = model):
    '''Classifies a data point with the trained weights'''
    _, output = feedforward(num, model)
    return sample(output)

def gradient_ascent(input_data):
    '''Training process: learns to approximate the best actions to take to arrive at the right outcome'''
    db_buffer = np.zeros(model['W1'].shape)
    dw_buffer = np.zeros(model['b1'].shape)

    for episode in xrange(episodes):
        correct = 0
        train_len = len(input_data)
        test_len = len(test_data)
        np.random.shuffle(input_data)

        for data, correct_action in input_data:
            # feedforward
            preactivation, output = feedforward(data)

            # sample an action
            action_taken = sample(output)

            # calculate the gradient based on reward
            if action_taken == correct_action:
                reward = 1.0
                correct += 1
            else:
                reward = -1.0

            # no loss function needed -> the loss w.r.t. your output is your "hard-coded" rewards i.e. your gradients
            gradient = np.zeros(action_space.shape)
            gradient[action_taken] = reward

            # backpropagate to calc the incremental parameter changes to make
            delta_w, delta_b = backprop(data, preactivation, gradient)
            db_buffer += delta_b
            dw_buffer += delta_w

        if episode%batch_size == 0:
            # update parameters by deltas
            update(dw_buffer, db_buffer, batch_size)
            db_buffer = np.zeros(model['W1'].shape)
            dw_buffer = np.zeros(model['b1'].shape)

        if episode%200 == 0:
            # check accuracy
            accuracy = round(float(correct)/train_len * 100,2)

            # check MSE of estimated action
            desired = np.zeros(output.shape)
            desired[correct_action] = correct_action
            cost = loss(desired, output) 

            print 'Accuracy: {0} / 100 '.format(accuracy) + 'Loss: ' + str(cost)
            res = [(round_num(x),y) for x,y in test_data]
            correct_count = sum(int(x==y) for x,y in res)
            test_accuracy = round(float(correct_count)/ test_len * 100, 2)
            print 'Test data accuracy: {0} / 100'.format(test_accuracy)
            if accuracy > 90 and test_accuracy > 90:
                return


if __name__ == "__main__":
    gradient_ascent(input_data)
    pickle.dump(model, open('model.p', 'wb'))
