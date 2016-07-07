import cPickle as pickle

from toy_pg import *

# load model
trained_model = pickle.load(open('model.p', 'rb'))

count, correct = 0, 0

while True:
    number = float(raw_input('Give me a number from 0 to 1 to round: '))
    count+=1

    print 'That rounds to: ', round_num(number, model = trained_model)

    if raw_input('Was my guess correct (y/n)? ') == 'y':
        correct+=1

    if raw_input('Want to keep going (y/n)? ') == 'n':
        break

print 'Accuracy: ', round(float(correct)/count * 100, 2)

