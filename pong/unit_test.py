import cPickle as pickle

from toy_pg import *

trained_model = pickle.load(open('model.p', 'rb'))

repeat = 'n'
while repeat == 'n':
    number = float(raw_input('Give me a number from 0 to 1 to round: '))
    print round_num(number, model = trained_model)

    repeat = raw_input('Was my guess correct (y/n)? ')
    if repeat == 'y':
        print 'Great!'
        break

