import tensorflow as tf
import random
import math

import load
import models
import constants

predictor = models.Predictor()

x = []
y = []
for match_num, match in enumerate(load.matches):
    print('\u001b[2K\r{:.2f}%'.format(match_num / len(load.matches) * 100), end='', flush=True)
    
    blue, red = load.match_to_tensors(match)

    x.append(blue[0])
    y.append(blue[1])

    x.append(red[0])
    y.append(red[1])

print('\u001b[2K\r100.00%')

zipped = list(zip(x, y))
random.shuffle(zipped)
x, y = zip(*zipped)

x = tf.stack(x)
y = tf.stack(y)

x_train = x[constants.test_size:]
y_train = y[constants.test_size:]

x_test = x[:constants.test_size]
y_test = y[:constants.test_size]

i = 0
curr_loss = math.inf
prev_loss = math.inf
while True:
    prev_loss = curr_loss

    predictor.fit(x_train, y_train)

    curr_loss = predictor.evaluate(x_test, y_test)
    if curr_loss > prev_loss:
        break
    else:
        i += 1

predictor = models.Predictor()

predictor.fit(x, y, epochs=i)

predictor.save()