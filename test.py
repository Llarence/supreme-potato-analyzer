import tensorflow as tf

import load
import models

team_indices = list(load.teams_to_indices.values())
one_hot_cache = tf.one_hot(team_indices, len(team_indices))

predictor = models.Predictor()
predictor.load()

team = 'frc' + input('Team Number?\n')

x = []
y = []
matches_played = []
for match_num, match in enumerate(load.matches):
    print('\u001b[2K\r{:.2f}%'.format(match_num / len(load.matches) * 100), end='', flush=True)
    if match[0][0][0] == team or match[0][0][1] == team or match[0][0][2] == team:
        x.append(load.match_to_tensors(match)[0][0])
        matches_played.append(match)

    if match[1][0][0] == team or match[1][0][1] == team or match[1][0][2] == team:
        x.append(load.match_to_tensors(match)[1][0])
        matches_played.append(match)

print('\u001b[2K\r100.00%')

x = tf.stack(x)

model_ys = predictor.predict(x)

for i in range(len(matches_played)):
    print(matches_played[i], model_ys[i])

x = []
y = []
x_blue = []
y_blue = []
x_red = []
y_red = []
for match_num, match in enumerate(load.matches):
    print('\u001b[2K\r{:.2f}%'.format(match_num / len(load.matches) * 100), end='', flush=True)
    
    blue, red = load.match_to_tensors(match)

    x.append(blue[0])
    y.append(blue[1])

    x_blue.append(blue[0])
    y_blue.append(blue[1])

    x.append(red[0])
    y.append(red[1])

    x_red.append(red[0])
    y_red.append(red[1])

print('\u001b[2K\r100.00%')

x = tf.stack(x)
y = tf.stack(y)

x_blue = tf.stack(x_blue)
y_blue = tf.stack(y_blue)

x_red = tf.stack(x_red)
y_red = tf.stack(y_red)

model_y = predictor.predict(x)

print(tf.reduce_mean(tf.abs(model_y - y), 0).numpy())
print(tf.reduce_mean(tf.abs(tf.reduce_sum(model_y[:][:4] - y[:][:4], 1))).numpy())

accuracy = 0
for i in range(len(y) // 2):
    print('\u001b[2K\r{:.2f}%'.format(i / len(y) * 200), end='', flush=True)
    blue_index = i * 2
    red_index = i * 2 + 1

    blue_score = sum(y[blue_index][:4])
    red_score = sum(y[red_index][:4])

    predicted_blue_score = sum(model_y[blue_index][:4])
    predicted_red_score = sum(model_y[red_index][:4])

    if blue_score == red_score:
        if predicted_blue_score == predicted_red_score:
            accuracy += 1
        else:
            accuracy += 0
    elif blue_score > red_score:
        if predicted_blue_score > predicted_red_score:
            accuracy += 1
        else:
            accuracy += 0
    else:
        if predicted_blue_score < predicted_red_score:
            accuracy += 1
        else:
            accuracy += 0

print('\u001b[2K\r100.00%')

accuracy /= len(y) // 2
print(accuracy)