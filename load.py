import tensorflow as tf
import numpy as np

import shared

with open(shared.teams_location) as file:
    teams = file.read().split('\n')

num_teams = len(teams)

team_ids = range(num_teams)
teams_to_ids = dict(zip(teams, team_ids))

one_hot_teams = tf.one_hot(team_ids, num_teams)

x = np.load(shared.xs_location)
y = np.load(shared.ys_location)

x = tf.constant(x)
y = tf.constant(y)

team_vector_size = num_teams
meta_vector_size = x.shape[1] - (2 * team_vector_size)

y_shape = y.shape
if len(y_shape) == 1:
    output_size = 1
else:
    output_size = y_shape[1]
