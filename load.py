import tensorflow as tf
import numpy as np

from . import shared

with open(shared.teams_location) as file:
    teams = file.read().split('\n')

num_teams = len(teams)

team_ids = range(num_teams)
teams_to_ids = dict(zip(teams, team_ids))

one_hot_teams = tf.one_hot(team_ids, num_teams)

x_offense = np.load(shared.x_offense_location)
x_defense = np.load(shared.x_defense_location)
x_meta = np.load(shared.x_meta_location)
y = np.load(shared.y_location)

x_offense = tf.constant(x_offense)
x_defense = tf.constant(x_defense)
x_meta = tf.constant(x_meta)
y = tf.constant(y)

test_x_offense = np.load(shared.test_x_offense_location)
test_x_defense = np.load(shared.test_x_defense_location)
test_x_meta = np.load(shared.test_x_meta_location)
test_y = np.load(shared.test_y_location)

test_x_offense = tf.constant(test_x_offense)
test_x_defense = tf.constant(test_x_defense)
test_x_meta = tf.constant(test_x_meta)
test_y = tf.constant(test_y)

team_vector_size = x_offense.shape[1]
meta_vector_size = x_meta.shape[1]

output_size = y.shape[1]
