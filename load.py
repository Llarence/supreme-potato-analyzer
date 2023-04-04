import json
import tensorflow as tf
import numpy as np

import constants

matches = None
teams_to_indices = None
with open(constants.cache_location) as file:
    matches, teams_to_indices = json.loads(file.read())

team_indices = list(teams_to_indices.values())
one_hot_cache = tf.one_hot(team_indices, len(team_indices))

def match_to_tensors(match):
    blue_robots = one_hot_cache[teams_to_indices[match[0][0][0]]] + one_hot_cache[teams_to_indices[match[0][0][1]]] + one_hot_cache[teams_to_indices[match[0][0][2]]]
    blue_output = match[0][1]
    blue_output[5] = int(blue_output[5])
    red_output = tf.convert_to_tensor(blue_output, dtype=tf.float32)

    red_robots = one_hot_cache[teams_to_indices[match[1][0][0]]] + one_hot_cache[teams_to_indices[match[1][0][1]]] + one_hot_cache[teams_to_indices[match[1][0][2]]]
    red_output = match[1][1]
    red_output[5] = int(red_output[5])
    red_output = tf.convert_to_tensor(red_output, dtype=tf.float32)

    match_meta = np.zeros(len(team_indices), dtype=np.float32)
    match_meta[0] = float(match[2][0])
    match_meta = tf.convert_to_tensor(match_meta, dtype=tf.float32)

    return ((tf.cast(tf.stack([blue_robots, red_robots, match_meta]), dtype=tf.float32), blue_output), (tf.cast(tf.stack([red_robots, blue_robots, match_meta]), dtype=tf.float32), red_output))
