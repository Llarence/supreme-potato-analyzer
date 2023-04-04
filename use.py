import tensorflow as tf

import load
import models

team_indices = list(load.teams_to_indices.values())
one_hot_cache = tf.one_hot(team_indices, len(team_indices))

predictor = models.Predictor()
predictor.load()

is_defense = input('Defense (y)?\n')
if is_defense == 'y':
    vectorizer = predictor.defense_vectorizer
else:
    vectorizer = predictor.offense_vectorizer
vector_cache = vectorizer.predict(one_hot_cache)

inp_team = 'frc' + input('Team Number?\n')
inp_vector = vector_cache[load.teams_to_indices[inp_team]]

teams_and_dists = []
for team, index in load.teams_to_indices.items():
    if inp_team != team:
        teams_and_dists.append((team, tf.math.reduce_euclidean_norm(tf.stack(inp_vector - vector_cache[load.teams_to_indices[team]])).numpy()))

teams_and_dists.sort(key=lambda x: x[1], reverse=True)
for team_and_dist in teams_and_dists:
    print(team_and_dist)