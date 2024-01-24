import tensorflow as tf

import load
import models

predictor = models.Predictor()
predictor.load()

is_defense = input('Defense (y)?\n')
if is_defense == 'y':
    vectorizer = predictor.defense_vectorizer
else:
    vectorizer = predictor.offense_vectorizer
vectors = vectorizer.predict(load.one_hot_teams)

inp_team = 'frc' + input('Team Number?\n')
inp_vector = vectors[load.teams_to_ids[inp_team]]

teams_and_dists = []
for team, index in load.teams_to_ids.items():
    if inp_team != team:
        teams_and_dists.append((team, tf.math.reduce_euclidean_norm(inp_vector - vectors[index]).numpy()))

teams_and_dists.sort(key=lambda x: x[1], reverse=True)
for team_and_dist in teams_and_dists:
    print(team_and_dist)
