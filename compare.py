import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.decomposition as dc

import load
import models

model, offense_vectorizer, defense_vectorizer = models.create_models()
models.load_model(model)

is_defense = input('Defense (y)?\n')
if is_defense == 'y':
    vectorizer = defense_vectorizer
else:
    vectorizer = offense_vectorizer
vectors = vectorizer(load.one_hot_teams)

inp_team = 'frc' + input('Team Number?\n')
inp_vector = vectors[load.teams_to_ids[inp_team]]

teams_and_dists = []
for team, index in load.teams_to_ids.items():
    if inp_team != team:
        teams_and_dists.append((team, tf.math.reduce_euclidean_norm(inp_vector - vectors[index]).numpy()))

teams_and_dists.sort(key=lambda x: x[1], reverse=True)
for team_and_dist in teams_and_dists:
    print(team_and_dist)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(*dc.PCA(3).fit_transform(vectors).T)
plt.show()
