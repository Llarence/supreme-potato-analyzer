import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.decomposition as dc
import numpy as np

from .. import load, models
from . import config

data = load.Data(config.year)
model = models.GameModel(config.year, data.team_vector_size, data.meta_vector_size, data.output_size)
model.load_model()

is_offense = input('Offense (y)?\n')
is_mean = input('Mean (y)?\n')
value_type = int(input('Total (0), Auto (1), Teleop (2), Foul (3)?\n'))

if is_offense == 'y':
    if is_mean == 'y':
        vectorizer = model.mean_offense_vectorizers[value_type]
    else:
        vectorizer = model.deviation_offense_vectorizers[value_type]
else:
    if is_mean == 'y':
        vectorizer = model.mean_defense_vectorizers[value_type]
    else:
        vectorizer = model.deviation_defense_vectorizers[value_type]

vectors = vectorizer(data.one_hot_teams)

inp_team = 'frc' + input('Team Number?\n')
inp_id = data.teams_to_ids[inp_team]
inp_vector = vectors[inp_id]

teams_and_dists = []
for team, index in data.teams_to_ids.items():
    if inp_team != team:
        teams_and_dists.append((team, tf.math.reduce_euclidean_norm(inp_vector - vectors[index]).numpy()))

teams_and_dists.sort(key=lambda x: x[1], reverse=True)
for team_and_dist in teams_and_dists:
    print(team_and_dist)

vector_size = vectors.shape[1]

fig = plt.figure(figsize=(12, 12))

if vector_size > 2:
    ax = fig.add_subplot(projection='3d')
else:
    ax = fig.add_subplot()

colors = [(1, 0, 0)] * len(vectors)
colors[inp_id] = (0, 0, 1)
if vector_size > 3:
    ax.scatter(*dc.PCA(3).fit_transform(vectors).T, c=colors)
else:
    if vector_size == 1:
        ax.scatter(vectors, np.zeros_like(vectors), c=colors)
    else:
        ax.scatter(*tf.transpose(vectors), c=colors)

plt.show()
