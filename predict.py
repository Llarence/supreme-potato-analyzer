import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.decomposition as dc

import load
import models

model, _, _ = models.create_models()
models.load_model(model)

blue_vector = sum([load.one_hot_teams[load.teams_to_ids['frc' + input('Blue Team Number?\n')]]
                   for i in range(3)])
red_vector = sum([load.one_hot_teams[load.teams_to_ids['frc' + input('Red Team Number?\n')]]
                   for i in range(3)])
meta_vector = tf.constant([bool(input('Elims?\n')), int(input('Week?\n'))], dtype=tf.float32)

output = model((tf.stack([blue_vector, red_vector], 0),
                tf.stack([red_vector, blue_vector], 0),
                tf.stack([meta_vector, meta_vector], 0)))

print(f'Blue Mean: {output.mean()[0, 0]}, Blue Standard Deviation: {output.stddev()[0, 0]}')
print(f'Red Mean: {output.mean()[1, 0]}, Red Standard Deviation: {output.stddev()[1, 0]}')