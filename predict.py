import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.decomposition as dc

from . import load, models

def print_score(name, means, stddevs):
    print(f'{name}:')

    print('  Total:')
    print(f'    Mean:               {means[0]}')
    print(f'    Standard Deviation: {stddevs[0]}')

    print('  Auto:')
    print(f'    Mean:               {means[1]}')
    print(f'    Standard Deviation: {stddevs[1]}')

    print('  Teleop:')
    print(f'    Mean:               {means[2]}')
    print(f'    Standard Deviation: {stddevs[2]}')

    print('  Foul:')
    print(f'    Mean:               {means[3]}')
    print(f'    Standard Deviation: {stddevs[3]}')


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

means = output.mean()
stddevs = output.stddev()

print_score('Blue', means[0], stddevs[0])
print_score('Red', means[1], stddevs[1])
