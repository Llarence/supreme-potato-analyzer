import matplotlib.pyplot as plt
import tensorflow as tf

from .. import load, models
from . import config

data = load.Data(config.year)

team = 'frc' + input('Team Number?\n')
id = data.teams_to_ids[team]

x = tf.concat([data.x_offense, data.test_x_offense], 0)

y = tf.concat([data.y, data.test_y], 0)
y = tf.gather(y, tf.where(x[:, id])[:, 0])

plt.hist(y[:, 0], bins=25)
plt.show()

plt.hist(y[:, 1], bins=25)
plt.show()

plt.hist(y[:, 2], bins=25)
plt.show()

plt.hist(y[:, 3], bins=25)
plt.show()
