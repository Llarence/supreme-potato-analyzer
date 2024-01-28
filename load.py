import tensorflow as tf
import numpy as np

from . import paths

class Data():
    def __init__(self, year):
        self.year = year

        with open(paths.teams_location(year)) as file:
            self.teams = file.read().split('\n')

        self.num_teams = len(self.teams)

        self.team_ids = range(self.num_teams)
        self.teams_to_ids = dict(zip(self.teams, self.team_ids))

        self.one_hot_teams = tf.one_hot(self.team_ids, self.num_teams)

        self.x_offense = np.load(paths.x_offense_location(year))
        self.x_defense = np.load(paths.x_defense_location(year))
        self.x_meta = np.load(paths.x_meta_location(year))
        self.y = np.load(paths.y_location(year))

        self.x_offense = tf.constant(self.x_offense)
        self.x_defense = tf.constant(self.x_defense)
        self.x_meta = tf.constant(self.x_meta)
        self.y = tf.constant(self.y)

        self.test_x_offense = np.load(paths.test_x_offense_location(year))
        self.test_x_defense = np.load(paths.test_x_defense_location(year))
        self.test_x_meta = np.load(paths.test_x_meta_location(year))
        self.test_y = np.load(paths.test_y_location(year))

        self.test_x_offense = tf.constant(self.test_x_offense)
        self.test_x_defense = tf.constant(self.test_x_defense)
        self.test_x_meta = tf.constant(self.test_x_meta)
        self.test_y = tf.constant(self.test_y)

        self.team_vector_size = self.x_offense.shape[1]
        self.meta_vector_size = self.x_meta.shape[1]

        self.output_size = self.y.shape[1]
