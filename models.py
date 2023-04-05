import tensorflow as tf

import constants
import load

def generate_offense_vectorizer():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(len(list(load.teams_to_indices.values())), activation='linear'),
        tf.keras.layers.Dense(2048, activation='linear'),
        tf.keras.layers.Dense(512, activation='linear'),
        tf.keras.layers.Dense(constants.team_vector_size, activation='linear')
    ])

    return model


def generate_defense_vectorizer():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(len(list(load.teams_to_indices.values())), activation='linear'),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(constants.team_vector_size, activation='linear')
    ])

    return model


def generate_mean_predictor():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(constants.team_vector_size * 2 + constants.meta_vector_size, activation='linear'),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(constants.outputs, activation='linear')
    ])

    return model


class Predictor(tf.keras.models.Model):
    def __init__(self):
        super(Predictor, self).__init__()   
        self.offense_vectorizer = generate_offense_vectorizer()
        self.defense_vectorizer = generate_defense_vectorizer()
        self.mean_predictor = generate_mean_predictor()

        self.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError())


    def call(self, x):
        offense_vector = self.offense_vectorizer(x[:, 0])
        defense_vector = self.defense_vectorizer(x[:, 1])

        match_vector = tf.concat([offense_vector, defense_vector, x[:, 2, :constants.meta_vector_size]], 1)

        prediction = self.mean_predictor(match_vector)
        return prediction


    def save(self):
        self.offense_vectorizer.save_weights('model/offense_vectorizer/model')
        self.defense_vectorizer.save_weights('model/defense_vectorizer/model')
        self.mean_predictor.save_weights('model/mean_predictor/model')


    def load(self):
        self.offense_vectorizer.load_weights('model/offense_vectorizer/model')
        self.defense_vectorizer.load_weights('model/defense_vectorizer/model')
        self.mean_predictor.load_weights('model/mean_predictor/model')