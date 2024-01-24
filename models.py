import tensorflow as tf

import load
import config

offense_vector_size = 4
defense_vector_size = 4
meta_vector_size = 4
predictor_vector_size = offense_vector_size + defense_vector_size + meta_vector_size

offense_location = f'data/{config.year}/model/offense/'
defense_location = f'data/{config.year}/model/defense/'
meta_location = f'data/{config.year}/model/meta/'
predictor_location = f'data/{config.year}/model/predictor/'

class Predictor(tf.keras.models.Model):
    def __init__(self):
        super(Predictor, self).__init__()   
        self.offense_vectorizer = self._generate_offense_vectorizer()
        self.defense_vectorizer = self._generate_defense_vectorizer()
        self.meta_vectorizer = self._generate_meta_vectorizer()
        self.predictor = self._generate_predictor()

        self.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.MeanAbsoluteError())

    def _generate_offense_vectorizer(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(load.team_vector_size, activation='leaky_relu'),
            tf.keras.layers.Dense(2048, activation='leaky_relu'),
            tf.keras.layers.Dense(512, activation='leaky_relu'),
            tf.keras.layers.Dense(offense_vector_size, activation='linear')
        ])

        return model


    def _generate_defense_vectorizer(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(load.team_vector_size, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(defense_vector_size, activation='linear')
        ])

        return model


    def _generate_meta_vectorizer(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(load.meta_vector_size, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(meta_vector_size, activation='linear')
        ])

        return model


    def _generate_predictor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(predictor_vector_size, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(256, activation='leaky_relu'),
            tf.keras.layers.Dense(load.output_size, activation='linear')
        ])

        return model


    # Does this have to be a tf function
    @tf.function
    def call(self, x):
        offense_vector = self.offense_vectorizer(x[:, :load.team_vector_size])
        defense_vector = self.defense_vectorizer(x[:, load.team_vector_size:(2 * load.team_vector_size)])
        meta_vector = self.meta_vectorizer(x[:, (2 * load.team_vector_size):])

        prediction = self.predictor(tf.concat([offense_vector, defense_vector, meta_vector], 1))
        return prediction


    def save(self):
        self.offense_vectorizer.save_weights(offense_location)
        self.defense_vectorizer.save_weights(defense_location)
        self.meta_vectorizer.save_weights(meta_location)
        self.predictor.save_weights(predictor_location)


    def load(self):
        # I don't know what expect partial means
        self.offense_vectorizer.load_weights(offense_location).expect_partial()
        self.defense_vectorizer.load_weights(defense_location).expect_partial()
        self.meta_vectorizer.load_weights(meta_location).expect_partial()
        self.predictor.load_weights(predictor_location).expect_partial()
