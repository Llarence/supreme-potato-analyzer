import tensorflow as tf
import tensorflow_probability as tfp

import load
import config

model_location = f'data/{config.year}/model/'

def create_models():
    offense_inp = tf.keras.layers.Input(shape=(load.team_vector_size,))
    offense = tf.keras.layers.Dense(128, activation='linear', use_bias=False)(offense_inp)
    offense = tf.keras.layers.Dense(4, activation='linear', use_bias=False)(offense)

    defense_inp = tf.keras.layers.Input(shape=(load.team_vector_size,))
    defense = tf.keras.layers.Dense(128, activation='linear', use_bias=False)(defense_inp)
    defense = tf.keras.layers.Dense(2, activation='linear', use_bias=False)(defense)

    meta_inp = tf.keras.layers.Input(shape=(load.meta_vector_size,))
    meta = tf.keras.layers.Dense(128, activation='leaky_relu')(meta_inp)
    meta = tf.keras.layers.Dense(1, activation='leaky_relu')(meta)

    analyzer = tf.keras.layers.Concatenate()([offense, defense, meta])
    analyzer = tf.keras.layers.Dense(128, activation='leaky_relu')(analyzer)
    analyzer = tf.keras.layers.Dense(128, activation='leaky_relu')(analyzer)
    analyzer = tf.keras.layers.Dense(128, activation='leaky_relu')(analyzer)
    analyzer = tf.keras.layers.Dense(64, activation='leaky_relu')(analyzer)

    means = []
    deviations = []
    for i in range(load.output_size):
        output = tf.keras.layers.Dense(64, activation='leaky_relu')(analyzer)
        output = tf.keras.layers.Dense(32, activation='leaky_relu')(output)

        mean = tf.keras.layers.Dense(32, activation='leaky_relu')(output)
        mean = tf.keras.layers.Dense(32, activation='leaky_relu')(mean)
        mean = tf.keras.layers.Dense(1, activation='linear')(mean)

        deviation = tf.keras.layers.Dense(32, activation='leaky_relu')(output)
        deviation = tf.keras.layers.Dense(32, activation='leaky_relu')(deviation)
        deviation = tf.keras.layers.Dense(1, activation='linear')(deviation)

        means.append(mean)
        deviations.append(deviation)

    outputs = tf.keras.layers.Concatenate()(means + deviations)
    outputs = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=x[:, :load.output_size], 
                                           scale=tf.math.softplus(x[:, load.output_size:])))(outputs)

    model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [outputs])

    model.compile(optimizer=tf.optimizers.Adam(), loss=lambda y, pred_y: -pred_y.log_prob(y))

    return model, tf.keras.Model([offense_inp], [offense]), tf.keras.Model([defense_inp], [defense])


def save_model(model):
    model.save_weights(model_location)


def load_model(model):
    model.load_weights(model_location).expect_partial()
