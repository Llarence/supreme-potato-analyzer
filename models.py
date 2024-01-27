import tensorflow as tf
import tensorflow_probability as tfp

import load
import config

model_location = f'data/{config.year}/model/'

@tf.function
def to_scale(x):
    # The 0.02 will prevent nans if log_prob is used
    return 0.02 + tf.math.softplus(x)

@tf.function
def prob_loss(y, pred_y):
    return pred_y.tensor_distribution.prob(y)


# Batch size has to be even and unshuffled for this to work because it relies
#  on how the matches are stored (blue score then red score)
# Even when they data is shuffled it still will kinda work
@tf.function
def outcome_accuracy(y, pred_y):
    # For load matches come in pairs blue then red
    blue_indices = tf.range(0, len(y), delta=2)
    red_indices = blue_indices + 1

    blue_y = tf.gather(y, blue_indices)
    red_y = tf.gather(y, red_indices)

    blue_pred_y = tf.round(tf.gather(pred_y, blue_indices))
    red_pred_y = tf.round(tf.gather(pred_y, red_indices))

    blue_wins = blue_y > red_y
    blue_pred_wins = blue_pred_y > red_pred_y

    ties = blue_y == red_y
    pred_ties = blue_pred_y == red_pred_y

    return tf.logical_or(blue_wins == blue_pred_wins, tf.logical_and(ties, pred_ties))


def create_game_analyzer(offense_inp, defense_inp, meta_inp):
    offense = tf.keras.layers.Dense(4, activation='linear', use_bias=False)(offense_inp)
    defense = tf.keras.layers.Dense(4, activation='linear', use_bias=False)(defense_inp)
    meta = tf.keras.layers.Dense(2, activation='linear')(meta_inp)

    analyzer = tf.keras.layers.Concatenate()([offense, defense, meta])
    analyzer = tf.keras.layers.BatchNormalization()(analyzer)
    analyzer = tf.keras.layers.Dense(256, activation='tanh')(analyzer)
    analyzer = tf.keras.layers.Dropout(0.4)(analyzer)
    analyzer = tf.keras.layers.Dense(256, activation='linear')(analyzer)

    return analyzer


def create_means(offense_inp, defense_inp, meta_inp):
    means = []
    for i in range(load.output_size):
        analyzer = create_game_analyzer(offense_inp, defense_inp, meta_inp)

        mean = tf.keras.layers.Dense(256, activation='linear')(analyzer)
        mean = tf.keras.layers.Dropout(0.4)(mean)
        mean = tf.keras.layers.Dense(256, activation='linear')(mean)
        mean = tf.keras.layers.Dropout(0.4)(mean)
        mean = tf.keras.layers.Dense(1, activation='linear')(mean)

        means.append(mean)

    return means


def create_deviations(offense_inp, defense_inp, meta_inp):
    deviations = []
    for i in range(load.output_size):
        analyzer = create_game_analyzer(offense_inp, defense_inp, meta_inp)

        deviation = tf.keras.layers.Dense(256, activation='linear')(analyzer)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.Dense(256, activation='linear')(deviation)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.Dense(1, activation='linear')(deviation)

        deviations.append(deviation)

    return deviations


def create_models():
    offense_inp = tf.keras.layers.Input(shape=(load.team_vector_size,))
    defense_inp = tf.keras.layers.Input(shape=(load.team_vector_size,))
    meta_inp = tf.keras.layers.Input(shape=(load.meta_vector_size,))

    means = create_means(offense_inp, defense_inp, meta_inp)
    deviations = create_deviations(offense_inp, defense_inp, meta_inp)

    means_output = tf.keras.layers.Concatenate()(means)
    deviations_output = tf.keras.layers.Concatenate()(deviations)
    deviations_output = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=0, 
                                           scale=to_scale(x)))(deviations_output)

    means_model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [means_output])
    means_model.compile(optimizer=tf.optimizers.Adam(), loss='mse', metrics=['mae', outcome_accuracy])
    
    deviations_model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [deviations_output])
    deviations_model.compile(optimizer=tf.optimizers.Adam(), loss=prob_loss)

    outputs = tf.keras.layers.Concatenate()(means + deviations)
    outputs = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=x[:, :load.output_size], 
                                           scale=to_scale(x[:, load.output_size:])))(outputs)

    model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [outputs])

    return model, means_model, deviations_model


def save_model(model):
    model.save_weights(model_location)


def load_model(model):
    model.load_weights(model_location).expect_partial()
