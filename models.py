import tensorflow as tf
import tensorflow_probability as tfp

from . import load, config

model_location = f'data/{config.year}/model/'

@tf.function
def to_scale(x):
    # The 0.02 will prevent nans if log_prob is used
    return 0.02 + tf.math.softplus(x)


@tf.function
def prob_loss(y, pred_y):
    return -pred_y.log_prob(y)


# Batch size has to be even and unshuffled for this to work because it relies
#  on how the matches are stored (blue score then red score)
# Even when they data is shuffled it still will kinda work
@tf.function
def outcome_accuracy(y, pred_y):
    # The first value is the total points
    y = y[:, 0]
    pred_y = pred_y[:, 0]

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
    analyzer = tf.keras.layers.Dense(128, activation='tanh')(analyzer)
    analyzer = tf.keras.layers.Dropout(0.4)(analyzer)
    analyzer = tf.keras.layers.Dense(128, activation='tanh')(analyzer)
    analyzer = tf.keras.layers.Dropout(0.4)(analyzer)
    analyzer = tf.keras.layers.Dense(128, activation='leaky_relu')(analyzer)
    analyzer = tf.keras.layers.Dropout(0.4)(analyzer)
    analyzer = tf.keras.layers.Dense(128, activation='leaky_relu')(analyzer)
    analyzer = tf.keras.layers.Dropout(0.4)(analyzer)
    analyzer = tf.keras.layers.Dense(256, activation='linear')(analyzer)

    return analyzer, offense, defense, meta


def create_means(offense_inp, defense_inp, meta_inp):
    means = []
    offenses = []
    defenses = []
    metas = []
    for i in range(load.output_size):
        analyzer, offense, defense, meta = create_game_analyzer(offense_inp, defense_inp, meta_inp)
        offenses.append(offense)
        defenses.append(defense)
        metas.append(meta)

        mean = tf.keras.layers.Dense(256, activation='linear')(analyzer)
        mean = tf.keras.layers.Dropout(0.4)(mean)
        mean = tf.keras.layers.Dense(256, activation='linear')(mean)
        mean = tf.keras.layers.Dropout(0.4)(mean)
        mean = tf.keras.layers.Dense(1, activation='linear')(mean)

        means.append(mean)

    return means, offenses, defenses, metas


def create_deviations(offense_inp, defense_inp, meta_inp):
    deviations = []
    offenses = []
    defenses = []
    metas = []
    for i in range(load.output_size):
        analyzer, offense, defense, meta = create_game_analyzer(offense_inp, defense_inp, meta_inp)
        offenses.append(offense)
        defenses.append(defense)
        metas.append(meta)

        deviation = tf.keras.layers.Dense(128, activation='linear')(analyzer)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.BatchNormalization()(deviation)
        deviation = tf.keras.layers.Dense(128, activation='tanh')(deviation)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.Dense(128, activation='tanh')(deviation)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.Dense(64, activation='linear')(deviation)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.Dense(64, activation='linear')(deviation)
        deviation = tf.keras.layers.Dropout(0.4)(deviation)
        deviation = tf.keras.layers.Dense(1, activation='linear')(deviation)

        deviations.append(deviation)

    return deviations, offenses, defenses, metas


def create_models():
    offense_inp = tf.keras.layers.Input(shape=(load.team_vector_size,))
    defense_inp = tf.keras.layers.Input(shape=(load.team_vector_size,))
    meta_inp = tf.keras.layers.Input(shape=(load.meta_vector_size,))

    means, mean_offenses, mean_defenses, mean_metas \
        = create_means(offense_inp, defense_inp, meta_inp)
    deviations, deviation_offenses, deviation_defenses, deviation_metas \
        = create_deviations(offense_inp, defense_inp, meta_inp)
    
    mean_offense_vectorizers = [tf.keras.Model([offense_inp], [mean_offense])
                                for mean_offense in mean_offenses]
    mean_defense_vectorizers = [tf.keras.Model([defense_inp], [mean_defense])
                                for mean_defense in mean_defenses]
    mean_meta_vectorizers = [tf.keras.Model([meta_inp], [mean_meta])
                             for mean_meta in mean_metas]
    
    deviation_offense_vectorizers = [tf.keras.Model([offense_inp], [deviation_offense])
                                     for deviation_offense in deviation_offenses]
    deviation_defense_vectorizers = [tf.keras.Model([defense_inp], [deviation_defense])
                                     for deviation_defense in deviation_defenses]
    deviation_meta_vectorizers = [tf.keras.Model([meta_inp], [deviation_meta])
                                  for deviation_meta in deviation_metas]

    means_output = tf.keras.layers.Concatenate()(means)
    deviations_output = tf.keras.layers.Concatenate()(deviations)
    deviations_output = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=0, 
                                           scale=to_scale(x)))(deviations_output)

    means_model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [means_output])
    means_model.compile(optimizer=tf.optimizers.Adam(), loss='mse', metrics=['mae', outcome_accuracy])
    
    deviations_model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [deviations_output])
    deviations_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss=prob_loss)

    outputs = tf.keras.layers.Concatenate()(means + deviations)
    outputs = tfp.layers.DistributionLambda(
        lambda x: tfp.distributions.Normal(loc=x[:, :load.output_size], 
                                           scale=to_scale(x[:, load.output_size:])))(outputs)

    model = tf.keras.Model([offense_inp, defense_inp, meta_inp], [outputs])

    return model, \
        (means_model, deviations_model), \
        ((mean_offense_vectorizers, mean_defense_vectorizers, mean_meta_vectorizers), \
            (deviation_offense_vectorizers, deviation_defense_vectorizers, deviation_meta_vectorizers))


def save_model(model):
    model.save_weights(model_location)


def load_model(model):
    model.load_weights(model_location).expect_partial()
