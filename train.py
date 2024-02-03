import tensorflow as tf
import matplotlib.pyplot as plt

from . import load, models
from .test import config

def plot_metric(history, name):
    val_name = 'val_' + name
    plt.plot(history.history[name], label=name)
    plt.plot(history.history[val_name], label=val_name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def train(model, data, plot=False):
    history = model.model.fit(
        (data.x_offense, data.x_defense, data.x_meta),
        data.y,
        epochs=10000,
        batch_size=32,
        validation_data=((data.test_x_offense,
                          data.test_x_defense,
                          data.test_x_meta), data.test_y),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=15,
                                                    restore_best_weights=True)]
    )

    if plot:
        plot_metric(history, 'loss')
        plot_metric(history, 'prob_mae')
        plot_metric(history, 'outcome_accuracy')


if __name__ == '__main__':
    data = load.Data(config.year)

    model = models.GameModel(config.year, data.team_vector_size, data.meta_vector_size, data.output_size)
    train(model, data, True)

    model.save_model()
