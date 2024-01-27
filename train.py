import tensorflow as tf
import matplotlib.pyplot as plt

import load
import models

def plot(history, name):
    val_name = 'val_' + name
    plt.plot(history.history[name], label=name)
    plt.plot(history.history[val_name], label=val_name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


model, means_model, deviations_model = models.create_models()

# models.load_model(model)

history1 = means_model.fit((load.x_offense, load.x_defense, load.x_meta),
                           load.y,
                           batch_size=32,
                           epochs=10000,
                           validation_data=((load.test_x_offense,
                                             load.test_x_defense,
                                             load.test_x_meta), load.test_y),
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=25,
                                                                       restore_best_weights=True)])

models.save_model(model)

errors = means_model((load.x_offense, load.x_defense, load.x_meta)) - load.y
test_errors = means_model((load.test_x_offense, load.test_x_defense, load.test_x_meta)) - load.test_y

history2 = deviations_model.fit((load.x_offense, load.x_defense, load.x_meta),
                                load.y,
                                batch_size=32,
                                epochs=10000,
                                validation_data=((load.test_x_offense,
                                                  load.test_x_defense,
                                                  load.test_x_meta), load.test_y),
                                callbacks=[tf.keras.callbacks.EarlyStopping(patience=25,
                                                                            restore_best_weights=True)])

models.save_model(model)

plot(history1, 'loss')
plot(history1, 'mae')
plot(history1, 'outcome_accuracy')
plot(history2, 'loss')
