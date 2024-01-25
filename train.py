import tensorflow as tf
import matplotlib.pyplot as plt

import load
import models

def plot(name):
    val_name = 'val_' + name
    plt.plot(history.history[name], label=name)
    plt.plot(history.history[val_name], label=val_name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


model, _, _ = models.create_models()

# models.load_model(model)

history = model.fit((load.x_offense, load.x_defense, load.x_meta),
                    load.y,
                    epochs=1000,
                    validation_data=((load.test_x_offense, load.test_x_defense, load.test_x_meta), load.test_y),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)])

models.save_model(model)

plot('loss')
plot('mae')
