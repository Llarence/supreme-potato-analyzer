import tensorflow as tf
import matplotlib.pyplot as plt

import load
import models

model, _, _ = models.create_models()

# models.load_model(model)

history = model.fit((load.x_offense, load.x_defense, load.x_meta),
                    load.y,
                    epochs=30,
                    validation_data=((load.test_x_offense, load.test_x_defense, load.test_x_meta), load.test_y),
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

models.save_model(model)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
