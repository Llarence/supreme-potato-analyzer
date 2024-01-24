import tensorflow as tf
import matplotlib.pyplot as plt

import load
import models

predictor = models.Predictor()

# predictor.load()

history = predictor.fit(load.x,
                        load.y,
                        epochs=10,
                        validation_data=(load.test_x, load.test_y),
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)])

predictor.save()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
