import tensorflow as tf

import load
import models

predictor = models.Predictor()
predictor.load()

pred_y = predictor.predict(load.x)[:, 0]

# For load matches come in pairs blue then red
blue_indices = tf.range(0, len(load.x), delta=2)
red_indices = blue_indices + 1

blue_y = tf.gather(load.y, blue_indices)
red_y = tf.gather(load.y, red_indices)

blue_pred_y = tf.gather(pred_y, blue_indices)
red_pred_y = tf.gather(pred_y, red_indices)

# This doesn't account for ties, but since the model should never
#  predict a tie it evens out
blue_wins = blue_y > red_y
blue_pred_wins = blue_pred_y > red_pred_y

accurate_preds = blue_wins == blue_pred_wins
print(tf.reduce_mean(tf.cast(accurate_preds, tf.float32)).numpy())

predictor.evaluate(load.x, load.y)
