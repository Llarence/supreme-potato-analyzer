import tensorflow as tf

import load
import models

predictor = models.Predictor()
predictor.load()

def test(x, y):
    pred_y = predictor.predict(x)[:, 0]

    # For load matches come in pairs blue then red
    blue_indices = tf.range(0, len(x), delta=2)
    red_indices = blue_indices + 1

    blue_y = tf.gather(y, blue_indices)
    red_y = tf.gather(y, red_indices)

    blue_pred_y = tf.gather(pred_y, blue_indices)
    red_pred_y = tf.gather(pred_y, red_indices)

    # This doesn't account for ties, but since the model should never
    #  predict a tie it evens out
    blue_wins = blue_y > red_y
    blue_pred_wins = blue_pred_y > red_pred_y

    accurate_preds = blue_wins == blue_pred_wins
    print(tf.reduce_mean(tf.cast(accurate_preds, tf.float32)).numpy())


test(load.test_x, load.test_y)
test(load.x, load.y)
