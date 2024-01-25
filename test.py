import tensorflow as tf

import load
import models

model, _, _ = models.create_models()
models.load_model(model)

def test(x, y):
    model.evaluate(x, y)

    pred_y = model(x).mean()

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

    accurate_preds = tf.logical_or(blue_wins == blue_pred_wins, tf.logical_and(ties, pred_ties))
    print(tf.reduce_mean(tf.cast(accurate_preds, tf.float32)).numpy())


test((load.test_x_offense, load.test_x_defense, load.test_x_meta), load.test_y)
test((load.x_offense, load.x_defense, load.x_meta), load.y)
