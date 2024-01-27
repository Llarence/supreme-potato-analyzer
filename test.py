import tensorflow as tf

import load
import models

model, (means_model, deviations_model), _ = models.create_models()
models.load_model(model)

means_model.evaluate((load.test_x_offense, load.test_x_defense, load.test_x_meta), load.test_y, batch_size=32)
means_model.evaluate((load.x_offense, load.x_defense, load.x_meta), load.y, batch_size=32)

deviations_model.evaluate((load.test_x_offense, load.test_x_defense, load.test_x_meta), load.test_y, batch_size=32)
deviations_model.evaluate((load.x_offense, load.x_defense, load.x_meta), load.y, batch_size=32)
