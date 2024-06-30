from .. import load, models
from . import config

data = load.Data(config.year)
model = models.GameModel(config.year, data.team_vector_size, data.meta_vector_size, data.output_size)
model.load_model()

# Batch size must be even for the game accuracy metric to work

model.model.evaluate((data.test_x_offense, data.test_x_defense, data.test_x_meta), data.test_y, batch_size=32)
model.model.evaluate((data.x_offense, data.x_defense, data.x_meta), data.y, batch_size=32)
