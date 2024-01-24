import load
import models

predictor = models.Predictor()

predictor.fit(load.x, load.y, epochs=5)

predictor.save()
