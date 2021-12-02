from data import *
from model_light import *

testGene = testGenerator("testData")
model = unet()
model.load_weights("unet_model_light.hdf5")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("result",results)
