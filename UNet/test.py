from data import *
from model import *

testGene = testGenerator("testData")
model = unet()
model.load_weights("unet_model.hdf5")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("result",results)
