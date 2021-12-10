from data import *
from model_light import *

testGene = testGenerator("testData")
model = unet()
#model.load_weights("unet_model_light.hdf5")
model.load_weights("./tianwen_light.hdf5")
results = model.predict_generator(testGene,30,verbose=1)
saveResult(npyfile=results, save_path="result_light_8bit")
