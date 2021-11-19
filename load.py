from UNet.model import unet
import sys

model = unet()
model.load_weights(sys.argv[1])

for layer in model.layers:
    if len(layer.get_weights()) > 0 :
        print (layer)
        print (layer.get_weights()[0].shape)
        print (layer.get_weights()[1].shape)
