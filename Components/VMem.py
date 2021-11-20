import numpy as np
from typing import Union
from UNet.Layer import Layer, LayerType, PadType

'''
    1. Stores all the kernel weights and biases for each layer

    2. Stores the current computed feature map

    3. When the data/kernel buffer asks for new data, return the feature map/kernel for the next layer

    - self.layerList stores all the operational layers in order
    - self.layerMap stores the name mapping of each layer
    - self.featureMapStorage is the storage for feature map, the maximum storage reuqired is
      input_width * input_height * 64 (the # of filters for the first convolutional layer)
'''
class VMem():
    def __init__(self, input_dim: tuple) -> None:
        self.layerList = list()
        self.layerMap = dict()
        self.featureMapStorage = np.zeros(shape=(input_dim[0]*input_dim[1]*64), dtype=np.float32)

    def addConvLayer(self, name: str, kernel_size: Union[int,tuple], strides: Union[int,tuple], pad: Union[str,PadType]) -> None:
        kernel_size, strides, pad = self._paramTypeTransfrom(kernel_size, strides, pad)
        newLayer = Layer(LayerType.CONV, name)
        newLayer.setConvParam(kernel_size=kernel_size, strides=strides, pad=pad)
        self.layerList.append(newLayer)
        self.layerMap[name] = self.layerList[-1]

    def addMaxPoolingLayer(self, name: str, kernel_size: Union[int,tuple], strides: Union[int,tuple], pad: Union[str,PadType]) -> None:
        kernel_size, strides, pad = self._paramTypeTransfrom(kernel_size, strides, pad)
        newLayer = Layer(LayerType.DOWN_SAMPLE, name)
        newLayer.setMaxPoolingParam(kernel_size=kernel_size, strides=strides, pad=pad)
        self.layerList.append(newLayer)
        self.layerMap[name] = self.layerList[-1]

    def addUpSamplingLayer(self, name: str, kernel_size: Union[int,tuple]) -> None:
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        newLayer = Layer(LayerType.UP_SAMPLE, name)
        newLayer.setUpSamplingParam(kernel_size=kernel_size)
        self.layerList.append(newLayer)
        self.layerMap[name] = self.layerList[-1]

    def addConcatLayer(self, name: str, src: str, dst: str) -> None:
        '''
        Concatenate 'src' to 'dst'
        'src' precede 'dst' in the third dimension
        '''
        newLayer = Layer(LayerType.CONCAT, name)
        newLayer.setConcatParam(src=src, dst=dst)
        self.layerList.append(newLayer)
        self.layerMap[name] = self.layerList[-1]

    def addInputLayer(self, name: str, input_shape: tuple) -> None:
        newLayer = Layer(LayerType.INPUT, name)
        newLayer.setInputParam(input_shape=input_shape)
        self.layerList.append(newLayer)
        self.layerMap[name] = self.layerList[-1]

    def setWeigtsBias(self, layer_name: str, weights: np.array, bias: np.array) -> None:
        self.layerMap[layer_name].setWeigtsBias(weights, bias)

    def layerStat(self):
        for layer in self.layerList: print (layer)

    def _paramTypeTransfrom(self, kernel_size: Union[int,tuple], strides: Union[int,tuple], pad: Union[str,PadType]) -> tuple:
        if type(strides) == tuple:
            assert strides[0] == strides[1]
            strides = strides[0]
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        if type(pad) == str:
            padStr = str(pad)
            pad = PadType.NONE
            pad = PadType.SAME if padStr == 'same' else pad
            pad = PadType.VALID if padStr == 'valid' else pad

        return (kernel_size, strides, pad)
