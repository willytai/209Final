import numpy as np
from typing import Union
from UNet.Layer import Layer, LayerType, PadType
from Utility import *
from . import OutputBuffer as OutputBuffer

'''
    1. Stores all the kernel weights and biases for each layer

    2. Stores the current computed feature map

    3. When the data/kernel buffer asks for new data, return the feature map/kernel for the next layer

    - self.layerList stores all the operational layers in order
    - self.layerMap stores the name mapping of each layer
    - self.featureMapStorage is the storage for feature map, the maximum storage reuqired is
      input_width * input_height * #_of_filters_in_1st_conv
    - self.outputBuffer references the output buffer for convenience
'''
class VMem():
    def __init__(self) -> None:
        self.layerList = list()
        self.layerMap = dict()
        self.featureMapStorage = None
        self.outputBuffer = None

    def loadInput(self, image: np.array) -> None:
        '''
      ✓ 1. initialize self.featureMapStorage
      ✓ 2. quantize image to 8 bit
      ✓ 3. load to self.featureMapStorage
        '''
        inputDim = self.layerList[0].inputShape # from the first layer (input layer)
        filters = self.layerList[1].filters     # from the first conv layer (right after the input layer)
        image_q = quantize8(image, fl=7)
        self.featureMapStorage = np.zeros(shape=(inputDim[0]*inputDim[1]*filters), dtype=np.float32)
        memcpy(self.featureMapStorage, image_q.reshape(-1))

    def requestOutput(self) -> bool:
        '''
      ✓ 1. request output of the next layer from the output buffer
      ✓ 2. return the state
        '''
        assert self.outputBuffer is not None
        ret = self.outputBuffer.vMemWrite(self)
        raise NotImplementedError
        return ret

    def linkOutputBuffer(self, output_buffer: OutputBuffer) -> None:
        self.outputBuffer = output_buffer

    def addConvLayer(self, name: str, filters: int, kernel_size: Union[int,tuple], strides: Union[int,tuple], pad: Union[str,PadType]) -> None:
        kernel_size, strides, pad = self._paramTypeTransfrom(kernel_size, strides, pad)
        newLayer = Layer(LayerType.CONV, name)
        newLayer.setConvParam(filters=filters, kernel_size=kernel_size, strides=strides, pad=pad)
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

    def getModelInputShape(self) -> tuple:
        return self.layerList[0].inputShape

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
