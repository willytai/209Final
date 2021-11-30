import numpy as np
from typing import Union
from Layer import Layer, LayerType, PadType
from Utility import *
from . import OutputBuffer as OutputBuffer
from . import InputBuffer as InputBuffer

'''
    1. Stores all the kernel weights and biases for each layer

    2. Stores the current computed feature map

    3. When the data/kernel buffer asks for new data, return the feature map/kernel for the next layer

    - self.layerList stores all the operational layers in order
    - self.layerMap stores the name mapping of each layer
    - self.layerID is the index of the current layer
    - self.featureMapStorage is the storage for feature map, the maximum storage reuqired is
      input_width * input_height * #_of_filters_in_1st_conv
    - self.outputBuffer references the output buffer for convenience
'''
class VMem():
    def __init__(self) -> None:
        self.layerList = list()
        self.layerMap = dict()
        self.layerID = 0
        self.featureMapStorage = None
        self.outputBuffer = None

    def loadInput(self, image: np.array) -> None:
        '''
      ✓ 1. initialize self.featureMapStorage
      ✓ 2. quantize image to 8 bit
      ✓ 3. load to self.featureMapStorage
        '''
        inputDim = self.layerList[0].getInputShape() # from the first layer (input layer)
        filters = self.layerList[1].filters          # from the first conv layer (right after the input layer)
        image_q = quantize8(image, fl=7)
        self.featureMapStorage = np.zeros(shape=(inputDim[0]*inputDim[1]*filters), dtype=np.float32)
        self.write(image_q)

    def requestOutput(self) -> bool:
        '''
      ✓ 1. request output of the next layer from the output buffer
      ✓ 2. return the state
        '''
        assert self.outputBuffer is not None
        ret = self.outputBuffer.vMemWrite(self)
        raise NotImplementedError
        return ret

    def send(self, input_buffer: InputBuffer) -> None:
        '''
      ✓ 1. read next conv to input buffer
            - do the padding here
      ✓ 2. read next kernel to input buffer
      ✓ 3. read next strides and set kernel pos to input buffer
            - reset input_buffer.krnlPos
      ✓ 4. send instruction to the output buffer for buffer initialization
        5. perpare output buffer for post-processing
        '''
        # skip input layer
        if self.layerList[self.layerID].type == LayerType.INPUT:
            self.layerID += 1
        assert self.layerList[self.layerID].type == LayerType.CONV
        targetLayer = self.layerList[self.layerID]
        inputShape = targetLayer.getInputShape()
        inputSize = inputShape[0]*inputShape[1]*inputShape[2]

        # padding stuffs and feature map
        assert targetLayer.pad == PadType.SAME or targetLayer.kernel_size[0] == 1
        padFirst = (targetLayer.kernel_size[0] - 1) // 2
        padLast = (targetLayer.kernel_size[0] - 1) // 2
        if targetLayer.kernel_size[0] % 2 == 0:
            padFirst += 1
        input_buffer.fmBuffer = np.zeros((inputShape[0]+padFirst+padLast, inputShape[1]+padFirst+padLast, inputShape[2]))
        input_buffer.fmBuffer[padFirst:padFirst+inputShape[0],
                              padFirst:padFirst+inputShape[1],
                              :] = self.featureMapStorage[:inputSize].reshape(inputShape)

        # kernel buffer (kernel_x, kernel_y, kernel_z, # kernels)
        input_buffer.krnlBuffer = targetLayer.weights

        # kernel position initialization
        input_buffer.resetKernelPosition(strides=targetLayer.strides,
                                         kernel_num=targetLayer.filters,
                                         kernel_size=targetLayer.kernel_size,
                                         output_shape=targetLayer.outputShape)

        # output buffer initialization (reset)
        self.outputBuffer.resetBuffer(targetLayer.outputShape)

        # queue post-processing actions
        self.outputBuffer.setBiasActivation(bias=targetLayer.bias, activation=targetLayer.activation)
        self.layerID += 1
        while self.layerID < len(self.layerList) and self.layerList[self.layerID].type != LayerType.CONV:
            raise NotImplementedError

    def write(self, data: np.array) -> None:
        '''
        copy data to self.featureMapStorage
        '''
        memcpy(self.featureMapStorage, data.reshape(-1))

    def linkOutputBuffer(self, output_buffer: OutputBuffer) -> None:
        self.outputBuffer = output_buffer

    def addConvLayer(self, name: str, filters: int, activation: str, kernel_size: Union[int,tuple], strides: Union[int,tuple], pad: Union[str,PadType]) -> None:
        kernel_size, strides, pad = self._paramTypeTransfrom(kernel_size, strides, pad)
        newLayer = Layer(LayerType.CONV, name)
        newLayer.setConvParam(filters=filters, activation=activation, kernel_size=kernel_size, strides=strides, pad=pad)
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

    def addDropoutLayer(self, name: str, rate: float) -> None:
        newLayer = Layer(LayerType.DROP_OUT, name)
        newLayer.setDropoutParam(rate=rate)
        self.layerList.append(newLayer)
        self.layerMap[name] = self.layerList[-1]

    def setWeigtsBias(self, layer_name: str, weights: np.array, bias: np.array) -> None:
        self.layerMap[layer_name].setWeigtsBias(weights, bias)

    def getModelInputShape(self) -> tuple:
        return self.layerList[0].getInputShape()

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
