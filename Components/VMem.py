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
      input_width * input_height * #_of_filters_in_1st_conv * 2
    - self.outputBuffer references the output buffer for convenience
    - self.concatCandidates is a list of layers that need to be saved for concatenation in the future
'''
class VMem():
    def __init__(self, verbose: int = 0) -> None:
        self.layerList = list()
        self.layerMap = dict()
        self.layerID = 0
        self.featureMapStorage = None
        self.outputBuffer = None
        self.concatCandidates = list()
        self.verbose = verbose

    def loadInput(self, image: np.array) -> None:
        '''
      ✓ 1. initialize self.featureMapStorage
      ✓ 2. quantize image to 8 bit
      ✓ 3. load to self.featureMapStorage
        '''
        inputDim = self.layerList[0].getInputShape() # from the first layer (input layer)
        filters = self.layerList[1].filters          # from the first conv layer (right after the input layer)
        image_q = Quantizer.getInstance().quantize(image)
        self.featureMapStorage = np.zeros(shape=(inputDim[0]*inputDim[1]*filters*2), dtype=np.float32)
        self.write(image_q)

    def requestOutput(self) -> bool:
        '''
      ✓ 1. request output of the next layer from the output buffer
      ✓ 2. return the state
        '''
        assert self.outputBuffer is not None
        ret = self.outputBuffer.vMemWrite(self)
        return ret

    def send(self, input_buffer: InputBuffer) -> None:
        '''
      ✓ 1. read next conv to input buffer
            - do the padding here
      ✓ 2. read next kernel to input buffer
      ✓ 3. read next strides and set kernel pos to input buffer
            - reset input_buffer.krnlPos
      ✓ 4. send instruction to the output buffer for buffer initialization
      ✓ 5. perpare output buffer for post-processing
        '''
        # skip input layer
        if self.layerList[self.layerID].type == LayerType.INPUT:
            print ('[{}/{}] processing layer: {}'.format(self.layerID+1, len(self.layerList), self.layerList[self.layerID]))
            self.layerID += 1
        assert self.layerList[self.layerID].type == LayerType.CONV

        targetLayer = self.layerList[self.layerID]
        print ('[{}/{}] processing layer: {}'.format(self.layerID+1, len(self.layerList), targetLayer))

        inputShape = targetLayer.getInputShape()
        inputSize = inputShape[0]*inputShape[1]*inputShape[2]

        # padding stuffs and feature map
        assert targetLayer.pad == PadType.SAME or targetLayer.kernel_size[0] == 1
        padFirst = (targetLayer.kernel_size[0] - 1) // 2
        padLast = (targetLayer.kernel_size[0] - 1) // 2
        if targetLayer.kernel_size[0] % 2 == 0:
            padLast += 1
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

        # post-processing actions
        # bias & activation
        self.outputBuffer.setBiasActivation(bias=targetLayer.bias, activation=targetLayer.activation)

        # save intermediate feature map for concatenation
        if targetLayer.name in self.concatCandidates:
            self.outputBuffer.setSaveResidual()

        # pooling or upsampling and concatenation
        self.layerID += 1
        while self.layerID < len(self.layerList) and self.layerList[self.layerID].type != LayerType.CONV:
            targetLayer = self.layerList[self.layerID]
            print ('[{}/{}] processing layer: {}'.format(self.layerID+1, len(self.layerList), targetLayer))
            if targetLayer.type == LayerType.DOWN_SAMPLE:
                self.outputBuffer.setPooling(targetLayer.kernel_size)
            elif targetLayer.type == LayerType.UP_SAMPLE:
                self.outputBuffer.setUpsample(targetLayer.kernel_size)
            elif targetLayer.type == LayerType.CONCAT:
                self.outputBuffer.setConcate()
            elif targetLayer.type == LayerType.DROP_OUT:
                self.outputBuffer.setDropoutRatio(targetLayer.dropRate)
                if targetLayer.name in self.concatCandidates:
                    self.outputBuffer.setSaveResidual()
            else:
                raise NotImplementedError('unsupported layer type: {}'.format(targetLayer.type))
            self.layerID += 1

        # stop or not
        if self.layerID == len(self.layerList):
            self.outputBuffer.setFinalRound()

        input_buffer.krnlPos.resetStatusLine()

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
        self.concatCandidates.append(dst)
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
        if self.verbose != 0:
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
