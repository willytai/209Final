import enum
import numpy as np

class LayerType(enum.Enum):
    CONV = 1
    DOWN_SAMPLE = 2
    UP_SAMPLE = 3
    CONCAT = 4
    INPUT = 5
    DROP_OUT = 6

class PadType(enum.Enum):
    SAME = 1
    VALID = 2
    NONE = 3  # for non-convolutional layers

'''
    The layer class
    To make it independent of Keras
    Only storing the information need for the uArch
'''
class Layer():
    def __init__(self, layer_type: LayerType, name: str) -> None:
        self.type = layer_type
        self.name = name
        self.pad = None
        self.activation = None
        self.kernel_size = None
        self.filters = None
        self.strides = None
        self.concatSrc = None
        self.concatDst = None
        self.dropRate = None
        self.weights = None
        self.bias = None
        self.inputShape = None # a list of input shapes that the layer takes
        self.outputShape = None

    def setConvParam(self, filters: int, activation: str, kernel_size: tuple, strides: int, pad: PadType) -> None:
        assert PadType is not None
        self.filters = filters
        self.activation = activation
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad

    def setMaxPoolingParam(self, kernel_size: tuple, strides: int, pad: PadType) -> None:
        assert PadType is not None
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad

    def setUpSamplingParam(self, kernel_size: tuple) -> None:
        self.kernel_size = kernel_size

    def setConcatParam(self, src: str, dst: str) -> None:
        self.concatSrc = src
        self.concatDst = dst

    def setInputParam(self, input_shape: tuple) -> None:
        '''
        remove the first dimension (batch size)
        '''
        if len(input_shape) == 4:
            input_shape = (input_shape[1], input_shape[2], input_shape[3])
        self.inputShape = input_shape

    def setDropoutParam(self, rate: float) -> None:
        self.dropRate = rate

    def setWeigtsBias(self, weights: np.array, bias: np.array) -> None:
        '''
        self.weights.shape = (filter_width, filter_height, filter_depth, # of fiters)
        self.bias.shape = (# of filters, )
        '''
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def getInputShape(self) -> tuple:
        assert len(self.inputShape) == 1
        return self.inputShape[0]

    def computeShape(self, layer_map: dict, input_shape: tuple) -> tuple:
        '''
        compute output shape according to input shape
        '''
        self.inputShape = [input_shape]
        if self.type == LayerType.INPUT:
            self.outputShape = tuple(self.inputShape[0])
        elif self.type == LayerType.CONV:
            self.outputShape = self._computeConvShape(input_shape)
        elif self.type == LayerType.DOWN_SAMPLE:
            self.outputShape = self._computeDownSampleShape(input_shape)
        elif self.type == LayerType.UP_SAMPLE:
            self.outputShape = self._computeUpSampleShape(input_shape)
        elif self.type == LayerType.CONCAT:
            self.outputShape = self._computeConcatShape(layer_map)
        elif self.type == LayerType.DROP_OUT:
            self.outputShape = input_shape
        else:
            assert False, 'unrecognized type: {}'.format(self.type)
        return self.outputShape

    def _computeConvShape(self, input_shape: tuple) -> tuple:
        assert self.kernel_size[0] == self.kernel_size[1], 'only support square kernels, {}'.format(self)
        assert self.pad == PadType.SAME or self.kernel_size[0] == 1, 'only support zero padding for conv layers with kernel size greater than 1: {}'.format(self)
        return (input_shape[0] // self.strides, input_shape[1] // self.strides, self.filters)

    def _computeDownSampleShape(self, input_shape: tuple) -> tuple:
        assert self.kernel_size[0] == self.kernel_size[1], 'only support square kernels'
        assert self.pad == PadType.VALID, 'only support valid padding for max pooling layers'
        return (input_shape[0] // self.strides, input_shape[1] // self.strides, input_shape[2])

    def _computeUpSampleShape(self, input_shape: tuple) -> tuple:
        assert self.kernel_size[0] == self.kernel_size[1], 'only support square kernels'
        return (input_shape[0]*self.kernel_size[0], input_shape[1]*self.kernel_size[1], input_shape[2])

    def _computeConcatShape(self, layer_map: dict) -> tuple:
        o1 = layer_map[self.concatSrc].outputShape
        o2 = layer_map[self.concatDst].outputShape
        assert o1[0] == o2[0] and o1[1] == o2[1], 'center crop concatenation not supported'
        self.inputShape = [o1, o2]
        return (o1[0], o1[1], o1[2]+o2[2])

    def __str__(self) -> str:
        return '{}, \'{}\', input: {}, output: {}'.format(self.type, self.name, self.inputShape, self.outputShape)

