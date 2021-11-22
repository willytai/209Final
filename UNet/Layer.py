import enum
import numpy as np

class LayerType(enum.Enum):
    CONV = 1
    DOWN_SAMPLE = 2
    UP_SAMPLE = 3
    CONCAT = 4
    INPUT = 5

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
        self.kernel_size = None
        self.filters = None
        self.strides = None
        self.concatSrc = None
        self.concatDst = None
        self.weights = None
        self.bias = None
        self.inputShape = None

    def setConvParam(self, filters: int, kernel_size: tuple, strides: int, pad: PadType) -> None:
        assert PadType is not None
        self.filters = filters
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

    def setWeigtsBias(self, weights: np.array, bias: np.array) -> None:
        '''
        self.weights.shape = (filter_width, filter_height, filter_depth, # of fiters)
        self.bias.shape = (# of filters, )
        '''
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def __str__(self) -> str:
        return '{}, \'{}\''.format(self.type, self.name)

