import enum

class LayerType(enum.Enum):
    CONV = 1
    DOWN_SAMPLE = 2
    UP_SAMPLE = 3

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
    def __init__(self, layer_type: LayerType) -> None:
        self.type = layer_type
        self.pad = None
        self.kernel_size = None
        self.strid = None
        self.concatSrc = None
        self.concatDst = None
        self.weights = None
        self.bias = None

    def setConvParam(self, kernel_size: tuple, stride: int, pad: PadType) -> None:
        assert PadType is not None
        self.kernel_size = kernel_size
        self.strid = stride
        self.pad = pad

    def setMaxPoolingParam(self, kernel_size: tuple, stride: int, pad: PadType) -> None:
        assert PadType is not None
        self.kernel_size = kernel_size
        self.strid = stride
        self.pad = pad

    def setUpSamplingParam(self, kernel_size: tuple) -> None:
        self.kernel_size = kernel_size

    def setConcatParam(self, src: str, dst: str) -> None:
        self.concatSrc = src
        self.concatDst = dst
