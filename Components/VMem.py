import numpy as np
from typing import Union
from UNet.Layer import Layer, PadType

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
        pass

    def addMaxPoolingLayer(self, name: str, kernel_size: Union[int,tuple], strides: Union[int,tuple], pad: Union[str,PadType]) -> None:
        pass

    def addUpSamplingLayer(self, name: str, kernel_size: Union[int,tuple]) -> None:
        pass

    def addConcatLayer(self, name: str, src: str, dst: str) -> None:
        '''
        Concatenate 'src' to 'dst'
        'src' precede 'dst' in the third dimension
        '''
        pass

    def addInputLayer(self, name: str, input_shape: tuple) -> None:
        pass
