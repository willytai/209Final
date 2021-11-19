from Components.VMem import VMem
from Components.ComputationUnit import ComputationUnit
from tensorflow.keras.models import model_from_json
import numpy as np

'''
    Contains ...
    Loading model and weights with the keras API fow now
'''
class uArch():
    def __init__(self, pe_array_size: int = 32, input_dim: tuple = (256, 256)) -> None:
        self.vMem = VMem(input_dim=input_dim)
        self.computationalUnit = ComputationUnit(pe_array_size=pe_array_size)

        # just for convenience
        self.model = None

    def loadModel(self, model_path: str) -> None:
        with open(model_path, 'r') as f:
            self.model = model_from_json(f.read())
        print ('model successfully read from {}'.format(model_path))

    def loadWeight(self, weight_path: str) -> None:
        self.model.load_weights(weight_path)
        print ('weights successfully loaded from {}'.format(weight_path))
        self._loadVMem()

    def run(self, input_path: str) -> np.array:
        pass

    def _loadVMem(self) -> None:
        '''
        Load the desired layers and their corresponding weights to the VMem
        Only Conv, MaxPool, UpSampling, Concat layers are recorded

      ✓ 1. Create Layer class via the VMem API
        2. Copy the weights and biases
        3. Release the memory used by self.model
        '''
        for layer in self.model.layers:
            layerConfig = layer.get_config()
            layerClassName = layer.__class__.__name__
            if layerClassName == 'Conv2D':
                self.vMem.addConvLayer(name=layerConfig['name'], kernel_size=layerConfig['kernel_size'], strides=layerConfig['strides'], pad=layerConfig['padding'])
            elif layerClassName == 'MaxPooling2D':
                self.vMem.addMaxPoolingLayer(name=layerConfig['name'], kernel_size=layerConfig['pool_size'], strides=layerConfig['strides'], pad=layerConfig['padding'])
            elif layerClassName == 'UpSampling2D':
                self.vMem.addUpSamplingLayer(name=layerConfig['name'], kernel_size=layerConfig['size'])
            elif layerClassName == 'Concatenate':
                self.vMem.addConcatLayer(name=layerConfig['name'], src=layer.input[1].name.split('/')[0], dst=layer.input[0].name.split('/')[0])
            elif layerClassName == 'InputLayer':
                self.vMem.addInputLayer(name=layerConfig['name'], input_shape=layerConfig['batch_input_shape'])
            else:
                print ('skipping {} ({}) layer'.format(layerClassName, layerConfig['name']))
