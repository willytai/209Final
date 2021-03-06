from Components.VMem import VMem
from Components.ComputationUnit import ComputationUnit
from Components.OutputBuffer import OutputBuffer
from Components.InputBuffer import InputBuffer
from Utility import Quantizer
from tensorflow.keras.models import model_from_json
import numpy as np
import skimage.io as io
import skimage.transform as trans

'''
Loading model and weights with the keras API fow now
'''
class uArch():
    def __init__(self, pe_array_size: int = 32, verbose: int = 0) -> None:
        self.vMem = VMem(verbose)
        self.outputBuffer = OutputBuffer()
        self.computationUnit = ComputationUnit(pe_array_size=pe_array_size)
        self.inputBuffer = InputBuffer()
        self.finalOutputShape = None

        # link the components
        self.vMem.linkOutputBuffer(self.outputBuffer)
        self.outputBuffer.linkComputationUnit(self.computationUnit)
        self.computationUnit.linkInputBuffer(self.inputBuffer)
        self.inputBuffer.linkVMem(self.vMem)

        # indicates whether the inference loop has completed
        self.done = False

        # just for convenience
        self.model = None

    def showUsage(self) -> None:
        self.computationUnit.usage()

    def loadModel(self, model_path: str) -> None:
        '''
        load model structure
        '''
        with open(model_path, 'r') as f:
            self.model = model_from_json(f.read())
        print ('model successfully read from {}'.format(model_path))

    def loadWeight(self, weight_path: str) -> None:
        '''
        load model weight and bias
        '''
        self.model.load_weights(weight_path)
        print ('weights successfully loaded from {}'.format(weight_path))
        self._loadVMem()

    def setComputationMode(self, word_length: int) -> None:
        if word_length != -1:
            self.outputBuffer.setQuantize()
            Quantizer.getInstance().setWordLength(word_length)

    def run(self, input_path: str) -> np.array:
        '''
      ✓ 1. read input image
      ✓ 2. place it in vMem
      ✓ 3. start inference loop
      ✓ 4. return the result
        '''
        image = self._readInput(input_path)
        self.vMem.loadInput(image)

        while not self.done:
            self.done = self.vMem.requestOutput()

        size = self.finalOutputShape[0]*self.finalOutputShape[1]*self.finalOutputShape[2]
        return self.vMem.featureMapStorage[:size].reshape((1,)+self.finalOutputShape)

    def _loadVMem(self) -> None:
        '''
        Load the desired layers and their corresponding weights to the VMem
        Only Conv, MaxPool, UpSampling, Concat layers are recorded

      ✓ 1. Add Layer class via the VMem API
      ✓ 2. Copy the weights and biases
      ✓ 3. Release the memory used by self.model
      ✓ 4. compute input/output shape for all layers
        '''
        for layer in self.model.layers:
            layerConfig = layer.get_config()
            layerClassName = layer.__class__.__name__
            if layerClassName == 'Conv2D':
                self.vMem.addConvLayer(name=layerConfig['name'],
                                       filters=layerConfig['filters'],
                                       activation=layerConfig['activation'],
                                       kernel_size=layerConfig['kernel_size'],
                                       strides=layerConfig['strides'],
                                       pad=layerConfig['padding'])
                self.vMem.setWeigtsBias(layer_name=layerConfig['name'], weights=layer.get_weights()[0], bias=layer.get_weights()[1])
            elif layerClassName == 'MaxPooling2D':
                self.vMem.addMaxPoolingLayer(name=layerConfig['name'],
                                             kernel_size=layerConfig['pool_size'],
                                             strides=layerConfig['strides'],
                                             pad=layerConfig['padding'])
            elif layerClassName == 'UpSampling2D':
                self.vMem.addUpSamplingLayer(name=layerConfig['name'],
                                             kernel_size=layerConfig['size'])
            elif layerClassName == 'Concatenate':
                self.vMem.addConcatLayer(name=layerConfig['name'],
                                         src=layer.input[1].name.split('/')[0],
                                         dst=layer.input[0].name.split('/')[0])
            elif layerClassName == 'InputLayer':
                self.vMem.addInputLayer(name=layerConfig['name'],
                                        input_shape=layerConfig['batch_input_shape'])
            elif layerClassName == 'Dropout':
                self.vMem.addDropoutLayer(name=layerConfig['name'], rate=layerConfig['rate'])
            else:
                assert False, 'unsupported layer: {} ({})'.format(layerClassName, layerConfig['name'])

        del self.model
        self.model = None

        self._computeShape()

        self.vMem.layerStat()

    def _readInput(self, input_path: str) -> np.array:
        '''
        returns the gray scaled image in the shape of
        (width, height, depth)
        '''
        img = io.imread(input_path, as_gray=True)
        img = img / 255
        img = trans.resize(image=img, output_shape=self.vMem.getModelInputShape())
        return img

    def _computeShape(self) -> None:
        '''
        compute all input/output shapes
        set the shape of the last (output) layer
        '''
        prevShape = self.vMem.layerList[0].inputShape
        for layer in self.vMem.layerList:
            prevShape = layer.computeShape(self.vMem.layerMap, prevShape)
        self.finalOutputShape = tuple(prevShape)
