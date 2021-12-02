import numpy as np
from skimage.measure import block_reduce
from . import VMem as VMem
from . import ComputationUnit as ComputationUnit

'''
- self.residuals stores the required feature maps for the decoding path
- self.activeBuffer is the storage of the output feature map for the current layer
  re-initialize it when the computation loop enters the next layer might be more convenient
- self.computationUnit references the computation unit for convenience
- self.dataReady indicates whether the data is ready to be written to the external memory
- self.end indicates whether the loop has ended (last layer or not)
'''


class OutputBuffer():
    def __init__(self) -> None:
        self.residuals = list()
        self.activeBuffer = None
        self.computationUnit = None
        self.dataReady = False
        self.end = False
        self.postProcessInfo = dict()

        self.read_layer1 = True
        self.conv_out_count = 1
        self.pool_out_count = 1
        self.upsample_out_count = 1
        self.concat_out_count = 1

    def vMemWrite(self, v_mem: VMem) -> bool:
        '''
      ✓ 1. verify the current data
      ✓ 2. post-processing
      ✓ 3. mem write
      ✓ 4. reset the active buffer
      ✓ 5. check for the status of the output
            - if the computed output is for the final layer, return True
            - otherwise, return False
        '''

        # if self.read_layer1:
        #     print ('this is temporary, reading precomputed layer3 output')
        #     v_mem.layerID += 2+2+1
        #     self.read_layer1 = False
        #     self.activeBuffer = np.load('layer3_conv_output.npy')
        #     self.dataReady = True
        #     v_mem.write(self.activeBuffer)
        #     self.activeBuffer = None
        #     self.dataReady = False
        #     if self.end is None:
        #         raise NotImplementedError
        #     return self.end

        self.checkData()
        assert self.activeBuffer is not None
        self.postProcess()
        v_mem.write(self.activeBuffer)
        self.activeBuffer = None
        self.dataReady = False
        # if self.end is None:
        #     raise NotImplementedError
        return self.end

    def checkData(self) -> None:
        '''
      ✓ 1. check if the current data is ready
            - if not, request data from the computation unit until the whole output is computed
        '''
        while not self.dataReady:
            self.requestData()
        # raise NotImplementedError

    def requestData(self) -> None:
        '''
      ✓ 1. request new data from the computation unit
            - computation
            - data write
      ✓ 2. update the status accordingly
        '''
        assert self.computationUnit is not None
        self.computationUnit.computeNextRound()
        self.dataReady, self.end = self.computationUnit.dataWrite(self)

    def writeData(self, data: float, position: tuple, channels: range) -> None:
        '''
        accumulate the calculated value to the desired position of the acitve buffer
        '''
        assert self.activeBuffer is not None
        self.activeBuffer[position[0], position[1], channels] += data

    def resetBuffer(self, shape: tuple) -> None:
        print ('output buffer reset to {}'.format(shape))
        self.activeBuffer = np.zeros(shape)

    def setBiasActivation(self, bias: np.array, activation: str) -> None:
        self.postProcessInfo['bias'] = bias
        self.postProcessInfo['activation'] = activation

    def setPooling(self, pool_size: tuple) -> None:
        self.postProcessInfo['pooling'] = pool_size

    def setUpsample(self, kernel_size: tuple) -> None:
        self.postProcessInfo['upsample'] = kernel_size

    def setSaveResidual(self) -> None:
        self.postProcessInfo['save'] = True

    def setConcate(self) -> None:
        self.postProcessInfo['concat'] = True

    def postProcess(self) -> None:
        '''
        1. add bias and do activation post-processing
        2. check for other required post-processing (pooling, upsampling, concatenation)
        '''
        # bias
        self.activeBuffer = self.activeBuffer + self.postProcessInfo['bias']

        # activation
        assert self.postProcessInfo['activation'] == 'relu'
        self.activeBuffer = np.maximum(0, self.activeBuffer)

        np.save('layer{}_conv_output.npy'.format(self.conv_out_count), self.activeBuffer)
        self.conv_out_count += 1

        # save or not
        if 'save' in self.postProcessInfo:
            assert self.postProcessInfo['save']
            self.residuals.append(self.activeBuffer)

        # pooling with skimage API
        if 'pooling' in self.postProcessInfo:
            assert 'upsample' not in self.postProcessInfo
            print ('pooling')
            pool_size = self.postProcessInfo['pooling']
            pool_size = (pool_size[0], pool_size[1], 1)
            self.activeBuffer = block_reduce(self.activeBuffer, block_size=(pool_size), func=np.max)

            np.save('layer{}_maxpool_output.npy'.format(self.pool_out_count), self.activeBuffer)
            self.pool_out_count += 1

        # upsampling with numpy
        if 'upsample' in self.postProcessInfo:
            assert 'pooling' not in self.postProcessInfo
            print ('upsampling')
            kernel_size = self.postProcessInfo['upsample']
            self.activeBuffer = self.activeBuffer.repeat(kernel_size[1], axis=1).repeat(kernel_size[0], axis=0)

            np.save('layer{}_upsample_output.npy'.format(self.upsample_out_count), self.activeBuffer)
            self.upsample_out_count += 1

        # concat
        if 'concat' in self.postProcessInfo:
            assert self.postProcessInfo['concat']
            self.activeBuffer = np.concatenate(( self.activeBuffer, self.residuals.pop()), axis=2)

            np.save('layer{}_concatenate_output.npy'.format(self.concat_out_count), self.activeBuffer)
            self.concat_out_count += 1

        self.postProcessInfo = dict()
        # raise NotImplementedError

    def linkComputationUnit(self, computation_unit: ComputationUnit) -> None:
        self.computationUnit = computation_unit
