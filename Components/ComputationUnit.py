import numpy as np
from . import OutputBuffer as OutputBuffer
from . import InputBuffer as InputBuffer

'''
    A PE is capable of doing 32 multiplications by default
'''
PE_COMPUTATIONAL_CAPABILITY = 32
class PE():
    def __init__(self):
        pass

'''
    1. Contains the PE array
    2. Channel based parallelism
    - self.inputBuffer references the InputBuffer for convenience
'''
class ComputationUnit():
    def __init__(self, pe_array_size: int) -> None:
        self.PEArray = np.zeros((3, pe_array_size*PE_COMPUTATIONAL_CAPABILITY))
        self.inputBuffer = None
        self._resetStats()

    def computeNextRound(self) -> None:
        '''
      ✓ 1. data fetch
      ✓ 2. assign task to PE
      ✓ 3. compute
      ✓ 4. record self.numInput and self.numOutput
        '''
        assert self.inputBuffer is not None
        kernelWeights, features, outputPos, outputChannels = self.dataFetch()
        # some sanity checks
        assert kernelWeights.shape[0] == 1 and kernelWeights.shape[0] == 1 and kernelWeights.shape[2] == features.shape[2] and features.shape[0] == 1 and features.shape[1] == 1

        # (kernelid, weights)
        kernelWeights = np.transpose(kernelWeights.reshape((kernelWeights.shape[2], kernelWeights.shape[3])))
        features = features.reshape((features.shape[2],))
        numInput = kernelWeights.shape[1]
        numOutput = kernelWeights.shape[0]

        # assign task to PE
        for i in range(numOutput):
            self.PEArray[0,i*numInput:(i+1)*numInput] = features
        self.PEArray[1,:numInput*numOutput] = kernelWeights.reshape(-1)

        # compute multiplication (element-wise multiplication)
        self.PEArray[2,:] = self.PEArray[0,:] * self.PEArray[1,:]

        # record stats for further operations
        self.numInput = numInput
        self.numOutput = numOutput
        self.outputPos = outputPos
        self.outputChannels = outputChannels

        '''
        print (self.PEArray[1,:numInput*numOutput])
        print (self.PEArray[0,:numInput*numOutput])
        print (self.PEArray[2,:numInput*numOutput])
        print (kernelWeights)
        print (kernelWeights.shape)
        print (features.shape)
        print ('numInput/numOutput', numInput, numOutput)
        print ('outputPos', outputPos)
        print ('outputChannels',outputChannels)
        raise NotImplementedError
        '''

    def dataFetch(self) -> tuple:
        return self.inputBuffer.sendData(len(self.PEArray)*PE_COMPUTATIONAL_CAPABILITY)

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        '''
      ✓ 1. acccumulate values for corresponding output channels
      ✓ 2. write (add) to output buffer
        3. check stats
        '''
        assert self.numInput is not None and self.numOutput is not None and self.outputPos is not None and self.outputChannels is not None
        for i, channel in enumerate(self.outputChannels):
            output_buffer.writeData(data=self.PEArray[2,i*self.numInput:(i+1)*self.numInput].sum(),
                                    position=self.outputPos,
                                    channel=channel)
        '''
        print (output_buffer.activeBuffer.shape)
        print (self.numInput)
        print (self.numOutput)
        print (self.outputPos)
        print (self.outputChannels)
        self._resetStats()
        raise NotImplementedError
        '''
        return (self.inputBuffer.isRoundFinished(), None)

    def linkInputBuffer(self, input_buffer: InputBuffer) -> None:
        self.inputBuffer = input_buffer

    def _resetStats(self):
        self.numInput = None
        self.numOutput = None
        self.outputPos = None
        self.outputChannels = None

