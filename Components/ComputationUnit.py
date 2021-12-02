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
        self.PEArrayLength = pe_array_size
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
        assert kernelWeights.shape[1] == features.shape[0]

        # (kernelid, weights)
        numInput = kernelWeights.shape[1]
        numOutput = kernelWeights.shape[0]

        # assign task to PE
        self.PEArray[0,:numInput*numOutput] = np.tile(features, numOutput)
        self.PEArray[1,:numInput*numOutput] = kernelWeights.reshape(-1)

        # compute multiplication (element-wise multiplication)
        self.PEArray[2,:numInput*numOutput] = self.PEArray[0,:numInput*numOutput] * self.PEArray[1,:numInput*numOutput]

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
        return self.inputBuffer.sendData(self.PEArrayLength*PE_COMPUTATIONAL_CAPABILITY)

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        '''
      ✓ 1. acccumulate values for corresponding output channels
      ✓ 2. write (add) to output buffer
        3. check stats
        '''
        assert self.numInput is not None and self.numOutput is not None and self.outputPos is not None and self.outputChannels is not None
        data = np.zeros(self.numOutput)
        for i in range(self.numOutput):
            data[i] = np.add.reduce(self.PEArray[2,i*self.numInput:(i+1)*self.numInput])
        output_buffer.writeData(data=data,
                                position=self.outputPos,
                                channels=self.outputChannels)
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

