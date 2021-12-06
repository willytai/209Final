import numpy as np
from . import OutputBuffer as OutputBuffer
from . import InputBuffer as InputBuffer

PE_COMPUTATIONAL_CAPABILITY = 32

class ComputationUnit():
    def __init__(self, pe_array_size: int) -> None:
        '''
        self.PEArray[0,:]: tiled feature map
        self.PEArray[1,:]: kernel weights
        self.PEArray[2,:]: output register
        '''
        self.PEArray = np.zeros((3, pe_array_size*PE_COMPUTATIONAL_CAPABILITY))
        self.PEArrayLength = pe_array_size
        self.inputBuffer = None
        self.numInput = None
        self.numOutput = None
        self.outputPos = None
        self.outputChannels = None
        self.cycles = 0
        self.multiplicationCount = 0
        self.wordLength = 0

    def setWordLength(self, word_length: int) -> None:
        self.wordLength = word_length

    def computeNextRound(self) -> None:
        '''
      ✓ 1. data fetch
      ✓ 2. assign task to PE
      ✓ 3. compute
      ✓ 4. record self.numInput and self.numOutput
      ✓ 5. increment cycle
      ✓ 6. acccumulate number of multiplications
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
        assert self.wordLength == 0, 'quantize after addition, determine the fraction length dynamically, wrap the quantize function into a class'

        # record stats for further operations
        self.numInput = numInput
        self.numOutput = numOutput
        self.outputPos = outputPos
        self.outputChannels = outputChannels
        self.cycles += 1
        self.multiplicationCount += numInput*numOutput

    def dataFetch(self) -> tuple:
        return self.inputBuffer.sendData(self.PEArrayLength*PE_COMPUTATIONAL_CAPABILITY)

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        '''
      ✓ 1. acccumulate values for corresponding output channels
      ✓ 2. write (add) to output buffer
      ✓ 3. check stats
        '''
        assert self.numInput is not None and self.numOutput is not None and self.outputPos is not None and self.outputChannels is not None
        data = np.zeros(self.numOutput)
        for i in range(self.numOutput):
            data[i] = np.add.reduce(self.PEArray[2,i*self.numInput:(i+1)*self.numInput])
        output_buffer.writeData(data=data,
                                position=self.outputPos,
                                channels=self.outputChannels)

        return self.inputBuffer.isRoundFinished()

    def usage(self) -> None:
        print ('------ Resource Usage ------')
        print ('- # PE: {}'.format(self.PEArrayLength))
        print ('- Cycles: {}'.format(self.cycles))
        print ('- PE Utilization: {:.2f}%'.format(self.multiplicationCount/(self.cycles*self.PEArrayLength*PE_COMPUTATIONAL_CAPABILITY)*100))
        print ('----------------------------')

    def linkInputBuffer(self, input_buffer: InputBuffer) -> None:
        self.inputBuffer = input_buffer
