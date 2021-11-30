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
        self.PEArray = [PE() for i in range(pe_array_size)]
        self.inputBuffer = None

    def computeNextRound(self) -> None:
        '''
      ✓ 1. data fetch
        3. assign task to PE
        4. acccumulate values for corresponding output channels
        '''
        assert self.inputBuffer is not None
        kernelWeights, features, outputPos, outputChannels = self.dataFetch()
        # some sanity checks
        assert kernelWeights.shape[0] == 1 and kernelWeights.shape[0] == 1 and kernelWeights.shape[2] == features.shape[2] and features.shape[0] == 1 and features.shape[1] == 1
        kernelWeights = kernelWeights.reshape((kernelWeights.shape[2], kernelWeights.shape[3]))
        features = features.reshape((features.shape[2],))
        numInput = kernelWeights.shape[0]
        numOutput = kernelWeights.shape[1]
        print (kernelWeights.shape)
        print (features.shape)
        print (numInput, numOutput)
        print (outputPos)
        print (outputChannels)
        raise NotImplementedError

    def dataFetch(self) -> tuple:
        return self.inputBuffer.sendData(len(self.PEArray)*PE_COMPUTATIONAL_CAPABILITY)

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        raise NotImplementedError

    def linkInputBuffer(self, input_buffer: InputBuffer) -> None:
        self.inputBuffer = input_buffer
