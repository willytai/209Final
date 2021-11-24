from . import OutputBuffer as OutputBuffer
from . import InputBuffer as InputBuffer

'''
    A PE is capable of doing 32 multiplications by default
'''
class PE():
    def __init__(self):
        pass

'''
    1. Contains the PE array
    2. Channel based parallelism
    - self.outputChannelIdx is the corresponding channels of the output feature that will be computed. (np.array)
    - self.inputBuffer references the InputBuffer for convenience
'''
class ComputationUnit():
    def __init__(self, pe_array_size: int) -> None:
        self.PEArray = list()
        self.outputChannelIdx = None
        self.inputBuffer = None

    def computeNextRound(self) -> None:
        '''
      ✓ 1. data fetch
        3. assign task to PE
        4. acccumulate values for corresponding output channels
        '''
        assert self.inputBuffer is not None
        data = self.dataFetch()
        '''
        do this somewhere
        1. record self.outputChannelIdx
        '''
        raise NotImplementedError

    def dataFetch(self) -> tuple:
        return self.inputBuffer.sendData()

    def dataWrite(self, output_buffer: OutputBuffer) -> tuple:
        raise NotImplementedError

    def linkInputBuffer(self, input_buffer: InputBuffer) -> None:
        self.inputBuffer = input_buffer
